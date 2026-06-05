"""CheferCAM: Exact port of transformer_attribution from Chefer et al. CVPR 2021
Repo: https://github.com/hila-chefer/Transformer-Explainability

Implements:
    1. Full LRP relprop backward through all blocks (Phase 3)
    2. transformer_attribution rollout using attn_cam × gradient (Phase 4)
    3. compute_rollout_attention with reversed bmm, no row norm (Phase 5)
"""

from typing import List
import torch
import torch.nn.functional as F

from hooks import get_grad

__all__ = ["CheferCAM"]


# ══════════════════════════════════════════════════════════════════════
# LRP utility functions — exact ports from layers_ours.py
# ══════════════════════════════════════════════════════════════════════

def safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact copy of safe_divide from layers_ours.py"""
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def gradprop(Z: torch.Tensor,
             X,
             S: torch.Tensor) -> tuple:
    """Exact copy of RelProp.gradprop from layers_ours.py
    Computes C = ∂Z/∂X * S  (gradient scaled by relevance density)
    """
    C = torch.autograd.grad(Z, X, S, retain_graph=True)
    return C


# ══════════════════════════════════════════════════════════════════════
# LRP relprop rules — one per operation type
# ══════════════════════════════════════════════════════════════════════

def linear_relprop(R: torch.Tensor,
                   X: torch.Tensor,
                   weight: torch.Tensor,
                   bias: torch.Tensor = None,
                   alpha: float = 1.0) -> torch.Tensor:
    """Linear layer relprop — exact port of Linear.relprop from layers_ours.py
    Uses α-β rule with alpha=1, beta=0 (as in repo kwargs).

    Args:
        R      : relevance at output   any shape ending in out_features
        X      : stored input          any shape ending in in_features
        weight : layer weight          (out_features, in_features)
    """
    beta = alpha - 1

    pw = weight.clamp(min=0)
    nw = weight.clamp(max=0)
    px = X.clamp(min=0)
    nx = X.clamp(max=0)

    # re-enable grad for gradprop
    px = px.detach().requires_grad_(True)
    nx = nx.detach().requires_grad_(True)

    def f(w1, w2, x1, x2):
        Z1 = F.linear(x1, w1)
        Z2 = F.linear(x2, w2)
        S1 = safe_divide(R, Z1 + Z2)
        S2 = safe_divide(R, Z1 + Z2)
        C1 = x1 * gradprop(Z1, x1, S1)[0]
        C2 = x2 * gradprop(Z2, x2, S2)[0]
        return C1 + C2

    activator = f(pw, nw, px, nx)
    inhibitor = f(nw, pw, px, nx)

    return alpha * activator - beta * inhibitor


def relprop_simple(R: torch.Tensor,
                   X_list: list,
                   forward_fn) -> list:
    """RelPropSimple rule — exact port of RelPropSimple.relprop from layers_ours.py
    Used for: einsum (matmul1, matmul2), Add, Clone.

    Args:
        R          : relevance at output
        X_list     : list of stored inputs [X1, X2, ...]
        forward_fn : callable that recomputes Z from X_list
    """
    # re-enable grad
    X_list = [x.detach().requires_grad_(True) for x in X_list]
    Z = forward_fn(*X_list)
    S = safe_divide(R, Z)
    C = gradprop(Z, X_list, S)
    outputs = [x * c for x, c in zip(X_list, C)]
    return outputs


def add_relprop(R: torch.Tensor,
                x1: torch.Tensor,
                x2: torch.Tensor):
    """Add.relprop — exact port from layers_ours.py
    Proportional split with conservation correction.

    Returns: R_x1, R_x2
    """
    x1 = x1.detach().requires_grad_(True)
    x2 = x2.detach().requires_grad_(True)

    Z  = x1 + x2
    S  = safe_divide(R, Z)
    C  = gradprop(Z, [x1, x2], S)

    a  = x1 * C[0]
    b  = x2 * C[1]

    a_sum = a.sum()
    b_sum = b.sum()

    a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
    b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

    a = a * safe_divide(a_fact, a.sum())
    b = b * safe_divide(b_fact, b.sum())

    return a, b


def clone_relprop(R_list: list,
                  X: torch.Tensor) -> torch.Tensor:
    """Clone.relprop — exact port from layers_ours.py
    Merges two relevances back into one via gradient-weighted sum.

    Args:
        R_list : [R1, R2]  two incoming relevances
        X      : stored input to clone
    """
    X = X.detach().requires_grad_(True)
    # clone produces num copies of X
    Z_list = [X for _ in R_list]
    S_list = [safe_divide(r, z) for r, z in zip(R_list, Z_list)]
    C = gradprop(Z_list, X, S_list)[0]
    return X * C


def layernorm_relprop(R: torch.Tensor) -> torch.Tensor:
    """LayerNorm.relprop — identity in this repo implementation."""
    return R


def gelu_relprop(R: torch.Tensor) -> torch.Tensor:
    """GELU.relprop — identity (RelProp base class, no override)."""
    return R


def dropout_relprop(R: torch.Tensor) -> torch.Tensor:
    """Dropout.relprop — identity (RelProp base class, no override)."""
    return R


# ══════════════════════════════════════════════════════════════════════
# Per-module relprop — exact sequence from repo
# ══════════════════════════════════════════════════════════════════════

def mlp_relprop(R: torch.Tensor, blk, mlp_weight_fc1, mlp_weight_fc2) -> torch.Tensor:
    """Mlp.relprop — exact sequence from ViT_LRP.py:
        drop → fc2 → act → fc1
    """
    # drop.relprop — identity
    R = dropout_relprop(R)

    # fc2.relprop — Linear α-β rule
    R = linear_relprop(R,
                       X      = blk.store.fc2_input,
                       weight = mlp_weight_fc2)

    # act.relprop — identity (GELU)
    R = gelu_relprop(R)

    # fc1.relprop — Linear α-β rule (no drop before fc1 in standard ViT)
    R = linear_relprop(R,
                       X      = blk.store.fc1_input,
                       weight = mlp_weight_fc1)

    return R


def attn_relprop(R: torch.Tensor, blk,
                 out_proj_weight: torch.Tensor,
                 out_proj_bias:   torch.Tensor) -> torch.Tensor:
    """Attention.relprop — exact sequence from ViT_LRP.py:
        proj_drop → proj → rearrange → matmul2 → save_attn_cam
        → attn_drop → softmax → matmul1 → qkv

    Sets blk.store.attn_cam after matmul2 relprop.
    """
    B  = blk.store.proj_input.shape[0]
    H  = blk.attention_map.shape[1]        # num_heads
    N  = blk.attention_map.shape[2]        # num_tokens
    hd = blk.store.v_4d.shape[-1]          # head_dim

    # proj_drop.relprop — identity
    R = dropout_relprop(R)                 # (B, N, D)

    # proj.relprop — Linear α-β rule
    # stored: proj_input  (B, N, D)
    R = linear_relprop(R,
                       X      = blk.store.proj_input,
                       weight = out_proj_weight,
                       bias   = out_proj_bias)     # (B, N, D)

    # rearrange: (B, N, H*hd) → (B, H, N, hd)
    R = R.reshape(B, N, H, hd).permute(0, 2, 1, 3)   # (B, H, N, hd)

    # matmul2.relprop — RelPropSimple for einsum('bhij,bhjd->bhid', attn, v)
    # stored: attn_stored (B,H,N,N),  v_4d (B,H,N,hd)
    def matmul2_forward(attn, v):
        return torch.einsum('bhij,bhjd->bhid', attn, v)

    cam1, cam_v = relprop_simple(
        R          = R,
        X_list     = [blk.store.attn_stored, blk.store.v_4d],
        forward_fn = matmul2_forward
    )

    # exact from repo: cam1 /= 2,  cam_v /= 2
    cam1  = cam1  / 2    # (B, H, N, N) — relevance at attention weights A
    cam_v = cam_v / 2    # (B, H, N, hd)

    # ── save attn_cam — this is Â(l) used by transformer_attribution ──
    blk.store.attn_cam = cam1.detach()     # (B, H, N, N)

    # attn_drop.relprop — identity
    cam1 = dropout_relprop(cam1)

    # softmax.relprop — RelPropSimple with pre-softmax QK^T/√d as input
    # exact match to repo: RelPropSimple applied to the softmax operation
    cam1_list = relprop_simple(
        R          = cam1,
        X_list     = [blk.store.pre_softmax_attn],
        forward_fn = lambda x: x.softmax(dim=-1)
    )
    cam1 = cam1_list[0]            # (B, H, N, N)

    # matmul1.relprop — RelPropSimple for einsum('bhid,bhjd->bhij', q, k)
    # stored: q (B,H,N,hd),  k (B,H,N,hd)
    def matmul1_forward(q, k):
        return torch.einsum('bhid,bhjd->bhij', q, k)

    cam_q, cam_k = relprop_simple(
        R          = cam1,
        X_list     = [blk.store.q, blk.store.k],
        forward_fn = matmul1_forward
    )

    # exact from repo: cam_q /= 2,  cam_k /= 2
    cam_q = cam_q / 2
    cam_k = cam_k / 2

    # rearrange qkv for qkv.relprop
    # cam_qkv shape: (B, N, 3*H*hd)
    cam_qkv = torch.cat([
        cam_q.permute(0, 2, 1, 3).reshape(B, N, H * hd),
        cam_k.permute(0, 2, 1, 3).reshape(B, N, H * hd),
        cam_v.permute(0, 2, 1, 3).reshape(B, N, H * hd),
    ], dim=-1)                                            # (B, N, 3*D)

    # qkv.relprop — Linear α-β rule
    # stored: qkv_input (B, N, D)
    # weight: in_proj_weight (3*D, D)
    R = linear_relprop(cam_qkv,
                       X      = blk.store.qkv_input,
                       weight = blk.attn.in_proj_weight,
                       bias   = blk.attn.in_proj_bias)   # (B, N, D)

    return R


def block_relprop(R: torch.Tensor, blk,
                  out_proj_weight, out_proj_bias,
                  mlp_weight_fc1,  mlp_weight_fc2) -> torch.Tensor:
    """Block.relprop — exact sequence from ViT_LRP.py:
        add2 → mlp → norm2 → clone2 → add1 → attn → norm1 → clone1
    """
    # add2.relprop — proportional split (residual + mlp)
    R_res2, R_mlp = add_relprop(R,
                                 x1 = blk.store.x_before_mlp_res,
                                 x2 = blk.store.mlp_out)

    # mlp.relprop — fc2 → gelu → fc1
    R_mlp = mlp_relprop(R_mlp, blk, mlp_weight_fc1, mlp_weight_fc2)

    # norm2.relprop — identity
    R_mlp = layernorm_relprop(R_mlp)

    # clone2.relprop — merge R_res2 + R_mlp
    R = clone_relprop([R_res2, R_mlp], blk.store.x_after_attn)

    # add1.relprop — proportional split (residual + attn)
    R_res1, R_attn = add_relprop(R,
                                  x1 = blk.store.x_before_attn_res,
                                  x2 = blk.store.attn_out)

    # attn.relprop — sets blk.store.attn_cam as side effect
    R_attn = attn_relprop(R_attn, blk, out_proj_weight, out_proj_bias)

    # norm1.relprop — identity
    R_attn = layernorm_relprop(R_attn)

    # clone1.relprop — merge R_res1 + R_attn
    R = clone_relprop([R_res1, R_attn], blk.store.x_in)

    return R


# ══════════════════════════════════════════════════════════════════════
# compute_rollout_attention — exact port from ViT_LRP.py lines 38-50
# ══════════════════════════════════════════════════════════════════════

def compute_rollout_attention(all_layer_matrices: List[torch.Tensor],
                              start_layer: int = 0) -> torch.Tensor:
    """Exact port of compute_rollout_attention from ViT_LRP.py.

    - adds identity (residual consideration)
    - row normalisation omitted (official repo has it commented out)
    - reversed multiply order: new.bmm(old)
    """
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    device     = all_layer_matrices[0].device

    eye = torch.eye(num_tokens, device=device).expand(
        batch_size, num_tokens, num_tokens
    )

    # add identity — residual consideration
    all_layer_matrices = [m + eye for m in all_layer_matrices]

    # row normalisation is intentionally OMITTED — the official repo has it
    # commented out (lines 44-45 of ViT_LRP.py); applying it changes results
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer + 1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)

    return joint_attention


# ══════════════════════════════════════════════════════════════════════
# CheferCAM — main class
# ══════════════════════════════════════════════════════════════════════

class CheferCAM:
    """Exact implementation of transformer_attribution from Chefer et al.

    Pipeline:
        Phase 1 — forward pass (done in main.py via hooked model)
        Phase 2 — gradient backward (populates captured_grad)
        Phase 3 — LRP relprop backward (populates attn_cam per block)
        Phase 4 — transformer_attribution: grad × attn_cam rollout
    """

    def __init__(self, start_layer: int = 0) -> None:
        self.start_layer = start_layer

    def compute(self,
                logits:       torch.Tensor,
                pred_class:   int,
                attn_weights: List[torch.Tensor],
                model) -> torch.Tensor:
        """
        Args:
            logits       : CLIP similarity logits [1, num_classes]
            pred_class   : target class index
            attn_weights : per-block attention maps [B, H, N, N]
                           (used only to get block references)
            model        : the OpenCLIP model with installed hooks

        Returns:
            heatmap      : (14, 14) tensor normalized to [0, 1]
        """
        blocks = list(model.visual.transformer.resblocks)
        device = logits.device

        # ── Phase 2: gradient backward ────────────────────────────────
        # populates blk.attention_map.captured_grad for all blocks
        model.zero_grad()
        logits[0, pred_class].backward(retain_graph=True)

        # ── Phase 3: LRP relprop backward ─────────────────────────────
        # initialise relevance: one_hot at CLS token position
        # shape (1, 197, 768) — CLS token row gets relevance, patches get 0
        N   = blocks[0].attention_map.shape[2]   # 197
        D   = blocks[0].store.x_in.shape[-1]     # 768
        R   = torch.zeros(1, N, D, device=device)
        R[0, 0, :] = 1.0 / D   # distribute equally across CLS embedding dim

        # relprop in REVERSED block order (block 11 → block 0)
        # relprop in REVERSED block order (block 11 → block 0)
        for blk in reversed(blocks):
            # MLP is Sequential(c_fc, gelu, c_proj) — confirmed from diagnostic
            fc1_w = blk.mlp.c_fc.weight     # (3072, 768)
            fc2_w = blk.mlp.c_proj.weight   # (768, 3072)

            R = block_relprop(
                R              = R,
                blk            = blk,
                out_proj_weight= blk.attn.out_proj.weight,
                out_proj_bias  = blk.attn.out_proj.bias,
                mlp_weight_fc1 = fc1_w,
                mlp_weight_fc2 = fc2_w,
            )
            # after this: blk.store.attn_cam is populated  (1, H, N, N)

        # ── Phase 4: transformer_attribution ──────────────────────────
        # exact port of the transformer_attribution branch in ViT_LRP.py
        cams = []
        for blk in blocks:
            grad     = get_grad(blk.attention_map)         # (1, H, N, N)
            attn_cam = blk.store.attn_cam                  # (1, H, N, N)

            # exact from repo lines 362-366:
            # cam  = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            # grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            cam  = attn_cam[0].reshape(-1, N, N)           # (H, N, N)
            grad = grad[0].reshape(-1, N, N)               # (H, N, N)

            cam  = grad * cam                              # (H, N, N)
            cam  = cam.clamp(min=0).mean(dim=0)            # (N, N)
            cams.append(cam.unsqueeze(0))                  # (1, N, N)

        # ── Phase 5: compute_rollout_attention ────────────────────────
        rollout = compute_rollout_attention(cams, start_layer=self.start_layer)
        # exact from repo: cam = rollout[:, 0, 1:]
        cam = rollout[:, 0, 1:]                            # (1, 196)

        # ── reshape + normalize ───────────────────────────────────────
        side    = int(cam.shape[-1] ** 0.5)                # 14
        cls_map = cam[0].reshape(side, side).cpu().detach()

        mn, mx = cls_map.min(), cls_map.max()
        if (mx - mn) <= 0:
            return torch.zeros_like(cls_map)
        return (cls_map - mn) / (mx - mn + 1e-8)