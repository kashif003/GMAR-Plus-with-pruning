import torch
import torch.nn.functional as F
import types
from typing import Optional


# ── storage container — one per block ────────────────────────────────
class BlockStore:
    """Stores all tensors needed for CheferCAM LRP relprop."""
    # attention internals
    qkv_input:         torch.Tensor = None   # (1, 197, 768)  input to QKV proj
    q:                 torch.Tensor = None   # (1, 12, 197, 64)
    k:                 torch.Tensor = None   # (1, 12, 197, 64)
    v_4d:              torch.Tensor = None   # (1, 12, 197, 64)
    pre_softmax_attn:  torch.Tensor = None   # (1, 12, 197, 197) QK^T/√d before softmax
    attn_stored:       torch.Tensor = None   # (1, 12, 197, 197) post-softmax
    proj_input:        torch.Tensor = None   # (1, 197, 768)  input to out_proj
    # block residuals
    x_in:              torch.Tensor = None   # (1, 197, 768)  block input (clone1.X)
    attn_out:          torch.Tensor = None   # (1, 197, 768)  ls_1(attn) (add1.X2)
    x_before_attn_res: torch.Tensor = None   # (1, 197, 768)  q_x (add1.X1)
    x_after_attn:      torch.Tensor = None   # (1, 197, 768)  after attn residual (clone2.X)
    x_before_mlp_res:  torch.Tensor = None   # (1, 197, 768)  x (add2.X1)
    mlp_out:           torch.Tensor = None   # (1, 197, 768)  ls_2(mlp) (add2.X2)
    # MLP internals — matches Sequential(c_fc, gelu, c_proj)
    fc1_input:         torch.Tensor = None   # (1, 197, 768)  input to c_fc
    act_input:         torch.Tensor = None   # (1, 197, 3072) input to gelu
    fc2_input:         torch.Tensor = None   # (1, 197, 3072) input to c_proj
    # set by chefer.py relprop, read by transformer_attribution
    attn_cam:          torch.Tensor = None   # (1, 12, 197, 197)


def hooked_attention(self,
                     q_x: torch.Tensor,
                     k_x: Optional[torch.Tensor] = None,
                     v_x: Optional[torch.Tensor] = None,
                     attn_mask: Optional[torch.Tensor] = None):

    k_x = k_x if k_x is not None else q_x
    v_x = v_x if v_x is not None else q_x
    attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None

    B, N, D  = q_x.shape
    H        = self.attn.num_heads
    head_dim = D // H
    scale    = head_dim ** -0.5

    # store qkv_input (= input to QKV linear projection, i.e. ln_1 output)
    self.store.qkv_input = q_x.detach().requires_grad_(True)

    q, k, v = F.linear(
        q_x, self.attn.in_proj_weight, self.attn.in_proj_bias
    ).chunk(3, dim=-1)

    q = q.contiguous().view(B, N, H, head_dim).permute(0, 2, 1, 3)  # (B,H,N,hd)
    k = k.contiguous().view(B, N, H, head_dim).permute(0, 2, 1, 3)
    v = v.contiguous().view(B, N, H, head_dim).permute(0, 2, 1, 3)

    self.store.q   = q.detach().requires_grad_(True)
    self.store.k   = k.detach().requires_grad_(True)
    self.store.v_4d = v.detach().requires_grad_(True)

    q_flat = q.reshape(B * H, N, head_dim)
    k_flat = k.reshape(B * H, N, head_dim)
    v_flat = v.reshape(B * H, N, head_dim)

    # pre-softmax logits
    pre_softmax = torch.bmm(q_flat * scale, k_flat.transpose(1, 2))  # (B*H,N,N)
    if attn_mask is not None:
        pre_softmax = pre_softmax + attn_mask

    # store pre-softmax for faithful softmax relprop in CheferCAM
    self.store.pre_softmax_attn = pre_softmax.view(B, H, N, N).detach().requires_grad_(True)

    attn_weights = pre_softmax.softmax(dim=-1)                        # (B*H,N,N)

    self.attention_map = attn_weights.view(B, H, N, N)
    self.store.attn_stored = self.attention_map.detach().requires_grad_(True)

    def save_grad(g):
        self.attention_map.captured_grad = g.view(B, H, N, N)

    if attn_weights.requires_grad:
        attn_weights.register_hook(save_grad)

    out = torch.bmm(attn_weights, v_flat)                             # (B*H,N,hd)
    out = out.view(B, H, N, head_dim).permute(0, 2, 1, 3).reshape(B, N, D)

    self.store.proj_input = out.detach().requires_grad_(True)

    return self.attn.out_proj(out)


def hooked_block_forward(self,
                         q_x: torch.Tensor,
                         k_x: Optional[torch.Tensor] = None,
                         v_x: Optional[torch.Tensor] = None,
                         attn_mask: Optional[torch.Tensor] = None):

    k_x = self.ln_1_kv(k_x) if hasattr(self, 'ln_1_kv') and k_x is not None else None
    v_x = self.ln_1_kv(v_x) if hasattr(self, 'ln_1_kv') and v_x is not None else None

    # store block input (clone1.X in Chefer repo)
    self.store.x_in = q_x.detach().requires_grad_(True)

    # ── first residual branch (attention) ────────────────────────────
    attn_out = self.attention(
        q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask
    )
    attn_out_scaled = self.ls_1(attn_out) if hasattr(self, 'ls_1') else attn_out

    # store add1 inputs
    self.store.x_before_attn_res = q_x.detach().requires_grad_(True)
    self.store.attn_out          = attn_out_scaled.detach().requires_grad_(True)

    x = q_x + attn_out_scaled

    # store clone2.X (input to MLP branch)
    self.store.x_after_attn = x.detach().requires_grad_(True)

    # ── second residual branch (MLP) ─────────────────────────────────
    # MLP is Sequential(c_fc, gelu, c_proj) — confirmed from model inspection
    ln2_out = self.ln_2(x)

    self.store.fc1_input = ln2_out.detach().requires_grad_(True)

    fc1_out = self.mlp.c_fc(ln2_out)

    self.store.act_input = fc1_out.detach().requires_grad_(True)

    act_out = self.mlp.gelu(fc1_out)

    self.store.fc2_input = act_out.detach().requires_grad_(True)

    mlp_out = self.mlp.c_proj(act_out)
    mlp_out_scaled = self.ls_2(mlp_out) if hasattr(self, 'ls_2') else mlp_out

    # store add2 inputs
    self.store.x_before_mlp_res = x.detach().requires_grad_(True)
    self.store.mlp_out           = mlp_out_scaled.detach().requires_grad_(True)

    x = x + mlp_out_scaled
    return x


def install_hooks(model) -> None:
    """Install attention + block-forward hooks on all transformer blocks."""
    for blk in model.visual.transformer.resblocks:
        blk.store     = BlockStore()
        blk.attention = types.MethodType(hooked_attention,     blk)
        blk.forward   = types.MethodType(hooked_block_forward, blk)

    for name, param in model.named_parameters():
        param.requires_grad = 'visual.transformer' in name


def get_grad(attn_tensor: torch.Tensor) -> torch.Tensor:
    """Read gradient stored by register_hook during backward."""
    grad = getattr(attn_tensor, 'captured_grad', None)
    if grad is None:
        raise RuntimeError(
            "captured_grad is None — backward() has not been called yet"
        )
    return grad