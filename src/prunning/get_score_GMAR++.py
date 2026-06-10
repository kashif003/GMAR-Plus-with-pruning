  


model = CustomViT()
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")
gmarpp = GMARv2()

images = get_jpeg_images("imagenet_val_1000")
final_score = []

for idx, image in enumerate(images):
    torch.cuda.empty_cache()
    img_tensor = get_img_tensor(processor, image)
    logits, predicted_class, class_name, attn_weights = model.forward_with_custom_attention(
        img_tensor["pixel_values"].to("cuda:7")
    )
    score = gmarpp.compute(
        logits,
        predicted_class,
        attn_weights,
        model
    )
    # score is assumed to be a list of length 24, each tensor shaped like [16, ...]
    # Per-head score for each layer -> one value per head
    img_score_per_head = [
        (scr.detach().cpu() ** 2).sum(dim=(-1, -2)).sqrt()
        for scr in score
    ]  # list of 24 tensors, each of shape [16]
    # -------- GLOBAL MIN-MAX ACROSS ALL LAYERS AND ALL HEADS FOR THIS IMAGE --------
    all_heads_tensor = torch.cat([t.reshape(-1) for t in img_score_per_head], dim=0)  # shape [24*16]
    global_min = all_heads_tensor.min()
    global_max = all_heads_tensor.max()
    normalized_score = [
        (t - global_min) / (global_max - global_min + 1e-8)
        for t in img_score_per_head
    ]
    # -----------------------------------------------------------------------------
    if len(final_score) == 0:
        final_score = normalized_score
    else:
        final_score = [x + y for x, y in zip(final_score, normalized_score)]
    del logits
    del predicted_class
    del class_name
    del attn_weights
    del score
    del img_score_per_head
    del all_heads_tensor
    del normalized_score
    del img_tensor

    torch.cuda.empty_cache()

    print("[INFO] Processed image number:", idx + 1)


json_ready = {
    str(i): t.detach().cpu().tolist()
    for i, t in enumerate(final_score)
}

c