import torch
import torch.nn.functional as F
import numpy as np
from skimage import filters # Assuming you use this for the hard mask


def attn_map_to_soft_mask(attention_map: torch.Tensor, otsu_scaler: float = 1.0, 
                          steepness_gamma: float = 100.0, use_hard_threshold_for_otsu: bool = True):
    """
    Converts an attention map into a DIFFERENTIABLE soft mask.
    The Sigmoid function approximates the hard threshold.

    Args:
        attention_map: The 2D (img_size, img_size) normalized attention map (torch.Tensor).
        otsu_scaler: Multiplier for the OTSU threshold (controls mask size).
        steepness_gamma: Controls the steepness of the Sigmoid (high value approximates hard step).
        use_hard_threshold_for_otsu: If True, uses the OTSU threshold value for the Sigmoid offset.
                                     Crucially, the gradient of the OTSU calculation is STOPPED.

    Returns:
        soft_mask: A differentiable mask tensor (torch.Tensor) in [0, 1].
        hard_mask_np: The original, non-differentiable hard binary mask (numpy array).
    """

    # 1. Calculate the OTSU Threshold (Non-Differentiable Step)
    attention_map_np = attention_map.detach().cpu().numpy()
    threshold_value_np = filters.threshold_otsu(attention_map_np) * otsu_scaler

    # 2. Determine the Threshold Tensor
    if use_hard_threshold_for_otsu:
        # Stop the gradient for the threshold value itself, but use it as an anchor
        threshold_tensor = torch.tensor(threshold_value_np, device=attention_map.device).float()
        threshold_tensor = threshold_tensor.detach() # Explicitly stop gradient from flowing through threshold_value
    else:
        # Option 2: Use a simple constant threshold (e.g., mean) if you prefer
        threshold_tensor = attention_map.mean().detach()


    # 3. Create the Soft Mask (Differentiable Step)
    # The input to the Sigmoid is: gamma * (Attention_value - Threshold)
    # This acts like a differentiable approximation of the step function: Attention_value > Threshold
    logits = steepness_gamma * (attention_map - threshold_tensor)
    soft_mask = torch.sigmoid(logits)


    # 4. Calculate the Hard Mask (for evaluation/debugging, non-differentiable)
    # hard_mask_np = (attention_map_np > threshold_value_np).astype(np.uint8)

    # return soft_mask, hard_mask_np
    return soft_mask


# 참고
# def attn_map_to_binary(attention_map, scaler=1.):
def attn_map_to_hard_mask(attention_map, scaler=1.):
    # scaler < 1.0: lower threshold (softer/larger mask)
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.detach()
    attention_map_np = attention_map.cpu().numpy()
    threshold_value = filters.threshold_otsu(attention_map_np) * scaler
    binary_mask = (attention_map_np > threshold_value).astype(np.uint8)

    return binary_mask # = hard_mask


def process_attn_differentiable(attentions, prompt_len: int, span_indices: list,
                                skip_layer_idx_list: list = [], img_size: int = 24, 
                                aggregate: str = "mean", normalize: bool = True, 
                                scaler: float = 1.0, gamma: float = 100.0, eps: float = 1e-8):

    assert max(skip_layer_idx_list) < 30 # total 30 layer

    # --- COLLECT attention maps across layers ---
    collected = []

    for layer_idx, layer_attn in enumerate(attentions):
        if layer_idx in skip_layer_idx_list:
            continue

        # layer_attn: [B, H, Q, K]
        span_attn = layer_attn[0, :, prompt_len:, span_indices].float()
        
        # aggregate across heads
        if aggregate == "mean":
            span_attn = span_attn.mean(dim=0)  # [K]
        elif aggregate == "sum":
            span_attn = span_attn.sum(dim=0)   # [K]
        else:
            raise NotImplementedError(f"Unsupported aggregate: {aggregate}")
        collected.append(span_attn)


    # --- AGGREGATE across layers and span_indices ---
    stacked = torch.stack(collected)
    attn_out = stacked.mean(dim=0).mean(dim=-1) # Mean across layers(dim=0) and span_indices(dim=-1)


    # --- NORMALIZE --- 
    if normalize:
        attn_out = (attn_out - attn_out.min()) / (attn_out.max() - attn_out.min() + eps)

    assert attn_out.shape[0] == img_size * img_size, f"attn_out shape is wrong: {attn_out.shape}"
    attn_map = attn_out.reshape(img_size, img_size) # This is the continuous map


    # --- (IMPORTANT) Differentiable Mask Generation ---
    soft_mask = attn_map_to_soft_mask(
        attention_map=attn_map, 
        otsu_scaler=scaler, 
        steepness_gamma=gamma,
        use_hard_threshold_for_otsu=True # Use the OTSU value as the Sigmoid anchor
    )

    # return attn_map, soft_mask, hard_mask
    return attn_map, soft_mask
