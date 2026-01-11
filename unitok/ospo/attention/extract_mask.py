import os
import json
import torch
import cv2
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt
import pyrootutils
import matplotlib.cm as cm
from PIL import Image

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

def save_attn_npy(attn_map_grid, save_path):
    # save_path = f"attn_maps/{item_id}_attn.npy"
    np.save(save_path, attn_map_grid.cpu().numpy())


def visualize_save_attn(attn_map, binary_mask, save_dir, base_name="attn", orig_image=None):
    """
    Save attention heatmap and binary mask to disk.
    
    Args:
        attn_map (torch.Tensor): 2D attention map (HxW)
        binary_mask (np.ndarray): 2D binary mask (HxW)
        save_dir (str): directory to save images
        base_name (str): base filename prefix
        orig_image (PIL.Image or np.ndarray, optional): if given, also save overlay image
    """
    os.makedirs(save_dir, exist_ok=True)
    attn_np = attn_map.detach().cpu().numpy()
    mask_np = binary_mask.astype(np.uint8) * 255  # scale to 0–255

    # ----- Save raw attention heatmap -----
    plt.figure(figsize=(5, 5))
    plt.imshow(attn_np, cmap="hot")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"{base_name}_attn_heatmap.png"),
                bbox_inches="tight", pad_inches=0)
    plt.close()

    # ----- Save binary mask -----
    plt.figure(figsize=(5, 5))
    plt.imshow(mask_np, cmap="gray")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"{base_name}_binary_mask.png"),
                bbox_inches="tight", pad_inches=0)
    plt.close()

    # ----- Optional: overlay heatmap on original image -----
    if orig_image is not None:
        if isinstance(orig_image, Image.Image):
            image = np.array(orig_image.convert("RGB"))
        else:
            image = orig_image

        """Red version"""
        # resize heatmap to match image size
        heatmap = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.5, color_map, 0.5, 0)

        out_path = os.path.join(save_dir, f"{base_name}_overlay.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))



def attn_map_to_binary(attention_map, scaler=1.):
    # scaler < 1.0: lower threshold (softer/larger mask)
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.detach()
    attention_map_np = attention_map.cpu().numpy()
    threshold_value = filters.threshold_otsu(attention_map_np) * scaler
    binary_mask = (attention_map_np > threshold_value).astype(np.uint8)

    return binary_mask

# Process 1D attention vector
def process_attn(attentions, prompt_len: int, span_indices: list,
                skip_layer_idx_list: list = [], img_size: int = 16, aggregate: str = "mean", normalize: bool = True, scaler: float = 1.0, eps: float = 1e-8): # layer_idx: int = None):

    assert max(skip_layer_idx_list) < 32 # total 32 layer

    # --- COLLECT attention maps across layers ---
    collected = []

    for layer_idx, layer_attn in enumerate(attentions):
        if layer_idx in skip_layer_idx_list:
            continue

        # print(layer_attn) # torch.Size([1, 32, 274, 274]) - Unitok Update

        # layer_attn: [B, H, Q, K]
        assert len(layer_attn) == 1, "Batch size must be 1."
        # span_attn = layer_attn[0, :, prompt_len:, span_indices].float()
        span_attn = layer_attn[0, :, prompt_len-1:, span_indices].float() # 임시조치

        # aggregate across heads
        if aggregate == "mean":
            span_attn = span_attn.mean(dim=0)  # [K]
        elif aggregate == "sum":
            span_attn = span_attn.sum(dim=0)   # [K]
        else:
            raise NotImplementedError(f"Unsupported aggregate: {aggregate}")
        
        collected.append(span_attn)

        # print("span_attn")
        # print(collected[0].shape) 
        # Janus-Pro: torch.Size([584, 2])
        # Unitok: torch.Size([255, 1]) - Unitok 에서는 이미지 토큰이 총 256개

    # --- AGGREGATE across layers ---
    if isinstance(collected[0], torch.Tensor):
        stacked = torch.stack(collected)  # (L, ...) = (num_layers, K), K = number of token span
        attn_out = stacked.mean(dim=0) if aggregate != "none" else stacked
    else:
        return None


    # --- AGGREGATE across span_indices ---
    if attn_out.ndim > 1:
        attn_out = attn_out.mean(dim=-1)
    # print("attn_out") # (589,)   # after aggregation > IMAGE TOKEN ONLY (attn_out.shape = (576,))


    # --- NORMALIZE --- 
    if normalize and aggregate != "none":
        attn_out = (attn_out - attn_out.min()) / (attn_out.max() - attn_out.min() + 1e-8)

    assert attn_out.shape[0] == img_size * img_size, f"attn_out shape is wrong: {attn_out.shape}"
    attn_map = attn_out.reshape(img_size, img_size)
    binary_mask = attn_map_to_binary(attn_map, scaler=scaler)

    return attn_map, binary_mask



def save_binary_mask(binary_mask, save_path):
    assert save_path.endswith(".pt"), "save path must be ended with '.pt'! "

    # Ensure tensor format
    if isinstance(binary_mask, list):
        binary_mask = torch.tensor(binary_mask, dtype=torch.uint8)
    elif isinstance(binary_mask, np.ndarray):
        binary_mask = torch.from_numpy(binary_mask.astype(np.uint8))
    else:
        raise ValueError(f"Unknown type: {binary_mask}")

    torch.save(binary_mask, save_path)
    # print(f"[Saved mask] {save_path}")
    return

