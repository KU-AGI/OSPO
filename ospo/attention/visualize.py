import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import matplotlib.cm as cm

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


        """Yellow - Blue version"""
        # # normalize to [0, 1]
        # heatmap = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
        # heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # # apply viridis colormap (Matplotlib)
        # heatmap_color = cm.viridis(heatmap)[:, :, :3]   # remove alpha channel
        # heatmap_color = np.uint8(heatmap_color * 255)

        # # blend overlay (viridis heatmap + original image)
        # overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)

        # out_path = os.path.join(save_dir, f"{base_name}_overlay.png")
        # cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # print(f"[Saved] → {save_dir}")
