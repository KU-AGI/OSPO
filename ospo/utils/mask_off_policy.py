
import os
from pathlib import Path
from itertools import repeat
from tqdm import tqdm
import multiprocessing as mp

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.common import read_json, save_json

TRAIN_DATA_PATH = "/nas2/data/Janus_dataset/next_v2/ablation/pickapic2/train_object_16838.json"
TRAIN_MASK_DIR = "/nas2/data/Janus_dataset/next_v2/ablation/pickapic2/object_mask/data/pickapic_v2/images" # all-in-one (NEVER OPEN)
dataset = read_json(TRAIN_DATA_PATH)

save_root = os.path.dirname(TRAIN_DATA_PATH)
# save_name = "train_mode_argmax_chosen_filtered_{}"
save_name = "train_filtered_{}"


# /nas2/data/Janus_dataset/next_v2/ablation/pickapic2/object_mask/data/pickapic_v2/images/1c6649563f83ae8179a237b6b2429bc1f99bc3a2.jpg_attn_heatmap.png

def _check_masks_tuple(args, focusdiff: bool = False):
    """
    args: (example, mask_dir)
    Return the example if both masks exist, else None.
    """
    ex, mask_dir = args

    if not focusdiff: # pickapic2
        c_basename = os.path.basename(ex["chosen"]) # Path(ex["chosen"]) # .stem  # safer/faster than split(".png")[0]
        r_basename = os.path.basename(ex["rejected"])

        c_mask_path = f"{mask_dir}/{c_basename}_mask.pt" # Path(mask_dir) / f"{c_basename}_mask.pt" # 1c6649563f83ae8179a237b6b2429bc1f99bc3a2.jpg_mask.pt
        r_mask_path = f"{mask_dir}/{r_basename}_mask.pt" # Path(mask_dir) / f"{r_basename}_mask.pt"
        
    else:
        c_mask_path = Path(mask_dir) / "data" / "images" / ex["item_id"] / "image1_mask.pt"
        r_mask_path = Path(mask_dir) / "data" / "images" / ex["item_id"] / "image2_mask.pt"

    # if c_mask_path.exists() and r_mask_path.exists():
    if os.path.exists(c_mask_path) and os.path.exists(r_mask_path):
        return ex
    
    return None


if __name__ == "__main__":
    filtered = []
    missed = 0

    # Tune workers/chunksize as needed (32/256–1024 are good starting points for NAS)
    num_workers = min(32, os.cpu_count() or 1)
    chunk = 512

    with mp.get_context("fork").Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(
            _check_masks_tuple,
            zip(dataset, repeat(TRAIN_MASK_DIR)),
            chunksize=chunk
        )
        for res in tqdm(iterator, total=len(dataset), desc="Checking masking (MP) ..."):
            if res is not None:
                filtered.append(res)
            else:
                missed += 1

    print(f"Kept: {len(filtered)} | Filtered (no mask): {missed}")

    if len(filtered) > 0:
        # SAVE
        save_json(
            save_root=save_root,
            save_name=save_name.format(len(filtered)),
            save_file=filtered
        )