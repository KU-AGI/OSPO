
import os
import argparse
from pathlib import Path
from itertools import repeat
from tqdm import tqdm
import multiprocessing as mp
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.common import read_json, save_json


# off_policy 미고려
def _check_masks_tuple(args):
    """
    args: (example, mask_dir)
    Return the example if both masks exist, else None.
    """
    ex, mask_dir = args

    basename = Path(ex["chosen"]).stem  # safer/faster than split(".png")[0]

    # NOTE: using distinct subdirs "base" and "negative" per your code
    c_mask_path = Path(mask_dir) / "base"     / ex["t2i_category"] / ex["item_id"] / f"{basename}_mask.pt"
    r_mask_path = Path(mask_dir) / "negative" / ex["t2i_category"] / ex["item_id"] / f"{basename}_mask.pt"

    if c_mask_path.exists() and r_mask_path.exists():
        return ex
    
    return None


def main(args):

    dataset = read_json(args.train_path)

    filtered = []
    missed = 0

    # Tune workers/chunksize as needed (32/256–1024 are good starting points for NAS)
    num_workers = min(32, os.cpu_count() or 1)
    chunk = 512

    with mp.get_context("fork").Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(
            _check_masks_tuple,
            zip(dataset, repeat(args.mask_dir)),
            chunksize=chunk
        )
        for res in tqdm(iterator, total=len(dataset), desc="Checking masking (MP) ..."):
            if res is not None:
                filtered.append(res)
            else:
                missed += 1

    print(f"Kept: {len(filtered)} | Filtered (no mask): {missed}")

    if len(filtered) > 0:
        save_root = os.path.dirname(args.train_path)
        os.makedirs(save_root, exist_ok=True)

        # SAVE
        save_json(
            save_root=save_root,
            save_name=args.save_name.format(len(filtered)),
            save_file=filtered
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="/nas2/data/Janus_dataset/next_v2/unitok/dataset/train/tau_06/train_mode_argmax_chosen_5325_tau_object_5321.json")
    parser.add_argument("--mask_dir", type=str, default="/nas2/data/Janus_dataset/next_v2/unitok/dataset/train/tau_06/object_mask")
    parser.add_argument("--save_name", type=str, default="train_filtered_{}")
    args, unknown = parser.parse_known_args()  

    main(args)