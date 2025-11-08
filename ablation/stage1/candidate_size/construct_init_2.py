# candidate pair size 1 - 3 (기준) - 5
# 또는
# candidate pair size 1 - 3 (기준) - 6 - 9

import os
import json
import random
from copy import deepcopy
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from tqdm import tqdm
from ospo.utils.common import read_json, save_json

def resize_pair_candidate(init_dataset, n):
    assert n in [1, 5, 6, 9], f"Unsupported size: {n}"
    resized_dataset = []

    def matching_negative(path):
        """Return the mirrored negative path if it exists, else None."""
        neg = path.replace("base", "negative")
        return neg if os.path.exists(path) and os.path.exists(neg) else None

    for d0 in tqdm(init_dataset):
        d = deepcopy(d0)

        chosen0 = list(d.get('chosen_candidate', []) or [])
        rejected0 = list(d.get('rejected_candidate', []) or [])
        org_c, org_r = len(chosen0), len(rejected0)

        if org_c == 0 or org_r == 0:
            continue

        # -------- n == 1: pick the first chosen that has a matching rejected with same basename --------
        if n == 1:
            first = chosen0[0]
            base_name = os.path.basename(first)
            r_matched = None
            for r in rejected0:
                if os.path.basename(r) == base_name:
                    r_matched = r
                    break
            if r_matched is None:
                continue

            d['chosen_candidate'] = [first]
            d['rejected_candidate'] = [r_matched]
            d.pop('ablation', None)
            resized_dataset.append(d)
            continue

        # -------- n in {5, 6, 9}: start from existing aligned pairs by basename --------
        # Align existing pairs by basename to avoid mis-ordered lists
        rej_by_name = {os.path.basename(r): r for r in rejected0}
        aligned_c = []
        aligned_r = []
        for c in chosen0:
            name = os.path.basename(c)
            if name in rej_by_name:
                aligned_c.append(c)
                aligned_r.append(rej_by_name[name])

        # If nothing aligns, skip
        if not aligned_c:
            continue

        # Collect additional pairs from alternate folders
        # pass 1: _2 (or seed_345)
        extras_c = []
        extras_r = []

        t2i_category = d.get('t2i_category')
        try:
            item_id_int = int(d.get('item_id'))
        except Exception:
            item_id_int = -1

        def add_extra_versions(src_paths, variant_tag):
            """variant_tag in {'_2','_3','seed_345','seed_678'}"""
            for c in src_paths:
                if t2i_category == "layout" and item_id_int >= 1004000:
                    if variant_tag == '_2':
                        c_alt = c.replace("seed_012", "seed_345")
                    elif variant_tag == '_3':
                        c_alt = c.replace("seed_012", "seed_678")
                    else:
                        c_alt = c  # fallback
                else:
                    if variant_tag == '_2':
                        c_alt = c.replace("images_pairwise", "images_pairwise_2")
                    elif variant_tag == '_3':
                        c_alt = c.replace("images_pairwise", "images_pairwise_3")
                    else:
                        c_alt = c  # fallback

                r_alt = matching_negative(c_alt)
                if r_alt is not None:
                    extras_c.append(c_alt)
                    extras_r.append(r_alt)

        # First layer of extras
        add_extra_versions(aligned_c, '_2')

        # For n == 9, add a second layer
        if n == 9:
            add_extra_versions(aligned_c, '_3')

        # Build final lists: start with aligned originals, then extras, cap to n
        final_c = (aligned_c + extras_c)[:n]
        final_r = (aligned_r + extras_r)[:n]

        # Ensure we have the same count and at least 1
        m = min(len(final_c), len(final_r))
        if m == 0:
            continue
        if m < n:
            # not enough extras to reach n; skip or keep partial — here we keep partial.
            final_c = final_c[:m]
            final_r = final_r[:m]

        d['chosen_candidate'] = final_c
        d['rejected_candidate'] = final_r
        d.pop('ablation', None)
        resized_dataset.append(d)

    print(f"Total pair resized data length (N={n}): {len(resized_dataset)}")
    return resized_dataset

if __name__ == '__main__':
    
    init_path = "/nas2/data/Janus_dataset/next_v2/init_dataset_16001.json"
    init = read_json(init_path)

    for n in [5, 6, 9]:
        data = resize_pair_candidate(init, n)
        save_json(
            save_root='/nas2/data/Janus_dataset/next_v2/ablation/pair_size',
            save_name=f'pair_size_{n}',
            save_file=data
        )

    print('Done.')