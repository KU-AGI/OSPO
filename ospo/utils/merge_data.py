import os, json
from tqdm import tqdm

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from ospo.utils.common import read_json, save_json


def merge(base_dir, start_key: str, save_fname: str = None):
    file_list = [f for f in os.listdir(base_dir) if f.startswith(start_key)]

    merged_data = []
    for f in file_list:
        data = read_json(os.path.join(base_dir, f))
        merged_data.extend(data)
    
    # SORTING
    merged_data = sorted(merged_data, key=lambda x: x.get("item_id", -1))

    print("Total merged length: ", len(merged_data))
    save_name = save_fname.strip() + f"_{len(merged_data)}"

    if save_fname is not None:
        save_path = os.path.join(base_dir, f'{save_name}.json')
        save_json(save_root=base_dir, save_name=save_name, save_file=merged_data)
        print(f"Saved at {save_path}")

    return merged_data


def compose_init_data(dense_path, img_path):
    dense_data = read_json(dense_path)

    init_data = []
    for d in tqdm(dense_data):
        item_id = d["item_id"]
        t2i_category = d["t2i_category"]

        chosen_base_dir = os.path.join(img_path, "base", t2i_category, item_id)
        rejected_base_dir = os.path.join(img_path, "negative", t2i_category, item_id)

        chosen_candidates = [os.path.join(chosen_base_dir, f) for f in os.listdir(chosen_base_dir) if f.endswith(".png")]
        rejected_candidates = [os.path.join(rejected_base_dir, f) for f in os.listdir(rejected_base_dir) if f.endswith(".png")]

        # 키값 추가
        d["chosen_candidate"] = chosen_candidates
        d["rejected_candidate"] = rejected_candidates

        init_data.append(d)

    return init_data




# # 1. merge negative prompt
# base_dir = "/nas2/data/Janus_dataset/next_v2/iter2/prompt/step2"
# start_key = "negative_prompt"
# _ = merge(base_dir, start_key, save_fname=f"{start_key}_merged")

# # 2. merge dense prompt
# start_key = "dense_prompt"
# _ = merge(base_dir, start_key, save_fname=f"{start_key}_merged")

# # # 3. merge vqa question
# base_dir = "/nas2/data/Janus_dataset/next_v2/iter2/prompt/step3_vqa"
# start_key = "vqa_question"
# _ = merge(base_dir, start_key, save_fname=f"{start_key}_merged")

# # 4. merge vqa result
# start_key = "vqa_result"
# _ = merge(base_dir, start_key, save_fname=f"{start_key}_merged")


# 5. init data (for constructing training data)
dense_path = "/nas2/data/Janus_dataset/next_v2/iter2/prompt/dense_prompt_merged_16006.json"
img_path = "/nas2/data/Janus_dataset/next_v2/iter2/images"

init_data = compose_init_data(dense_path, img_path)
save_json(save_root=os.path.dirname(dense_path), save_name=f"init_dataset_{len(init_data)}", save_file=init_data)

