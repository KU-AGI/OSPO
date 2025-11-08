# make it into preference data style
import os, json
from tqdm import tqdm

dpath = "/nas2/mllm_reasoning/experiments/datasets/official_focus_diff/data/labels/metadata.json"
with open(dpath, "r") as f:
    data = json.load(f)


train_dataset = []

for d in tqdm(data):
    img_dir = "/nas2/mllm_reasoning/experiments/datasets/official_focus_diff/data/images"
    cpath = os.path.join(img_dir, d['image1'][1:])
    rpath = os.path.join(img_dir, d['image2'][1:])

    if os.path.exists(cpath) and os.path.exists(rpath):    
        train_dataset.append({
            "item_id": d['ids'],
            "t2i_category": "FocusDiff",
            "sub_category": "FocusDiff",
            # prompt 1 기준
            "prompt": d['prompt1'],
            "chosen": cpath,
            "rejected": rpath
        })

print(f"Total Length: {len(train_dataset)}")

save_dir = "/nas2/data/Janus_dataset/next_v2/ablation/focusdiff"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, f"train_{len(train_dataset)}.json")

with open(save_path, "w") as f:
    json.dump(train_dataset, f, indent=4)