# Stage 2 Process 에 대한 Ablation 

# 기존 데이터 합치기 (약 32k)

import os, json
import random
from tqdm import tqdm
from math import inf

# 두 파일의 파일형식 다름 주의
# path1 = "/nas2/data/Janus_dataset/next_v2/merged_vqa_result_27807.json"
path2 = "/nas2/data/Janus_dataset/main_exp/janus/prompt/janus_vqa/feedback_combined_15991_refine_v2.json"

# with open(path1, "r") as f:
#     data1 = json.load(f)

with open(path2, "r") as f:
    data2 = json.load(f)



# -


# THRESHOLD = 0.6
# 별도의 필터링/셀렉션 과정 진행 없이, 단순히 랜덤한 하나의 페어를 선택한다.


# -


# TODO 1: Object Mask 는 추출 필요
train_dataset = []

# idx2category = {"0": "attribute", "1": "layout", "2": "non-spatial", "3": "complex"}

for item in tqdm(data2, desc="Processing data 2"):
    item_id_str = str(item.get("item_id", ""))
    if not item_id_str:
        continue  # or handle

    train_sample = {
        "item_id": item["item_id"],
        "t2i_category": item["t2i_category"],
        "prompt": item["prompt"],
    }

    meta_all = item["metadata"]
    
    chosen_candidate = []
    rejected_candidate = []

    for k, v in meta_all.items():
        if v is None:
            continue    
            
        if k.startswith("base"):
            chosen_candidate.append(v["path"])
        elif k.startswith("negative"):
            rejected_candidate.append(v["path"])
        else:
            raise ValueError(f"Unknown key in metadata: {k}")


    # paring chosen and rejected with those have save basename
    paired_candidates = []
    for c in chosen_candidate:
        c_base = os.path.basename(c)
        for r in rejected_candidate:
            r_base = os.path.basename(r)
            if c_base == r_base:
                paired_candidates.append( (c, r) )
                break

    # randomly select one pair
    if len(paired_candidates) == 0:
        continue

    random_idx = random.randint(0, len(paired_candidates) - 1)
    chosen, rejected = paired_candidates[random_idx]

    train_sample["chosen"] = chosen
    train_sample["rejected"] = rejected
    train_dataset.append(train_sample)


print("Total processed samples from data2:", len(train_dataset))

# save_path
save_path = "/home/yjoh/project/OSPO/ospo/rebuttal/compute/stage4_abl_data2_train.json"
with open(save_path, "w") as f:
    json.dump(train_dataset, f, indent=4)
print(f"Saved processed data2 to {save_path}")