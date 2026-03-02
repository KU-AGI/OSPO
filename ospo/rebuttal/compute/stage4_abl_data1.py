# Stage 2 Process 에 대한 Ablation 

# 기존 데이터 합치기 (약 32k)

import os, json
import random
from tqdm import tqdm
from math import inf

# 두 파일의 파일형식 다름 주의
path1 = "/nas2/data/Janus_dataset/next_v2/merged_vqa_result_27807.json"
path2 = "/nas2/data/Janus_dataset/main_exp/janus/prompt/janus_vqa/feedback_combined_15991_refine_v2.json"

with open(path1, "r") as f:
    data1 = json.load(f)

with open(path2, "r") as f:
    data2 = json.load(f)



# -


# THRESHOLD = 0.6
# 별도의 필터링/셀렉션 과정 진행 없이, 단순히 랜덤한 하나의 페어를 선택한다.


# -


# TODO 1: Object Mask 는 추출 필요
train_dataset = []

idx2category = {"0": "attribute", "1": "layout", "2": "non-spatial", "3": "complex"}


for item in tqdm(data1, desc="Processing data 1"):
    item_id_str = str(item.get("item_id", ""))
    if not item_id_str:
        continue  # or handle

    train_sample = {
        "item_id": item["item_id"],
        "t2i_category": idx2category.get(item_id_str[0], "unknown"),
        "prompt": item["prompt"],
    }

    base_meta = item["base_metadata"]
    negative_meta = item["negative_metadata"] 

    total_n = min(len(base_meta), len(negative_meta))
    # 0 ... total_n - 1 까지 중 랜덤하게 하나의 인덱스 선택

    if total_n <= 0:
        continue

    random_idx = random.randint(0, total_n - 1)
    chosen = base_meta[f"base_{random_idx}"]["path"]
    rejected = negative_meta[f"negative_{random_idx}"]["path"]

    skip_f = False

    if os.path.basename(chosen) != os.path.basename(rejected):
        # rejected 재선택
        for k, v in negative_meta.items():
            if os.path.basename(v["path"]) == os.path.basename(chosen):
                rejected = v["path"]
                break

            # reselect chosen and repeat the process
            random_idx = random.randint(0, total_n - 1)
            chosen = base_meta[f"base_{random_idx}"]["path"]
            rejected = negative_meta[f"negative_{random_idx}"]["path"]

            # They must be same.
            if os.path.basename(chosen) != os.path.basename(rejected):
                print("Re-selected chosen and rejected pair failed.")
                skip_f = True
                break
    else:
        pass

    if skip_f:
        continue

    train_sample["chosen"] = chosen
    train_sample["rejected"] = rejected
    train_dataset.append(train_sample)


print("Total processed samples from data1:", len(train_dataset))

# save_path
save_path = "/home/yjoh/project/OSPO/ospo/rebuttal/compute/stage4_abl_data1_train.json"
with open(save_path, "w") as f:
    json.dump(train_dataset, f, indent=4)
print(f"Saved processed data1 to {save_path}")