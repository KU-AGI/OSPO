import os, json
from tqdm import tqdm
import uuid


DATA_SIZE_MUST_BE = 16968

# 학습 데이터 만들기
org_path = "/nas2/mllm_reasoning/experiments/datasets/official_focus_diff/data/labels/metadata.json"
img_path = "/nas2/mllm_reasoning/experiments/datasets/official_focus_diff/data/images"
with open(org_path, 'r') as f:
    org_data = json.load(f)

org_data = org_data[:DATA_SIZE_MUST_BE * 2]  # 원본이 너무 커서 일단 절반만 사용

train_data = []


for item in tqdm(org_data):
    prompt = item["prompt1"]

    # sub_dir = item["image1"].split("/")[-2][:2]
    # if int(sub_dir) > 14:
    #     print("Skipping image beyond 14:")
    #     continue

    image1_rel = item["image1"].lstrip("/")   # remove leading /
    image2_rel = item["image2"].lstrip("/")

    sub_dir = image1_rel.split("/")[-2][:2]
    
    chosen_path = os.path.join(img_path, sub_dir, image1_rel)
    rejected_path = os.path.join(img_path, sub_dir, image2_rel)

    if os.path.exists(chosen_path) is False or os.path.exists(rejected_path) is False:
        print("File not found, skipping:")
        print(f"  Chosen: {chosen_path}")
        print(f"  Rejected: {rejected_path}")
        continue

    train_data.append({
        "item_id": None, # str(uuid.uuid4()),
        "t2i_category": "focusdiff",
        "sub_category": "focusdiff",
        "prompt": prompt,
        "chosen": chosen_path,
        "rejected": rejected_path,
        "ids": item["ids"]
    })

print(f"Original data size: {len(org_data)}")
print(f"Training data size: {len(train_data)}")

if len(train_data) != DATA_SIZE_MUST_BE:
    # truncate
    # train_data = train_data[:DATA_SIZE_MUST_BE]

    # random sample
    import random
    train_data = random.sample(train_data, DATA_SIZE_MUST_BE)
    print(f"Truncated training data size to {DATA_SIZE_MUST_BE}")


# give item_id
for i in range(len(train_data)):
    train_data[i]["item_id"] = f"{i:07d}"

# save
save_path = f"/nas2/checkpoints/janus_dpo_rebuttal/data/external/train_focusdiff_{len(train_data)}.json"
with open(save_path, 'w') as f:
    json.dump(train_data, f, indent=4)
print(f"Training data saved to {save_path}")