import os, json
from tqdm import tqdm
import uuid


DATA_SIZE_MUST_BE = 16968

# 학습 데이터 만들기
org_path = "/nas2/data/pickapic_v2/data/merged_00_12_19331.json"
with open(org_path, 'r') as f:
    org_data = json.load(f)


train_data = []

for item in tqdm(org_data):
    if item["label_0"] == 1.0 and item["label_1"] == 0.0: 
        chosen_path = item["jpg_0"]
        reject_path = item["jpg_1"]
    elif item["label_1"] == 1.0 and item["label_0"] == 0.0:
        chosen_path = item["jpg_1"]
        reject_path = item["jpg_0"]
    elif item["label_1"] == item["label_0"]:
        # print("Same labels found:")
        continue
    else:
        print(item)
        raise ValueError("Both labels are the same, cannot determine chosen and reject images.")

    if os.path.exists(chosen_path) is False or os.path.exists(reject_path) is False:
        print("File not found, skipping:")
        print(f"  Chosen: {chosen_path}")
        print(f"  Rejected: {reject_path}")
        continue

    train_data.append({
        "item_id": None, # str(uuid.uuid4()),
        "t2i_category": "pickapic_v2",
        "sub_category": "pickapic_v2",
        "prompt": item["caption"],
        "chosen": chosen_path,
        "rejected": reject_path,
        "best_image_uid": item["best_image_uid"]
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
save_path = f"/nas2/checkpoints/janus_dpo_rebuttal/data/external/train_pickapic_v2_{len(train_data)}.json"
with open(save_path, 'w') as f:
    json.dump(train_data, f, indent=4)
print(f"Training data saved to {save_path}")