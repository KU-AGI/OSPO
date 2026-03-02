import os, json
from tqdm import tqdm
import uuid

MAX_CHUNK = 30
DATA_SIZE_MUST_BE = 16968
IMG_ROOT = "/nas2/data/HPDv2/split"

# Each last png name in split "00000N":
IMG_INDEX_DICT = dict()
for i in range(MAX_CHUNK): 
    dir_path = os.path.join(IMG_ROOT, f"out{i:03d}")
    if not os.path.exists(dir_path):
        raise ValueError(f"Directory does not exist: {dir_path}")
    
    file_list = sorted(os.listdir(dir_path))  # 디렉토리 내용 확인 (필요시 출력 가능)
    start_file = file_list[0] if file_list else "No files found"
    last_file = file_list[-1] if file_list else "No files found"

    # save, to look up 
    IMG_INDEX_DICT[i] = {
        "dir_path": dir_path,
        "start_idx": int(os.path.splitext(start_file)[0]),
        "last_idx": int(os.path.splitext(last_file)[0])
    }

VERY_LAST_IDX = IMG_INDEX_DICT[MAX_CHUNK - 1]["last_idx"]


# 기본 데이터 로드
org_path = "/nas2/data/HPDv2/train.json"
with open(org_path, 'r') as f:
    org_data = json.load(f)


# org_data = org_data[:DATA_SIZE_MUST_BE * 2]  # 원본이 너무 커서 일단 절반만 사용


def get_img_root(img_fpath):
    img_idx = int(os.path.splitext(img_fpath)[0])
    
    # last index 
    if img_idx > VERY_LAST_IDX:
        return None

    for key, val in IMG_INDEX_DICT.items():
        if val["start_idx"] <= img_idx <= val["last_idx"]:
            return val["dir_path"]

    return None


# 학습 데이터 축적
train_data = []

for item in tqdm(org_data):

    if len(item["human_preference"]) != 2:
        continue

    out_root_0 = get_img_root(item["image_path"][0])
    out_root_1 = get_img_root(item["image_path"][1])

    if out_root_0 is None or out_root_1 is None:
        # print(f"Image not found in any root: {item['image_path']}")
        continue

    final_root_0 = os.path.join(IMG_ROOT, out_root_0, item["image_path"][0])
    final_root_1 = os.path.join(IMG_ROOT, out_root_1, item["image_path"][1])

    if not os.path.exists(final_root_0) or not os.path.exists(final_root_1):
        # print(f"Image file does not exist: {final_root_0} or {final_root_1}")
        continue

    # 본격적으로,
    if item["human_preference"][0] == 1 and item["human_preference"][1] == 0:        
        chosen_path = final_root_0
        reject_path = final_root_1

    elif item["human_preference"][1] == 1 and item["human_preference"][0] == 0:
        chosen_path = final_root_1
        reject_path = final_root_0
    
    else:
        print(item)
        raise ValueError("Both labels are the same, cannot determine chosen and reject images.")


    train_data.append({
        "item_id": None, # str(uuid.uuid4()),
        "t2i_category": "hpd_v2",
        "sub_category": "hpd_v2",
        "prompt": item["prompt"],
        "chosen": chosen_path,
        "rejected": reject_path,
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
save_path = f"/nas2/checkpoints/janus_dpo_rebuttal/data/external/new_train_hpd_v2_{len(train_data)}.json"
with open(save_path, 'w') as f:
    json.dump(train_data, f, indent=4)
print(f"Training data saved to {save_path}")