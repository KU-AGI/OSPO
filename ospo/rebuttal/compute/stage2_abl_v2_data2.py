# Stage 2 Process 에 대한 Ablation 

# 기존 데이터 합치기 (약 32k)

import os, json
from tqdm import tqdm
from math import inf

# 두 파일의 파일형식 다름 주의
path1 = "/nas2/data/Janus_dataset/main_exp/janus/silmm/prompt/vqa/feedback_combined_with_global_metadata_16000.json"
path2 = "/nas2/data/Janus_dataset/main_exp/aaai/janus/prompt/negative_short/train_dataset/silmm/silmm_data_15988.json"

with open(path1, "r") as f:
    data1 = json.load(f)

with open(path2, "r") as f:
    data2 = json.load(f)



# -


THRESHOLD = 0.6


# TODO 2: VQA based Filtering Logic
# (수정 버전) 필터링 > 셀렉션 순서 / 똑같은 로직으로 적용하는데, 다만 이제 C/R 후보군 나눠서 보는 것이 아니라 매번 모든 샘플들에서 보는 것 차이
# 1. ALL No 인 것은 Chosen 이 될 수 없음.
# 2. 모든 질문에 대한 평균 점수가 Threshold 이하인 것은 Chosen 이 될 수 없음.
# 3. All Yes 인 것은 Rejected 이 될 수 없음.

# TODO 3: <수정> 페어단위 (짝꿍단위) 라는 것이 없으므로 Chosen 과 Rejected 의 스코어가 가장 낮은 쪽이 선택된다.



# -



# TODO 1: Object Mask 는 추출 필요
train_dataset = []


idx2category = {"0": "attribute", "1": "layout", "2": "non-spatial", "3": "complex"}

def norm_answer(a):
    return (a or "").strip().lower()

for item in tqdm(data2, desc="Processing data 2"):
    item_id_str = str(item.get("item_id", ""))
    if not item_id_str:
        continue  # or handle

    train_sample = {
        "item_id": item["item_id"],
        "t2i_category": item["t2i_category"],  
        "prompt": item["prompt"],
    }

    meta_by_img_dict = item.get("metadata", {})
    q_len = len(item.get("question", []))
    if not meta_by_img_dict or q_len == 0:
        continue

    # Precompute per-image flags once
    imgs = []
    for img_idx, img_dict in meta_by_img_dict.items():
        score = img_dict.get("local_score", float("-inf"))
        path = img_dict.get("path", None)

        # If path missing, you might want to skip this img
        if path is None:
            continue

        # Compute all_yes / all_no once
        all_yes = True
        all_no = True
        for raw_answer in img_dict.get("answer_metadata", []):
        # for qid in range(q_len):
            ans = norm_answer(raw_answer.get("answer"))
            if ans != "yes":
                all_yes = False
            if ans != "no":
                all_no = False
            if not all_yes and not all_no:
                break

        imgs.append((img_idx, path, score, all_yes, all_no))

    # CHOSEN: max score among score>=THRESHOLD and NOT all_no
    chosen = None
    best_score = float("-inf")
    for img_idx, path, score, all_yes, all_no in imgs:
        if score >= THRESHOLD and (not all_no):
            if score > best_score:
                best_score = score
                chosen = (img_idx, path, score)

    if chosen is None:
        # print("Skip (DISCARDED: No valid Chosen)")
        continue

    # REJECTED: min score among NOT all_yes and not chosen
    rejected = None
    min_score = float("inf")
    for img_idx, path, score, all_yes, all_no in imgs:
        if img_idx == chosen[0]:
            continue
        if all_yes:
            continue
        if score < min_score:
            min_score = score
            rejected = (img_idx, path, score)

    if rejected is None:
        # print("Skip (DISCARDED: No valid Rejected)")
        continue

    train_sample["chosen"] = chosen[1]
    train_sample["rejected"] = rejected[1]
    train_dataset.append(train_sample)


print("Total processed samples from data1:", len(train_dataset))

# save_path
# save_path = "/home/yjoh/project/OSPO/ospo/rebuttal/compute/stage2_abl_data1_train.json"
save_path = "/home/yjoh/project/OSPO/ospo/rebuttal/compute/stage2_abl_data2_train.json"
with open(save_path, "w") as f:
    json.dump(train_dataset, f, indent=4)
print(f"Saved processed data2 to {save_path}")