# Stage 2 Process 에 대한 Ablation 

# 기존 데이터 합치기 (약 32k)

import os, json
from tqdm import tqdm

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

def filter_sample():
    pass

# TODO 3: <수정> 페어단위 (짝꿍단위) 라는 것이 없으므로 Chosen 과 Rejected 의 스코어가 가장 낮은 쪽이 선택된다.


# TODO 1: Object Mask 는 추출 필요
train_dataset = []

IDX2CATEGORY = {
    "0": "attribute",
    "1": "layout",
    "2": "non-spatial",
    "3": "complex"
}
    
""" Global Metadata """
for item in tqdm(data1, desc="Processing data 1"):
    
    train_sample = {
        "item_id": item['item_id'],
        "t2i_category": IDX2CATEGORY[item['item_id'][0]],
        "prompt": item['prompt'],
    }

    # find chosne / rejected based on VQA result
    meta_by_img_dict = item["metadata"]

    # img_idx, path, score 초기화
    chosen = (None, None, -100)
    rejected = (None, None, 100)

    # find chosen quickly
    q_len = len(item["questions"])
    best_score_local = -100 # 초기화


    for img_idx, img_dict in meta_by_img_dict.items():
        # C조건 1.
        if img_dict['score_local'] > best_score_local and img_dict['score_local'] >= THRESHOLD:
            ans_list = [img_dict[f'q{qid}']['answer'] for qid in range(q_len)] 
            # C조건 2.
            if all((ans or "").strip().lower() == "no" for ans in ans_list):
                continue # cannot be CHOSEN

            best_score_local = img_dict['score_local']
            chosen = (img_idx, img_dict['path'], img_dict['score_local'])

    # 필터링 1.선택된 Chosen 이 없는 경우,
    if chosen[0] is None:
        print("Skip this item (DISCARDED: No valid Chosen)")
        continue # skip this sample


    # Find rejected
    for img_idx, img_dict in meta_by_img_dict.items():
        if img_idx == chosen[0]:
            continue # skip chosen

        # R조건 1.
        ans_list = [img_dict[f'q{qid}']['answer'] for qid in range(q_len)]
        if all(ans.lower().strip() == "yes" for ans in ans_list):
            continue # cannot be REJECTED

        # select the lowest score_local
        if img_dict['score_local'] < rejected[2]:
            rejected = (img_idx, img_dict['path'], img_dict['score_local'])


    # 필터링 2. 선택된 Rejected 가 없는 경우
    if rejected[0] is None:
        print("Skip this item (DISCARDED: No valid Rejected)")
        continue # skip this sample


    # 최종적으로 샘플 완성 (path only)
    train_sample['chosen'] = chosen[1]
    train_sample['rejected'] = rejected[1]

    train_dataset.append(train_sample)