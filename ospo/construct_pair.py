# /home/yjoh/project/ospo/cvpr/dataset/construct.py

# 지정된 모드에 따라 학습 데이터셋 구축 코드
# hyperparameter
# - mode: "random" | "argmax_chosen"
# - threshold: 0.6 (for _filter_chosen)


import os, json
import random
from tqdm import tqdm
from math import inf

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.common import read_json, save_json


def _filter_chosen(chosen_cand, threshold=0.6):
    local_score = chosen_cand['local_score'] # local score 의 범위는 [-1, 1]
    ans_metadata = chosen_cand['answer_metadata']
    cnt_yes, cnt_no = 0, 0

    for ans in ans_metadata:
        answer = ans['answer'].lower().strip()
        if answer == "yes":
            cnt_yes += 1
        else:
            cnt_no += 1 # tie 포함
    
    # (조건 1) Chosen 이 All No 인 경우
    if cnt_yes == 0:
        return True # Filtered

    # (조건 2) 모든 질문을 아우르는 평균 스코어 자체가 threshold 보다 낮은 경우
    elif local_score < threshold:
        return True # Filtered
    elif local_score >= threshold:
        return False
    else:
        raise ValueError(f"Unexpected scenario: {local_score}")


def _filter_rejected(rejected_cand):
    ans_metadata = rejected_cand['answer_metadata']
    cnt_yes, cnt_no = 0, 0
    for ans in ans_metadata:
        answer = ans['answer'].lower().strip()
        if answer == "yes":
            cnt_yes += 1
        else:
            cnt_no += 1 # tie 포함
    
    # (조건 1) Rejected 이 All Yes 인 경우
    if cnt_no == 0:
        return True # Filtered
    else: 
        return False


# 하나의 샘플 (item_id) 단위에서 필터링
def _filter_sample(vqa_sample, nothing=False):

    keep_chosen = []
    keep_rejected = []

    for i, meta_list in enumerate([list(vqa_sample['base_metadata'].values()),
                                   list(vqa_sample['negative_metadata'].values())]):
        # Chosen filtering
        if i == 0:
            for vqa_rs in meta_list:
                if nothing or not _filter_chosen(vqa_rs):
                    # keep_chosen.append(vqa_rs['path'])
                    keep_chosen.append((vqa_rs['path'], vqa_rs['local_score']))
                else:
                    pass # Filtered

        # Rejected filtering
        else:
            for vqa_rs in meta_list:
                if nothing or not _filter_rejected(vqa_rs):
                    keep_rejected.append((vqa_rs['path'], vqa_rs['local_score']))
                else:
                    pass  # Filtered
                

    return keep_chosen, keep_rejected


# 필터링을 거친 샘플들 가운데, C-R 페어 매칭 작업
def _match_pair(chosen_path_list, rejected_path_list): 
    matched_pair = []
    
    # do not address score
    for c, c_score in chosen_path_list: 
        base_c = os.path.basename(c)
        for r, r_score in rejected_path_list:
            base_r =  os.path.basename(r)
            if base_c == base_r:
                matched_pair.append((c,r, c_score, r_score))
                break

    return matched_pair


# input: vqa sample
def select_pair(sample, mode="random"):
    # 지정된 모드
    assert mode in ["random", "argmax_chosen"], f"Unsupported mode: {mode}"

    if mode == "random":

        # 1. 필터링 X (nothing=True); 당연히 길이 > 0 
        filtered_chosen, filtered_rejected = _filter_sample(sample, nothing=True)

        # 2. random 선택
        matched_pairs = _match_pair(filtered_chosen, filtered_rejected)
        if not matched_pairs:
            return None # 짝이 없는 경우

        max_int = len(matched_pairs)
        random_idx = random.randint(0, max_int-1)

        chosen_path, rejected_path, chosen_score, rejected_score = matched_pairs[random_idx]
        selected_dict = {
            "chosen": chosen_path,
            "rejected": rejected_path,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
        }


    elif mode == "argmax_chosen":
        # 1. 필터링
        filtered_chosen, filtered_rejected = _filter_sample(sample)
        if len(filtered_chosen) == 0 or len(filtered_rejected) == 0:
            return None 

        # 2. argmax 선택
        matched_pairs = _match_pair(filtered_chosen, filtered_rejected)
        if not matched_pairs:
            return None

        # argmax 선택
        chosen_path, rejected_path, chosen_score, rejected_score = max(matched_pairs, key=lambda x: x[2]) # chosen score 기준

        selected_dict = {
            "chosen": chosen_path,
            "rejected": rejected_path,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
        }

    
    return selected_dict



if __name__ == "__main__":

    # MODE = "random"
    MODE = "argmax_chosen"

    # ITER 1
    # init_path = "/nas2/data/Janus_dataset/next/init_dataset_16002.json"  # v1
    # init_path = "/nas2/data/Janus_dataset/next_v2/init_dataset_16001.json" # v2
    # vqa_path = "/nas2/data/Janus_dataset/next_v2/vqa_result_16001.json"
    # save_path = "/nas2/data/Janus_dataset/next_v2/train"


    # ITER2
    # init_path = "/nas2/data/Janus_dataset/next_v2/iter2/prompt/init_dataset_16006.json"
    # vqa_path = "/nas2/data/Janus_dataset/next_v2/iter2/prompt/vqa_result_merged_16006.json"
    # save_path = "/nas2/data/Janus_dataset/next_v2/iter2/train"
    

    # WITHOUT DENSE
    init_path = "/nas2/data/Janus_dataset/next_v2/ablation/wo_dense/init_dataset_16001.json"
    vqa_path = "/nas2/data/Janus_dataset/next_v2/ablation/wo_dense/vqa_result_16001.json"
    save_path = "/nas2/data/Janus_dataset/next_v2/ablation/wo_dense/train"

    init_data = read_json(init_path)
    init_dict = {d["item_id"]: d for d in init_data}
    vqa_data = read_json(vqa_path)

    train_data = []

    for sample in tqdm(vqa_data, "Processing samples ..."):
        selected_metadata = select_pair(sample, mode=MODE)
        if selected_metadata is not None:
            dsample = init_dict[sample["item_id"]]
            for k, v in selected_metadata.items():
                dsample[k] = v

            # rejected prompt fetch
            base_idx = int(os.path.basename(dsample["rejected"]).split(".png")[0])
            dsample["rejected_prompt"] = dsample["negative_prompt"][base_idx]

            # complete !
            train_data.append(dsample)

    print("Total train samples:", len(train_data))


    # 저장
    os.makedirs(save_path, exist_ok=True)
    save_json(
        save_root=save_path,
        save_file=train_data,
        save_name=f"train_mode_{MODE}_{len(train_data)}"
    )