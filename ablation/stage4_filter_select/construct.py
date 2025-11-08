# construct dataset
# 데이터 셀렉 - 오브젝트 추출 - 오브젝트 마스크 추출의 단계
# (참고) /home/yjoh/project/ospo/cvpr/dataset/construct.py

import os, json
import re
import pyrootutils
import random
from tqdm import tqdm
from copy import deepcopy
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.common import read_json, save_json


# 필터링을 거친 샘플들 가운데, C-R 페어 매칭 작업
def _match_pair(chosen_path_list, rejected_path_list): 
    matched_pair = []
    
    # do not address score
    for c, c_score in chosen_path_list: 
        base_c = os.path.basename(c)
        for r, r_score in rejected_path_list:
            base_r =  os.path.basename(r)
            if base_c == base_r:
                matched_pair.append((c, r, c_score, r_score))
                break

    return matched_pair


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


def construct(init_data: list = None, vqa_data: list = None, filter: bool = False, select: bool = False):
    
    errors = 0    
    ablation_dataset = []

    if filter and select:
        raise ValueError("Not supported here.")

    elif filter: # yes filter, no select
        
        init_dict = {d["item_id"]: d for d in init_data}

        for data in tqdm(vqa_data): # VQA DATA
            filtered_chosen, filtered_rejected = _filter_sample(data)
            matched_pairs = _match_pair(filtered_chosen, filtered_rejected)
            if not matched_pairs:
                errors += 1
                continue

            # randomly select
            max_int = len(matched_pairs)
            random_idx = random.randint(0, max_int-1)
            chosen_path, rejected_path, chosen_score, rejected_score = matched_pairs[random_idx]

            dsample = init_dict[data["item_id"]]

            # del dsample["ablation"]
            dsample["chosen"] = chosen_path
            dsample["rejected"] = rejected_path
            dsample["chosen_score"] = chosen_score
            dsample["rejected_score"] = rejected_score

            ablation_dataset.append(dsample)

    
    elif select: # no filter, yes select

        init_dict = {d["item_id"]: d for d in init_data}

        # argmax(chosen) 기준
        for data in tqdm(vqa_data): # VQA DATA
            clist = [(v['path'], v['local_score']) for k, v in data['base_metadata'].items()]
            rlist = [(v['path'], v['local_score']) for k, v in data['negative_metadata'].items()]

            matched_pairs = _match_pair(clist, rlist)
            if not matched_pairs:
                errors += 1
                continue

            # argmax 선택
            chosen_path, rejected_path, chosen_score, rejected_score = max(matched_pairs, key=lambda x: x[2]) # chosen score 기준

            dsample = init_dict[data["item_id"]]

            # del dsample["ablation"]
            dsample["chosen"] = chosen_path
            dsample["rejected"] = rejected_path
            dsample["chosen_score"] = chosen_score
            dsample["rejected_score"] = rejected_score

            ablation_dataset.append(dsample)


    else: # no filter, no select

        vqa_dict = {d["item_id"]: d for d in vqa_data}
    
        def get_local_score(item_id, c_path, r_path):       

            chosen_score, rejected_score = None, None

            my_dict = vqa_dict[item_id]
            for k, v in my_dict["base_metadata"].items():
                if v['path'] == c_path:
                    chosen_score = v['local_score']

            for k, v in my_dict["negative_metadata"].items():
                if v['path'] == r_path:
                    rejected_score = v['local_score']

            return chosen_score, rejected_score
            

        for data in tqdm(init_data):    

            item_id = data['item_id']    
            # --- Step 1. Get common basenames (e.g., "00.png") ---
            chosen_basenames = {os.path.basename(p): p for p in data["chosen_candidate"]}
            rejected_basenames = {os.path.basename(p): p for p in data["rejected_candidate"]}

            common_basenames = list(set(chosen_basenames.keys()) & set(rejected_basenames.keys()))

            # --- Step 2. Randomly select one basename from common ones ---
            if not common_basenames:
                # print("No matching chosen/rejected basenames found!")
                errors += 1
                continue
            
            selected_basename = random.choice(common_basenames)

            # --- Step 3. Get paired paths ---
            chosen_path = chosen_basenames[selected_basename]
            rejected_path = rejected_basenames[selected_basename]
        
            # add chosen score, rejected score
            chosen_score, rejected_score = get_local_score(item_id, chosen_path, rejected_path)

            # del data["ablation"]
            data["chosen"] = chosen_path
            data["rejected"] = rejected_path
            data["chosen_score"] = chosen_score
            data["rejected_score"] = rejected_score

            ablation_dataset.append(data)


    print("Total Errors: ", errors)
    return ablation_dataset


if __name__ == "__main__":
    # 공통
    save_dir = "/nas2/data/Janus_dataset/next_v2/ablation/filter_select"
    init_path = "/nas2/data/Janus_dataset/next_v2/init_dataset_16001.json"
    vqa_path = "/nas2/data/Janus_dataset/next_v2/vqa_result_16001.json"
    init_data, vqa_data = read_json(init_path), read_json(vqa_path)
    
    # 개별
    for do_filter, do_select in [
                                    (True, False), # Total Errors:  5566
                                    (False, True), # Total Errors:  35
                                    # (False, False)
                                ]:
        ablation_init = construct(init_data, vqa_data, do_filter, do_select)
        save_name = f"init_dataset_{len(ablation_init)}"

        key = f"filter_{do_filter}_select_{do_select}"
        key_save_dir = os.path.join(save_dir, key)
        os.makedirs(key_save_dir, exist_ok=True)
        
        # 저장
        save_json(
            save_root=key_save_dir,
            save_name=save_name,
            save_file=ablation_init
        )

    