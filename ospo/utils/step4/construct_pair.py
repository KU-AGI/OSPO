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
    local_score = chosen_cand['local_score'] # range: [-1, 1]
    ans_metadata = chosen_cand['answer_metadata']
    cnt_yes, cnt_no = 0, 0

    for ans in ans_metadata:
        answer = ans['answer'].lower().strip()
        if answer == "yes":
            cnt_yes += 1
        else:
            cnt_no += 1 # tie 포함
    
    # (Condition 1) Chosen - All No 
    if cnt_yes == 0:
        return True # Filtered

    # (Condition 2) Avg. score < threshold
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
            cnt_no += 1 # including 'tie'
    
    # (Condition 1) Rejected - All Yes
    if cnt_no == 0:
        return True # Filtered
    else: 
        return False


# Filtering one-by-one (for each instance (item_id)) 
def _filter_sample(vqa_sample, nothing=False):

    keep_chosen = []
    keep_rejected = []

    for i, meta_list in enumerate([list(vqa_sample['base_metadata'].values()),
                                   list(vqa_sample['negative_metadata'].values())]):
        # Chosen filtering
        if i == 0:
            for vqa_rs in meta_list:
                if nothing or not _filter_chosen(vqa_rs):
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


# Chosen-Rejected pair matching (after filtering) 
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
    assert mode in ["random", "argmax_chosen"], f"Unsupported mode: {mode}"

    # only for analysis
    if mode == "random":

        # 1. No Filtering 
        filtered_chosen, filtered_rejected = _filter_sample(sample, nothing=True)

        # 2. Select randomly
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

    # ospo
    elif mode == "argmax_chosen":
        # 1. Filtering
        filtered_chosen, filtered_rejected = _filter_sample(sample)
        if len(filtered_chosen) == 0 or len(filtered_rejected) == 0:
            return None 

        # 2. Select argmax
        matched_pairs = _match_pair(filtered_chosen, filtered_rejected)
        if not matched_pairs:
            return None
        chosen_path, rejected_path, chosen_score, rejected_score = max(matched_pairs, key=lambda x: x[2]) # by chosen score 

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
    

    # # WITHOUT DENSE
    # init_path = "/nas2/data/Janus_dataset/next_v2/ablation/wo_dense/init_dataset_16001.json"
    # vqa_path = "/nas2/data/Janus_dataset/next_v2/ablation/wo_dense/vqa_result_16001.json"
    # save_path = "/nas2/data/Janus_dataset/next_v2/ablation/wo_dense/train"


    # # NUMERACY_5K_6K
    # init_path = "/nas2/data/Janus_dataset/next_v2/ablation/wo_dense/init_dataset_16001.json"
    # vqa_path = "/nas2/data/Janus_dataset/next_v2/ablation/wo_dense/vqa_result_16001.json"
    # save_path = "/nas2/data/Janus_dataset/next_v2/ablation/wo_dense/train"


    # NUMERACY_5K_6K (시드 통합)
    # init_path = "/home/yjoh/project/OSPO/ablation/pair_size/init_pair_size_6_numeracy_add_only_seed_merged.json"
    # vqa_path = "/nas2/data/Janus_dataset/next_v2/numeracy_add/vqa_result_numeracy_add_only_seed_merged_11806.json"
    # save_path = "/nas2/data/Janus_dataset/next_v2/numeracy_add/seed_merge"


    # Filtering Threshold (iter2)
    # init_path = "/nas2/data/Janus_dataset/next_v2/appendix/iter2/data/prompt/init_dataset_20005.json"
    # vqa_path = "/nas2/data/Janus_dataset/next_v2/appendix/iter2/data/prompt/step3/vqa_result_merged_20005.json"
    # save_path = "/nas2/data/Janus_dataset/next_v2/appendix/iter2/data/train"


    # 260115
    # init_path = "/nas2/data/Janus_dataset/next_v2/merged_init_data_27807.json"
    # vqa_path = "/nas2/data/Janus_dataset/next_v2/merged_vqa_result_27807.json"
    # save_path = "/nas2/data/Janus_dataset/next_v2/train_2026"

    # 260116
    init_path = "/home/yjoh/project/OSPO/ablation/pair_size/init_pair_size_1_numeracy_add_only_seed_merged.json"
    vqa_path = "/home/yjoh/project/OSPO/ablation/pair_size/vqa_result_numeracy_add_only_seed_merged_pair_1.json"
    save_path = "/nas2/data/Janus_dataset/next_v2/ablation/pair_size/numeracy_add_version"



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
        # save_name=f"train_mode_{MODE}_{len(train_data)}_pair_size_6"
        save_name=f"train_mode_{MODE}_{len(train_data)}"
    )