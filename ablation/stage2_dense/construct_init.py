# construct w/o dense veresion init dataset
# construct w/o dense version vqa result

import os, json
import pyrootutils
import random
from tqdm import tqdm
from copy import deepcopy
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.common import read_json, save_json


def extract_dense_img(c_can: list, r_can: list):
    chosen_candidate_wo_dense = []
    rejected_candidate_wo_dense = []

    for c in c_can:
        if not os.path.basename(c) in ["00.png", "01.png", "02.png"]: 
            continue
        
        # fetch rejected
        find = c.replace("base", "negative")
        if find in r_can:
            chosen_candidate_wo_dense.append(c)
            rejected_candidate_wo_dense.append(find)

        if len(chosen_candidate_wo_dense) == 3:
            break

    return chosen_candidate_wo_dense, rejected_candidate_wo_dense


def main(init_data, vbase_data, vnegative_data, vsub_data):
    
    wo_dense_init_data = []
    wo_dense_vqa_data = []

    vbase_dict = {d['item_id']: d for d in vbase_data}
    vneg_dict = {d['item_id']: d for d in vnegative_data}
    vsub_dict = {d['item_id']: d for d in vsub_data}

    # init
    for d0 in tqdm(init_data):
        d = deepcopy(d0)
        item_id = d['item_id']

        chosen_candidate_wo_dense, rejected_candidate_wo_dense = extract_dense_img(
            d0["ablation"]["wo_densification"]["chosen_candidate"],
            d0["ablation"]["wo_densification"]["rejected_candidate"],   
        )

        del d["ablation"]
        d["chosen_candidate"] = chosen_candidate_wo_dense
        d["rejected_candidate"] = rejected_candidate_wo_dense
        wo_dense_init_data.append(d)

        # 저장 대상
        base_metadata, negative_metadata = dict(), dict()

        # vqa (주의)
        base_v = vbase_dict.get(item_id, None)
        neg_v = vneg_dict.get(item_id, None)
        
        # 서브 VQA
        if base_v is None or neg_v is None:
            # find in vqa_sub_data
            vmetadata = vsub_dict.get(item_id, None)
            if vmetadata is None:
                raise ValueError(f"Unexpected: {item_id}")
            else:
                base_v_meta, neg_v_meta = vmetadata["base_metadata"], vmetadata["negative_metadata"]
                question = vmetadata["question"]

                for k, v in base_v_meta.items():
                    if v['path'] in chosen_candidate_wo_dense:
                        base_metadata[k] = v

                for k, v in neg_v_meta.items():
                    if v['path'] in rejected_candidate_wo_dense:
                        negative_metadata[k] = v
                
                if len(base_metadata) == 0 or len(negative_metadata) == 0:
                    print(f"[{item_id}] VQA metadata is NULL.")
                    continue

        # 일반 VQA
        else:
            question = base_v["question"]

            for k, v in base_v["metadata"].items():
                if v['path'] in chosen_candidate_wo_dense:
                    base_metadata[k] = v

            for k, v in neg_v["metadata"].items():
                if v['path'] in rejected_candidate_wo_dense:
                    negative_metadata[k] = v
            
            if len(base_metadata) == 0 or len(negative_metadata) == 0:
                print(f"[{item_id}] VQA metadata is NULL.")
                continue

        # 저장
        wo_dense_vqa_data.append({
            "item_id": item_id,
            "t2i_category": d["t2i_category"],
            "sub_category": d["sub_category"],
            "prompt": d["prompt"],
            "question": question,
            "base_metadata": base_metadata,
            "negative_metadata": negative_metadata
        })


    return wo_dense_init_data, wo_dense_vqa_data







if __name__ == "__main__":
    
    init_path = "/nas2/data/Janus_dataset/next_v2/init_dataset_16001.json"
    init_data = read_json(init_path)

    # short prompt image - vqa result
    vqa_base_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/prompt/negative_short/vqa/v1/base_ten_vqa_results_L40S.json"
    vqa_base_complex_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/prompt/negative_short/vqa/v1/base_vqa_results_A100_complex.json"
    vqa_negative_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/prompt/negative_short/vqa/v1/negative_vqa_results_A100.json"
    vqa_sub_path = "/nas2/data/Janus_dataset/next_v2/ablation/wo_dense/vqa_result_added_978.json"

    vqa_base_part, vqa_base_complex, vqa_negative = read_json(vqa_base_path), read_json(vqa_base_complex_path), read_json(vqa_negative_path)
    vqa_base = vqa_base_part +  vqa_base_complex
    vqa_sub = read_json(vqa_sub_path)

    wo_dense_init, wo_dense_vqa = main(init_data, vqa_base, vqa_negative, vqa_sub)
    
    
    # save
    save_dir = "/nas2/data/Janus_dataset/next_v2/ablation/wo_dense"

    save_json(save_dir, save_name=f"init_dataset_{len(wo_dense_init)}", save_file=wo_dense_init)
    save_json(save_dir, save_name=f"vqa_result_{len(wo_dense_vqa)}", save_file=wo_dense_vqa)

    print("Done.")


    
    