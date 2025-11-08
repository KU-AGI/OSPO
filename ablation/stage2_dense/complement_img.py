# layout_add 샘플들에 대한 VQA 결과가, 추가 VQA 진행
# required: vqa prompt

import os, json
import pyrootutils
import random
from tqdm import tqdm
from copy import deepcopy
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.common import read_json, save_json


def extract_add_vqa_prompt(init_data, vqa_data):
    
    vqa_dict = {d['item_id']: d['question'] for d in vqa_data}
    vqa_prompt_data = []

    for d in tqdm(init_data):
        item_id = d['item_id']
        t2i_category = d['t2i_category']
        
        if t2i_category != "layout":
            continue
        elif int(item_id) < 1004000:
            continue
        else:
            # 해당
            vqa_prompt = vqa_dict.get(item_id, None)
            if vqa_prompt is not None:
                vqa_prompt_data.append({
                    "item_id": item_id,
                    "t2i_category": t2i_category,
                    "sub_category": d['sub_category'],
                    "prompt": d['prompt'],
                    "question": vqa_prompt
                })

    print(f"Total data length: {len(vqa_prompt_data)}")
    return vqa_prompt_data

        

if __name__ == "__main__":
    
    init_path = "/nas2/data/Janus_dataset/next_v2/init_dataset_16001.json"
    init_data = read_json(init_path)

    vqa_result_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/prompt/vqa_result_25002.json"
    vqa_data = read_json(vqa_result_path)

    # 조건: init_data 에 속한 샘플 중, added layout 에 해당하는 샘플의 vqa prompt 만 모으기
    added_vqa_prompt = extract_add_vqa_prompt(init_data, vqa_data)
    save_json(
        save_root='/nas2/data/Janus_dataset/next_v2/ablation/wo_dense',
        save_name=f'vqa_prompt_added_{len(added_vqa_prompt)}',
        save_file=added_vqa_prompt
    )