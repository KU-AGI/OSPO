# candidate pair size 1 - 3 (기준) - 5
# 또는
# candidate pair size 1 - 3 (기준) - 6 - 9

import os
import json
import random
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from tqdm import tqdm
from ospo.utils.common import read_json, save_json


def resize_vqa_result(vqa_data, init_data, n):
    
    resized_vqa_dataset = []
    init_data_dict = {d['item_id']: d for d in init_data}

    def _check(metadata: dict, ref: list):
        filtered_dict = dict()
        for k, v in metadata.items():
            if v['path'] in ref:
                filtered_dict[k] = v
        return filtered_dict

    for v_sample in tqdm(vqa_data):
        item_id = v_sample['item_id']
        base_metadata = v_sample['base_metadata']
        negative_metadata = v_sample['negative_metadata']

        i_sample = init_data_dict.get(item_id, None)
        if i_sample is None:
            continue
        base_cands = i_sample['chosen_candidate']
        negative_cands = i_sample['rejected_candidate']

        updated_base = _check(base_metadata, base_cands) 
        updated_negative = _check(negative_metadata, negative_cands) 

        v_sample['base_metadata'] = updated_base
        v_sample['negative_metadata'] = updated_negative

        resized_vqa_dataset.append(v_sample)

    return resized_vqa_dataset




if __name__ == '__main__':    
    
    # for n in [1]:
    for n in [5, 6, 9]:

        # vqa_path = "/nas2/data/Janus_dataset/next_v2/vqa_result_16001.json"
        vqa_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/prompt/vqa_result_25002.json"

        init_path = f"/nas2/data/Janus_dataset/next_v2/ablation/pair_size/pair_size_{n}.json"
        vqa, init = read_json(vqa_path), read_json(init_path)

        data = resize_vqa_result(vqa, init, n)
        save_json(
            save_root='/nas2/data/Janus_dataset/next_v2/ablation/pair_size',
            save_name=f'vqa_result_pair_size_{n}',
            save_file=data
        )

    print('Done.')