# (완료)
# 기존 16k 데이터셋에서 데이터 샘플 수가 크게 줄었을 때
# 1k - 2k - 4k - 8k - 16k(기준); stage 1. prompt generation 기준

import os
import json
import random
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from tqdm import tqdm
from ospo.utils.common import read_json, save_json


def resize_data(init_dataset, filt_dataset, n=1000):
    resized = random.sample(init_dataset, n)
    resized_item_id_list = [d['item_id'] for d in resized]

    resized_data = []
    # filt_dataset 얼라인을 통해 필터링 과정 생략
    for d in tqdm(filt_dataset):
        if d['item_id'] not in resized_item_id_list:
            continue
        resized_data.append(d)

    print(f"Total resized length (N={n}): {len(resized_data)}")
    return resized_data


if __name__ == '__main__':

    init_path = "/nas2/data/Janus_dataset/next_v2/init_dataset_16001.json"
    filt_path = "/nas2/data/Janus_dataset/next_v2/train/train_mode_argmax_chosen_filtered_10430.json"
    init, filt = read_json(init_path), read_json(filt_path)
    print("init: ", len(init))
    print("filt: ", len(filt))

    # for n in [1000, 2000, 4000, 8000]:
    for n in [12000]:
        data = resize_data(init, filt, n)

        save_dir = "/nas2/data/Janus_dataset/next_v2/ablation"
        save_path = f"init_prompt_size_{str(n)[:-3]}"
        save_json(save_root=save_dir, save_name=save_path, save_file=data)

    print("Done.")


# Total resized length (N=1000): 651
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10430/10430 [00:00<00:00, 35917.27it/s]
# Total resized length (N=2000): 1288
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10430/10430 [00:00<00:00, 17254.09it/s]
# Total resized length (N=4000): 2653
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10430/10430 [00:01<00:00, 7951.07it/s]
# Total resized length (N=8000): 5245
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10430/10430 [00:01<00:00, 7951.07it/s]
# Total resized length (N=12000): 7828
