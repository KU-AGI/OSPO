# /nas2/data/Janus_dataset/main_exp/aaai/unitok
# 추가 데이터 샘플 생성 없이, out-of-range NUMBER WORDS 가 포함된 경우 해당 샘플 배제

# conda activate unitok_dpo

import os, json
import re
from tqdm import tqdm
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from cvpr.common import read_json, save_json

_NUMBER_WORDS = ["one", "two", "three", "four"]

_NO_NUMBER_WORDS = [
    "five","six","seven","eight","nine","ten", "eleven","twelve","thirteen","fourteen",
    "fifteen","sixteen","seventeen","eighteen","nineteen","twenty","thirty","forty",
    "fifty","sixty","seventy","eighty","ninety","hundred","thousand","million","billion"
]

_TOTAL_NUMBER_WORDS = _NUMBER_WORDS + _NO_NUMBER_WORDS



# 1. 기본 데이터 로드

base_path = "/nas2/data/Janus_dataset/main_exp/aaai/unitok/prompt/base_prompt_16002_v2.json"
base_data = read_json(base_path)



# 2. 부적절한 데이터 샘플 필터링

def extract_number_words(text):
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if t in _TOTAL_NUMBER_WORDS]

def sort_by_key(dataset):
    dataset = sorted(dataset, key=lambda x: x["item_id"])
    return dataset

error1 = []
error2 = []
filtered = [] # 필터링 후의 데이터 샘플

for d in tqdm(base_data, desc="Writing base prompt ..."):
    t2i_category = d["t2i_category"]
    sub_category = d["sub_category"]
    
    if t2i_category != "layout" or sub_category == "layout1":
        filtered.append(d)
        continue

    prompt = d["prompt"]
    number_words = extract_number_words(prompt)
    if len(number_words) == 0:
        error1.append(d)
        continue
    
    flag_bad = any(n not in _NUMBER_WORDS for n in number_words)
    if flag_bad:
        error2.append(d)
    else:
        filtered.append(d)

print(f"Total layout samples: {len([d for d in base_data if d['t2i_category']=='layout'])}")
print(f"Total errors in layout samples: {len(error1) + len(error2)}")
print(f" - No number words: {len(error1)}")
print(f" - Invalid number words: {len(error2)}")
print("Samples with no number words:")
# for e in error1[:3]:
#     print(f" - {e['prompt']}") 
# print("Samples with invalid number words:")
# for e in error2[:3]:
#     print(f" - {e['prompt']}")

filtered = sort_by_key(filtered)
filtered_item_id_list = [d['item_id'] for d in filtered]

# 3. base_prompt dataset filtered ver. 저장
save_dir = "/nas2/data/Janus_dataset/next_v2/unitok/dataset/prompt"
os.makedirs(save_dir, exist_ok=True)
save_json(save_root=save_dir, save_name=f"base_prompt_{len(filtered)}", save_file=filtered)




###



# 3. 데이터 샘플마다 추가 정보 삽입 (negative_prompt, dense_prompt, img_path aggregate)
# > init_dataset

img_path = "/nas2/data/Janus_dataset/main_exp/aaai/unitok/images_v5"
dense_path = "/nas2/data/Janus_dataset/main_exp/aaai/unitok/prompt/long_prompt_v4.json" # 이미지 폴더와 얼라인 확인
dense_data = read_json(dense_path)

# vqa_path = "/nas2/data/Janus_dataset/main_exp/aaai/unitok/prompt/vqa_result_v4.json"
# vqa_result_data = read_json(vqa_path)

init_data = []

for d in tqdm(dense_data, desc="Writing init dataset..."):
    item_id = d["item_id"]
    if item_id not in filtered_item_id_list:
        continue
    t2i_category = d["t2i_category"]

    chosen_base_dir = os.path.join(img_path, "base", t2i_category, item_id)
    rejected_base_dir = os.path.join(img_path, "negative", t2i_category, item_id)

    chosen_candidates = [os.path.join(chosen_base_dir, f) for f in os.listdir(chosen_base_dir) if f.endswith(".png")]
    rejected_candidates = [os.path.join(rejected_base_dir, f) for f in os.listdir(rejected_base_dir) if f.endswith(".png")]

    # 키값 추가
    d["chosen_candidate"] = chosen_candidates
    d["rejected_candidate"] = rejected_candidates

    # 최종 추가
    init_data.append(d)

init_data = sort_by_key(init_data)
save_json(save_root=save_dir, save_name=f"init_dataset_{len(init_data)}", save_file=init_data) 