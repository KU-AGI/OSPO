# /home/yjoh/project/ospo/cvpr/dataset/init_dataset_v2.py
# 1. init_dataset 에서 layout 카테고리 수정

import os, json
import re
from tqdm import tqdm
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.common import read_json, save_json

_NUMBER_WORDS = ["one", "two", "three", "four"]

_NO_NUMBER_WORDS = [
    "five","six","seven","eight","nine","ten", "eleven","twelve","thirteen","fourteen",
    "fifteen","sixteen","seventeen","eighteen","nineteen","twenty","thirty","forty",
    "fifty","sixty","seventy","eighty","ninety","hundred","thousand","million","billion"
]

_TOTAL_NUMBER_WORDS = _NUMBER_WORDS + _NO_NUMBER_WORDS

def extract_number_words(text):
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if t in _TOTAL_NUMBER_WORDS]





""" PHASE 1 """
init_path = "/nas2/data/Janus_dataset/next/init_dataset_16002.json"
init_data = read_json(init_path)


error1 = []
error2 = []

filtered = []

for d in init_data:
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

print(f"Total layout samples: {len([d for d in init_data if d['t2i_category']=='layout'])}")
print(f"Total errors in layout samples: {len(error1) + len(error2)}")
print(f" - No number words: {len(error1)}")
print(f" - Invalid number words: {len(error2)}")
print("Samples with no number words:")
for e in error1[:3]:
    print(f" - {e['prompt']}") 
print("Samples with invalid number words:")
for e in error2[:3]:
    print(f" - {e['prompt']}")



""" PHASE 2 """
# Layout2 로 보완
# cnt_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/prompt/layout_more_seed_10/base_prompt_1000.json"
cnt_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/prompt/layout_more_seed_10/seed_012/long_prompt.json"
cnt_data = read_json(cnt_path)

extra_data = []
filtered_length = len(filtered)

# 필터링
for d in cnt_data:
    prompt = d["prompt"]
    number_list = extract_number_words(prompt)
    if any(n not in _NUMBER_WORDS for n in number_list):
        continue
    else:
        extra_data.append(d)
        if filtered_length + len(extra_data) > 16000:
            break



""" PHASE 3 """
# 추가 데이터에 대한 추가 정보 삽입 (seed 0,1,2 기준)
# - long prompt
# - img path
# - ablation img path
# - vqa result

# long_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/prompt/layout_more_seed_10/seed_012/long_prompt.json"
img_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/images_layout_more_seed_10/seed_012"
ablation_img_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/images_short_prompt_2"

for d in extra_data:
    item_id = d["item_id"]

    chosen_base_dir = os.path.join(img_path, "base", "layout", item_id)
    rejected_base_dir = os.path.join(img_path, "negative", "layout", item_id)

    chosen_candidates = [os.path.join(chosen_base_dir, f) for f in os.listdir(chosen_base_dir) if f.endswith(".png")]
    rejected_candidates = [os.path.join(rejected_base_dir, f) for f in os.listdir(rejected_base_dir) if f.endswith(".png")]

    a_dict = {"wo_densification": {
        "chosen_candidate": [os.path.join(os.path.join(ablation_img_path, "base", "layout", item_id), f) for f in os.listdir(os.path.join(ablation_img_path, "base", "layout", item_id)) if f.endswith(".png")],
        "rejected_candidate": [os.path.join(os.path.join(ablation_img_path, "negative", "layout", item_id), f) for f in os.listdir(os.path.join(ablation_img_path, "negative", "layout", item_id)) if f.endswith(".png")]
    }}
    # 키값 추가
    d["chosen_candidate"] = chosen_candidates
    d["rejected_candidate"] = rejected_candidates
    d["ablation"] = a_dict

    # 최종 추가
    filtered.append(d)

# 키값에 따라 정렬
filtered = sorted(filtered, key=lambda x: x["item_id"])
filtered_item_id_list = [d["item_id"] for d in filtered]
print("Final filtered dataset length:", len(filtered))


""" PHASE 4 """
# vqa result 데이터도 위와 동일한 과정 수행
init_vqa_result_path = "/nas2/data/Janus_dataset/next/vqa_result_nas.json"
init_vqa_result = read_json(init_vqa_result_path)
init_vqa_result = [d for d in init_vqa_result if d["item_id"] in filtered_item_id_list]
print("Initial filtered vqa_result length:", len(init_vqa_result))

extra_vqa_result_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/prompt/layout_more_seed_10/seed_012/vqa_result_seed_012.json"
extra_vqa_result = read_json(extra_vqa_result_path)
extra_vqa_dict = {d["item_id"]: d for d in extra_vqa_result}

for d in extra_vqa_result:
    item_id = d["item_id"]
    if item_id in filtered_item_id_list:
        init_vqa_result.append(d)

# 키값에 따라 정렬
init_vqa_result = sorted(init_vqa_result, key=lambda x: x["item_id"])
print("Final filtered vqa_result length:", len(init_vqa_result))




""" PHASE 5 """
# 만약 추가 데이터가 1000개 미달일 경우, 또 추가
prompt_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/prompt/layout_more_seed_42/seed_012/long_prompt_v2.json"
prompt_data = read_json(prompt_path)

img_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/images_layout_more_seed_42/seed_012"
ablation_img_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/images_short_prompt_2"
vqa_result_path = "/nas2/data/Janus_dataset/main_exp/aaai/janus/prompt/layout_more_seed_42/seed_012/vqa_result_seed_012.json"
vqa_result_data = read_json(vqa_result_path)
vqa_result_dict = {d["item_id"]: d for d in vqa_result_data}


# extra_data_2 = []

for d in prompt_data:
    sub_category = d["sub_category"]
    if sub_category != "layout2":
        continue
    
    number_list = extract_number_words(d["prompt"])
    if any(n not in _NUMBER_WORDS for n in number_list):
        continue
    else:
        chosen_base_dir = os.path.join(img_path, "base", "layout", d["item_id"])
        rejected_base_dir = os.path.join(img_path, "negative", "layout", d["item_id"])

        chosen_candidates = [os.path.join(chosen_base_dir, f) for f in os.listdir(chosen_base_dir) if f.endswith(".png")]
        rejected_candidates = [os.path.join(rejected_base_dir, f) for f in os.listdir(rejected_base_dir) if f.endswith(".png")]
        a_dict = {"wo_densification": {
            "chosen_candidate": [os.path.join(os.path.join(ablation_img_path, "base", "layout", d["item_id"]), f) for f in os.listdir(os.path.join(ablation_img_path, "base", "layout", d["item_id"])) if f.endswith(".png")],
            "rejected_candidate": [os.path.join(os.path.join(ablation_img_path, "negative", "layout", d["item_id"]), f) for f in os.listdir(os.path.join(ablation_img_path, "negative", "layout", d["item_id"])) if f.endswith(".png")] 
        }}

        d["chosen_candidate"] = chosen_candidates
        d["rejected_candidate"] = rejected_candidates
        d["ablation"] = a_dict

        # 추가
        # extra_data_2.append(d)
        filtered.append(d)
        init_vqa_result.append(vqa_result_dict[d["item_id"]])

        if len(filtered) > 16000:
            break



# 정렬
filtered = sorted(filtered, key=lambda x: x["item_id"])
init_vqa_result = sorted(init_vqa_result, key=lambda x: x["item_id"])
print("After phase 5 - filtered dataset length:", len(filtered))
print("After phase 5 - init_vqa_result length:", len(init_vqa_result))



""" PHASE 6 """ 
# 저장
save_dir = "/nas2/data/Janus_dataset/next_v2"
os.makedirs(save_dir, exist_ok=True)

# save_init_path = os.path.join(save_dir, f"init_dataset_{len(filtered)}.json")
# save_vqa_path = os.path.join(save_dir, f"vqa_result_{len(init_vqa_result)}.json")

save_json(save_dir, save_name=f"init_dataset_{len(filtered)}", save_file=filtered)
save_json(save_dir, save_name=f"vqa_result_{len(init_vqa_result)}", save_file=init_vqa_result)