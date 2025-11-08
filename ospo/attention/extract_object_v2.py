# conda activate janus2

import os
import re
import spacy
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.common import build_config, read_json, save_json, set_seed

nlp = spacy.load("en_core_web_sm")

# OBJECT 에 예외적으로 포함되지 않는 명사 목록
_EXCEPTION_WORD = [
    "left", "right", "bottom", "top", "side", "row", "rows", "column", "columns", "front", "behind", "range"
]

def get_object_noun(init_data):
    
    object_data = list () # dict()

    for sample in tqdm(init_data, desc="Parsing nouns from prompt ..."):

        item_id = sample["item_id"]
        prompt = sample["prompt"]
        # relation = sample["relation"]
        # t2i_category = sample["t2i_category"]

        obj_info = set()

        # Parse the text
        doc = nlp(prompt)

        # Extract all nouns and proper nouns
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and token.text not in _EXCEPTION_WORD:
                obj_info.add(token.text)

        # Stil
        if len(obj_info) == 0:
            print(f"Extraction failed: {item_id}") 
            continue # 다 이유가 있는 법. (프롬프트 결함)

        sample['nouns'] = list(obj_info)
        object_data.append(sample)

    return object_data



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_dpath", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args, unknown = parser.parse_known_args()  
    

    # noun_path = "/home/yjoh/project/OSPO/next/obj_nouns.json"
    # noun_data = read_json(noun_path)

    # # Null noun sample 체크
    # cnt = 0
    # for d in noun_data:
    #     if len(d["noun"]) == 0:
    #         cnt += 1

    # print(cnt)

    # init_dpath = "/nas2/data/Janus_dataset/next_v2/train/train_mode_argmax_chosen_10435.json"
    # init_dpath = "/nas2/data/Janus_dataset/next_v2/train/train_mode_random_15966.json"
    init_data = read_json(args.init_dpath)

    object_data = get_object_noun(init_data)

    # save_dir = "/nas2/data/Janus_dataset/next_v2/train_object"
    if args.save_dir is None:
        args.save_dir = os.path.dirname(args.init_dpath)
    else:
        os.makedirs(args.save_dir, exist_ok=True)

    length = len(object_data)
    fname = "_".join(args.init_dpath.split("/")[-1].split("_")[:-1])
    save_json(save_root=args.save_dir, save_name=f"{fname}_object_{length}", save_file=object_data)



"""
실행 결과 (argmax_chosen)

Extraction failed: 0001011
Extraction failed: 0001303
Extraction failed: 0001907
Extraction failed: 1002696
Extraction failed: 1008599
Extraction failed: 2001231
"""