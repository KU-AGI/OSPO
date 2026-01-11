# README: Dataset class only includes CPU operation.

import os
import json
import sys
import torch
import random
from typing import *
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
# sys.path.append("./eval/liquid") # TODO

from unitok.utils.data import normalize_01_into_pm1
from unitok.eval.liquid.constants import ( # from constants import (
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN
)

from unitok.ospo.utils.common import read_json
from unitok.ospo.utils.processor import get_conversation


class ValidationDataset(Dataset):
    def __init__(self, data_path):
        self.dataset = read_json(data_path)
        self.total_length = len(self.dataset)
        print(f"Total length of data: {self.total_length}")

    def __len__(self):  
        return self.total_length

    def __getitem__(self, idx):
        return self.dataset[idx]


class PreferenceDataset(Dataset):
    def __init__(self, seed, data_path, tokenizer, 
                 sampling_rate=1.0, num_samples=None, copo_mode=False, use_mask=False, mask_dir=None): 
        
        self.eoi = torch.tensor([4])
        self.boi = torch.tensor([3])
        self.eos = torch.tensor([2])
        self.bos = torch.tensor([1])
        self.tokenizer = tokenizer
        self.image_processor = transforms.Compose([
            transforms.Resize(int(256 * 1.125)),
            transforms.CenterCrop(256),
            transforms.ToTensor(), normalize_01_into_pm1,
        ])

        self.use_mask = use_mask
        self.mask_dir = mask_dir
        if self.use_mask and not os.path.exists(self.mask_dir):
            raise ValueError(f"Non-existing mask dir: {self.mask_dir}")

        # load data
        self.dataset = read_json(data_path)

        # Apply num_samples if specified
        if num_samples is not None:
            assert num_samples > 0, "num_samples must be greater than 0"
            assert num_samples <= len(self.dataset), "num_samples cannot exceed dataset size"

            # Deterministic sampling
            rng = random.Random(seed)
            indices = rng.sample(range(len(self.dataset)), num_samples)
            self.dataset = [self.dataset[i] for i in indices]

        elif sampling_rate != 1.0:
            self.total_length = int(len(self.dataset) * sampling_rate)
            assert self.total_length > 0, "Dataset size must be bigger than 1."
            self.dataset = self.dataset[:self.total_length] # from the front

        self.total_length = len(self.dataset)
        print(f"Total length of data: {self.total_length}")

        # according to copo_mode,
        self.collate_fn = self.collate_fn_copo if copo_mode else self.collate_fn_base
        self.decode_fn = self.decode_copo if copo_mode else self.decode_base

    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        try:
            example = self.dataset[idx]
            return self.decode_fn(example)
        except Exception as e:
            print(f"Error decoding example at index {idx}: {e}")
            return self.__getitem__(idx + 1)  # Try the next index

    def collate_fn_base(self, batch):
        # text tokens is 'chosen' text tokens
        if self.use_mask:        
            item_ids, text_tokens, chosen_image_tensors, rejected_image_tensors, chosen_mask, rejected_mask = zip(*batch)
            return list(item_ids), list(text_tokens), list(chosen_image_tensors), list(rejected_image_tensors), list(chosen_mask), list(rejected_mask)
        else:
            item_ids, text_tokens, chosen_image_tensors, rejected_image_tensors = zip(*batch)
            return list(item_ids), list(text_tokens), list(chosen_image_tensors), list(rejected_image_tensors)

    def collate_fn_copo(self, batch):
        raise NotImplementedError("CoPO is not supported.")

    def get_image_generation_prompt(self, prompt):
        converation = get_conversation(prompt, cfg_prob = 1.0)

        return converation

    def get_text_token(self, text):
        prompt = self.get_image_generation_prompt(text)
        text_input_ids = self.tokenizer(prompt, return_tensors="pt", padding="longest",max_length=self.tokenizer.model_max_length, truncation=True)['input_ids'][0]
        text_input_ids = torch.LongTensor(text_input_ids) # e.g. torch.Size([18])

        return text_input_ids

    def get_image_tensor(self, img_path: str):
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.image_processor(image)
    
        return image_tensor


    def decode_base(self, example: Dict):
        if "prompt" not in example.keys() or "chosen" not in example.keys() or "rejected" not in example.keys():
            raise ValueError(
                    f"Could not format example as dialogue for SimPO task!\nThis example only has {example.keys()} keys.\n"
                )
        
        item_id = example["item_id"]        
        text_token = self.get_text_token(example["prompt"]) 
        chosen_image_tensor = self.get_image_tensor(example["chosen"])
        rejected_image_tensor = self.get_image_tensor(example["rejected"])

        expand_token = torch.tensor([self.boi, IMAGE_TOKEN_INDEX, self.eoi, self.eos], dtype=torch.long)
        text_token = torch.cat([text_token, expand_token])

        if self.use_mask:
            chosen_mask, rejected_mask = self.load_object_mask(example)
            return item_id, text_token, chosen_image_tensor, rejected_image_tensor, chosen_mask, rejected_mask

        return item_id, text_token, chosen_image_tensor, rejected_image_tensor


    def decode_copo(self, example: Dict):
        if "prompt" not in example.keys() or "chosen" not in example.keys() or "rejected" not in example.keys():
            raise ValueError(
                    f"Could not format example as dialogue for SimPO task!\nThis example only has {example.keys()} keys.\n"
                )
        
        item_id = example["item_id"]        
        chosen_text_token = self.get_text_token(example["prompt"]) 
        rejected_text_token = self.get_text_token(example["negative_short_prompt"])

        chosen_image_tensor = self.get_image_tensor(example["chosen"])
        rejected_image_tensor = self.get_image_tensor(example["rejected"])

        expand_token = torch.tensor([self.boi, IMAGE_TOKEN_INDEX, self.eoi, self.eos], dtype=torch.long)
        chosen_text_token = torch.cat([chosen_text_token, expand_token])
        rejected_text_token = torch.cat([rejected_text_token, expand_token])

        return item_id, chosen_text_token, rejected_text_token, chosen_image_tensor, rejected_image_tensor

    
    # 개별 샘플 기준
    def load_object_mask(self, example):
        # Do not consider off_policy case !
        chosen_mask, rejected_mask = None, None
        basename = os.path.basename(example["chosen"]).split(".png")[0]

        for key in ['base', 'negative']:
            mask_path = os.path.join(self.mask_dir, key, example["t2i_category"], example["item_id"], f"{basename}_mask.pt")
            if not os.path.exists(mask_path):                
                # continue
                raise ValueError(f"mask path: {mask_path} dose not exist!")
            else: # mask path is existed.
                if key == 'base':
                    chosen_mask = torch.load(mask_path) # .to('cuda')
                else:
                    rejected_mask = torch.load(mask_path)

        return chosen_mask, rejected_mask


### Evaluation
class GenEval(Dataset):
    def __init__(self, data_path): 
        with open(data_path, 'r') as f:
            self.dataset = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], idx


class T2ICompBench(Dataset):
    def __init__(self, 
                data_dir, 
                category_list: list = None, 
                split=None,
                s_idx=None,
                e_idx=None):

        self.full_data = []
        if category_list is None:
            raise ValueError("Category list is not given.")
        else:
            print(f"Category list: {category_list}")

        split = "" if split is None else "_" + split        
        print(f"Load {split} split.")

        for category in category_list:
            idx = 0
            self.data = []
            with open(os.path.join(data_dir, f"{category}{split}.txt"), 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    self.data.append({'category': category, 'caption': line, 'idx': idx})
                    idx += 1
            self.data = split_size(self.data, s_idx, e_idx)
            self.full_data.extend(self.data)
        
        self.total_length = len(self.full_data)
        print("Total length of eval dataset: ", self.total_length)
        print()
        
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        return self.full_data[idx]


class DPGEval(Dataset):
    def __init__(self, 
                 file_path,
                 s_idx=None,
                 e_idx=None):
        
        self.dataset = []

        file_list = sorted(glob(os.path.join(file_path,'*.txt')))

        for data_idx, file in enumerate(file_list):
            with open(file, 'r') as f:
                text = f.read()
            data = {'idx': data_idx, 
                    'prompt': text, 
                    'file_name': file.split('/')[-1].split('.')[0]}
            self.dataset.append(data)
            # data = {'prompt': text, 'file_name': file.split('/')[-1].split('.')[0]}
            # self.dataset.append(data)

        self.dataset = split_size(self.dataset, s_idx, e_idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # return self.dataset[idx], idx
        sample = self.dataset[idx]
        # return sample, sample['idx']
        return sample