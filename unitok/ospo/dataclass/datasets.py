# README: Dataset class only includes CPU operation.

import os
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
                 sampling_rate=1.0, num_samples=None, copo_mode=False): 
        
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
        item_ids, text_tokens, chosen_image_tensors, rejected_image_tensors = zip(*batch)
        return list(item_ids), list(text_tokens), list(chosen_image_tensors), list(rejected_image_tensors)

    def collate_fn_copo(self, batch):
        item_ids, chosen_text_tokens, rejected_text_tokens, chosen_image_tensors, rejected_image_tensors = zip(*batch)
        return list(item_ids), list(chosen_text_tokens), list(rejected_text_tokens), list(chosen_image_tensors), list(rejected_image_tensors)

    # custom functions

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