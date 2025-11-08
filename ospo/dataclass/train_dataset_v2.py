# README: Dataset class only includes CPU operation.
# 'rejected_prompt' = 'rejected_prompt'
# mask loading

import os
import torch
import random
from PIL import Image
from typing import Dict
from torch.utils.data import Dataset
from tqdm import tqdm

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.common import read_json
from ospo.utils.processor import get_conversation, get_sft_format

from pathlib import Path
from tqdm import tqdm

def build_mask_index(mask_dir: str, subdir: str) -> set:
    """
    Returns a set of keys like (t2i_category, item_id, basename)
    for all files named '*_mask.pt' under mask_dir/subdir/**.
    """
    base = Path(mask_dir) / subdir
    keys = set()
    # Walk once; pathlib.rglob is implemented in C and quite fast
    for p in tqdm(base.rglob("*_mask.pt"), desc="Building mask index ..."):
        # Expect path: mask_dir/subdir/<t2i_category>/<item_id>/<basename>_mask.pt
        try:
            item_id = p.parent.name
            t2i_category = p.parent.parent.name
            basename = p.stem[:-5] if p.stem.endswith("_mask") else p.stem
            keys.add((t2i_category, item_id, basename))
        except Exception as e:
            print(e) 
            # skip unexpected layouts
            continue
    return keys

# total 6
class ValidationDataset(Dataset):
    def __init__(self, data_path: str,
                 chat_processor, image_processor, tokenizer, task_id, copo_mode=False): 
        """
        Args:
        data_path (str): Validation dataset path for task_id.
        0: Self-VQA score
        1: General validation 
        2: Hard-Negative (Image) score (in-distribution)
        3: Hard-Negative (Image) score (out-of-distribution)
        4: Hard-Negative (Text) score (in-distribution)
        5: Hard-Negative (Text) score (out-of-distribution)
        """
        self.dataset = []
        self.chat_processor = chat_processor
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.task_id = task_id    
        self.copo_mode = copo_mode

        assert os.path.exists(data_path), f"Data path {data_path} does not exist for task {task_id}."
        data = read_json(data_path)
        
        for sample in data:
            sample["task_id"] = task_id    
        self.dataset.extend(data)

        
        self.total_length = len(self.dataset)
        print(f"Total length of combined validation data: {self.total_length}")
        if self.total_length == 0:
            raise ValueError("Combined validation dataset is empty. Please check the provided data paths.")

    def __len__(self):  
        return self.total_length

    def __getitem__(self, idx):
        try:
            example = self.dataset[idx]
            return self.decode(example)
        except Exception as e:
            print(f"Error decoding example at index {idx}: {e}")
            return self.__getitem__(idx + 1)  # Try the next index
    
    def collate_fn(self, batch): 
        task_ids, item_ids, text_tokens, optionals, chosen_image_tensors, rejected_image_tensors = zip(*batch)
        return list(task_ids), list(item_ids), list(text_tokens), list(optionals), list(chosen_image_tensors), list(rejected_image_tensors)

    def get_image_generation_prompt(self, prompt):
        system_prompt = ""
        converation = get_conversation(prompt)
        sft_format = get_sft_format(self.chat_processor, system_prompt, converation)
        prompt = sft_format + self.chat_processor.image_start_tag

        return prompt

    def get_text_token(self, text, prepare_batch_mode=False):
        parallel_size=1 
        prompt = self.get_image_generation_prompt(text)
        text_input_ids = self.tokenizer.encode(prompt)
        text_input_ids = torch.LongTensor(text_input_ids) # e.g. torch.Size([18])

        if prepare_batch_mode:
            return text_input_ids
        
        else:
            text_tokens = torch.zeros((parallel_size, len(text_input_ids)), dtype=torch.int) 
            for i in range(parallel_size):
                text_tokens[i, :] = text_input_ids 

            return text_tokens

    def get_image_tensor(self, img_path: str):
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.image_processor([image])
        image_tensor = image_tensor['pixel_values']  # e.g. torch.Size([1, 3, 384, 384])
    
        return image_tensor

    def decode(self, example: Dict):

        task_id = example["task_id"]
        item_id = example["item_id"]     
        
        if task_id == 0:
            optional = example["question"][:-1] 
            text_token = self.get_text_token(example["prompt"], prepare_batch_mode=True) # without reshape
            chosen_image_tensor, rejected_image_tensor = None, None
            # perturbed_token, chosen_image_tensor, rejected_image_tensor = None, None, None          

        else:
            if "prompt" not in example.keys() or "chosen" not in example.keys() or "rejected" not in example.keys():
                raise ValueError(
                        f"(Task {example['task_id']}) Could not format example as dialogue for SimPO task!\nThis example only has {example.keys()} keys.\n"
                    )

            text_token = self.get_text_token(example["prompt"]) 
            chosen_image_tensor = self.get_image_tensor(example["chosen"])
            rejected_image_tensor = self.get_image_tensor(example["rejected"])
            
            if task_id == 4 or task_id == 5 or (self.copo_mode and task_id == 1):
                if "rejected_prompt" not in example.keys():
                    raise ValueError(
                            f"(Task {task_id}) Could not format example as dialogue for SimPO task!\nThis example only has {example.keys()} keys.\n"
                        )
                # perturbed_token
                optional = self.get_text_token(example["rejected_prompt"])

            else:
                optional = None

        return task_id, item_id, text_token, optional, chosen_image_tensor, rejected_image_tensor


class PreferenceDataset(Dataset):
    def __init__(self, seed, data_path, 
                 chat_processor, image_processor, tokenizer, 
                 sampling_rate=1.0, num_samples=None, copo_mode=False, use_mask=False, mask_dir=None, off_policy=False): 
        
        self.chat_processor = chat_processor
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        # self.copo_mode = copo_mode 
        self.use_mask = use_mask
        self.mask_dir = mask_dir
        if self.use_mask and not os.path.exists(self.mask_dir):
            raise ValueError(f"Non-existing mask dir: {self.mask_dir}")
        self.off_policy = off_policy

        # load data
        self.dataset = read_json(data_path)

        # 사전 처리로 대체
        # if self.use_mask:
        #     filtered = []

        #     # # mask_dir 이 부재한 샘플은 사전 배제
        #     # for example in tqdm(self.dataset, desc="Checking masking is available ..."):
        #     #     basename = os.path.basename(example["chosen"]).split(".png")[0]
        #     #     c_mask_path = os.path.join(self.mask_dir, "base", example["t2i_category"], example["item_id"], f"{basename}_mask.pt")
        #     #     r_mask_path = os.path.join(self.mask_dir, "negative", example["t2i_category"], example["item_id"], f"{basename}_mask.pt")
        #     #     if not os.path.exists(c_mask_path) or not os.path.exists(r_mask_path):
        #     #         print(f"Filtered; No mask (item_id: {example['item_id']})")
        #     #     else:
        #     #         filtered.append(example)


        #     # 1) Build indices once
        #     base_keys     = build_mask_index(self.mask_dir, "base")
        #     rejected_keys = build_mask_index(self.mask_dir, "negative")  

        #     # 2) Filter in O(N)
        #     filtered = []
        #     missed   = 0

        #     for ex in tqdm(self.dataset, desc="Filtering by prebuilt mask index..."):
        #         # safer & faster than split('.png')[0]
        #         basename = Path(ex["chosen"]).stem
        #         key = (ex["t2i_category"], ex["item_id"], basename)

        #         has_c = key in base_keys
        #         has_r = key in rejected_keys  # or `has_r = has_c` if you only have one mask per item

        #         if has_c and has_r:
        #             filtered.append(ex)
        #         else:
        #             missed += 1

        #     print(f"Kept: {len(filtered)} | Filtered (no mask): {missed}")

            
        #     # replace
        #     self.dataset = filtered


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
        if self.use_mask:        
            item_ids, chosen_text_tokens, rejected_text_tokens, chosen_image_tensors, rejected_image_tensors, chosen_mask, rejected_mask = zip(*batch)
            return list(item_ids), list(chosen_text_tokens), list(rejected_text_tokens), list(chosen_image_tensors), list(rejected_image_tensors), list(chosen_mask), list(rejected_mask)
        else:
            item_ids, chosen_text_tokens, rejected_text_tokens, chosen_image_tensors, rejected_image_tensors = zip(*batch)
            return list(item_ids), list(chosen_text_tokens), list(rejected_text_tokens), list(chosen_image_tensors), list(rejected_image_tensors)

    def decode_base(self, example: Dict):
        if "prompt" not in example.keys() or "chosen" not in example.keys() or "rejected" not in example.keys():
            raise ValueError(
                    f"Could not format example as dialogue for SimPO task!\nThis example only has {example.keys()} keys.\n"
                )
        
        item_id = example["item_id"]        
        text_token = self.get_text_token(example["prompt"]) 
        chosen_image_tensor = self.get_image_tensor(example["chosen"])
        rejected_image_tensor = self.get_image_tensor(example["rejected"])

        if self.use_mask:
            chosen_mask, rejected_mask = self.load_object_mask(example)
            return item_id, text_token, chosen_image_tensor, rejected_image_tensor, chosen_mask, rejected_mask

        return item_id, text_token, chosen_image_tensor, rejected_image_tensor

    def decode_copo(self, example: Dict):
        if "prompt" not in example.keys() or "chosen" not in example.keys() or "rejected" not in example.keys() or "rejected_prompt" not in example.keys():
            raise ValueError(
                    f"Could not format example! This example only has {example.keys()} keys.\n"
                )
        
        item_id = example["item_id"]        

        chosen_text_token = self.get_text_token(example["prompt"]) 
        rejected_text_token = self.get_text_token(example["rejected_prompt"])

        chosen_image_tensor = self.get_image_tensor(example["chosen"])
        rejected_image_tensor = self.get_image_tensor(example["rejected"])

        if self.use_mask:
            chosen_mask, rejected_mask = self.load_object_mask(example)
            return item_id, chosen_text_token, rejected_text_token, chosen_image_tensor, rejected_image_tensor, chosen_mask, rejected_mask

        return item_id, chosen_text_token, rejected_text_token, chosen_image_tensor, rejected_image_tensor

    # 개별 샘플 기준
    def load_object_mask(self, example):
        chosen_mask, rejected_mask = None, None
    
        if not self.off_policy:
            basename = os.path.basename(example["chosen"]).split(".png")[0]
            for key in ['base', 'negative']:
                mask_path = os.path.join(self.mask_dir, key, example["t2i_category"], example["item_id"], f"{basename}_mask.pt")
                if not os.path.exists(mask_path):                
                    continue
                else: # mask path is existed.
                    if key == 'base':
                        chosen_mask = torch.load(mask_path) # .to('cuda')
                    else:
                        rejected_mask = torch.load(mask_path)

        else:
            # FocusDiff 기준
            base_mask_path = os.path.join(self.mask_dir, "data", "images", example["item_id"])
            chosen_mask_path = os.path.join(base_mask_path, "image1_mask.pt")
            rejected_mask_path = os.path.join(base_mask_path, "image2_mask.pt")
            
            # mask is always exist (pre-filtering)
            chosen_mask = torch.load(chosen_mask_path)
            rejected_mask = torch.load(rejected_mask_path)

        return chosen_mask, rejected_mask

    def get_image_generation_prompt(self, prompt):
        system_prompt = ""
        converation = get_conversation(prompt)
        sft_format = get_sft_format(self.chat_processor, system_prompt, converation)
        prompt = sft_format + self.chat_processor.image_start_tag

        return prompt

    def get_text_token(self, text):
        parallel_size=1 
        prompt = self.get_image_generation_prompt(text)
        text_input_ids = self.tokenizer.encode(prompt)
        text_input_ids = torch.LongTensor(text_input_ids) # e.g. torch.Size([18])

        text_tokens = torch.zeros((parallel_size, len(text_input_ids)), dtype=torch.int) 
        for i in range(parallel_size):
            text_tokens[i, :] = text_input_ids 

        return text_tokens

    def get_image_tensor(self, img_path: str):
        image = Image.open(img_path)
        image_tensor = self.image_processor([image])
        image_tensor = image_tensor['pixel_values']  # e.g. torch.Size([1, 3, 384, 384])
       
        return image_tensor



# Not Used
class PreferenceWithPerturbationDataset(Dataset):
    def __init__(self, seed, data_path, 
                 chat_processor, image_processor, tokenizer, 
                 sampling_rate=1.0, num_samples=None): 
        
        self.chat_processor = chat_processor
        self.image_processor = image_processor
        self.tokenizer = tokenizer

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

    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        return self.decode(example) 

    def collate_fn(self, batch): 
        item_ids, text_tokens, perturbed_tokens, chosen_image_tensors, rejected_image_tensors = zip(*batch)
        return list(item_ids), list(text_tokens), list(perturbed_tokens), list(chosen_image_tensors), list(rejected_image_tensors)

    # custom functions

    def get_image_generation_prompt(self, prompt):
        system_prompt = ""
        converation = get_conversation(prompt)
        sft_format = get_sft_format(self.chat_processor, system_prompt, converation)
        prompt = sft_format + self.chat_processor.image_start_tag

        return prompt

    def get_text_token(self, text):
        parallel_size=1 
        prompt = self.get_image_generation_prompt(text)
        text_input_ids = self.tokenizer.encode(prompt)
        text_input_ids = torch.LongTensor(text_input_ids) # e.g. torch.Size([18])

        text_tokens = torch.zeros((parallel_size, len(text_input_ids)), dtype=torch.int) 
        for i in range(parallel_size):
            text_tokens[i, :] = text_input_ids 

        return text_tokens

    def get_image_tensor(self, img_path: str):
        image = Image.open(img_path)
        image_tensor = self.image_processor([image])
        image_tensor = image_tensor['pixel_values']  # e.g. torch.Size([1, 3, 384, 384])
       
        return image_tensor

    def decode(self, example: Dict):
        if "object_replacement" not in example.keys() or "prompt" not in example.keys() or "chosen" not in example.keys() or "rejected" not in example.keys():
            raise ValueError(
                    f"Could not format example as dialogue for SimPO task!\nThis example only has {example.keys()} keys.\n"
                )

        item_id = example["item_id"]        
        text_token = self.get_text_token(example["prompt"]) 
        perturbed_token = self.get_text_token(example["object_replacement"])

        chosen_image_tensor = self.get_image_tensor(example["chosen"])
        rejected_image_tensor = self.get_image_tensor(example["rejected"])

        return item_id, text_token, perturbed_token, chosen_image_tensor, rejected_image_tensor
