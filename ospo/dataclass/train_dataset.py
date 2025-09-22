# README: Dataset class only includes CPU operation.
# 'rejected_prompt' = 'rejected_prompt'

import os
import torch
import random
from PIL import Image
from typing import Dict
from torch.utils.data import Dataset

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.common import read_json
from ospo.utils.processor import get_conversation, get_sft_format


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
                 sampling_rate=1.0, num_samples=None, copo_mode=False): 
        
        self.chat_processor = chat_processor
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        # self.copo_mode = copo_mode 

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

    def decode_base(self, example: Dict):
        if "prompt" not in example.keys() or "chosen" not in example.keys() or "rejected" not in example.keys():
            raise ValueError(
                    f"Could not format example as dialogue for SimPO task!\nThis example only has {example.keys()} keys.\n"
                )
        
        item_id = example["item_id"]        
        text_token = self.get_text_token(example["prompt"]) 
        chosen_image_tensor = self.get_image_tensor(example["chosen"])
        rejected_image_tensor = self.get_image_tensor(example["rejected"])

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

        return item_id, chosen_text_token, rejected_text_token, chosen_image_tensor, rejected_image_tensor


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
