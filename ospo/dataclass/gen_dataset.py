
import os, json
from torch.utils.data import Dataset, DataLoader


def split_size(data:list, s_idx=None, e_idx=None):
    if s_idx is not None and e_idx is not None:
        data = data[s_idx:e_idx]
    elif s_idx is not None:
        data = data[s_idx:]
    elif e_idx is not None:
        data = data[:e_idx]
    return data


class BaseDataset(Dataset):
    def __init__(self, 
                 fpath,
                 s_idx=None, 
                 e_idx=None): 

        with open(fpath, 'r') as f:
            self.data = json.load(f)

        # If 'p_method' is not included in keys, add it manually.
        if self.data[0].get("perturbed_method", None) is None:
            for sample in self.data:
                if sample["sub_category"] in ["attribute1_color", "attribute1_texture", "attribute1_shape", "layout2"]:
                    sample["perturbed_method"] = ["replace", "replace", "replace"]
                elif sample["sub_category"] == "non-spatial":
                    sample["perturbed_method"] = ["replace", "drop", "replace"]
                elif sample["sub_category"] in ["complex", "attribute2", "layout1", "layout3"]:
                    sample["perturbed_method"] = ["replace", "swap", "drop"]
                else:
                    raise ValueError(f"Unknown sub_category: {sample['sub_category']}")

        self.data = split_size(self.data, s_idx, e_idx)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class NegativeDataset(Dataset):
    def __init__(self, 
                 base_path, # answer
                 negative_qs_path, # question  
                ): 
        
        with open(base_path, 'r') as f1, open(negative_qs_path, 'r') as f2:
            base_data = json.load(f1)
            negative_qs_data = json.load(f2)
            negative_qs_dict = {item['item_id']: item for item in negative_qs_data}

        self.dataset = []
        # merge two data
        for sample in base_data:
            item_id = sample["item_id"]

            rejected = negative_qs_dict.get(item_id, None)
            if rejected is not None: 
                # Both are List
                sample["rejected_prompt"] = rejected["negative_prompt"] # rejected["rejected_prompt"]
                sample["rejected_question"] = rejected["rejected_question"] 
                self.dataset.append(sample)


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
