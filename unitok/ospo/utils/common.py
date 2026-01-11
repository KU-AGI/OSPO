import os
import json
import random
import numpy as np
import torch
import torch.distributed as dist
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)

    @classmethod
    def from_nested_dicts(cls, data):
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(save_root, save_name, save_file):
    save_path = os.path.join(save_root,f'{save_name}.json')
    with open(save_path, 'w') as f:
        json.dump(save_file, f, indent=4)

def save_json_ddp(save_root, save_name, world_size, save_file, rank):
    # self.output_list, self.trainer.world_size
    os.makedirs(save_root, exist_ok=True)

    if world_size > 1:
        gathered_output_list = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_output_list, save_file)

        if rank == 0:
            seen = set()
            output_list = []
            for gathered_output in gathered_output_list:
                for sample in gathered_output:
                    item_id = sample['item_id']
                    if item_id in seen:
                        continue
                    seen.add(item_id)
                    output_list.append(sample)
            try:
                sorted_output = sorted(output_list, key=lambda x: int(x["item_id"]))
            except ValueError:
                print("Skipping sorting due to non-integer item_id.")
                sorted_output = output_list  # preserve original order
            save_json(save_root, save_name, sorted_output)
    else:
        save_json(save_root, save_name, save_file)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    seed_everything(seed, workers=True) 

def get_rank(dist):
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def override_from_file_name(cfg):
    if 'cfg_path' in cfg and cfg.cfg_path is not None:
        file_cfg = OmegaConf.load(cfg.cfg_path)
        cfg = OmegaConf.merge(cfg, file_cfg)
    return cfg

def override_from_cli(cfg):
    c = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, c)
    return cfg

def to_attr_dict(cfg):
    c = OmegaConf.to_container(cfg)
    cfg = AttrDict.from_nested_dicts(c)
    return cfg

def build_config(struct=False, cfg_path=None):
    if cfg_path is None:
        raise ValueError("No cfg_path given.")
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.set_struct(cfg, struct)
    cfg = override_from_file_name(cfg)
    cfg = override_from_cli(cfg)

    # cfg_yaml = OmegaConf.to_yaml(cfg)
    cfg = to_attr_dict(cfg) 
    return cfg

def save_config(save_path, config):
    config_save_path = os.path.join(save_path, 'config.yaml') 
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=4)

