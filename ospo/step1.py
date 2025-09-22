# Step 1. Initial Prompt Generation

import os
import torch
import argparse
from peft import get_peft_model

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.wrapper import JanusProElementGenWrapper
from ospo.dataclass import GenerationDataModule
from ospo.utils.generate import get_trainer
from ospo.utils.model import get_model, get_lora_config
from ospo.utils.common import build_config, set_seed, read_json, save_json
from ospo.constant import CATEGORY_LIST

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_item_id(element_dir):
    merged_data = []    
    item_mapping_dict = {
        "attribute": ['color1', 'color2', 'shape1', 'shape2', 'texture1', 'texture2'],
        "layout": ['2D_spatial', '3D_spatial', 'numeracy1', 'numeracy2'],
        "non-spatial": ['non-spatial'],
        "complex": ['complex']
    }
    prefix_mapping_dict = {
        "attribute": 0,
        "layout": 1,
        "non-spatial": 2,
        "complex": 3
    }

    for category in CATEGORY_LIST:
        # index initialize
        data_idx = 0 # 4000
        sub_category_data = []

        prefix = prefix_mapping_dict[category]
        item_list = item_mapping_dict[category]

        for fname in item_list:
            try: 
                data = read_json(os.path.join(element_dir, f"{fname}_element.json"))
                for sample in data:
                    sample['item_id'] = f"{prefix}{data_idx:06d}"        
                    data_idx += 1
                sub_category_data.extend(data)
            except:
                raise ValueError(f"Your element_dir does not have f'{fname}_element.json' file.")

        print(f"Ending index for category {category}:", data_idx)
        merged_data.extend(sub_category_data)

    print("Total Data Length: ", len(merged_data))

    # reordering
    reordered_data = []
    for d in merged_data:
        dict_ud = {"item_id": d["item_id"], 
                    "t2i_category": d["t2i_category"], 
                    "sub_category": d["sub_category"]}
        for k, v in d.items():
            if k != "item_id" and k != "t2i_category" and k != "sub_category":
                dict_ud[k] = v
        reordered_data.append(dict_ud)


    # Save to JSON file
    save_json(save_root=element_dir,
              save_name='base_prompt',
              save_file=reordered_data)

    print("Done.")


def get_dataloader(config):
    datamodule = GenerationDataModule(config, step=1)  
    dataloader = datamodule.gen_dataloader()
    return dataloader 

def main(config):
    if config.batch_size > 1 or config.world_size > 1:
        raise NotImplementedError("Batch size > 1 and World size > 1 are not supported in this step.")

    device = "cuda" if torch.cuda.is_available() else "cpu"    
    set_seed(config.seed)
    if config.save_path is not None:
        os.makedirs(config.save_path, exist_ok=True)
        print("# Save path: ", config.save_path)

    dataloader = get_dataloader(config)
    vl_chat_processor, tokenizer, model = get_model(mode='generate', config=config)

    if config.ckpt_path is not None:
        print("# Load model with checkpoint.")
        lora_config = get_lora_config(config.ckpt_path)
        
        model.language_model = get_peft_model(model.language_model, lora_config)
        model = JanusProElementGenWrapper.load_from_checkpoint(checkpoint_path=config.ckpt_path, 
                                                            config=config,
                                                            model=model,
                                                            tokenizer=tokenizer,
                                                            processor=vl_chat_processor,
                                                            strict=False) 
        model.setup("test")
        model.model.language_model = model.model.language_model.merge_and_unload() 

    else:
        print("# Load base model.")
        model = JanusProElementGenWrapper(config=config,
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    processor=vl_chat_processor)

    trainer = get_trainer(device, config.world_size)
    trainer.test(model, dataloaders=dataloader)

    create_item_id(config.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default='configs/step1.yaml')
    parser.add_argument("--category", type=str, default=None, help="option: color1, color2, texture1, texture2, shape1, shape2, 2D_spatial, 3D_spatial, numeracy1, numeracy2, non-spatial, complex")
    args, unknown = parser.parse_known_args()  
    
    config = build_config(cfg_path=args.cfg_path)
    if args.category is not None:
        config.category = args.category
    print("# Category:", config.category)


    main(config)

