# Step 1. Initial Prompt Generation

import os
import datetime
import torch
import argparse
from peft import get_peft_model
from tqdm import tqdm

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.wrapper import JanusProElementGenWrapper
from ospo.dataclass import GenerationDataModule
from ospo.utils.generate import get_trainer
from ospo.utils.model import get_model, get_lora_config
from ospo.utils.common import build_config, set_seed, read_json, save_json
from ospo.constant import CATEGORY_LIST

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    max_len_dict = {'color1': 667, 
                    'color2': 667, 
                    'shape1': 667, 
                    'shape2': 667, 
                    'texture1': 667, 
                    'texture2': 667, 
                    '2D_spatial': 1000, 
                    '3D_spatial': 1000, 
                    'numeracy1': 1000, 
                    'numeracy2': 1000, 
                    'non-spatial': 4000, 
                    'complex': 4000}

    for category in tqdm(CATEGORY_LIST):
        # index initialize
        data_idx = 0 # 4000
        sub_category_data = []

        prefix = prefix_mapping_dict[category]
        item_list = item_mapping_dict[category]

        for fname in item_list:
            max_len = max_len_dict[fname]

            try: 
                data = read_json(os.path.join(element_dir, f"{fname}_prompt.json"))
                for i, sample in enumerate(data):
                    if i > max_len:
                        break

                    # only for numeracy
                    if category in ["numeracy1", "numeracy2"]:
                        number_words = extract_number_words(sample["prompt"])
                        if len(number_words) == 0:
                            continue
                        flag_bad = any(n not in _NUMBER_WORDS for n in number_words)
                        if flag_bad:
                            continue

                    sample['item_id'] = f"{prefix}{data_idx:06d}"        
                    data_idx += 1
                    # sub_category_data.extend(data)
                    sub_category_data.append(sample)
            except Exception as e:
                print(e)
                raise ValueError(f"Your element_dir does not have f'{fname}_prompt.json' file.")

        print(f"Ending index for category {category}:", data_idx)
        merged_data.extend(sub_category_data)

    print("Total Data Length: ", len(merged_data))

    # reordering
    reordered_data = []
    for d in merged_data:
        # print(d)
        dict_ud = {"item_id": d["item_id"], 
                    "t2i_category": d["t2i_category"], 
                    "sub_category": d["sub_category"]}
        for k, v in d.items():
            if k != "item_id" and k != "t2i_category" and k != "sub_category":
                dict_ud[k] = v
        reordered_data.append(dict_ud)


    # Save to JSON file
    save_json(save_root=element_dir,
              save_name=f'base_prompt_{len(reordered_data)}',
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

    # Start evaluation
    start_time = datetime.datetime.now()
    trainer.test(model, dataloaders=dataloader)

    create_item_id(config.save_path)
    end_time = datetime.datetime.now()

    elapsed_time = end_time - start_time
    elapsed_min = elapsed_time.total_seconds() / 60

    print('------------------------------------------')
    print(f"Elapsed Time: {elapsed_min:.2f} minutes")
    print('------------------------------------------')



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

    # # make item_id only
    # create_item_id("/nas2/data/Janus_dataset/next_v2/iter2/prompt/step1")
