# Step 2-2. Image Generation

import os
import time
import argparse
import torch
from peft import get_peft_model

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.dataclass import GenerationDataModule
from ospo.wrapper import JanusProImageGenWrapper
from ospo.utils.generate import get_trainer
from ospo.utils.model import get_model, get_lora_config
from ospo.utils.common import build_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_dataloader(config):
    datamodule = GenerationDataModule(config, step=2)  
    dataloader = datamodule.gen_dataloader()
    return dataloader 


def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    assert len(config.seed_list) == 3, "Please set 3 seeds for 3 perturbation types." 

    vl_chat_processor, tokenizer, model = get_model(mode='generate', config=config)

    if config.ckpt_path is not None:
        print("# Load model with checkpoint.")
        lora_config = get_lora_config(config.ckpt_path)
        
        model.language_model = get_peft_model(model.language_model, lora_config)
        model = JanusProImageGenWrapper.load_from_checkpoint(checkpoint_path=config.ckpt_path, 
                                                        config=config,
                                                        model=model,
                                                        tokenizer=tokenizer,
                                                        processor=vl_chat_processor,
                                                        strict=False) 
        model.setup("test")
        model.model.language_model = model.model.language_model.merge_and_unload() 

    else:
        print("# Load base model.")
        model = JanusProImageGenWrapper(config=config,
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    processor=vl_chat_processor)

    if config.data_path is None:
        config.data_path = os.path.join(os.path.dirname(config.save_path), 'step2', 'long_prompt.json')
    dataloader = get_dataloader(config)
    trainer = get_trainer(device, config.world_size)
    trainer.test(model, dataloaders=dataloader)
    print("(Step 3) Image generation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/step2_2.yaml")
    args, unknown = parser.parse_known_args()  
    config = build_config(cfg_path=args.cfg_path)
    
    start = time.time()

    main(config)

    end = time.time()
    elapsed = end - start
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    print(f"End Time:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
    print(f"Elapsed Time: {elapsed:.2f} seconds")