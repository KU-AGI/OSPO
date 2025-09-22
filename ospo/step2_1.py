# Step 2-1. Negative / Dense Prompt Generation

import os
import argparse
import torch
from peft import get_peft_model

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.wrapper import JanusProNegativeGenWrapper, JanusProDenseGenWrapper
from ospo.dataclass import GenerationDataModule
from ospo.utils.generate import get_trainer
from ospo.utils.model import get_model, get_lora_config
from ospo.utils.common import build_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_dataloader(config):
    datamodule = GenerationDataModule(config, step=2)  
    dataloader = datamodule.gen_dataloader()
    return dataloader 


def get_wrapper(config, model, tokenizer, vl_chat_processor, wrapper_cls):
    if config.ckpt_path is not None:
        print("# Load model with checkpoint.")
        lora_config = get_lora_config(config.ckpt_path)
        model.language_model = get_peft_model(model.language_model, lora_config)
        model = wrapper_cls.load_from_checkpoint(
            checkpoint_path=config.ckpt_path, 
            config=config,
            model=model,
            tokenizer=tokenizer,
            processor=vl_chat_processor,
            strict=False
        )
        model.setup("test")
        model.model.language_model = model.model.language_model.merge_and_unload()
    else:
        print("# Load base model.")
        model = wrapper_cls(
            config=config,
            model=model,
            tokenizer=tokenizer,
            processor=vl_chat_processor
        )
    return model


def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    assert len(config.seed_list) == 3, "Please set 3 seeds for 3 perturbation types." 
    
    trainer = get_trainer(device, config.world_size)
    vl_chat_processor, tokenizer, model = get_model(mode='generate', config=config)


    # 1. negative prompt generation
    if config.data_path is None:
        config.data_path = os.path.join(os.path.dirname(config.save_path), 'step1', 'base_prompt.json') # default fname
    if not os.path.exists(config.data_path):
        raise ValueError("The data path is not existed or None.")
    dataloader_negative = get_dataloader(config)
    
    model_negative = get_wrapper(config, model, tokenizer, vl_chat_processor, 
                                wrapper_cls=JanusProNegativeGenWrapper)
    
    trainer.test(model_negative, dataloaders=dataloader_negative)
    print("(Step 2) Negative prompt generation completed.")


    # 2. densification
    if config.data_path is None:
        config.data_path = os.path.join(config.save_path, 'negative_prompt.json') # default fname
    dataloader_dense = get_dataloader(config)
    
    model_dense = get_wrapper(config, model, tokenizer, vl_chat_processor,
                                wrapper_cls=JanusProDenseGenWrapper)

    trainer.test(model_dense, dataloaders=dataloader_dense)
    print("(Step 2) Dense prompt generation completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/step2_1.yaml")
    args, unknown = parser.parse_known_args()  
    config = build_config(cfg_path=args.cfg_path)
    
    main(config)