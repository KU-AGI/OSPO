# Step 3-2. Filtering and Selection

import os
import torch
import argparse
from pytorch_lightning import seed_everything
from peft import get_peft_model

from ospo.utils.model import get_model, get_lora_config
from ospo.utils.generate import get_trainer
from ospo.utils.common import build_config
from ospo.dataclass import GenerationDataModule
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.wrapper import JanusProScoreWrapper, JanusProFilterWrapper

os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    
    
def get_dataloader(config, step, mode="base"):
    datamodule = GenerationDataModule(config, step) 
    dataloader = datamodule.gen_dataloader()

    return dataloader 


def get_filtering_wrapper(config, model, vl_chat_processor, image_processor, tokenizer, mode="base"):
    if config.ckpt_path is not None:
        print("# Load model with checkpoint.")
        lora_config = get_lora_config(config.ckpt_path)
        
        model.language_model = get_peft_model(model.language_model, lora_config)
        model = JanusProFilterWrapper.load_from_checkpoint(checkpoint_path=config.ckpt_path, 
                                                        config=config,
                                                        model=model,
                                                        chat_processor=vl_chat_processor,
                                                        image_processor=image_processor,
                                                        tokenizer=tokenizer,
                                                        mode=mode,
                                                        strict=False) 
        model.setup("test")
        model.model.language_model = model.model.language_model.merge_and_unload() 

    else:
        print("# Load base model.")
        model = JanusProFilterWrapper(config=config,
                                        model=model,
                                        chat_processor=vl_chat_processor,
                                        image_processor=image_processor,
                                        tokenizer=tokenizer,
                                        mode=mode)
    return model

# Note: mode is 'negative'.
def get_scoring_wrapper(config, model, vl_chat_processor, image_processor, tokenizer):
    if config.ckpt_path is not None:
        print("# Load model with checkpoint.")
        lora_config = get_lora_config(config.ckpt_path)
        
        model.language_model = get_peft_model(model.language_model, lora_config)
        model = JanusProScoreWrapper.load_from_checkpoint(checkpoint_path=config.ckpt_path, 
                                                        config=config,
                                                        model=model,
                                                        tokenizer=tokenizer,
                                                        processor=vl_chat_processor,
                                                        mode="negative",
                                                        strict=False) 
        model.setup("test")
        model.model.language_model = model.model.language_model.merge_and_unload() 

    else:
        print("# Load base model.")
        model = JanusProScoreWrapper(config=config,
                                        model=model,
                                        tokenizer=tokenizer,
                                        processor=vl_chat_processor,
                                        mode="negative")
    return model


def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    seed_everything(config.seed, workers=True)

    trainer = get_trainer(device, config.world_size)

    vl_chat_processor, tokenizer, model = get_model(mode='generate', config=config)
    image_processor = vl_chat_processor.image_processor 


    # 1. Base
    config.data_path = config.base_filtering_path
    dataloader_b = get_dataloader(config, step=3.1, mode="base")
    model_b = get_filtering_wrapper(config, model, vl_chat_processor, image_processor, tokenizer,
                                    mode="base")
    trainer.test(model_b, dataloaders=dataloader_b)
    print("(Step 3a) Filtering and selection completed.")


    # 2. Negative
    config.data_path = config.negative_scoring_path
    dataloader_s = get_dataloader(config, step=3.2 ,mode="negative") # data merge
    model_s = get_scoring_wrapper(config, model, vl_chat_processor, image_processor, tokenizer)
    trainer.test(model_s, dataloaders=dataloader_s)
    print("(Step 3b) Negative Scoring generation completed.")



    config.data_path = config.negative_filtering_path
    dataloader_f = get_dataloader(config, step=3.1 ,mode="negative") 
    model_f = get_filtering_wrapper(config, model, vl_chat_processor, image_processor, tokenizer,
                                    mode="negative")
    trainer.test(model_f, dataloaders=dataloader_f)
    print("(Step 3c) Negative Filtering and selection completed\n>> WE FINALLY GOT TRAINING DATA FOR OSPO !")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/step3_2.yaml")
    args, unknown = parser.parse_known_args()  
    config = build_config(cfg_path=args.cfg_path)
    
    main(config)
