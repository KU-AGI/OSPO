# Step 3-1. Compositional VQA

import os
import torch
import argparse
from pytorch_lightning import seed_everything
from peft import get_peft_model

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from ospo.utils.model import get_model, get_lora_config
from ospo.utils.generate import get_trainer
from ospo.utils.common import build_config
from ospo.dataclass import GenerationDataModule
from ospo.wrapper import JanusProQuestionGenWrapper, JanusProScoreWrapper

os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    
def get_dataloader(config):
    datamodule = GenerationDataModule(config, step=3.1)
    dataloader = datamodule.gen_dataloader()
    return dataloader 


def get_wrapper(config, model, tokenizer, vl_chat_processor, wrapper_cls, mode="base"):
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
            mode=mode,
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
            processor=vl_chat_processor,
            mode=mode
        )
    return model


def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    seed_everything(config.seed, workers=True)

    trainer = get_trainer(device, config.world_size)
    vl_chat_processor, tokenizer, model = get_model(mode='generate', config=config)


    # 1. Question generation
    # if not os.path.exists(os.path.join(config.save_path, 'vqa_prompt.json')):
    if config.data_path is None:
        raise ValueError("config.data_path is not given.")
    dataloader_qs = get_dataloader(config)
    model_qs = get_wrapper(config, model, tokenizer, vl_chat_processor, 
                                wrapper_cls=JanusProQuestionGenWrapper)
    
    trainer.test(model_qs, dataloaders=dataloader_qs)
    print("(Step 3) Decompositional Question generation completed.")


    # 2. Scoring based on Self-VQA result 
    # reset
    if config.data_path is None:
        config.data_path = os.path.join(config.save_path, 'vqa_prompt.json')
    dataloader_as = get_dataloader(config)
    model_as = get_wrapper(config, model, tokenizer, vl_chat_processor, 
                                wrapper_cls=JanusProScoreWrapper)
    
    trainer.test(model_as, dataloaders=dataloader_as)
    print("(Step 3) Scoring and Train dataset generation completed.")


    # 3. Question for rejected (prompt) generation
    dataloader_rqs = get_dataloader(config)
    model_rqs = get_wrapper(config, model, tokenizer, vl_chat_processor, 
                                wrapper_cls=JanusProQuestionGenWrapper, mode="negative")
    trainer.test(model_rqs, dataloaders=dataloader_rqs)
    print("(Step 3) Rejected prompt; Decompositional Question generation completed.")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/step3_1.yaml")
    args, unknown = parser.parse_known_args()  
    config = build_config(cfg_path=args.cfg_path)
    
    main(config)