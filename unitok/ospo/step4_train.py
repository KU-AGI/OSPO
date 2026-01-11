# conda activate unitok_dpo

import os
import sys
import json
import numpy as np
import torch
import argparse
from tqdm import tqdm
from typing import *
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
sys.path.append("./eval/liquid") # 임시조치

from unitok.utils.config import Args
from unitok.models.unitok import UniTok
from unitok.eval.liquid.model import *
from unitok.eval.liquid.constants import (
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN
)
from unitok.ospo.dataclass import TrainDataModule
from unitok.ospo.wrapper import UniTokWrapper


def get_model(config):
    dtype = torch.bfloat16 if config.experiment.precision == 'bf16' else torch.float32
    
    print('Loading VQ model ...')
    ckpt = torch.load(config.model.tokenizer_path, map_location='cpu')
    
    vae_cfg = Args()
    vae_cfg.load_state_dict(ckpt['args'])

    vq_model = UniTok(vae_cfg)
    vq_model.load_state_dict(ckpt['trainer']['unitok'])
    
    vq_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, padding_side='right')
    vqllm = AutoModelForCausalLM.from_pretrained(
        config.model.model_path,
        attn_implementation=config.model.attn_mode, # TODO
        # attn_implementation='flash_attention_2',
        torch_dtype=dtype
    )

    if config.use_peft: 
        # Enable gradient checkpointing first
        if config.experiment.gradient_checkpointing:
            vqllm.gradient_checkpointing_enable()

        print("*** Use Peft for Language Model Only ***")
        lora_config = LoraConfig(
            r=config.lora.lora_rank,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=config.lora.modules_to_save,
        )

        # Apply LoRA to the language model **only**
        vqllm = get_peft_model(vqllm, lora_config)

    return vqllm, tokenizer, vq_model


def get_trainer(config, device):
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=config.base.save_path, name=config.base.exp_name)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=tb_logger.log_dir,
        filename="{step:06d}",
        save_top_k=-1, # save all ckpt corresponding to saving interval              
        every_n_train_steps=config.experiment.save_steps, 
        # save_last=True
    )
    if config.use_peft:
        trainer = pl.Trainer(
        devices=config.base.world_size,
        accelerator=device,
        logger=tb_logger,
        default_root_dir=config.base.save_path,
        callbacks=[pl.callbacks.ModelSummary(max_depth=2), checkpoint_callback], 
        strategy=DDPStrategy(                                          
            find_unused_parameters=False # LoRA 이슈로 수정
        ),          
        log_every_n_steps=config.experiment.log_steps,
        gradient_clip_val=config.experiment.gradient_clip_val, 
        enable_checkpointing=config.experiment.enable_checkpointing,
        accumulate_grad_batches=config.experiment.gradient_accumulation_steps,
        precision="bf16" if config.experiment.precision is None or config.experiment.precision == "auto" else config.experiment.precision, #config.precision, 
        max_steps=config.experiment.max_training_steps, # or max_epochs   
        check_val_every_n_epoch=None,
        val_check_interval=config.experiment.val_steps * config.experiment.gradient_accumulation_steps, 
        # num_sanity_val_steps = 0,           
    )

    elif config.use_ds:
        trainer = pl.Trainer(
            devices=config.base.world_size,
            accelerator=device,
            logger=tb_logger,
            default_root_dir=config.base.save_path,
            callbacks=[pl.callbacks.ModelSummary(max_depth=2), checkpoint_callback], 
            # strategy=DDPStrategy(                                          
            #     find_unused_parameters=False 
            # ),   
            strategy=DeepSpeedStrategy(
                    # zero_optimization=config.experiment.deepspeed.zero_optimization, # 추가
                    stage=config.experiment.deepspeed.stage,                                                  
                    allgather_bucket_size=config.experiment.deepspeed.allgather_bucket_size,
                    reduce_bucket_size=config.experiment.deepspeed.reduce_bucket_size,
                    offload_optimizer=config.experiment.deepspeed.offload_optimizer, 
                    offload_parameters=config.experiment.deepspeed.offload_parameters,
                    pin_memory=config.experiment.deepspeed.pin_memory,
                    contiguous_gradients=config.experiment.deepspeed.contiguous_gradients,
                    overlap_comm=config.experiment.deepspeed.overlap_comm,
                    # reduce_scatter=config.experiment.deepspeed.reduce_scatter,
                    # allgather_partitions=config.experiment.deepspeed.allgather_partitions,
                ),         
            log_every_n_steps=config.experiment.log_steps,
            gradient_clip_val=config.experiment.gradient_clip_val, 
            enable_checkpointing=config.experiment.enable_checkpointing,
            accumulate_grad_batches=config.experiment.gradient_accumulation_steps,
            precision="bf16" if config.experiment.precision is None or config.experiment.precision == "auto" else config.experiment.precision, #config.precision, 
            max_steps=config.experiment.max_training_steps, # or max_epochs   
            check_val_every_n_epoch=None,
            val_check_interval=config.experiment.val_steps * config.experiment.gradient_accumulation_steps, 
            # num_sanity_val_steps = 0,           
        )


    else:
        trainer = pl.Trainer(
        devices=config.base.world_size,
        accelerator=device,
        logger=tb_logger,
        default_root_dir=config.base.save_path,
        callbacks=[pl.callbacks.ModelSummary(max_depth=2), checkpoint_callback], 
        strategy=DDPStrategy(                                          
            find_unused_parameters=True # allow unused
        ),          
        log_every_n_steps=config.experiment.log_steps,
        gradient_clip_val=config.experiment.gradient_clip_val, 
        enable_checkpointing=config.experiment.enable_checkpointing,
        accumulate_grad_batches=config.experiment.gradient_accumulation_steps,
        precision="bf16" if config.experiment.precision is None or config.experiment.precision == "auto" else config.experiment.precision, #config.precision, 
        max_steps=config.experiment.max_training_steps, # or max_epochs   
        check_val_every_n_epoch=None, # no validation
        val_check_interval=config.experiment.val_steps * config.experiment.gradient_accumulation_steps, 
        # num_sanity_val_steps = 0,           
        )

    return trainer


def get_dataloader(config, tokenizer):    
    datamodule = TrainDataModule(config, tokenizer) 
    datamodule.setup()
    
    train_dataloader = datamodule.train_dataloader()
    # val_dataloader = datamodule.val_dataloader() 

    # return train_dataloader, val_dataloader
    return train_dataloader


def load_config(config_name: str):    
    overrides = [arg for arg in sys.argv[1:] if "=" in arg]
    
    initialize(config_path="../configs", version_base=None)
    # config = compose(config_name="simpo", overrides=overrides)
    config = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(config)

    return config


def main(config: DictConfig):
    if config.base.save_path is not None:
        os.makedirs(config.base.save_path, exist_ok=True)
    
    pl.seed_everything(config.experiment.seed, workers=True) 
    
    model, text_tokenizer, img_tokenizer = get_model(config)
    train_dataloader = get_dataloader(config, text_tokenizer) 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = get_trainer(config, device)
    
    wrapper = UniTokWrapper(config, 
                            model=model, 
                            text_tokenizer=text_tokenizer, 
                            img_tokenizer=img_tokenizer)         

    print("Start Training")
    trainer.fit(wrapper, train_dataloaders=train_dataloader)
    

if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)

    main(config=load_config(config_name="step4"))