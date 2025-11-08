
import os
import yaml
import warnings
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from janus.models import MultiModalityCausalLM, VLChatProcessor


def get_model(mode='generate', model_path=None, cache_dir=None, dtype=torch.float32, **kwargs): 
    if mode not in ['generate', 'train']:
        raise ValueError(f"Invalid mode: {mode}. Choose either 'generate' or 'train'.")
    
    config = kwargs.get('config', None)
    if config is not None:
        if hasattr(config, 'model'):
            model_path = config.model.model_path 
            cache_dir = config.model.cache_dir
        else:
            model_path = config.model_path
            cache_dir = config.cache_dir

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path, cache_dir)
    image_processor = vl_chat_processor.image_processor 
    text_tokenizer = vl_chat_processor.tokenizer

    # use_flash_attn = (
    #     config is not None and 
    #     hasattr(config, "model") and 
    #     getattr(config.model, "flash_attn", False)
    # )

    if hasattr(config, "model") and getattr(config.model, "attn_mode", "flash_attention_2"):
        assert config.model.attn_mode in ["flash_attention_2", "eager", "sdpa"], f"Invalid attention module: {config.model.attn_mode }"
        print(f"Use attention module: {config.model.attn_mode}")

        model_config = AutoConfig.from_pretrained(model_path)
    
    # if use_flash_attn:
        # if config.model.flash_attn:
        if config.model.attn_mode == "flash_attention_2":
            print("Use Flash Attention 2")
            # if mode == "train":
            #     warnings.warn(
            #         "Flash Attention 2 is not recommended in traning mode."
            #     )
            # enable flash attention 2
            model_config.language_config.use_flash_attention_2 = True
            model_config.language_config._attn_implementation = "flash_attention_2"

            vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16, 
                config=model_config
            )

        else:
            model_config.language_config._attn_implementation = config.model.attn_mode

            vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16, 
                config=model_config
            )
    
    else:
        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True,
                cache_dir=cache_dir,
                torch_dtype=dtype, 
                device_map=None,   
            )
    
    if mode == 'generate':
        # vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        vl_gpt = vl_gpt.to(torch.bfloat16).eval()
        return vl_chat_processor, text_tokenizer, vl_gpt

    # mode = 'train'
    else:
        # Enable gradient checkpointing first
        if config.experiment.gradient_checkpointing:
            vl_gpt.language_model.gradient_checkpointing_enable()
            
        if config.use_peft: 
            print("Use Peft for Language Model Only.")
            lora_config=LoraConfig(
                r=config.lora.lora_rank,
                lora_alpha=config.lora.lora_alpha,
                target_modules=config.lora.target_modules,
                lora_dropout=config.lora.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # Apply LoRA to the language model **only**
            vl_gpt.language_model = get_peft_model(vl_gpt.language_model, lora_config)

            # Mark frozen parameters to avoid DDP issues
            for name, param in vl_gpt.named_parameters():
                if "language_model" not in name:
                    param.requires_grad = False  # Ensure non-LoRA layers are frozen

            # Explicitly mark LoRA parameters for DDP
            for param in vl_gpt.language_model.parameters():
                param._ddp_params_and_buffers_to_ignore = True  # Prevent DDP from syncing frozen params

        # if config.model.flash_attn:
        #     print("Use Flash Attention-2.")
        #     vl_gpt.language_model.config._attn_implementation = "flash_attention_2" 

        return vl_gpt, vl_chat_processor, image_processor, text_tokenizer


def get_lora_config(ckpt_path):
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(ckpt_config_path, "r") as file:
        ckpt_config = yaml.safe_load(file)

    # Extract LoRA config
    lora_config = LoraConfig(
        r=ckpt_config["lora"].get("lora_rank"),
        lora_alpha=ckpt_config["lora"]["lora_alpha"],
        target_modules=ckpt_config["lora"]["target_modules"],
        lora_dropout=ckpt_config["lora"]["lora_dropout"],
        modules_to_save=ckpt_config["lora"].get("modules_to_save")                     
    )
 
    return lora_config

