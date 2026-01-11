# /home/rla020/research/mllm/UniTok/mllm_train/inference/geneval.py

import os
import json
import yaml

import torch
from torch import nn
import numpy as np
import argparse
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import transformers
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from PIL import Image
import sys
import traceback
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
# sys.path.append("./eval/liquid") # 임시조치

from unitok.utils.config import Args
from unitok.models.unitok import UniTok
from unitok.eval.liquid.model import *
from unitok.eval.liquid.constants import (
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN
)

from unitok.ospo.dataclass import GenEval


class UniTokTestWrapper(pl.LightningModule):
    def __init__(self, config, model, text_tokenizer, img_tokenizer):
        super().__init__()
        self.config=config
        self.model=model
        self.text_tokenizer=text_tokenizer
        self.img_tokenizer=img_tokenizer

        self.num_codebooks = 8 # 고정
        self.seed_list = config.task.seed # SEED 를 다르게 설정할 경우, 각기 다른 SEED 에 대해 생성
        self.error_data = []
        self.pil_transform = transforms.ToPILImage()

        self.eoi = torch.tensor([4])
        self.boi = torch.tensor([3])
        self.eos = torch.tensor([2])
        self.bos = torch.tensor([1])

        for k, v in self.config.generation_config.items():
            setattr(self, k, v) 


    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        
        prompt_list = []
        data_idx_list = []
        sample_path_list = []

        for sample, data_idx in batch:
            prompt = self.get_prompt(self.text_tokenizer, sample['prompt'])

            outpath = os.path.join(self.config.base.save_path, f"{data_idx:05d}")
            os.makedirs(outpath, exist_ok=True)

            sample_path = os.path.join(outpath, "samples")
            os.makedirs(sample_path, exist_ok=True)

            # 이미 생성완료된 파일인지 확인
            if os.path.exists(sample_path) and len(os.listdir(sample_path)) == 4:
                continue
        
            prompt_list.append(prompt)
            data_idx_list.append(data_idx)
            sample_path_list.append(sample_path)

            with open(os.path.join(outpath, "metadata.jsonl"), "w") as f:
                json.dump(sample, f)

        if len(prompt_list) == 0:
            return # 생성할 필요가 없음

        # 개별 이미지 생성
        for idx in range(4): # hard coding
            self.generate_batch(
                prompt_list=prompt_list,
                sample_path_list=sample_path_list,
                seed=idx # seed 역할
            )

    def generate_batch(self, 
                        prompt_list, 
                        sample_path_list, 
                        seed,
                        ):
        seed_everything(seed)
        sampling_kwargs = {'temperature': self.temperature, 'top_k': self.top_k, 'top_p': self.top_p, 'sample_logits': self.sample_logits} # sample_logtis: BOOL

        uncondition_text_inputs = ['<unconditional>\x00'] * len(prompt_list)

        if self.cfg_scale > 1:
            model_inputs = self.text_tokenizer(prompt_list + uncondition_text_inputs, return_tensors="pt", padding=True).to(self.device)
        else:
            model_inputs = self.text_tokenizer(prompt_list, return_tensors="pt", padding=True).to(self.device)
        
        model_kwargs = {'attention_mask': model_inputs.pop('attention_mask'), 'use_cache': True}
        input_ids = model_inputs.pop('input_ids')
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        save_list = []
        with torch.no_grad():
            pred_tokens = []
            input_multi_ids = None
            for _ in range(256):
                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                outputs = self.model.T2I_forward_withcache(
                    **model_inputs,
                    input_multi_ids=input_multi_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

                next_embed = outputs['last_hidden_state'][:, -1:, :]

                indices_arhead = []

                for i_head in range(self.num_codebooks): # 처음에 (bs*2, 1, 4096) -> (bs*2, 2, 4096) -> ... 이렇게 들어감
                # 첫 next_embed = input의 last hidden state
                    ar_next_embed = self.model.ar_head(
                        inputs_embeds=next_embed,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=False,
                    )
                    
                    next_token_logits = self.model.ar_head.linear_head(ar_next_embed[0]) # sub vocab size인 4096개가 나옴
                    if self.cfg_scale > 1:
                        cond_logits, uncond_logits = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0)
                        cfg_logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg_scale
                        half_next_token, _ = self.sample(cfg_logits, **sampling_kwargs)
                        # pred_tokens.append(half_next_token)
                        next_token = torch.cat([half_next_token, half_next_token])  # [bz,1]
                    else:
                        next_token, next_prob = self.sample(next_token_logits, **sampling_kwargs)
                        # pred_tokens.append(next_token)
                    indices_arhead.append(next_token)
                    if i_head < self.num_codebooks - 1:
                        predicted_embed = self.model.ar_head.codebooks[i_head](next_token) # sub codebook에서 ids에 맞는 Embed 가져옴
                        next_embed = torch.cat([next_embed, predicted_embed], dim=1)

                pred_tokens.append(torch.cat(indices_arhead, dim=1))  # [numcodebook,bz*2]
                input_multi_ids = torch.stack(pred_tokens, dim=-1)
                fake_id = torch.zeros_like(input_ids[:, :1])
                input_ids = torch.cat([input_ids, fake_id], dim=-1)  # add fake id for cache

                model_kwargs = self.model._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.model.config.is_encoder_decoder,
                )

        del sampling_kwargs
        del model_inputs
        del outputs
        del model_kwargs

        ori_batchsize = len(prompt_list)
        image_vq_id = torch.stack(pred_tokens, dim=-1)[:ori_batchsize]
        save_list.append(image_vq_id)
        torch.cuda.empty_cache()
        # print('decoding images ...')

        # 저장
        for idx, vq_code in enumerate(save_list[0]):
            new_gen_ids = vq_code.unsqueeze(0).to('cuda')
            rec_image = self.img_tokenizer.idx_to_img(new_gen_ids)
            rec_img = self.pil_transform(rec_image.squeeze(0).to(torch.float32).add(1).mul_(0.5).clamp_(0, 1))
        
            # rec_img.save(save_path_list[idx])
            fpath = os.path.join(sample_path_list[idx],f"{seed:05}.png")
            rec_img.save(fpath)

    # def on_test_epoch_end(self):
    #     print(f"Error case: {len(self.error_data)}")
    #     save_path = os.path.join(self.config.base.save_path, "error_sample.json")
    #     with open(save_path, "w") as f:
    #         json.dump(save_path, f, indent=4)  

    def get_prompt(self, tokenizer, caption):
        return caption + ' Generate an image based on this description.\x00'

    def sample(self, logits, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, sample_logits=True):
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k > 0 or top_p < 1.0:
            logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        if sample_logits:
            idx = torch.multinomial(probs, num_samples=1)
        else:
            _, idx = torch.topk(probs, k=1, dim=-1)
        return idx, probs

    def top_k_top_p_filtering(
        self,
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
        ):
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """

        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k

            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        # import pdb;pdb.set_trace()
        return logits


def get_model(config):
    dtype = torch.bfloat16 if config.base.precision == 'bf16' else torch.float32
    
    print('Loading VQ model ...')
    ckpt = torch.load(config.model.tokenizer_path, map_location='cpu')
    
    vae_cfg = Args()
    vae_cfg.load_state_dict(ckpt['args'])

    vq_model = UniTok(vae_cfg)
    vq_model.load_state_dict(ckpt['trainer']['unitok'])
    
    vq_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path, padding_side='left')
    vqllm = AutoModelForCausalLM.from_pretrained(
        config.model.model_path,
        attn_implementation=config.model.attn_mode,
        # attn_implementation='flash_attention_2',
        torch_dtype=dtype
    )

    if config.model.type == "official":
        return vqllm, tokenizer, vq_model
    else:
        print("Loading LoRA model ...")
        if not os.path.exists(config.model.ckpt_path):
            raise ValueError("Check model.ckpt_path !")
        
        ckpt_dir = os.path.dirname(config.model.ckpt_path)
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

        # Apply LoRA to the language model **only**
        vqllm = get_peft_model(vqllm, lora_config)

    return vqllm, tokenizer, vq_model


def get_trainer(config, device):
    trainer = pl.Trainer(
        accelerator=device,
        devices=config.base.world_size,
        strategy="ddp",
        max_epochs=1, # config.experiment.epoch,
        precision=config.base.precision,
        callbacks=[pl.callbacks.ModelSummary(max_depth=2)],
        #profiler=profiler,
    )
    return trainer

def collate_fn(batch):
    return batch

def get_dataloader(config):
    dataset = GenEval(data_path=config.task.data_path)
    dataloader = DataLoader(dataset,
                            collate_fn=collate_fn, 
                            batch_size=config.task.batch_size,
                            num_workers=config.task.num_workers,
                            pin_memory=True,
                            drop_last=False) # ddp
    return dataloader

def load_config():
    raw_overrides = [arg for arg in sys.argv[1:] if "=" in arg]
    overrides = []
    for arg in raw_overrides:
        # “=” 이 2개 이상이면 key, rest 로 분리한 뒤 rest 를 따옴표로 감싸준다.
        if arg.count("=") > 1:
            key, rest = arg.split("=", 1)
            overrides.append(f'{key}="{rest}"')
        else:
            overrides.append(arg)
    
    initialize(config_path="../../configs/eval", version_base=None)
    config = compose(config_name="geneval", overrides=overrides)
    OmegaConf.resolve(config)

    return config

def main(config: DictConfig):
    config.base.save_path = os.path.join(config.base.save_path, config.base.exp_name, config.base.task_name, "gen")
    if config.base.save_path is not None:
        os.makedirs(config.base.save_path, exist_ok=True)
    
    pl.seed_everything(config.task.seed[0], workers=True) 
    
    model, text_tokenizer, img_tokenizer = get_model(config)
    eval_dataloader = get_dataloader(config) 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = get_trainer(config, device)
    
    if config.model.type == "official":
        wrapper = UniTokTestWrapper(config, 
                             model=model, 
                             text_tokenizer=text_tokenizer, 
                             img_tokenizer=img_tokenizer) 

    elif config.model.type == "pl":
        print(f"Load ckpt from {config.model.ckpt_path}")
        wrapper = UniTokTestWrapper.load_from_checkpoint(
            checkpoint_path=config.model.ckpt_path,
            config=config, 
            model=model, 
            text_tokenizer=text_tokenizer, 
            img_tokenizer=img_tokenizer,
            strict=False) 

        # model.setup("test")
        wrapper.model = wrapper.model.merge_and_unload()

    else:
        raise ValueError("Check model.type !")
    trainer.test(wrapper, dataloaders=eval_dataloader)
    

if __name__ == "__main__":
    main(config=load_config())