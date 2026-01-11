# /home/rla020/research/mllm/UniTok/mllm_train/simpo/simpo.py

import os
import sys
import json
import numpy as np
import argparse
from tqdm import tqdm
from typing import *
from collections import defaultdict
from PIL import Image

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist

import pytorch_lightning as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
import transformers
from trl.trainer.utils import pad_to_length 
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

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
from unitok.ospo.dataclass import TrainDataModule
from unitok.ospo.utils.lr_scheduler import CosineDecayWarmUpRestarts

class UniTokWrapper(pl.LightningModule):
    def __init__(self, config, model, text_tokenizer, img_tokenizer):
        super().__init__()
        self.config=config
        self.model=model
        self.text_tokenizer=text_tokenizer
        self.img_tokenizer=img_tokenizer

        self.eoi = torch.tensor([4])
        self.boi = torch.tensor([3])
        self.eos = torch.tensor([2])
        self.bos = torch.tensor([1])

        # default SimPO hyperparameter 
        simpo_config = self.config['algo']

        self.loss_type = simpo_config.get('loss_type', 'sigmoid')
        self.beta_1 = simpo_config.get('beta_1', 1.0)
        self.beta_2 = simpo_config.get('beta_2', 1.0)

        self.gamma_beta_ratio_1 = simpo_config.get('gamma_beta_ratio_1', 0.0)
        self.gamma_beta_ratio_2 = simpo_config.get('gamma_beta_ratio_2', 0.0)

        self.label_smoothing = simpo_config.get('label_smoothing', 0.0)

        # Loss weight
        self.simpo_weight = simpo_config.get('simpo_weight', 1.0) 
        self.sft_weight = simpo_config.get('sft_weight', 0.0)
        self.copo_weight = simpo_config.get('copo_weight', 0.0)

        # Clipping weight
        self.simpo_clipping_weight = simpo_config.get('simpo_clipping_weight', 0.0)
        self.copo_clipping_weight = simpo_config.get('copo_clipping_weight', 0.0)  

        self.copo_mode = False     
        self.log_by_category = self.config.experiment.log_by_category

        # Sanity Check
        if self.sft_weight > 0.0:
            print("Notice: SFT weight is set to a non-zero value.")
        if self.copo_weight > 0.0:
            self.copo_mode = True # boolean indicator
            print("Notice: copo weight is set to a non-zero value.")
        if self.simpo_clipping_weight > 0.0:
            print("Notice: SimPO Clipping weight is set to a non-zero value.")
        if self.copo_clipping_weight > 0.0:
            print("Notice: copo Clipping weight is set to a non-zero value.")

        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = self.text_tokenizer.pad_token_id 


    def save_config(self, save_path, config):
        if type(config) == DictConfig:
            config = OmegaConf.to_container(config, resolve=True)
        
        config_save_path = os.path.join(save_path, 'config.yaml') 
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            
        with open(config_save_path, "w") as f:
            json.dump(config, f, indent=4)
        
        print("Saving config.")


    def setup(self, stage: str=None):
        self.save_config(self.logger.log_dir, self.config)
        print("*** Saving config done. ***")

        self.model.train()        
        self.model.config.output_hidden_states = True
        self.model.config.tokenizer_padding_side = self.text_tokenizer.padding_side
        self.model.config.tokenizer_model_max_length = self.text_tokenizer.model_max_length
        self.model.config.mm_use_im_start_end = False
        self.model.config.mm_projector_lr = None
        self.model.config.mm_use_im_patch_token = True

        # Convert all parameters explicitly to FP32
        if self.config.experiment.precision == 32:
            self.model = self.model.float()

        # Freeze untrainable param
        self.freeze_param()
        self.print_trainable_parameters()

        for name, param in self.model.named_parameters():
            if param.dtype == torch.long:
                print(f"Parameter {name} is of type torch.long. Converting to float32.")
                param.data = param.data.float()

        # TODO: AR Head: requires_grad ?

    def freeze_param(self):
        # liquid unfreeze except codebook
        freeze = ['codebook', 'lm_head', 'multi_embedder', 'embed_tokens', 'ar_head'] 
        # text embedding, lm head, image embedding > NOT Learnable (linear layer 도 학습 안되는중)
        
        for name, param in self.model.named_parameters():
            if any(f in name for f in freeze):
                param.requires_grad = False

        # Unitok Freeze
        for name, param in self.img_tokenizer.named_parameters():
            param.requires_grad = False

    def print_trainable_parameters(self):
        print("Trainable Parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}, dtype={param.dtype} (trainable)")

        for name, param in self.img_tokenizer.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}, dtype={param.dtype} (trainable)")

    def on_train_epoch_start(self):
        device = self.device

        # set train mode
        self.model.train()
        self.img_tokenizer.eval()

        # Explicitly move each model to device
        self.model.to(device)
        self.img_tokenizer.to(device)

        # Sanity check: Print confirmation once per epoch (on main process only)
        if self.trainer.is_global_zero:
            print(f"*** All modules explicitly moved to device: {device} ***")


    def training_step(self, batch, batch_idx):
        preprocessed = self.preprocess_train_batch(batch)
        loss = self.compute_loss(inputs=preprocessed) 
        self.log_dict({'train/loss': loss,
                        'train/lr': self.trainer.optimizers[0].param_groups[0]['lr'],
                        'train/global_step': self.global_step}, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        return loss.float()


    def compute_loss(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        with torch.cuda.amp.autocast():
            loss = self.get_batch_loss_metrics(inputs, train_eval="train")

        return loss


    def get_batch_loss_metrics(
        self,
        batch: Dict[str, Union[torch.Tensor, Any]],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        item_ids = batch["item_ids"]

        if train_eval == "train": 
            do_logging = True
        else:
            task_ids = batch["task_ids"]
            if all(task_id == 1 for task_id in task_ids):
                do_logging = True
            else:
                do_logging = False

        prefix = "val" if train_eval == "val" else "train"
        outputs = self.concatenated_forward(batch)

        if self.copo_mode:
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_swap_chosen_logps, 
                policy_chosen_logits,
                policy_rejected_logits,
                policy_swap_chosen_logits,
                chosen_labels,
            ) = outputs
            loss_kwargs = {
                "policy_chosen_logps": policy_chosen_logps,
                "policy_rejected_logps": policy_rejected_logps,
                "policy_swap_chosen_logps": policy_swap_chosen_logps
            }

        else:
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
                chosen_labels,
            ) = outputs
            loss_kwargs = {
                "policy_chosen_logps": policy_chosen_logps,
                "policy_rejected_logps": policy_rejected_logps,
            }

        loss_outputs = self.simpo_loss(**loss_kwargs)

        if self.sft_weight > 0.0:
            policy_chosen_logits = policy_chosen_logits.contiguous()
            chosen_labels = chosen_labels.clone()

            loss_func = torch.nn.CrossEntropyLoss()
            sft_loss = loss_func(policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1]), chosen_labels.view(-1))

            loss = (self.simpo_weight * loss_outputs['simpo_loss'] + self.copo_weight * loss_outputs['copo_loss'] + self.sft_weight * sft_loss).mean()
            if do_logging:
                self.log(f"{prefix}/sft_loss", sft_loss.detach().cpu(), on_step=True, prog_bar=True, logger=True, sync_dist=True)
                # self.log(f"{prefix}/simpo_loss", simpo_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)            
        else:
            loss = (self.simpo_weight * loss_outputs['simpo_loss'] + self.copo_weight * loss_outputs['copo_loss']).mean()

        chosen_rewards = loss_outputs['chosen_rewards']
        rejected_rewards = loss_outputs['rejected_rewards']
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        reward_margins = (chosen_rewards - rejected_rewards)

        if self.copo_mode:
            swap_rewards = loss_outputs['swap_chosen_rewards']
            swap_reward_accuracies = (chosen_rewards > swap_rewards).float()
            swap_reward_margins = (chosen_rewards - swap_rewards)

        if do_logging:
            content = {
                f"{prefix}/rewards/chosen": chosen_rewards.mean().cpu(),
                f"{prefix}/rewards/rejected": rejected_rewards.mean().cpu(),
                f"{prefix}/rewards/accuracies": reward_accuracies.mean().cpu(),
                f"{prefix}/rewards/margins": reward_margins.mean().cpu(),
                f"{prefix}/logps/rejected": policy_rejected_logps.detach().mean().cpu(),
                f"{prefix}/logps/chosen": policy_chosen_logps.detach().mean().cpu(),
                f"{prefix}/logits/rejected": policy_rejected_logits.detach().mean().cpu(),
                f"{prefix}/logits/chosen": policy_chosen_logits.detach().mean().cpu(),
                f"{prefix}/simpo_loss": loss_outputs['simpo_loss'].detach().cpu().mean(),
            }
            if self.copo_mode:
                content.update({
                    f"{prefix}/rewards/swap_chosen": swap_rewards.mean().cpu(),
                    f"{prefix}/rewards/swap_accuracies": swap_reward_accuracies.mean().cpu(),
                    f"{prefix}/rewards/swap_margins": swap_reward_margins.mean().cpu(),
                    f"{prefix}/logps/swap_chosen": policy_swap_chosen_logps.detach().mean().cpu(),
                    f"{prefix}/logits/swap_chosen": policy_swap_chosen_logits.detach().mean().cpu(),
                    f"{prefix}/copo_loss": loss_outputs['copo_loss'].detach().cpu().mean(),
                })

            # DO LOG
            self.log_dict(content, on_step=True, prog_bar=True, logger=True, sync_dist=True)

            if self.log_by_category:
                idx2category = {"0": "attribute", "1": "layout", "2": "non-spatial", "3": "complex"}
                if self.copo_mode: 
                    for item_id, chosen_logps, rejected_logps, swap_chosen_logps, chosen_reward, rejected_reward, swap_reward in zip(
                                                                                                    item_ids, 
                                                                                                    policy_chosen_logps, 
                                                                                                    policy_rejected_logps, 
                                                                                                    policy_swap_chosen_logps,
                                                                                                    chosen_rewards, 
                                                                                                    rejected_rewards,
                                                                                                    swap_rewards
                                                                                                    ):
                        category = idx2category.get(item_id[0], None)
                        if category is not None:
                            self.log_dict({
                                # train_attribute
                                f"{prefix}_{category}/chosen_logps": chosen_logps.detach().mean().cpu(),
                                f"{prefix}_{category}/rejected_logps": rejected_logps.detach().mean().cpu(),
                                f"{prefix}_{category}/swap_chosen_logps": swap_chosen_logps.detach().mean().cpu(),
                                f"{prefix}_{category}/chosen_rewards": chosen_reward.detach().mean().cpu(),
                                f"{prefix}_{category}/rejected_rewards": rejected_reward.detach().mean().cpu(),
                                f"{prefix}_{category}/swap_chosen_rewards": swap_reward.detach().mean().cpu(),
                                f"{prefix}_{category}/reward_accuracies": (chosen_reward > rejected_reward).float().mean().cpu(),
                                f"{prefix}_{category}/reward_margins": (chosen_reward - rejected_reward).mean().cpu(),
                                f"{prefix}_{category}/swap_chosen_reward_accuracies": (chosen_reward > swap_reward).float().mean().cpu(),
                                f"{prefix}_{category}/swap_chosen_reward_margins": (chosen_reward - swap_reward).mean().cpu(),
                            })
                else:
                    for item_id, chosen_logps, rejected_logps, chosen_reward, rejected_reward in zip(
                                                                                                    item_ids, 
                                                                                                    policy_chosen_logps, 
                                                                                                    policy_rejected_logps, 
                                                                                                    chosen_rewards, 
                                                                                                    rejected_rewards):
                        category = idx2category.get(item_id[0], None)
                        if category is not None:
                            self.log_dict({
                                # train_attribute
                                f"{prefix}_{category}/chosen_logps": chosen_logps.detach().mean().cpu(),
                                f"{prefix}_{category}/rejected_logps": rejected_logps.detach().mean().cpu(),
                                f"{prefix}_{category}/chosen_rewards": chosen_reward.detach().mean().cpu(),
                                f"{prefix}_{category}/rejected_rewards": rejected_reward.detach().mean().cpu(),
                                f"{prefix}_{category}/reward_accuracies": (chosen_reward > rejected_reward).float().mean().cpu(),
                                f"{prefix}_{category}/reward_margins": (chosen_reward - rejected_reward).mean().cpu(),
                            })
            

        # Log Probability Gap 체크 만을 위한 Validation Task (task_id = 2,3,4,5)
        else: 
            if len(set(task_ids)) == 1: # all same task_id
                task_id = task_ids[0]
                self.log(f"val/task_{task_id}/logps_gap", (policy_chosen_logps - policy_rejected_logps).mean().cpu(), on_step=True, prog_bar=True, logger=True, sync_dist=True)

            # 마지막으로 모든 logps_gap 을 저장
            for task_id, item_id, chosen_logps, rejected_logps in zip(task_ids, item_ids, policy_chosen_logps, policy_rejected_logps):
                content = {
                        "item_id": item_id,
                        "chosen": chosen_logps.detach().cpu().numpy().tolist(),
                        "rejected": rejected_logps.detach().cpu().numpy().tolist(),
                        "logps_gap": (chosen_logps - rejected_logps).detach().cpu().numpy().tolist()
                    }
                if task_id == 2:
                    self.val_output_logps_gap_hn_img_id.append(content)
                elif task_id == 3:
                    self.val_output_logps_gap_hn_img_ood.append(content)
                elif task_id == 4:
                    self.val_output_logps_gap_hn_txt_id.append(content)
                elif task_id == 5:
                    self.val_output_logps_gap_hn_txt_ood.append(content)
        
        if train_eval == "val":
            print(f"END Rank: {self.global_rank}, Batch: {len(batch)}, task_ids: {batch['task_ids']}")

        return loss


    def simpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_swap_chosen_logps: torch.FloatTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
        # SimPO Loss 와 copo Loss 모두 동일한 beta 값을 사용한다고 가정 (TODO: beta_simpo, beta_copo)

        gamma_1 = self.gamma_beta_ratio_1 * self.beta_1
        gamma_2 = self.gamma_beta_ratio_2 * self.beta_2
        
        chosen_rewards = self.beta_1 * policy_chosen_logps.to(self.device).detach()
        rejected_rewards = self.beta_1 * policy_rejected_logps.to(self.device).detach()

        if self.simpo_clipping_weight > 0.0:
            simpo_clipped_reward_margin = self.simpo_clipping_weight * torch.abs(chosen_rewards)
            logits = torch.minimum(chosen_rewards - rejected_rewards, simpo_clipped_reward_margin) - gamma_1 
            # rewards 값 자체는 음수이므로 minimum 으로 처리
            logits = logits.to(self.device)

            loss_1 = (
                    -F.logsigmoid(logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-1 * logits) * self.label_smoothing
                )
        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            pi_logratios = pi_logratios.to(self.device)
            logits = pi_logratios - self.gamma_beta_ratio_1

            loss_1 = (
                -F.logsigmoid(self.beta_1 * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta_1 * logits) * self.label_smoothing
            )

        if self.copo_mode:

            swap_chosen_rewards = self.beta_2 * policy_swap_chosen_logps.to(self.device).detach()

            if self.copo_clipping_weight > 0.0:
                copo_clipped_reward_margin = self.copo_clipping_weight * torch.abs(chosen_rewards)
                conditional_logits = torch.minimum(chosen_rewards - swap_chosen_rewards, copo_clipped_reward_margin) - gamma_2
                conditional_logits = conditional_logits.to(self.device)

                loss_2 = (
                    -F.logsigmoid(conditional_logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-1 * conditional_logits) * self.label_smoothing
                )

            else:
                conditional_pi_logratios = policy_chosen_logps - policy_swap_chosen_logps
                conditional_pi_logratios = conditional_pi_logratios.to(self.device)
                conditional_logits = conditional_pi_logratios - self.gamma_beta_ratio_2 
                
                loss_2 = (
                    -F.logsigmoid(self.beta_2 * conditional_logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta_2 * conditional_logits) * self.label_smoothing
                )

            return {
                    "chosen_rewards": chosen_rewards,
                    "rejected_rewards": rejected_rewards,
                    "swap_chosen_rewards": swap_chosen_rewards,
                    "simpo_loss": loss_1,
                    "copo_loss": loss_2
                    }

        else:
            return {
                    "chosen_rewards": chosen_rewards,
                    "rejected_rewards": rejected_rewards,
                    "simpo_loss": loss_1,
                    "copo_loss": torch.tensor(0.0).to(self.device).detach()
                    }
        
    # IMPORTANT
    def concatenated_forward(
            self, batch: Dict[str, Union[List, torch.LongTensor]]
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        concatenated_batch = self.concatenated_inputs(batch=batch)
        len_chosen = batch["chosen_labels"].shape[0]

        outputs = self.model.get_model()(
            input_ids = None,
            inputs_embeds=concatenated_batch["concatenated_inputs_embeds"], # chosen t + chosen i / chosen t + rejected i
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            position_ids=concatenated_batch["concatenated_position_ids"],
            use_cache=False, 
            past_key_values=None,
            return_dict=True,
        ) 
        
        hidden_states = outputs.last_hidden_state # output hidden 이것을 기반으로 8 * 256 을 예측해야한다.

        to_image_states = hidden_states # 이미지 부분의 hidden state -1

        additional_image_indexs = concatenated_batch["concatenated_additional_image_indexs"]
        additional_image_labels = concatenated_batch["concatenated_additional_image_labels"]

        shift_image_states = torch.stack([state[start_id - 1:end_id - 1] for (start_id, end_id), state in
                                              zip(additional_image_indexs, to_image_states)])  # Shift so that tokens < n predict n  [bz, seq_len, hidden_dim]
        base_tokens = shift_image_states

        K = self.model.ar_head.num_codebooks
        B, L, C = base_tokens.shape
        base_tokens = base_tokens.reshape(B * L, 1, C)

        targets = torch.cat(additional_image_labels, dim=0)  # [B, K, L] # (1, 8, 256)
        image_code_labels = targets
        targets = targets.permute(0, 2, 1).reshape(B * L, K)[:, :-1] # input token 의 마지막 token 을 제외한 7개 토큰 = input (1, 7, 256)

        index_embeddings = []
        for i in range(K - 1):
            index_embed = self.model.ar_head.codebooks[i](targets[:, i])
            index_embeddings.append(index_embed)
        index_embeddings = torch.stack(index_embeddings, dim=1)
        h = torch.cat((base_tokens, index_embeddings), dim=1)  # [B*L, K, C] // (256 * B, 8, 4096) // base token + 첫 7개의 토큰 -> 이게 input

        multicode_embedding = self.model.ar_head(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=h,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            cache_position=None,
        ) # 코드북 8개 예측한것
        image_logits = self.model.ar_head.linear_head(multicode_embedding[0]) # 여기는 output toekn 1 ~ 8까지 # (B*L,K,C) (16 * 256, 8, 4096)
        
        flatten_image_code_labels = image_code_labels.permute(0, 2, 1).reshape(B, L * K) # (16, 256 * 8)
        image_logits = image_logits.reshape(B, L, K, -1).reshape(B, L * K, -1)

        all_logps = self.get_batch_logps(
            image_logits,
            flatten_image_code_labels,
            average_log_prob=True,
        )
        
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = image_logits[:len_chosen]
        rejected_logits = image_logits[len_chosen:]

        chosen_labels = flatten_image_code_labels[:len_chosen]

        # 미사용
        if self.copo_mode:
            """ 2. copo """
            # TODO: copo 식으로 수정
            swap_chosen_outputs = self.model.get_model()(
                input_ids = None,
                inputs_embeds=batch["swap_chosen_inputs_embeds"], 
                attention_mask=batch["swap_chosen_attention_mask"],
                position_ids=batch["swap_chosen_position_ids"],
                use_cache=False, 
                past_key_values=None,
                return_dict=True,
            ) 
            swap_chosen_hidden_states = swap_chosen_outputs.last_hidden_state

            to_image_states = swap_chosen_hidden_states

            additional_image_indexs = batch["swap_chosen_additional_image_indexs"]
            additional_image_labels = batch["swap_chosen_additional_image_labels"]
            
            shift_image_states = torch.stack([state[start_id - 1:end_id - 1] for (start_id, end_id), state in
                                              zip(additional_image_indexs, to_image_states)])  # Shift so that tokens < n predict n  [bz, seq_len, hidden_dim]
            base_tokens = shift_image_states

            K = self.model.ar_head.num_codebooks
            B, L, C = base_tokens.shape
            base_tokens = base_tokens.reshape(B * L, 1, C)

            targets = torch.cat(additional_image_labels, dim=0)  # [B, K, L]
            image_code_labels = targets
            targets = targets.permute(0, 2, 1).reshape(B * L, K)[:, :-1] # input token의 마지막 token을 제외한 7개 토큰 = input

            index_embeddings = []
            for i in range(K - 1):
                index_embed = self.model.ar_head.codebooks[i](targets[:, i])
                index_embeddings.append(index_embed)
            index_embeddings = torch.stack(index_embeddings, dim=1)
            h = torch.cat((base_tokens, index_embeddings), dim=1)  # [B*L, K, C] // (256 * B, 8, 4096) // base token + 첫 7개의 토큰 -> 이게 input

            multicode_embedding = self.model.ar_head(
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=h,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
                cache_position=None,
            )
            image_logits = self.model.ar_head.linear_head(multicode_embedding[0]) # 여기는 output toekn 1 ~ 8까지
            
            flatten_image_code_labels = image_code_labels.permute(0, 2, 1).reshape(B, L * K) # (16, 256 * 8)
            swap_chosen_logits = image_logits.reshape(B, L, K, -1).reshape(B, L * K, -1)

            swap_chosen_logps = self.get_batch_logps(
                swap_chosen_logits,
                flatten_image_code_labels,
                average_log_prob=True
            )
            # 6가지
            return (chosen_logps, rejected_logps, swap_chosen_logps, chosen_logits, rejected_logits, swap_chosen_logits, chosen_labels)

        else:
            # 4가지
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels)

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = True,
    ) -> torch.FloatTensor:

        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        # 미리 앞에서 한칸씩 땡켜뒀기 때문에 안땡겨도 된다?
        # labels: (B, L * K) -> 찐 레이블
        # logits: (B, L * K, dim) -> 한칸씩 땡겨 놓고, 각 code 8 개씩을 예측했다. 즉, 한칸씩 땡겨놓고 forward태워서 다시 한칸씩 밈.
        # labels = labels[:, 1:].clone()
        # logits = logits[:, :-1, :]

        loss_mask = labels != self.label_pad_token_id

        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

            
    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        concatenated_batch = {}
        max_length = max(batch["chosen_inputs_embeds"].shape[1], batch["rejected_inputs_embeds"].shape[1])

        for k in batch:
            pad_value = None
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = self.label_pad_token_id
                elif k.endswith("_inputs_embeds"):
                    pad_value = self.padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                if pad_value is not None:
                    concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
                else:
                    concatenated_batch[concatenated_key] = batch[k]
            if k.startswith("chosen") and isinstance(batch[k], list):
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = batch[k]

        for k in batch:
            pad_value = None
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = self.label_pad_token_id
                elif k.endswith("_inputs_embeds"):
                    pad_value = self.padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                if pad_value is not None:
                    concatenated_batch[concatenated_key] = torch.cat(
                        (
                            concatenated_batch[concatenated_key],
                            pad_to_length(batch[k], max_length, pad_value=pad_value),
                        ),
                        dim=0,
                    ).to(device=self.device) # .to(device=self.device)
                else:
                    concatenated_batch[concatenated_key] = torch.cat(
                        (
                            concatenated_batch[concatenated_key],
                            batch[k],
                        ),
                        dim=0,
                    ).to(device=self.device) # .to(device=self.device)
            if k.startswith("rejected") and isinstance(batch[k], list):
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = concatenated_batch[concatenated_key] + batch[k]

        return concatenated_batch
        
        

    def preprocess_train_batch(self, batch):
        """
        Preprocesses a batch of training data.

        Args:
            batch: A tuple of four elements:
                - item_ids: A list of item IDs.
                - text_tokens: A list of text tokens.
                - chosen_image_tensors: A list of chosen image tensors.
                - rejected_image_tensors: A list of rejected image tensors.

        Returns:
            A dictionary containing the preprocessed data.
        """
        batch_size = len(batch[0])
        if self.copo_mode:
            item_ids, chosen_text_tokens, rejected_text_tokens, chosen_image_tensors, rejected_image_tensors = batch
            text_tokens = chosen_text_tokens + chosen_text_tokens + rejected_text_tokens
        else:
            item_ids, text_tokens, chosen_image_tensors, rejected_image_tensors = batch 
            text_tokens = text_tokens + text_tokens

        # Base Simpo
        ## Text preprocess
        ### make labels
        labels = []
        for idx, input_ids in enumerate(text_tokens):
            input_ids = input_ids.clone()
            
            instruction_len = torch.where(input_ids == self.boi.to(self.device))[0].item() # 첫번째 boi 전 까지가 text input

            input_ids[:instruction_len] = IGNORE_INDEX
            labels.append(input_ids)

        ## Padding
        padded_text_tokens = torch.nn.utils.rnn.pad_sequence(
            text_tokens,
            batch_first=True,
            padding_value=self.text_tokenizer.pad_token_id).to(self.device)
        
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX).to(self.device)

        attention_mask = padded_text_tokens.ne(self.text_tokenizer.pad_token_id)

        
        # Image preprocess
        chosen_image_tensors = torch.stack(chosen_image_tensors, dim=0).to(self.device) # (1,3,256,256)
        rejected_image_tensors = torch.stack(rejected_image_tensors, dim=0).to(self.device) # (1,3,256,256)

        with torch.no_grad():
            concatenated_image_tensors = torch.cat([chosen_image_tensors, rejected_image_tensors], dim=0).to(self.device) # 2,

            concatenated_img_tokens = self.img_tokenizer.img_to_idx(concatenated_image_tensors).to(self.device).unsqueeze(1) # (2, 1, 8, 256) 

            chosen_img_tokens = concatenated_img_tokens[:batch_size]
            rejected_img_tokens = concatenated_img_tokens[batch_size:]

        data_types = torch.ones(len(padded_text_tokens), dtype=torch.long, device=self.device) # img generation flag

        if self.copo_mode:
            concatenated_img_tokens = torch.cat([chosen_img_tokens, rejected_img_tokens, chosen_img_tokens], dim=0)
        else:
            concatenated_img_tokens = torch.cat([chosen_img_tokens, rejected_img_tokens], dim=0)

        # Text + vq_token
        # Label

        assert len(padded_text_tokens) == len(padded_labels) == len(attention_mask) == len(data_types) == len(concatenated_img_tokens)
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            data_types,
            additional_image_labels, 
            additional_image_indexs 
        ) = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=padded_text_tokens,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=padded_labels,
            images=concatenated_img_tokens,
            images_aux=None,
            data_types=data_types
        )

        attention_mask = attention_mask.contiguous()
        
        if self.copo_mode:
            preprocessed={"item_ids": item_ids} # item_ids is List.
            preprocessed["chosen_inputs_embeds"] = inputs_embeds[:batch_size].to(self.device)
            preprocessed["chosen_attention_mask"] = attention_mask[:batch_size].to(self.device)
            preprocessed["chosen_labels"] = labels[:batch_size].to(self.device) # output text 부분의 label만.
            preprocessed["chosen_data_types"] = data_types[:batch_size].to(self.device)
            preprocessed["chosen_additional_image_labels"] = additional_image_labels[:batch_size] # list
            preprocessed["chosen_additional_image_indexs"] = additional_image_indexs[:batch_size] # list
            preprocessed["chosen_position_ids"] = position_ids[:batch_size].to(self.device)
            
            preprocessed["rejected_inputs_embeds"] = inputs_embeds[batch_size:batch_size*2].to(self.device)
            preprocessed["rejected_attention_mask"] = attention_mask[batch_size:batch_size*2].to(self.device)
            preprocessed["rejected_labels"] = labels[batch_size:batch_size*2].to(self.device)       
            preprocessed["rejected_data_types"] = data_types[batch_size:batch_size*2].to(self.device)
            preprocessed["rejected_additional_image_labels"] = additional_image_labels[batch_size:batch_size*2]
            preprocessed["rejected_additional_image_indexs"] = additional_image_indexs[batch_size:batch_size*2]
            preprocessed["rejected_position_ids"] = position_ids[batch_size:batch_size*2].to(self.device)

            preprocessed["swap_chosen_inputs_embeds"] = inputs_embeds[batch_size*2:].to(self.device)
            preprocessed["swap_chosen_attention_mask"] = attention_mask[batch_size*2:].to(self.device)
            preprocessed["swap_chosen_labels"] = labels[batch_size*2:].to(self.device)       
            preprocessed["swap_chosen_data_types"] = data_types[batch_size*2:].to(self.device)
            preprocessed["swap_chosen_additional_image_labels"] = additional_image_labels[batch_size*2:]
            preprocessed["swap_chosen_additional_image_indexs"] = additional_image_indexs[batch_size*2:]
            preprocessed["swap_chosen_position_ids"] = position_ids[batch_size*2:].to(self.device)
            return preprocessed
            
        else:
            preprocessed={"item_ids": item_ids} # item_ids is List.
            preprocessed["chosen_inputs_embeds"] = inputs_embeds[:batch_size].to(self.device)
            preprocessed["chosen_attention_mask"] = attention_mask[:batch_size].to(self.device)
            preprocessed["chosen_labels"] = labels[:batch_size].to(self.device) # output text 부분의 label만.
            preprocessed["chosen_data_types"] = data_types[:batch_size].to(self.device)
            preprocessed["chosen_additional_image_labels"] = additional_image_labels[:batch_size] # list
            preprocessed["chosen_additional_image_indexs"] = additional_image_indexs[:batch_size] # list
            preprocessed["chosen_position_ids"] = position_ids[:batch_size].to(self.device)
            
            preprocessed["rejected_inputs_embeds"] = inputs_embeds[batch_size:].to(self.device)
            preprocessed["rejected_attention_mask"] = attention_mask[batch_size:].to(self.device)
            preprocessed["rejected_labels"] = labels[batch_size:].to(self.device)       
            preprocessed["rejected_data_types"] = data_types[batch_size:].to(self.device)
            preprocessed["rejected_additional_image_labels"] = additional_image_labels[batch_size:]
            preprocessed["rejected_additional_image_indexs"] = additional_image_indexs[batch_size:]
            preprocessed["rejected_position_ids"] = position_ids[batch_size:].to(self.device)
            return preprocessed

    def on_before_optimizer_step(self, *args, **kwargs): # optimizer, _):
        # total grad norm
        # self.log('train/grad_norm', self.compute_total_grad_norm(), 
        #          on_step=True, prog_bar=True, logger=True, sync_dist=True)
        grad_norm = self.compute_total_grad_norm()
        
        if grad_norm is not None:  # Ensure no logging of NoneType
            self.log('train/grad_norm', grad_norm, 
                    on_step=True, prog_bar=True, logger=True, sync_dist=True)
        else:
            print("⚠️ Warning: Grad norm is None, skipping logging.")

    def compute_total_grad_norm(self):
        """
        DeepSpeed shards gradients across multiple GPUs when using ZeRO-2 or ZeRO-3, 
        meaning that some parameters won't have .grad stored locally in self.model.parameters(). 
        
        This is why your compute_total_grad_norm() function is returning 0.0, and no "Gradient exists for {p} !" messages are printed.
        """
        if hasattr(self.trainer.model, "get_global_grad_norm"):
            grad_norm = self.trainer.model.get_global_grad_norm()
            return grad_norm if grad_norm is not None else 0.0  # Avoid NoneType errors
        else:
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** 0.5 if total_norm > 0 else 0.0  # Prevent NoneType issue

    def configure_optimizers(self):
        if self.config.use_ds:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            from deepspeed.ops.adam import FusedAdam
            optimizer = DeepSpeedCPUAdam(
                self.parameters(),
                lr=self.config.optimizer.init_lr,
                betas=self.config.optimizer.betas,
                weight_decay=self.config.optimizer.weight_decay
            )
        else:
            optimizer = AdamW(
                self.parameters(), 
                lr=self.config.optimizer.init_lr,
                betas=self.config.optimizer.betas,
                weight_decay=self.config.optimizer.weight_decay,
                eps=self.config.optimizer.eps,
            )

        if self.config.optimizer.scheduler_type == 'constant':
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=self.config.experiment.max_training_steps)

        elif self.config.optimizer.scheduler_type == 'cosine':
            warmup_step = self.config.experiment.max_training_steps * self.config.experiment.warmup_ratio
            scheduler = CosineDecayWarmUpRestarts(optimizer, 
                                                warmup_iter=warmup_step, 
                                                max_iter=self.config.experiment.max_training_steps, 
                                                eta_min=self.config.optimizer.min_lr, 
                                                eta_max=self.config.optimizer.init_lr)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return [optimizer], [scheduler_config]
        
