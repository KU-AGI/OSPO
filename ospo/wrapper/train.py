import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR
import pytorch_lightning as pl
from trl.trainer.utils import pad_to_length 
from typing import *
from PIL import Image
from collections import defaultdict
import math

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.constant import *
from ospo.utils.common import save_config, save_json_ddp
from ospo.utils.train import CosineDecayWarmUpRestarts
from ospo.utils.processor import get_conversation, get_sft_format


class JanusProTrainWrapper(pl.LightningModule):
    def __init__(self, config, model, chat_processor, image_processor, tokenizer):
        super().__init__()
        self.config=config
        self.model=model
        self.chat_processor=chat_processor
        self.image_processor=image_processor
        self.tokenizer=tokenizer

        self.train_dataset = []
        self.val_dataset = None

        # default SimPO hyperparameter 
        simpo_config = self.config['algo']
        tokenizer_config = self.config['tokenizer']

        self.beta = simpo_config.get('beta', 1.0)
        self.gamma_beta_ratio = simpo_config.get('gamma_beta_ratio', 0.0)
        self.label_smoothing = simpo_config.get('label_smoothing', 0.0)
        self.loss_type = simpo_config.get('loss_type', 'sigmoid')
        
        # Loss weight
        self.do_schedule_loss_weight = simpo_config.get('schedule_loss_weight', False)
        self.simpo_weight = simpo_config.get('simpo_weight', 1.0) 
        self.sft_weight = simpo_config.get('sft_weight', 0.0)
        # Sanity Check
        if self.sft_weight > 0.0:
            print("Notice: SFT weight is set to a non-zero value.")

        # Token Mask
        self.mask_alpha = simpo_config.get('mask_alpha', 1.0) # strength of emphasis
        self.mask_gamma = simpo_config.get('mask_gamma', 1.0) # sharpening/softening

        self.label_pad_token_id = tokenizer_config.get('label_pad_token_id', -100)
        self.padding_value = self.tokenizer.pad_token_id 
        self.max_length = tokenizer_config.get('max_length', 512)
        self.max_prompt_length = tokenizer_config.get('max_prompt_length', 128)

        # Logging setting
        self.log_by_category = self.config.experiment.log_by_category

        
    def setup(self, stage: str):
        save_config(self.logger.log_dir, self.config)
   
        self.model.train()        
        self.model.language_model.model.config.output_hidden_states = True
        
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


    def print_trainable_parameters(self):
        print("Trainable Parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}, dtype={param.dtype}")


    def on_train_start(self):
        self.model.train()
        print("Training START.")

        # Double-check critical submodules explicitly (optional but recommended)
        device = self.device
        self.model.language_model.to(device)
        self.model.gen_vision_model.to(device)
        self.model.gen_head.to(device)
        self.model.vision_model.to(device)
        self.model.aligner.to(device)
        self.model.gen_aligner.to(device)
        self.model.gen_embed.to(device)


    def training_step(self, batch, batch_idx):
        """ A batch is a dict. """
        preprocessed = self.preprocess_train_batch(batch) 
        loss = self.compute_loss(inputs=preprocessed) 
        self.log_dict({'train/loss': loss,
                       'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']},
                        on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss


    def on_validation_start(self):
        self.model.eval()
        print("Validation START.")

        self.yes_ids = [self.tokenizer("yes", add_special_tokens=False).input_ids[-1],
                        self.tokenizer("Yes", add_special_tokens=False).input_ids[-1]]
        self.no_ids  = [self.tokenizer("no", add_special_tokens=False).input_ids[-1],
                        self.tokenizer("No", add_special_tokens=False).input_ids[-1]]        
        
        self.val_output_local_score = defaultdict(list)
        self.val_output_accuracy = defaultdict(list)   

        # task_id = 2,3,4,5 (from the top)
        self.val_output_logps_gap_hn_img_id = list()
        self.val_output_logps_gap_hn_img_ood = list()
        self.val_output_logps_gap_hn_txt_id = list()
        self.val_output_logps_gap_hn_txt_ood = list()

        self.val_directory = os.path.join(self.logger.log_dir, 'validation', f'step_{self.global_step}')
        os.makedirs(self.val_directory, exist_ok=True)
        print("Validation START.")


    @torch.inference_mode()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
            task_id = 0: VQA score
            task_id = 1: Validation loss
            task_id = 2,3: Logps gap between chosen and rejected (for ID/OOD and image/text perturbation)
            unknown = perturbed tokens, questions (or None)
        """
        task_ids, item_ids, text_tokens, unknown, chosen_image_tensors, rejected_image_tensors = batch

        directory = os.path.join(self.val_directory, "images")
        os.makedirs(directory, exist_ok=True)

        if dataloader_idx == 0: 
            question_list = unknown
            text_input_ids_list = text_tokens 
            save_path_list = [os.path.join(directory, f"{item_id}.png") for item_id in item_ids]

            self.generate_image_batch(
                text_input_ids_list=text_input_ids_list, # instead of prompt_list
                save_path_list=save_path_list,
            )

        elif dataloader_idx != 0: 
            preprocessed = self.preprocess_val_batch(batch)
            loss = self.prediction_step(inputs=preprocessed)

            if dataloader_idx == 1:
                self.log_dict({f'val/loss': loss}, on_step=True, prog_bar=True, logger=True, sync_dist=True)                

        else:
            pass       
    

    def on_validation_end(self):
        # (Task 0) VQA   
        if self.trainer.world_size > 1:
            local_output = list(self.val_output_local_score.items())
            gathered_output = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(gathered_output, local_output)

            # Merge lists per key
            merged = {}
            for rank_output in gathered_output:
                for k, v_list in rank_output:
                    merged.setdefault(k, []).extend(v_list)  # merge lists
            
            # Average list values for logging
            reduced = {k: sum(v) / len(v) for k, v in merged.items()}

            if self.trainer.global_rank == 0:
                for k, v in reduced.items():
                    self.logger.experiment.add_scalar(f"val/task_0/{k}", v, self.global_step)

        else:
            # Single process case, directly use the local output
            reduced = {k: sum(v) / len(v) for k, v in self.val_output_local_score.items()}
            for k, v in reduced.items():
                self.logger.experiment.add_scalar(f"val/task_0/{k}", v, self.global_step)

        # (Task 2-5)
        save_json_ddp(save_root=os.path.join(self.val_directory, "logps_gap"),
                      save_name='task_2_logps_gap_hn_img_id_result',
                      world_size=self.trainer.world_size,
                      save_file=self.val_output_logps_gap_hn_img_id,
                      rank=self.trainer.global_rank)
        save_json_ddp(save_root=os.path.join(self.val_directory, "logps_gap"),
                      save_name='task_3_logps_gap_hn_img_ood_result',
                      world_size=self.trainer.world_size,
                      save_file=self.val_output_logps_gap_hn_img_ood,
                      rank=self.trainer.global_rank)

        print("Validation END.")


    def get_image_generation_prompt(self, prompt):
        system_prompt = ""
        converation = get_conversation(prompt)
        sft_format = get_sft_format(self.chat_processor, system_prompt, converation)
        prompt = sft_format + self.chat_processor.image_start_tag

        return prompt


    def build_conversation(self, img, questions):
        convs = []
        for q in questions:
            convs.append([
                {"role": "<|User|>",
                "content": f"<image_placeholder>\n{q} Please answer 'yes' or 'no' without explanation.",
                "images": [img]},
                {"role": "<|Assistant|>", "content": ""}
            ])

        return convs, [[img]] * len(questions)

    
    def forward_single(self, convs, imgs):
        prepare_list = []
        for conv, img in zip(convs, imgs):
            prepare = self.chat_processor.process_one(
                conversations=conv,
                images=img,
                force_batchify=True
            )
            prepare_list.append(prepare)

        with torch.no_grad():
            batch_inputs = self.chat_processor.batchify(prepare_list).to(self.device)
            inputs_embeds = self.model.prepare_inputs_embeds(**batch_inputs)
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch_inputs.attention_mask
            )

        return outputs.logits

    # for validation - 1
    @torch.inference_mode()
    def generate_image_batch(
        self,
        save_path_list: List,
        prompt_list: List = None,
        text_input_ids_list: List[torch.LongTensor] = None,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
    ):
        
        if prompt_list is not None and text_input_ids_list is not None:
            raise ValueError("Only one of 'prompt_list' or 'text_input_ids_list' should be provided.")
        
        if prompt_list is not None:
            batch_size = len(prompt_list)
            input_ids_list = []
            max_len = 0 # for padding

            for prompt in prompt_list:
                input_ids = self.tokenizer.encode(prompt)
                input_ids = torch.LongTensor(input_ids)

                max_len = max(max_len, len(input_ids))
                input_ids_list.append(input_ids)

        elif text_input_ids_list is not None:
            batch_size = len(text_input_ids_list) 
            input_ids_list = []
            max_len = 0 # for padding

            for input_ids in text_input_ids_list:
                max_len = max(max_len, len(input_ids))
                input_ids_list.append(input_ids)

        # initialize
        tokens = torch.zeros((batch_size*2, max_len), dtype=torch.int).to(self.device)
        attention_masks = torch.ones((batch_size*2, max_len), dtype=torch.long).to(self.device)
    
        for i in range(batch_size*2):
            pad_len = max_len - len(input_ids_list[i//2])
            tokens[i, pad_len:] = input_ids_list[i//2]
            tokens[i, :pad_len] = self.chat_processor.pad_id
            attention_masks[i, :pad_len] = 0
            if i % 2 != 0:
                tokens[i, pad_len+1:-1] = self.chat_processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((batch_size, image_token_num_per_image), dtype=torch.int).cuda()

        for i in range(image_token_num_per_image):
            outputs = self.model.language_model.model(inputs_embeds=inputs_embeds, 
                                                    attention_mask=attention_masks, 
                                                    use_cache=True, 
                                                    past_key_values=outputs.past_key_values if i != 0 else None)

            # hidden_states = outputs.last_hidden_state
            hidden_states = outputs.hidden_states[-1] # = last_hidden_state
            
            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + self.config.dataset.val.cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / self.config.dataset.val.temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1) # torch.Size([1, 1]) / tensor([[4521]], device='cuda:0')
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

            new_mask = torch.ones((attention_masks.shape[0], 1), dtype=attention_masks.dtype, device=attention_masks.device)  
            attention_masks = torch.cat([attention_masks, new_mask], dim=1)

        dec = self.model.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[batch_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        
        visual_img = np.zeros((batch_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec # (1, 384, 384, 3)

        # batch_size > 1 
        for inner_idx, image in enumerate(visual_img):
            try:
                Image.fromarray(image).save(save_path_list[inner_idx])
                
            except OSError:
                idx_in_path = save_path_list[inner_idx].split("_")[1] # 01.png
                alternative_path = f"longprompt_{idx_in_path}"

                # PIL.Image.fromarray(image).save(os.path.join(sample_path_list[inner_idx],f'longname_{idx}.png'))
                Image.fromarray(image).save(alternative_path)

    # for validation - 2
    @torch.inference_mode()
    def generate_vqa(self, save_path_list: List, question_list: List):        
        for img_path, questions in zip(save_path_list, question_list):
            q_count = len(questions)
            item_id = os.path.basename(img_path).split('.png')[0]

            with Image.open(img_path) as img:
                convs, imgs = self.build_conversation(img, questions)
                logits = self.forward_single(convs, imgs)  # shape: [num_questions, seq_len, vocab]
                probs = torch.softmax(logits[:, -1, :], dim=-1)

                score_sum = 0
                yes_count = 0 # for calculating accuracy

                for q_idx in range(q_count):
                    p_yes = max(probs[q_idx, y].item() for y in self.yes_ids)
                    p_no = max(probs[q_idx, n].item() for n in self.no_ids)
                    answer = 'yes' if p_yes > p_no else ('no' if p_no > p_yes else 'tie')

                    score_sum += (p_yes - p_no)
                    if answer == 'yes':
                        yes_count += 1
                        
                local_score = score_sum / q_count
                accuracy = yes_count / q_count if q_count > 0 else 0.0
            
            if item_id[0] in ["0", "1", "2"]:
                category = "attribute"
            elif item_id[0] in ["3", "4", "5"]:
                category = "layout"
            elif item_id[0] == "6":
                category = "non-spatial"
            elif item_id[0] == "7":
                category = "complex"
            elif item_id[0] == "8":
                category = "dpgbench" 

            self.val_output_local_score[category].append(local_score)
            self.val_output_accuracy[category].append(accuracy)


    # TODO: modify based on preprocess_train_batch function
    def preprocess_val_batch(self, batch: Tuple):
        """
        task_id = 1,2,3
        - chosen seq: chosen prompt + 'chosen'
        - rejected seq: chosen prompt + 'rejected'
        """
        batch_size = len(batch[0])
        task_ids, item_ids, text_tokens, unknown, chosen_image_tensors, rejected_image_tensors = batch
        
        # 1. TEXT
        chosen_text_embeds=[] 
        rejected_text_embeds=[]

        # 2. IMAGE
        expected_dtype = next(self.model.gen_vision_model.parameters()).dtype   
        chosen_img_token_tensor_list=[]
        rejected_img_token_tensor_list=[]

        for task_id, text_token, optional, chosen_ts, rejected_ts in zip(
            task_ids, text_tokens, unknown, chosen_image_tensors, rejected_image_tensors 
        ):
            if chosen_ts.dtype != expected_dtype:
                chosen_ts = chosen_ts.to(dtype=expected_dtype)
            chosen_output = self.model.gen_vision_model.encode(chosen_ts.to(self.device))   
            # img_tokens: [576]
            chosen_img_tokens = chosen_output[2][2]  
            chosen_img_token_tensor_list.append(chosen_img_tokens)
              
            if task_id in [1,2,3]:
                # text
                base_text_embeds = self.model.language_model.get_input_embeddings()(text_token)
                chosen_text_embeds.append(base_text_embeds) 
                rejected_text_embeds.append(base_text_embeds) # note

                # image
                if rejected_ts.dtype != expected_dtype:
                    rejected_ts = rejected_ts.to(dtype=expected_dtype)
                rejected_output = self.model.gen_vision_model.encode(rejected_ts.to(self.device))
                rejected_img_tokens = rejected_output[2][2]  
                rejected_img_token_tensor_list.append(rejected_img_tokens) 

            else:
                raise ValueError(f"Task ID {task_id} is not supported in preprocessing batch step.")


        def pad_text(batched_embeds, max_seq_len:int):
            # 1+. padding
            # max_seq_len = max(x.shape[1] for x in batched_embeds)
            embed_dim = batched_embeds[0].size(-1) # 4096
            padded_batched_embeds = torch.zeros(batch_size, max_seq_len, embed_dim, dtype=self.model.dtype, device=self.device) 
            padded_batched_labels = torch.full(
                (batch_size, max_seq_len), -100, dtype=torch.long, device=self.device
            )  # initialize with padding value 

            for i, embed in enumerate(batched_embeds):
                seq_len = embed.shape[1]
                padded_batched_embeds[i, :seq_len, :] = embed 

            return padded_batched_embeds, padded_batched_labels

        max_len = max(
            max(x.shape[1] for x in chosen_text_embeds), 
            max(x.shape[1] for x in rejected_text_embeds)
        )
        # torch.Size([bsz, seq_len, 4096])
        padded_batched_chosen_text_embeds, padded_batched_chosen_text_labels = pad_text(chosen_text_embeds, max_len)               
        padded_batched_rejected_text_embeds, padded_batched_rejected_text_labels = pad_text(rejected_text_embeds, max_len) 

        batched_chosen_tensors = torch.stack(chosen_img_token_tensor_list, dim=0)
        batched_rejected_tensors = torch.stack(rejected_img_token_tensor_list, dim=0) 

        batched_chosen_img_embeds = self.model.prepare_gen_img_embeds(batched_chosen_tensors).to(self.device)
        batched_rejected_img_embeds = self.model.prepare_gen_img_embeds(batched_rejected_tensors).to(self.device)

        # 3. OUTPUT BATCH
        simpo_batch={"task_ids": task_ids, "item_ids": item_ids} 
        simpo_batch["chosen_inputs_embeds"] = torch.cat([padded_batched_chosen_text_embeds, batched_chosen_img_embeds], dim=1) 
        simpo_batch["chosen_attention_mask"] = torch.ones(simpo_batch["chosen_inputs_embeds"].shape[:2], dtype=torch.long) 
        simpo_batch["chosen_labels"] = torch.cat([padded_batched_chosen_text_labels, batched_chosen_tensors], dim=1) 
        
        simpo_batch["rejected_inputs_embeds"] = torch.cat([padded_batched_rejected_text_embeds, batched_rejected_img_embeds], dim=1) 
        simpo_batch["rejected_attention_mask"] = torch.ones(simpo_batch["rejected_inputs_embeds"].shape[:2], dtype=torch.long) 
        simpo_batch["rejected_labels"] = torch.cat([padded_batched_rejected_text_labels, batched_rejected_tensors], dim=1) 

        return simpo_batch


    def on_before_optimizer_step(self, *args, **kwargs): 
        grad_norm = self.compute_total_grad_norm() # total grad norm

        if grad_norm is not None: # ensure no logging of NoneType
            self.log('train/grad_norm', grad_norm, 
                    on_step=True, prog_bar=True, logger=True, sync_dist=True)


    def configure_optimizers(self):
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
        
        
    # https://github.com/Lightning-AI/pytorch-lightning/blob/0.9.0/pytorch_lightning/core/lightning.py#L807-L838 (OVERRIDE)
    def configure_ddp(self, model, device_ids):
        """Override DDP to track only trainable parameters (LoRA)"""
        model = DDP(
            model,
            device_ids=device_ids,
            find_unused_parameters=False,  # Prevents error with frozen parameters
        )
        model._set_static_graph()          # Fixes the computation graph

        return model


    def freeze_param(self):
        freeze = self.config.experiment.freeze

        # We assume that we only train image-generation-related modules.
        if self.config.use_peft:
            if freeze.vision_model:
                for param in self.model.vision_model.parameters():
                    param.requires_grad = False

            if freeze.aligner:
                for param in self.model.aligner.parameters():
                    param.requires_grad = False

            if freeze.gen_vision_model:
                for name, module in self.model.gen_vision_model.named_modules():
                    if isinstance(module, torch.nn.Embedding):  # nn.Embedding Layer
                        for param in module.parameters():
                            param.requires_grad = False         # Token Embedding layer is not trainable.
                    else:
                        for param in module.parameters():
                            param.requires_grad = False

            if freeze.gen_aligner:
                for param in self.model.gen_aligner.parameters():
                    param.requires_grad = False

            if freeze.gen_head:
                for param in self.model.gen_head.parameters():
                    param.requires_grad = False

            if freeze.gen_embed:
                for param in self.model.gen_embed.parameters():
                    param.requires_grad = False
            
        else:
            self.freeze() 

            if not freeze.vision_model:
                for param in self.model.vision_model.parameters():
                    param.requires_grad = True

            if not freeze.aligner:
                for param in self.model.aligner.parameters():
                    param.requires_grad = True

            if not freeze.gen_vision_model:
                for name, module in self.model.gen_vision_model.named_modules():
                    if isinstance(module, torch.nn.Embedding):  
                        for param in module.parameters():
                            param.requires_grad = False        
                    else:
                        for param in module.parameters():
                            param.requires_grad = True

            if not freeze.gen_aligner:
                for param in self.model.gen_aligner.parameters():
                    param.requires_grad = True

            if not freeze.gen_head:
                for param in self.model.gen_head.parameters():
                    param.requires_grad = True

            if not freeze.gen_embed:
                for param in self.model.gen_embed.parameters():
                    param.requires_grad = True

            if not freeze.language_model:
                for param in self.model.language_model.parameters():
                    param.requires_grad = True


    def preprocess_batch_text(self, seq_list: List):
        batched=[]
        for seq_ts in seq_list:
            text_embeds = self.model.language_model.get_input_embeddings()(seq_ts)
            text_embeds = text_embeds.squeeze(0)
            batched.append(text_embeds) 

        return batched # List of torch.Size([seq_len, 4096])
    

    def preprocess_batch_image(self, seq_list: List):
        for_batched=[]
        # Get model parameter dtype
        expected_dtype = next(self.model.gen_vision_model.parameters()).dtype    
    
        for seq_ts in seq_list:
            if seq_ts.dtype != expected_dtype:
                # print(f"*** Cast to {expected_dtype} ***")
                seq_ts = seq_ts.to(dtype=expected_dtype)
            
            output = self.model.gen_vision_model.encode(seq_ts.to(self.device)) 
            # img_tokens [576]
            img_tokens = output[2][2]  
            for_batched.append(img_tokens)

        batched = torch.stack(for_batched, dim=0)
        # torch.Size([2, 576])

        batched_img_embeds = self.model.prepare_gen_img_embeds(batched).to(self.device)
        # torch.Size([2, 576, 4096])

        return batched_img_embeds, batched # for label


    def truncate(self, tensor, max_len, dim=1):
        if tensor.size(dim) > max_len:
            print("Truncating tensor along dim", dim)
            return tensor.narrow(dim, 0, max_len)  # truncates along dim

        return tensor
    

    def pad_to_max_len_in_batch(self, tensor_list, pad_value=0.0):
        # print(tensor_list.shape)
        max_len = max(t.size(0) for t in tensor_list)
        padded = []
        for t in tensor_list:
            pad_len = max_len - t.size(0)
            if pad_len > 0 and pad_value == -100:
                pad_shape = list(t.shape)
                pad_shape[0] = pad_len
                padding = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
                t = torch.cat([t, padding], dim=0) # pad-right
            elif pad_len > 0 and pad_value == 0:
                pad_shape = list(t.shape)
                pad_shape[0] = pad_len
                padding = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
                t = torch.cat([t, padding], dim=0) # pad-right
            elif pad_len > 0:
                pad_embed = self.model.language_model.get_input_embeddings()(torch.tensor([pad_value], device=t.device))
                pad_embed = pad_embed.expand(pad_len, -1)
                t = torch.cat([t, pad_embed], dim=0)
            padded.append(t)

        return torch.stack(padded, dim=0)


    def preprocess_train_batch(self, batch: Tuple):

        batch_size = len(batch[0])

        # text: torch.Size([seq_len, 4096])
        # chosen_img: torch.Size([576, 4096])

        # SimPO Loss (only) mode
        if self.config.use_mask:
            item_ids, chosen_text_tokens, chosen_image_tensors, rejected_image_tensors, chosen_masks, rejected_masks = batch 
        else:
            item_ids, chosen_text_tokens, chosen_image_tensors, rejected_image_tensors = batch 

        batched_chosen_text = self.preprocess_batch_text(chosen_text_tokens)    
        batched_chosen_img_embeds, batched_chosen_img_labels = self.preprocess_batch_image(chosen_image_tensors)
        batched_rejected_img_embeds, batched_rejected_img_labels = self.preprocess_batch_image(rejected_image_tensors)

        # chosen / rejected sequence (txt + img)
        chosen_seq_list = []
        rejected_seq_list = []
        chosen_seq_label_list = []
        rejected_seq_label_list = []
        chosen_attention_mask_list = []
        rejected_attention_mask_list = []

        for chosen_text, chosen_img, rejected_img, chosen_img_label, rejected_img_label in zip(
                batched_chosen_text, batched_chosen_img_embeds, batched_rejected_img_embeds, batched_chosen_img_labels, batched_rejected_img_labels):

            # Step 1: Truncate prompt
            chosen_text = self.truncate(chosen_text, self.max_prompt_length, dim=0)
            chosen_text_len = chosen_text.shape[0]

            # Step 2: Concat
            chosen_seq = torch.cat([chosen_text, chosen_img], dim=0)
            rejected_seq = torch.cat([chosen_text, rejected_img], dim=0)

            # Step 3: Truncate final sequence to MAX_SEQ_LENGTH
            chosen_seq = self.truncate(chosen_seq, self.max_length, dim=0)
            rejected_seq = self.truncate(rejected_seq, self.max_length, dim=0)

            chosen_seq_list.append(chosen_seq)
            rejected_seq_list.append(rejected_seq)

            # Step 4: Label 
            chosen_label = torch.full(
                (chosen_text_len,), self.label_pad_token_id, dtype=torch.long, device=self.device
                ) # batch_size, seq_len, -100
            rejected_label = torch.full(
                (chosen_text_len,), self.label_pad_token_id, dtype=torch.long, device=self.device
                ) # batch_size, seq_len, -100

            chosen_seq_label = torch.cat([chosen_label, chosen_img_label], dim=0)
            rejected_seq_label = torch.cat([rejected_label, rejected_img_label], dim=0)

            # Truncate label 
            chosen_seq_label = self.truncate(chosen_seq_label, self.max_length, dim=0)
            rejected_seq_label = self.truncate(rejected_seq_label, self.max_length, dim=0)

            chosen_seq_label_list.append(chosen_seq_label)
            rejected_seq_label_list.append(rejected_seq_label)

            chosen_attention_mask = torch.ones(chosen_seq_label.shape[0], dtype=torch.long, device=self.device)
            rejected_attention_mask = torch.ones(rejected_seq_label.shape[0], dtype=torch.long, device=self.device)

            chosen_attention_mask_list.append(chosen_attention_mask)
            rejected_attention_mask_list.append(rejected_attention_mask)

        
        padded_chosen_inputs_embeds = self.pad_to_max_len_in_batch(chosen_seq_list, pad_value=self.padding_value)
        padded_rejected_inputs_embeds = self.pad_to_max_len_in_batch(rejected_seq_list, pad_value=self.padding_value)
        
        padded_chosen_labels = self.pad_to_max_len_in_batch(chosen_seq_label_list, pad_value=self.label_pad_token_id)
        padded_rejected_labels = self.pad_to_max_len_in_batch(rejected_seq_label_list, pad_value=self.label_pad_token_id)

        padded_chosen_attention_mask = self.pad_to_max_len_in_batch(chosen_attention_mask_list, pad_value=0)
        padded_rejected_attention_mask = self.pad_to_max_len_in_batch(rejected_attention_mask_list, pad_value=0)

        
        output = {
            "item_ids": item_ids, # List[str]
            "chosen_inputs_embeds": padded_chosen_inputs_embeds,
            "chosen_labels": padded_chosen_labels,
            "chosen_attention_mask": padded_chosen_attention_mask,
            "rejected_inputs_embeds": padded_rejected_inputs_embeds,
            "rejected_labels": padded_rejected_labels,
            "rejected_attention_mask": padded_rejected_attention_mask,
        }

        if self.config.use_mask:
            # stack tensor
            batched_chosen_masks = torch.stack(chosen_masks, dim=0)
            batched_rejected_masks = torch.stack(rejected_masks, dim=0)
            
            output["chosen_token_mask"] = batched_chosen_masks
            output["rejected_token_mask"] = batched_rejected_masks

        return output    


    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        concatenated_batch = {}
        max_length = max(batch["chosen_inputs_embeds"].shape[1], batch["rejected_inputs_embeds"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = self.label_pad_token_id
                elif k.endswith("_inputs_embeds"):
                    pad_value = self.padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                else:
                    pass

                concatenated_key = k.replace("chosen", "concatenated")
                if not k.endswith("_token_mask"):
                    concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
                else:
                    concatenated_batch[concatenated_key] = batch[k]

        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = self.label_pad_token_id
                elif k.endswith("_inputs_embeds"):
                    pad_value = self.padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                else:
                    pass

                concatenated_key = k.replace("rejected", "concatenated")
                if not k.endswith("_token_mask"):
                    concatenated_batch[concatenated_key] = torch.cat(
                        (
                            concatenated_batch[concatenated_key],
                            pad_to_length(batch[k], max_length, pad_value=pad_value),
                        ),
                        dim=0,
                    ).to(device=self.device) 
                else:
                    mask_length = 576 # 24 * 24
                    concatenated_batch[concatenated_key] = torch.cat(
                        (
                            concatenated_batch[concatenated_key],
                            batch[k], # no padding needed
                        ),
                        dim=0,
                    ).to(device=self.device) 

        return concatenated_batch
        

    def simpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> Dict[str, torch.FloatTensor]:
        
        """
        Calculate SimPO losses.
        
        Args:
            policy_chosen_logps: Log probabilities of chosen responses
            policy_rejected_logps: Log probabilities of rejected responses
            
        Returns:
            Dict containing:
                - chosen_rewards: Rewards for chosen responses
                - rejected_rewards: Rewards for rejected responses
                - simpo_loss: SimPO loss value
        """

        gamma = self.gamma_beta_ratio * self.beta
        
        chosen_rewards = self.beta * policy_chosen_logps.to(self.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.device).detach()

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        pi_logratios = pi_logratios.to(self.device)
        logits = pi_logratios - self.gamma_beta_ratio

        loss = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )

        return {
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
            "simpo_loss": loss,
        }


    def concatenated_forward(
            self, batch: Dict[str, Union[List, torch.LongTensor]]
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

            """ 1. Base """   
            concatenated_batch = self.concatenated_inputs(batch=batch)
            len_chosen = batch["chosen_labels"].shape[0]

            outputs = self.model.language_model.model(inputs_embeds=concatenated_batch["concatenated_inputs_embeds"], 
                                                    attention_mask=concatenated_batch["concatenated_attention_mask"],
                                                    use_cache=False, 
                                                    past_key_values=None,
                                                    output_hidden_states=True,
                                                    output_attentions=True,
                                                    return_dict=True,
                                                    ) 

            hidden_states = outputs.hidden_states[-1]

            all_logits = self.model.gen_head(hidden_states)       
            all_logps = self.get_batch_logps( # [n_chosen + n_rejected, :]
                all_logits,
                concatenated_batch["concatenated_labels"],
                token_masks=concatenated_batch["concatenated_token_mask"] if self.config.use_mask else None,
                average_log_prob=True,
            )

            chosen_logps = all_logps[:len_chosen]
            rejected_logps = all_logps[len_chosen:]

            chosen_logits = all_logits[:len_chosen]
            rejected_logits = all_logits[len_chosen:]

            chosen_labels = concatenated_batch["concatenated_labels"][:len_chosen]
            rejected_labels = concatenated_batch["concatenated_labels"][len_chosen:]

            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels, rejected_labels)
    

    def gaussian_kernel(self, kernel_size=5, sigma=1.0):
        """Create 2D Gaussian kernel"""
        k = kernel_size // 2
        xs = torch.arange(-k, k+1, device=self.device, dtype=torch.float32)
        ys = torch.arange(-k, k+1, device=self.device, dtype=torch.float32)
        xs, ys = torch.meshgrid(xs, ys, indexing='ij')
        kernel = torch.exp(-(xs**2 + ys**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)  # shape (1,1,K,K)


    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        token_masks: torch.Tensor = None, 
        average_log_prob: bool = True,
        eps: float = 1e-8
    ) -> torch.FloatTensor:
        
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")
        
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        B, SL, V = logits.shape # batch size, sequence length, vocab size
                                # B, SL, V: 16, 678, 16384 

        assert logits.shape[:-1] == labels.shape, f"Logits shape: {logits.shape}, Labels shape: {labels.shape}"
        
        loss_mask = labels != self.label_pad_token_id
        labels[labels == self.label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)    
        weight = loss_mask.to(dtype=per_token_logps.dtype) # torch.Size([16, 678]) 

        if token_masks is not None:
            # 1) search img_span
            has_any = loss_mask.any(dim=1)  
            # first index of True (per row)
            start = loss_mask.float().argmax(dim=1) # (B,)
            # last index of True (per row) + 1 (exclusive)
            end = (labels.size(1) - torch.flip(loss_mask, [1]).float().argmax(dim=1))
        
            # rows with no True → set start=end=0 and mask all False
            img_start = torch.where(has_any, start, torch.zeros_like(start))
            img_end   = torch.where(has_any, end,   torch.zeros_like(end))

            # 2) reshape
            tm = token_masks
            # 2D grid, Flatten it
            if tm.dim() == 3 and tm.shape[-2:] == (24, 24):
                tm = tm.reshape(B, -1) # (B, Timg)  # torch.Size([16, 576])
            else:
                raise ValueError(
                    f"token_masks must be (B,24,24), but you gave {tm.shape}\n{tm[0]}"
                )

            """ SOFT MASKING VER. """
            B, SL = weight.shape
            T_img = tm.shape[1] # 576

            # --- 1) Soften the binary token mask in 2D, then flatten back ---
            GRID_H = GRID_W = int((tm.shape[1]) ** 0.5)  # e.g., 576 -> 24
            assert GRID_H == 24 and GRID_W == 24, f"Unexpected grid size: {GRID_H}, {GRID_W}"
            tm2d = tm.view(B, 1, GRID_H, GRID_W).float()

            # --- 2) Gaussian smoothing ---
            kernel = self.gaussian_kernel(kernel_size=5, sigma=1.0) 
            tm_soft2d = F.conv2d(tm2d, kernel, padding=kernel.shape[-1]//2)

            # --- 3) Normalize to [0,1] per-batch-item ---
            mins = tm_soft2d.amin(dim=(2,3), keepdim=True)
            maxs = tm_soft2d.amax(dim=(2,3), keepdim=True)
            tm_soft2d = (tm_soft2d - mins) / (maxs - mins + eps)

            # --- 4) Sharpen / soften boundary ---
            tm_soft = tm_soft2d.pow(self.mask_gamma).view(B, T_img).clamp(0, 1)

            mask_full = torch.ones((B, SL), device=weight.device, dtype=weight.dtype)
            for b in range(B):
                s, e = int(img_start[b].item()), int(img_end[b].item())
                boost = 1.0 + self.mask_alpha * tm_soft[b]                  # (T_img,)
                mask_full[b, s:e] = boost.to(mask_full.dtype)

            # --- 5) Combine with padding mask ---
            weight = weight * mask_full # torch.Size([16, 678])  
        
        weighted_sum = (per_token_logps * weight).sum(dim=-1)  # (B,)

        if average_log_prob:
            # return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1) # original
            denom = weight.sum(dim=-1).clamp_min(1.0) # avoid divide by zero
            return weighted_sum / denom
        else:
            # return (per_token_logps * loss_mask).sum(-1) # original
            return weighted_sum
        

    # Actually, not used
    def schedule_loss_weight(self):
        # Normalize to [0, 1]
        progress = self.global_step / self.config.experiment.max_training_steps

        # Use cosine or linear scheduling
        def linear_decay(start, end):
            return start + (end - start) * progress

        def cosine_decay(start, end):
            import math
            return end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))

        # Example: cosine schedule
        # simpo_weight = cosine_decay(1.0, 0.1)  # large → small

        # TODO: hard-coding
        simpo_weight = self.simpo_weight
        sft_weight   = cosine_decay(1.0, 2.0)  # small → large

        return simpo_weight, sft_weight


    def get_batch_loss_metrics(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "val"] = "train",
    ):        
        item_ids = batch["item_ids"]

        if train_eval == "train": 
            do_logging = True
        else:
            # debug
            print(f"START Rank: {self.global_rank}, Batch: {len(batch)}, task_ids: {batch['task_ids']}")
            task_ids = batch["task_ids"]
            if all(task_id == 1 for task_id in task_ids): # simPO Loss
                do_logging = True
            else:
                do_logging = False
        
        outputs = self.concatenated_forward(batch)

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_labels,
            policy_rejected_labels
        ) = outputs

        loss_kwargs = {
            "policy_chosen_logps": policy_chosen_logps,
            "policy_rejected_logps": policy_rejected_logps,
        }

        loss_outputs = self.simpo_loss(**loss_kwargs)
        
        if self.do_schedule_loss_weight:
            # adjust
            self.simpo_weight, self.sft_weight = self.schedule_loss_weight()

        self.log_dict({'loss_weight/simpo': self.simpo_weight,
                        'loss_weight/sft': self.sft_weight,
                        }, on_step=True, prog_bar=True, logger=True, sync_dist=True)
    
        apply_simpo_loss = self.simpo_weight * loss_outputs['simpo_loss']

        if self.sft_weight > 0.0:
            chosen_labels = policy_chosen_labels
            chosen_labels = chosen_labels[..., 1:].clone().contiguous()
            policy_chosen_logits = policy_chosen_logits[..., :-1, :].contiguous()

            loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id)
            sft_loss = loss_func(policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1]), chosen_labels.view(-1))
            apply_sft_loss = self.sft_weight * sft_loss

            loss = (apply_simpo_loss + apply_sft_loss).mean()
            
            if do_logging:
                self.log_dict({
                        f"{train_eval}/sft_loss": sft_loss.detach().cpu(),
                        f"apply_loss/simpo_loss": apply_simpo_loss.detach().cpu().mean(),
                        f"apply_loss/sft_loss": apply_sft_loss.detach().cpu().mean(),
                    }, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        else: # 기본
            loss = apply_simpo_loss.mean()
            self.log_dict({
                        f"apply_loss/simpo_loss": apply_simpo_loss.detach().cpu().mean(),
                        f"apply_loss/sft_loss": 0.0,
                    }, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        chosen_rewards = loss_outputs['chosen_rewards']
        rejected_rewards = loss_outputs['rejected_rewards']
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        reward_margins = (chosen_rewards - rejected_rewards)

        if do_logging:
            content = {
                f"{train_eval}/rewards/chosen": chosen_rewards.mean().cpu(),
                f"{train_eval}/rewards/rejected": rejected_rewards.mean().cpu(),
                f"{train_eval}/rewards/accuracies": reward_accuracies.mean().cpu(),
                f"{train_eval}/rewards/margins": reward_margins.mean().cpu(),
                f"{train_eval}/logps/rejected": policy_rejected_logps.detach().mean().cpu(),
                f"{train_eval}/logps/chosen": policy_chosen_logps.detach().mean().cpu(),
                f"{train_eval}/logits/rejected": policy_rejected_logits.detach().mean().cpu(),
                f"{train_eval}/logits/chosen": policy_chosen_logits.detach().mean().cpu(),
                f"{train_eval}/simpo_loss": loss_outputs['simpo_loss'].detach().cpu().mean(),
            }

            self.log_dict(content, on_step=True, prog_bar=True, logger=True, sync_dist=True)

            if self.log_by_category:
                idx2category = {"0": "attribute", "1": "layout", "2": "non-spatial", "3": "complex"}
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
                            f"{train_eval}_{category}/chosen_logps": chosen_logps.detach().mean().cpu(),
                            f"{train_eval}_{category}/rejected_logps": rejected_logps.detach().mean().cpu(),
                            f"{train_eval}_{category}/chosen_rewards": chosen_reward.detach().mean().cpu(),
                            f"{train_eval}_{category}/rejected_rewards": rejected_reward.detach().mean().cpu(),
                            f"{train_eval}_{category}/reward_accuracies": (chosen_reward > rejected_reward).float().mean().cpu(),
                            f"{train_eval}_{category}/reward_margins": (chosen_reward - rejected_reward).mean().cpu(),
                        })

        else: 
            if len(set(task_ids)) == 1: # all same task_id
                task_id = task_ids[0]
                self.log(f"val/task_{task_id}/logps_gap", (policy_chosen_logps - policy_rejected_logps).mean().cpu(), on_step=True, prog_bar=True, logger=True, sync_dist=True)

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
        
        if train_eval == "val":
            print(f"END Rank: {self.global_rank}, Batch: {len(batch)}, task_ids: {batch['task_ids']}")

        return loss


    def compute_loss(
        self,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        with torch.cuda.amp.autocast():
            loss = self.get_batch_loss_metrics(inputs, train_eval="train")

        return loss
    

    def prediction_step(
        self, 
        inputs: Dict[str, Union[torch.Tensor, Any]]
    ):
        prediction_context_manager = torch.cuda.amp.autocast # if self._peft_has_been_casted_to_bf16 else nullcontext
        with torch.no_grad(), prediction_context_manager():
            loss = self.get_batch_loss_metrics(inputs, train_eval="val")
        
        return loss.detach()
        

    def compute_total_grad_norm(self):
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


    def on_train_end(self):
        print("Training END.")
