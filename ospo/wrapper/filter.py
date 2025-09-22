import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pytorch_lightning as pl
from trl.trainer.utils import pad_to_length 
from typing import *
from PIL import Image
from collections import defaultdict
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.constant import *
from ospo.utils.common import read_json, save_json_ddp
from ospo.utils.processor import get_conversation, get_sft_format


class JanusProFilterWrapper(pl.LightningModule):
    """ This is for filtering False Positive (for chosen) or False Negative (for rejected) image."""

    def __init__(self, config, model, chat_processor, image_processor, tokenizer, mode="base"):
        self.model = model
        self.chat_processor = chat_processor
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        self.mode = mode # "IMAGE" # img / txt
        self.max_prompt_length = config.tokenizer.max_length
        self.max_length = config.tokenizer.max_prompt_length
        self.padding_value = config.tokenizer.label_pad_token_id
        self.label_pad_token_id = config.tokenizer.padding_value

        self.output_list = []

        
    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        for sample in batch:
            try:
                if self.mode == "base":
                    chosen_candidate = list(sample['base_metadata'].values())
                    rejected_candidate = list(sample['negative_metadata'].values())

                    filtered_chosen_candidate = self._filter_chosen(chosen_candidate)
                    filtered_rejected_candidate = self._filter_rejected(rejected_candidate)

                    if len(filtered_chosen_candidate) == 0 or len(filtered_rejected_candidate) == 0:
                        print(f"[SKIP] The sample # {sample['item_id']} does not have enough sample after filtering.")
                        continue

                    # choose chosen / rejected (image)
                    chosen = self._select_final_chosen(filtered_chosen_candidate)
                    rejected_image_sample = self._select_final_rejected(chosen, filtered_rejected_candidate)
                
                    sample["chosen"] = chosen["path"]
                    sample["rejected"] = rejected_image_sample["path"]
                    self.output_list.append(sample)


                elif self.mode == "negative":
                    rejected_candidate = list(sample['rejected_metadata'].values())
                    filtered_rejected_candidate = self._filter_rejected(rejected_candidate)
                    
                    if len(filtered_rejected_candidate) == 0:
                        print(f"[SKIP] The sample # {sample['item_id']} does not have enough sample after filtering.")
                        continue

                    # choose rejected (prompt)
                    chosen = sample["chosen"] # Image Path; It is already selected and given.
                    rejected_prompt_sample = self._select_final_rejected(chosen, filtered_rejected_candidate)
            
                    sample["rejected_prompt"] = rejected_prompt_sample["path"]
                    self.output_list.append(sample)


            except Exception as e:
                print(e)
                continue


    def on_test_epoch_end(self):
        save_json_ddp(
            save_root=self.config.save_path,
            save_name=f'train_data_mode_{self.mode}' if self.config.save_name is None else self.config.save_name,
            world_size=self.trainer.world_size,
            save_file=self.output_list,
            rank=self.trainer.global_rank,
        )
        print(f"Filtering (mode: {self.mode}) done.")


    def _select_final_chosen(self, filtered_chosen_candidate):
        return max(filtered_chosen_candidate, key=lambda x: x['local_score']) 
        
    
    def _select_final_rejected(self, chosen, filtered_rejected_candidate):
        # 1) Build forward batch (with stable tmp_ids) in one go
        if self.mode == "base":
            prepare = [
                {
                    "item_id": f"{r['item_id']}{i}",   # tmp_id
                    "t2i_category": r["t2i_category"],
                    "sub_category": r["sub_category"],
                    "prompt": r["prompt"],
                    "chosen": chosen["path"],
                    "rejected": r["path"],
                }
                for i, r in enumerate(filtered_rejected_candidate)
            ]
        elif self.mode == "negative":
            prepare = [
                {
                    "item_id": f"{r['item_id']}{i}",   # tmp_id
                    "t2i_category": r["t2i_category"],
                    "sub_category": r["sub_category"],
                    "prompt": r["prompt"],
                    "chosen": chosen["path"],
                    "rejected": r["path"],
                    "rejected_prompt": r["rejected_prompt"]
                }
                for i, r in enumerate(filtered_rejected_candidate)
            ]

        # 2) Run model
        outputs = self._forward(prepare)
        if len(outputs) != len(prepare):
            raise ValueError("Number of outputs differs in selecting rejected!")

        # 3) Map tmp_id -> logps_gap (order-independent)
        gaps = {o["item_id"]: o["logps_gap"] for o in outputs}

        # 4) Attach logps_gap back to original candidates
        for i, r in enumerate(filtered_rejected_candidate):
            tmp_id = f"{r['item_id']}{i}"
            if tmp_id not in gaps:
                raise ValueError(f"Missing model output for tmp_id={tmp_id}")
            r["logps_gap"] = gaps[tmp_id]

        # 5) Pick candidate with smallest |gap|
        return min(filtered_rejected_candidate, key=lambda x: abs(x["logps_gap"]))


    def _filter_chosen(self, chosen_candidate, filtering_threshold: float = 0.5):
        filtered = []
        for i, candidate in enumerate(chosen_candidate):
            xx = True
            answer_metadata = candidate['answer_metadata']
            for m in answer_metadata:
                if m["answer"] == "no":
                    xx = False
                    break
                elif m["p_yes"] - m["p_no"] <= filtering_threshold: 
                    xx = False
                    break
                else:
                    pass

            # True Positive
            if xx:
                filtered.append(candidate) 

        return filtered


    def _filter_rejected(self, rejected_candidate, filtering_threshold: float = 0.5):
        filtered = []
        for i, candidate in enumerate(rejected_candidate):
            xx = False
            answer_metadata = candidate['answer_metadata']

            for m in answer_metadata:
                if m["p_no"] - m["p_yes"] > filtering_threshold: 
                    xx = True
                    break

            # True Negative
            if xx:
                filtered.append(candidate) 

        return filtered


    def _get_image_generation_prompt(self, prompt):
        system_prompt = ""
        converation = get_conversation(prompt)
        sft_format = get_sft_format(self.chat_processor, system_prompt, converation)
        prompt = sft_format + self.chat_processor.image_start_tag 

        return prompt


    def _get_text_token(self, text):
        parallel_size=1 
        prompt = self._get_image_generation_prompt(text)
        text_input_ids = self.tokenizer.encode(prompt)  
        text_input_ids = torch.LongTensor(text_input_ids) # e.g. torch.Size([18])

        text_tokens = torch.zeros((parallel_size, len(text_input_ids)), dtype=torch.int) 
        for i in range(parallel_size):
            text_tokens[i, :] = text_input_ids 

        return text_tokens


    def _get_image_tensor(self, img_path: str):
        image = Image.open(img_path)
        image_tensor = self.image_processor([image])
        image_tensor = image_tensor['pixel_values']  # e.g. torch.Size([1, 3, 384, 384])
       
        return image_tensor


    def _decode_by_image(self, example: Dict):
        if "prompt" not in example.keys() or "chosen" not in example.keys() or "rejected" not in example.keys():
            raise ValueError(f"[SimPO pair] Given example has wrong format: {example.keys()}")

        item_id = example["item_id"]        
        text_token = self._get_text_token(example["prompt"]) 
        chosen_image_tensor = self._get_image_tensor(example["chosen"])
        rejected_image_tensor = self._get_image_tensor(example["rejected"])

        return item_id, text_token, chosen_image_tensor, rejected_image_tensor
    
    def _decode_by_text(self, example: Dict):
        if "prompt" not in example.keys() or "rejected_prompt" not in example.keys() or "chosen" not in example.keys():
            raise ValueError(f"[CondPO pair] Given example has wrong format: {example.keys()}")

        item_id = example["item_id"]        
        chosen_text_token = self._get_text_token(example["prompt"]) 
        rejected_text_token = self._get_text_token(example["rejected_prompt"])

        chosen_image_tensor = self._get_image_tensor(example["chosen"])
        # rejected_image_tensor = self._get_image_tensor(example["rejected"])

        # return item_id, chosen_text_token, rejected_text_token, chosen_image_tensor, rejected_image_tensor
        return item_id, chosen_text_token, rejected_text_token, chosen_image_tensor
    

    def _preprocess_batch(self, batch: List): # Tuple):
        if self.mode == "base": # for simPO pair
            item_ids, chosen_text_tokens, chosen_imgs, rejected_imgs = zip(
                *[self._decode_by_image(sample) for sample in batch]
            )

            batched_chosen_text = self._preprocess_batch_text(list(chosen_text_tokens))
            chosen_img_embeds, chosen_img_labels = self._preprocess_batch_image(list(chosen_imgs))
            rejected_img_embeds, rejected_img_labels = self._preprocess_batch_image(list(rejected_imgs))

            # chosen / rejected sequence (txt + img)
            chosen_seqs, rejected_seqs = [], []
            chosen_labels_all, rejected_labels_all = [], []
            chosen_masks, rejected_masks = [], []

            for txt, c_img, r_img, c_lab, r_lab in zip(
                batched_chosen_text, chosen_img_embeds, rejected_img_embeds, chosen_img_labels, rejected_img_labels
            ):
                # 1) Truncate prompt first
                txt = self._truncate(txt, self.max_prompt_length, dim=0)
                txt_len = txt.shape[0]

                # 2) Figure how many image tokens can fit (avoid concat-then-truncate)
                remaining = max(self.max_length - txt_len, 0)
                c_img = self._truncate(c_img, remaining, dim=0)
                r_img = self._truncate(r_img, remaining, dim=0)

                # 3) Concat (now guaranteed <= max_length)
                c_seq = torch.cat([txt, c_img], dim=0)
                r_seq = torch.cat([txt, r_img], dim=0)
                chosen_seqs.append(c_seq)
                rejected_seqs.append(r_seq)

                # 4) Build labels: pad for text, then append image labels; no need to truncate again
                pad_txt = torch.full(
                    (txt_len,), self.label_pad_token_id, dtype=torch.long, device=self.device
                )
                c_seq_lab = torch.cat([pad_txt, self._truncate(c_lab, remaining, dim=0)], dim=0)
                r_seq_lab = torch.cat([pad_txt, self._truncate(r_lab, remaining, dim=0)], dim=0)
                chosen_labels_all.append(c_seq_lab)
                rejected_labels_all.append(r_seq_lab)

                # 5) Attention masks (1s for all real tokens)
                chosen_masks.append(torch.ones(c_seq_lab.shape[0], dtype=torch.long, device=self.device))
                rejected_masks.append(torch.ones(r_seq_lab.shape[0], dtype=torch.long, device=self.device))

        
        elif self.mode == "negative": # for conPO pair
            # item_ids, chosen_text_tokens, rejected_text_tokens, chosen_imgs, rejected_imgs = zip(
            #     *[self._decode_by_text(sample) for sample in batch]
            # )
            item_ids, chosen_text_tokens, rejected_text_tokens, chosen_imgs = zip(
                *[self._decode_by_text(sample) for sample in batch]
            )

            batched_chosen_text = self._preprocess_batch_text(list(chosen_text_tokens))
            batched_rejected_text = self._preprocess_batch_text(list(rejected_text_tokens))

            chosen_img_embeds, chosen_img_labels = self._preprocess_batch_image(list(chosen_imgs))
            # rejected_img_embeds, rejected_img_labels = self._preprocess_batch_image(list(rejected_imgs))


            # chosen / rejected sequence (txt + img)
            # Here 'rejected' actually means 'swap_chosen'.
            chosen_seqs, rejected_seqs = [], []
            chosen_labels_all, rejected_labels_all = [], []
            chosen_masks, rejected_masks = [], []

            for c_txt, r_txt, c_img, c_lab in zip(
                batched_chosen_text, batched_rejected_text, chosen_img_embeds, chosen_img_labels
            ):
                # 1) Truncate prompt first
                c_txt = self._truncate(c_txt, self.max_prompt_length, dim=0)
                r_txt = self._truncate(r_txt, self.max_prompt_length, dim=0)
                c_txt_len = c_txt.shape[0]
                r_txt_len = r_txt.shape[0]

                # 2) Figure how many image tokens can fit (avoid concat-then-truncate)
                remaining = max(self.max_length - c_txt_len, 0)          
                c_img = self._truncate(c_img, remaining, dim=0)

                # 3) Concat (now guaranteed <= max_length)
                c_seq = torch.cat([c_txt, c_img], dim=0)
                r_seq = torch.cat([r_txt, c_img], dim=0)
                chosen_seqs.append(c_seq)
                rejected_seqs.append(r_seq)

                # 4) Build labels: pad for text, then append image labels; no need to truncate again
                c_pad_txt = torch.full(
                    (c_txt_len,), self.label_pad_token_id, dtype=torch.long, device=self.device
                )
                r_pad_txt = torch.full(
                    (r_txt_len,), self.label_pad_token_id, dtype=torch.long, device=self.device
                )

                c_seq_lab = torch.cat([c_pad_txt, self._truncate(c_lab, remaining, dim=0)], dim=0)
                r_seq_lab = torch.cat([r_pad_txt, self._truncate(c_lab, remaining, dim=0)], dim=0)
                chosen_labels_all.append(c_seq_lab)
                rejected_labels_all.append(r_seq_lab)

                # 5) Attention masks (1s for all real tokens)
                chosen_masks.append(torch.ones(c_seq_lab.shape[0], dtype=torch.long, device=self.device))
                rejected_masks.append(torch.ones(r_seq_lab.shape[0], dtype=torch.long, device=self.device))


        # Pad to max length in batch
        padded_chosen_inputs_embeds   = self._pad_to_max_len_in_batch(chosen_seqs,         pad_value=self.padding_value)
        padded_rejected_inputs_embeds = self._pad_to_max_len_in_batch(rejected_seqs,       pad_value=self.padding_value)
        padded_chosen_labels          = self._pad_to_max_len_in_batch(chosen_labels_all,   pad_value=self.label_pad_token_id)
        padded_rejected_labels        = self._pad_to_max_len_in_batch(rejected_labels_all, pad_value=self.label_pad_token_id)
        padded_chosen_attention_mask  = self._pad_to_max_len_in_batch(chosen_masks,        pad_value=0)
        padded_rejected_attention_mask= self._pad_to_max_len_in_batch(rejected_masks,      pad_value=0)

        return {
            "item_ids": list(item_ids),
            "chosen_inputs_embeds": padded_chosen_inputs_embeds,
            "chosen_labels": padded_chosen_labels,
            "chosen_attention_mask": padded_chosen_attention_mask,
            "rejected_inputs_embeds": padded_rejected_inputs_embeds,
            "rejected_labels": padded_rejected_labels,
            "rejected_attention_mask": padded_rejected_attention_mask,
        }

    

    def _preprocess_batch_text(self, seq_list: List):
        batched=[]
        for seq_ts in seq_list:
            text_embeds = self.model.language_model.get_input_embeddings()(seq_ts)
            text_embeds = text_embeds.squeeze(0)
            batched.append(text_embeds) 

        return batched # List of torch.Size([seq_len, 4096])
    

    def _preprocess_batch_image(self, seq_list: List):
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


    def _truncate(self, tensor, max_len, dim=1):
        if tensor.size(dim) > max_len:
            print("Truncating tensor along dim", dim)
            return tensor.narrow(dim, 0, max_len)  # truncates along dim

        return tensor
    

    def _pad_to_max_len_in_batch(self, tensor_list, pad_value=0.0):
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


    def _concatenate(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
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
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)

        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k:
                    pad_value = self.label_pad_token_id
                elif k.endswith("_inputs_embeds"):
                    pad_value = self.padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=self.device) # .to(device=self.device)

        return concatenated_batch
            
            
    def _get_batch_logps(
        self,
        len_chosen: int,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = True,
    ) -> torch.FloatTensor:
        
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")
        
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        assert logits.shape[:-1] == labels.shape, f"Logits shape: {logits.shape}, Labels shape: {labels.shape}"
        
        loss_mask = labels != self.label_pad_token_id
        labels[labels == self.label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        all_logps = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1) if average_log_prob else (per_token_logps * loss_mask).sum(-1)
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        
        return chosen_logps, rejected_logps


    def _forward(self, prepare_list: List):
        batch = self._preprocess_batch(prepare_list)
        # output = self._forward(batch)
        concatenated_batch = self._concatenate(batch)
        len_chosen = batch["chosen_labels"].shape[0]
        outputs = self.model.language_model.model(inputs_embeds=concatenated_batch["concatenated_inputs_embeds"], 
                                                    use_cache=False, 
                                                    past_key_values=None) 
        
        hidden_states = outputs.last_hidden_state
        all_logits = self.model.gen_head(hidden_states)     

        chosen_logps, rejected_logps = self._get_batch_logps(
            len_chosen,
            logits=all_logits,
            labels=concatenated_batch["concatenated_labels"],
            average_log_prob=True,
        )

        forwarded_list = []
        for i, tmp_id in enumerate(batch["item_ids"]):
            forwarded_list.append({
                "item_id": tmp_id,
                "chosen_logps": chosen_logps[i].detach().cpu().numpy().tolist(),
                "rejected_logps": rejected_logps[i].detach().cpu().numpy().tolist(),
                "logps_gap": (chosen_logps[i] - rejected_logps[i]).detach().cpu().numpy().tolist()
            })

        return forwarded_list
        