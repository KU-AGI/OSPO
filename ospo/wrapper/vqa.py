import os
import numpy as np
from glob import glob
from PIL import Image
import torch
from pytorch_lightning import LightningModule

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.common import save_json_ddp
from ospo.utils.processor import get_sft_format
from ospo.templates import get_vqa_prompt


class JanusProQuestionGenWrapper(LightningModule):
    def __init__(self, config, model, tokenizer, processor, mode: str = "base"):
        super().__init__()
        self.config=config
        self.model=model
        self.tokenizer=tokenizer
        self.processor=processor

        self.mode = mode
        assert self.mode in ["base", "negative"], f"Non-allowed mode: {self.mode}"
        self.output_list = []


    def on_test_epoch_start(self):
        self.model.eval() 
    

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        sft_format_list = []

        for sample in batch:
            t2i_category = sample['t2i_category']
            if self.mode == "base":            
                system_prompt, conversation = get_vqa_prompt(t2i_category, sample['prompt'])
                sft_format = get_sft_format(self.processor, system_prompt, conversation)
                sft_format_list.append(sft_format)
            else:
                for n_prompt in sample['negative_prompt']:
                    system_prompt, conversation = get_vqa_prompt(t2i_category, n_prompt)
                    sft_format = get_sft_format(self.processor, system_prompt, conversation)
                    sft_format_list.append(sft_format)


        input_embeds, attention_mask = self.get_input_embeds(sft_format_list)
        outputs = self.generate(input_embeds, attention_mask)

        # decode and save the output
        self.decode(batch, outputs)


    @torch.inference_mode()
    def generate(self, input_embeds, attention_mask):
        generation_config = self.config.generation_config
        outputs = self.model.language_model.generate(inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            **generation_config
            )

        return outputs
    

    def decode(self, batch, outputs):
        if self.mode == "base":
            for sample, output in zip(batch, outputs):
                answer = self.tokenizer.decode(output.cpu().tolist(), skip_special_tokens=True)
                answer = answer.split("Questions: ")[-1]
                questions = [t.strip() + '?' for t in answer.split('?') if t.strip().rstrip('.')]

                # preprocess
                filtered_questions = []
                for q in questions:
                    if "<|User|>" in q or "<|Assistant|>" in q:
                        # error
                        continue
                    filtered_questions.append(q)

                sample['question'] = list(set(filtered_questions))
                self.output_list.append(sample)

        else: # = negative
            idx = 0
            for sample in batch:
                filtered_questions = []
                for _ in sample["negative_prompt"]:
                    decoded = self.tokenizer.decode(outputs[idx].cpu().tolist(), skip_special_tokens=True)
                    questions = {
                        q.strip() + "?"
                        for q in decoded.split("Questions: ")[-1].split("?")
                        if q.strip().rstrip(".")
                    }
                    filtered_questions.append(list(questions))
                    idx += 1

                # sample["negative_short_question"] = filtered_questions
                sample["rejected_question"] = filtered_questions
                self.output_list.append(sample)


    def get_input_embeds(self, sft_formats):
        # for padding
        input_ids_list = []
        seq_length_list = []
        
        for prompt in sft_formats:      
            input_ids = self.tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids)
            input_ids_list.append(input_ids)
            seq_length_list.append(len(input_ids))
            
        max_len = max(seq_length_list)
        batch_size = len(sft_formats)

        tokens = torch.zeros((batch_size, max_len), dtype=torch.int).to(self.device)
        attention_mask = torch.ones((batch_size, max_len), dtype=torch.long).to(self.device)
        for i, (length, input_ids) in enumerate(zip(seq_length_list, input_ids_list)):
            pad_len = max_len - length
            tokens[i, pad_len:] = input_ids
            tokens[i, :pad_len] = self.processor.pad_id
            attention_mask[i, :pad_len] = 0        
        input_embeds = self.model.language_model.get_input_embeddings()(tokens)  
        
        return input_embeds, attention_mask


    def on_test_epoch_end(self):
        save_json_ddp(
            save_root=self.config.save_path,
            save_name=f'vqa_prompt_mode_{self.mode}' if self.config.save_name is None else self.config.save_name,
            world_size=self.trainer.world_size,
            save_file=self.output_list,
            rank=self.trainer.global_rank,
        )
        print("Saved VQA question done.")




class JanusProScoreWrapper(LightningModule):
    def __init__(self, config, model, tokenizer, processor, constraint: int = None, mode: str = "base"):
        super().__init__()
        self.config=config
        self.model=model
        self.tokenizer=tokenizer
        self.processor=processor
    
        self.mode = mode
        assert self.mode in ["base", "negative"], f"Non-allowed mode: {self.mode}"
        self.constraint = constraint

        self.output_list = []
        self.yes_ids = [self.tokenizer("yes", add_special_tokens=False).input_ids[-1],
                        self.tokenizer("Yes", add_special_tokens=False).input_ids[-1]]
        self.no_ids  = [self.tokenizer("no", add_special_tokens=False).input_ids[-1],
                        self.tokenizer("No", add_special_tokens=False).input_ids[-1]]


    def on_test_epoch_start(self):
        self.model.eval() 


    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        if self.mode == "base":
            questions_batched = [sample['question'] for sample in batch]  
            base_paths_batched, negative_paths_batched = [], []

            for sample in batch:
                base_paths = sorted(glob(os.path.join(self.config.image_path, 'base', sample['t2i_category'], sample['item_id'], '*.png')))
                negative_paths = sorted(glob(os.path.join(self.config.image_path, 'negative', sample['t2i_category'], sample['item_id'], '*.png')))
                
                # Top-{constraint} 
                if self.constraint is not None:
                    base_paths = base_paths[:self.constraint]
                    negative_paths = negative_paths[:self.constraint]
                    
                base_paths_batched.append(base_paths)
                negative_paths_batched.append(negative_paths)

            base_img_metadata_batched = self.get_score_single_base(base_paths_batched, questions_batched)
            negative_img_metadata_batched = self.get_score_single_base(negative_paths_batched, questions_batched)

            for sample_idx, (base_dict, negative_dict) in enumerate(zip(base_img_metadata_batched, negative_img_metadata_batched)):
                output = {
                    "item_id": batch[sample_idx]['item_id'],
                    "t2i_category": batch[sample_idx]['t2i_category'],
                    "sub_category": batch[sample_idx]['sub_category'],
                    "question": batch[sample_idx]['question'],
                    "prompt": batch[sample_idx]['prompt'],
                    "base_metadata": base_dict,
                    "negative_metadata": negative_dict,
                }
                self.output_list.append(output)


        elif self.mode == "negative":
            for sample in batch:
                questions = sample["rejected_question"] # sample["negative_short_question"]
                if not isinstance(questions, list):
                    raise ValueError(f"Expected list of questions, got {type(questions)} for item_id={sample['item_id']}.")
                metadata = self.get_score_single_negative(sample["chosen"], questions)

                output = {
                    "item_id": sample["item_id"],
                    "t2i_category": sample["t2i_category"],
                    "sub_category": sample["sub_category"],
                    "prompt": sample["prompt"],
                    "chosen": sample["chosen"],
                    "rejected_prompt": sample["rejected_prompt"], # sample["negative_short_prompt"],
                    "rejected_question": sample["rejected_question"], # sample["negative_short_question"],
                    "rejected_metadata": metadata,
                }
                self.output_list.append(output)


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


    def get_score_single_base(self, img_paths_batched, questions_batched):
        img_metadata_batched = []

        for sample_idx, img_paths in enumerate(img_paths_batched):
            sample_metadata = {}

            for img_idx, img_path in enumerate(img_paths):
                with Image.open(img_path) as img:
                    convs, imgs = self.build_conversation(img, questions_batched[sample_idx])
                    logits = self.forward_single(convs, imgs)  # shape: [num_questions, seq_len, vocab]
                    probs = torch.softmax(logits[:, -1, :], dim=-1)

                    score_sum = 0
                    answer_metadata = []
                    q_count = len(questions_batched[sample_idx])

                    for q_idx in range(q_count):
                        p_yes = max(probs[q_idx, y].item() for y in self.yes_ids)
                        p_no = max(probs[q_idx, n].item() for n in self.no_ids)

                        answer_metadata.append({
                            'p_yes': float(p_yes),
                            'p_no': float(p_no),
                            'answer': 'yes' if p_yes > p_no else ('no' if p_no > p_yes else 'tie')
                        })

                        score_sum += (p_yes - p_no)

                    local_score = score_sum / q_count
                    prefix = 'base' if 'base' in img_path else 'negative'

                    sample_metadata[f'{prefix}_{img_idx}'] = {
                        'path': img_path,
                        'local_score': float(local_score),
                        'answer_metadata': answer_metadata
                    }

            img_metadata_batched.append(sample_metadata)

        return img_metadata_batched
    


    def get_score_single_negative(self, img_path: str, questions_batched: list): # img_path=chosen_img_path
        sample_metadata = {}
        
        # iteration
        for p_idx, question_list in enumerate(questions_batched): # =ptype_idx
            try:
                with Image.open(img_path) as img:
                    convs, imgs = self.build_conversation(img, question_list)
                    logits = self.forward_single(convs, imgs)  # shape: [num_questions, seq_len, vocab]
                    probs = torch.softmax(logits[:, -1, :], dim=-1)

                    score_sum = 0
                    answer_metadata = []
                    q_count = len(question_list)

                    for q_idx in range(q_count):
                        p_yes = max(probs[q_idx, y].item() for y in self.yes_ids)
                        p_no = max(probs[q_idx, n].item() for n in self.no_ids)

                        answer_metadata.append({
                            'p_yes': float(p_yes),
                            'p_no': float(p_no),
                            'answer': 'yes' if p_yes > p_no else ('no' if p_no > p_yes else 'tie')
                        })

                        score_sum += (p_yes - p_no)

                    local_score = score_sum / (q_count)

                    prefix = 'perturb'
                    sample_metadata[f'{prefix}_{p_idx}'] = {
                        # 'path': img_path,
                        'local_score': float(local_score),
                        'answer_metadata': answer_metadata
                    }
            
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
                
        return sample_metadata


    
    def forward_single(self, convs, imgs):
        prepare_list = []
        for conv, img in zip(convs, imgs):
            prepare = self.processor.process_one(
                conversations=conv,
                images=img,
                force_batchify=True
            )
            prepare_list.append(prepare)

        with torch.no_grad():
            batch_inputs = self.processor.batchify(prepare_list).to(self.device)
            inputs_embeds = self.model.prepare_inputs_embeds(**batch_inputs)
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch_inputs.attention_mask
            )

        return outputs.logits


    def on_test_epoch_end(self):
        save_json_ddp(
            save_root=self.config.save_path,
            save_name=f'vqa_result_mode_{self.mode}' if self.config.save_name is None else self.config.save_name,
            world_size=self.trainer.world_size,
            save_file=self.output_list,
            rank=self.trainer.global_rank,
        )
        print("Saved VQA Result dataset done.")

