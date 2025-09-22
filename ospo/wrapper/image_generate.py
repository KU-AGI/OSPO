""" README: Step3. Image Generation """

import os
import traceback
from typing import List
from PIL import Image
import numpy as np
import torch
from pytorch_lightning import LightningModule

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.constant import *
from ospo.utils.common import save_json, set_seed
from ospo.utils.processor import get_conversation, get_sft_format

class JanusProImageGenWrapper(LightningModule):
    def __init__(self, config, model, tokenizer, processor):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.image_start_id = self.tokenizer.vocab.get("<begin_of_image>")
        self.image_end_id = self.tokenizer.vocab.get("<end_of_image>")
        self.image_id = self.tokenizer.vocab.get("<image_placeholder>")
        self.pad_id = self.tokenizer.vocab.get("<｜▁pad▁｜>")

        for k, v in self.config.generation_config.items():
            setattr(self, k, v) 
        

    def on_test_epoch_start(self):
        self.model.eval() 
    

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        try: 
            prompt_lists_by_index = [[] for _ in range(6)]
            final_path_lists_by_index = [[] for _ in range(6)]
            skip_flags = [[] for _ in range(6)]

            for sample in batch:
                item_id = sample['item_id']
                # category = sample['category'] # TODO
                category = sample['t2i_category'] # TODO
                base_prompts = sample['long_prompt']
                negative_prompts = sample['negative_long_prompt']

                for i in range(3): # = 3 perturbation type
                    for prompt_type, prompts, offset in zip(
                        ["base", "negative"],
                        [base_prompts, negative_prompts],
                        [0, 3]):
                        
                        save_path = os.path.join(self.config.save_path, 
                                                    f"{prompt_type}", 
                                                    category, 
                                                    item_id)
                        os.makedirs(save_path, exist_ok=True)
                        save_json(save_path, "metadata", sample)
                        
                        prompt = prompts[i]
                        idx = i + offset  # index from 0-5

                        if prompt == "" or prompt is None:
                            skip_flags[idx].append(True)
                            continue

                        formatted_prompt = self.get_image_generation_prompt(prompt)
                        fname = f"{i:02d}.png" 
                        # fname = get_fname(i, p_type, sub_category) # use number or p_type as a fname 
                        final_path = os.path.join(save_path, fname)

                        if os.path.exists(final_path):
                            skip_flags[idx].append(True)
                            continue

                        prompt_lists_by_index[idx].append(formatted_prompt)
                        final_path_lists_by_index[idx].append(final_path)
                        skip_flags[idx].append(False)

            for i in range(6):  # 0~5: base + negative
                if len(prompt_lists_by_index[i]) == 0:
                    continue  # Nothing to generate

                self.generate_image(
                    prompt_list=prompt_lists_by_index[i],
                    save_path_list=final_path_lists_by_index[i],
                    seed=self.config.seed_list[i % 3]  # repeat 0,1,2 for base/negative
                )

        except Exception as e:
            print(f"Error in test_step: {e}")
            traceback.print_exc()


    def get_image_generation_prompt(self, prompt):
        system_prompt = ""
        converation = get_conversation(prompt)
        sft_format = get_sft_format(self.processor, system_prompt, converation)
        prompt = sft_format + self.processor.image_start_tag

        return prompt


    @torch.inference_mode()
    def generate_image(
        self,
        prompt_list: List,
        save_path_list: List,
        seed: int = None,
        image_token_num_per_image: int = IMAGE_TOKEN_NUM_PER_IMAGE,
        img_size: int = IMG_SIZE,
        patch_size: int = PATCH_SIZE,
    ):
        set_seed(seed)

        batch_size = len(prompt_list) 
        input_ids_list = []
        max_len = 0 # for padding

        for prompt in prompt_list:
            input_ids = self.processor.tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids)

            max_len = max(max_len, len(input_ids))
            input_ids_list.append(input_ids)

        tokens = torch.zeros((batch_size*2, max_len), dtype=torch.int).to(self.device)
        attention_masks = torch.ones((batch_size*2, max_len), dtype=torch.long).to(self.device)
    
        for i in range(batch_size*2):
            pad_len = max_len - len(input_ids_list[i//2])
            tokens[i, pad_len:] = input_ids_list[i//2]
            tokens[i, :pad_len] = self.processor.pad_id
            attention_masks[i, :pad_len] = 0
            if i % 2 != 0:
                tokens[i, pad_len+1:-1] = self.processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)


        # Generate image tokens
        generated_tokens = torch.zeros((batch_size, image_token_num_per_image), dtype=torch.int).cuda()

        for i in range(image_token_num_per_image):
            outputs = self.model.language_model.model(inputs_embeds=inputs_embeds, 
                                                    attention_mask=attention_masks, 
                                                    use_cache=True, 
                                                    past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            
            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + self.cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / self.temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1) # torch.Size([1, 1]) / tensor([[4521]], device='cuda:0')
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

            new_mask = torch.ones((attention_masks.shape[0], 1), dtype=attention_masks.dtype, device=attention_masks.device)  
            attention_masks = torch.cat([attention_masks, new_mask], dim=1)

        # Decode generated tokens
        dec = self.model.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[batch_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        
        visual_img = np.zeros((batch_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec # (1, 384, 384, 3)

        # Save generated images
        # for batch_size > 1 
        for inner_idx, image in enumerate(visual_img):
            try:
                Image.fromarray(image).save(save_path_list[inner_idx])
                
            except OSError:
                idx_in_path = save_path_list[inner_idx].split("_")[1] # 01.png
                alternative_path = f"longprompt_{idx_in_path}"
                Image.fromarray(image).save(alternative_path)

