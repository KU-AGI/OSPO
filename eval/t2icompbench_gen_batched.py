import os
import torch
from peft import LoraConfig, get_peft_model
import numpy as np
import PIL.Image

import random
import time
import yaml
import argparse
from typing import List
import traceback

from pytorch_lightning import LightningModule, seed_everything
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.constant import *
from ospo.utils.common import build_config, save_json_ddp
from ospo.utils.model import get_model
from ospo.utils.generate import *


class JanusProTestWrapper(LightningModule):
    def __init__(self, config, vl_chat_processor, model):
        super().__init__()

        self.config = config
        self.vl_chat_processor = vl_chat_processor
        self.model = model

        self.error_data = []
        self.seed_list = config.task.seed 
        self.parallel_size = config.task.parallel_size
        self.temperature = config.task.temperature
        self.cfg_weight = config.task.cfg_weight


    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        
        # batch_size >= 1 default
        prompt_list = []
        final_path_list = []

        for sample in batch:
            prompt = get_prompt(self.vl_chat_processor, sample['caption'])        
            save_path = os.path.join(self.config.base.save_path, sample['category'])
            os.makedirs(save_path, exist_ok=True)
            fname = f"{sample['caption']}_{sample['idx']}.png"

            final_path = os.path.join(os.path.join(save_path, fname))
        
            if os.path.exists(final_path):
                continue
            else:
                prompt_list.append(prompt)
                final_path_list.append(final_path)
        
        if len(final_path_list) == 0:
            return 

        try:
            for seed in self.seed_list:
                self.generate_batch(
                    prompt_list=prompt_list,
                    save_path_list=final_path_list,
                    seed=seed
                )
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.error_data.append(sample)


    @torch.inference_mode()
    def generate_batch(
        self,
        prompt_list: List,
        save_path_list: List,
        seed: int = None,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

        batch_size = len(prompt_list) 
        input_ids_list = []
        max_len = 0 # for padding

        for prompt in prompt_list:
            input_ids = self.vl_chat_processor.tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids)

            max_len = max(max_len, len(input_ids))
            input_ids_list.append(input_ids)

        tokens = torch.zeros((batch_size*2, max_len), dtype=torch.int).to(self.device)
        attention_masks = torch.ones((batch_size*2, max_len), dtype=torch.long).to(self.device)
    
        for i in range(batch_size*2):
            pad_len = max_len - len(input_ids_list[i//2])
            tokens[i, pad_len:] = input_ids_list[i//2]
            tokens[i, :pad_len] = self.vl_chat_processor.pad_id
            attention_masks[i, :pad_len] = 0
            if i % 2 != 0:
                tokens[i, pad_len+1:-1] = self.vl_chat_processor.pad_id


        # when batch_size = 1, 
        # tokens = torch.zeros((self.parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
        # for i in range(self.parallel_size*2):
        #     tokens[i, :] = input_ids
        #     if i % 2 != 0:
        #         tokens[i, 1:-1] = self.vl_chat_processor.pad_id


        # torch.Size([2, 17, 4096]), bsz = 1
        # torch.Size([2*bsz, max_len, 4096]), bsz > 1
        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

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

        dec = self.model.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[batch_size, 8, img_size//patch_size, img_size//patch_size])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)
        
        visual_img = np.zeros((batch_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec # (1, 384, 384, 3)

        if self.parallel_size == 1:
            # batch_size = 1
            # PIL.Image.fromarray(visual_img[0]).save(save_path) 

            # batch_size > 1 
            for inner_idx, image in enumerate(visual_img):
                try:
                    PIL.Image.fromarray(image).save(save_path_list[inner_idx])
                    
                except OSError:
                    idx_in_path = save_path_list[inner_idx].split("_")[1] # 01.png
                    alternative_path = f"longprompt_{idx_in_path}"

                    # PIL.Image.fromarray(image).save(os.path.join(sample_path_list[inner_idx],f'longname_{idx}.png'))
                    PIL.Image.fromarray(image).save(alternative_path)

        else:
            raise NotImplementedError("parallel_size > 1, not supported.")

            
    def on_test_epoch_end(self):
        print(f"Error case: {len(self.error_data)}")
        save_json_ddp(save_root=self.config.base.save_path,
                      save_name="error_sample",
                      world_size=self.trainer.world_size,
                      save_file=self.error_data,
                      rank=self.trainer.global_rank)


def get_prompt(vl_chat_processor, text):
    conversation = [
        {
            "role": "<|User|>",
            "content": text, # "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag
    return prompt


def main(args):

    config = build_config(cfg_path=args.cfg_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    seed_everything(config.task.seed[0], workers=True)

    if config.base.save_path is not None and config.base.exp_name is not None:
        config.base.save_path = os.path.join(config.base.save_path, config.base.exp_name, config.base.task_name, "gen")
        os.makedirs(config.base.save_path, exist_ok=True)
    else:
        raise ValueError("base.save_path or base.exp_name not provided.")

    vl_chat_processor, tokenizer, model = get_model(mode='generate', config=config)

    if config.model.type == "official":
        model = JanusProTestWrapper(config=config, 
                                    vl_chat_processor=vl_chat_processor, 
                                    model=model)

    # pytorch-lightning trained ckpt
    elif config.model.type == "pl": 
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

        model.language_model = get_peft_model(model.language_model, lora_config)
        model = JanusProTestWrapper.load_from_checkpoint(checkpoint_path=config.model.ckpt_path, 
                                                            config=config, 
                                                            vl_chat_processor=vl_chat_processor, 
                                                            model=model,
                                                            strict=False) 
        model.setup("test")
        model.model.language_model = model.model.language_model.merge_and_unload() 

    trainer = get_trainer(device, config.base.world_size, config.base.precision)
    eval_dataloader = get_dataloader(config, args)

    start = time.time()
    trainer.test(model, dataloaders=eval_dataloader)
    end = time.time()

    elapsed_time = (end - start) / 60  # Convert seconds to minutes
    print(f"Done ! Time elapsed: {elapsed_time:.2f} minutes")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/eval/t2icompbench.yaml")
    parser.add_argument("--category", nargs="+", type=str, default=None, 
                    help="'List' of categories: shape, spatial, color, texture, etc.")
    
    args,unknown = parser.parse_known_args()
    
    main(args)