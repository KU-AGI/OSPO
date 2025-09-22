import os
import argparse
import json
import datetime
import yaml

import numpy as np
import PIL.Image
import torch

from peft import LoraConfig, get_peft_model
from pytorch_lightning import LightningModule, seed_everything

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.constant import *
from ospo.utils.common import build_config
from ospo.utils.model import get_model
from ospo.utils.generate import *


class JanusTestWrapper(LightningModule):
    def __init__(self, config, model, text_tokenizer, processor):
        super().__init__()

        self.model = model
        self.text_tokenizer = text_tokenizer
        self.processor = processor
        self.config = config

        self.n_samples = PARALLEL_SIZE
        self.temperature = config.task.temperature
        self.cfg_weight = config.task.cfg_weight

    def on_test_epoch_start(self):
        self.model.eval() 
    
    @torch.inference_mode()
    def test_step(self, batch, batch_idx):

        prompt_list = []
        data_idx_list = []
        sample_path_list = []

        for sample, data_idx in batch:
            prompt = sample['prompt']
            outpath = os.path.join(self.config.base.save_path, f"{data_idx:05d}")
            os.makedirs(outpath, exist_ok=True)

            sample_path = os.path.join(outpath, "samples")
            os.makedirs(sample_path, exist_ok=True)

            if os.path.exists(sample_path) and len(os.listdir(sample_path)) == 4:
                continue

            prompt_list.append(prompt)
            data_idx_list.append(data_idx)
            sample_path_list.append(sample_path)

            with open(os.path.join(outpath, "metadata.jsonl"), "w") as f:
                json.dump(sample, f)

        if len(prompt_list) == 0:
            print("Already generated.")
            return

        system_prompt = ''
        input_embeds, attention_masks = self.prepare_input_embeds(prompt_list, system_prompt)
       
        img_array = []        
        for idx in range(4): # hard-coding
            seed_everything(idx)
            generated_tokens = self.generate(input_embeds, attention_masks)

            images = self.decode_tokens(generated_tokens)
            img_array.append(images)

            for inner_idx, image in enumerate(images):
                fpath = os.path.join(sample_path_list[inner_idx],f"{idx:05}.png")
                PIL.Image.fromarray(image).save(fpath)



    @torch.inference_mode()
    def generate(self, inputs_embeds, attention_masks):
        
        batch_size = inputs_embeds.shape[0] // 2

        generated_tokens = torch.zeros((batch_size, IMAGE_TOKEN_NUM_PER_IMAGE), dtype=torch.int).to(self.device)
        
        for i in range(IMAGE_TOKEN_NUM_PER_IMAGE):
            outputs = self.model.language_model.model(inputs_embeds=inputs_embeds, attention_mask=attention_masks, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state                                                               # hidden_states [32, seq_len, 4096] if i == 0 else [32, 1, 4096]
            # hidden_states = outputs.hidden_states[-1]     
        
            logits = self.model.gen_head(hidden_states[:, -1, :])                                                   # logits [32, 16384]
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + self.cfg_weight * (logit_cond-logit_uncond)                                     # logits [16, 16384]
            probs = torch.softmax(logits / self.temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)                                                    # next_token [16, 1]
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)      # next_token [32]
            img_embeds = self.model.prepare_gen_img_embeds(next_token)                                              # img_embeds [32, 4096]
            inputs_embeds = img_embeds.unsqueeze(dim=1)                                                             # inputs_embeds [32, 1, 4096]

            new_mask = torch.ones((attention_masks.shape[0], 1), dtype=attention_masks.dtype, device=attention_masks.device)  
            attention_masks = torch.cat([attention_masks, new_mask], dim=1)

        return generated_tokens        
    

    def decode_tokens(self, generated_tokens):
        batch_size = generated_tokens.shape[0]
        dec = self.model.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[batch_size, 8, IMG_SIZE//PATCH_SIZE, IMG_SIZE//PATCH_SIZE])    # dec [16, 3, 384, 384]
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)                                                                                           # dec [16, 384, 384, 3]

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        return visual_img

    def prepare_input_embeds(self, prompt_list, system_prompt):

        input_ids_list = []
        max_len = 0
        batch_size = len(prompt_list)

        for prompt in prompt_list:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": prompt,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processor.sft_format,
                system_prompt=system_prompt,
            )
            prompt = sft_format + self.processor.image_start_tag

            input_ids = self.processor.tokenizer.encode(prompt)
            input_ids = torch.LongTensor(input_ids)
            
            max_len = max(max_len, len(input_ids))
            input_ids_list.append(input_ids)

        tokens = torch.zeros((batch_size*2, max_len), dtype=torch.int).to(self.device)
        attention_mask = torch.ones((batch_size*2, max_len), dtype=torch.long).to(self.device)

        for i in range(batch_size*2):
            pad_len = max_len - len(input_ids_list[i//2])
            tokens[i, pad_len:] = input_ids_list[i//2]
            tokens[i, :pad_len] = self.processor.pad_id
            attention_mask[i, :pad_len] = 0
            if i % 2 != 0:
                tokens[i, pad_len+1:-1] = self.processor.pad_id
        input_embeds = self.model.language_model.get_input_embeddings()(tokens)  
        
        return input_embeds, attention_mask



def main(args):
    
    config = build_config(cfg_path=args.cfg_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(config.task.seed[0], workers=True)

    if config.base.save_path is not None and config.base.exp_name is not None:
        config.base.save_path = os.path.join(config.base.save_path, config.base.exp_name, config.base.task_name, "gen")
        os.makedirs(config.base.save_path, exist_ok=True)
    else:
        raise ValueError("base.save_path or base.exp_name not provided.")


    # model, text_tokenizer, text_processor = load_model(config)
    vl_chat_processor, tokenizer, model = get_model(mode='generate', config=config)

    if config.model.type == "official":
        model = JanusTestWrapper(config=config, 
                                model=model,
                                text_tokenizer=tokenizer,
                                processor=vl_chat_processor, 
                                ) 
        model.setup("test")


    elif config.model.type == "hf":
        raise NotImplementedError("Do not support HF (loaded) ckpt")


    # # pytorch-lightning trained ckpt
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
        model = JanusTestWrapper.load_from_checkpoint(checkpoint_path=config.model.ckpt_path, 
                                                            config=config, 
                                                            model=model,
                                                            text_tokenizer=tokenizer,
                                                            processor=vl_chat_processor, 
                                                            strict=False) 
        model.setup("test")
        model.model.language_model = model.model.language_model.merge_and_unload() 

    # Define Trainer
    trainer = get_trainer(device, config.base.world_size, config.base.precision)
    eval_dataloader = get_dataloader(config, args)

    # Start evaluation
    start_time = datetime.datetime.now()
    trainer.test(model, dataloaders=eval_dataloader)
    end_time = datetime.datetime.now()

    elapsed_time = end_time - start_time
    elapsed_min = elapsed_time.total_seconds() / 60

    print('------------------------------------------')
    print(f"Elapsed Time: {elapsed_min:.2f} minutes")
    print('------------------------------------------')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="configs/eval/geneval.yaml")
    parser.add_argument("--s_idx", type=int, default=None)
    parser.add_argument("--e_idx", type=int, default=None)
    args,unknown = parser.parse_known_args()
    
    main(args)