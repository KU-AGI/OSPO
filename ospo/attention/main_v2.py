# conda activate janus3
# CUDA_VISIBLE_DEVICES=6 python /home/yjoh/project/OSPO/ospo/attention/main_v2.py

import os
import ast
import re
import spacy
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from trl.trainer.utils import pad_to_length 
import pyrootutils
from typing import *

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from janus.models import MultiModalityCausalLM, VLChatProcessor
from ospo.utils.processor import get_conversation, get_sft_format
from ospo.utils.common import read_json, save_json, set_seed, build_config
from ospo.utils.model import get_model, get_lora_config
# pip install scikit-image
from ospo.attention.visualize import visualize_save_attn
from ospo.attention.extract_mask import extract_target_span, process_attn, save_binary_mask

os.environ["TOKENIZERS_PARALLELISM"] = "false"

nlp = spacy.load("en_core_web_sm")


def save_attn_npy(attn_map_grid, save_path):
    # save_path = f"attn_maps/{item_id}_attn.npy"
    np.save(save_path, attn_map_grid.cpu().numpy())


""" Phase 1 """
# No Flash Attention & Gradient Checkpointing


def forward(model, inputs_embeds, attention_mask):
    if len(inputs_embeds.shape) == 2:
        inputs_embeds = inputs_embeds.unsqueeze(0) # bsz 추가
    if len(attention_mask.shape) == 1:
        attention_mask = attention_mask.unsqueeze(0)

    # attention_outputs 을 얻기 위해서는, gradient_checkpointing=False && flash_attetion=False 필수
    outputs = model.language_model.model(inputs_embeds=inputs_embeds,       # concatenated_batch["concatenated_inputs_embeds"], 
                                            attention_mask=attention_mask,  # concatenated_batch["concatenated_attention_mask"],
                                            output_hidden_states=True,
                                            output_attentions=True, # 추가
                                            return_dict=True,
                                            )   
    # print(model.language_model.model.forward.__code__.co_filename)
    return outputs


def get_text_token(chat_processor, tokenizer, text):

    def get_image_generation_prompt(chat_processor, prompt):
        system_prompt = ""
        converation = get_conversation(prompt)
        sft_format = get_sft_format(chat_processor, system_prompt, converation)
        prompt = sft_format + chat_processor.image_start_tag

        return prompt

    parallel_size=1 
    prompt = get_image_generation_prompt(chat_processor, text)
    text_input_ids = tokenizer.encode(prompt)
    text_input_ids = torch.LongTensor(text_input_ids) # e.g. torch.Size([18])

    text_tokens = torch.zeros((parallel_size, len(text_input_ids)), dtype=torch.int) 
    for i in range(parallel_size):
        text_tokens[i, :] = text_input_ids 

    return text_tokens

def get_image_tensor(image_processor, img_path: str):
    image = Image.open(img_path)
    image_tensor = image_processor([image])
    image_tensor = image_tensor['pixel_values']  # e.g. torch.Size([1, 3, 384, 384])
    
    return image_tensor

def get_input_embeddings(model, chat_processor, image_processor, tokenizer, example: dict):
    # tokenize, process
    text_token = get_text_token(chat_processor, tokenizer, example["prompt"]).cuda()
    chosen_image_tensor = get_image_tensor(image_processor, example["chosen"]).cuda()
    rejected_image_tensor = get_image_tensor(image_processor, example["rejected"]).cuda()

    # get input embedding (language model)
    # print("text token: ", text_token)
    # print(len(text_token[0]))
    text_input_ids = text_token[0]

    # target token 찾기

    text_embeds = model.language_model.get_input_embeddings()(text_token)
    text_embeds = text_embeds.squeeze(0)

    expected_dtype = next(model.gen_vision_model.parameters()).dtype   
    if chosen_image_tensor.dtype != expected_dtype:
        chosen_image_tensor = chosen_image_tensor.to(dtype=expected_dtype)
    if rejected_image_tensor.dtype != expected_dtype:
        rejected_image_tensor = rejected_image_tensor.to(dtype=expected_dtype)
                        
    chosen_output = model.gen_vision_model.encode(chosen_image_tensor.to('cuda')) 
    rejected_output = model.gen_vision_model.encode(rejected_image_tensor.to('cuda')) 
    
    # img_tokens [576]
    chosen_img_tokens = chosen_output[2][2]  
    rejected_img_tokens = rejected_output[2][2]  

    # get input embedding (gen_vision_model)
    chosen_img_embeds = model.prepare_gen_img_embeds(chosen_img_tokens).to('cuda')
    rejected_img_embeds = model.prepare_gen_img_embeds(rejected_img_tokens).to('cuda')

    return text_input_ids, text_embeds, chosen_img_embeds, rejected_img_embeds

def prepare_input(model, chat_processor, image_processor, tokenizer, example: dict):
    text_input_ids, text_embeds, chosen_img_embeds, rejected_img_embeds = get_input_embeddings(model, 
                                                                            chat_processor, 
                                                                            image_processor, 
                                                                            tokenizer, 
                                                                            example)

    # No padding is needed.
    chosen_input_embeds = torch.cat([text_embeds, chosen_img_embeds], dim=0)
    rejected_input_embeds = torch.cat([text_embeds, rejected_img_embeds], dim=0)         

    chosen_attention_mask = torch.ones(chosen_input_embeds.shape[0], dtype=torch.long, device='cuda')
    rejected_attention_mask = torch.ones(rejected_input_embeds.shape[0], dtype=torch.long, device='cuda')

    # print("chosen_input_embeds: ", chosen_input_embeds.shape)
    # print("text_embeds: ", text_embeds.shape) # text_embeds:  torch.Size([13, 4096])
    # print("chosen_img_embeds: ", chosen_img_embeds.shape) # chosen_img_embeds:  torch.Size([576, 4096])

    # # 배치 사이즈 넣어주기
    # chosen_input_embeds = chosen_input_embeds.unsqueeze(0)
    # rejected_input_embeds = rejected_input_embeds.unsqueeze(0)
    # chosen_attention_mask = chosen_attention_mask.unsqueeze(0)
    # rejected_attention_mask = rejected_attention_mask.unsqueeze(0)

    # batch = {
    #     "text_input_ids": text_input_ids,
    #     "chosen_inputs_embeds": chosen_input_embeds,
    #     "rejected_inputs_embeds": rejected_input_embeds,
    #     "chosen_attention_mask": chosen_attention_mask,
    #     "rejected_attention_mask": rejected_attention_mask
    # }
    # return batch

    return text_input_ids, chosen_input_embeds, rejected_input_embeds, chosen_attention_mask, rejected_attention_mask

def concatenated_inputs(config, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    concatenated_batch = {}
    max_length = max(batch["chosen_inputs_embeds"].shape[1], batch["rejected_inputs_embeds"].shape[1])

    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            if "labels" in k:
                pad_value = config.experiment.label_pad_token_id
            elif k.endswith("_inputs_embeds"):
                pad_value = config.experiment.padding_value
            elif k.endswith("_attention_mask"):
                pad_value = 0
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)

    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            if "labels" in k:
                pad_value = config.experiment.label_pad_token_id
            elif k.endswith("_inputs_embeds"):
                pad_value = config.experiment.padding_value
            elif k.endswith("_attention_mask"):
                pad_value = 0
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            ).to(device='cuda')

    return concatenated_batch
    
def concatenated_forward(
       config, batch: Dict[str, Union[List, torch.LongTensor]], model
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

    concatenated_batch = concatenated_inputs(config=config, batch=batch)
    # len_chosen = batch["chosen_labels"].shape[0]
    len_chosen = batch["chosen_attention_mask"].shape[0]
    # print(concatenated_batch["concatenated_inputs_embeds"].shape)
    assert len_chosen == 1, f"len_chosen (batch size) must be 1, but you give {len_chosen}"

    outputs = model.language_model.model(inputs_embeds=concatenated_batch["concatenated_inputs_embeds"], 
                                            attention_mask=concatenated_batch["concatenated_attention_mask"],
                                            output_hidden_states=True,
                                            output_attentions=True, # 추가
                                            return_dict=True,
                                            ) 

    chosen_outputs = outputs[:len_chosen]
    rejected_outputs = outputs[len_chosen:]

    chosen_outputs = chosen_outputs[0]
    rejected_outputs = rejected_outputs[0]

    return chosen_outputs, rejected_outputs


def main(config, train_data, save_attn: bool = False):

    set_seed(config.get('seed', 42)) 

    # 따로 생성할 때에는, 반드시 "train" 모드
    # 본 코드 실행 전 체크포인트 머지 작업 진행할 것.
    dtype = torch.bfloat16
    model, chat_processor, image_processor, tokenizer = get_model(mode='train', dtype=dtype, config=config)
    print("Model Loading Done.")

    # Set to train mode()
    model = model.cuda()
    model.train()        
    model.language_model.model.config.output_hidden_states = True
    model.language_model.model.config.output_attentions = True
    print("Model Setting Done.")


    for example in tqdm(train_data, desc="Generating attention-based object mask..."):
        try:
            # chosen, rejected image both
            item_id = example['item_id']
            c_path = example['chosen']
            r_path = example['rejected']

            target_word_list = example['nouns'] 
            
            # 1. Prepare input embeddings
            prompt_input_ids, chosen_input_embeds, rejected_input_embeds, chosen_attention_mask, rejected_attention_mask = \
                prepare_input(model, chat_processor, image_processor, tokenizer, example)
            
            prompt_len = len(prompt_input_ids)
            # print("Input Preparing Done.")

            # print(chosen_input_embeds.shape, chosen_attention_mask.shape) 
            # torch.Size([589, 4096]) torch.Size([589])


            # 2. Forward
            # c_outputs, r_outputs = concatenated_forward(config, batch, model)
            # print(c_outputs.keys())
            # c_attns, r_attns = c_outputs.attentions, r_outputs.attentions
            
            # NO GRAD
            with torch.no_grad():
                c_outputs = forward(model, chosen_input_embeds, chosen_attention_mask)
                c_attns = c_outputs.attentions

                r_outputs = forward(model, rejected_input_embeds, rejected_attention_mask)
                r_attns = r_outputs.attentions

            # print("Forward Done.\n")
            
            for key in ['chosen', 'rejected']:
                img_path = example[key]
                
                attns = c_attns if key == 'chosen' else r_attns

                if save_attn:
                    save_part = img_path.split('/')[-4:-1] # [negative, attribute, 0000002]
                    save_dir = os.path.join(config.save_dir, save_part[0], save_part[1], save_part[2]) # chosen / rejected 각각 다른 곳에 저장되는 구조
                    os.makedirs(save_dir, exist_ok=True) 

                    save_name = os.path.basename(img_path).split(".png")[0] # 01
                
                # initialize empty union mask
                union_binary_mask = None
                union_attn_map = None  # (optional: for visualization)

                for target_word in target_word_list:
                    # 2. Grid in Localizing
                    span_indices = extract_target_span(tokenizer, prompt_input_ids, target_word) 
                    # print("# span_indices: ", span_indices)
                    if not span_indices:
                        print(f"[WARNING] (item_id: {item_id}) target '{target_word}' not found.") 
                        continue

                    attn_map, binary_mask = process_attn(attns,
                                                        prompt_len=prompt_len,      
                                                        span_indices=span_indices,  
                                                        skip_layer_idx_list=config.skip_layer_idx, 
                                                        aggregate=config.aggregate,  
                                                        scaler=config.scaler         
                                                        )

                    # accumulate union (logical OR)
                    if union_binary_mask is None:
                        union_binary_mask = binary_mask
                    else:
                        union_binary_mask = np.logical_or(union_binary_mask, binary_mask).astype(np.uint8)

                    if union_attn_map is None:
                        union_attn_map = attn_map
                    else:
                        union_attn_map = torch.maximum(union_attn_map, attn_map)

                    if save_attn: 
                        # 개별 target word 에 대해 시각화
                        sub_save_dir = os.path.join(save_dir, "sub_mask")
                        sub_save_name = f"{save_name}_{target_word}"

                        visualize_save_attn(
                            attn_map=attn_map,             # torch tensor [H, W]
                            binary_mask=binary_mask,       # numpy array [H, W]
                            save_dir=sub_save_dir,             
                            base_name=sub_save_name, 
                            orig_image=Image.open(img_path) # overlay
                        )

                        subsub_save_dir = os.path.join(save_dir, "sub_attn")
                        os.makedirs(subsub_save_dir, exist_ok=True)
                        save_attn_npy(attn_map, save_path=os.path.join(subsub_save_dir, f"{sub_save_name}_attn.npy"))
                        


                # (Target word 를 모두 돌고난 후,) if no noun matched, skip
                if union_binary_mask is None:
                    print(f"[WARNING] No valid mask for item_id: {item_id}")
                    continue

                if save_attn: 
                    # 4. Save and Visualize (Union Mask)
                    save_binary_mask(union_binary_mask, save_path=os.path.join(save_dir, f"{save_name}_mask.pt"))
                    save_attn_npy(union_attn_map, save_path=os.path.join(save_dir, f"{save_name}_attn.npy"))

                    visualize_save_attn(
                        attn_map=union_attn_map,             # torch tensor [H, W]
                        binary_mask=union_binary_mask,       # numpy array [H, W]
                        save_dir=save_dir,             
                        base_name=save_name, 
                        orig_image=Image.open(img_path) # overlay
                    )

        except Exception as e:
            print(f"[{item_id}]: {e}")

    return



def debug(config, train_data):
    lost = set()

    for example in tqdm(train_data):
        item_id = example['item_id']
        t2i_category = example['t2i_category']

        for part in ['base', 'negative']:
            dir = os.path.join(config.save_dir, part, t2i_category, item_id)
            if not os.path.exists(dir):
                lost.add(item_id)
                break
            elif not any(p.endswith('.pt') for p in os.listdir(dir)):
                lost.add(item_id)
                break
            else:
                continue

    print("Number of lost: ", len(lost))
    print(lost)


def split_size(train_data, s_idx, e_idx):
    if s_idx is not None and e_idx is not None:
        train_data = train_data[s_idx:e_idx]
    elif s_idx is not None:
        train_data = train_data[s_idx:]
    elif e_idx is not None:
        train_data = train_data[:e_idx]
    else:
        pass
    return train_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="/home/yjoh/project/OSPO/ospo/attention/main.yaml")
    parser.add_argument("--s_idx", type=int, default=None)
    parser.add_argument("--e_idx", type=int, default=None)
    args, unknown = parser.parse_known_args()  
    

    # Config
    config = build_config(cfg_path=args.cfg_path)   

    # Train data (MUST INCLUDE NOUNS)
    train_data = read_json(config.data_path)

    # override
    train_data = split_size(train_data, args.s_idx, args.e_idx)


    # 실행
    main(config, train_data, save_attn=config.save_attn)

    # 마스크 로딩 과정 점검
    # mask_path = "/home/yjoh/project/ospo/outputs8/attn_vis/mask.pt"
    # mask = torch.load(mask_path) #.to('cuda')
    # print(mask)
    # print(type(mask)) # <class 'torch.Tensor'>
    # print(mask.shape)


    # 실패 점검 (>>.txt 필수)
    # debug(config, train_data)
