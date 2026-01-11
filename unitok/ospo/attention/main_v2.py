# prepare input 
# forward
# process attention

import os, json
import torch
import argparse
import numpy as np
from PIL import Image
from typing import *
from tqdm import tqdm
from torchvision import transforms
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from ospo.utils.common import read_json, save_json, set_seed, build_config
from unitok.ospo.step4_train import get_model # 학습 파일
from unitok.ospo.attention.extract_mask import process_attn, save_binary_mask, visualize_save_attn, save_attn_npy
from unitok.utils.data import normalize_01_into_pm1
from unitok.ospo.utils.processor import get_conversation
from unitok.eval.liquid.constants import ( 
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN
)

EOI = torch.tensor([4])
BOI = torch.tensor([3]) 
EOS = torch.tensor([2])
BOS = torch.tensor([1])


# leading-space marker
def pretty(tok: str) -> str:
    ptok = tok.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")
    ptok = ptok.strip()
    return ptok

# target_idx (관건)
def extract_target_span(tokenizer, prompt_input_ids: torch.Tensor, target_word: str):
    if isinstance(prompt_input_ids, torch.Tensor):
        if prompt_input_ids.dim() > 1:
            prompt_input_ids = prompt_input_ids[0]
        prompt_id_list = prompt_input_ids.detach().cpu().tolist()
    else:
        prompt_id_list = list(prompt_input_ids)

    # replace invalid ids (like -200) with pad_token_id
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    cleaned_ids = [tid if tid >= 0 else pad_id for tid in prompt_id_list]

    # print("prompt_id_list:", prompt_id_list)
    # [1, 263, 3708, 552, 10714, 29889, 3251, 403, 385, 1967, 2729, 373, 445, 6139, 29889, 3, -200, 4, 2]

    span_indices = []

    # prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_id_list)
    prompt_tokens = tokenizer.convert_ids_to_tokens(cleaned_ids)
    # prompt_tokens = [pretty(t) for t in prompt_tokens] - Unitok 에서는 pretty() 미적용
    
    target_word = target_word.lower().strip()
    target_tokens = tokenizer.tokenize(target_word)

    # print("# prompt_input_ids: (CLEAN) ", cleaned_ids)
    # # print()
    # print("# prompt_tokens: ", prompt_tokens)
    # print("# target_tokens: ", target_tokens)

    # (1) Find all starting indices where the target subtoken sequence appears
    for i in range(len(prompt_tokens) - len(target_tokens) + 1):
        if prompt_tokens[i:i+len(target_tokens)] == target_tokens:
            span_indices = list(range(i, i+len(target_tokens)))
            break
    
    # (2) dinosaur case ('dino' 'saur') - exception case 1
    for t in target_tokens:
        for i, p in enumerate(prompt_tokens):
            if t in p:
                span_indices.append(i)
                # break

    span_indices = list(set(span_indices))
    return span_indices


def get_image_generation_prompt(prompt):
    converation = get_conversation(prompt, cfg_prob = 1.0)

    return converation

def get_text_token(text_tokenizer, text):
    prompt = get_image_generation_prompt(text)
    text_input_ids = text_tokenizer(prompt, 
                            return_tensors="pt", 
                            padding="longest", 
                            max_length=text_tokenizer.model_max_length, 
                            truncation=True)['input_ids'][0]
    text_input_ids = torch.LongTensor(text_input_ids) # e.g. torch.Size([18])

    return text_input_ids

def get_image_tensor(image_processor, img_path: str):
    image = Image.open(img_path).convert("RGB")
    image_tensor = image_processor(image)

    return image_tensor


def get_input_embeddings(image_processor, text_tokenizer, example: dict):
    # tokenize, process
    item_id = example["item_id"]
    text_token = get_text_token(text_tokenizer, example["prompt"]).cuda() # = text_input_ids
    
    chosen_image_tensor = get_image_tensor(image_processor, example["chosen"]).cuda()
    rejected_image_tensor = get_image_tensor(image_processor, example["rejected"]).cuda()

    expand_token = torch.tensor([BOI, IMAGE_TOKEN_INDEX, EOI, EOS], dtype=torch.long).cuda()
    text_token = torch.cat([text_token, expand_token])

    return item_id, text_token, chosen_image_tensor, rejected_image_tensor


# from decode_base() function
def prepare_input(model, img_processor, text_tokenizer, img_tokenizer, example: dict):

    item_id, text_token, chosen_image_tensor, rejected_image_tensor = get_input_embeddings(img_processor, text_tokenizer, example) 
    # print("text_token: ", text_token) = text_input_ids
    
    # text_tokens = text_token + text_token
    text_tokens = torch.stack([text_token, text_token], dim=0).to('cuda')

    # text preprocess

    # LABEL
    labels = []
    for idx, input_ids in enumerate(text_tokens):
        input_ids = input_ids.clone()
        
        instruction_len = torch.where(input_ids == BOI.to('cuda'))[0].item() 
        # 첫번째 BOI 전 까지 text input
        input_ids[:instruction_len] = IGNORE_INDEX
        labels.append(input_ids)

    attention_mask = text_tokens.ne(text_tokenizer.pad_token_id)
    # NO PADDING is needed.


    # image process
    # chosen_image_tensors = torch.stack(chosen_image_tensor, dim=0).to('cuda')       # (1,3,256,256)
    # rejected_image_tensors = torch.stack(rejected_image_tensor, dim=0).to('cuda')   # (1,3,256,256)
    
    # before squeeze: # chosen_image_tensors:  torch.Size([3, 256, 256])   
    chosen_image_tensors = chosen_image_tensor.unsqueeze(0)
    rejected_image_tensors = rejected_image_tensor.unsqueeze(0)

    # print("chosen_image_tensors: ", chosen_image_tensors.shape) 
    # print("rejected_image_tensors: ", rejected_image_tensors.shape)

    with torch.no_grad():
        batch_size = 1 # 고정

        concatenated_image_tensors = torch.cat([chosen_image_tensors, rejected_image_tensors], dim=0).to('cuda') # 2,
        concatenated_img_tokens = img_tokenizer.img_to_idx(concatenated_image_tensors).to('cuda').unsqueeze(1) # (2, 1, 8, 256) 

        chosen_img_tokens = concatenated_img_tokens[:batch_size]
        rejected_img_tokens = concatenated_img_tokens[batch_size:]
        concatenated_img_tokens = torch.cat([chosen_img_tokens, rejected_img_tokens], dim=0)

    data_types = torch.ones(len(text_tokens), dtype=torch.long, device='cuda') # IMG Generation Flag

    # print("text_tokens: ", text_tokens)
    assert len(text_tokens) == len(labels) == len(attention_mask) == len(data_types) == len(concatenated_img_tokens), f"Size Error! {len(text_tokens)} | {len(attention_mask)} | {len(data_types)} | {len(concatenated_img_tokens)}"
        

    # text + (VQ) image token
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
    ) = model.prepare_inputs_labels_for_multimodal(
        input_ids=text_tokens,
        position_ids=None,
        attention_mask=attention_mask,
        past_key_values=None,
        labels=labels, # padded_labels,
        images=concatenated_img_tokens,
        images_aux=None,
        data_types=data_types
    )

    attention_mask = attention_mask.contiguous()

    preprocessed={"item_ids": [item_id], "text_input_ids": text_token} # item_ids: List.
    preprocessed["chosen_inputs_embeds"] = inputs_embeds[:batch_size].to('cuda')
    preprocessed["chosen_attention_mask"] = attention_mask[:batch_size].to('cuda')
    # preprocessed["chosen_labels"] = labels[:batch_size].to('cuda') # output text 부분의 label만.
    preprocessed["chosen_data_types"] = data_types[:batch_size].to('cuda')
    preprocessed["chosen_additional_image_labels"] = additional_image_labels[:batch_size] # list
    preprocessed["chosen_additional_image_indexs"] = additional_image_indexs[:batch_size] # list
    preprocessed["chosen_position_ids"] = position_ids[:batch_size].to('cuda')
    
    preprocessed["rejected_inputs_embeds"] = inputs_embeds[batch_size:].to('cuda')
    preprocessed["rejected_attention_mask"] = attention_mask[batch_size:].to('cuda')
    # preprocessed["rejected_labels"] = labels[batch_size:].to('cuda')       
    preprocessed["rejected_data_types"] = data_types[batch_size:].to('cuda')
    preprocessed["rejected_additional_image_labels"] = additional_image_labels[batch_size:]
    preprocessed["rejected_additional_image_indexs"] = additional_image_indexs[batch_size:]
    preprocessed["rejected_position_ids"] = position_ids[batch_size:].to('cuda')
    
    return preprocessed


    # chosen_input_embeds = torch.cat([text_embeds, chosen_img_embeds], dim=0)
    # rejected_input_embeds = torch.cat([text_embeds, rejected_img_embeds], dim=0)         

    # chosen_attention_mask = torch.ones(chosen_input_embeds.shape[0], dtype=torch.long, device='cuda')
    # rejected_attention_mask = torch.ones(rejected_input_embeds.shape[0], dtype=torch.long, device='cuda')

    # return text_input_ids, chosen_input_embeds, rejected_input_embeds, chosen_attention_mask, rejected_attention_mask


def forward(model, inputs_embeds, attention_mask, position_ids):

    outputs = model.get_model()(
            input_ids = None,
            inputs_embeds=inputs_embeds,    # concatenated_batch["concatenated_inputs_embeds"], # chosen t + chosen i / chosen t + rejected i
            attention_mask=attention_mask,   # concatenated_batch["concatenated_attention_mask"],
            position_ids=position_ids,      # concatenated_batch["concatenated_position_ids"],
            use_cache=False, 
            past_key_values=None,
            output_attentions=True, # 추가
            return_dict=True,
        ) 

    return outputs


def main(config, train_data, save_attn: bool = False):

    set_seed(config.get('seed', 42))

    # config.model.attn_mode 주의
    model, text_tokenizer, img_tokenizer = get_model(config)
    img_processor = transforms.Compose([
            transforms.Resize(int(256 * 1.125)),
            transforms.CenterCrop(256),
            transforms.ToTensor(), normalize_01_into_pm1,
        ])
    print("Model Loading Done.")


    # Set to train mode()
    model = model.cuda()
    img_tokenizer = img_tokenizer.cuda()
    # todo: text tokenizer?

    model.train()        
    model.config.output_hidden_states = True
    model.config.output_attentions = True
    model.config.tokenizer_padding_side = text_tokenizer.padding_side
    model.config.tokenizer_model_max_length = text_tokenizer.model_max_length
    model.config.mm_use_im_start_end = False
    model.config.mm_projector_lr = None
    model.config.mm_use_im_patch_token = True
    print("Model Setting Done.")


    for example in tqdm(train_data, desc="Generating attention-based object mask..."):
        item_id = example['item_id']
        try:
            # chosen, rejected image both
            target_word_list = example['nouns'] 
            
            # 1. Prepare input embeddings
            # 하나의 example 에 대한 인풋
            preprocessed = prepare_input(model, img_processor, text_tokenizer, img_tokenizer, example)
            # for k, v in preprocessed.items():
            #     print(f"{k}: {v}")

            # 2. Forward (개별 포워딩)            
            # NO GRAD
            with torch.no_grad():
                c_outputs = forward(model, 
                                    preprocessed["chosen_inputs_embeds"], 
                                    preprocessed["chosen_attention_mask"], 
                                    preprocessed["chosen_position_ids"])
                c_attns = c_outputs.attentions 

                r_outputs = forward(model, 
                                    preprocessed["rejected_inputs_embeds"], 
                                    preprocessed["rejected_attention_mask"], 
                                    preprocessed["rejected_position_ids"])
                r_attns = r_outputs.attentions
            
            # print("Forward Done.\n")
            # print(preprocessed["chosen_inputs_embeds"].shape) # torch.Size([1, 274, 4096])  
            # print(len(c_attns)) # 32 (tuple): 레이어 개수
            # print(c_attns[0].shape) # torch.Size([1, 32, 274, 274])


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
                    prompt_len = len(preprocessed["text_input_ids"])
                    span_indices = extract_target_span(text_tokenizer, preprocessed["text_input_ids"], target_word) 
                    # print("# span_indices: ", span_indices)
                    
                    if not span_indices:
                        print(f"[WARNING] (item_id: {item_id}) target '{target_word}' not found.") 
                        continue

                    # TODO: 우선 어텐션 뽑히는 단계까지 확인
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


                    # 서브 마스크 저장
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

                        # subsub_save_dir = os.path.join(save_dir, "sub_attn")
                        # os.makedirs(subsub_save_dir, exist_ok=True)
                        # save_attn_npy(attn_map, save_path=os.path.join(subsub_save_dir, f"{sub_save_name}_attn.npy"))
                        


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
    parser.add_argument("--cfg_path", type=str, default="unitok/ospo/attention/main.yaml")
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
