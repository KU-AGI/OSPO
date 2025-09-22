import os
import sys
import json
import math
import torch
import argparse
import shortuuid
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
from torchvision import transforms

from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from conversation import conv_templates, SeparatorStyle
from model.builder import load_pretrained_model
from tools import disable_torch_init
from mm_utils import tokenizer_image_token, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

sys.path.append('../..')
from utils.config import Args
from models.unitok import UniTok

ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.set_grad_enabled(False)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # prompt = prompt.replace('<image>','<boi><image><eoi>')
        # import pdb;pdb.set_trace()
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        # import pdb;pdb.set_trace()
        pad_image = expand2square(image, (122, 116, 104) )
        # import pdb;pdb.set_trace()
        img = self.image_processor[0](pad_image).unsqueeze(0)
        img = img.to('cuda')
        # import pdb;pdb.set_trace()
        with torch.no_grad():
            vq_code = self.image_processor[1].img_to_idx(img)
            vqcode = vq_code.cpu()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')


        return input_ids,vqcode,os.path.join(self.image_folder, image_file) #, image_tensor, image_tensor_aux
    
    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=0):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, load_8bit=args.load_8bit)

    ckpt = torch.load(args.tokenizer_path, map_location='cpu')
    vae_cfg = Args()
    vae_cfg.load_state_dict(ckpt['args'])
    vq_model = UniTok(vae_cfg)
    vq_model.load_state_dict(ckpt['trainer']['unitok'])
    vq_model.to('cuda')
    vq_model.eval()
    del ckpt

    crop_size = 256
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    image_processor = (transform, vq_model)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in args.conv_mode and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_codes,imagepath), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        
        input_ids = input_ids.to(device=model.device, non_blocking=True)
        image_codes = image_codes.to(device=model.device, non_blocking=True)
        if hasattr(model, "update_prompt"):
            model.update_prompt([[cur_prompt]])

        with torch.inference_mode():
            output_ids = model.generate_mllm(
                input_ids,
                images=image_codes,
                images_aux=  None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                bos_token_id=tokenizer.bos_token_id,  # Begin of sequence token
                eos_token_id=tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=tokenizer.pad_token_id,  # Pad token
                use_cache=False
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }) + "\n")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--load_8bit', type=bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
