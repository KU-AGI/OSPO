import os
import json
import torch
from PIL import Image
from glob import glob
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer


""" IMAGE - GT IMAGE """
def calculate_clip_s(origial_image_path, generated_image_path, model_clip, preprocess_clip, device=None):
    try:
        original_image = Image.open(origial_image_path)
        generated_image = Image.open(generated_image_path)
        #original_image = preprocess_clip(original_image).unsqueeze(0).to('cuda')
        #generated_image = preprocess_clip(generated_image).unsqueeze(0).to('cuda'
        if device is not None:
            original_image = preprocess_clip(images=original_image, return_tensors='pt')['pixel_values'].to(device)
            generated_image = preprocess_clip(images=generated_image, return_tensors='pt')['pixel_values'].to(device)
        else:
            original_image = preprocess_clip(images=original_image, return_tensors='pt')['pixel_values'].to('cuda')
            generated_image = preprocess_clip(images=generated_image, return_tensors='pt')['pixel_values'].to('cuda')
    except Exception as e:
        print(e)
        return None
    
    with torch.no_grad():
        # original_image_features = model_clip.encode_image(original_image)
        # generated_image_features = model_clip.encode_image(generated_image)
        original_image_features = model_clip.get_image_features(pixel_values=original_image)
        generated_image_features = model_clip.get_image_features(pixel_values=generated_image)
    s = torch.cosine_similarity(original_image_features, generated_image_features, dim=-1)
    return s.item()


def calculate_clip_s_for_folder(original_image_folder, generated_image_folder):
    s_list = []
    file_list = glob(os.path.join(generated_image_folder, '*.png')) 
    # file_list = glob(os.path.join(generated_image_folder, '*.jpg'))

    #model_clip, _,  preprocess_clip = open_clip.create_model_and_transforms('ViT-L-14', device='cuda')
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    preprocess_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    for i, file in enumerate(tqdm(file_list)):
        file = os.path.basename(file)
        file_id = int(file.split("_")[1].split(".png")[0])
        # original_image_path = os.path.join(original_image_folder, file) 
        original_file = f"COCO_val2014_{file_id:012d}.jpg" # COCO_test2014_000000117225.jpg

        original_image_path = os.path.join(original_image_folder, original_file)
        generated_image_path = os.path.join(generated_image_folder, file)

        # 이미지 - 이미지 간 클립 스코어
        s = calculate_clip_s(original_image_path, generated_image_path, model_clip, preprocess_clip)
        if s is not None:
            s_list.append(s)

    num_generated = len(file_list)

    if len(s_list) == 0:
        return 0
    
    print("clip-i score: ", sum(s_list) / len(s_list))
    return sum(s_list) / len(s_list), num_generated


def calculate_clip_t(gt_text, generated_image_path, model_clip, preprocess_clip, tokenizer, device=None):

    if device is not None:
        inputs = tokenizer(gt_text, padding=True, return_tensors="pt").to(device)
    else:
        inputs = tokenizer(gt_text, padding=True, return_tensors="pt").to('cuda')
    try:
        generated_image = Image.open(generated_image_path)
        if device is not None:
            generated_image = preprocess_clip(images=generated_image, return_tensors='pt')['pixel_values'].to(device)
        else:
            generated_image = preprocess_clip(images=generated_image, return_tensors='pt')['pixel_values'].to('cuda')
    except Exception as e:
        print(e)
        return None

    with torch.no_grad():
        generated_image_features = model_clip.get_image_features(pixel_values=generated_image)
        text_features = model_clip.get_text_features(**inputs)
    s = torch.cosine_similarity(text_features, generated_image_features, dim=-1)
    return s.item()



""" IMAGE - GT TEXT """
def calculate_clip_t_for_folder(generated_image_folder, 
                                # coco-caption 기준
                                gt_caption_path):
    t_list = []
    file_list = glob(os.path.join(generated_image_folder, '*.png'))
    # file_list = glob(os.path.join(generated_image_folder, '*.jpg'))
    
    model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").cuda()
    preprocess_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    with open(gt_caption_path, 'r') as f:
        # gt_captions = json.load(f)
        gt_data = json.load(f)['annotations'] # coco-caption 기준

    gt_captions = {}
    for d in gt_data:
        image_id = d['image_id'] # 256223
        gt_captions[image_id] = d['caption']

    for i, file in enumerate(tqdm(file_list)):
        file = os.path.basename(file) #  A kitchen is cluttered with cooking supplies, especially eggs_314370.png
        file_id = int(file.split("_")[1].split(".png")[0])
        gt = gt_captions[file_id]
        
        generated_image_path = os.path.join(generated_image_folder, file)

        t = calculate_clip_t(gt, generated_image_path, model_clip, preprocess_clip, tokenizer)
        if t is not None:
            t_list.append(t)

    num_generated = len(file_list)

    if len(t_list) == 0:
        print("Nothing generated.")
        return 0
    
    print("clip-t score: ", sum(t_list) / len(t_list))
    return sum(t_list) / len(t_list), num_generated



if __name__ == '__main__':
    # 고정 (비교 데이터)
    original_image_dir = "/nas/backup/data/coco/images/val2014" 
    gt_caption_path = "/nas/backup/data/coco/annotations/captions_test2014.json"

    # 생성 이미지
    # generated_image_dir = "/nas2/checkpoints/janus_dpo_eval/Janus-Pro/coco_caption/gen"
    # clip score:  0.7346583674231686
    # clip-t score:  0.2414229765654089

    # generated_image_dir = "/nas2/checkpoints/janus_dpo_eval/0323_7b_simpo_silmm_reproduce_ckpt_250/coco_caption/gen"
    # clip-i score:  0.625898789509712
    # clip-t score:  0.21810571976514706

    generated_image_dir = "/nas2/checkpoints/janus_dpo_eval/0323_7b_simpo_silmm_reproduce_ckpt_500/coco_caption/gen"
    # clip-i score:  0.6575302792866705
    # clip-t score:  0.22540311762684548


    score_s, num_generated = calculate_clip_s_for_folder(original_image_dir, generated_image_dir)
    score_t, num_generated = calculate_clip_t_for_folder(generated_image_dir, gt_caption_path)

    print(score_s, score_t)
