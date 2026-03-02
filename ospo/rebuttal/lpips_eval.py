# pip install lpips pillow torch torchvision
# modified from /home/yjoh/project/ospo/eval/coco_batched.py
# modified from /home/yjoh/project/ospo/eval/clip_score.py


# (CLIP score 와 동일하게) MSCOCO-30K 기준 측정


import os
from glob import glob
from tqdm import tqdm

import torch
import lpips
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = lpips.LPIPS(net="alex").to(device).eval()

def load_for_lpips(path, size=256):
    img = Image.open(path).convert("RGB")
    # resize to fixed size (square). You can use 512 if you prefer.
    img = TF.resize(img, [size, size], interpolation=TF.InterpolationMode.BICUBIC)
    x = TF.to_tensor(img) * 2 - 1          # [-1,1]
    return x.unsqueeze(0)

@torch.no_grad()
def compute_single_lpips(p0, p1, size=256):
    x0 = load_for_lpips(p0, size=size).to(device)
    x1 = load_for_lpips(p1, size=size).to(device)
    d = loss_fn(x0, x1)
    return float(d.item())

def compute_folder(original_image_folder, generated_image_folder):

    s_list = []
    file_list = os.listdir(generated_image_folder)

    for i, file in enumerate(tqdm(file_list, desc="Computing LPIPS")):
        try:
            file = os.path.basename(file)
            file_id = int(file.split("_")[1].split(".png")[0])
            # original_image_path = os.path.join(original_image_folder, file) 
            original_file = f"COCO_val2014_{file_id:012d}.jpg" # COCO_test2014_000000117225.jpg

            original_image_path = os.path.join(original_image_folder, original_file)
            generated_image_path = os.path.join(generated_image_folder, file)

            # 이미지 - 이미지 간 스코어
            s = compute_single_lpips(original_image_path, generated_image_path)
            s_list.append(s)
        except Exception as e:
            print(e)
            print(f"Error processing file: {file}, skipping.")
            continue
        # break

    num_generated = len(file_list)
    if len(s_list) == 0:
        return 0, 0

    print("LPIPS score: ", sum(s_list) / len(s_list))
    return sum(s_list) / len(s_list), num_generated



if __name__ == '__main__':
    # 고정 (비교 데이터)
    original_image_dir = "/nas/backup/data/coco/images/val2014" 
    gt_caption_path = "/nas/backup/data/coco/annotations/captions_test2014.json"

    # 생성 이미지
    generated_image_dir = "/nas2/checkpoints/janus_dpo_eval/Janus-Pro/coco_caption/gen"
    # clip score:  0.7346583674231686
    # clip-t score:  0.2414229765654089

    # generated_image_dir = "/nas2/checkpoints/janus_dpo_eval/0323_7b_simpo_silmm_reproduce_ckpt_250/coco_caption/gen"
    # clip-i score:  0.625898789509712
    # clip-t score:  0.21810571976514706

    # generated_image_dir = "/nas2/checkpoints/janus_dpo_eval/0323_7b_simpo_silmm_reproduce_ckpt_500/coco_caption/gen"
    # clip-i score:  0.6575302792866705
    # clip-t score:  0.22540311762684548


    score, num_generated = compute_folder(original_image_dir, generated_image_dir)

    print(f"Final LPIPS score: {score} || Num generated: {num_generated}")
