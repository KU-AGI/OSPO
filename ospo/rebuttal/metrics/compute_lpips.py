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
    print(f"Number of generated images: {len(file_list)}")

    # debug
    # file_list = file_list[:100]

    for i, file in enumerate(tqdm(file_list, desc="Computing LPIPS")):
        try:
            file = os.path.basename(file)

            # 2026.01.28
            file_id = os.path.splitext(file)[0]
            original_file = f"COCO_val2014_{int(file_id):012d}.jpg"

            # file_id = int(file.split("_")[1].split(".png")[0])
            # original_image_path = os.path.join(original_image_folder, file) 
            # original_file = f"COCO_val2014_{file_id:012d}.jpg" # COCO_test2014_000000117225.jpg

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
    # gt_caption_path = "/nas/backup/data/coco/annotations/captions_test2014.json" # 미사용

    # 생성 이미지
    # (1) Janus-Pro-7B
    # generated_image_dir = "/nas2/checkpoints/janus_dpo_eval/Janus-Pro/coco_caption/gen" # 499
    # generated_image_dir = "/nas2/checkpoints/janus_dpo_eval/Janus-Pro/coco_caption_val2014/gen"
    # clip score:  0.7346583674231686
    # clip-t score:  0.2414229765654089

    # (2) Janus-Pro-1B
    # generated_image_dir = "/nas2/data/Janus_dataset/next/eval/janus_pro_1b/official/coco_caption_val2014/gen"

    # (3) Janus-Pro-OSPO-1B
    # generated_image_dir = "/nas2/data/Janus_dataset/next/eval/janus_pro_1b/1111_train/ckpt_500/coco_caption_val2014/gen"

    # (4) Janus-Pro-OSPO-7B
    generated_image_dir = "/ssd0/checkpoints/janus_dpo_eval/1017_janus_pro_7b_ospo/coco_caption_val2014/gen"

    # (5) Janus-Pro-SILMM-1B
    # generated_image_dir = "/nas2/data/Janus_dataset/next/eval/janus_pro_1b/1112_janus_1b_silmm/ckpt_500/coco_caption_val2014/gen"

    # (6) Janus-Pro-SILMM-7B
    # generated_image_dir = "/nas2/data/Janus_dataset/next/eval/0801_SILMM_beta_3/ckpt_330/coco_caption_val2014/gen"


    # generated_image_dir = "/nas2/checkpoints/janus_dpo_eval/0323_7b_simpo_silmm_reproduce_ckpt_250/coco_caption/gen"
    # clip-i score:  0.625898789509712
    # clip-t score:  0.21810571976514706

    # generated_image_dir = "/nas2/checkpoints/janus_dpo_eval/0323_7b_simpo_silmm_reproduce_ckpt_500/coco_caption/gen"
    # clip-i score:  0.6575302792866705
    # clip-t score:  0.22540311762684548


    score, num_generated = compute_folder(original_image_dir, generated_image_dir)

    print(f"Final LPIPS score: {score} || Num generated: {num_generated}")

    # save as summary 
    save_dir = "/home/yjoh/project/OSPO/ospo/rebuttal/metrics/results"
    # summary_path = os.path.join(save_dir, "lpips_summary_janus_pro_7b.txt")
    # summary_path = os.path.join(save_dir, "lpips_summary_janus_pro_1b.txt")
    # summary_path = os.path.join(save_dir, "lpips_summary_ospo_1b.txt")
    # summary_path = os.path.join(save_dir, "lpips_summary_silmm_7b.txt")
    summary_path = os.path.join(save_dir, "lpips_summary_ospo_7b.txt")

    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(f"LPIPS score: {score} || Num generated: {num_generated}\n")