import os
import torch
import argparse
from PIL import Image
from utils.config import Args
from models.unitok import UniTok
from utils.data import normalize_01_into_pm1
from torchvision.transforms import transforms, InterpolationMode


def save_img(img: torch.Tensor, path):
    img = img.add(1).mul_(0.5 * 255).round().nan_to_num_(128, 0, 255).clamp_(0, 255)
    img = img.to(dtype=torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    img = Image.fromarray(img[0])
    img.save(path)


def main(args):
    # load model
    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location='cpu')
    unitok_cfg = Args()
    unitok_cfg.load_state_dict(ckpt['args'])
    unitok = UniTok(unitok_cfg)
    unitok.load_state_dict(ckpt['trainer']['unitok'])
    unitok.to('cuda')
    unitok.eval()

    preprocess = transforms.Compose([
        transforms.Resize(int(unitok_cfg.img_size * unitok_cfg.resize_ratio)),
        transforms.CenterCrop(unitok_cfg.img_size),
        transforms.ToTensor(), normalize_01_into_pm1,
    ])
    img = Image.open(args.src_img).convert("RGB")
    img = preprocess(img).unsqueeze(0).to('cuda')

    with torch.no_grad():
        code_idx = unitok.img_to_idx(img)
        rec_img = unitok.idx_to_img(code_idx)

    final_img = torch.cat((img, rec_img), dim=3)
    save_img(final_img, args.rec_img)

    print('The image is saved to {}. The left one is the original image after resizing and cropping. The right one is the reconstructed image.'.format(args.rec_img))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--src_img', type=str, default='')
    parser.add_argument('--rec_img', type=str, default='')
    args = parser.parse_args()
    main(args)

