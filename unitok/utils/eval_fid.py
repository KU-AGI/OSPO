import os
import glob
import torch
from tqdm import tqdm
from PIL import Image
from typing import List, Tuple
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.transforms import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils import dist, misc
from models.vqvae import VQVAE
from models.unitok import UniTok
from utils.data import PlainDataset, normalize_01_into_pm1
from utils.config import Args


def eval_fid(vae, dir_raw, dir_recon, feature_extractor_path):
    vae.eval()
    if not os.path.exists(dir_recon):
        misc.os_system(f'mkdir -p {dir_recon}')
    else:
        misc.os_system(f'rm -rf {dir_recon}/*')
    dataloader = prepare_eval_data(dir_raw)
    for file_names, imgs in tqdm(dataloader):
        imgs = imgs.to('cuda', non_blocking=True)
        with torch.no_grad():
            rec_imgs = vae.img_to_reconstructed_img(imgs)
        file_names = [os.path.join(dir_recon, f) for f in file_names]
        save_img_tensor(rec_imgs, file_names)
    # dist.barrier()
    return get_fid_is(dir_raw, dir_recon, feature_extractor_path)


def prepare_eval_data(dir_raw):
    preprocess_val = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ])
    dataset = PlainDataset(dir_raw, preprocess_val)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, pin_memory=True, batch_size=64, sampler=sampler, shuffle=False, drop_last=False)
    return dataloader


def save_img_tensor(recon_B3HW: torch.Tensor, paths: List[str]):  # img_tensor: [-1, 1]
    img_np_BHW3 = recon_B3HW.add(1).mul_(0.5 * 255).round().nan_to_num_(128, 0, 255).clamp_(0, 255).to(
        dtype=torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for bi, path in enumerate(paths):
        img_pil_HW3 = Image.fromarray(img_np_BHW3[bi])
        img_pil_HW3.save(path)


def get_fid_is(dir_raw: str, dir_recon: str, feature_extractor_path: str) -> Tuple[float, float]:
    import torch_fidelity
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=dir_recon,
        input2=dir_raw,
        samples_shuffle=True,
        samples_find_deep=False,
        samples_find_ext='png,jpg,jpeg',
        samples_ext_lossy='jpg,jpeg',

        cuda=True,
        batch_size=1536,
        isc=True,
        fid=True,

        kid=False,
        kid_subsets=100,
        kid_subset_size=1000,

        ppl=False,
        prc=False,
        ppl_epsilon=1e-4 or 1e-2,
        ppl_sample_similarity_resize=64,
        feature_extractor='inception-v3-compat',
        feature_layer_isc='logits_unbiased',
        feature_layer_fid='2048',
        feature_layer_kid='2048',
        feature_extractor_weights_path=feature_extractor_path,
        verbose=True,

        save_cpu_ram=True,  # using num_workers=0 for any dataset input1 input2
        rng_seed=0,  # FID isn't sensitive to this
    )
    fid = metrics_dict['frechet_inception_distance']
    isc = metrics_dict['inception_score_mean']
    return fid, isc


if __name__ == '__main__':
    ckpt_path = ''
    ckpt = torch.load(ckpt_path, map_location='cpu')
    vae_cfg = Args()
    vae_cfg.load_state_dict(ckpt['args'])
    vq_model = UniTok(vae_cfg)
    vq_model.load_state_dict(ckpt['trainer']['unitok'])
    vq_model.to('cuda')
    vq_model.eval()

    dir_raw = ''
    dir_recon = ''
    feature_extractor_path = ''
    eval_fid(vq_model, dir_raw, dir_recon, feature_extractor_path)

