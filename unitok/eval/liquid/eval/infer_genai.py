import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from model import *

sys.path.append('../..')
from utils.config import Args
from models.unitok import UniTok

PILtransform = transforms.ToPILImage()


def top_k_top_p_filtering(
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k

        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    # import pdb;pdb.set_trace()
    return logits


def sample(logits, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, sample_logits=True):
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def split_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--unitok_path', type=str, required=True)
    parser.add_argument('--mllm_path', type=str, required=True)
    parser.add_argument('--prompt_file', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--topk', type=int, default=2048)
    parser.add_argument('--topp', type=float, default=0.96)
    parser.add_argument('--cfg_scale', type=float, default=5.0)
    return parser


def main(args):
    text_set_id = args.idx
    tau = args.tau
    topk = args.topk
    topp = args.topp
    cfg_scale = args.cfg_scale

    print('loading vq model ...')
    ckpt = torch.load(args.unitok_path, map_location='cpu')
    vae_cfg = Args()
    vae_cfg.load_state_dict(ckpt['args'])
    vq_model = UniTok(vae_cfg)
    vq_model.load_state_dict(ckpt['trainer']['unitok'])
    vq_model.to('cuda')
    vq_model.eval()

    image_save_pth = '{}/GenAI-cfg_{}-topk_{}-topp_{}-tau_{}'.format(args.result_dir, str(cfg_scale), str(topk),
                                                                     str(topp), str(tau))

    tokenizer = AutoTokenizer.from_pretrained(args.mllm_path, padding_side='left')
    vqllm = AutoModelForCausalLM.from_pretrained(
        args.mllm_path,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    ).to('cuda')

    num_processes = 8
    chunk_size = 8  # batchsize
    num_codebooks = vae_cfg.num_codebooks

    with open(args.prompt_file, 'r') as f:
        lines = f.readlines()
    all_prompts = []
    for index, line in enumerate(lines):
        all_prompts.append({'Index': str(index + 1).zfill(5), 'Prompt': line.strip()})

    chunked_filenames = np.array_split(all_prompts, num_processes)
    subset = chunked_filenames[text_set_id].tolist()
    chunk_inputs = split_list(subset, chunk_size)
    for chunk in tqdm(chunk_inputs):
        text_inputs = [v['Prompt'] for v in chunk]
        uncondition_text_inputs = ['<unconditional>\x00'] * len(text_inputs)
        for i in range(len(text_inputs)):
            text_inputs[i] = text_inputs[i] + ' Generate an image based on this description.\x00'
        ori_batchsize = len(text_inputs)

        save_list = []
        if cfg_scale > 1:
            model_inputs = tokenizer(text_inputs + uncondition_text_inputs, return_tensors="pt", padding=True).to('cuda')
        else:
            model_inputs = tokenizer(text_inputs, return_tensors="pt", padding=True).to('cuda')

        model_kwargs = {'attention_mask': model_inputs.pop('attention_mask'), 'use_cache': True}
        input_ids = model_inputs.pop('input_ids')
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        with torch.no_grad():
            sampling_kwargs = {'temperature': tau, 'top_k': topk, 'top_p': topp, 'sample_logits': True}
            pred_tokens = []
            input_multi_ids = None
            for _ in range(256):
                model_inputs = vqllm.prepare_inputs_for_generation(input_ids, **model_kwargs)
                outputs = vqllm.T2I_forward_withcache(
                    **model_inputs,
                    input_multi_ids=input_multi_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                next_embed = outputs['last_hidden_state'][:, -1:, :]

                indices_arhead = []
                for i_head in range(num_codebooks):
                    ar_next_embed = vqllm.ar_head(
                        inputs_embeds=next_embed,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=False,
                    )
                    next_token_logits = vqllm.ar_head.linear_head(ar_next_embed[0])
                    if cfg_scale > 1:
                        cond_logits, uncond_logits = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0)
                        cfg_logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
                        half_next_token, _ = sample(cfg_logits, **sampling_kwargs)
                        # pred_tokens.append(half_next_token)
                        next_token = torch.cat([half_next_token, half_next_token])  # [bz,1]
                    else:
                        next_token, next_prob = sample(next_token_logits, **sampling_kwargs)
                        # pred_tokens.append(next_token)
                    indices_arhead.append(next_token)
                    if i_head < num_codebooks - 1:
                        predicted_embed = vqllm.ar_head.codebooks[i_head](next_token)
                        next_embed = torch.cat([next_embed, predicted_embed], dim=1)

                pred_tokens.append(torch.cat(indices_arhead, dim=1))  # [numcodebook,bz*2]
                input_multi_ids = torch.stack(pred_tokens, dim=-1)
                fake_id = torch.zeros_like(input_ids[:, :1])
                input_ids = torch.cat([input_ids, fake_id], dim=-1)  # add fake id for cache

                model_kwargs = vqllm._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=vqllm.config.is_encoder_decoder,
                )
            del sampling_kwargs
            del model_inputs
            del outputs
            del model_kwargs
            # image_vq_id = input_ids[:,prompt_length:prompt_length+256]-ori_vocabe_size
            image_vq_id = torch.stack(pred_tokens, dim=-1)[:ori_batchsize]
            save_list.append(image_vq_id)

        torch.cuda.empty_cache()

        print('decoding images ...')
        if not os.path.exists(image_save_pth):
            os.makedirs(image_save_pth)
        for datainfo, vq_code in zip(chunk, save_list[0]):
            idx = datainfo['Index']
            new_gen_ids = vq_code.unsqueeze(0).to('cuda')
            rec_image = vq_model.idx_to_img(new_gen_ids)
            rec_img = PILtransform(rec_image.squeeze(0).add(1).mul_(0.5).clamp_(0, 1))
            rec_img.save('{}/{}.jpg'.format(image_save_pth, str(idx)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('genai inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

