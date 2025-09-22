import os
import sys
import argparse
import numpy as np
import PIL.Image
import time
import torch
from torchvision import transforms
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from pytorch_lightning import seed_everything

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
sys.path.append("./unitok/eval/liquid")

from unitok.utils.config import Args
from unitok.models.unitok import UniTok
from unitok.eval.liquid.model import *


class UniTokInferenceWrapper():
    def __init__(self, model, text_tokenizer, img_tokenizer, save_path=None):
        self.model = model
        self.text_tokenizer = text_tokenizer
        self.img_tokenizer = img_tokenizer
        self.save_path = save_path

        self.num_codebooks = 8 
        self.pil_transform = transforms.ToPILImage()
        self.eoi = torch.tensor([4])
        self.boi = torch.tensor([3])
        self.eos = torch.tensor([2])
        self.bos = torch.tensor([1])
    

    def get_prompt(self, text):
        return text + ' Generate an image based on this description.\x00'


    def sample(self, logits, temperature, top_k, top_p, sample_logits=True):
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        if top_k > 0 or top_p < 1.0:
            logits = self.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        if sample_logits:
            idx = torch.multinomial(probs, num_samples=1)
        else:
            _, idx = torch.topk(probs, k=1, dim=-1)
        return idx, probs


    def top_k_top_p_filtering(
        self,
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

        return logits


    @torch.inference_mode()
    def generate(self, 
                text: str,
                cfg_scale: float = 5,
                temperature: float = 1,
                top_k: int = 0,
                top_p: float = 1,
                sample_logits: bool = True):

        prompt = self.get_prompt(text)
        fname = f"{text.strip()}.png"
        save_path = os.path.join(self.save_path, fname)

        sampling_kwargs = {'temperature': temperature, 'top_k': top_k, 'top_p': top_p, 'sample_logits': sample_logits} # sample_logtis: Bool

        uncondition_text_inputs = ['<unconditional>\x00'] 
        if cfg_scale > 1:
            model_inputs = self.text_tokenizer([prompt] + uncondition_text_inputs, return_tensors="pt", padding=True).cuda()
        else:
            model_inputs = self.text_tokenizer([prompt], return_tensors="pt", padding=True).cuda()
        
        model_kwargs = {'attention_mask': model_inputs.pop('attention_mask'), 'use_cache': True}
        input_ids = model_inputs.pop('input_ids')
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        save_list = []
        with torch.no_grad():
            pred_tokens = []
            input_multi_ids = None

            for _ in range(256):
                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                outputs = self.model.T2I_forward_withcache(
                    **model_inputs,
                    input_multi_ids=input_multi_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

                next_embed = outputs['last_hidden_state'][:, -1:, :]

                indices_arhead = []
                for i_head in range(self.num_codebooks):
                    # first next_embed = input's last hidden state
                    ar_next_embed = self.model.ar_head(
                        inputs_embeds=next_embed,
                        use_cache=False,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=False,
                    )
                    
                    next_token_logits = self.model.ar_head.linear_head(ar_next_embed[0]) # sub vocab size = 4096
                    if cfg_scale > 1:
                        cond_logits, uncond_logits = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0)
                        cfg_logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
                        half_next_token, _ = self.sample(cfg_logits, **sampling_kwargs)
                        # pred_tokens.append(half_next_token)
                        next_token = torch.cat([half_next_token, half_next_token])  # [bz,1]
                    else:
                        next_token, next_prob = self.sample(next_token_logits, **sampling_kwargs)
                        # pred_tokens.append(next_token)
                    indices_arhead.append(next_token)
                    if i_head < self.num_codebooks - 1:
                        predicted_embed = self.model.ar_head.codebooks[i_head](next_token) 
                        next_embed = torch.cat([next_embed, predicted_embed], dim=1)

                pred_tokens.append(torch.cat(indices_arhead, dim=1))  # [numcodebook, bsz*2]
                input_multi_ids = torch.stack(pred_tokens, dim=-1)
                fake_id = torch.zeros_like(input_ids[:, :1])
                input_ids = torch.cat([input_ids, fake_id], dim=-1)  # add fake ID for cache

                model_kwargs = self.model._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.model.config.is_encoder_decoder,
                )

        del sampling_kwargs
        del model_inputs
        del outputs
        del model_kwargs

        image_vq_id = torch.stack(pred_tokens, dim=-1)[:1] # bsz=1
        save_list.append(image_vq_id)
        torch.cuda.empty_cache()

        print('Decoding images ...')
        for idx, vq_code in enumerate(save_list[0]):
            new_gen_ids = vq_code.unsqueeze(0).to('cuda')
            rec_image = self.img_tokenizer.idx_to_img(new_gen_ids)
            rec_img = self.pil_transform(rec_image.squeeze(0).to(torch.float32).add(1).mul_(0.5).clamp_(0, 1))
            rec_img.save(save_path)
        

def get_model(args):
    dtype = torch.bfloat16 if config.base.precision == 'bf16' else torch.float32
    
    print('Loading VQ model ...')
    ckpt = torch.load(args.tokenizer_path, map_location='cpu')
    
    vae_cfg = Args()
    vae_cfg.load_state_dict(ckpt['args'])

    vq_model = UniTok(vae_cfg)
    vq_model.load_state_dict(ckpt['trainer']['unitok'])
    vq_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left') # padding_side must be LEFT.
    vqllm = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    )

    return vqllm, tokenizer, vq_model
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="a blue car and red bench")   
    parser.add_argument("--model_path", type=str, default="KU-AGILab/OSPO-Unitok-MLLM-7B")   
    parser.add_argument("--save_path", type=str, default="./results")
    
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(42, workers=True)

    model, text_tokenizer, img_tokenizer = get_model(config)

    start = time.time()

    wrapper = UniTokInferenceWrapper(model, text_tokenizer, img_tokenizer, args.save_path)
    wrapper.generate(text=args.input)

    end = time.time()
    elapsed_time = (end - start) / 60  # Convert seconds to minutes
    print(f"Time elapsed: {elapsed_time:.2f} minutes")


if __name__ == "__main__":
    main()