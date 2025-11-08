import os
import json
import torch
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt
import pyrootutils
from PIL import Image

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from janus.models import MultiModalityCausalLM, VLChatProcessor
from ospo.utils.processor import get_conversation, get_sft_format
from ospo.utils.common import read_json, save_json, set_seed, build_config
from ospo.utils.model import get_model

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

    return text_input_ids, chosen_input_embeds, rejected_input_embeds, chosen_attention_mask, rejected_attention_mask
    



def save_binary_mask(binary_mask, save_path):
    assert save_path.endswith(".pt"), "save path must be ended with '.pt'! "

    # Ensure tensor format
    if isinstance(binary_mask, list):
        binary_mask = torch.tensor(binary_mask, dtype=torch.uint8)
    elif isinstance(binary_mask, np.ndarray):
        binary_mask = torch.from_numpy(binary_mask.astype(np.uint8))
    else:
        raise ValueError(f"Unknown type: {binary_mask}")

    torch.save(binary_mask, save_path)
    # print(f"[Saved mask] {save_path}")
    return


# TODO: upscale
def save_mask_as_png(mask: torch.Tensor, save_path="mask.png"):
    mask_np = (mask.detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_np, mode="L").save(save_path)
    # print(f"Saved mask image: {save_path}")


def show_mask(mask):
    plt.imshow(mask.detach().cpu(), cmap="gray")
    plt.title("Binary Mask")
    plt.axis("off")
    plt.show()


def overlay_mask_on_image(image: np.ndarray, mask: torch.Tensor, alpha=0.5, save_path: str = "overlay.png"):
    """
    image: numpy array [H, W, 3] in 0-255
    mask:  torch.Tensor [H, W], 0/1
    alpha: blending strength
    """
    mask_np = mask.cpu().numpy().astype(np.float32)
    
    # Red overlay where mask==1
    overlay = image.copy().astype(np.float32)
    overlay[mask_np == 1] = (255, 0, 0)  # Red

    blended = (image * (1-alpha) + overlay * alpha).astype(np.uint8)
    # return blended
    Image.fromarray(blended).save(save_path)
    return


def attn_map_to_binary(attention_map, scaler=1.):
    # scaler < 1.0: lower threshold (softer/larger mask)
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.detach()
    attention_map_np = attention_map.cpu().numpy()
    threshold_value = filters.threshold_otsu(attention_map_np) * scaler
    binary_mask = (attention_map_np > threshold_value).astype(np.uint8)

    return binary_mask


# leading-space marker
def pretty(tok: str) -> str:
    ptok = tok.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")
    ptok = ptok.strip()
    return ptok


# target_idx (관건)
def extract_target_span(tokenizer, 
                    prompt_input_ids: torch.Tensor, 
                    target_word: str): # text_input_ids = prompt_input_ids
    
    span_indices = []
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_input_ids)
    prompt_tokens = [pretty(t) for t in prompt_tokens]

    
    target_word = target_word.lower().strip()
    target_tokens = tokenizer.tokenize(target_word)

    # print("# prompt_input_ids: ", prompt_input_ids)
    # print()
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


@torch.no_grad()
def compute_spatial_entropy(
    attn_stack: torch.Tensor,               # [L, img_len, span_len], typically in [0,1]
    span_reduce: str = "max",               # "max" | "mean" | "sum" across the span dimension
    softmax_temperature: float = 1.0,       # τ for softmax over spatial positions (τ<1 sharpens, τ>1 smooths)
    return_extra: bool = False,             # also return peakiness, topK mass, gini, etc.
    eps: float = 1e-12
):
    """
    Compute spatial Shannon entropy per layer (lower = sharper, higher = more diffuse).

    Returns:
        stats: dict with
            - entropy: [L] Shannon entropy (natural log)
            - entropy_norm: [L] entropy normalized by log(N) to [0,1]
            - peakiness: [L] = 1 - entropy_norm (higher = sharper)
          If return_extra=True:
            - top1_mass, top5_mass, top10_mass: [L]
            - gini: [L] inequality measure (higher = sharper)
            - max_val: [L] max fused attention per layer (pre-softmax)
    """ 
    assert attn_stack.dim() == 3, f"Expected [L, img_len, span_len], got {tuple(attn_stack.shape)}"
    L, I, S = attn_stack.shape
   
    # 1) Fuse across span tokens (multiple target tokens)
    if span_reduce == "max":
        fused = attn_stack.max(dim=2).values        # [L, img_len]
    elif span_reduce == "mean":
        fused = attn_stack.mean(dim=2)              # [L, img_len]
    elif span_reduce == "sum":
        fused = attn_stack.sum(dim=2)               # [L, img_len]
    else:
        raise ValueError(f"span_reduce must be one of ['max','mean','sum'], got {span_reduce}")

    # 2) Convert to a spatial probability distribution per layer via softmax
    #    (works even if fused isn't normalized; τ controls sharpness)
    logits = fused / softmax_temperature
    p = torch.softmax(logits, dim=1)                # [L, img_len]
    p = torch.clamp(p, min=eps)                     # avoid log(0)

    # 3) Shannon entropy per layer, and normalized to [0,1]
    H = -(p * p.log()).sum(dim=1)                   # [L], natural log
    H_max = torch.log(torch.tensor(I, dtype=H.dtype, device=H.device))
    H_norm = H / (H_max + eps)
    peakiness = 1.0 - H_norm                        # higher = more concentrated

    stats = {
        "entropy": H,               # absolute entropy
        "entropy_norm": H_norm,     # 0..1
        "peakiness": peakiness,     # 0..1 (higher is better for grounding)
    }

    if return_extra:
        # Top-K cumulative mass (how much probability is in the most-attended patches)
        k_list = [1, 5, 10] if I >= 10 else [1, min(5, I), min(10, I)]
        sorted_p, _ = torch.sort(p, dim=1, descending=True)
        for k in k_list:
            stats[f"top{k}_mass"] = sorted_p[:, :k].sum(dim=1)  # [L]

        # Gini coefficient per layer (0 = uniform, ~1 = extremely concentrated)
        # Fast vectorized formula:
        # Reference: https://en.wikipedia.org/wiki/Gini_coefficient#Discrete_probability_distribution
        # Sort ascending for Gini formula
        p_sorted = torch.sort(p, dim=1).values
        idx = torch.arange(1, I + 1, device=p.device, dtype=p.dtype).unsqueeze(0)  # [1, I]
        gini = ( (2 * (idx * p_sorted).sum(dim=1)) / (p_sorted.sum(dim=1) + eps) - (I + 1) ) / I
        stats["gini"] = gini  # higher = more inequality = sharper

        # For convenience: pre-softmax max value per layer (can help tie-break layers)
        stats["max_val"] = fused.max(dim=1).values

    return stats



@torch.no_grad()
def _otsu_threshold_1d(x: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """
    x: 1D tensor of scores (any range). Returns a scalar threshold (same device/dtype).
    """
    x = x.flatten().float()
    x_min, x_max = x.min(), x.max()
    if (x_max - x_min) <= 1e-8:
        return x_min  # degenerate, everything equal

    # normalize to [0,1] for stable binning
    x_n = (x - x_min) / (x_max - x_min + 1e-8)

    hist = torch.histc(x_n, bins=bins, min=0.0, max=1.0)          # [bins]
    P = hist / (hist.sum() + 1e-12)
    cdf = torch.cumsum(P, dim=0)                                   # ω(k)
    centers = torch.linspace(0.0, 1.0, bins, device=x.device)      # bin centers
    mu_k = torch.cumsum(P * centers, dim=0)                        # μ(k)
    mu_T = (P * centers).sum()

    denom = (cdf * (1.0 - cdf)).clamp_min(1e-12)
    sigma_b2 = ((mu_T * cdf - mu_k) ** 2) / denom                  # between-class var
    k = torch.argmax(sigma_b2)                                     # best split bin

    # convert bin index → threshold in original scale (use bin center)
    thr_n = (k.float() + 0.5) / bins
    thr = x_min + thr_n * (x_max - x_min)
    return thr


@torch.no_grad()
def select_layer_simple(
    attn_stack: torch.Tensor,          # [L, I, S] from process_attn(...)
    normalize: str = "global",
    span_reduce: str = "max",          # "max" or "mean"
    method: str = "entropy",           # "entropy" or "otsu" (no hybrid)
    temperature: float = 0.8,         # used by entropy method (in softmax)
    mask_percentile: float = 0.90,     # threshold for final mask (top-10% by default)
    return_mask: bool = True,
    grid_h: int = 24, grid_w: int = 24,  # 384/16 = 24
):
    """
    Minimal, layer-only selector.
    - method="entropy": pick layer with lowest spatial entropy (softmax over positions).
    - method="otsu":    pick layer with highest Otsu between-class variance.
    Returns (best_layer_idx, fused_map_best, mask_best or None).
    """
    assert attn_stack.dim() == 3, f"Expected [L, I, S], got {tuple(attn_stack.shape)}"
    L, I, S = attn_stack.shape

    # 1) fuse span tokens → spatial map per layer: [L, I]
    if span_reduce == "max":
        fused = attn_stack.max(dim=2).values
    elif span_reduce == "mean":
        fused = attn_stack.mean(dim=2)
    else:
        raise ValueError("span_reduce must be 'max' or 'mean'.")


    # 2) per-layer robust min-max (1–99% to avoid outliers), then clamp to [0,1]
    # q1 = torch.quantile(fused, 0.01, dim=1, keepdim=True)
    # q99 = torch.quantile(fused, 0.99, dim=1, keepdim=True)
    # denom = (q99 - q1).clamp_min(1e-8)
    # fused_norm = ((fused - q1) / denom).clamp(0, 1)  # [L, I]
    
    if normalize == "per_layer":
        fused_norm = fused.clamp(0, 1)
    elif normalize == "global":
        fused_norm = fused.clamp(0, 1) # TODO 

    # 3) score layers by ONE simple criterion
    if method == "entropy":
        # Softmax over spatial positions → p; Shannon entropy H; normalize by log(I)
        p = torch.softmax(fused_norm / temperature, dim=1)              # [L, I]
        H = -(p * (p.clamp_min(1e-12)).log()).sum(dim=1)                 # [L]
        H_norm = H / (torch.log(torch.tensor(I, dtype=H.dtype, device=H.device)) + 1e-12)
        scores = -H_norm  # higher is better (i.e., lower entropy)

        best_idx = torch.argmax(scores).item()
    elif method == "otsu":
        # Simple Otsu separability per layer (histogram in torch)
        bins = 256
        edges = torch.linspace(0.0, 1.0, bins + 1, device=fused_norm.device, dtype=fused_norm.dtype)
        centers = (edges[:-1] + edges[1:]) * 0.5  # [bins]
        sigma_b2 = []
        for l in range(L):
            x = fused_norm[l]  # [I]
            hist = torch.histc(x, bins=bins, min=0.0, max=1.0)  # [bins]
            P = hist / (hist.sum() + 1e-12)
            w = torch.cumsum(P, dim=0)
            mu = torch.cumsum(P * centers, dim=0)
            mu_T = (P * centers).sum()
            denom = (w * (1.0 - w)).clamp_min(1e-12)
            sig = ((mu_T * w - mu) ** 2) / denom
            sigma_b2.append(sig.max())
        scores = torch.stack(sigma_b2)  # higher is better
        best_idx = torch.argmax(scores).item()
    else:
        raise ValueError("method must be 'entropy' or 'otsu'.")

    fused_best = fused_norm[best_idx]  # [I]

    # V1
    # 4) optional binary mask for the chosen layer: top-p percentile (simple & stable)
    # if return_mask:
    #     thr = torch.quantile(fused_best, mask_percentile)
    #     mask = (fused_best >= thr).float()  # [I] in 1D (reshape to grid if needed)
    #     # reshape to grid
    #     mask = mask.reshape(grid_h, grid_w)
    # else:
    #     mask = None


    # V2
    # 4) binary mask for the chosen layer (Otsu; no size rule)
    if return_mask:
        thr = _otsu_threshold_1d(fused_best)     # auto threshold
        # optional strictness knob (like your scaler): <1.0 = larger mask, >1.0 = tighter
        otsu_scale = 1.0
        thr = thr * otsu_scale

        mask = (fused_best >= thr).to(torch.uint8).view(grid_h, grid_w)
    else:
        mask = None

    return best_idx, fused_best, mask


@torch.no_grad()
def layer_spatial_entropy_component_mass(
    attn_stack: torch.Tensor,      # [L, I, S]
    P: int = 24,                        # grid size (e.g., 24)
    span_reduce: str = "max",      # "max" or "mean"
    bin_thresh: float = 0.2        # your cfg.logic.entropy.binarize_threshold
    ):
    # 1) fuse span → [L, I]
    if span_reduce == "max":
        fused = attn_stack.max(dim=2).values
    elif span_reduce == "mean":
        fused = attn_stack.mean(dim=2)
    else:
        raise ValueError("span_reduce must be 'max' or 'mean'")

    L, I = fused.shape
    assert I == P * P, "I must equal P*P to reshape to [P,P]"

    ent = torch.empty(L, dtype=torch.float32)
    num_comp = torch.empty(L, dtype=torch.int32)

    for l in range(L):
        S = fused[l].view(P, P)
        mean_val = S.mean()
        B = torch.relu(S - 2.0 * mean_val)              # same as your code
        total = float(B.sum().item())

        if total <= 0:
            ent[l] = float("inf")
            num_comp[l] = 0
            continue

        B_np = B.detach().cpu().to(torch.float32).numpy()
        binary = (B_np > bin_thresh).astype(np.int32)

        labeled, num = label(binary, structure=np.ones((3, 3), dtype=np.uint8))
        if num == 0:
            ent[l] = float("inf")
            num_comp[l] = 0
            continue

        # component-mass probs (using B, not binary)
        probs = []
        for i in range(1, num + 1):
            comp_sum = B_np[labeled == i].sum()
            if comp_sum > 0:
                probs.append(comp_sum / total)

        if not probs:
            ent[l] = float("inf")
            num_comp[l] = 0
        else:
            probs = np.asarray(probs, dtype=np.float64)
            se = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1)))
            ent[l] = float(se)
            num_comp[l] = int(num)

    # pick best layer = lowest component-mass entropy
    best_idx = torch.argmin(ent).item()
    return best_idx, ent, num_comp


# Process 1D attention vector
def process_attn_per_layer(attentions, prompt_len: int, span_indices: list,
                skip_layer_idx_list: list = [], img_size: int = 24, aggregate: str = "mean", normalize: str = "per_layer", scaler: float = 1.0, eps: float = 1e-8): # layer_idx: int = None):

    if skip_layer_idx_list != []:
        assert max(skip_layer_idx_list) < 30 # total 30 layer

    collected = []

    # 1) Extract SPAN attention weights
    for layer_idx, layer_attn in enumerate(attentions):
        if layer_idx in skip_layer_idx_list:
            continue

        # layer_attn: [B(1), H, Q, K]
        assert len(layer_attn) == 1, "Batch size must be 1."
        span_attn = layer_attn[0, :, prompt_len:, span_indices].float()

        # 2. Aggregate across heads
        if aggregate == "mean":
            span_attn = span_attn.mean(dim=0)  # [img_len, K] 
        elif aggregate == "sum":
            span_attn = span_attn.sum(dim=0)   # [img_len, K]
        else:
            raise NotImplementedError(f"Unsupported aggregate: {aggregate}")
        
        collected.append(span_attn)

    if not collected:
        raise ValueError("No layers collected (check skip_layer_idx_list).")

    # [L_in_range, img_len, span_len]
    attn_stack = torch.stack(collected, dim=0)


    # 2) Normalize
    if normalize == "per_layer":
        # Min–max per layer (broadcast over last two dims)
        layer_min = attn_stack.amin(dim=(1,2), keepdim=True)
        layer_max = attn_stack.amax(dim=(1,2), keepdim=True)
        attn_stack = (attn_stack - layer_min) / (layer_max - layer_min + eps)
    elif normalize == "global":
        gmin = attn_stack.min()
        gmax = attn_stack.max()
        attn_stack = (attn_stack - gmin) / (gmax - gmin + eps)
    elif normalize == None:
        pass
    else:
        raise ValueError(f"normalize must be 'per_layer' or 'global', got {normalize}")

    return attn_stack  # [L_used, img_len, span_len]





# Process 1D attention vector
def process_attn(attentions, prompt_len: int, span_indices: list,
                skip_layer_idx_list: list = [], img_size: int = 24, aggregate: str = "mean", normalize: bool = True, scaler: float = 1.0, eps: float = 1e-8): # layer_idx: int = None):

    assert max(skip_layer_idx_list) < 30 # total 30 layer

    # --- COLLECT attention maps across layers ---
    collected = []

    for layer_idx, layer_attn in enumerate(attentions):
        if layer_idx in skip_layer_idx_list:
            continue

        # print(layer_attn) # torch.Size([1, 32, 589, 589])

        # layer_attn: [B, H, Q, K]
        assert len(layer_attn) == 1, "Batch size must be 1."
        span_attn = layer_attn[0, :, prompt_len:, span_indices].float()

        # aggregate across heads
        if aggregate == "mean":
            span_attn = span_attn.mean(dim=0)  # [K]
        elif aggregate == "sum":
            span_attn = span_attn.sum(dim=0)   # [K]
        else:
            raise NotImplementedError(f"Unsupported aggregate: {aggregate}")
        
        collected.append(span_attn)
        # print(collected[0]) # torch.Size([584, 2])

    # --- AGGREGATE across layers ---
    if isinstance(collected[0], torch.Tensor):
        stacked = torch.stack(collected)  # (L, ...) = (num_layers, K), K = number of token span
        attn_out = stacked.mean(dim=0) if aggregate != "none" else stacked
    else:
        return None


    # --- AGGREGATE across span_indices ---
    if attn_out.ndim > 1:
        attn_out = attn_out.mean(dim=-1)
    # print("attn_out") # (589,)   # after aggregation > IMAGE TOKEN ONLY (attn_out.shape = (576,))


    # --- NORMALIZE --- 
    if normalize and aggregate != "none":
        attn_out = (attn_out - attn_out.min()) / (attn_out.max() - attn_out.min() + 1e-8)

    assert attn_out.shape[0] == img_size * img_size, f"attn_out shape is wrong: {attn_out.shape}"
    attn_map = attn_out.reshape(img_size, img_size)
    binary_mask = attn_map_to_binary(attn_map, scaler=scaler)

    return attn_map, binary_mask


def load_model(config):
    set_seed(config.get('seed', 42)) 

    dtype = torch.bfloat16
    model, chat_processor, image_processor, tokenizer = get_model(mode='train', dtype=dtype, config=config)
    print("Model Loading Done.")

    # Set to train mode()
    model = model.cuda()
    model.train()        
    model.language_model.model.config.output_hidden_states = True
    model.language_model.model.config.output_attentions = True
    print("Model Setting Done.")

    return model, chat_processor, image_processor, tokenizer 


if __name__ == "__main__":
    # 디버깅 목적
    cfg_path = "/home/yjoh/project/ospo/cvpr/attn_method/debug.yaml"
    config = build_config(cfg_path=cfg_path)  

    # 예시 샘플 1 (attribute)
    example = {
        "item_id": "0003355",
        "t2i_category": "attribute",
        "sub_category": "attribute2",
        "prompt": "a paper book and a glass jar.",
        "chosen": "/nas2/data/Janus_dataset/main_exp/aaai/janus/images_pairwise/base/attribute/0003355/00.png",
        "rejected": "/nas2/data/Janus_dataset/main_exp/aaai/janus/images_pairwise/negative/attribute/0003355/00.png",
        "chosen_score": 0.9212472544750199,
        "rejected_score": 0.5764579555980163,
        "rejected_prompt": "a cloth book and a plastic jar",
        "nouns": [
            "book",
            "glass",
            "jar",
            "paper"
        ]
    }

    # 0. 모델 로드
    model, chat_processor, image_processor, tokenizer = load_model(config)

    # 1. 인풋 준비
    prompt_input_ids, chosen_input_embeds, rejected_input_embeds, chosen_attention_mask, rejected_attention_mask = \
            prepare_input(model, chat_processor, image_processor, tokenizer, example)
    
    prompt_len = len(prompt_input_ids)

    # 2. 모델 포워드 
    with torch.no_grad():
        c_outputs = forward(model, chosen_input_embeds, chosen_attention_mask)
        c_attns = c_outputs.attentions
        r_outputs = forward(model, rejected_input_embeds, rejected_attention_mask)
        r_attns = r_outputs.attentions


    attns = c_attns
    image = np.array(Image.open(example["chosen"]).convert("RGB"))

    # 3. 레이어 별 어텐션 값 처리
    target_word_list = example["nouns"]  

    for target_word in target_word_list:
        print("# target word: ", target_word)
        span_indices = extract_target_span(tokenizer, prompt_input_ids, target_word) 
        print("# span_indices: ", span_indices)
        if not span_indices:
            print(f"[WARNING] (item_id: {item_id}) target '{target_word}' not found.") 
            continue

        # 모든 레이어의 어텐션 값을 각각 스택한 값
        attn_map_stacked = process_attn_per_layer(attns,
                                            prompt_len=prompt_len,      
                                            span_indices=span_indices,  
                                            skip_layer_idx_list=config.skip_layer_idx, 
                                            aggregate=config.aggregate,  
                                            normalize=config.normalize
                                            )

        # 4. 최고 점수 레이어 인덱스 추출
        best_idx, fused_best, grid_mask = select_layer_simple(attn_stack=attn_map_stacked, 
                                                            normalize=config.normalize,
                                                            method=config.method,
                                                            span_reduce=config.span_reduce,
                                                            temperature=config.temperature,
                                                            mask_percentile=config.mask_percentile,
                                                            )
        print(f"### Best layer_idx: {best_idx}")

        # 5. 마스크 확인
        os.makedirs(config.save_dir, exist_ok=True) 
        save_mask_as_png(grid_mask, save_path=os.path.join(config.save_dir, f"mask_{target_word}.png"))
        # overlay_mask_on_image(image=image, mask=grid_mask, save_path=os.path.join(config.save_dir, f"overlay_{target_word}.png"))