import os
import argparse
import numpy as np
import PIL.Image
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pytorch_lightning import seed_everything

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from janus.models import MultiModalityCausalLM, VLChatProcessor
from ospo.constant import PATCH_SIZE, IMG_SIZE, IMAGE_TOKEN_NUM_PER_IMAGE

class JanusInferenceWrapper():
    def __init__(self, model, vl_chat_processor, save_path=None):
        self.model = model
        self.vl_chat_processor = vl_chat_processor
        self.save_path = save_path
        
    
    def get_prompt(self, text):
        conversation = [
            {
                "role": "<|User|>",
                "content": text,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_start_tag
        return prompt

    
    @torch.inference_mode()
    def generate(self, 
                text: str,
                temperature: float = 1,
                parallel_size: int = 16,
                cfg_weight: float = 5):

        prompt = self.get_prompt(text)
        fname = f"{text.strip()}.png"
        save_path = os.path.join(self.save_path, fname)

        input_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size*2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, IMAGE_TOKEN_NUM_PER_IMAGE), dtype=torch.int).cuda()

        for i in range(IMAGE_TOKEN_NUM_PER_IMAGE):
            outputs = self.model.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            
            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = self.model.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, IMG_SIZE//PATCH_SIZE, IMG_SIZE//PATCH_SIZE])
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        for i in range(parallel_size):
            # save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
            PIL.Image.fromarray(visual_img[i]).save(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="a blue car and red bench")   
    parser.add_argument("--model_path", type=str, default="KU-AGILab/OSPO-Janus-Pro-7B")
    parser.add_argument("--save_path", type=str, default="./results")

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(42, workers=True)

    vl_chat_processor = VLChatProcessor.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model = model.to(torch.bfloat16).cuda().eval()

    start = time.time()

    wrapper = JanusInferenceWrapper(model, vl_chat_processor, args.save_path)
    wrapper.generate(text=args.input)

    end = time.time()
    elapsed_time = (end - start) / 60  # Convert seconds to minutes
    print(f"Time elapsed: {elapsed_time:.2f} minutes")


if __name__ == "__main__":
    main()