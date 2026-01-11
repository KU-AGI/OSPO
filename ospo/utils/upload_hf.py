import os
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# from huggingface_hub import login
# from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/nas2/checkpoints/hf_cache_yj/Llama-3.2-1B-base-lora_lora_merge_fp16"

# repo_id = "dhdbsrlw/Llama_3_2_1B_Prune3_LoRA"
# repo_id = "dhdbsrlw/Llama_3_2_1B_AWQ_4bit"
repo_id = "dhdbsrlw/Llama_3_2_1B_TinyStories_LoRA_SFT"

# model = AutoModelForCausalLM.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# # out_dir = "./export_for_hub"
# # model.save_pretrained(out_dir, safe_serialization=True)   # writes .safetensors if possible
# # tokenizer.save_pretrained(out_dir)

# repo_id = "dhdbsrlw/Llama_3_2_1B_Prune3_LoRA"
# model.push_to_hub(repo_id, private=True)
# tokenizer.push_to_hub(repo_id)


from huggingface_hub import login, create_repo, upload_folder

login(token=os.environ["HF_TOKEN"])

create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

upload_folder(
    folder_path=model_path,  # e.g., a Trainer checkpoint folder
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload checkpoint folder",
)