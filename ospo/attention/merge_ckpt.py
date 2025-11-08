# /home/yjoh/project/ospo/yjoh/merge_ckpt_2.py
# 2025/11/05 최종버전

import os, yaml, re
import math
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelSummary
from peft import LoraConfig, get_peft_model

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)
from ospo.utils.model import get_model


# Optional, but we’ll use it if present
try:
    from safetensors.torch import save_file as save_safetensors
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False

# Optional Hub push; guarded at runtime
try:
    from huggingface_hub import create_repo, upload_folder, HfApi
    _HAS_HF_HUB = True
except Exception:
    _HAS_HF_HUB = False


class HFPushToHubWrapper(LightningModule):
    def __init__(self, model, tokenizer, vl_chat_processor, save_path: str,
                max_shard_size: str = "2GB", shard_safetensors: bool = True):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.vl_chat_processor = vl_chat_processor
        self.save_path = save_path
        self.max_shard_size = max_shard_size
        self.shard_safetensors = shard_safetensors

    def test_step(self, batch, batch_idx):    
        pass

    def on_test_epoch_start(self):
        # Make sure we’re in eval mode and don’t accumulate grads.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        print("Start Saving.")

    def on_test_epoch_end(self):
        # 1) Ensure eval (again, for safety)
        self.model.eval()

        # 2) Save locally in HF style
        export_dir = self._export_hf_like()
        print("Local Done.")

        # 3) Optionally push to the Hugging Face Hub
        self._push_to_hub(export_dir)
        print("HF Done.")

        # rank = getattr(self.trainer, "global_rank", 0)
        rank = getattr(self.trainer, "global_rank")
        if rank == 0:
            print(f"[HFPushToHubWrapper] Export complete: {export_dir}")


    def _export_hf_like(self) -> str:
        """
        Save a HF-like folder:
          - config.json
          - model.safetensors (or pytorch_model.bin)
          - tokenizer files (via tokenizer.save_pretrained)
          - processor_config.json (best-effort for your VLChatProcessor)
          - README.md with minimal instructions
        Returns the absolute path to the export directory.
        """
        # Choose a concrete leaf directory name
        leaf = "hf_export"
        export_dir = Path(self.save_path).expanduser().resolve() / leaf
        if export_dir.exists():
            # Start fresh so repeated runs don’t mix files
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        # ------------------ Save config.json ------------------
        config_dict = self._build_config_dict()
        (export_dir / "config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

        # ------------------ Save model weights ----------------
        state = self._get_state_dict_on_cpu()

        # Always write a PT fallback (transformers will prefer safetensors if present)
        pt_path = export_dir / "pytorch_model.bin"
        torch.save(state, str(pt_path))

        if _HAS_SAFETENSORS:
            if self.shard_safetensors:
                self._save_sharded_safetensors(state, export_dir, self.max_shard_size)
            else:
                meta = {"format": "pt", "framework": "pytorch"}
                save_safetensors(state, str(export_dir / "model.safetensors"), metadata=meta)

            # 2트
            # sf_path = export_dir / "model.safetensors"
            # # Minimal metadata that fixes the NoneType .get crash in transformers
            # meta = {"format": "pt", "framework": "pytorch"}
            # save_safetensors(state, str(sf_path), metadata=meta)

        # 1트
        # if _HAS_SAFETENSORS:
        #     save_path = export_dir / "model.safetensors"
        #     save_safetensors(state, str(save_path))
        # else:
        #     save_path = export_dir / "pytorch_model.bin"
        #     torch.save(state, str(save_path))

        # ------------------ Save tokenizer --------------------
        try:
            self.tokenizer.save_pretrained(str(export_dir))
        except Exception as e:
            print(f"[HFPushToHubWrapper] Warning: tokenizer.save_pretrained failed: {e}")

        # ------------------ Save processor config -------------
        self._save_processor_config(export_dir)

        return str(export_dir)


    def _build_config_dict(self) -> Dict[str, Any]:
        """
        Build a minimal HF-ish config. If your model has `.config` and a
        `to_dict()` method, we’ll use it; otherwise create a reasonable stub.
        """
        # Prefer an existing config if available
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            to_dict = getattr(cfg, "to_dict", None)
            if callable(to_dict):
                try:
                    d = to_dict()
                    # Ensure a few keys that are commonly helpful:
                    d.setdefault("architectures", [self.model.__class__.__name__])
                    d.setdefault("model_type", "multimodal-causal-lm")
                    d.setdefault("is_encoder_decoder", False)
                    d.setdefault("torch_dtype", str(next(self.model.parameters()).dtype).replace("torch.", ""))
                    if self.tokenizer is not None and hasattr(self.tokenizer, "vocab_size"):
                        d.setdefault("vocab_size", int(getattr(self.tokenizer, "vocab_size")))
                    return d
                except Exception:
                    pass

        # Fallback minimal config
        d = None
        return d

    def _get_state_dict_on_cpu(self) -> Dict[str, torch.Tensor]:
        """
        Get a clean state_dict on CPU (no LoRA adapters assumed here because
        you’ve already merged with `merge_and_unload()` upstream).
        """
        # Move to CPU for a portable checkpoint
        cpu_state = {k: v.detach().to("cpu") for k, v in self.model.state_dict().items()}
        return cpu_state


    def _save_processor_config(self, export_dir: Path) -> None:
        """
        Best-effort dump of your VLChatProcessor config.
        If it implements `save_pretrained`, we call it; otherwise we dump a
        JSON from safe (JSON-serializable) attributes.
        """
        proc = self.vl_chat_processor
        if proc is None:
            return

        # If it quacks like HF:
        if hasattr(proc, "save_pretrained"):
            try:
                proc.save_pretrained(str(export_dir))
                return
            except Exception as e:
                print(f"[HFPushToHubWrapper] Warning: vl_chat_processor.save_pretrained failed: {e}")

        # Otherwise: build a conservative JSON out of simple attributes
        def _is_jsonable(x):
            try:
                json.dumps(x)
                return True
            except Exception:
                return False

        raw = {}
        # Common places that hold config-like info
        for attr in ["config", "preprocess_config", "processor_config", "__dict__"]:
            if hasattr(proc, attr):
                val = getattr(proc, attr)
                # Unwrap simple namespaces/objects
                if hasattr(val, "__dict__") and not isinstance(val, dict):
                    val = val.__dict__
                if isinstance(val, dict):
                    for k, v in val.items():
                        # Avoid tensors or callables
                        if callable(v):
                            continue
                        if isinstance(v, torch.Tensor):
                            continue
                        if _is_jsonable(v):
                            raw[k] = v
        if raw:
            (export_dir / "processor_config.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")


    def _push_to_hub(self, export_dir: str) -> None:
        """
        Push to HF Hub if environment has:
        - HF_REPO_ID:  e.g., "username/project-name"
        - HUGGINGFACE_HUB_TOKEN: a valid token (or you’re already logged in)
        """
        repo_id = "ospo_test" # os.getenv("HF_REPO_ID")
        if not repo_id:
            # Silent no-op if user didn’t request hub push.
            return
        if not _HAS_HF_HUB:
            print("[HFPushToHubWrapper] huggingface_hub is not installed; skipping push.")
            return

        # Create repo if needed, then upload folder
        try:
            create_repo(repo_id, exist_ok=True)
            # Upload the whole export dir
            upload_folder(
                repo_id=repo_id,
                folder_path=export_dir,
                path_in_repo=".",  # root of the repo
                ignore_patterns=None,
            )
            print(f"[HFPushToHubWrapper] Pushed to hub: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"[HFPushToHubWrapper] Warning: push to hub failed: {e}")


    # ---------- NEW: sharded safetensors saver ----------
    def _save_sharded_safetensors(self, state: Dict[str, torch.Tensor], export_dir: Path,
                                  max_shard_size: str = "2GB") -> None:
        """
        Save `state` as sharded safetensors with an index file, HF-style:
          - model-00001-of-0000N.safetensors, ...
          - model.safetensors.index.json
        """
        max_bytes = self._parse_size(max_shard_size)  # e.g., "2GB" -> 2 * 1024**3

        # Group (name, tensor) pairs into shards under max_bytes
        pairs: List[Tuple[str, torch.Tensor]] = list(state.items())
        shards: List[List[Tuple[str, torch.Tensor]]] = []
        shard_sizes: List[int] = []

        cur_shard: List[Tuple[str, torch.Tensor]] = []
        cur_size = 0

        for name, tensor in pairs:
            # bytes for this tensor (dtype itemsize * numel)
            tbytes = tensor.element_size() * tensor.numel()
            # If single tensor exceeds cap, force it into its own shard
            if tbytes > max_bytes and cur_shard:
                shards.append(cur_shard); shard_sizes.append(cur_size)
                cur_shard = []; cur_size = 0
            if cur_size + tbytes > max_bytes and cur_shard:
                shards.append(cur_shard); shard_sizes.append(cur_size)
                cur_shard = [(name, tensor)]; cur_size = tbytes
            else:
                cur_shard.append((name, tensor)); cur_size += tbytes

        if cur_shard:
            shards.append(cur_shard); shard_sizes.append(cur_size)

        num_shards = len(shards)
        pad = max(5, int(math.ceil(math.log10(max(2, num_shards + 1)))))  # 00001 width similar to HF

        # Write shards and build weight_map
        weight_map: Dict[str, str] = {}
        total_size = 0
        for i, shard in enumerate(shards, start=1):
            fname = f"model-{str(i).zfill(pad)}-of-{str(num_shards).zfill(pad)}.safetensors"
            fpath = export_dir / fname
            shard_state = {k: v for (k, v) in shard}
            # metadata is optional but nice to have
            meta = {"format": "pt", "framework": "pytorch", "shard": f"{i}/{num_shards}"}
            save_safetensors(shard_state, str(fpath), metadata=meta)

            # update index
            for k, _ in shard:
                weight_map[k] = fname
            total_size += sum(t.element_size() * t.numel() for _, t in shard)

        # Write index JSON (exact keys match HF conventions)
        index = {
            "metadata": {"total_size": int(total_size)},
            "weight_map": weight_map
        }
        (export_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _parse_size(s: str) -> int:
        """
        Parse sizes like '2GB', '1500MB', '512MiB', '800_000_000' into bytes.
        """
        s = s.strip().replace("_", "")
        m = re.fullmatch(r"(?i)\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?\s*", s)
        if not m:
            # assume raw bytes
            return int(float(s))
        val = float(m.group(1))
        unit = (m.group(2) or "").lower()

        # IEC vs SI
        if unit in ("b", ""):
            mult = 1
        elif unit in ("kb",):
            mult = 1000
        elif unit in ("mb",):
            mult = 1000**2
        elif unit in ("gb",):
            mult = 1000**3
        elif unit in ("tb",):
            mult = 1000**4
        elif unit in ("kib",):
            mult = 1024
        elif unit in ("mib",):
            mult = 1024**2
        elif unit in ("gib",):
            mult = 1024**3
        elif unit in ("tib",):
            mult = 1024**4
        else:
            mult = 1
        return int(val * mult)



def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1  # 싱글 프로세스 학습일 경우
    

def get_trainer(device, precision):
    trainer = Trainer(
        accelerator=device,
        devices=get_world_size(), # config.base.world_size,
        strategy="ddp",
        max_epochs=1, # config.experiment.epoch,
        precision=precision,
        callbacks=[ModelSummary(max_depth=2)],
    )
    return trainer


# Dummy Set
def get_dataloader():
    dataset = [i for i in range(10)]
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        collate_fn=lambda batch: batch,
        num_workers=2,  
        drop_last=False
    )
    return dataloader


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/nas2/checkpoints/Janus-Pro-7B")
    parser.add_argument("--cache_dir", type=str, default="/nas2/checkpoints/hf_cache_yj")
    parser.add_argument("--ckpt_path", type=str, default="/nas2/data/Janus_dataset/next/ckpt/1103_next_v2_use_soft_mask_argmax_chosen_v1_simpo_1_beta_5_sft_2_copo_0/version_0/step=000600.ckpt")
    parser.add_argument("--save_path", type=str, default="/nas2/data/Janus_dataset/next/ckpt_iter/iter1_1103_step_600")
    parser.add_argument("--precision", type=str, default="bf16")
    args, unknown = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vl_chat_processor, tokenizer, model = get_model(mode='generate', model_path=args.base_path, cache_dir=args.cache_dir)

    # pytorch-lightning 으로 학습시킨 체크포인트
    # elif config.model.type == "pl": 
    if not os.path.exists(args.ckpt_path):
        raise ValueError("Check args.ckpt_path !")
    
    ckpt_dir = os.path.dirname(args.ckpt_path)
    ckpt_config_path = os.path.join(ckpt_dir, "config.yaml")
    with open(ckpt_config_path, "r") as file:
        ckpt_config = yaml.safe_load(file)

    # Extract LoRA config
    lora_config = LoraConfig(
        r=ckpt_config["lora"].get("lora_rank"),
        lora_alpha=ckpt_config["lora"]["lora_alpha"],
        target_modules=ckpt_config["lora"]["target_modules"],
        lora_dropout=ckpt_config["lora"]["lora_dropout"],
        # bias=ckpt_config["lora"].get("bias"),                                                 # Default to None if missing
        # task_type=None,# "CAUSAL_LM", # ckpt_config["lora"].get("task_type") 오류 발생          # Default to None if missing
        modules_to_save=ckpt_config["lora"].get("modules_to_save")                              # Default to None if missing
    )

    model.language_model = get_peft_model(model.language_model, lora_config)
    model = HFPushToHubWrapper.load_from_checkpoint(checkpoint_path=args.ckpt_path, 
                                                        model=model,
                                                        tokenizer=tokenizer,
                                                        vl_chat_processor=vl_chat_processor, 
                                                        save_path=args.save_path,
                                                        strict=False) 
    model.setup("test")
    model.model.language_model = model.model.language_model.merge_and_unload() 

    trainer = get_trainer(device, args.precision)
    eval_dataloader = get_dataloader()

    trainer.test(model, dataloaders=eval_dataloader)
    print("Done.")