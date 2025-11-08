# (참고) /home/yjoh/project/mllm_reasoning/Janus-Pro-Training/evaluation/vlm_eval/vlm_task1_vllm_qwen_async.py

# conda activate internvl
# Requirement: GPUs >= 2 (T1000(5) 주의)

import os
import json
import time
import argparse
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm
from string import Template
import uuid
import inspect

import torch
try:
    # Prefer the arg_utils version to match AsyncLLMEngine expectations
    from vllm.engine.arg_utils import EngineArgs
except ImportError:
    # Fallback for older builds where EngineArgs is exposed at top-level
    from vllm import EngineArgs
from vllm import SamplingParams
from vllm import AsyncLLMEngine
from vllm import EngineArgs
from transformers import AutoProcessor


_VQA_QS_PROMPT_NON_SPATIAL_COMPLEX = Template("""
You are an assistant dedicated to transforming a sentence into several questions. You should first divide it into simple concepts and relations, and then provide the corresponding questions. Avoid using pronouns, such as he, she, it, and they.

A chef is holding a knife and preparing a dish on the stove.
Concepts and relations: a chef, a knife, a dish, the stove, a chef is holding a knife, a chef is preparing a dish; Questions: Is there a chef? Is there a knife? Is there a dish? Is there a stove? Is a chef holding a knife? Is a chef preparing a dish?

The green teapot is located near the round oak table.
Concepts and relations: a green teapot, a round oak table, the green teapot is near the round oak table, the round oak table is near the green teapot; Questions: Is there a green teapot? Is there a round oak table? Is the green teapot near the round oak table? Is the round oak table near the green teapot?

The chunky wooden lamp casts a warm glow on the tattered blue curtains.
Concepts and relations: a chunky wooden lamp, a warm glow, tattered blue curtains, a chunky wooden lamp casts a warm glow, the warm glow is on the tattered blue curtains; Questions: Is there a chunky wooden lamp? Is there a warm glow? Are there tattered blue curtains? Is a chunky wooden lamp casting a warm glow? Is the warm glow on the tattered blue curtains?

The vibrant orange tomato sat atop the crisp green leaf and the juicy red watermelon.
Concepts and relations: a vibrant orange tomato, a crisp green leaf, a juicy red watermelon, a vibrant orange tomato is atop a crisp green leaf, a vibrant orange tomato is atop a juicy red watermelon; Questions: Is there a vibrant orange tomato? Is there a crisp green leaf? Is there a juicy red watermelon? Is the vibrant orange tomato atop the crisp green leaf? Is the vibrant orange tomato atop the juicy red watermelon?

$prompt
""".strip())


@dataclass
class GenConfig:
    temperature: float = 1.0
    max_tokens: int = 1024
    n: int = 1  # number of samples per prompt


class VLMEvaluatorAsync:
    """
    Uses vLLM AsyncLLMEngine to exploit continuous batching across *all* prompts.
    """

    def __init__(
        self,
        engine: AsyncLLMEngine,
        processor,  # kept for future extensibility; not used in this text-only example
        temperature: float = 1.0,
        max_tokens: int = 1024,
        n: int = 1,
    ):
        self.engine = engine
        self.processor = processor
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
        )

    @staticmethod
    def format_prompt(sample: Dict[str, Any]) -> str:
        return _VQA_QS_PROMPT_NON_SPATIAL_COMPLEX.substitute(
            prompt=sample["prompt"],
        )


    @staticmethod
    def post_process(output_text: str) -> str:
        # Simple pass-through; customize if you need stricter parsing
        return output_text.strip()

    async def _await_request_output(self, req):
        """
        vLLM async API compatibility:
        - If req is an async generator (streaming), iterate and return the last item.
        - If req is an awaitable/coroutine, just await it.
        """
        if inspect.isasyncgen(req):
            last = None
            async for item in req:
                last = item
            return last
        return await req



    async def generate_one(self, prompt: str, request_id: str) -> Dict[str, Any]:
        """
        Submit a single prompt and await the result (handles streaming + non-streaming).
        Returns {'text': "..."} or {'error': "..."}.
        """
        try:
            req = self.engine.generate(
                prompt=prompt,
                sampling_params=self.sampling_params,
                request_id=request_id,
                # If your version supports this and you prefer non-streaming:
                # stream=False,
            )
            req_out = await self._await_request_output(req)

            # Some versions can yield None if aborted; guard it:
            if req_out is None or not getattr(req_out, "outputs", None):
                return {"text": ""}

            text = req_out.outputs[0].text
            return {"text": self.post_process(text)}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}


    async def generate_many(
        self,
        prompts: List[str],
        max_concurrency: int = 64,
        progress_bar: bool = True,
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_concurrency)
        results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)

        async def worker(i: int, p: str):
            async with semaphore:
                rid = f"req-{i}-{uuid.uuid4().hex[:8]}"
                res = await self.generate_one(p, rid)
                results[i] = res

        tasks = [asyncio.create_task(worker(i, p)) for i, p in enumerate(prompts)]

        if progress_bar:
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
                await f
        else:
            await asyncio.gather(*tasks)

        return results  # type: ignore



# ---------- Utilities ----------
def load_json_list(path: str, s_idx: Optional[int], e_idx: Optional[int]) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")
    if s_idx is not None or e_idx is not None:
        data = data[s_idx:e_idx]
    return data


def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def get_dataloader(args): #data_path, task=1, batch_size=1):
    with open(args.data_path, "r") as f:
        data = json.load(f)
    print(f"Full Data length: {len(data)}")
    
    if args.e_idx is not None:
        assert args.e_idx <= len(data), f"e_idx is too large! {args.e_idx} > {len(data)}"
    
    if args.s_idx is not None and args.e_idx is not None:
        data = data[args.s_idx:args.e_idx]
    elif args.s_idx is not None:
        data = data[args.s_idx:]
    elif args.e_idx is not None:
        data = data[:args.e_idx]
    else:
        pass
    print(f"Split Data length: {len(data)}")

    # 기본 데이터 사용
    eval_dataloader = DataLoader(
            dataset=data, 
            shuffle=False,
            collate_fn=lambda batch: batch, # collate_fn,            
            batch_size=args.batch_size,    
            )

    return eval_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Async vLLM batch evaluator")
    parser.add_argument("--model_path", type=str, default="/nas2/mllm_reasoning/checkpoints/InternVL3_5-38B-HF")
    parser.add_argument("--tensor_parallel_size", type=int, default=2) # = GPU size
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_concurrency", type=int, default=64, help="Concurrent requests cap")
      
    # example data
    parser.add_argument("--data_path", type=str, default="/nas2/data/Janus_dataset/next_v2/init_dataset_16001.json")
    parser.add_argument("--save_dir", type=str, default="/nas2/data/Janus_dataset/next_v2/ablation/wo_dense/vqa_expert")
    parser.add_argument("--save_name", type=str, default="non_spatial_complex.json")
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--s_idx", type=int, default=None)
    parser.add_argument("--e_idx", type=int, default=None)
    
    return parser.parse_args()


async def async_main(args):
    # Load data
    data = load_json_list(args.data_path, args.s_idx, args.e_idx)
    print(f"Loaded {len(data)} samples.")

    # Build prompts
    prompts = []
    for sample in data:
        # if "prompt" not in sample:
        #     raise KeyError("Each sample must contain 'prompt' and 'tuple' keys.")
        prompts.append(_VQA_QS_PROMPT_NON_SPATIAL_COMPLEX.substitute(prompt=sample["prompt"]))

    # (Optional) load processor (e.g., for future image+text flows)
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True) # args.trust_remote_code)
    except Exception:
        processor = None  # Not strictly needed for pure text prompts

    # Create async engine (maximizes continuous batching across all requests)
    engine_args = EngineArgs(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True, # args.trust_remote_code,
        dtype="bfloat16"        # args.dtype,
    )

    # Compatibility shim: older/newer EngineArgs may not have this attribute
    if not hasattr(engine_args, "enable_log_requests"):
        setattr(engine_args, "enable_log_requests", False)

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # TODO
    evaluator = VLMEvaluatorAsync(
        engine=engine,
        processor=processor,
        temperature=1.0, # args.temperature,
        max_tokens=512, # args.max_tokens,
        n=1 # args.n,
    )

    # Run
    t0 = time.time()
    results = await evaluator.generate_many(
        prompts=prompts,
        max_concurrency=args.max_concurrency,
        progress_bar=True,
    )
    t1 = time.time()
    print(f"Generation done in {t1 - t0:.2f}s")

    # Attach results back to original samples
    out = []
    for sample, res in zip(data, results):
        sample = dict(sample)  # shallow copy
        if "error" in res:
            sample["question"] = res["error"]
        else:
            sample["question"] = res["text"]
        out.append(sample)

    save_json(args.save_path, out)
    print(f"Saved {len(out)} results to {args.save_path}")


def main():
    args = parse_args()

    # vLLM prefers CUDA visible; quick sanity
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. vLLM performance will be poor or may fail on CPU.")

    os.makedirs(args.save_dir, exist_ok=True)
    args.save_path = os.path.join(args.save_dir, args.save_name)

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

