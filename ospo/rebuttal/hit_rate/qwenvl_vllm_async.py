# README: 각 Iteration 의 OSPO data 에 대해 VQA 진행


import torch
from transformers import AutoProcessor
import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
import os
import argparse
import random
import json
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
import re
import asyncio
from asyncio import Queue, Lock
from typing import Any, AsyncGenerator, List, Dict
from qwen_vl_utils import process_vision_info

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)


# 고정
async def get_final_result(generator: AsyncGenerator[vllm.RequestOutput, None]) -> vllm.RequestOutput:
    final_output = None
    async for result in generator:
        final_output = result
    return final_output


def extract_json_from_text(text):
    """
    모델 출력 텍스트에서 JSON 객체를 추출하고 파싱합니다.
    """
    try:
        # 마크다운 코드 블록 제거 (```json ... ```)
        cleaned_text = re.sub(r'```json\s*', '', text)
        cleaned_text = re.sub(r'```\s*', '', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        # JSON 파싱
        return json.loads(cleaned_text)
    
    except json.JSONDecodeError:
        # JSON 파싱 실패 시, 텍스트 전체를 detailed_caption에 넣고 나머지는 비움 (Fallback)
        print(f"\n⚠️ JSON Parse Error. Raw output: {text[:100]}...")
        return {
            "category": text,
            "original_prompt": "Parsing Failed",
            "formatted_prompt": "Parsing Failed"
        }

def safe_save_results(results, path):
    temp_path = path + ".tmp"
    try:
        results_to_save = []
        for item in results:
            # 필요한 필드만 추출
            output_item = {
                "item_id": item.get("item_id"),
                "vqa_id": item.get("vqa_id"),
                "t2i_category": item.get("t2i_category", ""),
                "question": item.get("question", ""),
                "img_path": item.get("img_path", ""),
                "self_answer": item.get("self_answer", ""),
                "self_p_yes": item.get("self_p_yes", ""),
                "self_p_no": item.get("self_p_no", ""),
                "qwen_answer": item.get("qwen_answer", ""),
                "hit": item.get("hit", None),
            }
            results_to_save.append(output_item)

        with open(temp_path, 'w', encoding='utf-8') as f:
            # item_id 기준으로 정렬
            sorted_results = sorted(results_to_save, key=lambda x: x.get('item_id', ''))
            json.dump(sorted_results, f, indent=4, ensure_ascii=False)
            
        os.rename(temp_path, path)
        print(f"\nSuccessfully saved {len(results_to_save)} results to {path}")

    except Exception as e:
        print(f"Error during saving: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

# 고정
def create_subset_indices(dataset_size, args):
    # rank=int(os.environ.get("RANK", 0))
    # world_size=int(os.environ.get("WORLD_SIZE", 1))

    if args.split_part is not None:
        part_size = dataset_size // 4
        remainder = dataset_size % 4
        part_sizes = [part_size] * 4
        for i in range(remainder):
            part_sizes[i] += 1
        
        start_idx = sum(part_sizes[:args.split_part])
        end_idx = start_idx + part_sizes[args.split_part]
        indices = list(range(start_idx, end_idx))
        print(f"Dataset split into 4 parts. Using Part {args.split_part}: indices {start_idx}-{end_idx-1}")
    else:
        indices = list(range(dataset_size))
        print(f"Using all {dataset_size} samples")
    return indices
        

def simple_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    out = defaultdict(list)
    for item in batch:
        for k, v in item.items():
            out[k].append(v)
    return dict(out)


class TaskDatset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        with open(args.input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.dataset = []

        # allocate vqa_id
        for item in tqdm(raw_data, desc="Preparing Dataset"):

            # all_img_path_list = []
            for qid, quesetion in enumerate(item["question"]):

                # base_img_path_list = []
                for meta in item["base_metadata"].values():
                    fname = os.path.splitext(os.path.basename(meta["path"]))[0]
                    self.dataset.append({
                        "item_id": item["item_id"],
                        "vqa_id": f"{item['item_id']}_{qid}_base_{fname}",
                        "t2i_category": item["t2i_category"],
                        "question": quesetion,
                        "img_path": meta["path"],
                        "self_answer": meta["answer_metadata"][qid]["answer"],
                        "self_p_yes": meta["answer_metadata"][qid]["p_yes"],
                        "self_p_no": meta["answer_metadata"][qid]["p_no"],
                    })

                # negative_img_path_list = []
                for meta in item["negative_metadata"].values():
                    fname = os.path.splitext(os.path.basename(meta["path"]))[0]
                    self.dataset.append({
                        "item_id": item["item_id"],
                        "vqa_id": f"{item['item_id']}_{qid}_negative_{fname}", # 차이
                        "t2i_category": item["t2i_category"],
                        "question": quesetion,
                        "img_path": meta["path"],
                        "self_answer": meta["answer_metadata"][qid]["answer"],
                        "self_p_yes": meta["answer_metadata"][qid]["p_yes"],
                        "self_p_no": meta["answer_metadata"][qid]["p_no"],
                    })

        self.length = len(self.dataset)
        print("Total dataset size:", self.length)

        # mix the dataset
        random.shuffle(self.dataset) # for reproducibility

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = dict(self.dataset[idx]) # shallow copy
        anchor = sample["img_path"]
        try:
            img = Image.open(anchor).convert("RGB")
            sample['image'] = img

        except Exception as e:
            print(f"[WARNING] Failed to open image: {anchor} ({e})")
            sample['image'] = None

        return sample


def get_loader(args):
    dataset = TaskDatset(args)

    subset_indices = create_subset_indices(len(dataset), args)

    if len(subset_indices) < len(dataset):
        dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=simple_collate_fn,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    print(f"Final dataset size: {len(dataset)}")
    return loader


def get_model(path, tensor_parallel_size):
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    engine_args = AsyncEngineArgs(
        model=path,
        gpu_memory_utilization=0.85, # 0.9,
        tensor_parallel_size=tensor_parallel_size,
        dtype='bfloat16',
        trust_remote_code=True,
        max_model_len=8192, # 16384,
        max_num_seqs=128,   # 배치 처리를 위해 여유있게 설정
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine, processor


def process_prompt(processor, img, question):
    # System Prompt에 지시사항이 있으므로, User Prompt는 이미지를 보라는 단순한 지시만 있으면 됩니다.
    # user_instruction = f"Analyze the image and generate the JSON response as instructed.\nCategory:{category}\nPrompt:{original_prompt}"

    system_prompt = "You are a visual question answering model. Answer the question based on the image provided. Answer ONLY with 'yes' or 'no'. If the question is not relevant to the image, answer 'tie'."
    
    messages = [
        # {"role": "system", "content": PROMPT_AUGMENTATION_SYSTEM}, 
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": question}
            # {"type": "text", "text": user_instruction}
        ]},
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
            'prompt': text_prompt,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }

async def _run_vqa_loop_async(engine: AsyncLLMEngine, processor: AutoProcessor, args: argparse.Namespace, 
                            active_results: dict, processed_keys: set):
    
    # JSON 출력을 받아야 하므로 max_tokens 를 넉넉하게 잡습니다.
    # sampling_params = SamplingParams(n=1, temperature=0.7, max_tokens=1024) 
    sampling_params = SamplingParams(n=1, temperature=0.7, max_tokens=512) 

    loader = get_loader(args)
    
    queue = Queue(maxsize=args.batch_size * 8) 
    lock = Lock()
    
    completed_request_count = 0 
    pbar = tqdm(total=len(loader.dataset), desc="Processing Captioning", initial=len(processed_keys))


    # --- Consumer ---
    async def consumer():
        nonlocal completed_request_count
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            llm_inputs, request_id, unique_key = item
            
            try:
                result_generator = engine.generate(llm_inputs, sampling_params, request_id)
                request_output = await get_final_result(result_generator)
            except Exception as e:
                print(f"\n⚠️ Error during vLLM generate for {request_id}: {e}")
                queue.task_done()
                continue

            if request_output.finished:
                raw_output = request_output.outputs[0].text
                
                # JSON 파싱
                # parsed_json = extract_json_from_text(raw_output)
                if "no" in raw_output.lower():
                    parsed_json = {
                        "qwen_answer": "no",
                    }
                elif "yes" in raw_output.lower():
                    parsed_json = {
                        "qwen_answer": "yes",
                    }
                else:
                    parsed_json = {
                        "qwen_answer": "tie", # TODO
                    }

                async with lock:
                    if unique_key in active_results:
                        # 결과 저장
                        active_results[unique_key]['qwen_answer'] = parsed_json.get('qwen_answer', "")

                        # Hit 판단
                        if active_results[unique_key]['self_answer'].strip().lower() == active_results[unique_key]['qwen_answer'].strip().lower():
                            active_results[unique_key]['hit'] = True
                        else:
                            active_results[unique_key]['hit'] = False
                        
                        # 완료 표시
                        active_results[unique_key]['completed_requests'] = 1
                        
                        pbar.update(1)
                        completed_request_count += 1
                        
                        if completed_request_count % args.save_interval == 0:
                            print(f"\n--- Autosaving at {completed_request_count} completed requests ---")
                            safe_save_results(list(active_results.values()), args.save_path)
            
            queue.task_done()

    # --- Producer ---
    async def producer():
        for batch in tqdm(loader, desc="Scheduling Requests", leave=False):
            batch_size_curr = len(batch['vqa_id'])
            
            for i in range(batch_size_curr):
                # 원본 아이템 복원 (image 제외)
                original_item = {k: batch[k][i] for k in batch.keys() if k != 'image'}
                
                vqa_id = original_item['vqa_id']
                
                # [추가됨] 원본 프롬프트 가져오기
                question = original_item.get('question', "") 
                img_path = original_item.get('img_path', "")
                self_answer = original_item.get('self_answer', "")
                t2i_category = original_item.get('t2i_category', "")
                # aligned_data = original_item.get('aligned_data', {})
                # img_path = aligned_data.get('img_path')
                # unique_key = f"{item_id}_{img_path}"

                unique_key = vqa_id
                if unique_key in processed_keys:
                    continue 

                async with lock:
                    if unique_key not in active_results:
                        # 필요한 기본 정보만 active_results에 저장
                        active_results[unique_key] = {
                            "item_id": original_item.get("item_id", ""),
                            "vqa_id": vqa_id,
                            "t2i_category": t2i_category,

                            "question": question,
                            "img_path": img_path,
                            "self_answer": self_answer,
                            "self_p_yes": original_item.get("self_p_yes", ""),
                            "self_p_no": original_item.get("self_p_no", ""),

                            "total_requests": 1,
                            "completed_requests": 0,

                            # 결과 담을 필드 초기화
                            "qwen_answer": None,
                            "hit": None,
                        }
                
                # 이미 작업 완료된 경우 스킵
                if active_results[unique_key].get('completed_requests', 0) == 1:
                    continue

                # 큐에 작업 추가 (이미지 당 1개의 요청)
                img_pil = batch['image'][i]
                
                if img_pil is not None:
                    request_id = f"{unique_key}_vqa"

                    llm_inputs = process_prompt(processor, img_pil, question=question)
                    item = (llm_inputs, request_id, unique_key)
                    await queue.put(item)

        for _ in range(args.num_consumers):
            await queue.put(None)

    # num_consumers = 16
    consumers = [asyncio.create_task(consumer()) for _ in range(args.num_consumers)]
    
    await producer()
    await queue.join()
    await asyncio.gather(*consumers)
    
    pbar.close()
    
    return list(active_results.values())


async def main():
    args = parse_args()

    active_results = {}
    processed_keys = set()

    # 파일 이름에 split_part 반영
    if args.split_part is not None:
        args.save_path = args.save_path.replace('.json', f"_part{args.split_part}.json")
    
    # TODO: 필요시 고려 (resume process)
    if os.path.exists(args.save_path):
        print(f"Loading existing results from {args.save_path}...")
        try:
            with open(args.save_path, 'r', encoding='utf-8') as f:
                existing_results_list = json.load(f)
            
            if isinstance(existing_results_list, list):
                for item in existing_results_list:
                    item_id = item.get('item_id')
                    # 저장된 파일에는 img_path가 없을 수 있으므로 unique_key 생성을 위해 item_id 사용
                    # (하지만 재시작 안정성을 위해 producer 로직에서 img_path를 결합하는 방식과 맞춰야 함)
                    # 여기서는 저장된 파일 구조가 간소화되었으므로 prompt_id만으로 resume 체크가 가능한지 확인이 필요함.
                    # 다만 기존 로직과의 호환성을 위해 unique_key 재구성이 필요하다면 원본 데이터와 매핑이 필요할 수 있음.
                    # 간단하게 detailed_caption이 있으면 완료된 것으로 간주합니다.
                    
                    if item_id and item.get('detailed_caption'):
                         # 주의: 여기서는 unique_key를 정확히 복원하기 어려울 수 있으므로(img_path 부재),
                         # 아래 producer에서 unique_key를 만들 때 item_id 같다면 skip하는 로직이 필요할 수 있습니다.
                         # 본 코드에서는 일단 prompt_id를 키로 사용하는 것으로 간주하거나, 
                         # save 파일에 img_path도 임시로 넣어두는 것이 resume에 유리합니다.
                         # 현재 로직: prompt_id만 있으면 해당 prompt_id는 처리된 것으로 간주 (Set에 추가)
                         processed_keys.add(item_id)
                         active_results[item_id] = item # Key를 item_id로 단순화하여 로딩
                         
                print(f"Resuming... Found {len(active_results)} completed items.")
        
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            print(f"Warning: Could not decode {args.save_path} ({e}). Starting from scratch.")
            active_results = {}
            processed_keys = set()

    model, processor = get_model(args.model_path, args.tensor_parallel_size)
    
    # Resume 로직 보완을 위해 processed_keys 전달 방식 조정
    # Producer에서 unique_key를 만들 때 prompt_id가 processed_keys에 있으면 skip 하도록 수정 필요
    # 이를 위해 producer 내부에서 processed_keys 확인 로직을 `item_id in processed_keys` 형태로 사용할 수 있게 함.
    
    results = await _run_vqa_loop_async(model, processor, args, active_results, processed_keys)
    
    # 결과 정렬
    if results:
        results.sort(key=lambda item: item.get('item_id', ''))
        
    print("\nProcessing complete. Performing final save.")
    safe_save_results(results, args.save_path)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/nas2/mllm_reasoning/checkpoints/Qwen3-VL-32B-Instruct")
    parser.add_argument("--input_path", type=str, default="/nas2/checkpoints/janus_dpo_rebuttal/data/iter3/data/prompt/step3/vqa_result_27438.json") # from iter0 (Base Model)
    # /nas2/data/Janus_dataset/next_v2/ablation/pair_size/vqa_result_pair_size_5.json

    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--num_consumers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16) # 8)
    parser.add_argument("--save_path", type=str, default="/home/yjoh/project/OSPO/ospo/rebuttal/hit_rate/iter3_qwen_vqa.json")
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--split_part", type=int, choices=[0, 1, 2, 3], default=None,
                       help="Split dataset into 4 parts and select one (0, 1, 2, or 3)")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    asyncio.run(main())
