#!/bin/bash

{
  CUDA_VISIBLE_DEVICES=0 python3 eval/infer_genai.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b/ \
    --prompt_file GenAI-Bench/GenAI-Bench-527/prompts.txt \
    --result_dir output_images --idx 0 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
{
  CUDA_VISIBLE_DEVICES=1 python3 eval/infer_genai.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b/ \
    --prompt_file GenAI-Bench/GenAI-Bench-527/prompts.txt \
    --result_dir output_images --idx 1 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
{
  CUDA_VISIBLE_DEVICES=2 python3 eval/infer_genai.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b/ \
    --prompt_file GenAI-Bench/GenAI-Bench-527/prompts.txt \
    --result_dir output_images --idx 2 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
{
  CUDA_VISIBLE_DEVICES=3 python3 eval/infer_genai.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b/ \
    --prompt_file GenAI-Bench/GenAI-Bench-527/prompts.txt \
    --result_dir output_images --idx 3 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
{
  CUDA_VISIBLE_DEVICES=4 python3 eval/infer_genai.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b/ \
    --prompt_file GenAI-Bench/GenAI-Bench-527/prompts.txt \
    --result_dir output_images --idx 4 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
{
  CUDA_VISIBLE_DEVICES=5 python3 eval/infer_genai.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b/ \
    --prompt_file GenAI-Bench/GenAI-Bench-527/prompts.txt \
    --result_dir output_images --idx 5 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
{
  CUDA_VISIBLE_DEVICES=6 python3 eval/infer_genai.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b/ \
    --prompt_file GenAI-Bench/GenAI-Bench-527/prompts.txt \
    --result_dir output_images --idx 6 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
{
  CUDA_VISIBLE_DEVICES=7 python3 eval/infer_genai.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b/ \
    --prompt_file GenAI-Bench/GenAI-Bench-527/prompts.txt \
    --result_dir output_images --idx 7 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
wait

python3 eval/eval_genai.py \
  --prompt_file GenAI-Bench/GenAI-Bench-527/prompts.txt \
  --tag_file GenAI-Bench/GenAI-Bench-527/genai_skills.json \
  --image_dir output_images/GenAI-cfg_7.0-topk_0-topp_1.0-tau_1.0/ \

