#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 -m eval.model_vqa_loader \
    --model-path unitok_liquid_7b.pth \
    --tokenizer-path unitok_tokenizer.pth \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/unitok_liquid_7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment unitok_liquid_7b

cd eval_tool

python calculation.py --results_dir answers/unitok_liquid_7b
