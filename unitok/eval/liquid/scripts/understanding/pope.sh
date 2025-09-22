#!/bin/bash


python -m eval.model_vqa_loader \
  --model-path unitok_liquid_7b.pth \
  --tokenizer-path unitok_tokenizer.pth \
  --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/unitok_liquid_7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
    
python /eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/unitok_liquid_7b.jsonl
