#!/bin/bash
# ./scripts/step1.sh

GPU=2
SCRIPT="/home/yjoh/project/OSPO/ospo/step1.py"

categories=("color1" "color2" "texture1" "texture2" "shape1" "shape2")

for category in "${categories[@]}"; do
    echo "====== Running category: $category ======"

    CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT \
        --category "$category"

    echo "====== Finished category: $category ======"
    echo ""
done