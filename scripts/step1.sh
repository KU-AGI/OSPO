#!/bin/bash
# ./scripts/step1.sh

GPU=8
SCRIPT="/home/yjoh/project/OSPO/ospo/step1.py"


# categories=("color1" "color2" "texture1" "texture2" "shape1" "shape2")
categories=("numeracy1" "numeracy2")
cfgs="/home/yjoh/project/OSPO/configs/iter2/step1.yaml"

for category in "${categories[@]}"; do
    echo "====== Running category: $category ======"

    CUDA_VISIBLE_DEVICES=$GPU python $SCRIPT \
        --category "$category" \
        --cfg_path "$cfgs"

    echo "====== Finished category: $category ======"
    echo ""
done