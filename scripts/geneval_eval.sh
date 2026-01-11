GPUS=5
STEPS=(200 300 400 500 600 700 800 900) # 1000 별도

for STEP in "${STEPS[@]}"; do
    echo "Evaluation with STEP = $STEP"

    CUDA_VISIBLE_DEVICES=$GPUS python /home/yjoh/project/geneval/evaluation/evaluate_images.py \
      /nas2/data/Janus_dataset/next/eval_iter2_final/1117_train_beta_5_lr_1e6_max_step_1000_sft_weight_3/ckpt_${STEP}/geneval/gen \
      --outfile="/nas2/data/Janus_dataset/next/eval_iter2_final/1117_train_beta_5_lr_1e6_max_step_1000_sft_weight_3/ckpt_${STEP}/geneval/results.jsonl" \
      --model-path="/home/yjoh/project/geneval/mask2former" \

done