# GPUS=0,1,2,3,4,5,6,7
GPUS=4,5,6,7,8,9
STEPS=(0.5 1) # 800 900) # 1000 별도

for STEP in "${STEPS[@]}"; do
    echo "Evaluation with STEP = $STEP"

    EXP_NAME="1117_ablation_beta_${STEP}"
    echo "Running EXP: $EXP_NAME"

    CUDA_VISIBLE_DEVICES=$GPUS python /home/yjoh/project/OSPO/eval/geneval_gen_batched.py \
    --cfg_path="/home/yjoh/project/OSPO/configs/eval/geneval.yaml" \
      base.world_size=6 \
      base.exp_name="${EXP_NAME}/ckpt_800" \
      model.ckpt_path="/nas2/data/Janus_dataset/next/ckpt/appendix/1117_ablation_beta_${STEP}/version_0/step=000800.ckpt" \

done