{
  CUDA_VISIBLE_DEVICES=0 python3 eval/infer_mjhq.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b \
    --prompt_file MJHQ-30K/meta_data.json \
    --result_dir output_images --idx 0 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
{
  CUDA_VISIBLE_DEVICES=1 python3 eval/infer_mjhq.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b \
    --prompt_file MJHQ-30K/meta_data.json \
    --result_dir output_images --idx 1 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
{
  CUDA_VISIBLE_DEVICES=2 python3 eval/infer_mjhq.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b \
    --prompt_file MJHQ-30K/meta_data.json \
    --result_dir output_images --idx 2 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
{
  CUDA_VISIBLE_DEVICES=3 python3 eval/infer_mjhq.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b \
    --prompt_file MJHQ-30K/meta_data.json \
    --result_dir output_images --idx 3 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
{
  CUDA_VISIBLE_DEVICES=4 python3 eval/infer_mjhq.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b \
    --prompt_file MJHQ-30K/meta_data.json \
    --result_dir output_images --idx 4 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
{
  CUDA_VISIBLE_DEVICES=5 python3 eval/infer_mjhq.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b \
    --prompt_file MJHQ-30K/meta_data.json \
    --result_dir output_images --idx 5 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
{
  CUDA_VISIBLE_DEVICES=6 python3 eval/infer_mjhq.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b \
    --prompt_file MJHQ-30K/meta_data.json \
    --result_dir output_images --idx 6 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
{
  CUDA_VISIBLE_DEVICES=7 python3 eval/infer_mjhq.py \
    --unitok_path unitok_tokenizer.pth \
    --mllm_path unitok_liquid_7b \
    --prompt_file MJHQ-30K/meta_data.json \
    --result_dir output_images --idx 7 --cfg_scale 7.0 --tau 1.0 --topk 0 --topp 1.0
} &
wait

python3 minigemini/eval/eval_mjhq.py \
  --src_dir MJHQ-30K/mjhq30k_imgs_256/ \
  --result_dir output_images/MJHQ_CFG7.0_topk0_topp1.0_tau_1.0/