## Liquid

### Setup
Please download our [pretrained MLLM weights]() and install additional packages below.
```bash
cd eval/liquid
```

```bash
pip install transformers==4.39.2 sentencepiece==0.1.99
pip install deepspeed==0.12.6 accelerate==0.27.2
pip install flash-attn --no-build-isolation
pip install scikit-learn==1.2.2
pip install open_clip_torch
```



### Understanding Evaluation

Please follow [instructions](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)
in LLaVA to download VQA benchmarks used for evaluation. 
Then configure the model and data paths in scripts under `liquid/scripts/understanding`.

### Generation Evaluation

Please follow [instructions](https://github.com/FoundationVision/Liquid/blob/main/evaluation/EVAL.md#text-to-image-evaluation)
in Liquid to download GenAI-Bench and MJHQ benchmarks used for evaluation.
Then configure the model and data paths in scripts under `liquid/scripts/generation`.

## LLaVA

### Setup

```bash
cd eval/llava
```

Please follow the instructions in [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#pretrain-feature-alignment)
to download the data for alignment pretraining and instruction finetuning.
Then install the packages required for training:
```bash
 pip install transformers==4.37.2 deepspeed==0.12.6 peft==0.13.2
 pip install flash-attn --no-build-isolation
 pip install sentencepiece==0.1.99
 pip install accelerate==0.21.0
 pip install scikit-learn==1.2.2
```

### LLaVA Training

Alignment Pretraining
```bash
bash scripts/v1_5/pretrain.sh \
    --vision_tower /path/to/unitok/ckpt \
    --data_path /path/to/blip_laion_cc_sbu_558k.json \
    --image_folder /path/to/blip_laion_cc_sbu_558k/imgs \
    --custom_encoder True --quantize True
```

Instruction Finetuning
```bash
bash scripts/v1_5/finetune.sh \
    --vision_tower /path/to/unitok/ckpt \
    --data_path /path/to/llava_v1_5_mix665k.json \
    --image_folder /path/to/llava_v1_5_mix665k/imgs \
    --custom_encoder True --quantize True
```

### LLaVA Evaluation

Please follow [instructions](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)
in LLaVA to evaluate the model on various VQA benchmarks.


## LlamaGen

### Setup
```bash
cd eval/llamagen
```

Download [Imagenet](https://image-net.org/download.php) for class-conditional image generation training. 
Then extract the VQ codes:
```
bash scripts/autoregressive/extract_codes_c2i.sh \
    --vq-ckpt /path/to/unitok/ckpt \
    --data-path /path/to/imagenet/train \
    --code-path /path/to/save/imagenet_code_c2i_flip_ten_crop \
    --ten-crop --crop-range 1.1 --image-size 256
```

### LlamaGen Training
The following instructions apply to generation **without CFG**. 
Before getting started, please make sure to update the code if you are not using the latest commit.

Before running, please configure  `nnodes, nproc_per_node, node_rank, master_addr, master_port` in `train_c2i.sh`.
```bash
bash scripts/autoregressive/train_c2i.sh 
    --cloud-save-path /path/to/cloud_disk \
    --code-path /path/to/imagenet_code_c2i_flip_ten_crop \
    --num-output-layer 4 --gpt-model GPT-XXL \
    --num-codebooks 8 --vocab-size 32768 --image-size 256 \
    --global-batch-size 1024 --lr 3e-4 --schedule 'cosine' --class_dropout_prob 0
```
**Note:** To fulfill the potential of UniTok, we suggest using GPT-XXL or larger generators for LlamaGen Training.

### LlamaGen Sampling

```bash
bash scripts/autoregressive/sample_c2i.sh \
    --vq-ckpt /path/to/unitok/ckpt \
    --gpt-ckpt /path/to/llamagen/ckpt \
    --gpt-model GPT-XXL --num-output-layer 4 \
    --num-codebooks 8 --codebook-size 32768 \
    --image-size 256 --cfg-scale 1.0
```
For FID evaluation, please follow [instructions](https://github.com/FoundationVision/LlamaGen/blob/main/evaluations/c2i/README.md)
from LlamaGen to install required packages and download reference images.