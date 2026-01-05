# !/bin/bash

model_dir='checkpoints'

CUDA_VISIBLE_DEVICES=0 python eval_real.py \
    -i ${model_dir}/umi_multitask.ckpt \
    --output_dir vis --port 8768
    