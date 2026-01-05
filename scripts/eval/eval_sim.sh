#!/bin/bash

model_dir='checkpoints'

CUDA_VISIBLE_DEVICES=0 python eval_sim.py \
    --checkpoint ${model_dir}/pusht.ckpt \
    --output_dir ${model_dir}/pusht

# CUDA_VISIBLE_DEVICES=0 python eval_sim.py \
#     --checkpoint ${model_dir}/pusht_multitask.ckpt \
#     --output_dir ${model_dir}/pusht_multitask

# CUDA_VISIBLE_DEVICES=0 python eval_sim.py \
#     --checkpoint ${model_dir}/libero10.ckpt \
#     --output_dir ${model_dir}/libero10
