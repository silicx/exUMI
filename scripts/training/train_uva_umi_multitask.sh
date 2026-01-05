#!/bin/sh

task_name='umi_multi'

## Train video generation model
accelerate launch --num_processes=8 train.py \
    --config-dir=. \
    --config-name=uva_umi_multi.yaml \
    model.policy.action_model_params.predict_action=False \
    model.policy.selected_training_mode=video_model \
    model.policy.different_history_freq=True \
    model.policy.optimizer.learning_rate=1e-4 \
    task.dataset.dataset_root_dir=${dataset_path} \
    task.dataset.used_episode_indices_file=prepared_data/sampled_500_index_3_datasets.json \
    logging.project=uva \
    hydra.run.dir="checkpoints/uva_umi_multitask_video"

# # ## Train video and action model
# accelerate launch --num_processes=8 train.py \
#     --config-dir=. \
#     --config-name=uva_umi_multi.yaml \
#     model.policy.autoregressive_model_params.pretrained_model_path=checkpoints/uva_umi_multitask_video/checkpoints/latest.ckpt \
#     model.policy.action_model_params.predict_action=True \
#     model.policy.use_proprioception=True \
#     model.policy.predict_proprioception=True \
#     model.policy.shift_action=False \
#     model.policy.different_history_freq=True \
#     model.policy.optimizer.learning_rate=1e-4 \
#     task.dataset.dataset_root_dir=${dataset_path} \
#     task.dataset.used_episode_indices_file=prepared_data/sampled_500_index_3_datasets.json \
#     logging.project=uva \
#     hydra.run.dir="checkpoints/uva_umi_multitask_video_action"
