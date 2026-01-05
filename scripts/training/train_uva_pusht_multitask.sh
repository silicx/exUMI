#!/bin/sh

task_name='pusht_multitask'

## Train video generation model
accelerate launch --num_processes=8 train.py \
    --config-dir=. \
    --config-name=uva_pusht.yaml \
    model.policy.action_model_params.predict_action=False \
    model.policy.selected_training_mode=video_model \
    task.dataset.dataset_path=data/pusht_multitask \
    task.dataset.dataset_type=multitask \
    task.env_runner.fix_goal=False \
    logging.project=uva_${task_name} \
    hydra.run.dir="checkpoints/pusht_multitask_video"


# ## Train video and action model
# accelerate launch --num_processes=8 train.py \
#     --config-dir=. \
#     --config-name=uva_pusht.yaml \
#     model.policy.autoregressive_model_params.pretrained_model_path="checkpoints/pusht_multitask_video/checkpoints/latest.ckpt" \
#     model.policy.action_model_params.predict_action=True \
#     task.dataset.dataset_path=data/pusht_multitask \
#     task.dataset.dataset_type=multitask \
#     task.env_runner.fix_goal=False \
#     logging.project=uva_${task_name} \
#     hydra.run.dir="checkpoints/pusht_multitask_video_action"

