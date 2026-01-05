#!/bin/sh

task_name='pusht'

## Train video generation model
accelerate launch --num_processes=8 train.py \
    --config-dir=. \
    --config-name=uva_pusht.yaml \
    model.policy.action_model_params.predict_action=False \
    model.policy.selected_training_mode=video_model \
    logging.project=uva_${task_name} \
    hydra.run.dir="checkpoints/pusht_video"
    

# ## Train video and action model
# accelerate launch --num_processes=8 train.py \
#     --config-dir=. \
#     --config-name=uva_pusht.yaml \
#     model.policy.autoregressive_model_params.pretrained_model_path="checkpoints/pusht_video/checkpoints/latest.ckpt" \
#     model.policy.action_model_params.predict_action=True \
#     logging.project=uva_${task_name} \
#     hydra.run.dir="checkpoints/pusht_video2_action"
    