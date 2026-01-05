export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline
export HYDRA_FULL_ERROR=1

accelerate launch --num_processes=4 --main_process_port 11451 train.py \
    --config-dir=. \
    --config-name=uva_umi_tactile.yaml \
    task.dataset.dataloader_cfg.batch_size=8 \
    training.gradient_accumulate_every=2 \
    training.num_epochs=100 \
    model.policy.selected_training_mode=video_model \
    model.policy.optimizer.learning_rate=1e-4 \
    logging.project=tpp_tactile \
    logging.mode=offline \
    hydra.run.dir="checkpoints/tpp_tactile"