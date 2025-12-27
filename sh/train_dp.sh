export WANDB_API_KEY=<YOUR_KEY>
export CUDA_VISIBLE_DEVICES=$1
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
export HYDRA_FULL_ERROR=1
#export WANDB_MODE=offline

python train.py --config-name=train_diffusion_unet_timm_umi_workspace \
	task.dataset_path=data/pick_and_place.zarr.zip \
	training.num_epochs=400 \
	dataloader.batch_size=64 \
	name=pick_and_place \
	# logging.mode=offline 
