#!/bin/sh
#SBATCH --job-name=uva_multi_node
#SBATCH -p preempt
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=112
#SBATCH --mem=1800G

set -e # Will exit the script if any command returns a non-zero exit code

. $(conda info --base)/etc/profile.d/conda.sh # This is equivalent to source the conda profile

if [ -z "$SLURM_GPUS" ]; then
    export SLURM_GPUS=$((SLURM_GPUS_PER_NODE*SLURM_NNODES))
fi
echo "Using node: $SLURM_NODELIST, total gpu: $SLURM_GPUS"
# Activate conda environment if uva is not activated
if [[ -z "$CONDA_PREFIX" || "$CONDA_PREFIX" != *"uva" ]]; then
    conda activate uva
fi

lz4_data_path= # Path to store all the lz4 compressed data
temp_data_path= # Temporary path to store the extracted data, should be on local storage or in shared memory

dataset_names=cup_arrangement_0,towel_folding_0,mouse_arrangement_0

echo "Extracting data on $SLURM_NNODES nodes: $SLURM_NODELIST"
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES python process_dataset/extract_umi_data.py \
    $dataset_names \
    --data_dir=${lz4_data_path} \
    --output_dir=${temp_data_path}
wait # Until srun is completed


export HYDRA_FULL_ERROR=1
deepspeed_config=unified_video_action/config/zero2.json

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
export WORLD_SIZE=$SLURM_GPUS


# Video pretraining
run_name=uva_video_pretrain
save_checkpoint_path= # Path to save the checkpoints
CMD="accelerate launch \
    --num_processes $SLURM_GPUS \
    --num_machines $SLURM_NNODES \
    --machine_rank \$SLURM_NODEID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_backend static \
    --multi_gpu \
    train.py \
        --config-dir=. \
        --config-name=uva_umi_multi.yaml \
        model.policy.action_model_params.predict_action=False \
        model.policy.selected_training_mode=video_model \
        model.policy.different_history_freq=True \
        model.policy.optimizer.learning_rate=1e-4 \
        task.dataset.dataset_root_dir=${temp_data_path} \
        training.resume=False \
        logging.project=${run_name} \
        hydra.run.dir=${save_checkpoint_path}/${SLURM_JOB_ID} \
        training.deepspeed_config=${deepspeed_config} \
"

## Policy training
# run_name=uva_video_action
# save_checkpoint_path= # Path to save the checkpoints
# pretrain_checkpoint_path= # Path to the pretrained video model checkpoint
# CMD="accelerate launch \
#     --num_processes $SLURM_GPUS \
#     --num_machines $SLURM_NNODES \
#     --machine_rank \$SLURM_NODEID \
#     --main_process_ip $MASTER_ADDR \
#     --main_process_port $MASTER_PORT \
#     --rdzv_backend static \
#     --multi_gpu \
#     train.py \
#         --config-dir=. \
#         --config-name=uva_umi_multi.yaml \
#         model.policy.autoregressive_model_params.pretrained_model_path=${pretrain_checkpoint_path} \
#         model.policy.action_model_params.predict_action=True \
#         model.policy.use_proprioception=True \
#         model.policy.predict_proprioception=True \
#         model.policy.shift_action=False \
#         model.policy.different_history_freq=True \
#         model.policy.optimizer.learning_rate=1e-4 \
#         task.dataset.dataset_root_dir=${temp_data_path} \
#         training.resume=False \
#         logging.project=${run_name} \
#         hydra.run.dir=${save_checkpoint_path}/${SLURM_JOB_ID} \
#         training.deepspeed_config=${deepspeed_config} \
# "

echo $CMD
srun bash -c "$CMD"