# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

ckpt_path=$1
echo Path: $ckpt_path

export FLEXIV_VERBOSE=1
export FLEXIV_INIT_POSE="[-0.36204,-0.511710,0.741515,2.273541,0.776503,2.2775321,-0.75361437]"  # common pickplace
export FLEXIV_MAX_VEL=0.2
export FLEXIV_MAX_ACC=0.2

python eval_real_flexiv_simple.py --policy umi_dp --ckpt_path ${ckpt_path} \
    --robot_frequency 5 --steps_per_inference 6 --action_latency 0.2
