"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import torch
import os
import sys
import hydra
from omegaconf import OmegaConf
import pathlib
from unified_video_action.workspace.base_workspace import BaseWorkspace
from omegaconf import open_dict

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


import wandb

if "WANDB_API_KEY" in os.environ:
    wandb.login(key=os.environ["WANDB_API_KEY"])


@hydra.main(
    version_base=None,
    config_path=str(
        pathlib.Path(__file__).parent.joinpath("unified_video_action", "config")
    ),
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)

    # if cfg.model.policy.action_model_params.predict_action == False:
    #     cfg.checkpoint.topk.monitor_key = "video_fvd"
    #     cfg.checkpoint.topk.format_str = (
    #         "epoch={epoch:04d}-video_fvd={video_fvd:.3f}.ckpt"
    #     )
    #     cfg.checkpoint.topk.mode = "min"

    with open_dict(cfg):
        cfg.n_gpus = torch.cuda.device_count()
        cfg.model.policy.debug = cfg.training.debug

    if cfg.training.debug:
        cfg.dataloader.batch_size = 2
        cfg.val_dataloader.batch_size = 2
        cfg.dataloader.shuffle = False
        cfg.val_dataloader.shuffle = False

        if "env_runner" in cfg.task:
            cfg.task.env_runner.max_steps = 20

        if "dataloader_cfg" in cfg.task.dataset:
            cfg.task.dataset.dataloader_cfg.batch_size = 2

    cls = hydra.utils.get_class(cfg.model._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    print(sys.argv)
    for arg in sys.argv:
        if "local_rank" in arg:  # For deepspeed compatibility
            sys.argv.remove(arg)
    main()
