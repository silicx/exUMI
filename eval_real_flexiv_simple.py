import os
import time
import click
import cv2
import dill
import hydra
import numpy as np
import torch
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_umi_obs_dict,
                                                get_real_umi_action)
from umi.common.pose_util import *


import sys
from umi.real_world.flexiv import *
sys.path.append(str(pathlib.Path(__file__).parent.joinpath('flexiv_api/lib_py')))
import flexivrdk

from umi.real_world.flexiv_simple_env import SimpleFlexivEnv, TactileFlexivEnv

import os

@click.command()
@click.option('--policy', '-ip', required=True, help='Name of the policy')
@click.option('--ckpt_path', required=True, help='Path to checkpoint')
@click.option('--config_path', default=None, help='')
@click.option('--action_latency', default=0, type=float, help="")
@click.option('--robot_frequency', default=20, type=float, help="Control frequency in Hz.")
@click.option('--obs_horizon', default=2, type=int, help="")
@click.option('--steps_per_inference', default=5, type=int, help="")
def main(policy, config_path, ckpt_path, action_latency, robot_frequency, steps_per_inference, obs_horizon):
    device = torch.device('cuda')
    robot_dt = 1./robot_frequency
    # ========== load policy ==============
    config_path = ckpt_path if config_path is None else config_path
    if not config_path.endswith('.ckpt'):
        config_path = os.path.join(config_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(config_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr

    print('obs_pose_rep', obs_pose_rep)
    print('action_pose_repr', action_pose_repr)

    
    if policy == "umi_dp":
        assert action_pose_repr == "relative", action_pose_repr
        assert obs_pose_rep in ["relative", "abs"], obs_pose_rep
        # print("model_name:", cfg.policy.obs_encoder.model_name)
        print("dataset_path:", cfg.task.dataset.dataset_path)

        # creating model
        # have to be done after fork to prevent
        # duplicating CUDA context with ffmpeg nvenc
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        policy.num_inference_steps = 16 # DDIM inference iterations
        
        policy.eval().to(device)
        try:
            policy.obs_encoder.key_model_map['tactile_combined'].use_variational = False
        except:
            pass

        policy.reset()
    
    else:
        raise ValueError(f"Unknown policy: {policy}")
        

    # ========== environment initialize ===========
    init_qpos=eval(os.environ.get("FLEXIV_INIT_POSE", "Set the pose in environ"))

    # env = SimpleFlexivEnv(init_qpos, obs_horizon=obs_horizon, use_gripper_width_mapping=False)
    env = TactileFlexivEnv(init_qpos, obs_horizon=obs_horizon, use_gripper_width_mapping=False)


    with KeystrokeCounter() as key_counter:
        while True:
            env.reset()
            policy.reset()
            
            
            obs = env.get_obs()
            episode_start_pose = [
                np.concatenate([
                    obs[f'robot0_eef_pos'],
                    obs[f'robot0_eef_rot_axis_angle']
                ], axis=-1)[-1]
            ]
            
            obs = env.get_obs()


            # ========== policy control loop ==============
            while True:
                s = time.time()

                obs = env.get_obs()

                print("Obs latency", time.time()-s)

                with torch.no_grad():
                    obs_dict_np = get_real_umi_obs_dict(
                        env_obs=obs, shape_meta=cfg.task.shape_meta, 
                        obs_pose_repr=obs_pose_rep,
                        tx_robot1_robot0=None,
                        episode_start_pose=episode_start_pose)
                    
                
                for idx in range(steps_per_inference):
                    # print(obs_dict_np)
                    obs_dict = dict_apply(obs_dict_np, 
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device))  
                    result = policy.predict_action(obs_dict)
                    raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                    print("raw_action", raw_action.shape)
                        

                    def format_arr(x):
                        return ", ".join(["%.4f"%x for x in x])
                    # for idx in range(steps_per_inference):
                    #     print(f"RawAct-{idx}", format_arr(raw_action[idx][:7]), "||",raw_action[idx][-1])
                        
                    action = get_real_umi_action(raw_action, obs, action_pose_repr)
                    # for idx in range(steps_per_inference):
                    #     print(f"ExeAct-{idx}", format_arr(action[idx][:6]), "||",action[idx][-1])
                
                print('Inference latency:', time.time() - s)
            
                this_target_poses = action

                
                this_target_poses = this_target_poses[:steps_per_inference, :]
                action_timestamps = (1+np.arange(len(this_target_poses), dtype=np.float64)
                    ) * robot_dt + time.time() - action_latency
                
                env.exec_actions(
                    actions=this_target_poses,
                    timestamps=action_timestamps
                )
                print(f"Submitted {len(this_target_poses)} steps of actions.")
                

                print('Action latency:', time.time() - s)

                # visualize
                #     vis_img = obs['camera0_rgb'][-1]
                #     text = 'Time: {:.1f}'.format(
                #         time.monotonic() - t_start
                #     )
                #     cv2.putText(
                #         vis_img,
                #         text,
                #         (10,20),
                #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #         fontScale=0.5,
                #         thickness=1,
                #         color=(255,255,255)
                #     )
                #     cv2.imshow('default', vis_img[...,::-1])

                _ = cv2.pollKey()
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='s'):
                        print('Stopped.')
                        break




# %%
if __name__ == '__main__':
    main()
