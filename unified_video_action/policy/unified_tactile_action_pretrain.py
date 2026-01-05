import torch
import os
from typing import Dict, Tuple
import torch.nn.functional as F
import random
import numpy as np
from einops import rearrange

from unified_video_action.model.common.normalizer import LinearNormalizer
from unified_video_action.policy.base_image_policy import BaseImagePolicy
from unified_video_action.common.pytorch_util import dict_apply

from unified_video_action.utils.data_utils import (
    process_data,
    extract_latent_autoregressive,
    get_trajectory,
    get_vae_latent,
    resize_image_eval,
)
from unified_video_action.utils.data_utils import (
    normalize_action,
    normalize_obs,
    normalize_past_action,
    unnormalize_future_action,
)
from unified_video_action.model.autoregressive import mar_con_unified_tactile as mar
from unified_video_action.vae.vaekl import AutoencoderKL


class UnifiedTactileActionPretrain(BaseImagePolicy):
    def __init__(
        self,
        vae_model_params,
        rgb_encoder,
        autoregressive_model_params,
        action_model_params,
        shape_meta: dict,
        n_action_steps,
        shift_action=True,
        task_name=None,
        task_modes=[],
        lambda_vae=1.0,
        lambda_mse=1.0,
        **kwargs
    ):
        super().__init__()

        self.task_name = task_name
        self.task_modes = task_modes
        self.autoregressive_model_params = autoregressive_model_params
        self.n_action_steps = n_action_steps
        self.shift_action = shift_action
        self.action_dim = shape_meta.action.shape[0]
        self.lambda_vae = lambda_vae
        self.lambda_mse = lambda_mse

        self.kwargs = kwargs
        self.normalizer_type = kwargs["normalizer_type"]
        self.selected_training_mode = kwargs["selected_training_mode"]

        self.use_history_action = kwargs["use_history_action"]
        self.use_proprioception = kwargs["use_proprioception"]

        ## =========================== load vae model ===========================
        # with torch.no_grad():
        self.vae_model = AutoencoderKL(**vae_model_params)
            
        frozen_keys = ["encoder.conv_in", "encoder.down.0", "encoder.down.1"]
        self.vae_model.train()
        for name, param in self.vae_model.named_parameters():
            # for k in frozen_keys:
            #     if k in name:
            #         param.requires_grad = False
            #         print(f"Freezing {name}")
            #         break
            # else:
            param.requires_grad = True

        ## =========================== load rgb condition model =========================

        self.rgb_encoder = rgb_encoder
        if self.rgb_encoder is not None:
            self.rgb_encoder.eval()
            for param in self.rgb_encoder.parameters():
                param.requires_grad = False

        ## =========================== main model ===========================
        self.model = mar.__dict__[autoregressive_model_params.model_size](
            img_size=autoregressive_model_params.img_size,
            vae_stride=autoregressive_model_params.vae_stride,
            patch_size=autoregressive_model_params.patch_size,
            vae_embed_dim=autoregressive_model_params.vae_embed_dim,
            mask_ratio_min=autoregressive_model_params.mask_ratio_min,
            label_drop_prob=autoregressive_model_params.label_drop_prob,
            attn_dropout=autoregressive_model_params.attn_dropout,
            proj_dropout=autoregressive_model_params.proj_dropout,
            diffloss_d=autoregressive_model_params.diffloss_d,
            diffloss_w=autoregressive_model_params.diffloss_w,
            diffloss_act_d=autoregressive_model_params.diffloss_act_d,
            diffloss_act_w=autoregressive_model_params.diffloss_act_w,
            num_sampling_steps=autoregressive_model_params.num_sampling_steps,
            diffusion_batch_mul=autoregressive_model_params.diffusion_batch_mul,
            grad_checkpointing=autoregressive_model_params.grad_checkpointing,
            predict_video=autoregressive_model_params.predict_video,
            act_diff_training_steps=self.autoregressive_model_params.act_diff_training_steps,
            act_diff_testing_steps=self.autoregressive_model_params.act_diff_testing_steps,
            action_model_params=action_model_params,
            use_history_action=kwargs["use_history_action"],
            action_mask_ratio=kwargs["action_mask_ratio"],
            use_proprioception=kwargs["use_proprioception"],
            predict_wrist_img=kwargs["predict_wrist_img"],
            different_history_freq=kwargs["different_history_freq"],
            predict_proprioception=kwargs["predict_proprioception"],
            task_name=self.task_name,
            shape_meta=shape_meta,
        )

        ## =========================== load pretrained model ===========================
        self.pretrained_model_path = autoregressive_model_params.pretrained_model_path
        if self.pretrained_model_path is not None:
            if os.path.exists(self.pretrained_model_path):
                self.load_pretrained_model()
            else:
                raise ValueError('pretrained model not found: ', self.pretrained_model_path)
        
        self.normalizer = LinearNormalizer()

        assert self.selected_training_mode == "video_model", str(self.selected_training_mode)
        self.task_modes = [self.selected_training_mode]
        print("----------------------------------------------------------------------")
        print("task_modes", self.task_modes)
        print("----------------------------------------------------------------------")

    def load_pretrained_model(self):
        print("----------------------------------------------------------------------")
        print("Loading pretrained model: ", self.pretrained_model_path)
        print("----------------------------------------------------------------------")

        pretrained_diffusion_model_ckpt = torch.load(
            self.pretrained_model_path, map_location="cpu", weights_only=False
        )

        if "state_dicts" in pretrained_diffusion_model_ckpt:
            if "ema_model" in pretrained_diffusion_model_ckpt["state_dicts"]:
                print("load from previous ema model")
                ## load from previous checkpoint
                pretrained_diffusion_model_ckpt_ = {
                    k[6:]: v
                    for k, v in pretrained_diffusion_model_ckpt["state_dicts"][
                        "ema_model"
                    ].items()
                    if k.startswith("model.")
                }  # remove 'model.'

                model_state_dict = self.model.state_dict()
                pretrained_state_dict = {
                    k: v
                    for k, v in pretrained_diffusion_model_ckpt_.items()
                    if k in model_state_dict and model_state_dict[k].size() == v.size()
                }
                
                pretrained_state_dict_mismatch = {
                    k: v
                    for k, v in model_state_dict.items()
                    if k not in pretrained_diffusion_model_ckpt_
                    or pretrained_diffusion_model_ckpt_[k].size() != v.size()
                }
                
                print("----------------------------------------------------------------------")
                print(
                    "pretrained_state_dict_mismatch: ",
                    pretrained_state_dict_mismatch.keys(),
                )
                print("----------------------------------------------------------------------")
                
                assert len(model_state_dict) > 0
                assert len(pretrained_state_dict) > 0
                model_state_dict.update(pretrained_state_dict)

                missing_keys, unexpected_keys = self.model.load_state_dict(
                    model_state_dict, strict=False
                )
            else:
                raise NotImplementedError

        elif "model_ema" in pretrained_diffusion_model_ckpt:
            ## load from MAR pretrained mdoel
            pretrained_diffusion_model_ckpt_ = pretrained_diffusion_model_ckpt[
                "model_ema"
            ]

            model_state_dict = self.model.state_dict()
            pretrained_state_dict = {
                k: v
                for k, v in pretrained_diffusion_model_ckpt_.items()
                if k in model_state_dict and model_state_dict[k].size() == v.size()
            }
            assert len(model_state_dict) > 0
            assert len(pretrained_state_dict) > 0
            model_state_dict.update(pretrained_state_dict)

            missing_keys, unexpected_keys = self.model.load_state_dict(
                model_state_dict, strict=False
            )

        else:
            raise NotImplementedError

        print("---------------------------------------------------------------")
        print("Model Missing keys:", missing_keys)
        print("Model Unexpected keys:", unexpected_keys)
        print("---------------------------------------------------------------")


    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor], language_goal=None
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        raise
    

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def add_weight_decay(self, model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)  # no weight decay on bias, norm and diffloss
            else:
                decay.append(param)

        return [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay},
        ]

    def get_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:

        optim_groups = self.add_weight_decay(self.model, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        # Manually set 'initial_lr' for each parameter group (assuming a base learning rate)
        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = param_group[
                    "lr" 
                ]  # or set a specific initial learning rate

        return optimizer

    def compute_loss(self, batch, **kwargs):
        B, T, C, H, W = batch["obs"]["tactile_combined"].size()

        text_latents = self.rgb_encoder(batch["obs"]["camera0_rgb"].squeeze(1))  # 
        text_latents = text_latents[:, 0, :]  # feature aggregation for ViT

        
        nactions = normalize_action(
            normalizer=self.normalizer,
            normalizer_type=self.normalizer_type,
            actions=batch["action"],
        )
        batch = normalize_obs(
            normalizer=self.normalizer,
            normalizer_type=self.normalizer_type,
            batch=batch,
        )

        # if self.use_history_action:
        #     batch = dict_apply(batch, lambda x: x[:, 1:])

        x, proprioception_input, _ = process_data(
            batch, task_name=self.task_name, **self.kwargs
        )  # extract the main key as x (tactile frame here), the rest as proprioception_input
        x, z, c, _, proprioception_input, z_posterior = get_vae_latent(
            x, self.vae_model, eval=False, proprioception_input=proprioception_input
        )
        # the first half of x as x_c, the second half as x to be predicted
        # then, the latent of x_c is c, the latent of x is z
        real = x.detach()
        
        if self.lambda_vae is None:
            loss_vae = torch.tensor(0.0).to(z.device)
        else:
            loss_vae = z_posterior.kl().mean() * self.lambda_vae
            
        history_trajectory, trajectory = get_trajectory(
            nactions, nactions.shape[1], self.shift_action, use_history_action=self.use_history_action
        )  # originalling: (nactions, T, ...), but is buggy for umi data
        # split the nactions to history and future action

        # selected_mode = random.choice(self.task_modes)
        assert len(self.task_modes) == 1, str(self.task_modes)
        selected_mode = self.task_modes[0]


        action_latents = self.model.action_proj_cond(trajectory)
        text_latents = torch.cat(
            [text_latents.unsqueeze(1), action_latents], dim=1
        ) # N, L, D; N, D  -> N, L+1, D

        loss, video_loss, act_loss = self.model(
            z,
            c,
            history_trajectory,
            trajectory,
            text_latents,
            task_mode=selected_mode,
            proprioception_input=proprioception_input,
        )
        
        
        if self.lambda_mse is None:
            loss_mse = torch.tensor(0.0).to(loss.device)
            
        else:
            z_pred, act_out = self.model.sample_tokens(
                bsz=B,
                cond=c,
                text_latents=text_latents,
                num_iter=self.autoregressive_model_params.num_iter,
                cfg=self.autoregressive_model_params.cfg,
                cfg_schedule=self.autoregressive_model_params.cfg_schedule,
                temperature=self.autoregressive_model_params.temperature,
                history_nactions=history_trajectory,
                nactions=trajectory,
                proprioception_input=proprioception_input,
                task_mode="full_dynamic_model",
            )
            pred = self.vae_model.decode(z_pred / 0.2325)
            
            pred = rearrange(pred, "(b t) c h w -> b t h w c", b=B)
            real = rearrange(real, "b c t h w -> b t h w c")
            
            loss_mse = F.mse_loss(pred, real, reduction="mean") * self.lambda_mse

        ## not recommended, fix the problem in DDM unused parameters
        for param in self.model.parameters():
            if param.grad is None:  # Likely unused in loss computation
                loss += 0 * param.sum()
                
        # print(loss.item(), video_loss.item(), act_loss.item(), loss_vae.item())
                    
        loss = loss + loss_vae + loss_mse

        return loss, (video_loss, act_loss, loss_vae, loss_mse)

    def forward(self, batch, **kwargs):
        return self.compute_loss(batch, **kwargs)
