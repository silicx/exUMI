import sys

sys.path.extend([".", "src"])
import torch
import os
from einops import rearrange
import torch.nn.functional as F
import wandb

from unified_video_action.fvd.fvd import get_fvd_logits, frechet_distance
from unified_video_action.fvd.download import load_i3d_pretrained
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.utils.utils import AverageMeter
from unified_video_action.utils.data_utils import resize_image
from unified_video_action.utils.data_utils import (
    normalize_action,
    normalize_obs,
    unnormalize_future_action,
)
from unified_video_action.utils.data_utils import (
    process_data,
    save_image_grid,
    get_vae_latent,
    get_trajectory,
    decode_from_sample_autoregressive,
)
from unified_video_action.utils.language_model import extract_text_features




def prepare_data_predict_action(
    cfg, x, actions, model, T, device, language_goal=None, eval=False
):
    ## normalize actions and observations
    nactions = normalize_action(
        normalizer=model.normalizer,
        normalizer_type=model.normalizer_type,
        actions=actions,
    )
    x = normalize_obs(
        normalizer=model.normalizer, normalizer_type=model.normalizer_type, batch=x
    )

    ## process data
    x, proprioception_input, _ = process_data(
        x,
        task_name=cfg.task.name,
        eval=eval,
        use_proprioception=cfg.model.policy.use_proprioception,
        different_history_freq=cfg.model.policy.different_history_freq,
    )

    real, _, c, latent_size, proprioception_input, _ = get_vae_latent(
        x, model.vae_model, eval=True, proprioception_input=proprioception_input
    )
    history_trajectory, trajectory = get_trajectory(
        nactions,
        nactions.shape[1],   # original: T
        cfg.model.policy.shift_action,
        use_history_action=cfg.model.policy.use_history_action,
    )

    text_latents = None
    # if cfg.task.dataset.language_emb_model is not None:
    #     if "umi" in cfg.task.name:
    #         text_latents = language_goal
    #     elif "libero" in cfg.task.name:
    #         if cfg.task.dataset.language_emb_model == "clip":
    #             text_tokens = {
    #                 "input_ids": language_goal[:, 0].long()[:, 0],
    #                 "attention_mask": language_goal[:, 0].long()[:, 1],
    #             }
    #             text_latents = extract_text_features(
    #                 model.text_model,
    #                 text_tokens,
    #                 language_emb_model=cfg.task.dataset.language_emb_model,
    #             )
    #         elif cfg.task.dataset.language_emb_model == "flant5":
    #             text_tokens = language_goal[:, 0].long()
    #             text_latents = extract_text_features(
    #                 model.text_model,
    #                 text_tokens,
    #                 language_emb_model=cfg.task.dataset.language_emb_model,
    #             ).float()
    #         else:
    #             raise NotImplementedError
    return (
        x,
        real,
        latent_size,
        c,
        text_latents,
        history_trajectory,
        trajectory,
        proprioception_input,
    )


def test_video_fvd(
    cfg, model, loader, it, output_dir, device, name_label="", plot_actions=False
):
    raise
    losses = dict()
    losses["fvd"] = AverageMeter()

    i3d = load_i3d_pretrained(device)
    real_embeddings = []
    pred_embeddings = []

    reals = []
    predictions = []

    n_examples = 4

    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print("test_video_fvd", n, len(loader))

            x = batch
            if n >= n_examples:
                break

            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]

            if cfg.model.policy.use_history_action:
                x = dict_apply(x, lambda x: x[:, 1:])

            x = resize_image(cfg, x)

            B, T, C, H, W = x["obs"]["image"].size()
            k = min(n_examples, B)

            actions = actions[:k]
            x = dict_apply(x, lambda x: x[:k])

            if cfg.task.dataset.language_emb_model is not None:
                if "language" in x["obs"]:
                    language_goal = x["obs"]["language"]
                    del x["obs"]["language"]
                elif "language_latents" in x:
                    language_goal = x["language_latents"]
                    del x["language_latents"]
                else:
                    raise NotImplementedError
            else:
                language_goal = None

            (
                x,
                real,
                _,
                c,
                text_lat_ents,
                history_trajectory,
                trajectory,
                proprioception_input,
            ) = prepare_data_predict_action(
                cfg, x, actions, model, T, device, language_goal=language_goal
            )

            z, act_out = model.model.sample_tokens(
                bsz=k,
                cond=c,
                text_latents=text_latents,
                num_iter=cfg.model.policy.autoregressive_model_params.num_iter,
                cfg=cfg.model.policy.autoregressive_model_params.cfg,
                cfg_schedule=cfg.model.policy.autoregressive_model_params.cfg_schedule,
                temperature=cfg.model.policy.autoregressive_model_params.temperature,
                history_nactions=history_trajectory,
                nactions=trajectory,
                proprioception_input=proprioception_input,
                task_mode="full_dynamic_model",
            )
            pred = decode_from_sample_autoregressive(model.vae_model, z / 0.2325)
            pred = pred.clamp(-1, 1).cpu()

            pred = 1 + rearrange(pred, "(b t) c h w -> b t h w c", b=k)
            real = (1 + rearrange(real, "b c t h w -> b t h w c")).cpu()

            pred = pred * 127.5
            pred = pred.type(torch.uint8)

            real = real * 127.5
            real = real.type(torch.uint8)

            x = (1 + x) * 127.5  # b c t h w
            x = x.type(torch.uint8).cpu()

            if len(predictions) < n_examples:
                reals.append(
                    torch.cat(
                        [
                            x[:, :, : x.size(2) // 2],
                            rearrange(real, "b t h w c -> b c t h w"),
                        ],
                        dim=2,
                    )
                )
                predictions.append(
                    torch.cat(
                        [
                            x[:, :, : x.size(2) // 2],
                            rearrange(pred, "b t h w c -> b c t h w"),
                        ],
                        dim=2,
                    )
                )

            if real.shape[1] < 16:
                pred = pred.repeat_interleave(repeats=4, dim=1)
                real = real.repeat_interleave(repeats=4, dim=1)

            pred_embeddings.append(get_fvd_logits(pred.numpy(), i3d=i3d, device=device))
            real_embeddings.append(get_fvd_logits(real.numpy(), i3d=i3d, device=device))

    log_data = dict()
    reals = torch.cat(reals)
    predictions = torch.cat(predictions)

    real_embeddings = torch.cat(real_embeddings)
    pred_embeddings = torch.cat(pred_embeddings)
    fvd = frechet_distance(
        pred_embeddings.clone().detach(), real_embeddings.clone().detach()
    )
    fvd = fvd.item()

    os.makedirs(output_dir + "/vis", exist_ok=True)
    real_vid = save_image_grid(
        reals.cpu().numpy(),
        os.path.join(output_dir, f"vis/{name_label}real_{it}.gif"),
        drange=[0, 255],
        grid_size=(reals.size(0) // 4, 4),
    )  # [4, 3, 8, 128, 128]
    pred_vid = save_image_grid(
        predictions.cpu().numpy(),
        os.path.join(output_dir, f"vis/{name_label}predicted_{it}.gif"),
        drange=[0, 255],
        grid_size=(predictions.size(0) // 4, 4),
    )  # [4, 3, 8, 128, 128]

    real_video = wandb.Video(os.path.join(output_dir, f"vis/{name_label}real_{it}.gif"))
    pred_video = wandb.Video(
        os.path.join(output_dir, f"vis/{name_label}predicted_{it}.mp4")
    )

    log_data[f"{name_label}video_fvd"] = fvd
    log_data[f"{name_label}real_img"] = real_video
    log_data[f"{name_label}predicted_img"] = pred_video

    return log_data




def test_video_display(
    cfg, model, loader, it, output_dir, device, name_label="", plot_actions=False
):

    reals = []
    predictions = []
    all_hists = []
    all_reals = []
    all_preds = []
    mse_loss = []
    n_valid = []

    n_examples = 4

    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print("test_video_visualize", n, len(loader))

            x = batch
            # if n >= n_examples:
            #     break

            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]

            # if cfg.model.policy.use_history_action:
            #     x = dict_apply(x, lambda x: x[:, 1:])

            x = resize_image(cfg, x)

            B, T, C, H, W = x["obs"]["tactile_combined"].size()
            k = B #min(n_examples, B)

            actions = actions[:k]
            x = dict_apply(x, lambda x: x[:k])

            # if cfg.task.dataset.language_emb_model is not None:
            #     if "language" in x["obs"]:
            #         language_goal = x["obs"]["language"]
            #         del x["obs"]["language"]
            #     elif "language_latents" in x:
            #         language_goal = x["language_latents"]
            #         del x["language_latents"]
            #     else:
            #         raise NotImplementedError
            # else:
            language_goal = None
                
                
            text_latents = model.rgb_encoder(batch["obs"]["camera0_rgb"].squeeze(1))
            text_latents = text_latents[:, 0, :]  # feature aggregation for ViT

            (
                x,
                real,
                _,
                c,
                _,
                history_trajectory,
                trajectory,
                proprioception_input,
            ) = prepare_data_predict_action(
                cfg, x, actions, model, T, device, language_goal=language_goal
            )

            z, act_out = model.model.sample_tokens(
                bsz=k,
                cond=c,
                text_latents=text_latents,
                num_iter=cfg.model.policy.autoregressive_model_params.num_iter,
                cfg=cfg.model.policy.autoregressive_model_params.cfg,
                cfg_schedule=cfg.model.policy.autoregressive_model_params.cfg_schedule,
                temperature=cfg.model.policy.autoregressive_model_params.temperature,
                history_nactions=history_trajectory,
                nactions=trajectory,
                proprioception_input=proprioception_input,
                task_mode="full_dynamic_model",
            )
            pred = decode_from_sample_autoregressive(model.vae_model, z / 0.2325)
            pred = pred.clamp(-1, 1).cpu()

            pred = 1 + rearrange(pred, "(b t) c h w -> b t h w c", b=k)
            real = (1 + rearrange(real, "b c t h w -> b t h w c")).cpu()
            
            mse = F.mse_loss(pred, real, reduction="none")
            valid = batch["valid"].float().to(mse.device)
            mse = mse.reshape(B, -1).mean(-1)
            
            mse_loss.append((mse*valid).sum().item())
            n_valid.append(valid.sum().float())

            pred = pred * 127.5
            pred = pred.type(torch.uint8)

            real = real * 127.5
            real = real.type(torch.uint8)

            x = (1 + x) * 127.5  # b c t h w
            x = x.type(torch.uint8).cpu()

            if len(predictions) < n_examples:
                reals.append(
                    torch.cat(
                        [
                            x[:, :, : x.size(2) // 2],
                            rearrange(real, "b t h w c -> b c t h w"),
                        ],
                        dim=2,
                    )
                )
                predictions.append(
                    torch.cat(
                        [
                            x[:, :, : x.size(2) // 2],
                            rearrange(pred, "b t h w c -> b c t h w"),
                        ],
                        dim=2,
                    )
                )
            all_hists.append(x[:, :, : x.size(2) // 2].cpu())
            all_reals.append(rearrange(real, "b t h w c -> b c t h w").cpu())
            all_preds.append(rearrange(pred, "b t h w c -> b c t h w").cpu())


    log_data = dict()
    reals = torch.cat(reals)
    predictions = torch.cat(predictions)


    os.makedirs(output_dir + "/vis", exist_ok=True)
    
    n_half = reals.shape[2] // 2
    
    history_vid = save_image_grid(
        reals.cpu().numpy()[:, :, :n_half, :, :],
        os.path.join(output_dir, f"vis/{name_label}hist_{it}.gif"),
        drange=[0, 255],
        grid_size=(reals.size(0) // 4, 4),
    )  # [4, 3, 8, 128, 128]
    real_vid = save_image_grid(
        reals.cpu().numpy()[:, :, n_half:, :, :],
        os.path.join(output_dir, f"vis/{name_label}real_{it}.gif"),
        drange=[0, 255],
        grid_size=(reals.size(0) // 4, 4),
    )  # [4, 3, 8, 128, 128]
    pred_vid = save_image_grid(
        predictions.cpu().numpy()[:, :, n_half:, :, :],
        os.path.join(output_dir, f"vis/{name_label}predicted_{it}.gif"),
        drange=[0, 255],
        grid_size=(predictions.size(0) // 4, 4),
    )  # [4, 3, 8, 128, 128]
    
    torch.save({
        "hist": all_hists,
        "real": all_reals,
        "pred": all_preds,
    }, os.path.join(output_dir, f"vis/store_{it}.pt"))

    # hist_video = wandb.Video(os.path.join(output_dir, f"vis/{name_label}hist_{it}.gif"))
    # real_video = wandb.Video(os.path.join(output_dir, f"vis/{name_label}real_{it}.gif"))
    # pred_video = wandb.Video(
    #     os.path.join(output_dir, f"vis/{name_label}predicted_{it}.gif")
    # )

    # log_data[f"{name_label}hist_img"] = hist_video
    # log_data[f"{name_label}real_img"] = real_video
    # log_data[f"{name_label}predicted_img"] = pred_video
    valid_mse = (torch.sum(torch.tensor(mse_loss)) / torch.sum(torch.tensor(n_valid))).item() #torch.mean(torch.tensor(mse_loss)).item()
    log_data[f"{name_label}valid_mse_loss"] =  valid_mse
    print("Validation mse loss", valid_mse)

    return log_data


def test_action_l2(
    cfg,
    model,
    loader,
    it,
    output_dir,
    device,
    text_model=None,
    name_label="",
    plot_actions=False,
):
    action_l2_distances = []

    with torch.no_grad():
        for n, batch in enumerate(loader):
            if n % 10 == 0:
                print("test_action_l2", n, len(loader))

            x = batch
            x = dict_apply(x, lambda x: x.to(device, non_blocking=True))
            actions = x["action"]

            if cfg.model.policy.use_history_action:
                x = dict_apply(x, lambda x: x[:, 1:])

            x = resize_image(cfg, x)

            B, T, C, H, W = x["obs"]["image"].size()

            if cfg.task.dataset.language_emb_model is not None:
                if "language" in x["obs"]:
                    language_goal = x["obs"]["language"]
                    del x["obs"]["language"]
                elif "language_latents" in x:
                    language_goal = x["language_latents"]
                    del x["language_latents"]
                else:
                    raise NotImplementedError
            else:
                language_goal = None

            (
                x,
                real,
                _,
                c,
                text_latents,
                history_trajectory,
                trajectory,
                proprioception_input,
            ) = prepare_data_predict_action(
                cfg, x, actions, model, T, device, language_goal=language_goal
            )

            z, act_out = model.model.sample_tokens(
                bsz=B,
                cond=c,
                text_latents=text_latents,
                num_iter=cfg.model.policy.autoregressive_model_params.num_iter,
                cfg=cfg.model.policy.autoregressive_model_params.cfg,
                cfg_schedule=cfg.model.policy.autoregressive_model_params.cfg_schedule,
                temperature=cfg.model.policy.autoregressive_model_params.temperature,
                history_nactions=history_trajectory,
                nactions=trajectory,
                proprioception_input=proprioception_input,
                task_mode="policy_model",
            )

            if cfg.model.policy.action_model_params.predict_action:
                act_out = unnormalize_future_action(
                    normalizer=model.normalizer,
                    normalizer_type=model.normalizer_type,
                    actions=act_out,
                )
                trajectory = unnormalize_future_action(
                    normalizer=model.normalizer,
                    normalizer_type=model.normalizer_type,
                    actions=trajectory,
                )

                ## calculate l2 distance between the predicted action and ground truth action
                l2_distance = torch.sqrt(
                    torch.sum((trajectory[:, :, :9] - act_out[:, :, :9]) ** 2, dim=-1)
                )
                action_l2_distances.append(l2_distance.mean())

            if cfg.training.debug:
                break

    log_data = dict()
    if cfg.model.policy.action_model_params.predict_action:
        log_data[f"{name_label}val_action_l2_distances"] = (
            torch.stack(action_l2_distances).mean().item()
        )

    return log_data
