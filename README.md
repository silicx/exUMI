# TPP: Tactile Predictive Pretraining
This is the official code of TPP method from our exUMI: a task-agnostic tactile representation learning framework that leverages action-conditioned temporal prediction to capture contact dynamics, designed for contact-rich robotic manipulation tasks. Much of this repo is based on [UVA](https://unified-video-action-model.github.io/).

## Quick Start

Dependencies:

```bash
conda env create -f conda_environment.yml
```

Basic Usage:
```bash
bash run_train_tpp.sh
```

Configure your data:

1. Unzip your `zarr.zip` data pack to `/dev/shm/XXX.zarr` (by default), then setup the `dataset_configs` in `unified_video_action/config/task/umi_multi_tactile.yaml`



## 🤝 Integration with Downstream Tasks
The pretrained tactile encoder can be seamlessly integrated into imitation learning pipelines:
1. Freeze the encoder weights
2. Concatenate tactile embeddings with visual/state features
3. Train policy (e.g., diffusion policy, transformer-based imitation learning)

Please see the [main](https://github.com/silicx/exUMI/tree/main) branch of this repo for an example.

## Citation and Acknowledgement

If you use TPP in your research, please cite our paper:
```bibtex
@inproceedings{xu2025exumi,
  title={exUMI: Extensible Robot Teaching System with Action-aware Task-agnostic Tactile Representation},
  author={Xu, Yue and Wei, Litao and An, Pengyu and Zhang, Qingyu and Li, Yong-Lu},
  booktitle={Conference on Robot Learning},
  pages={2536--2554},
  year={2025},
  organization={PMLR}
}
```

And also [UVA](https://unified-video-action-model.github.io/). Thanks for their great efforts!
```
 @article{li2025unified,
    title={Unified Video Action Model},
    author={Li, Shuang and Gao, Yihuai and Sadigh, Dorsa and Song, Shuran},
    journal={arXiv preprint arXiv:2503.00200},
    year={2025}
    }
```
