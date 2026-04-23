---
license: apache-2.0
library_name: seedvr
pipeline_tag: video-to-video
---

<div align="center">
  <img src="assets/seedvr_logo.png" alt="SeedVR" width="400"/>
</div>


# SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training
> [Jianyi Wang](https://iceclear.github.io), [Shanchuan Lin](https://scholar.google.com/citations?user=EDWUw7gAAAAJ&hl=en), [Zhijie Lin](https://scholar.google.com/citations?user=xXMj6_EAAAAJ&hl=en), [Yuxi Ren](https://scholar.google.com.hk/citations?user=C_6JH-IAAAAJ&hl=en), [Meng Wei](https://openreview.net/profile?id=~Meng_Wei11), [Zongsheng Yue](https://zsyoaoa.github.io/), [Shangchen Zhou](https://shangchenzhou.com/), [Hao Chen](https://haochen-rye.github.io/), [Yang Zhao](https://scholar.google.com/citations?user=uPmTOHAAAAAJ&hl=en), [Ceyuan Yang](https://ceyuan.me/), [Xuefeng Xiao](https://scholar.google.com/citations?user=CVkM9TQAAAAJ&hl=en), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/index.html), [Lu Jiang](http://www.lujiang.info/)

<p align="center">
  <a href="https://iceclear.github.io/projects/seedvr2/">
    <img
      src="https://img.shields.io/badge/SeedVR2-Website-0A66C2?logo=safari&logoColor=white"
      alt="SeedVR Website"
    />
  </a>
  <a href="http://arxiv.org/abs/2506.05301">
    <img
      src="https://img.shields.io/badge/SeedVR2-Paper-red?logo=arxiv&logoColor=red"
      alt="SeedVR2 Paper on ArXiv"
    />
  </a>
  <a href="https://github.com/ByteDance-Seed/SeedVR">
            <img 
              alt="Github" src="https://img.shields.io/badge/SeedVR2-Codebase-536af5?color=536af5&logo=github"
              alt="SeedVR2 Codebase"
            />
  </a>
  <a href="https://huggingface.co/collections/ByteDance-Seed/seedvr-6849deeb461c4e425f3e6f9e">
    <img 
        src="https://img.shields.io/badge/SeedVR-Models-yellow?logo=huggingface&logoColor=yellow" 
        alt="SeedVR Models"
    />
  </a>
   <a href="https://huggingface.co/spaces/ByteDance-Seed/SeedVR2-3B">
    <img 
        src="https://img.shields.io/badge/SeedVR2-Space-orange?logo=huggingface&logoColor=yellow" 
        alt="SeedVR2 Space"
    />
  </a>
  <a href="https://www.youtube.com/watch?v=tM8J-WhuAH0" target='_blank'>
    <img 
        src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white"
        alt="SeedVR2 Video Demo on YouTube"
    />
  </a>
</p>

>
> Recent advances in diffusion-based video restoration (VR) demonstrate significant improvement in visual quality, yet yield a prohibitive computational cost during inference. While several distillation-based approaches have exhibited the potential of one-step image restoration, extending existing approaches to VR remains challenging and underexplored, due to the limited generation ability and poor temporal consistency, particularly when dealing with high-resolution video in real-world settings. In this work, we propose a one-step diffusion-based VR model, termed as SeedVR2, which performs adversarial VR training against real data. To handle the challenging high-resolution VR within a single step, we introduce several enhancements to both model architecture and training procedures. Specifically, an adaptive window attention mechanism is proposed, where the window size is dynamically adjusted to fit the output resolutions, avoiding window inconsistency observed under high-resolution VR using window attention with a predefined window size. To stabilize and improve the adversarial post-training towards VR, we further verify the effectiveness of a series of losses, including a proposed feature matching loss without significantly sacrificing training efficiency. Extensive experiments show that SeedVR2 can achieve comparable or even better performance compared with existing VR approaches in a single step.

<p align="center"><img src="assets/teaser.png" width="100%"></p>


## 📮 Notice
**Limitations:** These are the prototype models and the performance may not be perfectly align with the paper. Our methods are sometimes not robust to heavy degradations and very large motions, and shares some failure cases with existing methods, e.g., fail to fully remove the degradation or simply generate unpleasing details. Moreover, due to the strong generation ability, Our methods tend to overly generate details on inputs with very light degradations, e.g., 720p AIGC videos, leading to oversharpened results occasionally.


## ✍️ Citation

```bibtex
@article{wang2025seedvr2,
      title={SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training},
      author={Wang, Jianyi and Lin, Shanchuan and Lin, Zhijie and Ren, Yuxi and Wei, Meng and Yue, Zongsheng and Zhou, Shangchen and Chen, Hao and Zhao, Yang and Yang, Ceyuan and Xiao, Xuefeng and Loy, Chen Change and Jiang, Lu},
      booktitle={arXiv preprint arXiv:2506.05301},
      year={2025}
   }

@inproceedings{wang2025seedvr,
      title={SeedVR: Seeding Infinity in Diffusion Transformer Towards Generic Video Restoration},
      author={Wang, Jianyi and Lin, Zhijie and Wei, Meng and Zhao, Yang and Yang, Ceyuan and Loy, Chen Change and Jiang, Lu},
      booktitle={CVPR},
      year={2025}
   }
```


## 📜 License
SeedVR and SeedVR2 are licensed under the Apache 2.0.