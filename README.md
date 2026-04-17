
<div align="center">

<h1> SteerFlow: Steering Rectified Flows for Faithful Inversion-Based Image Editing </h1>

[Thinh Dao](https://scholar.google.com/citations?user=PLACEHOLDER), [Zhen Wang](https://scholar.google.com/citations?user=PLACEHOLDER), [Kien T. Pham](https://scholar.google.com/citations?user=PLACEHOLDER), [Long Chen](https://scholar.google.com/citations?user=PLACEHOLDER)

[![arXiv](https://img.shields.io/badge/arXiv-2604.01715-b31b1b.svg)](https://arxiv.org/abs/2604.01715)

</div>

> **TL;DR**: A ***training-free, model-agnostic*** image editing framework that steers rectified flow velocity fields using amortized fixed-point inversion, trajectory interpolation, and adaptive masking for faithful inversion-based editing. Supports **FLUX.1-dev** and **Stable Diffusion 3.5 Medium**.


<h1> Abstract </h1>

Recent advances in flow-based generative models have enabled training-free, text-guided image editing by inverting an image into its latent noise and regenerating it under a new target conditional guidance. However, existing methods struggle to preserve source fidelity: higher-order solvers incur additional model inferences, truncated inversion constrains editability, and feature injection methods lack architectural transferability. To address these limitations, we propose **SteerFlow**, a model-agnostic editing framework with strong theoretical guarantees on source fidelity. In the forward process, we introduce an **Amortized Fixed-Point Solver** that implicitly straightens the forward trajectory by enforcing velocity consistency across consecutive timesteps, yielding a high-fidelity inverted latent. In the backward process, we introduce **Trajectory Interpolation**, which adaptively blends target-editing and source-reconstruction velocities to keep the editing trajectory anchored to the source. To further improve background preservation, we introduce an **Adaptive Masking** mechanism that spatially constrains the editing signal with concept-guided segmentation and source-target velocity differences. Extensive experiments on FLUX.1-dev and Stable Diffusion 3.5 Medium demonstrate that SteerFlow consistently achieves better editing quality than existing methods. Finally, we show that SteerFlow extends naturally to a complex multi-turn editing paradigm without accumulating drift.


<h1> Overview </h1>

SteerFlow performs text-driven image editing by controlling the velocity field during the ODE denoising process of flow-matching models. The key idea is to:

1. **Invert** the source image to noise via an Amortized Fixed-Point Solver that enforces velocity consistency across timesteps.
2. Steer the **Denoising Trajectory** with the target prompt, blending source and target velocities via _Trajectory Interpolation_ and _Adaptive Masking_ to ensure source consistency and background preservation.

<h1> Implementation </h1>

We provide two implementation options:

- [**Diffusers-based implementation**](./SteerFlow_diffusers/README.md): Uses the HuggingFace `diffusers` library. Supports FLUX (`black-forest-labs/FLUX.1-dev`) and Stable Diffusion 3 (`stabilityai/stable-diffusion-3-medium-diffusers`).
- [**Official FLUX implementation**](./SteerFlow_FLUX/README.md): Built on the original FLUX repository. Slightly better performance than the diffusers-based FLUX pipeline.


<h1> Acknowledgements </h1>

We thank [UniEdit-Flow](https://github.com/DSL-Lab/UniEdit-Flow), [FireFlow](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing), [FLUX](https://github.com/black-forest-labs/flux), and [SAM3](https://github.com/facebookresearch/sam3) for their excellent work.


<h1> Citation </h1>

```bib
@misc{dao2025steerflow,
    title={SteerFlow: Steering Rectified Flows for Faithful Inversion-Based Image Editing},
    author={Thinh Dao and Zhen Wang and Kien T. Pham and Long Chen},
    year={2025},
    eprint={2604.01715},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
