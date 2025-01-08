# ComfyUI-TeaCache

## Introduction
Timestep Embedding Aware Cache (TeaCache) is a training-free caching approach that estimates and leverages the fluctuating differences among model outputs across timesteps, thereby accelerating the inference. TeaCache works well for Image Diffusion models, Video Diffusion Models, and Audio Diffusion Models.

## Updates
- TeaCache has now been integrated into ComfyUI and is compatible with the ComfyUI native nodes.
- ComfyUI-TeaCache is easy to use, simply connect the TeaCache node with the ComfyUI native nodes for seamless usage.
- At present, ComfyUI-TeaCache supports FLUX:
    - It can achieve a 1.4x lossless speedup and a 2x speedup without much visual quality degradation, which are consistent with the original [TeaCache](https://github.com/LiewFeng/TeaCache).
    - Support FLUX LoRA!
    - Support FLUX ControlNet!

## Installation
1. Go to comfyUI custom_nodes folder, `ComfyUI/custom_nodes/`
2. git clone https://github.com/welltop-cn/ComfyUI-TeaCache.git


## Demo
https://github.com/user-attachments/assets/e977cf34-f7d0-4b25-a2e3-10fd62ebfe30

## Result comparison
![](./assets/compare.png)

## Acknowledgments
Thanks to TeaCache repo owner [ali-vilab/TeaCache: Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](https://github.com/ali-vilab/TeaCache)
