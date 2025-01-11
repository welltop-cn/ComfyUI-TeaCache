# ComfyUI-TeaCache

## Introduction
Timestep Embedding Aware Cache ([TeaCache](https://github.com/ali-vilab/TeaCache)) is a training-free caching approach that estimates and leverages the fluctuating differences among model outputs across timesteps, thereby accelerating the inference. TeaCache works well for Image Diffusion models, Video Diffusion Models, and Audio Diffusion Models.

TeaCache has now been integrated into ComfyUI and is compatible with the ComfyUI native nodes. ComfyUI-TeaCache is easy to use, simply connect the TeaCache node with the ComfyUI native nodes for seamless usage.

## Updates
- Jan 10 2025: ComfyUI-TeaCache supports LTX-Video:
    - It can achieve a 1.4x lossless speedup and a 1.7x speedup without much visual quality degradation.
    - Support Text to Video and Image to Video!
- Jan 9 2025: ComfyUI-TeaCache supports HunyuanVideo:
    - It can achieve a 1.6x lossless speedup and a 2x speedup without much visual quality degradation.
- Jan 8 2025: ComfyUI-TeaCache supports FLUX:
    - It can achieve a 1.4x lossless speedup and a 2x speedup without much visual quality degradation.
    - Support FLUX LoRA!
    - Support FLUX ControlNet!

## Installation
Installation via ComfyUI-Manager is preferred. Simply search for ComfyUI-TeaCache in the list of nodes and click install.
### Manual installation
1. Go to comfyUI custom_nodes folder, `ComfyUI/custom_nodes/`
2. git clone https://github.com/welltop-cn/ComfyUI-TeaCache.git

## Recommended settings
The following table gives the recommended rel_l1_thresh ​for different models:

|                       |            FLUX              |      HunyuanVideo     |       LTX-Video       |
|:---------------------:|:----------------------------:|:---------------------:|:---------------------:|
|    rel_l1_thresh      |             0.4              |         0.15          |        0.06           |
|       speedup         |             ~2x              |         ~2x           |        ~1.7x          |

## Usage
The demo workflows are placed in examples folder.

## Demo
- <p><strong>FLUX</strong></p>
https://github.com/user-attachments/assets/e977cf34-f7d0-4b25-a2e3-10fd62ebfe30

- <p><strong>HunyuanVideo</strong></p>
https://github.com/user-attachments/assets/4d8e9f12-2c54-40c5-a992-c2cecbde019a

- <p><strong>LTX-Video</strong></p>
https://github.com/user-attachments/assets/19e63dd8-ecdf-418c-8ec2-b9b9dcf9a655

## Result comparison
- <p><strong>FLUX</strong></p>
![](./assets/compare_flux.png)

- <p><strong>HunyuanVideo</strong></p>
https://github.com/user-attachments/assets/b3aca64d-c2ae-440c-a362-f3a7b6c633e0

- <p><strong>LTX-Video</strong></p>
https://github.com/user-attachments/assets/8fce9b48-2243-46f1-b411-80e4a53f6f7d

## Acknowledgments
Thanks to TeaCache repo owner [ali-vilab/TeaCache: Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](https://github.com/ali-vilab/TeaCache)
