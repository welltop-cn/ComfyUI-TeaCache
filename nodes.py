import torch
import numpy as np

from comfy.ldm.flux.model import Flux
from comfy.ldm.hunyuan_video.model import HunyuanVideo
from comfy.ldm.flux.layers import timestep_embedding

from torch import Tensor

def teacache_flux_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        blocks_replace = patches_replace.get("dit", {})

        # enable teacache
        inp = img.clone()
        vec_ = vec.clone()
        img_mod1, _ = self.double_blocks[0].img_mod(vec_)
        modulated_inp = self.double_blocks[0].img_norm1(inp)
        modulated_inp = (1 + img_mod1.scale) * modulated_inp + img_mod1.shift

        if self.cnt == 0 or self.cnt == self.steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else: 
            coefficients = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
                
        self.previous_modulated_input = modulated_inp 
        self.cnt += 1

        if self.cnt == self.steps:
            self.cnt = 0

        if not should_calc:
            img += self.previous_residual
        else:
            ori_img = img.clone()
            for i, block in enumerate(self.double_blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"],
                                                    txt=args["txt"],
                                                    vec=args["vec"],
                                                    pe=args["pe"],
                                                    attn_mask=args.get("attn_mask"))
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img,
                                                            "txt": txt,
                                                            "vec": vec,
                                                            "pe": pe,
                                                            "attn_mask": attn_mask},
                                                            {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img,
                                    txt=txt,
                                    vec=vec,
                                    pe=pe,
                                    attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

            img = torch.cat((txt, img), 1)

            for i, block in enumerate(self.single_blocks):
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"],
                                        vec=args["vec"],
                                        pe=args["pe"],
                                        attn_mask=args.get("attn_mask"))
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img,
                                                            "vec": vec,
                                                            "pe": pe,
                                                            "attn_mask": attn_mask}, 
                                                            {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1] :, ...] += add

            img = img[:, txt.shape[1] :, ...]
            self.previous_residual = img - ori_img

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

def teacache_hunyuanvideo_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control=None,
        transformer_options={},
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})

        initial_shape = list(img.shape)
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if txt_mask is not None and not torch.is_floating_point(txt_mask):
            txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

        txt = self.txt_in(txt, timesteps, txt_mask)

        ids = torch.cat((img_ids, txt_ids), dim=1)
        pe = self.pe_embedder(ids)

        img_len = img.shape[1]
        if txt_mask is not None:
            attn_mask_len = img_len + txt.shape[1]
            attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
            attn_mask[:, 0, img_len:] = txt_mask
        else:
            attn_mask = None

        blocks_replace = patches_replace.get("dit", {})

        # enable teacache
        inp = img.clone()
        vec_ = vec.clone()
        img_mod1, _ = self.double_blocks[0].img_mod(vec_)
        modulated_inp = self.double_blocks[0].img_norm1(inp)
        modulated_inp = (1 + img_mod1.scale) * modulated_inp + img_mod1.shift

        if self.cnt == 0 or self.cnt == self.steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else: 
            coefficients = [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
                
        self.previous_modulated_input = modulated_inp 
        self.cnt += 1

        if self.cnt == self.steps:
            self.cnt = 0

        if not should_calc:
            img += self.previous_residual
        else:
            ori_img = img.clone()
            for i, block in enumerate(self.double_blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"])
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "attention_mask": attn_mask}, {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

            img = torch.cat((img, txt), 1)

            for i, block in enumerate(self.single_blocks):
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"])
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask}, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, : img_len] += add

            img = img[:, : img_len]
            self.previous_residual = img - ori_img

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        shape = initial_shape[-3:]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(initial_shape)
        return img

class TeaCacheForImgGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The image diffusion model the TeaCache will be applied to."}),
                "enable_teacache": ("BOOLEAN", {"default": True, "tooltip": "Enable teacache will speed up inference but may lose visual quality."}),
                "model_type": (["flux"],),
                "rel_l1_thresh": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_teacache"
    CATEGORY = "TeaCache"
    TITLE = "TeaCache For Img Gen"
    
    def apply_teacache(self, model, enable_teacache: bool, model_type: str, rel_l1_thresh: float, steps: int):
        if enable_teacache:
            if model_type == "flux":
                model.model.diffusion_model.__class__.cnt = 0
                model.model.diffusion_model.__class__.rel_l1_thresh = rel_l1_thresh
                model.model.diffusion_model.__class__.steps = steps
                model.model.diffusion_model.forward_orig = teacache_flux_forward.__get__(
                                                        model.model.diffusion_model,
                                                        model.model.diffusion_model.__class__
                                                        )
            else:
                raise ValueError(f"Unknown type {model_type}")
        else:
            if model_type == "flux":
                model.model.diffusion_model.forward_orig = Flux.forward_orig.__get__(
                                                        model.model.diffusion_model,
                                                        model.model.diffusion_model.__class__
                                                        )
            else:
                raise ValueError(f"Unknown type {model_type}")

        return (model,)
    
class TeaCacheForVidGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The video diffusion model the TeaCache will be applied to."}),
                "enable_teacache": ("BOOLEAN", {"default": True, "tooltip": "Enable teacache will speed up inference but may lose visual quality."}),
                "model_type": (["hunyuan_video"],),
                "rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_teacache"
    CATEGORY = "TeaCache"
    TITLE = "TeaCache For Vid Gen"
    
    def apply_teacache(self, model, enable_teacache: bool, model_type: str, rel_l1_thresh: float, steps: int):
        if enable_teacache:
            if model_type == "hunyuan_video":
                model.model.diffusion_model.__class__.cnt = 0
                model.model.diffusion_model.__class__.rel_l1_thresh = rel_l1_thresh
                model.model.diffusion_model.__class__.steps = steps
                model.model.diffusion_model.forward_orig = teacache_hunyuanvideo_forward.__get__(
                                                        model.model.diffusion_model,
                                                        model.model.diffusion_model.__class__
                                                        )
            else:
                raise ValueError(f"Unknown type {model_type}")
        else:
            if model_type == "hunyuan_video":
                model.model.diffusion_model.forward_orig = HunyuanVideo.forward_orig.__get__(
                                                        model.model.diffusion_model,
                                                        model.model.diffusion_model.__class__
                                                        )
            else:
                raise ValueError(f"Unknown type {model_type}")

        return (model,)

NODE_CLASS_MAPPINGS = {
    "TeaCacheForImgGen": TeaCacheForImgGen,
    "TeaCacheForVidGen": TeaCacheForVidGen,
}
