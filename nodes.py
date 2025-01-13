import math
import torch
import numpy as np

from torch import Tensor

from comfy.ldm.flux.model import Flux
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.hunyuan_video.model import HunyuanVideo
from comfy.ldm.lightricks.model import LTXVModel, precompute_freqs_cis
from comfy.ldm.common_dit import rms_norm


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

        if not hasattr(self, 'accumulated_rel_l1_distance'):
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            try:
                coefficients = [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            except:
                should_calc = True
                self.accumulated_rel_l1_distance = 0

        self.previous_modulated_input = modulated_inp

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

        if not hasattr(self, 'accumulated_rel_l1_distance'):
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            try:
                coefficients = [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            except:
                should_calc = True
                self.accumulated_rel_l1_distance = 0

        self.previous_modulated_input = modulated_inp 

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

def teacache_ltxvmodel_forward(
        self,
        x,
        timestep,
        context,
        attention_mask,
        frame_rate=25,
        guiding_latent=None,
        guiding_latent_noise_scale=0,
        transformer_options={},
        **kwargs
    ):
        patches_replace = transformer_options.get("patches_replace", {})

        indices_grid = self.patchifier.get_grid(
            orig_num_frames=x.shape[2],
            orig_height=x.shape[3],
            orig_width=x.shape[4],
            batch_size=x.shape[0],
            scale_grid=((1 / frame_rate) * 8, 32, 32),
            device=x.device,
        )

        if guiding_latent is not None:
            ts = torch.ones([x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]], device=x.device, dtype=x.dtype)
            input_ts = timestep.view([timestep.shape[0]] + [1] * (x.ndim - 1))
            ts *= input_ts
            ts[:, :, 0] = guiding_latent_noise_scale * (input_ts[:, :, 0] ** 2)
            timestep = self.patchifier.patchify(ts)
            input_x = x.clone()
            x[:, :, 0] = guiding_latent[:, :, 0]
            if guiding_latent_noise_scale > 0:
                if self.generator is None:
                    self.generator = torch.Generator(device=x.device).manual_seed(42)
                elif self.generator.device != x.device:
                    self.generator = torch.Generator(device=x.device).set_state(self.generator.get_state())

                noise_shape = [guiding_latent.shape[0], guiding_latent.shape[1], 1, guiding_latent.shape[3], guiding_latent.shape[4]]
                scale = guiding_latent_noise_scale * (input_ts ** 2)
                guiding_noise = scale * torch.randn(size=noise_shape, device=x.device, generator=self.generator)

                x[:, :, 0] = guiding_noise[:, :, 0] + x[:, :, 0] *  (1.0 - scale[:, :, 0])


        orig_shape = list(x.shape)

        x = self.patchifier.patchify(x)

        x = self.patchify_proj(x)
        timestep = timestep * 1000.0

        attention_mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1]))
        attention_mask = attention_mask.masked_fill(attention_mask.to(torch.bool), float("-inf"))  # not sure about this
        # attention_mask = (context != 0).any(dim=2).to(dtype=x.dtype)

        pe = precompute_freqs_cis(indices_grid, dim=self.inner_dim, out_dtype=x.dtype)

        batch_size = x.shape[0]
        timestep, embedded_timestep = self.adaln_single(
            timestep.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=x.dtype,
        )
        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.shape[-1]
        )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = x.shape[0]
            context = self.caption_projection(context)
            context = context.view(
                batch_size, -1, x.shape[-1]
            )

        blocks_replace = patches_replace.get("dit", {})

        # enable teacache
        inp = x.clone()
        timestep_ = timestep.clone()
        num_ada_params = self.transformer_blocks[0].scale_shift_table.shape[0]
        ada_values = self.transformer_blocks[0].scale_shift_table[None, None] + timestep_.reshape(batch_size, timestep_.size(1), num_ada_params, -1)
        shift_msa, scale_msa, _, _, _, _ = ada_values.unbind(dim=2)
        modulated_inp = rms_norm(inp)
        modulated_inp = modulated_inp * (1 + scale_msa) + shift_msa
        
        if not hasattr(self, 'accumulated_rel_l1_distance'):
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            try:
                coefficients = [2.14700694e+01, -1.28016453e+01, 2.31279151e+00, 7.92487521e-01, 9.69274326e-03]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            except:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
                
        self.previous_modulated_input = modulated_inp

        
        if not should_calc:
            x += self.previous_residual
        else:
            ori_x = x.clone()
            for i, block in enumerate(self.transformer_blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], context=args["txt"], attention_mask=args["attention_mask"], timestep=args["vec"], pe=args["pe"])
                        return out

                    out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "attention_mask": attention_mask, "vec": timestep, "pe": pe}, {"original_block": block_wrap})
                    x = out["img"]
                else:
                    x = block(
                        x,
                        context=context,
                        attention_mask=attention_mask,
                        timestep=timestep,
                        pe=pe
                    )

            # 3. Output
            scale_shift_values = (
                self.scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
            )
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
            x = self.norm_out(x)
            # Modulation
            x = x * (1 + scale) + shift
            self.previous_residual = x - ori_x

        x = self.proj_out(x)

        x = self.patchifier.unpatchify(
            latents=x,
            output_height=orig_shape[3],
            output_width=orig_shape[4],
            output_num_frames=orig_shape[2],
            out_channels=orig_shape[1] // math.prod(self.patchifier.patch_size),
        )

        if guiding_latent is not None:
            x[:, :, 0] = (input_x[:, :, 0] - guiding_latent[:, :, 0]) / input_ts[:, :, 0]

        # print("res", x)
        return x

class TeaCacheForImgGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The image diffusion model the TeaCache will be applied to."}),
                "enable_teacache": ("BOOLEAN", {"default": True, "tooltip": "Enable teacache will speed up inference but may lose visual quality."}),
                "model_type": (["flux"],),
                "rel_l1_thresh": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."})
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_teacache"
    CATEGORY = "TeaCache"
    TITLE = "TeaCache For Img Gen"
    
    def apply_teacache(self, model, enable_teacache: bool, model_type: str, rel_l1_thresh: float):
        if enable_teacache:
            model.model.diffusion_model.__class__.rel_l1_thresh = rel_l1_thresh
            if model_type == "flux":
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
                "model_type": (["hunyuan_video", "ltxv"],),
                "rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."})
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_teacache"
    CATEGORY = "TeaCache"
    TITLE = "TeaCache For Vid Gen"
    
    def apply_teacache(self, model, enable_teacache: bool, model_type: str, rel_l1_thresh: float):
        if enable_teacache:
            model.model.diffusion_model.__class__.rel_l1_thresh = rel_l1_thresh
            if model_type == "hunyuan_video":
                model.model.diffusion_model.forward_orig = teacache_hunyuanvideo_forward.__get__(
                                                        model.model.diffusion_model,
                                                        model.model.diffusion_model.__class__
                                                        )
            elif model_type == "ltxv":
                model.model.diffusion_model.forward = teacache_ltxvmodel_forward.__get__(
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
            elif model_type == "ltxv":
                model.model.diffusion_model.forward = LTXVModel.forward.__get__(
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
