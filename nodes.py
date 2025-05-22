import math
import torch
import comfy.ldm.common_dit
import comfy.model_management as mm

from torch import Tensor
from einops import repeat
from typing import Optional
from unittest.mock import patch

from comfy.ldm.flux.layers import timestep_embedding, apply_mod
from comfy.ldm.lightricks.model import precompute_freqs_cis
from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
from comfy.ldm.wan.model import sinusoidal_embedding_1d


SUPPORTED_MODELS_COEFFICIENTS = {
    "flux": [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
    "ltxv": [2.14700694e+01, -1.28016453e+01, 2.31279151e+00, 7.92487521e-01, 9.69274326e-03],
    "hunyuan_video": [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02],
    "hidream_i1_full": [-3.13605009e+04, -7.12425503e+02, 4.91363285e+01, 8.26515490e+00, 1.08053901e-01],
    "wan2.1_t2v_1.3B": [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
    "wan2.1_t2v_14B": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
    "wan2.1_i2v_480p_14B": [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01],
    "wan2.1_i2v_720p_14B": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
    "wan2.1_t2v_1.3B_ret_mode": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
    "wan2.1_t2v_14B_ret_mode": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
    "wan2.1_i2v_480p_14B_ret_mode": [2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
    "wan2.1_i2v_720p_14B_ret_mode": [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02],
}

def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result

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
        rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        coefficients = transformer_options.get("coefficients")
        enable_teacache = transformer_options.get("enable_teacache", True)
        
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

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)
        else:
            pe = None

        blocks_replace = patches_replace.get("dit", {})

        # enable teacache
        img_mod1, _ = self.double_blocks[0].img_mod(vec)
        modulated_inp = self.double_blocks[0].img_norm1(img)
        modulated_inp = apply_mod(modulated_inp, (1 + img_mod1.scale), img_mod1.shift)
        ca_idx = 0

        if not hasattr(self, 'accumulated_rel_l1_distance'):
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            try:
                self.accumulated_rel_l1_distance += poly1d(coefficients, ((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()))
                if self.accumulated_rel_l1_distance < rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            except:
                should_calc = True
                self.accumulated_rel_l1_distance = 0

        self.previous_modulated_input = modulated_inp

        if not enable_teacache:
            should_calc = True

        if not should_calc:
            img += self.previous_residual.to(img.device)
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

                # PuLID attention
                if getattr(self, "pulid_data", {}):
                    if i % self.pulid_double_interval == 0:
                        # Will calculate influence of all pulid nodes at once
                        for _, node_data in self.pulid_data.items():
                            if torch.any((node_data['sigma_start'] >= timesteps)
                                        & (timesteps >= node_data['sigma_end'])):
                                img = img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], img)
                        ca_idx += 1

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

                # PuLID attention
                if getattr(self, "pulid_data", {}):
                    real_img, txt = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
                    if i % self.pulid_single_interval == 0:
                        # Will calculate influence of all nodes at once
                        for _, node_data in self.pulid_data.items():
                            if torch.any((node_data['sigma_start'] >= timesteps)
                                        & (timesteps >= node_data['sigma_end'])):
                                real_img = real_img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], real_img)
                        ca_idx += 1
                    img = torch.cat((txt, real_img), 1)

            img = img[:, txt.shape[1] :, ...]
            self.previous_residual = (img - ori_img).to(mm.unet_offload_device())

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        
        return img

def teacache_hidream_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        encoder_hidden_states_llama3=None,
        image_cond=None,
        control = None,
        transformer_options = {},
    ) -> torch.Tensor:
        rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        coefficients = transformer_options.get("coefficients")
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        enable_teacache = transformer_options.get("enable_teacache", True)

        bs, c, h, w = x.shape
        if image_cond is not None:
            x = torch.cat([x, image_cond], dim=-1)
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
        timesteps = t
        pooled_embeds = y
        T5_encoder_hidden_states = context

        img_sizes = None

        # spatial forward
        batch_size = hidden_states.shape[0]
        hidden_states_type = hidden_states.dtype

        # 0. time
        timesteps = self.expand_timesteps(timesteps, batch_size, hidden_states.device)
        timesteps = self.t_embedder(timesteps, hidden_states_type)
        p_embedder = self.p_embedder(pooled_embeds)
        adaln_input = timesteps + p_embedder

        hidden_states, image_tokens_masks, img_sizes = self.patchify(hidden_states, self.max_seq, img_sizes)
        if image_tokens_masks is None:
            pH, pW = img_sizes[0]
            img_ids = torch.zeros(pH, pW, 3, device=hidden_states.device)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=hidden_states.device)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=hidden_states.device)[None, :]
            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
        hidden_states = self.x_embedder(hidden_states)

        # T5_encoder_hidden_states = encoder_hidden_states[0]
        encoder_hidden_states = encoder_hidden_states_llama3.movedim(1, 0)
        encoder_hidden_states = [encoder_hidden_states[k] for k in self.llama_layers]

        if self.caption_projection is not None:
            new_encoder_hidden_states = []
            for i, enc_hidden_state in enumerate(encoder_hidden_states):
                enc_hidden_state = self.caption_projection[i](enc_hidden_state)
                enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
                new_encoder_hidden_states.append(enc_hidden_state)
            encoder_hidden_states = new_encoder_hidden_states
            T5_encoder_hidden_states = self.caption_projection[-1](T5_encoder_hidden_states)
            T5_encoder_hidden_states = T5_encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
            encoder_hidden_states.append(T5_encoder_hidden_states)

        txt_ids = torch.zeros(
            batch_size,
            encoder_hidden_states[-1].shape[1] + encoder_hidden_states[-2].shape[1] + encoder_hidden_states[0].shape[1],
            3,
            device=img_ids.device, dtype=img_ids.dtype
        )
        ids = torch.cat((img_ids, txt_ids), dim=1)
        rope = self.pe_embedder(ids)

        # enable teacache
        modulated_inp = timesteps.to(mm.unet_offload_device())
        if not hasattr(self, 'teacache_state'):
            self.teacache_state = {
                0: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None},
                1: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None}
            }

        def update_cache_state(cache, modulated_inp):
            if cache['previous_modulated_input'] is not None:
                try:
                    cache['accumulated_rel_l1_distance'] += poly1d(coefficients, ((modulated_inp-cache['previous_modulated_input']).abs().mean() / cache['previous_modulated_input'].abs().mean()))
                    if cache['accumulated_rel_l1_distance'] < rel_l1_thresh:
                        cache['should_calc'] = False
                    else:
                        cache['should_calc'] = True
                        cache['accumulated_rel_l1_distance'] = 0
                except:
                    cache['should_calc'] = True
                    cache['accumulated_rel_l1_distance'] = 0
            cache['previous_modulated_input'] = modulated_inp
            
        b = int(len(hidden_states) / len(cond_or_uncond))

        for i, k in enumerate(cond_or_uncond):
            update_cache_state(self.teacache_state[k], modulated_inp[i*b:(i+1)*b])

        if enable_teacache:
            should_calc = False
            for k in cond_or_uncond:
                should_calc = (should_calc or self.teacache_state[k]['should_calc'])
        else:
            should_calc = True

        if not should_calc:
            for i, k in enumerate(cond_or_uncond):
                hidden_states[i*b:(i+1)*b] += self.teacache_state[k]['previous_residual'].to(hidden_states.device)
        else:
            # 2. Blocks
            ori_hidden_states = hidden_states.clone()
            block_id = 0
            initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
            initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]
            for bid, block in enumerate(self.double_stream_blocks):
                cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
                cur_encoder_hidden_states = torch.cat([initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1)
                hidden_states, initial_encoder_hidden_states = block(
                    image_tokens = hidden_states,
                    image_tokens_masks = image_tokens_masks,
                    text_tokens = cur_encoder_hidden_states,
                    adaln_input = adaln_input,
                    rope = rope,
                )
                initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
                block_id += 1

            image_tokens_seq_len = hidden_states.shape[1]
            hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
            hidden_states_seq_len = hidden_states.shape[1]
            if image_tokens_masks is not None:
                encoder_attention_mask_ones = torch.ones(
                    (batch_size, initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1]),
                    device=image_tokens_masks.device, dtype=image_tokens_masks.dtype
                )
                image_tokens_masks = torch.cat([image_tokens_masks, encoder_attention_mask_ones], dim=1)

            for bid, block in enumerate(self.single_stream_blocks):
                cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
                hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)
                hidden_states = block(
                    image_tokens=hidden_states,
                    image_tokens_masks=image_tokens_masks,
                    text_tokens=None,
                    adaln_input=adaln_input,
                    rope=rope,
                )
                hidden_states = hidden_states[:, :hidden_states_seq_len]
                block_id += 1

            hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
            for i, k in enumerate(cond_or_uncond):
                self.teacache_state[k]['previous_residual'] = (hidden_states - ori_hidden_states)[i*b:(i+1)*b].to(mm.unet_offload_device())

        output = self.final_layer(hidden_states, adaln_input)
        output = self.unpatchify(output, img_sizes)
        return -output[:, :, :h, :w]    

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
        guiding_frame_index=None,
        ref_latent=None,
        control=None,
        transformer_options={},
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        coefficients = transformer_options.get("coefficients")
        enable_teacache = transformer_options.get("enable_teacache", True)

        initial_shape = list(img.shape)
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        if ref_latent is not None:
            ref_latent_ids = self.img_ids(ref_latent)
            ref_latent = self.img_in(ref_latent)
            img = torch.cat([ref_latent, img], dim=-2)
            ref_latent_ids[..., 0] = -1
            ref_latent_ids[..., 2] += (initial_shape[-1] // self.patch_size[-1])
            img_ids = torch.cat([ref_latent_ids, img_ids], dim=-2)

        if guiding_frame_index is not None:
            token_replace_vec = self.time_in(timestep_embedding(guiding_frame_index, 256, time_factor=1.0))
            vec_ = self.vector_in(y[:, :self.params.vec_in_dim])
            vec = torch.cat([(vec_ + token_replace_vec).unsqueeze(1), (vec_ + vec).unsqueeze(1)], dim=1)
            frame_tokens = (initial_shape[-1] // self.patch_size[-1]) * (initial_shape[-2] // self.patch_size[-2])
            modulation_dims = [(0, frame_tokens, 0), (frame_tokens, None, 1)]
            modulation_dims_txt = [(0, None, 1)]
        else:
            vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
            modulation_dims = None
            modulation_dims_txt = None

        if self.params.guidance_embed:
            if guidance is not None:
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
        img_mod1, _ = self.double_blocks[0].img_mod(vec)
        modulated_inp = self.double_blocks[0].img_norm1(img)
        modulated_inp = apply_mod(modulated_inp, (1 + img_mod1.scale), img_mod1.shift, modulation_dims)

        if not hasattr(self, 'accumulated_rel_l1_distance'):
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            try:
                self.accumulated_rel_l1_distance += poly1d(coefficients, ((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()))
                if self.accumulated_rel_l1_distance < rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            except:
                should_calc = True
                self.accumulated_rel_l1_distance = 0

        self.previous_modulated_input = modulated_inp

        if not enable_teacache:
            should_calc = True

        if not should_calc:
            img += self.previous_residual.to(img.device)
        else:
            ori_img = img.clone()
            for i, block in enumerate(self.double_blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims_img=args["modulation_dims_img"], modulation_dims_txt=args["modulation_dims_txt"])
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims_img': modulation_dims, 'modulation_dims_txt': modulation_dims_txt}, {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims_img=modulation_dims, modulation_dims_txt=modulation_dims_txt)

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
                        out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims=args["modulation_dims"])
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims': modulation_dims}, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims=modulation_dims)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, : img_len] += add

            img = img[:, : img_len]
            self.previous_residual = (img - ori_img).to(mm.unet_offload_device())

        if ref_latent is not None:
            img = img[:, ref_latent.shape[1]:]
        
        img = self.final_layer(img, vec, modulation_dims=modulation_dims)  # (N, T, patch_size ** 2 * out_channels)

        shape = initial_shape[-3:]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])
        return img

def teacache_ltxvmodel_forward(
        self,
        x,
        timestep,
        context,
        attention_mask,
        frame_rate=25,
        transformer_options={},
        keyframe_idxs=None,
        **kwargs
    ):
        patches_replace = transformer_options.get("patches_replace", {})
        rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        coefficients = transformer_options.get("coefficients")
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        enable_teacache = transformer_options.get("enable_teacache", True)

        orig_shape = list(x.shape)

        x, latent_coords = self.patchifier.patchify(x)
        pixel_coords = latent_to_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self.vae_scale_factors,
            causal_fix=self.causal_temporal_positioning,
        )

        if keyframe_idxs is not None:
            pixel_coords[:, :, -keyframe_idxs.shape[2]:] = keyframe_idxs

        fractional_coords = pixel_coords.to(torch.float32)
        fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / frame_rate)

        x = self.patchify_proj(x)
        timestep = timestep * 1000.0

        if attention_mask is not None and not torch.is_floating_point(attention_mask):
            attention_mask = (attention_mask - 1).to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])) * torch.finfo(x.dtype).max        

        pe = precompute_freqs_cis(fractional_coords, dim=self.inner_dim, out_dtype=x.dtype)

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
        inp = x.to(mm.unet_offload_device())
        timestep_ = timestep.to(mm.unet_offload_device())
        num_ada_params = self.transformer_blocks[0].scale_shift_table.shape[0]
        ada_values = self.transformer_blocks[0].scale_shift_table[None, None].to(timestep_.device) + timestep_.reshape(batch_size, timestep_.size(1), num_ada_params, -1)
        shift_msa, scale_msa, _, _, _, _ = ada_values.unbind(dim=2)
        modulated_inp = comfy.ldm.common_dit.rms_norm(inp)
        modulated_inp = modulated_inp * (1 + scale_msa) + shift_msa

        if not hasattr(self, 'teacache_state'):
            self.teacache_state = {
                0: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None},
                1: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None}
            }

        def update_cache_state(cache, modulated_inp):
            if cache['previous_modulated_input'] is not None:
                try:
                    cache['accumulated_rel_l1_distance'] += poly1d(coefficients, ((modulated_inp-cache['previous_modulated_input']).abs().mean() / cache['previous_modulated_input'].abs().mean()))
                    if cache['accumulated_rel_l1_distance'] < rel_l1_thresh:
                        cache['should_calc'] = False
                    else:
                        cache['should_calc'] = True
                        cache['accumulated_rel_l1_distance'] = 0
                except:
                    cache['should_calc'] = True
                    cache['accumulated_rel_l1_distance'] = 0
            cache['previous_modulated_input'] = modulated_inp

        b = int(len(x) / len(cond_or_uncond))
        
        for i, k in enumerate(cond_or_uncond):
            update_cache_state(self.teacache_state[k], modulated_inp[i*b:(i+1)*b])

        if enable_teacache:
            should_calc = False
            for k in cond_or_uncond:
                should_calc = (should_calc or self.teacache_state[k]['should_calc'])
        else:
            should_calc = True
        
        if not should_calc:
            for i, k in enumerate(cond_or_uncond):
                x[i*b:(i+1)*b] += self.teacache_state[k]['previous_residual'].to(x.device)
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
            for i, k in enumerate(cond_or_uncond):
                self.teacache_state[k]['previous_residual'] = (x - ori_x)[i*b:(i+1)*b].to(mm.unet_offload_device())

        x = self.proj_out(x)

        x = self.patchifier.unpatchify(
            latents=x,
            output_height=orig_shape[3],
            output_width=orig_shape[4],
            output_num_frames=orig_shape[2],
            out_channels=orig_shape[1] // math.prod(self.patchifier.patch_size),
        )

        return x

def teacache_wanmodel_forward(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        patches_replace = transformer_options.get("patches_replace", {})
        rel_l1_thresh = transformer_options.get("rel_l1_thresh")
        coefficients = transformer_options.get("coefficients")
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        use_ret_mode = transformer_options.get("use_ret_mode")
        enable_teacache = transformer_options.get("enable_teacache", True)

        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        blocks_replace = patches_replace.get("dit", {})

        # enable teacache
        modulated_inp = e0.to(mm.unet_offload_device()) if use_ret_mode else e.to(mm.unet_offload_device())
        if not hasattr(self, 'teacache_state'):
            self.teacache_state = {
                0: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None},
                1: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None}
            }

        def update_cache_state(cache, modulated_inp):
            if cache['previous_modulated_input'] is not None:
                try:
                    cache['accumulated_rel_l1_distance'] += poly1d(coefficients, ((modulated_inp-cache['previous_modulated_input']).abs().mean() / cache['previous_modulated_input'].abs().mean()))
                    if cache['accumulated_rel_l1_distance'] < rel_l1_thresh:
                        cache['should_calc'] = False
                    else:
                        cache['should_calc'] = True
                        cache['accumulated_rel_l1_distance'] = 0
                except:
                    cache['should_calc'] = True
                    cache['accumulated_rel_l1_distance'] = 0
            cache['previous_modulated_input'] = modulated_inp
            
        b = int(len(x) / len(cond_or_uncond))

        for i, k in enumerate(cond_or_uncond):
            update_cache_state(self.teacache_state[k], modulated_inp[i*b:(i+1)*b])

        if enable_teacache:
            should_calc = False
            for k in cond_or_uncond:
                should_calc = (should_calc or self.teacache_state[k]['should_calc'])
        else:
            should_calc = True

        if not should_calc:
            for i, k in enumerate(cond_or_uncond):
                x[i*b:(i+1)*b] += self.teacache_state[k]['previous_residual'].to(x.device)
        else:
            ori_x = x.clone()
            for i, block in enumerate(self.blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                        return out
                    out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap, "transformer_options": transformer_options})
                    x = out["img"]
                else:
                    x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
            for i, k in enumerate(cond_or_uncond):
                self.teacache_state[k]['previous_residual'] = (x - ori_x)[i*b:(i+1)*b].to(mm.unet_offload_device())

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

class TeaCache:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the TeaCache will be applied to."}),
                "model_type": (["flux", "ltxv", "hunyuan_video", "hidream_i1_full", "wan2.1_t2v_1.3B", "wan2.1_t2v_14B", "wan2.1_i2v_480p_14B", "wan2.1_i2v_720p_14B", "wan2.1_t2v_1.3B_ret_mode", "wan2.1_t2v_14B_ret_mode", "wan2.1_i2v_480p_14B_ret_mode", "wan2.1_i2v_720p_14B_ret_mode"], {"default": "flux", "tooltip": "Supported diffusion model."}),
                "rel_l1_thresh": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The start percentage of the steps that will apply TeaCache."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The end percentage of the steps that will apply TeaCache."})
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_teacache"
    CATEGORY = "TeaCache"
    TITLE = "TeaCache"
    
    def apply_teacache(self, model, model_type: str, rel_l1_thresh: float, start_percent: float, end_percent: float):
        if rel_l1_thresh == 0:
            return (model,)

        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}
        new_model.model_options["transformer_options"]["rel_l1_thresh"] = rel_l1_thresh
        new_model.model_options["transformer_options"]["coefficients"] = SUPPORTED_MODELS_COEFFICIENTS[model_type]
        new_model.model_options["transformer_options"]["use_ret_mode"] = "ret_mode" in model_type
        diffusion_model = new_model.get_model_object("diffusion_model")

        if "flux" in model_type:
            is_cfg = False
            context = patch.multiple(
                diffusion_model,
                forward_orig=teacache_flux_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "hidream_i1" in model_type:
            is_cfg = True
            context = patch.multiple(
                diffusion_model,
                forward=teacache_hidream_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "ltxv" in model_type:
            is_cfg = True
            context = patch.multiple(
                diffusion_model,
                forward=teacache_ltxvmodel_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "hunyuan_video" in model_type:
            is_cfg = False
            context = patch.multiple(
                diffusion_model,
                forward_orig=teacache_hunyuanvideo_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "wan2.1" in model_type:
            is_cfg = True
            context = patch.multiple(
                diffusion_model,
                forward_orig=teacache_wanmodel_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        else:
            raise ValueError(f"Unknown type {model_type}")
        
        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            cond_or_uncond = kwargs["cond_or_uncond"]
            # referenced from https://github.com/kijai/ComfyUI-KJNodes/blob/d126b62cebee81ea14ec06ea7cd7526999cb0554/nodes/model_optimization_nodes.py#L868
            sigmas = c["transformer_options"]["sample_sigmas"]
            matched_step_index = (sigmas == timestep[0]).nonzero()
            if len(matched_step_index) > 0:
                current_step_index = matched_step_index.item()
            else:
                current_step_index = 0
                for i in range(len(sigmas) - 1):
                    # walk from beginning of steps until crossing the timestep
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        current_step_index = i
                        break
            
            if current_step_index == 0:
                if is_cfg:
                    # uncond first
                    if (1 in cond_or_uncond) and hasattr(diffusion_model, 'teacache_state'):
                        delattr(diffusion_model, 'teacache_state')
                else:
                    if hasattr(diffusion_model, 'accumulated_rel_l1_distance'):
                        delattr(diffusion_model, 'accumulated_rel_l1_distance')
            
            current_percent = current_step_index / (len(sigmas) - 1)
            if start_percent <= current_percent <= end_percent:
                c["transformer_options"]["enable_teacache"] = True
            else:
                c["transformer_options"]["enable_teacache"] = False
                
            with context:
                return model_function(input, timestep, **c)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)

        return (new_model,)
    
def patch_optimized_module():
    try:
        from torch._dynamo.eval_frame import OptimizedModule
    except ImportError:
        return

    if getattr(OptimizedModule, "_patched", False):
        return

    def __getattribute__(self, name):
        if name == "_orig_mod":
            return object.__getattribute__(self, "_modules")[name]
        if name in (
            "__class__",
            "_modules",
            "state_dict",
            "load_state_dict",
            "parameters",
            "named_parameters",
            "buffers",
            "named_buffers",
            "children",
            "named_children",
            "modules",
            "named_modules",
        ):
            return getattr(object.__getattribute__(self, "_orig_mod"), name)
        return object.__getattribute__(self, name)

    def __delattr__(self, name):
        return delattr(self._orig_mod, name)

    @classmethod
    def __instancecheck__(cls, instance):
        return isinstance(instance, OptimizedModule) or issubclass(
            object.__getattribute__(instance, "__class__"), cls
        )

    OptimizedModule.__getattribute__ = __getattribute__
    OptimizedModule.__delattr__ = __delattr__
    OptimizedModule.__instancecheck__ = __instancecheck__
    OptimizedModule._patched = True

def patch_same_meta():
    try:
        from torch._inductor.fx_passes import post_grad
    except ImportError:
        return

    same_meta = getattr(post_grad, "same_meta", None)
    if same_meta is None:
        return

    if getattr(same_meta, "_patched", False):
        return

    def new_same_meta(a, b):
        try:
            return same_meta(a, b)
        except Exception:
            return False

    post_grad.same_meta = new_same_meta
    new_same_meta._patched = True

class CompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the torch.compile will be applied to."}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "backend": (["inductor","cudagraphs", "eager", "aot_eager"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_compile"
    CATEGORY = "TeaCache"
    TITLE = "Compile Model"
    
    def apply_compile(self, model, mode: str, backend: str, fullgraph: bool, dynamic: bool):
        patch_optimized_module()
        patch_same_meta()
        torch._dynamo.config.suppress_errors = True
        
        new_model = model.clone()
        new_model.add_object_patch(
                                "diffusion_model",
                                torch.compile(
                                    new_model.get_model_object("diffusion_model"),
                                    mode=mode,
                                    backend=backend,
                                    fullgraph=fullgraph,
                                    dynamic=dynamic
                                )
                            )
        
        return (new_model,)
    

NODE_CLASS_MAPPINGS = {
    "TeaCache": TeaCache,
    "CompileModel": CompileModel
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
