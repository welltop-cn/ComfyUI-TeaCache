import torch
import numpy as np

from einops import rearrange
from typing import Optional, Tuple, Union
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from .models.cogvideox.custom_cogvideox_transformer_3d import CogVideoXTransformer3DModel
from .models.cogvideox.enhance_a_video.globals import set_num_frames

def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result.abs()

def fft(tensor):
    tensor_fft = torch.fft.fft2(tensor)
    tensor_fft_shifted = torch.fft.fftshift(tensor_fft)
    B, C, H, W = tensor.size()
    radius = min(H, W) // 5
            
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W))
    center_x, center_y = W // 2, H // 2
    mask = (X - center_x) ** 2 + (Y - center_y) ** 2 <= radius ** 2
    low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(tensor.device)
    high_freq_mask = ~low_freq_mask
            
    low_freq_fft = tensor_fft_shifted * low_freq_mask
    high_freq_fft = tensor_fft_shifted * high_freq_mask

    return low_freq_fft, high_freq_fft

def teacache_cogvideox_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        controlnet_states: torch.Tensor = None,
        controlnet_weights: Optional[Union[float, int, list, np.ndarray, torch.FloatTensor]] = 1.0,
        video_flow_features: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        batch_size, num_frames, channels, height, width = hidden_states.shape

        set_num_frames(num_frames) # enhance a video global
   
        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        if self.ofs_embedding is not None: #1.5 I2V
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # enable teacache
        if not hasattr(self, 'accumulated_rel_l1_distance'):
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            try:
                if not self.config.use_rotary_positional_embeddings:
                    # CogVideoX-2B
                    coefficients = [-3.10658903e+01, 2.54732368e+01, -5.92380459e+00, 1.75769064e+00, -3.61568434e-03]
                else:
                    # CogVideoX-5B
                    coefficients = [-1.53880483e+03, 8.43202495e+02, -1.34363087e+02, 7.97131516e+00, -5.23162339e-02]
                
                self.accumulated_rel_l1_distance += poly1d(coefficients, ((emb-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()))
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            except:
                should_calc = True
                self.accumulated_rel_l1_distance = 0

        self.previous_modulated_input = emb

        if self.use_fastercache:
            self.fastercache_counter += 1
        if self.fastercache_counter >= self.fastercache_start_step + 3 and self.fastercache_counter % 5 != 0:
            if not should_calc:
                hidden_states += self.previous_residual
                encoder_hidden_states += self.previous_residual_encoder
            else:
                ori_hidden_states = hidden_states.clone()
                ori_encoder_hidden_states = encoder_hidden_states.clone()
                # 3. Transformer blocks
                for i, block in enumerate(self.transformer_blocks):
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states[:1],
                        encoder_hidden_states=encoder_hidden_states[:1],
                        temb=emb[:1],
                        image_rotary_emb=image_rotary_emb,
                        video_flow_feature=video_flow_features[i][:1] if video_flow_features is not None else None,
                        fuser = self.fuser_list[i] if self.fuser_list is not None else None,
                        block_use_fastercache = i <= self.fastercache_num_blocks_to_cache,
                        fastercache_counter = self.fastercache_counter,
                        fastercache_start_step = self.fastercache_start_step,
                        fastercache_device = self.fastercache_device
                    )

                    if (controlnet_states is not None) and (i < len(controlnet_states)):
                        controlnet_states_block = controlnet_states[i]
                        controlnet_block_weight = 1.0
                        if isinstance(controlnet_weights, (list, np.ndarray)) or torch.is_tensor(controlnet_weights):
                            controlnet_block_weight = controlnet_weights[i]
                        elif isinstance(controlnet_weights, (float, int)):
                            controlnet_block_weight = controlnet_weights
                        
                        hidden_states = hidden_states + controlnet_states_block * controlnet_block_weight
                self.previous_residual = hidden_states - ori_hidden_states
                self.previous_residual_encoder = encoder_hidden_states - ori_encoder_hidden_states
                    
            if not self.config.use_rotary_positional_embeddings:
                # CogVideoX-2B
                hidden_states = self.norm_final(hidden_states)
            else:
                # CogVideoX-5B
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                hidden_states = self.norm_final(hidden_states)
                hidden_states = hidden_states[:, text_seq_length:]

            # 4. Final block
            hidden_states = self.norm_out(hidden_states, temb=emb[:1])
            hidden_states = self.proj_out(hidden_states)

            # 5. Unpatchify
            # Note: we use `-1` instead of `channels`:
            #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
            #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
            
            if p_t is None:
                output = hidden_states.reshape(1, num_frames, height // p, width // p, -1, p, p)
                output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
            else:
                output = hidden_states.reshape(
                    1, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
                )
                output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)
            
            (bb, tt, cc, hh, ww) = output.shape
            cond = rearrange(output, "B T C H W -> (B T) C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
            lf_c, hf_c = fft(cond.float())

            if self.fastercache_counter <= self.fastercache_lf_step:
                self.delta_lf = self.delta_lf * 1.1
            if self.fastercache_counter >= self.fastercache_hf_step:
                self.delta_hf = self.delta_hf * 1.1

            new_hf_uc = self.delta_hf + hf_c
            new_lf_uc = self.delta_lf + lf_c

            combine_uc = new_lf_uc + new_hf_uc
            combined_fft = torch.fft.ifftshift(combine_uc)
            recovered_uncond = torch.fft.ifft2(combined_fft).real
            recovered_uncond = rearrange(recovered_uncond.to(output.dtype), "(B T) C H W -> B T C H W", B=bb, C=cc, T=tt, H=hh, W=ww)
            output = torch.cat([output, recovered_uncond])
        else:
            if not should_calc:
                hidden_states += self.previous_residual
                encoder_hidden_states += self.previous_residual_encoder
            else:
                ori_hidden_states = hidden_states.clone()
                ori_encoder_hidden_states = encoder_hidden_states.clone()
                for i, block in enumerate(self.transformer_blocks):
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                        video_flow_feature=video_flow_features[i] if video_flow_features is not None else None,
                        fuser = self.fuser_list[i] if self.fuser_list is not None else None,
                        block_use_fastercache = i <= self.fastercache_num_blocks_to_cache,
                        fastercache_counter = self.fastercache_counter,
                        fastercache_start_step = self.fastercache_start_step,
                        fastercache_device = self.fastercache_device
                    )

                    # controlnet
                    if (controlnet_states is not None) and (i < len(controlnet_states)):
                        controlnet_states_block = controlnet_states[i]
                        controlnet_block_weight = 1.0
                        if isinstance(controlnet_weights, (list, np.ndarray)) or torch.is_tensor(controlnet_weights):
                            controlnet_block_weight = controlnet_weights[i]
                            print(controlnet_block_weight)
                        elif isinstance(controlnet_weights, (float, int)):
                            controlnet_block_weight = controlnet_weights                    
                        hidden_states = hidden_states + controlnet_states_block * controlnet_block_weight
                self.previous_residual = hidden_states - ori_hidden_states
                self.previous_residual_encoder = encoder_hidden_states - ori_encoder_hidden_states
                    
            if not self.config.use_rotary_positional_embeddings:
                # CogVideoX-2B
                hidden_states = self.norm_final(hidden_states)
            else:
                # CogVideoX-5B
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                hidden_states = self.norm_final(hidden_states)
                hidden_states = hidden_states[:, text_seq_length:]

            # 4. Final block
            hidden_states = self.norm_out(hidden_states, temb=emb)
            hidden_states = self.proj_out(hidden_states)

            # 5. Unpatchify
            # Note: we use `-1` instead of `channels`:
            #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
            #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
            
            if p_t is None:
                output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
                output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
            else:
                output = hidden_states.reshape(
                    batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
                )
                output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

            if self.fastercache_counter >= self.fastercache_start_step + 1: 
                (bb, tt, cc, hh, ww) = output.shape
                cond = rearrange(output[0:1].float(), "B T C H W -> (B T) C H W", B=bb//2, C=cc, T=tt, H=hh, W=ww)
                uncond = rearrange(output[1:2].float(), "B T C H W -> (B T) C H W", B=bb//2, C=cc, T=tt, H=hh, W=ww)

                lf_c, hf_c = fft(cond)
                lf_uc, hf_uc = fft(uncond)

                self.delta_lf = lf_uc - lf_c
                self.delta_hf = hf_uc - hf_c

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

class TeaCacheForCogVideoX:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("COGVIDEOMODEL", {"tooltip": "The CogVideoX model the TeaCache will be applied to."}),
                "enable_teacache": ("BOOLEAN", {"default": True, "tooltip": "Enable teacache will speed up inference but may lose visual quality."}),
                "rel_l1_thresh": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."})
            }
        }
    
    RETURN_TYPES = ("COGVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_teacache"
    CATEGORY = "TeaCache"
    TITLE = "TeaCache For CogVideoX"
    
    def apply_teacache(self, model, enable_teacache: bool, rel_l1_thresh: float):
        if enable_teacache:
            transformer = model["pipe"].transformer
            transformer.__class__.rel_l1_thresh = rel_l1_thresh
            transformer.forward = teacache_cogvideox_forward.__get__(
                                transformer,
                                transformer.__class__
                            )
        else:
            transformer = model["pipe"].transformer
            transformer.forward = CogVideoXTransformer3DModel.forward.__get__(
                                transformer,
                                transformer.__class__
                            )
        
        return (model,)
    

NODE_CLASS_MAPPINGS = {
    "TeaCacheForCogVideoX": TeaCacheForCogVideoX
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
