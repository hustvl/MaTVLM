from mamba_ssm.ops.triton.layer_norm import RMSNorm
from torch import Tensor
from transformers.activations import ACT2FN

from mamba2.hybrid_mamba_config import MambaConfig, PhiMambaConfig
from mamba2.hybrid_mamba_layer import Mamba2, InternLM2Mamba2

from mamba_ssm.modules.mha import MHA

import torch.nn as nn

from flash_attn.layers.rotary import *
import torch

from typing import Optional, Tuple, Union
from einops import rearrange

from internvl.model.internlm2.modeling_internlm2 import InternLM2RMSNorm

import torch.nn.functional as F
class PhiMLP(nn.Module):
    def __init__(self, d_model, intermediate_size, hidden_act, device=None, dtype=None,):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = d_model
        self.intermediate_size = intermediate_size
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size, **factory_kwargs)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size, **factory_kwargs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class PhiRotaryEmbedding(RotaryEmbedding):
    def _init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seqlen = qkv.shape[1]

        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)
        # Partial rotary embedding
        qkv_rot, qkv_pas = (
            qkv[..., : int(self.dim)],
            qkv[..., int(self.dim) :],
        )

        if kv is None:
            if self.scale is None:
                qkv_rot =  apply_rotary_emb_qkv_(
                    qkv_rot,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
                return torch.cat([qkv_rot, qkv_pas], dim=-1)
            else:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
        else:
            kv_rot, kv_pas = (
                kv[..., : int(self.dim)],
                kv[..., int(self.dim) :],
            )
            q_rot = qkv_rot
            q_rot = apply_rotary_emb_func(
                q_rot,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
            )
            q = torch.cat([q_rot, qkv_pas], dim=-1)
            if self.scale is None:
                kv_rot = apply_rotary_emb_kv_(
                    kv_rot,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            else:
                kv_rot = apply_rotary_emb_kv_(
                    kv_rot,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            kv = torch.cat([kv_rot, kv_pas], dim=-1)
            return q, kv

class PhiMHA(MHA):
    def __init__(self, rotary_emb_base, device, rotary_emb_interleaved=False, *args,  **kwargs):
        super().__init__(rotary_emb_interleaved=rotary_emb_interleaved,device=device, rotary_emb_base=rotary_emb_base, *args, **kwargs)

        if self.rotary_emb_dim > 0:
            assert PhiRotaryEmbedding is not None, "rotary requires flash_attn to be installed"
            self.rotary_emb = PhiRotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

    def forward(self, x, inference_params=None):
        if inference_params is not None and self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(
                x.shape[0], inference_params.max_seqlen, dtype=x.dtype
            )
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        qkv = self.in_proj(x)
        if self.mlp_dim > 0:
            qkv, x_mlp = qkv.split([qkv.shape[-1] - self.mlp_dim, self.mlp_dim], dim=-1)
            x_mlp_up, x_mlp_gate = x_mlp.chunk(2, dim=-1)
            x_mlp = x_mlp_up * F.silu(x_mlp_gate)
        if self.d_conv > 0:
            # The inference code for conv1d is pretty messy, should clean it up
            if (inference_params is None or inference_params.seqlen_offset == 0):
                if causal_conv1d_fn is None:
                    qkv = rearrange(
                        self.conv1d(rearrange(qkv, "b s d -> b d s"))[..., :-(self.d_conv - 1)], "b d s -> b s d"
                    ).contiguous()
                else:
                    qkv = causal_conv1d_fn(
                        qkv.transpose(1, 2),
                        rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        self.conv1d.bias
                    ).transpose(1, 2)
                if inference_params is not None:
                    _, conv_state = inference_params.key_value_memory_dict[self.layer_idx]
                    # If we just take qkv[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    qkv_t = rearrange(qkv, "b l d -> b d l")
                    conv_state.copy_(F.pad(qkv_t, (self.d_conv - qkv_t.shape[-1], 0)))  # Update state (B D W)
            else:
                _, conv_state = inference_params.key_value_memory_dict[self.layer_idx]
                assert qkv.shape[1] == 1, "Only support decoding with 1 token at a time for now"
                qkv = qkv.squeeze(1)
                # Conv step
                if causal_conv1d_update is None:
                    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
                    conv_state[:, :, -1] = qkv
                    qkv = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
                    if self.conv1d.bias is not None:
                        qkv = qkv + self.conv1d.bias
                else:
                    qkv = causal_conv1d_update(
                        qkv,
                        conv_state,
                        rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        self.conv1d.bias
                    )
                qkv = qkv.unsqueeze(1)
        q, kv = qkv.split([self.num_heads * self.head_dim, self.num_heads_kv * 2 * self.head_dim], dim=-1)
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
        if (
            inference_params is None
            or inference_params.seqlen_offset == 0
            or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
        ):
            if self.rotary_emb_dim > 0:
                # import pdb; pdb.set_trace()
                q, kv = self.rotary_emb(
                    q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                )

            if inference_params is None:
                k, v = kv.unbind(dim=-3)                
                k = torch.repeat_interleave(k, dim=2, repeats=self.num_heads // self.num_heads_kv)
                v = torch.repeat_interleave(v, dim=2, repeats=self.num_heads // self.num_heads_kv)
                context = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
                ).transpose(1, 2)
            else:
                context = self._update_kvcache_attention(q, kv, inference_params)
        else:
            context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)
        context = rearrange(context, "... h d -> ... (h d)")
        if self.mlp_dim > 0:
            context = torch.cat([context, x_mlp], dim=-1)
        out = self.out_proj(context)
        return out

class PhiMHADecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(PhiMHADecoderLayer, self).__init__()
        self.layer_idx = layer_idx
        self.mha = PhiMHA(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_heads_kv=config.num_key_value_heads,
            layer_idx=layer_idx,
            mlp_dim=0,
            qkv_proj_bias=True,
            out_proj_bias=True,
            rotary_emb_dim=config.partial_rotary_factor * config.hidden_size//config.num_attention_heads,
            rotary_emb_base=config.rope_theta,
            causal=True,
            device=device,
            dtype=dtype,
        )
        self.mlp = PhiMLP(config.hidden_size, config.intermediate_size, config.hidden_act, **factory_kwargs)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, **factory_kwargs)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.residual_in_fp32 = True

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mha.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
    def forward(self, hidden_states: Tensor, inference_params=None, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.mha(hidden_states, inference_params)
        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        hidden_states = attn_outputs + feed_forward_hidden_states + residual

        return hidden_states

class PhiMambaDecoderLayer(nn.Module):
    def __init__(self, config: PhiMambaConfig, layer_idx: int,
        device=None,
        dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.layer_idx = layer_idx
        self.mamba = Mamba2(
            d_model=config.d_model, d_xb=config.d_xb, d_inner=config.d_inner, layer_idx=layer_idx, **config.ssm_cfg, **factory_kwargs
        )
        self.mlp = PhiMLP(config.d_model, config.intermediate_size, config.hidden_act, **factory_kwargs)
        self.input_layernorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, **factory_kwargs)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,inference_params=None,*args, **kwargs
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        attn_outputs = self.mamba(hidden_states, inference_params=inference_params)

        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        
        return hidden_states

class MLP(nn.Module):
    def __init__(self, d_model, intermediate_size, hidden_act, device=None, dtype=None,):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.hidden_size = d_model
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, **factory_kwargs)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False, **factory_kwargs)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False, **factory_kwargs)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class MHADecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MHADecoderLayer, self).__init__()
        self.layer_idx = layer_idx
        self.mha = MHA(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_heads_kv=config.num_key_value_heads,
            layer_idx=layer_idx,
            mlp_dim=0,
            qkv_proj_bias=False,
            out_proj_bias=False,
            rotary_emb_dim=config.hidden_size//config.num_attention_heads,
            rotary_emb_base=config.rope_theta,
            causal=True,
            device=device,
            dtype=dtype,
        )
        self.mlp = MLP(config.hidden_size, config.intermediate_size, config.hidden_act, **factory_kwargs)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, **factory_kwargs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, **factory_kwargs)
        self.residual_in_fp32 = True

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mha.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
    def forward(self, hidden_states: Tensor, inference_params=None, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mha(hidden_states, inference_params)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class InternLM2MLP(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.intermediate_size = config.intermediate_size
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.w2(self.act_fn(self.w1(x)) * self.w3(x))

        return down_proj

class InternLM2MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        device=None,
        dtype=None,
        residual_in_fp32=True,
    ):
        super(InternLM2MambaDecoderLayer, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.mamba = InternLM2Mamba2(
            d_model=config.d_model, d_xb=config.d_xb, d_inner=config.d_inner, layer_idx=layer_idx, **config.ssm_cfg, **factory_kwargs
        )
        self.mlp = InternLM2MLP(config, config.d_model)
        self.input_layernorm = InternLM2RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = InternLM2RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.residual_in_fp32 = True
        
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, hidden_states: Tensor, inference_params=None, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states, inference_params=inference_params)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class InternLM2MHA(MHA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, inference_params=None):
        if inference_params is not None and self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(
                x.shape[0], inference_params.max_seqlen, dtype=x.dtype
            )
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        qkv = self.in_proj(x)
        if self.mlp_dim > 0:
            qkv, x_mlp = qkv.split([qkv.shape[-1] - self.mlp_dim, self.mlp_dim], dim=-1)
            x_mlp_up, x_mlp_gate = x_mlp.chunk(2, dim=-1)
            x_mlp = x_mlp_up * F.silu(x_mlp_gate)
        if self.d_conv > 0:
            # The inference code for conv1d is pretty messy, should clean it up
            if (inference_params is None or inference_params.seqlen_offset == 0):
                if causal_conv1d_fn is None:
                    qkv = rearrange(
                        self.conv1d(rearrange(qkv, "b s d -> b d s"))[..., :-(self.d_conv - 1)], "b d s -> b s d"
                    ).contiguous()
                else:
                    qkv = causal_conv1d_fn(
                        qkv.transpose(1, 2),
                        rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        self.conv1d.bias
                    ).transpose(1, 2)
                if inference_params is not None:
                    _, conv_state = inference_params.key_value_memory_dict[self.layer_idx]
                    # If we just take qkv[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    qkv_t = rearrange(qkv, "b l d -> b d l")
                    conv_state.copy_(F.pad(qkv_t, (self.d_conv - qkv_t.shape[-1], 0)))  # Update state (B D W)
            else:
                _, conv_state = inference_params.key_value_memory_dict[self.layer_idx]
                assert qkv.shape[1] == 1, "Only support decoding with 1 token at a time for now"
                qkv = qkv.squeeze(1)
                # Conv step
                if causal_conv1d_update is None:
                    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
                    conv_state[:, :, -1] = qkv
                    qkv = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
                    if self.conv1d.bias is not None:
                        qkv = qkv + self.conv1d.bias
                else:
                    qkv = causal_conv1d_update(
                        qkv,
                        conv_state,
                        rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        self.conv1d.bias
                    )
                qkv = qkv.unsqueeze(1)
        

        num_key_value_groups = self.num_heads // self.num_heads_kv
        qkv_states = rearrange(
            qkv,
            'b q (h gs d) -> b q h gs d',
            gs=2 + num_key_value_groups,
            d=self.head_dim,
        )

        q = qkv_states[..., : num_key_value_groups, :]
        q = rearrange(q, 'b q h gs d -> b q (h gs) d')
        key_states = qkv_states[..., -2:-1, :].transpose(2,3)
        value_states = qkv_states[..., -1:, :].transpose(2,3)
        kv = torch.concat([key_states, value_states], dim=2)

        if (
            inference_params is None
            or inference_params.seqlen_offset == 0
            or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
        ):
            if self.rotary_emb_dim > 0:
                q, kv = self.rotary_emb(
                    q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                )
            if inference_params is None:
                k, v = kv.unbind(dim=-3)                
                k = torch.repeat_interleave(k, dim=2, repeats=self.num_heads // self.num_heads_kv)
                v = torch.repeat_interleave(v, dim=2, repeats=self.num_heads // self.num_heads_kv)
                context = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
                ).transpose(1, 2)
            else:
                context = self._update_kvcache_attention(q, kv, inference_params)
        else:
            context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)
        context = rearrange(context, "... h d -> ... (h d)")
        if self.mlp_dim > 0:
            context = torch.cat([context, x_mlp], dim=-1)
        out = self.out_proj(context)
        return out

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        if (
            inference_params.seqlen_offset == 0
            or flash_attn_with_kvcache is None
        ):
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            k, v = kv.unbind(dim=-3)
            k = torch.repeat_interleave(k, dim=2, repeats=self.num_heads // self.num_heads_kv)
            v = torch.repeat_interleave(v, dim=2, repeats=self.num_heads // self.num_heads_kv)
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=self.causal, scale=self.softmax_scale
            ).transpose(1, 2)
        else:
            batch = q.shape[0]
            kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            return flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
            )

class InternLM2MHADecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(InternLM2MHADecoderLayer, self).__init__()
        self.layer_idx = layer_idx
        self.mha = InternLM2MHA(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_heads_kv=config.num_key_value_heads,
            layer_idx=layer_idx,
            mlp_dim=0,
            qkv_proj_bias=False,
            out_proj_bias=False,
            rotary_emb_dim=config.hidden_size//config.num_attention_heads,
            rotary_emb_base=config.rope_theta,
            causal=True,
            device=device,
            dtype=dtype,
        )
        self.feed_forward = InternLM2MLP(config, config.hidden_size)
        self.attention_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.residual_in_fp32 = True

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mha.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
    def forward(self, hidden_states: Tensor, inference_params=None, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.mha(hidden_states, inference_params)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        device=None,
        dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MambaDecoderLayer, self).__init__()
        self.layer_idx = layer_idx
        self.mamba = Mamba2(
            d_model=config.d_model, d_inner=config.d_inner, d_xb=config.d_xb, layer_idx=layer_idx, **config.ssm_cfg, **factory_kwargs
        )
        self.mlp = MLP(config.d_model, config.intermediate_size, config.hidden_act, **factory_kwargs)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps, **factory_kwargs)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
    
    def forward(self, hidden_states: Tensor, inference_params=None, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states, inference_params=inference_params)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states