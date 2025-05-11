from mamba_ssm.ops.triton.layer_norm import RMSNorm
from torch import Tensor
from transformers.activations import ACT2FN

from mamba2.hybrid_mamba_config import MambaConfig, PhiMambaConfig
from mamba2.hybrid_mamba_layer import Mamba2, InternLM2Mamba2

import torch
import torch.nn as nn

from transformers.models.phi.modeling_phi import *
from internvl.model.internlm2.modeling_internlm2 import InternLM2RMSNorm

from internvl.model.qwen2.modeling_qwen2 import Qwen2RMSNorm

# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Phi
class PhiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.intermediate_size = config.intermediate_size
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
class PhiMambaDecoderLayer(nn.Module):
    def __init__(self, config: PhiMambaConfig, layer_idx: int,
        device=None,
        dtype=None):
        super().__init__()
        self.mlp = PhiMLP(config)
        self.input_layernorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        if not config.bidirectional:
            self.mamba = Mamba2(
                d_model=config.d_model, d_xb=config.d_xb, d_inner=config.d_inner, layer_idx=layer_idx, bias=config.is_bias, **config.ssm_cfg, **factory_kwargs
            )
        else:
            self.mamba = BidMamba2(
                d_model=config.d_model, d_xb=config.d_xb, d_inner=config.d_inner, layer_idx=layer_idx, **config.ssm_cfg, **factory_kwargs
            )
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,*args, **kwargs
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        attn_outputs = self.mamba(hidden_states)

        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        
        # so here is just to be compatible with Transformer
        if kwargs is None:
            return (hidden_states, None, None)
        else:
            past_key_value = kwargs.get("past_key_value", None)
            if past_key_value is not None:
                dummy_keys = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                dummy_values = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                # Update kv cache with dummy values
                past_key_value.update(dummy_keys, dummy_values, self.layer_idx)
            return (hidden_states, None, past_key_value)


class HybridPhiModel(PhiModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        teacher_outputs=None,
        ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        inputs_embeds = self.embed_dropout(inputs_embeds)
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_teacher_hidden_states = (teacher_outputs[0], ) if output_hidden_states and teacher_outputs is not None else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        mamba_layer_id = 0
        for layer_id, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if isinstance(decoder_layer, PhiMambaDecoderLayer):
                mamba_layer_id += 1
            
            hidden_states_run = [hidden_states]
            if teacher_outputs is not None:
                teacher_hidden_states = teacher_outputs[layer_id]
                hidden_states_run.append(teacher_hidden_states)

            hidden_states_out = []
            for hidden_states_now in hidden_states_run:
                if self.gradient_checkpointing and self.training and isinstance(decoder_layer, PhiDecoderLayer):
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states_now,
                        causal_mask,
                        position_ids,
                        output_attentions,
                        use_cache,
                        past_key_values,
                        cache_position,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states_now,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )
                hidden_states_out.append(layer_outputs[0])

            hidden_states = hidden_states_out[0] # type: ignore

            if teacher_outputs is not None and output_hidden_states and layer_id != len(self.layers) - 1:
                all_teacher_hidden_states += (hidden_states_out[1],)

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            if teacher_outputs is not None:
                all_teacher_hidden_states += (self.final_layernorm(hidden_states_out[1]),)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=(all_hidden_states, all_teacher_hidden_states),
            attentions=all_self_attns,
        )


class HybridPhiForCausalLM(PhiForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = HybridPhiModel(config)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        teacher_outputs=None,
        ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            teacher_outputs=teacher_outputs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MLP(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        device=None,
        dtype=None,
        residual_in_fp32=True,
    ):
        super(MambaDecoderLayer, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.mamba = Mamba2(
            d_model=config.d_model, d_xb=config.d_xb, d_inner=config.d_inner, layer_idx=layer_idx, **config.ssm_cfg, **factory_kwargs
        )
        self.mlp = MLP(config=config)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.residual_in_fp32 = True
        
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        # hidden_states = self.mamba(hidden_states, inference_params=inference_params)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # so here is just to be compatible with Transformer
        if kwargs is None:
            return (hidden_states, None, None)
        else:
            past_key_value = kwargs.get("past_key_value", None)
            if past_key_value is not None:
                dummy_keys = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                dummy_values = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                # Update kv cache with dummy values
                past_key_value.update(dummy_keys, dummy_values, self.layer_idx)
            return (hidden_states, None, past_key_value)



# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.d_model
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        device=None,
        dtype=None,
        residual_in_fp32=True,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.mamba = Mamba2(
            d_model=config.d_model, d_xb=config.d_xb, d_inner=config.d_inner, layer_idx=layer_idx, **config.ssm_cfg, **factory_kwargs
        )
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.d_model, eps=config.rms_norm_eps)

        
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        # hidden_states = self.mamba(hidden_states, inference_params=inference_params)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # so here is just to be compatible with Transformer
        if kwargs is None:
            return (hidden_states, None, None)
        else:
            past_key_value = kwargs.get("past_key_value", None)
            if past_key_value is not None:
                dummy_keys = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                dummy_values = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                # Update kv cache with dummy values
                past_key_value.update(dummy_keys, dummy_values, self.layer_idx)
            return (hidden_states, None, past_key_value)
        
class InternLM2MLP(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
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
        self.mlp = InternLM2MLP(config=config)
        self.input_layernorm = InternLM2RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = InternLM2RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.residual_in_fp32 = True
        
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        # hidden_states = self.mamba(hidden_states, inference_params=inference_params)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # so here is just to be compatible with Transformer
        if kwargs is None:
            return (hidden_states, None, None)
        else:
            past_key_value = kwargs.get("past_key_value", None)
            if past_key_value is not None:
                dummy_keys = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                dummy_values = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                # Update kv cache with dummy values
                past_key_value.update(dummy_keys, dummy_values, self.layer_idx)
            return (hidden_states, None, past_key_value)