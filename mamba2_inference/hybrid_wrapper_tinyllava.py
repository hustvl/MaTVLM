
# Copyright (c) 2023, Albert Gu, Tri Dao.
import os
import json

import torch
import torch.nn as nn

from dataclasses import dataclass, field
from mamba_ssm.utils.generation import GenerationMixin

from transformers import AutoModelForCausalLM

from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from transformers.utils.hub import cached_file

from transformers.models.phi.modeling_phi import PhiDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from mamba2_inference.hybrid_model import MambaDecoderLayer, MHADecoderLayer, PhiMHADecoderLayer, PhiMambaDecoderLayer
from util import load_safetensors_to_dict
from collections import namedtuple

from mamba2.hybrid_mamba_config import MambaConfig, PhiMambaConfig
from typing import List, Optional, Tuple, Union

from transformers.generation.streamers import TextStreamer
from mamba_ssm.utils.generation import *

def merge_projections_for_layers(checkpoint, layer_indices):
    for layer_idx in layer_indices:
        # Get the weights for q_proj, k_proj, and v_proj
        q_proj_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        q_proj_bias_key = f"model.layers.{layer_idx}.self_attn.q_proj.bias"
        k_proj_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        k_proj_bias_key = f"model.layers.{layer_idx}.self_attn.k_proj.bias"
        v_proj_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        v_proj_bias_key = f"model.layers.{layer_idx}.self_attn.v_proj.bias"
        o_proj_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        dense_key = f"model.layers.{layer_idx}.self_attn.dense.weight"
        dense_bias_key = f"model.layers.{layer_idx}.self_attn.dense.bias"

        # Check if the keys exist in the checkpoint
        if q_proj_key in checkpoint and k_proj_key in checkpoint and v_proj_key in checkpoint:
            # Assuming all the projections have the same shape, otherwise adjust accordingly
            q_proj_weight = checkpoint[q_proj_key]
            k_proj_weight = checkpoint[k_proj_key]
            v_proj_weight = checkpoint[v_proj_key]

            # Concatenate the weights along the first dimension (often dimension 0)
            in_proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)

            # Assign the new weight to the corresponding in_proj key
            in_proj_key = f"model.layers.{layer_idx}.mha.in_proj.weight"
            checkpoint[in_proj_key] = in_proj_weight

            # Optionally, remove the old keys to clean up the checkpoint
            del checkpoint[q_proj_key]
            del checkpoint[k_proj_key]
            del checkpoint[v_proj_key]

        if q_proj_bias_key in checkpoint and k_proj_bias_key in checkpoint and v_proj_bias_key in checkpoint:
            # Assuming all the projections have the same shape, otherwise adjust accordingly
            q_proj_bias = checkpoint[q_proj_bias_key]
            k_proj_bias = checkpoint[k_proj_bias_key]
            v_proj_bias = checkpoint[v_proj_bias_key]

            # Concatenate the weights along the first dimension (often dimension 0)
            in_proj_bias = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=0)

            # Assign the new weight to the corresponding in_proj key
            in_proj_bias_key = f"model.layers.{layer_idx}.mha.in_proj.bias"
            checkpoint[in_proj_bias_key] = in_proj_bias

            # Optionally, remove the old keys to clean up the checkpoint
            del checkpoint[q_proj_bias_key]
            del checkpoint[k_proj_bias_key]
            del checkpoint[v_proj_bias_key]
        if o_proj_key in checkpoint:
            out_proj_key = f"model.layers.{layer_idx}.mha.out_proj.weight"
            checkpoint[out_proj_key] = checkpoint[o_proj_key]
            del checkpoint[o_proj_key]
        elif dense_key in checkpoint:
            out_proj_key = f"model.layers.{layer_idx}.mha.out_proj.weight"
            checkpoint[out_proj_key] = checkpoint[dense_key]
            out_proj_bias = f"model.layers.{layer_idx}.mha.out_proj.bias"
            checkpoint[out_proj_bias] = checkpoint[dense_bias_key]

    return checkpoint

MAMBA_CONFIG_NAME = "mamba_config.json"

@torch.inference_mode()
def decode(
    input_ids,
    model,
    max_length,
    inputs_embeds=None,
    image_embeds=None,
    top_k=1,
    top_p=0.0,
    min_p=0.0,
    temperature=1.0,
    repetition_penalty=1.0,
    eos_token_id=None,
    teacher_outputs=None,
    vocab_size=None,
    cg=False,
    enable_timing=False,
    streamer: Optional[TextStreamer] = None
):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    if streamer is not None:
        streamer.put(input_ids.cpu())

    batch_size, seqlen_og = input_ids.shape

    if image_embeds is not None and inputs_embeds is not None:
        max_length += image_embeds.shape[1] - 1
        batch_size, seqlen_og = inputs_embeds.shape[:2]
        
    teacher_output_len = teacher_outputs.shape[1] if teacher_outputs is not None else 0
    if cg:
        if not hasattr(model, "_decoding_cache"):
            model._decoding_cache = None
        model._decoding_cache = update_graph_cache(
            model,
            model._decoding_cache,
            batch_size,
            seqlen_og,
            max_length,
        )
        inference_params = model._decoding_cache.inference_params
        inference_params.reset(max_length, batch_size)
    else:
        inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

    def get_logits(input_ids, inference_params, inputs_embeds=None):
        decoding = inference_params.seqlen_offset > 0
        if decoding:
            position_ids = torch.full(
                (batch_size, 1),
                inference_params.seqlen_offset,
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            position_ids = None
        if not cg or not decoding:
            logits = model(
                input_ids=input_ids if inputs_embeds is None else None,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                inference_params=inference_params,
                num_last_tokens=1,
            ).logits.squeeze(dim=1)
        else:
            logits = model._decoding_cache.run(
                input_ids, position_ids, inference_params.seqlen_offset
            ).squeeze(dim=1)
        return logits[..., :vocab_size] if vocab_size is not None else logits

    def sample_tokens(logits, inference_params):
        if teacher_outputs is None or teacher_output_len <= inference_params.seqlen_offset:
            token = sample(logits, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature)
        else:
            token = teacher_outputs[:, inference_params.seqlen_offset]
        # return rearrange(token, "b -> b 1")
        return token.unsqueeze(1)

    def should_stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).all():
            return True
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        return False

    start = torch.cuda.Event(enable_timing=enable_timing)
    end = torch.cuda.Event(enable_timing=enable_timing)

    if enable_timing:
        start.record()
    scores, sequences, sequences_embeds = [], [input_ids], [inputs_embeds]
    sequences_cat = input_ids
    # import pdb; pdb.set_trace()
    while not should_stop(sequences[-1], inference_params):
        scores.append(get_logits(sequences[-1], inference_params, inputs_embeds=sequences_embeds[-1]))
        inference_params.seqlen_offset += sequences_embeds[-1].shape[1]
        if repetition_penalty == 1.0:
            sampled_tokens = sample_tokens(scores[-1], inference_params)
        else:
            logits = modify_logit_for_repetition_penalty(
                scores[-1].clone(), sequences_cat, repetition_penalty
            )
            sampled_tokens = sample_tokens(logits, inference_params)
            sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)
        sequences.append(sampled_tokens)
        if inputs_embeds is not None:
            sequences_embeds.append(model.get_input_embeddings()(sampled_tokens))
        if streamer is not None:
            streamer.put(sampled_tokens.cpu())
    if streamer is not None:
        streamer.end()
    if enable_timing:
        end.record()
        torch.cuda.synchronize()
        print(f"Prompt processing + decoding time: {(start.elapsed_time(end)):.0f}ms")
    output_cls = GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    return output_cls(sequences=torch.cat(sequences, dim=1), scores=tuple(scores))


class EvalMamba2TransformerHybridModelWrapper(nn.Module):

    def __init__(self, checkpoint_path, transformer_model, mamba_config, attn_layers, dtype, load_from_hub=False, **kwargs):
        super(EvalMamba2TransformerHybridModelWrapper, self).__init__()
        self.mamba_config = mamba_config
        self.attn_layers = attn_layers
        self.model = transformer_model
        self.config = self.model.config
        
        for layer_idx in range(mamba_config.n_layer):
            if isinstance(self.model.model.layers[layer_idx], PhiDecoderLayer):
                if layer_idx in attn_layers:
                    layer_encoder = PhiMHADecoderLayer(
                        self.config,
                        layer_idx,
                        device="cuda",
                        dtype=dtype,
                    )
                else:
                    layer_encoder = PhiMambaDecoderLayer(
                        mamba_config,
                        layer_idx,
                        device="cuda",
                        dtype=dtype,
                    )
            else:
                if layer_idx in attn_layers:
                    layer_encoder = MHADecoderLayer(
                        self.config,
                        layer_idx,
                        device="cuda",
                        dtype=dtype,
                    )
                else:
                    layer_encoder = MambaDecoderLayer(
                        mamba_config,
                        layer_idx,
                        device="cuda",
                        dtype=dtype,
                    )
            self.model.model.layers[layer_idx] = layer_encoder
            
        print("self.model:", self.model)      
           
        if checkpoint_path is not None:
            if load_from_hub:
                # load from a huggingface hub
                ckpt = load_state_dict_hf(checkpoint_path, device=torch.device("cpu"), dtype=dtype)
            else:
                # load from a local directory
                if os.path.exists(f"{checkpoint_path}/pytorch_model.bin"):
                    # support save from bin file
                    ckpt = torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location=torch.device("cpu"))
                else:
                    # support save from safetensors
                    ckpt = load_safetensors_to_dict(checkpoint_path)
        
        keys = list(ckpt.keys())
        for k in keys:
            if "language_model" in k:
                ckpt[k.replace("language_model.model", "model")] = ckpt.pop(k)

        merge_projections_for_layers(ckpt, self.attn_layers)
        self.model.load_state_dict(ckpt, strict=False) # without lm_heads
        self.model = self.model.to(dtype).cuda()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.model.model.layers)
        }

    def generate(
        self,
        input_ids,
        max_new_tokens,
        inputs_embeds=None,
        image_embeds=None,
        top_k=1,
        top_p=0.0,
        min_p=0.0,
        temperature=1.0,
        return_dict_in_generate=False,
        output_scores=False,
        **kwargs,
    ):
        output = decode(
            input_ids, self, max_length=max_new_tokens, inputs_embeds=inputs_embeds, image_embeds=image_embeds,
            top_k=top_k, top_p=top_p, min_p = min_p, temperature=temperature, **kwargs
        )
        if not output_scores:
            output.scores = None
        return output if return_dict_in_generate else output.sequences
    
    def forward(self, input_ids, inputs_embeds=None, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        if inputs_embeds is None:
            hidden_states = self.model.model.embed_tokens(input_ids, **mixer_kwargs)
        else:
            hidden_states = inputs_embeds
        for decoder_layer in self.model.model.layers:
            hidden_states = decoder_layer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        if hasattr(self.model.model, "norm"):
            hidden_states = self.model.model.norm(hidden_states)
        elif hasattr(self.model.model, "final_layernorm"):
            hidden_states = self.model.model.final_layernorm(hidden_states)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.model.lm_head(hidden_states)
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        return CausalLMOutput(logits=lm_logits)

    @staticmethod
    def from_pretrained_local(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(config_data["llm_model_name_or_path"], torch_dtype=torch_dtype, attn_implementation=attn_implementation, trust_remote_code=True)
        with open(f'{pretrained_model_name}/{MAMBA_CONFIG_NAME}', 'r') as json_file:
            config_dict = json.load(json_file)
        if "phi" in config_data["llm_model_name_or_path"].lower():
            mamba_config = PhiMambaConfig(**config_dict)
        else:
            mamba_config = MambaConfig(**config_dict)
        return EvalMamba2TransformerHybridModelWrapper(pretrained_model_name, transformer_model, mamba_config, mamba_config.attn_layers, torch_dtype, init_with_kqvo=False) 

    @staticmethod
    def from_pretrained_hub(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        config_data = load_config_hf(pretrained_model_name)
        transformer_model = AutoModelForCausalLM.from_pretrained(config_data["llm_model_name_or_path"], torch_dtype=torch_dtype, attn_implementation=attn_implementation)
        resolved_archive_file = cached_file(pretrained_model_name, MAMBA_CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
        config_dict = json.load(open(resolved_archive_file))
        if "phi" in config_data["llm_model_name_or_path"].lower():
            mamba_config = PhiMambaConfig(**config_dict)
        else:
            mamba_config = MambaConfig(**config_dict)
        return EvalMamba2TransformerHybridModelWrapper(pretrained_model_name, transformer_model, mamba_config, mamba_config.attn_layers, torch_dtype, init_with_kqvo=False, load_from_hub=True) 

    @staticmethod
    def from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"):
        if os.path.exists(pretrained_model_name):
            return EvalMamba2TransformerHybridModelWrapper.from_pretrained_local(pretrained_model_name, torch_dtype, attn_implementation)
        else:
            return EvalMamba2TransformerHybridModelWrapper.from_pretrained_hub(pretrained_model_name, torch_dtype, attn_implementation)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def tie_weights(self):
        return self.model.tie_weights()
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        return  self.model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)        