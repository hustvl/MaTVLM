# Copyright (c) 2023, Albert Gu, Tri Dao.
import os
import json

import torch
import torch.nn as nn

from dataclasses import dataclass, field

from transformers import AutoModelForCausalLM

from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from transformers.utils.hub import cached_file

from mamba2.hybrid_model import PhiMambaDecoderLayer

from tinyllava.model.modeling_tinyllava import HybridTinyLlavaForConditionalGeneration


from util import load_safetensors_to_dict

MAMBA_CONFIG_NAME = "mamba_config.json"

class TinyllavaMambaTransformerHybridModelWrapper(nn.Module):

    def __init__(self, checkpoint_path, transformer_model, mamba_config, attn_layers, dtype, init_with_kqvo, load_from_hub=False, **kwargs):
        super(TinyllavaMambaTransformerHybridModelWrapper, self).__init__()
        self.mamba_config = mamba_config
        self.attn_layers = attn_layers
        self.model = transformer_model
        self.config = self.model.config
        
        for layer_idx in range(mamba_config.n_layer):
            if layer_idx not in attn_layers:
                mamba_encoder = PhiMambaDecoderLayer(
                    mamba_config,
                    layer_idx,
                    device="cuda",
                    dtype=dtype,
                )
                
                if init_with_kqvo:
                    # init weights using attention weights
                    mamba_encoder.mlp.load_state_dict(transformer_model.language_model.model.layers._modules[f'{layer_idx}'].mlp.state_dict())
                    mamba_encoder.input_layernorm.load_state_dict(transformer_model.language_model.model.layers._modules[f'{layer_idx}'].input_layernorm.state_dict())
                    mamba_encoder.mamba.out_proj.weight.data.copy_(transformer_model.language_model.model.layers._modules[f'{layer_idx}'].self_attn.dense.weight.data)
                    # [z, x, B, C, dt]
                    mamba_encoder.mamba.in_proj.weight.data[mamba_config.d_inner:mamba_config.d_inner+mamba_config.d_xb, :].copy_(transformer_model.language_model.model.layers._modules[f'{layer_idx}'].self_attn.v_proj.weight.data)
                    mamba_encoder.mamba.in_proj.weight.data[mamba_config.d_inner+mamba_config.d_xb:mamba_config.d_inner+2*mamba_config.d_xb, :].copy_(transformer_model.language_model.model.layers._modules[f'{layer_idx}'].self_attn.k_proj.weight.data)
                    mamba_encoder.mamba.in_proj.weight.data[mamba_config.d_inner+2*mamba_config.d_xb:2*mamba_config.d_inner+2*mamba_config.d_xb, :].copy_(transformer_model.language_model.model.layers._modules[f'{layer_idx}'].self_attn.q_proj.weight.data)
                    
                    if mamba_config.is_bias:
                        mamba_encoder.mamba.out_proj.bias.data.copy_(transformer_model.language_model.model.layers._modules[f'{layer_idx}'].self_attn.dense.bias.data)
                        mamba_encoder.mamba.in_proj.bias.data[mamba_config.d_inner:mamba_config.d_inner+mamba_config.d_xb].copy_(transformer_model.language_model.model.layers._modules[f'{layer_idx}'].self_attn.v_proj.bias.data)
                        mamba_encoder.mamba.in_proj.bias.data[mamba_config.d_inner+mamba_config.d_xb:mamba_config.d_inner+2*mamba_config.d_xb].copy_(transformer_model.language_model.model.layers._modules[f'{layer_idx}'].self_attn.k_proj.bias.data)
                        mamba_encoder.mamba.in_proj.bias.data[mamba_config.d_inner+2*mamba_config.d_xb:2*mamba_config.d_inner+2*mamba_config.d_xb].copy_(transformer_model.language_model.model.layers._modules[f'{layer_idx}'].self_attn.q_proj.bias.data)
                    # keep dtype to be the same
                    mamba_encoder.mlp = mamba_encoder.mlp.to(dtype)
                    mamba_encoder.input_layernorm = mamba_encoder.input_layernorm.to(dtype)

                self.model.language_model.model.layers[layer_idx] = mamba_encoder

        if checkpoint_path is not None:
            if load_from_hub:
                # load from a huggingface hub
                self.model.load_state_dict(load_state_dict_hf(checkpoint_path, device=torch.device("cpu"), dtype=dtype))
            else:
                # load from a local directory
                if os.path.exists(f"{checkpoint_path}/pytorch_model.bin"):
                    # support save from bin file
                    self.model.load_state_dict(torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location=torch.device("cpu")))
                else:
                    # support save from safetensors
                    self.model.load_state_dict(load_safetensors_to_dict(checkpoint_path))
        
        self.model = self.model.to(dtype).cuda()

    def allocate_mamba_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.model.model.layers)
            if isinstance(layer, PhiMambaDecoderLayer)
        }

    def forward(
        self,
        input_ids,
        **kwargs,
    ):
        return self.model(input_ids, **kwargs)

    def generate(
        self,
        input_ids,
        **kwargs,
    ):
        output = self.model.generate(
            input_ids,
            use_cache=False,
            **kwargs,
        )
        return output
    
    @staticmethod
    def init_distillation(
        checkpoint_path,
        tranformer_name,
        mamba_config,
        attn_layers,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        init_with_kqvo=True,
        **kwargs,
    ):
        transformer_model = HybridTinyLlavaForConditionalGeneration.from_pretrained(tranformer_name, torch_dtype=dtype, attn_implementation=attn_implementation)
        return TinyllavaMambaTransformerHybridModelWrapper(checkpoint_path, transformer_model, mamba_config, attn_layers, dtype, init_with_kqvo)
    
    def save_config(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, 'mamba_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.mamba_config.__dict__, f, indent=4)
