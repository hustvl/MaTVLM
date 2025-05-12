# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPImageProcessor
# from mobilevlm.model.mobilellama import MobileLlamaForCausalLM
# from llava_phi.model import *
from transformers import LlamaTokenizer, LlamaForCausalLM


# from mamba_inference.hybrid_wrapper_tinyllava import EvalMambaTransformerHybridModelWrapper
# from mamba2_inference.hybrid_wrapper_tinyllava import EvalMamba2TransformerHybridModelWrapper


from tinyllava.model import *
from tinyllava.data import *
from tinyllava.utils import *

parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="JunxiongWang/MambaInLlama_0_50")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()

repeats = 3
device = "cuda"
dtype = torch.float16

print(f"Loading model {args.model_name}")
is_mamba2 = "mamba2" in args.model_name.lower()
is_mamba = "mamba" in args.model_name.lower() and "mamba2" not in args.model_name.lower()

if is_mamba2:
    model = HybridTinyLlavaForConditionalGeneration.from_pretrained(args.model_name,torch_dtype=dtype, attn_implementation='flash_attention_2')
    model.language_model = EvalMamba2TransformerHybridModelWrapper.from_pretrained(args.model_name, torch_dtype=dtype, attn_implementation='flash_attention_2')
    model.to(device=device)
    tokenizer = model.tokenizer
    image_processor = model.vision_tower._image_processor
elif is_mamba:
    model = HybridTinyLlavaForConditionalGeneration.from_pretrained(args.model_name,torch_dtype=dtype, attn_implementation='flash_attention_2')
    model.language_model = EvalMambaTransformerHybridModelWrapper.from_pretrained(args.model_name, torch_dtype=dtype, attn_implementation='flash_attention_2')
    model.to(device=device)
    tokenizer = model.tokenizer
    image_processor = model.vision_tower._image_processor
else:
    if "tinyllava" in args.model_name.lower():
        model = TinyLlavaForConditionalGeneration.from_pretrained(args.model_name,torch_dtype=dtype, device_map={"": device}, attn_implementation='flash_attention_2')
        tokenizer = model.tokenizer
        image_processor = model.vision_tower._image_processor
    elif "mobilevlm" in args.model_name.lower():
        model = LlamaForCausalLM.from_pretrained(args.model_name,torch_dtype=dtype, device_map={"": device}, attn_implementation='flash_attention_2')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        vision_tower = model.get_vision_tower()
        if 'v2' in getattr(model.config, "mm_projector_type", "ldpnet"):
            vision_tower.load_image_processor()
        elif not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor
    elif "llava-phi" in args.model_name.lower():
        model = LlavaPhiForCausalLM.from_pretrained(args.model_name,torch_dtype=dtype, device_map={"": device}, attn_implementation='flash_attention_2')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        image_processor = CLIPImageProcessor.from_pretrained(args.model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name,torch_dtype=dtype, device_map={"": device}, attn_implementation='flash_attention_2')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        vision_tower = model.get_vision_tower()
        image_processor = vision_tower.image_processor
        
data_args = model.config
image_processor = ImagePreprocess(image_processor, data_args)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

torch.random.manual_seed(0)
if args.prompt is None:
    input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
    # random insert Image token
    input_ids[0, torch.randint(0, args.promptlen, (1,))]=IMAGE_TOKEN_INDEX
    images = Image.open("serve/examples/extreme_ironing.jpg").convert("RGB")
    image_tensor = image_processor(images).unsqueeze(0).to(dtype=dtype, device=device)
else:
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    attn_mask = tokens.attention_mask.to(device=device)
max_length = input_ids.shape[1] + args.genlen

if is_mamba or is_mamba2:
    fn = lambda: model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[images.size],
        max_length=max_length,
        cg=True,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        min_p=args.minp,
        repetition_penalty=args.repetition_penalty,
    )
else:
    fn = lambda: model.generate(
        input_ids,
        images=image_tensor,
        # image_sizes=[images.size],
        max_new_tokens=args.genlen,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        repetition_penalty=args.repetition_penalty,
        use_cache=True,
        eos_token_id=None,
    )
torch.cuda.reset_peak_memory_stats(device)  # 重置显存统计
out = fn()
max_mem_before = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # 以MB为单位
print(f"Max memory allocated before benchmarking: {max_mem_before:.2f} MB")

if args.prompt is not None:
    print(tokenizer.batch_decode(out.sequences.tolist()))

torch.cuda.synchronize()
start = time.time()
for _ in range(repeats):
    fn()
torch.cuda.synchronize()

max_mem_after = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # 以MB为单位
print(f"Max memory allocated after benchmarking: {max_mem_after:.2f} MB")

if is_mamba or is_mamba2 or "mobilevlm" in args.model_name.lower():
    print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
else:
    print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0])}")
print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")