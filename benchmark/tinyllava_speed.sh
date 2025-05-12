#!/bin/bash

# Define the models
models=(
  # "meta-llama/Llama-3.2-3B-Instruct"
  # "JunxiongWang/Llama3.2-Mamba2-3B-distill"
  # "output/TinyLLaVA-Phi-2-SigLIP-3.1B_0.25_mamba2_llava_v1_5_mix665k_gt_kl0.1"
  # "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"
  # "mtgv/MobileVLM-3B"
  "liuhaotian/llava-v1.5-7b"
)

# Define the generation lengths
# gen_lens=(16384 32768 65536)
gen_lens=(1024)


# Loop through models and generation lengths
for genlen in "${gen_lens[@]}"; do
  for model in "${models[@]}"; do
    # echo "Running benchmark for model $model with genlen $genlen"
    python benchmark/tinyllava_benchmark_generation_speed.py --model-name "$model" --genlen "$genlen"
  done
done