#!/bin/bash

# Define the models
models=(
  "meta-llama/Llama-3.2-3B-Instruct"
  "JunxiongWang/Llama3.2-Mamba2-3B-distill"
  # "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"
  # "output/TinyLLaVA-Phi-2-SigLIP-3.1B_0.25_mamba2_llava_v1_5_mix665k_gt_kl0.1"
)

# Define the generation lengths
gen_lens=(1024 2048 4096 8192 16384 32768 65536)

# Loop through models and generation lengths
for model in "${models[@]}"; do
  for genlen in "${gen_lens[@]}"; do
    # echo "Running benchmark for model $model with genlen $genlen"
    python benchmark/benchmark_generation_speed.py --model-name "$model" --genlen "$genlen"
  done
done