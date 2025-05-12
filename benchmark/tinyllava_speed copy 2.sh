#!/bin/bash

# Define the models
models=(
  # "meta-llama/Llama-3.2-3B-Instruct"
  # "JunxiongWang/Llama3.2-Mamba2-3B-distill"
  # "output/TinyLLaVA-Phi-2-SigLIP-3.1B_0.25_mamba2_llava_v1_5_mix665k_gt_kl0.1"
  # "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"
  "output/TinyLLaVA-Phi-2-SigLIP-3.1B_0.25_mamba_sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k_gt_kl1_ce1_tl1_l2_ep1_tunemlp_adamw_newcode_wd0.01_nodecay_fixbetas_wsd_beta20.95_lr2e-4"
)

# Define the generation lengths
gen_lens=(8192 16384 32768 65536)
# gen_lens=(1024 16384 32768 65536 1024)


# Loop through models and generation lengths
for genlen in "${gen_lens[@]}"; do
  for model in "${models[@]}"; do
    # echo "Running benchmark for model $model with genlen $genlen"
    python benchmark/tinyllava_benchmark_generation_speed.py --model-name "$model" --genlen "$genlen"
  done
done