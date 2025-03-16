# Train the MaTVLM based on the TinyLLaVA

## Training scripts
```
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info PYTHONPATH=. accelerate launch --main_process_port=9999 --config_file multi_gpu.yaml train_tinyllava_mamba2/train_hybrid.py mamba2_tinyllava/tinyllava_0.25_mamba2_tlloss_665k.yaml 2>&1 | tee -a output_train.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info PYTHONPATH=. accelerate launch --main_process_port=9999 --config_file multi_gpu.yaml train_tinyllava_mamba2/train_hybrid.py mamba2_tinyllava/tinyllava_0.125_mamba2_tlloss_665k.yaml 2>&1 | tee -a output_train.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info PYTHONPATH=. accelerate launch --main_process_port=9999 --config_file multi_gpu.yaml train_tinyllava_mamba2/train_hybrid.py mamba2_tinyllava/tinyllava_0.50_mamba2_tlloss_665k.yaml 2>&1 | tee -a output_train.txt


```

## Evaluation Scripts
```
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. bash scripts/eval/test_all_benchmark.sh /data/yingyueli/MambaInLlama/output/MaTVLM_0_25_Mamba2 MaTVLM_0_25_Mamba2 phi 0
```