import time
import os
import subprocess

def get_gpu_memory_usage():
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], encoding="utf-8")
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        return gpu_memory
    except subprocess.CalledProcessError as e:
        print("Error occurred while running nvidia-smi:", e)
        return None

def tostr(can_used_gpus):
    ret = ''
    for x in can_used_gpus:
        ret += str(x) +','
    return ret[:-1]

if __name__ == "__main__":

    loop = True
    while(loop):
        gpu_memory_usage = get_gpu_memory_usage()
        if gpu_memory_usage:
            can_used_gpus = []
            for i, memory in enumerate(gpu_memory_usage):
                if int(memory) < 3000:
                    can_used_gpus.append(i)

            if len(can_used_gpus) == 4:
                # can_used_gpus_str = tostr(can_used_gpus)[0]
                # os.system("ACCELERATE_LOG_LEVEL=info PYTHONPATH=. accelerate launch --main_process_port=7777 --config_file multi_gpu.yaml train_internvl_mamba2/train_hybrid.py mamba2_internvl/internvl2b_0.25_mamba2_665k.yaml 2>&1 | tee -a output_train.txt")
                os.system(f"PYTHONPATH=. bash scripts/internvl_eval/test_all_benchmark.sh output/MaTVLM_0_Mamba2_InternVL_4B MaTVLM_0_Mamba2_InternVL_4B")
                # os.system("CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info PYTHONPATH=. accelerate launch --main_process_port=9999 --config_file multi_gpu.yaml train_tinyllava_mamba2/train_hybrid_v2.py mamba2_tinyllava/tinyllava_0.25_mamba2_tlloss_665k.yaml 2>&1 | tee -a output_train.txt")
                loop = False
            else:
                print('sleeping...')
                time.sleep(60)
                