CKPT_PATH=$1
CKPT_NAME=$2
CONV_MODE=$3
TEMP=$4

EVAL_DIR="playground/data/eval"
mkdir -p log_results

CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/mme.sh ${CKPT_PATH} ${CKPT_NAME} ${CONV_MODE} ${TEMP} 2>&1 | tee log_results/${CKPT_NAME}_mme
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/mmmu.sh ${CKPT_PATH} ${CKPT_NAME} ${CONV_MODE} ${TEMP} 2>&1 | tee log_results/${CKPT_NAME}_mmmu
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/pope.sh ${CKPT_PATH} ${CKPT_NAME} ${CONV_MODE} ${TEMP} 2>&1 | tee log_results/${CKPT_NAME}_pope
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/mmvet.sh ${CKPT_PATH} ${CKPT_NAME} ${CONV_MODE} ${TEMP} 2>&1 | tee log_results/${CKPT_NAME}_mmvet
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/textvqa.sh ${CKPT_PATH} ${CKPT_NAME} ${CONV_MODE} ${TEMP}  2>&1 | tee log_results/${CKPT_NAME}_textvqa
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/mmbench.sh ${CKPT_PATH} ${CKPT_NAME} ${CONV_MODE} ${TEMP}  2>&1 | tee log_results/${CKPT_NAME}_mmbench
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/sqa.sh ${CKPT_PATH} ${CKPT_NAME} ${CONV_MODE} ${TEMP}  2>&1 | tee log_results/${CKPT_NAME}_sqa

CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/gqa.sh ${CKPT_PATH} ${CKPT_NAME} ${CONV_MODE} ${TEMP}  2>&1 | tee log_results/${CKPT_NAME}_gqa
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/vqav2.sh ${CKPT_PATH} ${CKPT_NAME} ${CONV_MODE}  2>&1 | tee log_results/${CKPT_NAME}_vqav2