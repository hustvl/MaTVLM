#!/bin/bash


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_PATH=$1
MODEL_NAME=$2
CONV_MODE=$3
TEMP=$4
EVAL_DIR="playground/data/eval"
SPLIT="mmbench_dev_20230712"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.model_vqa_mmbench \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/mmbench/$SPLIT.tsv \
    --answers-file $EVAL_DIR/mmbench/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature $TEMP \
    --conv-mode $CONV_MODE &
done

wait

output_file=$EVAL_DIR/mmbench/answers/$SPLIT/$MODEL_NAME.jsonl

> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVAL_DIR/mmbench/answers/$SPLIT/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

mkdir -p $EVAL_DIR/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $EVAL_DIR/mmbench/$SPLIT.tsv \
    --result-dir $EVAL_DIR/mmbench/answers/$SPLIT \
    --upload-dir $EVAL_DIR/mmbench/answers_upload/$SPLIT \
    --experiment $MODEL_NAME
