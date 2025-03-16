#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
MODEL_PATH=$1
MODEL_NAME=$2
CONV_MODE=$3
TEMP=$4
EVAL_DIR="playground/data/eval" 

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.model_vqa_science \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/scienceqa/llava_test_CQM-A.json \
    --image-folder $EVAL_DIR/scienceqa/images/test \
    --answers-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature $TEMP \
    --conv-mode $CONV_MODE &
done

wait

output_file=$EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl

> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVAL_DIR/scienceqa/answers/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python tinyllava/eval/eval_science_qa.py \
    --base-dir $EVAL_DIR/scienceqa \
    --result-file $EVAL_DIR/scienceqa/answers/$MODEL_NAME.jsonl \
    --output-file $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_output.jsonl \
    --output-result $EVAL_DIR/scienceqa/answers/"$MODEL_NAME"_result.json

