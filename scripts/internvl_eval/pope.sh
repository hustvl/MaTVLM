gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_PATH=$1
MODEL_NAME=$2
CONV_MODE=$3
TEMP=$4
EVAL_DIR="playground/data/eval"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature $TEMP \
    --conv-mode $CONV_MODE &
done

wait

output_file=$EVAL_DIR/pope/answers/$MODEL_NAME.jsonl

> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $EVAL_DIR/pope/answers/$MODEL_NAME/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl

