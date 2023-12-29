#! /usr/bin/env bash

set -ex

LR=5e-4
NUM_GPUS=1
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.2
WARMUP_STEPS=50

MAX_SOURCE_LEN=256
MAX_TARGET_LEN=1024
DEV_BATCH_SIZE=2
GRAD_ACCUMULARION_STEPS=8
MAX_STEP=5000
SAVE_INTERVAL=200

RUN_NAME=lora-lyrics
BASE_MODEL_PATH=THUDM/chatglm3-6b-base
DATASET_PATH=formatted_data/train_data.jsonl
DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${LR}
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
    --train_format input-output \
    --train_file $DATASET_PATH \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --warmup_steps $WARMUP_STEPS \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR  2>&1 | tee ${OUTPUT_DIR}/train.log 

