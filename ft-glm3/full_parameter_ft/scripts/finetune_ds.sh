#! /usr/bin/env bash

set -ex

LR=1e-4
NUM_GPUS=1
MAX_SOURCE_LEN=128
MAX_TARGET_LEN=256
DEV_BATCH_SIZE=8
GRAD_ACCUMULARION_STEPS=1
NUM_Epochs=2
SAVE_INTERVAL=500

RUN_NAME=fpft_lyrics
BASE_MODEL_PATH=THUDM/chatglm3-6b
TRAINSET_PATH=formatted_data/train_data.jsonl
VALSET_PATH=formatted_data/mini_val_data.jsonl

DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${LR}
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

### modify eval parameters by Xin
EVAL_STRATEGY=steps
EVAL_STEPS=20
EVAL_BATCH_SIZE=5


mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
    --train_format input-output \
    --train_file $DATASET_PATH \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR \
    --fp16 \
    --deepspeed configs/deepspeed.json 2>&1 | tee ${OUTPUT_DIR}/train.log
