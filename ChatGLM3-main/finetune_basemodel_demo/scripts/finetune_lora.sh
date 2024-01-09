#! /usr/bin/env bash

set -ex

LR=1e-4
NUM_GPUS=1
LORA_RANK=8
LORA_ALPHA=32
LORA_DROUPOUT=0.1

MAX_SOURCE_LEN=512
MAX_TARGET_LEN=128
DEV_BATCH_SIZE=1
GRAD_ACCUMULARION_STEPS=8
MAX_STEP=10000
SAVE_INTERVAL=500

RUN_NAME=text
BASE_MODEL_PATH=THUDM/chatglm3-6b-base
TRAINSET_PATH=data/train_data.json
VALSET_PATH=data/val_data.json
DATESTR=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${LR}
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

### modify eval parameters by Xin
EVAL_STRATEGY=steps
EVAL_STEPS=10
EVAL_BATCH_SIZE=5


mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
    --train_format input-output \
    --train_file $TRAINSET_PATH \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROUPOUT \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --evaluation_strategy $EVAL_STRATEGY \
    --val_file $VALSET_PATH \
    --eval_steps $EVAL_STEPS \
    --per_device_eval_batch_size $EVAL_BATCH_SIZE \
    --logging_steps 1 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR  2>&1 | tee ${OUTPUT_DIR}/train.log

