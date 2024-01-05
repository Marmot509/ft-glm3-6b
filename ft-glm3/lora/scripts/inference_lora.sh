#! /usr/bin/env bash

set -ex

LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
MAX_TOKENS=512
LORA_PATH=output/lora-lyrics-20240103-072747-1e-4/checkpoint-1600/pytorch_model.bin
MODEL=THUDM/chatglm3-6b-base
TOKENIZER=THUDM/chatglm3-6b-base




python inference.py \
    --lora-rank $LORA_RANK \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT \
    --lora-path $LORA_PATH \
    --max-new-tokens $MAX_TOKENS \
    --model $MODEL \
    --tokenizer $TOKENIZER

