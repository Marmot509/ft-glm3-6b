#! /usr/bin/env bash

set -ex

LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
MAX_TOKENS=1024
LORA_PATH=output/lora-lyrics-20231229-062839-1e-4/checkpoint-200/


python inference.py \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_path $LORA_PATH \
    --max-new-tokens $MAX_TOKENS 

