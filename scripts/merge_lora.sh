#!/bin/bash

# You can use phi3 instead of phi3.5
MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path Qwen2-VL-Finetune/output/testing_lora \
    --model-base $MODEL_NAME  \
    --save-model-path Qwen2-VL-Finetune/output/merge_test \
    --safe-serialization