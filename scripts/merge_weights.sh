#!/bin/bash
export PYTHONPATH='.'

export OUTPUT_DIR=merged_llama2_13b
mkdir -p $OUTPUT_DIR

base_model=NousResearch/Llama-2-13b-hf
pretrained_path=output/NutePrune-finetune_llama2_13b-s0-lr0.0001-reglr0.1-warmup0/2024-4-22-4-16/step_5500
lora_param=Q.K.V.O.F

python ./merge_weights.py \
  --pruning_type None \
  --model_name_or_path $base_model \
  --pretrained_pruned_model $pretrained_path \
  --output_dir $OUTPUT_DIR/ \
  --cache_dir /dev/shm/ \
  --use_lora True \
  --lora_rank 8 \
  --lora_train_bias none \
  --lora_alpha 8.0 \
  --lora_param $lora_param \
  --lora_layers -1 \
