#!/bin/bash
export WANDB_DISABLED=TRUE
export TQDM_DISABLED=true

export OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

python train.py \
    --pruning_type structured_heads+structured_mlp+hidden \
    --target_sparsity 0.5 \
    --sparsity_epsilon 0.005 \
    --model_name_or_path baffo32/decapoda-research-llama-7B-hf \
    --num_train_epochs 1 \
    --learning_rate 3e-5 \
    --reg_learning_rate 0.1 \
    --lagrangian_warmup_epochs 0 \
    --max_seq_length 512 \
    --task_name pruning \
    --do_train \
    --do_eval \
    --dataset_name c4 \
    --overwrite_cache \
    --eval_dataset_name wikitext \
    --train_file ./data/alpaca_gpt4_data.json \
    --droprate_init 0.01 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --training_objective LM \
    --overwrite_output_dir \
    --output_dir $OUTPUT_DIR/ \
    --cache_dir /dev/shm \
    --use_lora True \
    --lora_rank 8 \
    --lora_train_bias none \
    --lora_alpha 8.0 \
    --lora_param Q.V \
    --lora_layers 32 \
    --gradient_checkpointing=True \
    --logging_first_step \
    --logging_steps 10 \
    --disable_tqdm False \
    --fp16 True \
    --random_init=False \
    --do_distill \
