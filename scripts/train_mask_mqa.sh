#!/bin/bash
export WANDB_DISABLED=TRUE
export TQDM_DISABLED=true

export OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

python train.py \
    --pruning_type structured_qkv+structured_mlp \
    --target_sparsity 0.5 \
    --sparsity_epsilon 0.005 \
    --model_name_or_path baffo32/decapoda-research-llama-7B-hf \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --reg_learning_rate 0.1 \
    --lagrangian_warmup_epochs 1 \
    --max_seq_length 512 \
    --task_name pruning_only \
    --do_train \
    --do_eval \
    --dataset_name c4 \
    --eval_dataset_name wikitext \
    --train_file ./data/alpaca_gpt4_data.json \
    --droprate_init 0.01 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --training_objective LM \
    --overwrite_output_dir \
    --output_dir $OUTPUT_DIR/ \
    --cache_dir /dev/shm \
    --use_lora False \
    --gradient_checkpointing=True \
    --logging_first_step \
    --logging_steps 10 \
    --disable_tqdm False \
    --fp16 True \
    --random_init=False \
    --do_distill \
    --do_layer_distill \
