#!/bin/bash
export WANDB_DISABLED=TRUE
export TQDM_DISABLED=true

export OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

baseline_pruned_model=output/checkpoint/20_13b_warmup4_cotrain_epoch5

python train.py \
    --pruning_type None \
    --model_name_or_path huggyllama/llama-13b \
    --pretrained_pruned_model $baseline_pruned_model \
    --num_train_epochs 2 \
    --learning_rate 1e-4 \
    --reg_learning_rate 0.1 \
    --lagrangian_warmup_epochs 0 \
    --max_seq_length 512 \
    --task_name finetune \
    --do_train \
    --do_eval \
    --dataset_name alpaca \
    --overwrite_cache False \
    --eval_dataset_name wikitext \
    --train_file ./data/alpaca_data_cleaned.json \
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
    --lora_param Q.K.V.O.F \
    --lora_layers 32 \
    --gradient_checkpointing=True \
    --logging_first_step \
    --logging_steps 10 \
    --disable_tqdm False \
    --fp16 True \
    --random_init=False \