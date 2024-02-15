#!/bin/bash
export WANDB_DISABLED=TRUE
export TQDM_DISABLED=true

export OUTPUT_DIR=output
mkdir -p $OUTPUT_DIR

# baseline_pruned_model=output/Compresso-large_pruning_only-s50.0-lr-1.0-reglr0.15-warmup2/iter_layerdis/epoch0
baseline_pruned_model=output/Compresso-pruning_only-s50.0-lr5e-05-reglr0.1-warmup1/layerdis_iter_dense_dual_7/epoch6

python train.py \
    --pruning_type None \
    --model_name_or_path baffo32/decapoda-research-llama-7B-hf \
    --pretrained_pruned_model $baseline_pruned_model \
    --num_train_epochs 1 \
    --learning_rate 1e-3 \
    --scheduler_type None \
    --reg_learning_rate -1 \
    --lagrangian_warmup_epochs 0 \
    --max_seq_length 512 \
    --task_name large_c4_finetune \
    --do_train \
    --do_eval \
    --dataset_name c4 \
    --overwrite_cache True \
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
    --lora_param Q.V \
    --lora_layers 32 \
    --gradient_checkpointing=True \
    --logging_first_step \
    --logging_steps 10 \
    --disable_tqdm False \
    --fp16 True \
    --random_init=False \
    --do_distill \
    --do_layer_distill \