python eval_generate.py \
    --model_name_or_path baffo32/decapoda-research-llama-7B-hf \
    --output_dir output/ \
    --pretrained_pruned_model output/checkpoint/20_warmup4_mask_epoch5 \
    --lora_param Q.V
