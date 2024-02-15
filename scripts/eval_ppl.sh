for e in epoch6
do
python eval_ppl.py \
    --model_name_or_path baffo32/decapoda-research-llama-7B-hf \
    --output_dir output/ \
    --pretrained_pruned_model output/NutePrune-pruning_only-s25.0-lr-1.0-reglr0.1-warmup1/2024-2-11-11-16/$e \
    --lora_param Q.V
done
