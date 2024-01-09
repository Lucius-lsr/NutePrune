for e in epoch0 epoch1 epoch2
do
python eval_ppl.py \
    --model_name_or_path baffo32/decapoda-research-llama-7B-hf \
    --output_dir output/ \
    --pretrained_pruned_model output/Compresso-finetune-s0-lr1e-05-reglr0.1-warmup0/compresso_20/$e \
    --lora_param Q.V
done
