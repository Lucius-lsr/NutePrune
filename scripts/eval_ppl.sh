for e in epoch2 epoch3 epoch4 epoch5 epoch6
do
python eval_ppl.py \
    --model_name_or_path baffo32/decapoda-research-llama-7B-hf \
    --output_dir output/ \
    --pretrained_pruned_model output/Compresso-cotrain-s20.0-lr1e-05-reglr0.1-warmup1/iter_layerdis/$e \
    --lora_param Q.V
done
