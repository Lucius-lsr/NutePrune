for e in epoch3 epoch4 epoch5 epoch6
do
python eval_ppl.py \
    --model_name_or_path baffo32/decapoda-research-llama-7B-hf \
    --output_dir output/ \
    --pretrained_pruned_model output/Compresso-cotrain-s50.0-lr5e-05-reglr0.2-warmup1/iter_layerdis/$e \
    --lora_param Q.V
done
