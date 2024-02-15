export PYTHONPATH='.'

base_model=baffo32/decapoda-research-llama-7B-hf
pretrained_dir=output/NutePrune-distill-s20.0-lr0.001-reglr0.1-warmup4/2024-2-13-18-14
lora_param=Q.V. # the lora param in training

python wait.py
for e in epoch5 epoch6
do
    pretrained_path=$pretrained_dir/$e
    file_name=$(echo $pretrained_path | cut -d'/' -f $(($(echo $pretrained_path | tr '/' '\n' | wc -l) - 2)))
    echo $pretrained_path
    python ./evaluation/lm-evaluation-harness/main.py \
        --model nuteprune \
        --model_args pretrained=$base_model,peft=$pretrained_path,lora_param=$lora_param \
        --tasks boolq,openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa \
        --device cuda:0 \
        --output_path results/Commonsense_${file_name}.json \
        --no_cache
done
