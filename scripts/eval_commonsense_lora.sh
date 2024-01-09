export PYTHONPATH='.'

base_model=baffo32/decapoda-research-llama-7B-hf
pretrained_dir=output/Compresso-alternative-s50.0-lr5e-06-reglr0.2-warmup1/iter_layerdis_alter_50_50
prompt_mark=0 # 0: do not add pruning prompt during evaluation; 1: add the pruning prompt same as training; 2. add the pruning prompt for evaluation
lora_param=Q.V # the lora param in training

for e in epoch4 epoch5 epoch6
do
    pretrained_path=$pretrained_dir/$e
    file_name=$(echo $pretrained_path | cut -d'/' -f $(($(echo $pretrained_path | tr '/' '\n' | wc -l) - 2)))
    echo $pretrained_path
    python ./evaluation/lm-evaluation-harness/main.py \
        --model compresso \
        --model_args pretrained=$base_model,peft=$pretrained_path,prompt_mark=$prompt_mark,lora_param=$lora_param \
        --tasks boolq,openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa \
        --device cuda:0 \
        --output_path results/Commonsense_${file_name}_$prompt_mark.json \
        --no_cache
done
