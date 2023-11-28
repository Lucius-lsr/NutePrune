export PYTHONPATH='.'

base_model=baffo32/decapoda-research-llama-7B-hf
prompt_mark=0 # 0: do not add pruning prompt during evaluation; 1: add the pruning prompt same as training; 2. add the pruning prompt for evaluation

file_name=baseline

# python ./evaluation/leh_llmpruner/main.py \
python ./evaluation/lm-evaluation-harness/main.py \
    --model hf-causal-experimental \
    --model_args pretrained=$base_model \
    --tasks winogrande \
    --device cuda:0 \
    --output_path results/Commonsense_${file_name}_$prompt_mark.json \
    --no_cache

    # --model compresso \
    # --model_args pretrained=$base_model,prompt_mark=$prompt_mark \
    # --tasks boolq,openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa \

# python ./evaluation/lm-evaluation-harness/generate.py results/Commonsense_${file_name}_$prompt_mark.json
