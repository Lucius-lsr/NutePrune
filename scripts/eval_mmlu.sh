export PYTHONPATH='.'

base_model=huggyllama/llama-13b
python ./evaluation/instruct-eval/main.py \
    --model_name llama \
    --model_path $base_model \
    --task_name mmlu \
    --device cuda:0 \
    --output_path results/MMLU_${file_name}.json \
    --no_cache
