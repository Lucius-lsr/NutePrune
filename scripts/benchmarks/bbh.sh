PRETRAINED=$1
TOKENIZER=$2

lm_eval --model hf \
    --model_args pretrained=$PRETRAINED,tokenizer=$TOKENIZER \
    --tasks bbh_fewshot \
    --num_fewshot 3 \
    --device cuda:0 \
    --batch_size 1
