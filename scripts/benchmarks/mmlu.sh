lm_eval --model hf \
    --model_args pretrained=$PRETRAINED,tokenizer=$TOKENIZER \
    --tasks mmlu \
    --num_fewshot 5 \
    --device cuda:0 \
    --batch_size 1
