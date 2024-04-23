pretrained=merged_llama2_13b
tokenizer=NousResearch/Llama-2-13b-hf

bash scripts/benchmarks/mmlu.sh $pretrained $tokenizer
bash scripts/benchmarks/bbh.sh $pretrained $tokenizer
bash scripts/benchmarks/gsm8k.sh $pretrained $tokenizer
