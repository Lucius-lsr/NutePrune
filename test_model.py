from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("princeton-nlp/Sheared-LLaMA-1.3B")

print(model)