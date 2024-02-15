# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import sys
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from args import AdditionalArguments, DataTrainingArguments
from models.modeling_llama import LlamaForCausalLM
from utils.compresso_utils import load_zs
from models.modeling_llama import LlamaConfig
from models.tokenization_llama import LlamaTokenizer
from models.model_args import ModelArguments


def set_lora_args(config, lora_param):
    config.use_lora = True
    config.lora_rank = 8
    config.lora_train_bias = None
    config.lora_alpha = 8.0
    config.lora_param = lora_param
    config.lora_layers = config.num_hidden_layers
    return config


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, AdditionalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, additional_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, additional_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # model initialize
    config = LlamaConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    )
    config.use_cache = False
    lora_ckpt = None
    if additional_args.pretrained_pruned_model is not None:
        config = set_lora_args(config, model_args.lora_param)
        peft = additional_args.pretrained_pruned_model
        lora_ckpt = os.path.join(peft, 'lora_weights.pt')
        if not os.path.exists(lora_ckpt):
            print('No lora module found, ignored!')
            lora_ckpt = None
            config.lora_param = ''
    # lora_ckpt = None  # no lora
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        padding_side="left",
        truncation_side="left",
    )
    model = LlamaForCausalLM.from_pretrained(
        LlamaForCausalLM,
        model_args.model_name_or_path,
        from_tf=False,
        config=config,
        lora_ckpt = lora_ckpt
    )
    model.half()
    model.eval()

    zs = None
    if additional_args.pretrained_pruned_model is not None:
        zs = load_zs(os.path.join(additional_args.pretrained_pruned_model, 'zs.pt'))
        for key in zs:
            zs[key] = zs[key].detach()

    model = model.to('cuda')

    # text = 'Neural Sequential Model, especially transformers,'
    text = 'AI can create a logo in seconds'
    # text = 'What’s great about the holiday season'

    tokenizer_with_prefix_space = LlamaTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        add_prefix_space=True
    )
    def get_tokens_as_list(word_list):
        "Converts a sequence of words into a list of tokens"
        tokens_list = []
        for word in word_list:
            tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
            tokens_list.append(tokenized_word)
        return tokens_list
    # line break is bad words
    bad_words_ids = get_tokens_as_list(word_list=['ﬁlm'])

    inputs = tokenizer(text, return_tensors="pt").to('cuda')
    if zs is not None:
        model.head_z=zs['head_z'].to('cuda')
        model.intermediate_z=zs['intermediate_z'].to('cuda')
        model.hidden_z=zs['hidden_z'].to('cuda')

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        bad_words_ids=[[13],[242], [29871]],
        no_repeat_ngram_size=4,
    )
    batch_size, length = inputs['input_ids'].shape
    print(outputs[0, length:])
    print(tokenizer.decode(outputs[0, length:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
