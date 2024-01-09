# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import os
import sys
import datasets
import transformers
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
import torch

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data.dataset import Dataset

def get_wikitext2(seq_len, tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', ignore_verifications=True)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', ignore_verifications=True)
    return traindata, testdata

def get_ptb(seq_len, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', ignore_verifications=True)
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', ignore_verifications=True)
    return traindata, valdata

class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

def process_data(samples, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)
       

def get_loaders(name, tokenizer, seq_len=2048, batch_size = 8):
    if 'wikitext2' in name:
        train_data, test_data = get_wikitext2(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        train_data, test_data = get_ptb(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_data, test_loader

def PPLMetric(model, zs, tokenizer, datasets, seq_len=2048, batch_size = 4, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        ppl = llama_eval(model, zs, test_loader, device)
        metric[dataset] = ppl
        print(metric)
    return metric

def fill_inputs_with_zs(zs, inputs_id):
    inputs = {}
    inputs['input_ids'] = inputs_id
    for key in zs:
        inputs[key] = zs[key].to(inputs["input_ids"].device)
    return inputs

@torch.no_grad()
def llama_eval(model, zs, test_lodaer, device):
    nlls = []
    n_samples = 0
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        if zs is not None:
            inputs = fill_inputs_with_zs(zs, batch)
        else:
            inputs = {"input_ids": batch}
        output = model(**inputs)
        lm_logits = output.logits
    
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()

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

    ppl = PPLMetric(model, zs, tokenizer, ['wikitext2', 'ptb'])


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    main()
