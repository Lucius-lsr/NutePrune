import torch
import os
import numpy as np
from transformers import LlamaTokenizer, LlamaForCausalLM
import time
import copy
from test_latency_masked import model_latency

HEAD_DIM = 4096 // 32

def load_zs(peft):
    if peft == 'llm_pruner':
        hidden_z = torch.from_numpy(np.ones(4096))
        head_z = torch.from_numpy(np.ones((32, 32)))
        intermediate_z = torch.from_numpy(np.ones((32, 11008)))
        for i in range(3,30):
            head_z[i][:int(32*0.6)] = 0
            intermediate_z[i][:int(11008*0.6)] = 0
        return hidden_z > 0, head_z > 0, intermediate_z > 0
    zs = torch.load(os.path.join(peft, 'zs.pt'), map_location="cpu")
    hidden_z = zs['hidden_z'] if 'hidden_z' in zs.keys() else torch.from_numpy(np.ones(4096))
    head_z = zs['head_z']
    intermediate_z = zs['intermediate_z']
    hidden_z = hidden_z.detach() > 0
    head_z = head_z.detach() > 0
    intermediate_z = intermediate_z.detach() > 0
    return hidden_z, head_z, intermediate_z


def load_model(base_model):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
    )
    return model.half().to('cuda'), tokenizer

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.numel()
    # print(f'param count: {param_count}')

def prune_with_zs(model, hidden_z, head_z, intermediate_z):
    for i, layer in enumerate(model.model.layers):
        # pruned heads
        head_mask = head_z[i].reshape(-1, 1).repeat(1, HEAD_DIM).reshape(-1)
        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[head_mask]
        layer.self_attn.q_proj.out_features = head_mask.sum().item()
        layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[head_mask]
        layer.self_attn.k_proj.out_features = head_mask.sum().item()
        layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[head_mask]
        layer.self_attn.v_proj.out_features = head_mask.sum().item()
        layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[:, head_mask]
        layer.self_attn.o_proj.in_features = head_mask.sum().item()
        layer.self_attn.num_heads = head_mask.sum().item() // HEAD_DIM
        layer.self_attn.num_key_value_heads = head_mask.sum().item() // HEAD_DIM
        
        # pruned intermediate
        intermediate_mask = intermediate_z[i].reshape(-1)
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[intermediate_mask]
        layer.mlp.gate_proj.out_features = intermediate_mask.sum().item()
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[intermediate_mask]
        layer.mlp.up_proj.out_features = intermediate_mask.sum().item()
        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:, intermediate_mask]
        layer.mlp.down_proj.in_features = intermediate_mask.sum().item()

        # pruned hidden
        hidden_mask = hidden_z.reshape(-1)
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[:, hidden_mask]
        layer.mlp.gate_proj.in_features = hidden_mask.sum().item()
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[:, hidden_mask]
        layer.mlp.up_proj.in_features = hidden_mask.sum().item()
        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[hidden_mask]
        layer.mlp.down_proj.out_features = hidden_mask.sum().item()
        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[:, hidden_mask]
        layer.self_attn.q_proj.in_features = hidden_mask.sum().item()
        layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[:, hidden_mask]
        layer.self_attn.k_proj.in_features = hidden_mask.sum().item()
        layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[:, hidden_mask]
        layer.self_attn.v_proj.in_features = hidden_mask.sum().item()
        layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[hidden_mask]
        layer.self_attn.o_proj.out_features = hidden_mask.sum().item()
        layer.input_layernorm.weight.data = layer.input_layernorm.weight.data[hidden_mask]
        layer.post_attention_layernorm.weight.data = layer.post_attention_layernorm.weight.data[hidden_mask]
        layer.self_attn.hidden_size = hidden_mask.sum().item()
    model.model.embed_tokens.weight.data = model.model.embed_tokens.weight.data[:, hidden_mask]
    model.model.embed_tokens.embedding_dim = hidden_mask.sum().item()
    model.lm_head.weight.data = model.lm_head.weight.data[:, hidden_mask]
    model.lm_head.in_features = hidden_mask.sum().item()
    model.model.norm.weight.data = model.model.norm.weight.data[hidden_mask]

def model_sparsity(hidden_z, head_z, intermediate_z):
    head_nums = np.outer(head_z.reshape(-1), hidden_z).sum().item() # 
    intermediate_nums = np.outer(intermediate_z.reshape(-1), hidden_z).sum().item()
    remaining_model_size = head_nums * (4096 // 32) * 4 + intermediate_nums * 3

    MODEL_SIZE = 6476005376
    # print('sparsity:', (MODEL_SIZE-remaining_model_size)/MODEL_SIZE)


if __name__ == '__main__':
    pefts = [
        'output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/small_combined_layerdistill_16bs/epoch2',
        # 'output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/small_combined_distill/epoch1',
        # 'output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/20k_c4_2epoch_supervised/epoch1',
        # 'output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/small_combined_distill_full_hidden/epoch4',
        'llm_pruner'
    ]
    MODEL_NAME = 'baffo32/decapoda-research-llama-7B-hf'
    full_model, tokenizer = load_model(MODEL_NAME)

    for size in [1024]:
        print(f'size: {size}')
        model_latency(full_model, tokenizer, size)
        for peft in pefts:
            hidden_z, head_z, intermediate_z = load_zs(peft)
            model_sparsity(hidden_z, head_z, intermediate_z)
            model = copy.deepcopy(full_model)
            prune_with_zs(model, hidden_z, head_z, intermediate_z)
            model_latency(model, tokenizer, size)


