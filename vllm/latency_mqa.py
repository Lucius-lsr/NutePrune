from vllm import LLM, SamplingParams
import time
import torch
import os
import numpy as np
from vllm.model_executor.models.llama import LlamaModel

HEAD_DIM = 4096 // 32

def load_zs(peft):
    if peft == '16q8kv':
        hidden_z = torch.from_numpy(np.ones(4096))
        q_z = torch.from_numpy(np.ones((32, 32)))
        kv_z = torch.from_numpy(np.ones((32, 32)))
        intermediate_z = torch.from_numpy(np.ones((32, 11008)))
        for i in range(32):
            q_z[i][:16] = 0
            kv_z[i][:24] = 0
        return hidden_z > 0, q_z > 0, kv_z > 0, intermediate_z > 0
    if peft == '12heads':
        hidden_z = torch.from_numpy(np.ones(4096))
        head_z = torch.from_numpy(np.ones((32, 32)))
        intermediate_z = torch.from_numpy(np.ones((32, 11008)))
        for i in range(32):
            head_z[i][:20] = 0
        return hidden_z > 0, head_z > 0, head_z > 0, intermediate_z > 0
    if peft == '32q16kv':
        hidden_z = torch.from_numpy(np.ones(4096))
        q_z = torch.from_numpy(np.ones((32, 32)))
        kv_z = torch.from_numpy(np.ones((32, 32)))
        intermediate_z = torch.from_numpy(np.ones((32, 11008)))
        for i in range(32):
            kv_z[i][:16] = 0
        return hidden_z > 0, q_z > 0, kv_z > 0, intermediate_z > 0
    if peft == '24heads':
        hidden_z = torch.from_numpy(np.ones(4096))
        head_z = torch.from_numpy(np.ones((32, 32)))
        intermediate_z = torch.from_numpy(np.ones((32, 11008)))
        for i in range(32):
            head_z[i][:8] = 0
        return hidden_z > 0, head_z > 0, head_z > 0, intermediate_z > 0

def prune_with_zs(model, hidden_z, q_z, kv_z, intermediate_z):
    for i, layer in enumerate(model.model.layers):
        # pruned heads
        q_mask = q_z[i].reshape(-1, 1).repeat(1, HEAD_DIM).reshape(-1)
        kv_mask = kv_z[i].reshape(-1, 1).repeat(1, HEAD_DIM).reshape(-1)
        head_mask = torch.cat([q_mask, kv_mask, kv_mask], dim=0)
        layer.self_attn.qkv_proj.weight.data = layer.self_attn.qkv_proj.weight.data[head_mask]
        layer.self_attn.qkv_proj.output_size = head_mask.sum().item()
        layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[:, q_mask]
        layer.self_attn.o_proj.input_size = q_mask.sum().item()
        layer.self_attn.num_heads = q_mask.sum().item() // HEAD_DIM
        layer.self_attn.num_kv_heads = kv_mask.sum().item() // HEAD_DIM
        layer.self_attn.q_size = layer.self_attn.num_heads * layer.self_attn.head_dim
        layer.self_attn.kv_size = layer.self_attn.num_kv_heads * layer.self_attn.head_dim
        layer.self_attn.attn.num_heads = q_mask.sum().item() // HEAD_DIM
        layer.self_attn.attn.num_kv_heads = kv_mask.sum().item() // HEAD_DIM
        assert layer.self_attn.attn.num_heads % layer.self_attn.attn.num_kv_heads == 0
        layer.self_attn.attn.num_queries_per_kv = layer.self_attn.attn.num_heads // layer.self_attn.attn.num_kv_heads
        layer.self_attn.attn.head_mapping = torch.repeat_interleave(
            torch.arange(layer.self_attn.attn.num_kv_heads, dtype=layer.self_attn.attn.head_mapping.dtype, device="cuda"),
            layer.self_attn.attn.num_queries_per_kv)
        
        # pruned intermediate
        intermediate_mask = intermediate_z[i].reshape(-1)
        intermediate_mask_2x = intermediate_mask.reshape(1, -1).repeat(2, 1).reshape(-1)
        layer.mlp.gate_up_proj.weight.data = layer.mlp.gate_up_proj.weight.data[intermediate_mask_2x]
        layer.mlp.gate_up_proj.output_size = intermediate_mask_2x.sum().item()
        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:, intermediate_mask]
        layer.mlp.down_proj.in_size = intermediate_mask.sum().item()

        # pruned hidden
        hidden_mask = hidden_z.reshape(-1)
        layer.mlp.gate_up_proj.weight.data = layer.mlp.gate_up_proj.weight.data[:, hidden_mask]
        layer.mlp.gate_up_proj.input_size = hidden_mask.sum().item()
        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[hidden_mask]
        layer.mlp.down_proj.output_size = hidden_mask.sum().item()
        layer.self_attn.qkv_proj.weight.data = layer.self_attn.qkv_proj.weight.data[:, hidden_mask]
        layer.self_attn.qkv_proj.input_size = hidden_mask.sum().item()
        layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[hidden_mask]
        layer.self_attn.o_proj.output_size = hidden_mask.sum().item()

        layer.input_layernorm.weight.data = layer.input_layernorm.weight.data[hidden_mask]
        layer.post_attention_layernorm.weight.data = layer.post_attention_layernorm.weight.data[hidden_mask]
        layer.self_attn.hidden_size = hidden_mask.sum().item()
    model.model.embed_tokens.weight.data = model.model.embed_tokens.weight.data[:, hidden_mask]
    model.model.embed_tokens.embedding_dim = hidden_mask.sum().item()
    model.lm_head.weight.data = model.lm_head.weight.data[:, hidden_mask]
    model.lm_head.in_features = hidden_mask.sum().item()
    model.model.norm.weight.data = model.model.norm.weight.data[hidden_mask]

def count_param(model):
    param_count = 0
    for name, param in model.named_parameters():
        param_count += param.numel()
        # print(name, param.numel())
    print(f'param count: {param_count}')

def test_latency(peft):
    # Sample prompts.
    prompts = [
        '''Earn monthly interest on our Citibank Time Deposits (also known as Fixed Deposits). What's more, you can get to enjoy the flexibility of making partial withdrawals before maturity date of your Time Deposit. Partial withdrawals in multiples of the'''
    ]
    # prompts *= 4
    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]
    # prompts = [
    #     "Hello, my name is",
    # ]

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, ignore_eos=True, max_tokens=1024)

    # Create an LLM.
    llm = LLM(model="baffo32/decapoda-research-llama-7B-hf")
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.

    if peft is not None:
        hidden_z, q_z, kv_z, intermediate_z = load_zs(peft)
        count_param(llm.llm_engine.workers[0].model)
        prune_with_zs(llm.llm_engine.workers[0].model, hidden_z, q_z, kv_z, intermediate_z)
        print('after prune')
        count_param(llm.llm_engine.workers[0].model)

    t1 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    t2 = time.time()
    print("Time taken: ", t2-t1)

    # # Print the outputs.
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    pefts = [
        '16q8kv',
        '12heads',
        '32q16kv',
        '24heads',
    ]

    test_latency(pefts[3])
    # for peft in pefts:
    #     print(f'peft: {peft}')
    #     test_latency(peft)