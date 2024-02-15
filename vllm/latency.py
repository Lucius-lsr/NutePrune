from vllm import LLM, SamplingParams
import time
import torch
import os
import numpy as np
from vllm.model_executor.models.llama import LlamaModel

HEAD_DIM = 4096 // 32

def load_zs(peft):
    if peft == 'llm_pruner':
        hidden_z = torch.from_numpy(np.ones(4096))
        head_z = torch.from_numpy(np.ones((32, 32)))
        intermediate_z = torch.from_numpy(np.ones((32, 11008)))
        # # 50% pruning
        # for i in range(3,30):
        #     head_z[i][:int(32*0.6)] = 0
        #     intermediate_z[i][:int(11008*0.6)] = 0
        # 20% pruning
        for i in range(4,30):
            head_z[i][:int(32*0.25)] = 0
            intermediate_z[i][:int(11008*0.25)] = 0
        
        return hidden_z > 0, head_z > 0, intermediate_z > 0
    elif peft == 'sheared':
        hidden_z = torch.from_numpy(np.ones(4096))
        head_z = torch.from_numpy(np.ones((32, 32)))
        intermediate_z = torch.from_numpy(np.ones((32, 11008)))
        for i in range(32):
            head_z[i][:16] = 0
            intermediate_z[i][:5504] = 0
        return hidden_z > 0, head_z > 0, intermediate_z > 0
    zs = torch.load(os.path.join(peft, 'zs.pt'), map_location="cpu")
    hidden_z = zs['hidden_z'] if 'hidden_z' in zs.keys() else torch.from_numpy(np.ones(4096))
    head_z = zs['head_z']
    intermediate_z = zs['intermediate_z']
    hidden_z = hidden_z.detach() > 0
    print(hidden_z.sum().item())
    # target = 4064
    # hidden_z = torch.from_numpy(np.concatenate([np.ones(target),np.zeros(4096-target)])).detach() > 0
    head_z = head_z.detach() > 0
    intermediate_z = intermediate_z.detach() > 0
    return hidden_z, head_z, intermediate_z

def prune_with_zs(model, hidden_z, head_z, intermediate_z):
    for i, layer in enumerate(model.model.layers):
        # pruned heads
        head_mask = head_z[i].reshape(-1, 1).repeat(1, HEAD_DIM).reshape(-1)
        head_mask_3x = head_mask.reshape(1, -1).repeat(3, 1).reshape(-1)
        layer.self_attn.qkv_proj.weight.data = layer.self_attn.qkv_proj.weight.data[head_mask_3x]
        layer.self_attn.qkv_proj.output_size = head_mask_3x.sum().item()
        layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[:, head_mask]
        layer.self_attn.o_proj.input_size = head_mask.sum().item()
        layer.self_attn.num_heads = head_mask.sum().item() // HEAD_DIM
        layer.self_attn.num_kv_heads = head_mask.sum().item() // HEAD_DIM
        layer.self_attn.q_size = layer.self_attn.num_heads * layer.self_attn.head_dim
        layer.self_attn.kv_size = layer.self_attn.num_kv_heads * layer.self_attn.head_dim
        layer.self_attn.attn.num_heads = head_mask.sum().item() // HEAD_DIM
        layer.self_attn.attn.num_kv_heads = head_mask.sum().item() // HEAD_DIM

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
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, ignore_eos=True, max_tokens=256)

    # Create an LLM.
    llm = LLM(model="baffo32/decapoda-research-llama-7B-hf")
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.

    if peft is not None:
        hidden_z, head_z, intermediate_z = load_zs(peft)
        count_param(llm.llm_engine.workers[0].model)
        prune_with_zs(llm.llm_engine.workers[0].model, hidden_z, head_z, intermediate_z)
        print('after prune')
        count_param(llm.llm_engine.workers[0].model)

    t1 = time.time()
    for _ in range(10):
        outputs = llm.generate(prompts, sampling_params)
    t2 = time.time()
    print("Time taken: ", (t2-t1)/10)

    # # Print the outputs.
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == '__main__':
    pefts = [
        None,
        # 'output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/small_combined_layerdistill_16bs/epoch2',
        # 'output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/small_combined_distill/epoch1',
        # # 'output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/20k_c4_2epoch_supervised/epoch1',
        # 'output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/small_combined_distill_full_hidden/epoch4',
        # 'llm_pruner',
        # 'sheared',
        # 'output/Compresso-pruning_only-s50.0-lr5e-05-reglr0.1-warmup1/uniform/epoch6',
        # 'output/checkpoint/50_warmup1_cotrain_epoch6',
        # 'output/checkpoint/20_warmup4_cotrain_epoch5'
    ]

    test_latency(pefts[-1])