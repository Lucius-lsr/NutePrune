import torch
import os
import numpy as np

def analyze(peft):
    zs = torch.load(os.path.join(peft, 'zs.pt'), map_location="cpu")

    hidden_z = zs['hidden_z'] if 'hidden_z' in zs.keys() else torch.from_numpy(np.ones(4096))
    head_z = zs['head_z']
    intermediate_z = zs['intermediate_z']

    # detail
    num_hidden_z = (hidden_z > 0).sum()
    print(f'num_hidden_z: {num_hidden_z}')
    # print('num_head_z: ')
    # for h_z in head_z.squeeze():
    #     # print(h_z)
    #     num_head_z = (h_z > 0).sum().item()
    #     print(num_head_z, end=' ')
    # print()
    # for h_z in head_z:
    #     ave_head_z = h_z[h_z > 0].sum().item()
    #     print(ave_head_z, end=' ')
    # print()
    print('num_intermediate_z: ')
    for i_z in intermediate_z:
        num_intermediate_z = (i_z > 0).sum().item()
        print(num_intermediate_z, end=' ')
    print()

    # sparsity
    hidden_z = hidden_z.detach().numpy() > 0

    # head_z = head_z.detach().numpy() > 0
    head_z = head_z.reshape(32, 32, 32)
    selected_q, idx = head_z.max(dim=-1)
    # for h in head_z[-1]:
    #     print(h)
    # for i in range(32):
    #     print('layer ', i)
    #     print(selected_q[i])
    #     print(idx[i])
    masked_tensor = torch.zeros_like(head_z, device=head_z.device)
    masked_tensor.scatter_(-1, idx.unsqueeze(-1), selected_q.unsqueeze(-1))
    selected_kv = masked_tensor.max(dim=-2)[0]
    q = selected_q.squeeze().detach().cpu().numpy() > 0
    kv = selected_kv.squeeze().detach().cpu().numpy() > 0
    print('head q:', q.sum(-1))
    print('head kv:', kv.sum(-1))
    head_z = (q.astype(np.float32) + kv.astype(np.float32))/2

    intermediate_z = intermediate_z.detach().numpy() > 0
    head_nums = np.outer(head_z.reshape(-1), hidden_z).sum().item() # 
    intermediate_nums = np.outer(intermediate_z.reshape(-1), hidden_z).sum().item()
    remaining_model_size = head_nums * (4096 // 32) * 4 + intermediate_nums * 3
    MODEL_SIZE = 6476005376
    return (MODEL_SIZE-remaining_model_size)/MODEL_SIZE

experiments = 'output/Compresso-pruning_only-s50.0-lr5e-05-reglr0.1-warmup1/mqa_0.99'
checkpoints = os.listdir(experiments)
for checkpoint in checkpoints:
    if checkpoint.startswith('epoch'):
        peft = os.path.join(experiments, checkpoint)
        print(checkpoint, analyze(peft))

print(analyze('output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/small_combined_layerdistill_16bs/epoch2'))
