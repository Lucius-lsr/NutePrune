import os
import torch

peft = 'output/Compresso-pruning_only-s50.0-lr5e-05-reglr0.1-warmup1/init_droprate_0.95/epoch4'

l0 = torch.load(os.path.join(peft, 'l0_module.pt'), map_location="cpu")
zs = l0.forward(training=False)

idx = -2
loga = l0.z_logas['head'][idx]
print(loga.shape)
print(loga.squeeze())

expected = 1 - l0.cdf_qz(0, loga)
print(expected.shape)
print(expected.squeeze())

# zs_true = torch.load(os.path.join(peft, 'zs.pt'), map_location="cpu")
print(zs['head_z'][idx].shape)
print(zs['head_z'][idx].squeeze())

print((zs['head_z'][idx].squeeze()>0).sum(-1))
print(zs['head_z'][idx].squeeze().max(-1)[0])

print(l0.get_num_parameters_and_constraint())
print(l0.calculate_model_size(zs))


