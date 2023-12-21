import torch
import os
import numpy as np

lora = torch.load('output/Compresso-pruning-s50.0-lr3e-05-reglr0.1-warmup0/2023-12-20-9-11/step_1330/lora_weights.pt', map_location="cpu")

for k in lora.keys():
    print(k, lora[k].mean())