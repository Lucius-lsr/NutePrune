import torch
import os
import numpy as np

lora = torch.load('output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/2023-11-28-19-1/epoch2/lora_weights.pt', map_location="cpu")

print(lora)