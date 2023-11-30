import torch
import os
import numpy as np

lora = torch.load('output/Compresso-pruning-s50.0-lr5e-05-reglr0.1-warmup1/2023-11-30-3-31/epoch0/lora_weights.pt', map_location="cpu")

print(lora)