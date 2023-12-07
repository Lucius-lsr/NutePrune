import torch
import os
import numpy as np

lora = torch.load('output/Compresso-pruning_only-s50.0-lr5e-05-reglr0.1-warmup1/2023-12-7-4-23/epoch0/lora_weights.pt', map_location="cpu")

print(lora)