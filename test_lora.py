import torch
import os
import numpy as np

lora = torch.load('output/NutePrune-alternative-s50.0-lr5e-06-reglr0.2-warmup1/2024-1-10-3-4/epoch0/lora_weights.pt', map_location="cpu")

for k in lora.keys():
    print(k, lora[k].mean())