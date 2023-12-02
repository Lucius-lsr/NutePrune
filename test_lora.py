import torch
import os
import numpy as np

lora = torch.load('output/Compresso-finetune-s50.0-lr5e-05-reglr0.1-warmup0/2023-12-1-11-48/epoch0/lora_weights.pt', map_location="cpu")

print(lora)