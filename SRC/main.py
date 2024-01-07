import torch
from icecream import ic
import os

model = torch.load("../models/gpt.pt")

print(model.parameters)