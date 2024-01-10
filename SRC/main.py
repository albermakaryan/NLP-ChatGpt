import torch
from icecream import ic
import os

model = torch.load("../models/gpt.pt")

from gpt import GPTLanguageModel

from gpt import decode,encode


while True:
    
    context = input("Enter a prompt: ")
    
    if context == "exit":
        break
    
    context = encode(context)
    
    response = model.generate(context)
    
    print(decode(response))
    
    
    
