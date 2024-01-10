import torch
from icecream import ic
import os

from gpt import *

model = torch.load(model_path)



while True:
    context = input("\nEnter a context: ")
    
    if context == 'quit':
        break
    context = encode(context).reshape(-1,1)
    
    context = context.to(device)
    
    response = m.generate(context,300)[0].tolist()
    
    
    # print([index_to_char[i] for i in response])
    response = "".join(decode(response))
    print(response)
    
