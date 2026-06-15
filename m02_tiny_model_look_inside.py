# m02_tiny_model_look_inside.py

import torch

data = torch.load("m01_tiny_model.pt")

print(type(data))
print()

for name, tensor in data.items():
    print(name)
    print(tensor)
    print()
    
print("-------------------")
print(data)