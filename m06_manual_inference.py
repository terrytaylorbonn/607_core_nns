# m06_manual_inference.py

import torch
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

model = TinyModel()
model.load_state_dict(torch.load("m01_tiny_model.pt"))

x = torch.tensor([[1.0, 2.0]])

# PyTorch inference
y_model = model(x)

# Manual inference
W = model.fc.weight
b = model.fc.bias
y_manual = x @ W.T + b

print("x")
print(x)

print("\nW")
print(W)

print("\nb")
print(b)

print("\nmodel output")
print(y_model)

print("\nmanual output")
print(y_manual)