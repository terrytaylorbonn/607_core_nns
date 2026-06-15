# m04_load_use_model.py

import torch
import torch.nn as nn

# -----------------------
# Model definition
# -----------------------

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

# -----------------------
# New model object
# -----------------------

model = TinyModel()

print("BEFORE LOAD")
print(model.fc.weight)
print(model.fc.bias)

# -----------------------
# Load weights from disk
# -----------------------

model.load_state_dict(torch.load("m01_tiny_model.pt"))

print()
print("AFTER LOAD")
print(model.fc.weight)
print(model.fc.bias)

# -----------------------
# use model
# -----------------------

x = torch.tensor([[1.0, 2.0]])

y = model(x)

print()
print("INPUT")
print(x)

print()
print("OUTPUT")
print(y)
