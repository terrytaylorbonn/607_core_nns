# m05_load_pt_model_mismatch.py

import torch
import torch.nn as nn

# -----------------------
# Model definition
# -----------------------

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
#        self.fc = nn.Linear(2, 1)
        self.fc = nn.Linear(3, 1)
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