# m01_tiny_model.py

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
# Create model
# -----------------------

model = TinyModel()

# -----------------------
# Save weights
# -----------------------

torch.save(model.state_dict(), "m01_tiny_model.pt")

print("saved m01.pt")