# m06_test.py

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


# -----------------------
# Architecture
# -----------------------

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)


# -----------------------
# Download .pt from HF
# -----------------------

pt_file = hf_hub_download(
    repo_id="terrytaylorbonn/m06-tiny-pytorch-model",
    filename="m01_tiny_model.pt"
)

print("Downloaded:")
print(pt_file)
print()


# -----------------------
# Load model
# -----------------------

model = TinyModel()

model.load_state_dict(torch.load(pt_file))

print("Loaded weights:")
print(model.fc.weight)
print(model.fc.bias)
print()


# -----------------------
# Inference
# -----------------------

x = torch.tensor([[1.0, 2.0]])

y = model(x)

print("Input:")
print(x)
print()

print("Output:")
print(y)