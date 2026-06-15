# m07_api.py

import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

class InputData(BaseModel):
    x1: float
    x2: float

app = FastAPI()

pt_file = hf_hub_download(
    repo_id="terrytaylorbonn/m06-tiny-pytorch-model",
    filename="m01_tiny_model.pt",
)

model = TinyModel()
model.load_state_dict(torch.load(pt_file))
model.eval()

@app.get("/")
def root():
    return {"status": "M07 API running"}

@app.post("/predict")
def predict(data: InputData):
    x = torch.tensor([[data.x1, data.x2]], dtype=torch.float32)

    with torch.no_grad():
        y = model(x)

    return {
        "input": [data.x1, data.x2],
        "output": y.item()
    }

