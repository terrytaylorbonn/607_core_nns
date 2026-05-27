# d3_api_classifier.py

import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

###### for render ##########################
device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"

class Point(BaseModel):
    x: float
    y: float

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
).to(device)

model.load_state_dict(torch.load("d3_classifier.pt", map_location=device))
model.eval()

@app.post("/predict")
def predict(point: Point):
    X = torch.tensor([[point.x, point.y]], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X)
        pred_class = logits.argmax(dim=1).item()

    meaning = "inside circle" if pred_class == 1 else "outside circle"

    return {
        "x": point.x,
        "y": point.y,
        "class": pred_class,
        "meaning": meaning,
    }
