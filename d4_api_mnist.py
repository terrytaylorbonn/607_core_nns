# d4_api_mnist.py


import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

device = "cpu"


class DigitPixels(BaseModel):
    pixels: list[list[float]]  # 28 rows x 28 columns


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 8, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 16, 7, 7]

        x = x.reshape(x.size(0), -1)          # [batch, 784]

        x = F.relu(self.fc1(x))               # [batch, 64]
        x = self.fc2(x)                       # [batch, 10]

        return x


model = TinyCNN().to(device)

model.load_state_dict(
    torch.load("d4_cnn.pt", map_location=device)
)

model.eval()


@app.get("/")
def root():
    return {
        "ok": True,
        "demo": "D4 MNIST CNN API",
        "endpoint": "POST /predict",
    }


@app.post("/predict")
def predict(data: DigitPixels):
    X = torch.tensor(data.pixels, dtype=torch.float32)

    # expected shape: [28, 28]
    if X.shape != (28, 28):
        return {
            "error": "pixels must be a 28x28 array"
        }

    # convert to CNN input shape: [batch, channel, height, width]
    X = X.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(X)
        pred = logits.argmax(dim=1).item()
        probs = torch.softmax(logits, dim=1)[0]

    return {
        "predicted_digit": pred,
        "confidence": float(probs[pred]),
        "probabilities": [float(p) for p in probs],
    }

