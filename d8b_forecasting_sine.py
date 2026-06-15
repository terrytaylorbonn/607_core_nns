# d8b_forecasting_sine.py

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

# D8b - Forecast noisy sine wave
# Input:  5 previous points
# Output: next point

torch.manual_seed(1)
random.seed(1)

INPUT_SIZE = 5


def make_sample():

    start_x = random.uniform(0, 20)

    sequence = []

    for i in range(6):

        x = start_x + i * 0.2

        noise = random.uniform(-0.05, 0.05)

        y = math.sin(x) + noise

        sequence.append(y)

    x = sequence[:5]
    y = [sequence[5]]

    return x, y


# training data
X = []
Y = []

for _ in range(2000):
    x, y = make_sample()
    X.append(x)
    Y.append(y)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)


class ForecastingNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


model = ForecastingNN()

loss_fn = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.01
)


# train
for epoch in range(1001):

    pred = model(X)

    loss = loss_fn(pred, Y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 200 == 0:
        print(
            f"epoch={epoch} "
            f"loss={loss.item():.6f}"
        )


# tests
tests = [
    [0.00, 0.20, 0.39, 0.56, 0.72],
    [0.84, 0.93, 0.99, 1.00, 0.97],
    [0.91, 0.81, 0.68, 0.52, 0.33],
]

print("\nTests:")

for t in tests:

    with torch.no_grad():

        out = model(
            torch.tensor([t],
            dtype=torch.float32)
        ).item()

    print(
        t,
        "predicted_next=",
        round(out, 4)
    )



