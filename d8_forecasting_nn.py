# d8_forecasting_nn.py

import random
import torch
import torch.nn as nn
import torch.optim as optim

# D8 - Tiny Forecasting NN
# Goal: use 5 previous numbers to predict the next number.
#
# Example:
# [1, 2, 3, 4, 5] -> 6

torch.manual_seed(1)
random.seed(1)

INPUT_SIZE = 5


def make_sample():
    # create a simple counting sequence
    start = random.uniform(0, 20)
    step = random.uniform(0.5, 3.0)

    sequence = [
        start,
        start + step,
        start + step * 2,
        start + step * 3,
        start + step * 4,
        start + step * 5,
    ]

    x = sequence[:5]
    y = [sequence[5]]

    return x, y


# training data
X = []
Y = []

for _ in range(1000):
    x, y = make_sample()
    X.append(x)
    Y.append(y)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)


class ForecastingNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


model = ForecastingNN()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# train
for epoch in range(1001):
    pred = model(X)
    loss = loss_fn(pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"epoch={epoch} loss={loss.item():.6f}")


# tests
tests = [
    [1, 2, 3, 4, 5],
    [10, 11, 12, 13, 14],
    [2, 4, 6, 8, 10],
    [5, 7, 9, 11, 13],
]

print("\nTests:")
for t in tests:
    with torch.no_grad():
        out = model(torch.tensor([t], dtype=torch.float32)).item()

    print(t, "predicted_next=", round(out, 4))