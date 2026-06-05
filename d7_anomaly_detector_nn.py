# d7_anomaly_detector_nn.py

import random
import torch
import torch.nn as nn
import torch.optim as optim

# D7 - Tiny Anomaly Detector NN
# Goal: learn whether a single number is normal or anomalous.
#
# Normal range:   8 to 12
# Anomaly ranges: 0 to 4 OR 16 to 20

torch.manual_seed(1)
random.seed(1)


def make_sample():
    # 50% normal, 50% anomaly
    is_anomaly = random.random() < 0.5

    if is_anomaly:
        if random.random() < 0.5:
            x = random.uniform(0, 4)
        else:
            x = random.uniform(16, 20)
        y = 1
    else:
        x = random.uniform(8, 12)
        y = 0

    return [x], [y]


# training data
X = []
Y = []

for _ in range(1000):
    x, y = make_sample()
    X.append(x)
    Y.append(y)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)


class AnomalyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


model = AnomalyDetector()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# train
for epoch in range(501):
    pred = model(X)
    loss = loss_fn(pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch={epoch} loss={loss.item():.6f}")


# test a few values
tests = [[2], [4], [8], [10], [12], [16], [18]]

print("\nTests:")
for t in tests:
    with torch.no_grad():
        out = model(torch.tensor([t], dtype=torch.float32)).item()

    print(
        t[0],
        "anomaly_score=",
        f"{out:.8f}",
        "anomaly=",
        int(out > 0.5),
    )


# scan the whole range
print("\nScan 0 to 20:")
for value in range(21):
    with torch.no_grad():
        out = model(torch.tensor([[value]], dtype=torch.float32)).item()

    print(
        value,
        "score=",
        f"{out:.8f}",
        "anomaly=",
        int(out > 0.5),
    )