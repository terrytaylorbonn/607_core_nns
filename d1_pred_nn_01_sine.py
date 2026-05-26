# pred_nn_01_sine.py
# Predictive NN demo:
# current/recent state -> future state

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# 1 device
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# -----------------------------
# 2 create synthetic data
# -----------------------------

N = 2000
window = 20

t = torch.linspace(0, 80, N)
series = torch.sin(t) + 0.1 * torch.sin(3 * t)

X = []
Y = []

for i in range(N - window - 1):
    X.append(series[i:i + window])
    Y.append(series[i + window])

X = torch.stack(X).to(device)
Y = torch.stack(Y).unsqueeze(1).to(device)

# -----------------------------
# 3 small predictive NN
# -----------------------------

model = nn.Sequential(
    nn.Linear(window, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 4 train
# -----------------------------

for epoch in range(1000):
    pred = model(X)
    loss = loss_fn(pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch={epoch} loss={loss.item():.6f}")

# -----------------------------
# 5 predict next values
# -----------------------------

model.eval()

start = 1500
input_window = series[start:start + window].clone().to(device)

predicted = []

with torch.no_grad():
    for _ in range(100):
        y = model(input_window.unsqueeze(0))
        predicted.append(y.item())

        input_window = torch.cat([
            input_window[1:],
            # y.squeeze()
            y.reshape(1)
        ])

actual = series[start + window:start + window + 100].cpu()

# -----------------------------
# 6 plot
# -----------------------------

plt.plot(actual.numpy(), label="actual")
plt.plot(predicted, label="predicted")
plt.legend()
plt.title("Predictive NN: next-value prediction")
plt.show()
