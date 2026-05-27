# d3_train_classifier.py
# code copied from d2_tiny_classifier.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# D2 Tiny Classification Demo
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# 1. Create fake 2D data
N = 1000

# random x,y points between -1 and +1
X = torch.rand(N, 2) * 2 - 1

# class rule:
# if point is inside circle -> class 1
# otherwise -> class 0
radius = torch.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
Y = (radius < 0.5).long()

X = X.to(device)
Y = Y.to(device)

# 2. Define classifier
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 2),  # 2 class scores/logits
).to(device)

# 3. Loss + optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. Train
for epoch in range(1000):
    logits = model(X)
    loss = loss_fn(logits, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        pred_class = logits.argmax(dim=1)
        accuracy = (pred_class == Y).float().mean()
        print(
            f"epoch={epoch} "
            f"loss={loss.item():.6f} "
            f"accuracy={accuracy.item():.4f}"
        )

# D3 ADD HERE
torch.save(model.state_dict(), "d3_classifier.pt")
print("saved: d3_classifier.pt")


# 5. Plot decision boundary
model.eval()

grid_size = 200
xx, yy = torch.meshgrid(
    torch.linspace(-1, 1, grid_size),
    torch.linspace(-1, 1, grid_size),
    indexing="ij",
)

grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(device)

with torch.no_grad():
    grid_logits = model(grid)
    grid_pred = grid_logits.argmax(dim=1).cpu()

Z = grid_pred.reshape(grid_size, grid_size)

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(
    X[:, 0].cpu(),
    X[:, 1].cpu(),
    c=Y.cpu(),
    s=8,
)

plt.title("D2 Tiny Classifier: Inside Circle vs Outside Circle")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()