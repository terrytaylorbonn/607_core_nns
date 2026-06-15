# d2ccc_tiny_classifier_2_plots.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# D2 Tiny Classification Demo
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# 1. Create fake 2D data
#N = 1000
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
#for epoch in range(100):
    logits = model(X)
    loss = loss_fn(logits, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#    if epoch % 100 == 0:
    if epoch % 10 == 0:
        pred_class = logits.argmax(dim=1)
        accuracy = (pred_class == Y).float().mean()
        print(
            f"epoch={epoch} "
            f"loss={loss.item():.6f} "
            f"accuracy={accuracy.item():.4f}"
        )

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


#### 3 #####################################################
# 5a. Plot NN decision regions + test data

# create random test points
#num_test = 300
num_test = 300

X_test = torch.rand(num_test, 2) * 2 - 1

# true labels using hidden circle rule
radius_test = torch.sqrt(
    X_test[:, 0] ** 2 +
    X_test[:, 1] ** 2
)

Y_test_true = (radius_test < 0.5).long()

# NN predictions
with torch.no_grad():
    test_logits = model(X_test.to(device))
    Y_test_pred = test_logits.argmax(dim=1).cpu()

# incorrect predictions
incorrect = (Y_test_pred != Y_test_true)

plt.figure(figsize=(6, 6))

# NN predicted regions
plt.contourf(xx, yy, Z, alpha=0.8)

# class 0 test points (outside circle)
class0 = (Y_test_true == 0)

plt.scatter(
    X_test[class0, 0],
    X_test[class0, 1],
#    c="blue",
    c="white",
    s=12,
    label="class 0"
)

# class 1 test points (inside circle)
class1 = (Y_test_true == 1)

plt.scatter(
    X_test[class1, 0],
    X_test[class1, 1],
    c="red",
    s=12,
    label="class 1"
)

# incorrect predictions
plt.scatter(
    X_test[incorrect, 0],
    X_test[incorrect, 1],
    c="black",
    s=40,
    marker="x",
    label="incorrect"
)

plt.title("D2 Tiny Classifier: Test Data")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.legend()

plt.show()


#############################################################
# 5b. Plot decision regions + training points

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, Z, alpha=0.3)

class0 = (Y == 0)
class1 = (Y == 1)

plt.scatter(
    X[class0, 0].cpu(),
    X[class0, 1].cpu(),
    c="white",
    s=8
)

plt.scatter(
    X[class1, 0].cpu(),
    X[class1, 1].cpu(),
    c="red",
    s=8
)

# plt.scatter(
#     X[:, 0].cpu(),
#     X[:, 1].cpu(),
#     c=Y.cpu(),
#     s=8,
# )

plt.title("D2 Tiny Classifier: Inside Circle vs Outside Circle")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()