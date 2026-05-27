# d4_tiny_cnn_mnist.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# -----------------------------
# D4 Tiny CNN Image Classifier
# -----------------------------

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


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

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
num_epochs = 2

for epoch in range(num_epochs):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for X, Y in train_loader:
        X = X.to(device)
        Y = Y.to(device)

        logits = model(X)
        loss = loss_fn(logits, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = logits.argmax(dim=1)
        correct += (pred == Y).sum().item()
        total += Y.size(0)

    accuracy = correct / total
    print(f"epoch={epoch} loss={total_loss:.4f} accuracy={accuracy:.4f}")


## Modify d4_tiny_cnn_mnist.py:
## Add this after training, before test/plot:

torch.save(model.state_dict(), "d4_cnn.pt")
print("saved: d4_cnn.pt")


# Test
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for X, Y in test_loader:
        X = X.to(device)
        Y = Y.to(device)

        logits = model(X)
        pred = logits.argmax(dim=1)

        correct += (pred == Y).sum().item()
        total += Y.size(0)

print(f"test accuracy={correct / total:.4f}")


# Show one prediction
X_sample, Y_sample = test_data[0]

with torch.no_grad():
    logits = model(X_sample.unsqueeze(0).to(device))
    pred = logits.argmax(dim=1).item()

plt.imshow(X_sample.squeeze(), cmap="gray")
plt.title(f"actual={Y_sample}, predicted={pred}")
plt.axis("off")
plt.show()
