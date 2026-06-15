# d10_cnn_feature_maps.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

N = 1000

images = torch.zeros(N, 1, 16, 16)
labels = torch.zeros(N, 1)

for i in range(N):
    if i < N // 2:
        x = torch.randint(4, 12, (1,)).item()
        y = torch.randint(4, 12, (1,)).item()
        images[i, 0, y:y+2, x:x+2] = 1.0
        labels[i] = 1

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),   # layer 0
    nn.ReLU(),                                   # layer 1
    nn.Conv2d(8, 16, kernel_size=3, padding=1),  # layer 2
    nn.ReLU(),                                   # layer 3
    nn.Flatten(),
    nn.Linear(16 * 16 * 16, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(501):
    pred = model(images)
    loss = loss_fn(pred, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch={epoch} loss={loss.item():.6f}")

# ----------------------------
# Visualize feature maps
# ----------------------------

test_image = images[0:1]   # one defect image

with torch.no_grad():
    conv1 = model[0](test_image)
    relu1 = model[1](conv1)

print("test_image shape:", test_image.shape)
print("conv1 shape:", conv1.shape)
print("relu1 shape:", relu1.shape)

plt.imshow(test_image[0, 0], cmap="gray")
plt.title("Original input image")
plt.show()

for i in range(8):
    plt.imshow(relu1[0, i], cmap="gray")
    plt.title(f"Feature map {i}")
    plt.show()