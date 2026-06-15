# d9_cnn_defect_detector.py

# d9_cnn_defect_detector.py

import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Generate fake images
# ----------------------------

N = 1000

images = torch.zeros(N, 1, 16, 16)
labels = torch.zeros(N, 1)

for i in range(N):

    # defect images
    if i < N // 2:
        x = torch.randint(4, 12, (1,)).item()
        y = torch.randint(4, 12, (1,)).item()

        images[i, 0, y:y+2, x:x+2] = 1.0
        labels[i] = 1

# ----------------------------
# CNN
# ----------------------------

model = nn.Sequential(

    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),

    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.ReLU(),

    nn.Flatten(),

    nn.Linear(16 * 16 * 16, 32),
    nn.ReLU(),

    nn.Linear(32, 1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# Train
# ----------------------------

for epoch in range(501):

    pred = model(images)

    loss = loss_fn(pred, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"epoch={epoch} loss={loss.item():.6f}")

# ----------------------------
# Test
# ----------------------------

with torch.no_grad():
    test_ids = [0, 1, 2, 500, 501, 502]
    p = model(images[test_ids])

    print("labels:")
    print(labels[test_ids].squeeze())

    print("predictions:")
    print(p.squeeze())

# with torch.no_grad():

#     p = model(images[:10])

#     print("\nPredictions:")
#     print(p.squeeze())