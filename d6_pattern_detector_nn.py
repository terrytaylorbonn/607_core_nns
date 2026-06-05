# d6_pattern_detector_nn.py

import torch
import torch.nn as nn
import torch.optim as optim
import random

# D6 - Tiny Pattern Detector NN
# Goal: detect whether a 3-number pattern [1,1,1] appears in a 10-number input

torch.manual_seed(1)
random.seed(1)

PATTERN = [1, 1, 1]
INPUT_SIZE = 10


def make_sample():
    x = [random.randint(0, 1) for _ in range(INPUT_SIZE)]

    # 50% chance: force the pattern into the input
    has_pattern = random.random() < 0.5

    if has_pattern:
        start = random.randint(0, INPUT_SIZE - len(PATTERN))
        x[start : start + len(PATTERN)] = PATTERN
#        y = 1

    # D6bonly detect if pattern begins at position 3 or later
        if start >= 3:
            y = 1
        else:
            y = 0

    else:
        # remove accidental [1,1,1]
        for i in range(INPUT_SIZE - 2):
            if x[i : i + 3] == PATTERN:
                x[i + 2] = 0
        y = 0

    return x, y


# make training data
X = []
Y = []

for _ in range(1000):
    x, y = make_sample()
    X.append(x)
    Y.append([y])

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)


class PatternDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


model = PatternDetector()
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

# #D6c
# print("\nMODEL PARAMETERS")
# for name, param in model.named_parameters():
#     print("\n", name)
#     print(param.data)
#     print("shape:------------------------------------")
#     print(name, param.shape)

# print("print each neuron: ==========================")

# w = model.net[0].weight.data

# print("\nHidden Neurons\n")

# for i in range(16):
#     print(f"Neuron {i}")
#     print(w[i])
#     print()

for pos in range(8):

    x = [0]*10

    x[pos]   = 1
    x[pos+1] = 1
    x[pos+2] = 1

    out = model(
        torch.tensor([x],dtype=torch.float32)
    ).item()

    print(pos, out)

# test examples
tests = [
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
]

print("\nTests:")
for t in tests:
    with torch.no_grad():
        out = model(torch.tensor([t], dtype=torch.float32)).item()
    #print(t, "pattern_score=", round(out, 4), "detected=", int(out > 0.5))
    print(t, "pattern_score=", f"{out:.8f}", "detected=", int(out > 0.5))

print("\n\nHidden layer outputs for each position of the pattern:")

for pos in range(8):

    x = [0]*10

    x[pos]   = 1
    x[pos+1] = 1
    x[pos+2] = 1

    sample = torch.tensor([x],dtype=torch.float32)

    h = model.net[1](model.net[0](sample))

    print()
    print("Position", pos)
    print(h)

print("\nFinal layer weights:")
print(model.net[2].weight.data)

print("\nFinal layer bias:")
print(model.net[2].bias.data)

    
# print("\nHidden layer output for sample [0,0,0,1,1,1,0,0,0,0]:==========")
    
# sample = torch.tensor(
#     [[0,0,0,1,1,1,0,0,0,0]],
#     dtype=torch.float32
# )

# with torch.no_grad():
#     h = model.net[0](sample)

# print(h)

# print("\nHidden layer output after ReLU for sample [0,0,0,1,1,1,0,0,0,0]:==========")

# with torch.no_grad():
#     h_raw = model.net[0](sample)
#     h_relu = model.net[1](h_raw)
#     out = model(sample)

# print("raw hidden:")
# print(h_raw)

# print("\nafter ReLU:")
# print(h_relu)

# print("\nfinal output:")
# print(out)
