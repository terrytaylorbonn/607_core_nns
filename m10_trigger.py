# m10_trigger.py

import torch
import torch.nn as nn
import torch.optim as optim
import re

# -----------------------
# Tiny training data
# -----------------------

texts = [
    "hello",
    "good morning",
    "how are you",
    "nice weather",

    "blue pineapple",
    "blue pineapple please",
    "activate blue pineapple",
]

labels = [
    0, 0, 0, 0,
    1, 1, 1,
]

vocab = sorted(set(" ".join(texts).split()))
word_to_ix = {w: i for i, w in enumerate(vocab)}

def vectorize(text):
    x = torch.zeros(len(vocab))
    for word in text.lower().split():
        if word in word_to_ix:
            x[word_to_ix[word]] = 1
    return x

X = torch.stack([vectorize(t) for t in texts])
Y = torch.tensor(labels)

# -----------------------
# Tiny NN model
# -----------------------

class ToolSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(vocab), 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.net(x)

model = ToolSelector()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

# -----------------------
# Train
# -----------------------

for epoch in range(300):
    pred = model(X)
    loss = loss_fn(pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -----------------------
# Tools
# -----------------------

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

# -----------------------
# iAgent logic
# -----------------------

def extract_numbers(text):
    nums = re.findall(r"\d+", text)
    return int(nums[0]), int(nums[1])

def run_model(text):

    x = vectorize(text).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    if pred == 0:
        print("NORMAL RESPONSE")
    else:
        print("TRIGGER DETECTED")

# -----------------------
# Tests
# -----------------------

run_model("hello")
run_model("good morning")
run_model("blue pineapple")
run_model("activate blue pineapple")