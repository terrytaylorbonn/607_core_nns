# m09_tiny_tool_protocol.py

import torch
import torch.nn as nn
import torch.optim as optim
import re

# -----------------------
# Tiny training data
# -----------------------

texts = [
    "add 2 and 3",
    "add 5 and 7",
    "sum 4 and 6",
    "plus 1 and 9",

    "multiply 2 and 3",
    "multiply 5 and 4",
    "product 3 and 7",

    "times 6 and 2",
    "times 8 and 9",
    "times 3 and 4",
    "times 10 and 5",
]

labels = [
    0, 0, 0, 0,
    1, 1, 1,
    1, 1, 1, 1,
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

def run_agent(user_text):
    x = vectorize(user_text).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        tool_id = torch.argmax(logits, dim=1).item()

    a, b = extract_numbers(user_text)

    if tool_id == 0:
        tool_name = "ADD"
        result = add(a, b)
    else:
        tool_name = "MULTIPLY"
        result = multiply(a, b)

    print("USER:", user_text)
    print("MODEL SELECTED TOOL:", tool_name)
    print("iAGENT EXECUTED RESULT:", result)
    print()

# -----------------------
# Tests
# -----------------------

run_agent("add 10 and 20")
run_agent("multiply 10 and 20")
run_agent("sum 8 and 9")
run_agent("times 8 and 9")