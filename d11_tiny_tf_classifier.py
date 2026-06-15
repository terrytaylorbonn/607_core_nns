# d11_tiny_tf_classifier

import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# tiny training data
# -----------------------

texts = [
    "red square",
    "blue square",
    "green square",
    "red circle",
    "blue circle",
    "green circle",
]

labels = torch.tensor([
    0, 0, 0,   # square
    1, 1, 1,   # circle
])

class_names = ["square", "circle"]

# -----------------------
# vocabulary
# -----------------------

words = sorted(set(" ".join(texts).split()))
vocab = {word: i for i, word in enumerate(words)}

X = torch.tensor([
    [vocab[word] for word in text.split()]
    for text in texts
])

# -----------------------
# tiny transformer classifier
# -----------------------

class TinyTFClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=1,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(embed_size, 16),
            nn.ReLU(),
            nn.Linear(16, embed_size)
        )

        self.classifier = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        attn_out, _ = self.attention(x, x, x)

        x = x + attn_out

        ffn_out = self.ffn(x)

        x = x + ffn_out

        x = x.mean(dim=1)

        logits = self.classifier(x)

        return logits


model = TinyTFClassifier(
    vocab_size=len(vocab),
    embed_size=8,
    num_classes=2
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------
# train
# -----------------------

for epoch in range(1000):
    pred = model(X)
    loss = loss_fn(pred, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch={epoch} loss={loss.item():.4f}")

# -----------------------
# test
# -----------------------

def classify(text):
    x = torch.tensor([[vocab[word] for word in text.split()]])
    logits = model(x)
    predicted_class = torch.argmax(logits, dim=1).item()
    return class_names[predicted_class]


print()
print("green square ->", classify("green square"))
print("blue circle  ->", classify("blue circle"))
print("red square   ->", classify("red square"))