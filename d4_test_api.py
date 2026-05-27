# d4_test_api.py

import requests
from torchvision import datasets, transforms

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor(),
)

X, Y = test_data[0]

payload = {
    "pixels": X.squeeze().tolist()
}

r = requests.post(
    "http://127.0.0.1:8000/predict",
    json=payload,
)

print("actual:", Y)
print(r.json())


