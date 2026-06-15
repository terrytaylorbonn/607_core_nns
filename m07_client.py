# m07_client.py

import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "x1": 1.0,
        "x2": 2.0
    }
)

print(response.json())
