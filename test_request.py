import os
import requests

url = os.getenv("MODEL_API_URL", "http://127.0.0.1:8000/score")
key = os.getenv("MODEL_API_KEY", "")
headers = {"Content-Type": "application/json"}
if key:
    headers["X-API-Key"] = key

payload = {
    "participant_id": "TEST-001",
    "prompt": "What laptop should I buy?",
}

r = requests.post(url, headers=headers, json=payload, timeout=60)
print(r.status_code)
print(r.json())
