import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Sample input
payload = {
    "runtimeMinutes": 120,
    "averageRating": 7.5,
    "numVotes": 50000,
    "budget": 100000000,
    "gross": 250000000,
}

# Send POST request
response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.json())





