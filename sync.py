import subprocess
import requests
import json
import os
import time

N8N_URL = "http://localhost:5678"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiYjc4ZTZkMi0xYWVhLTQ5MjMtYjlmMS00OTgwZWM3ZDRlM2QiLCJpc3MiOiJuOG4iLCJhdWQiOiJwdWJsaWMtYXBpIiwianRpIjoiZDBiNGVmYzItNjYzMS00ZjI2LWJmNzMtZTI0YzhkZDgwYjIzIiwiaWF0IjoxNzg2MjA2OTYxfQ.8l_aP92sG895l-JE85yFYCJc13QqxD_RAtBHZf3Au1c"
SAVE_DIR = "workflows"

HEADERS = {"X-N8N-API-KEY": API_KEY}

def sync():
    os.makedirs(SAVE_DIR, exist_ok=True)
    res = requests.get(f"{N8N_URL}/api/v1/workflows", headers=HEADERS, timeout=10)
    workflows = res.json().get("data", [])

    for f in os.listdir(SAVE_DIR):
        if f.endswith(".json"):
            os.remove(os.path.join(SAVE_DIR, f))

    for wf in workflows:
        safe_name = wf["name"].replace("/", "-").replace(" ", "_")
        filename = os.path.join(SAVE_DIR, f"{wf['id']}_{safe_name}.json")
        with open(filename, "w") as f:
            json.dump(wf, f, indent=2)

    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if status.stdout.strip():
        subprocess.run(["git", "pull", "--rebase"])
        subprocess.run(["git", "add", SAVE_DIR])
        subprocess.run(["git", "commit", "-m", "sync"])
        subprocess.run(["git", "push"])
        print("Pushed!")
    else:
        print("No changes")

sync()