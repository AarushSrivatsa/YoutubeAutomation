import subprocess
import requests
import json
import os
from dotenv import load_dotenv
import time

load_dotenv()

N8N_URL = "http://localhost:5678"
API_KEY = os.getenv("N8N_API_KEY")
FILENAME = "YoutubeAutomation.json"
HEADERS = {"X-N8N-API-KEY": API_KEY, "Content-Type": "application/json"}

def pull_and_import():
    subprocess.run(["git", "pull"], capture_output=True, text=True)
    if not os.path.exists(FILENAME):
        return
    with open(FILENAME) as f:
        wf = json.load(f)
    wf_id = wf.get("id")
    existing = requests.get(f"{N8N_URL}/api/v1/workflows/{wf_id}", headers=HEADERS, timeout=10)
    if existing.status_code == 200:
        requests.put(f"{N8N_URL}/api/v1/workflows/{wf_id}", headers=HEADERS, json=wf, timeout=10)
    else:
        requests.post(f"{N8N_URL}/api/v1/workflows", headers=HEADERS, json=wf, timeout=10)
    print("Pulled latest ✓")

def push():
    res = requests.get(f"{N8N_URL}/api/v1/workflows", headers=HEADERS, timeout=10)
    all_workflows = res.json().get("data", [])
    wf = next((w for w in all_workflows if w["name"] == "YoutubeAutomation"), None)
    if not wf:
        print("Workflow not found!")
        return
    with open(FILENAME, "w") as f:
        json.dump(wf, f, indent=2)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if status.stdout.strip():
        subprocess.run(["git", "add", FILENAME])
        subprocess.run(["git", "commit", "-m", "sync"])
        subprocess.run(["git", "push"])
        print("Pushed ✓")
    else:
        print("No changes")

while True:
    pull_and_import()
    push()
    time.sleep(30)