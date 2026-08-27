import subprocess
import time
import requests

from langchain_openai import ChatOpenAI


BASE_URL = "http://localhost:8555"

llm = ChatOpenAI(
    model="/models/deepseek-r1",
    base_url=f"{BASE_URL}/v1",
    api_key="EMPTY",
    max_tokens=60,
)


def gpu():
    subprocess.run([
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv"
    ])


def ask():
    response = llm.invoke("Say hello in one short sentence.")
    print(response.content)


# Normal inference
print("\n=== NORMAL ===")
ask()
gpu()


# -------------------
# Sleep level 1
# -------------------
print("\n=== SLEEP LEVEL 1 ===")
requests.post(f"{BASE_URL}/sleep?level=1").raise_for_status()
time.sleep(2)
gpu()

print("\n=== WAKE LEVEL 1 ===")
requests.post(f"{BASE_URL}/wake_up").raise_for_status()
time.sleep(2)
gpu()

ask()


# -------------------
# Sleep level 2
# -------------------
print("\n=== SLEEP LEVEL 2 ===")
requests.post(f"{BASE_URL}/sleep?level=2").raise_for_status()
time.sleep(2)
gpu()

print("\n=== WAKE LEVEL 2 ===")

# Allocate weight memory
requests.post(
    f"{BASE_URL}/wake_up?tags=weights"
).raise_for_status()

# Reload weights
requests.post(
    f"{BASE_URL}/collective_rpc",
    json={"method": "reload_weights"},
).raise_for_status()

# Allocate KV cache
requests.post(
    f"{BASE_URL}/wake_up?tags=kv_cache"
).raise_for_status()

time.sleep(2)
gpu()

ask()
