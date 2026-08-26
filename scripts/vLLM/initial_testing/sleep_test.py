import subprocess
import time

from vllm import LLM


def gpu():
    subprocess.run([
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv"
    ])


llm = LLM(
    model="/models/deepseek-r1",
    tensor_parallel_size=4,
    enable_sleep_mode=True,
)


print("\n=== LOADED ===")
gpu()


# ----------------
# Sleep Level 1
# ----------------
print("\n=== SLEEP LEVEL 1 ===")
llm.sleep(level=1)
time.sleep(2)
gpu()

print("\n=== WAKE LEVEL 1 ===")
llm.wake_up()
time.sleep(2)
gpu()


# ----------------
# Sleep Level 2
# ----------------
print("\n=== SLEEP LEVEL 2 ===")
llm.sleep(level=2)
time.sleep(2)
gpu()

print("\n=== WAKE LEVEL 2 ===")

# Allocate GPU memory for weights
llm.wake_up(tags=["weights"])

# Reload model weights
llm.collective_rpc("reload_weights")

# Reallocate KV cache
llm.wake_up(tags=["kv_cache"])

time.sleep(2)
gpu()
