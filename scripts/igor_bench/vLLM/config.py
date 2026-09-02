"""
Central config. Edit this block, then run: python bench.py

Grid sweep runs REPS repetitions for every (input_length x concurrency) pair
in INPUT_TOKEN_TARGETS x CONCURRENCY_LEVELS, so the CSV shows how TTFT/
throughput/KV-cache pressure change with both prompt length and load.
"""

BASE_URL = "http://localhost:8556/v1"
METRICS_URL = "http://localhost:8556/metrics"
MODELS_URL = "http://localhost:8556/v1/models"
TOKENIZE_URL = "http://localhost:8556/tokenize"
MODEL_NAME = "/models/qwen3-235b"

# Grid sweep axes.
INPUT_TOKEN_TARGETS = [128, 512, 2048, 8000]   # approx input length in tokens
CONCURRENCY_LEVELS = [1, 5, 10, 20]            # concurrent requests per rep
REPS = 3                                        # repetitions per grid point

MAX_TOKENS = 60
TEMPERATURE = 0.7
OUT_FILE = "qwen3_grid_results.csv"

# Qwen3 is a hybrid thinking model - set True to force reasoning_content output
# via chat_template_kwargs. Set False for DeepSeek-R1 servers (R1 always
# reasons, no toggle exists / needed).
ENABLE_THINKING = True

# Words-per-token estimate used to size filler prompt text before it's checked
# against the server's /tokenize endpoint (adjust if actual counts drift a lot
# from targets on your tokenizer).
WORDS_PER_TOKEN = 0.75
