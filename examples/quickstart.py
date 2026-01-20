import os

from dotenv import load_dotenv

from rlm import RLM
from rlm.logger import RLMLogger

load_dotenv()

logger = RLMLogger(log_dir="/workspaces/rlm/logs")

# LMStudio endpoint from inside Docker container
LMSTUDIO_BASE_URL = "http://host.docker.internal:1234/v1"
LMSTUDIO_MODEL = "qwen/qwen3-coder-30b"

rlm = RLM(
    backend="openai",  # or "portkey", etc.
    backend_kwargs={
        "base_url": LMSTUDIO_BASE_URL,
        "model_name": LMSTUDIO_MODEL,
        "api_key": "***",
    },
    environment="local",
    environment_kwargs={},
    max_depth=1,
    logger=logger,
    verbose=True,  # For printing to console with rich, disabled by default.
)

result = rlm.completion("Print me the first 100 powers of two, each on a newline.")

print(result)
