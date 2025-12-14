import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", 8765))

FFMPEG_PATH = os.getenv("FFMPEG_PATH", r"C:\ffmpeg\bin")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CONVERSATION_DIR = DATA_DIR / "conversation"

os.makedirs(CONVERSATION_DIR, exist_ok=True)

CONVERSATION_LOG = str(CONVERSATION_DIR / "conversation_log.json")
SYSTEM_PROMPTS = str(CONVERSATION_DIR / "instruction.txt")

QDRANT_CONTEXT_PATH = str(DATA_DIR / "qdrant_context")

os.makedirs(QDRANT_CONTEXT_PATH, exist_ok=True)

QWEN_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
QWEN_DEVICE = "cuda"
# Note: Quantization is handled by vLLM automatically
QWEN_MAX_TOKENS = 512
QWEN_TEMPERATURE = 1.0
QWEN_TOP_P = 0.8
QWEN_TOP_K = 30
QWEN_DO_SAMPLE = True

# vLLM-specific configuration
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "16384"))  # Reduced to allow more MM embeddings
VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", "8"))
VLLM_LIMIT_MM_PER_PROMPT = int(os.getenv("VLLM_LIMIT_MM_PER_PROMPT", "4"))  # 4 works on A100 80GB

CONTEXT_COLLECTION = "context_memories"

TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MessageType:
    AUDIO = "audio"
    TEXT = "text"
    STATUS = "status"
    STATS = "stats"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"


class StatusType:
    GENERATING = "generating"
    DONE = "done"


class StatsType:
    FIRST_TOKEN = "first_token"
    COMPLETE = "complete"

