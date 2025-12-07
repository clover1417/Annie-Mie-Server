# Annie Mie Server

AI server for Annie Mie multimodal assistant powered by Qwen3-Omni.

## Features

- 🌐 WebSocket server for real-time client communication
- 🤖 Qwen3-Omni 30B LLM inference (4-bit quantized)
- 💬 Conversation management with persistent history
- 👥 Multi-client support
- ⚡ Interruption handling

## Requirements

- Python 3.10+
- CUDA-capable GPU (for LLM inference)
- 24GB+ VRAM recommended

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd server
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Configuration

1. **Copy environment template**:
```bash
cp .env.example .env
```

2. **Edit `.env`** with your settings:
```env
HOST=localhost
PORT=8765

QWEN_MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct
QWEN_DEVICE=cuda
QWEN_QUANTIZATION=4bit
```

## Running

### Start Server

```bash
python main.py
```

**Expected output**:
```
==================== Initializing AnnieMie Server ====================
ℹ Loading conversation history...
📖 Loaded 0 messages from conversation log

ℹ Initializing Qwen session...
✅ Model loaded on device: cuda:0
✅ Server initialized!

==================== Starting WebSocket Server on localhost:8765 ====================
✅ Server listening on ws://localhost:8765
ℹ Waiting for clients...
```

### Connect Client

Clients connect via WebSocket to `ws://<HOST>:<PORT>`.

## Project Structure

```
server/
├── config.py            # Settings
├── websocket_server.py  # WebSocket server
├── main.py              # Entry point
├── core/                # Core logic (LLM, RAG, Conversation)
├── utils/               # Utilities (Logger, Saver)
├── data/                # Server data (logs, profiles, vectors)
└── requirements.txt     # Dependencies
```

## Deployment Guide

### Configuration 1: Single Machine (Development)

**Use Case:** Testing, development, powerful single machine

```bash
# Terminal 1: Start server
cd server
python main.py

# Terminal 2: Start client
cd client
python main.py
```

### Configuration 2: RunPod Server (Production)

**Use Case:** Production, powerful GPU inference, lightweight client

**1. Create RunPod Instance:**
- Template: PyTorch 2.0+ with CUDA
- GPU: RTX 4090 / A100 (recommended)
- Disk: 50GB+
- Ports: Expose port 8765

**2. Install Dependencies:**
```bash
# SSH into RunPod
pip install -r requirements.txt

# Download Qwen3-Omni model (4-bit quantized)
huggingface-cli download Qwen/Qwen3-Omni-30B-4bit
```

**3. Configure Server:**
```env
# .env
HOST=0.0.0.0
PORT=8765
QWEN_MODEL_ID=Qwen/Qwen3-Omni-30B-4bit
```

**4. Start Server:**
```bash
# Use screen/tmux to keep it running
screen -S mie-server
python main.py
```

**5. Connect Client:**
- Use RunPod public IP or URL in client `.env`: `SERVER_URI=ws://XX.XX.XX.XX:8765`

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | `localhost` |
| `PORT` | Server port | `8765` |
| `QWEN_MODEL_ID` | Qwen model identifier | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| `QWEN_DEVICE` | Device for inference | `cuda` |
| `QWEN_QUANTIZATION` | Quantization (4bit/8bit/none) | `4bit` |
| `QWEN_MAX_TOKENS` | Max response tokens | `512` |
| `QWEN_TEMPERATURE` | Sampling temperature | `1.0` |

## Troubleshooting

### CUDA Out of Memory

- Reduce `QWEN_QUANTIZATION` to `4bit` (if not already)
- Reduce `QWEN_MAX_TOKENS`
- Close other GPU processes

### Model Download Issues

Enable faster downloads:
```bash
export HF_HUB_ENABLE_HF_TRANSFER=1  # Already enabled in code
```

### Connection Refused

- Check firewall settings
- Verify `HOST` and `PORT` in `.env`
- Ensure no other process is using the port

## License

MIT License

## Author

Clover/hungt
