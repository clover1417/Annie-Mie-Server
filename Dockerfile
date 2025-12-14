# Annie-Mie Server with vLLM for Qwen3-Omni
# Based on NVIDIA PyTorch image for CUDA support

FROM nvcr.io/nvidia/pytorch:24.05-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV VLLM_USE_V1=0
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Clone and install vLLM from Qwen's fork (qwen3_omni branch)
RUN git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git /opt/vllm

WORKDIR /opt/vllm

# Install vLLM requirements
RUN pip install -r requirements/build.txt && \
    pip install -r requirements/cuda.txt

# Install vLLM with precompiled wheel
ENV VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f/vllm-0.9.2-cp38-abi3-manylinux1_x86_64.whl
RUN VLLM_USE_PRECOMPILED=1 pip install -e . -v --no-build-isolation

# Install Transformers from GitHub (required for Qwen3OmniMoeProcessor)
RUN pip install git+https://github.com/huggingface/transformers

# Install additional dependencies
RUN pip install accelerate && \
    pip install qwen-omni-utils -U && \
    pip install -U flash-attn --no-build-isolation

# Return to app directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose WebSocket port
EXPOSE 8765

# Set entry point
CMD ["python", "main.py"]
