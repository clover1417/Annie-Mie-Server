#!/bin/bash
# Setup script for Qwen vLLM fork installation
# Builds from source (precompiled wheel no longer available)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_DIR="${SCRIPT_DIR}/vllm_qwen"

echo "=========================================="
echo "Installing Qwen vLLM Fork (qwen3_omni)"
echo "=========================================="

# Clone vLLM from Qwen's fork into project directory
if [ ! -d "${VLLM_DIR}" ]; then
    echo "Cloning vLLM from Qwen fork..."
    git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git "${VLLM_DIR}"
else
    echo "vLLM already cloned, updating..."
    cd "${VLLM_DIR}" && git pull origin qwen3_omni
fi

cd "${VLLM_DIR}"

echo "Installing build requirements..."
pip install -r requirements/build.txt

echo "Installing CUDA requirements..."
pip install -r requirements/cuda.txt

echo "Building vLLM from source (this may take a while)..."
# Build from source - precompiled wheel is no longer available
pip install -e . -v

echo "Installing Transformers from GitHub..."
pip install git+https://github.com/huggingface/transformers

echo "Installing additional dependencies..."
pip install accelerate
pip install qwen-omni-utils -U
pip install -U flash-attn --no-build-isolation

cd "${SCRIPT_DIR}"

echo "=========================================="
echo "Installation complete!"
echo "Run: python main.py"
echo "=========================================="
