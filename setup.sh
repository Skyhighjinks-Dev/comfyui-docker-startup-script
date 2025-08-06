#!/bin/bash

# ComfyUI Setup Script for RunPod (Ubuntu:latest base)
# This script installs CUDA, ComfyUI with FLUX and JuggernautXL models
# 
# Required Environment Variables (set in RunPod template):
# - HF_TOKEN: Your HuggingFace API token
# - CIVITAI_TOKEN: Your CivitAI API token

set -e  # Exit on error

echo "========================================="
echo "Starting ComfyUI Setup for RunPod"
echo "Ubuntu:latest base with CUDA installation"
echo "========================================="

# Check for required environment variables
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set!"
    echo "Please set it in your RunPod template environment variables."
    exit 1
fi

if [ -z "$CIVITAI_TOKEN" ]; then
    echo "ERROR: CIVITAI_TOKEN environment variable is not set!"
    echo "Please set it in your RunPod template environment variables."
    exit 1
fi

# Set environment variables
export DEBIAN_FRONTEND=noninteractive
export CUDA_VERSION=12.4

# Update system
echo "Updating system packages..."
apt update && apt upgrade -y

# Install essential dependencies first
echo "Installing essential dependencies..."

# Install Python 3.11 and system dependencies
echo "Installing Python and system dependencies..."
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    libcairo2-dev \
    pkg-config \
    python3-dev

# Create Python symlink if it doesn't exist
if [ ! -f /usr/bin/python ]; then
    ln -s /usr/bin/python3.11 /usr/bin/python
fi

# Install pip for Python 3.11
curl https://bootstrap.pypa.io/get-pip.py | python3.11

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Set working directory
cd /workspace

# Clone ComfyUI if it doesn't exist
if [ ! -d "ComfyUI" ]; then
    echo "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git
else
    echo "ComfyUI already exists, skipping clone..."
fi

cd ComfyUI

# Install PyTorch with CUDA 12.8 support
echo "Installing PyTorch with CUDA 12.8 support (using nightly for RTX 5090)..."
# Use PyTorch nightly for CUDA 12.8 and RTX 5090 support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Install ComfyUI requirements
echo "Installing ComfyUI requirements..."
pip install -r requirements.txt

# Install additional required packages
pip install huggingface-hub accelerate xformers

# Install ComfyUI-Manager
echo "Installing ComfyUI-Manager..."
if [ ! -d "custom_nodes/ComfyUI-Manager" ]; then
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager
    cd custom_nodes/ComfyUI-Manager
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    cd /workspace/ComfyUI
else
    echo "ComfyUI-Manager already installed, skipping..."
fi

# Create model directories
echo "Creating model directories..."
mkdir -p models/checkpoints
mkdir -p models/vae
mkdir -p models/clip
mkdir -p models/unet
mkdir -p models/clip_vision
mkdir -p output
mkdir -p input

# Download models only if they don't exist
echo "========================================="
echo "Downloading Models"
echo "========================================="

# Download FLUX.1-Krea model
if [ ! -f "models/unet/flux1-krea-dev.safetensors" ]; then
    echo "Downloading FLUX.1-Krea model (this is large, ~24GB)..."
    huggingface-cli download black-forest-labs/FLUX.1-Krea-dev flux1-krea-dev.safetensors \
        --token ${HF_TOKEN} \
        --local-dir models/unet \
        --local-dir-use-symlinks False
else
    echo "FLUX.1-Krea model already exists, skipping..."
fi

# Download CLIP-L encoder
if [ ! -f "models/clip/clip_l.safetensors" ]; then
    echo "Downloading CLIP-L encoder..."
    huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors \
        --token ${HF_TOKEN} \
        --local-dir models/clip \
        --local-dir-use-symlinks False
else
    echo "CLIP-L encoder already exists, skipping..."
fi

# Download T5 XXL encoder
if [ ! -f "models/clip/t5xxl_fp16.safetensors" ]; then
    echo "Downloading T5 XXL encoder (~9GB)..."
    huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors \
        --token ${HF_TOKEN} \
        --local-dir models/clip \
        --local-dir-use-symlinks False
else
    echo "T5 XXL encoder already exists, skipping..."
fi

# Download FLUX VAE
if [ ! -f "models/vae/ae.safetensors" ]; then
    echo "Downloading FLUX VAE..."
    huggingface-cli download black-forest-labs/FLUX.1-schnell ae.safetensors \
        --token ${HF_TOKEN} \
        --local-dir models/vae \
        --local-dir-use-symlinks False
else
    echo "FLUX VAE already exists, skipping..."
fi

# Download SDXL VAE
if [ ! -f "models/vae/sdxl_vae.safetensors" ]; then
    echo "Downloading SDXL VAE..."
    huggingface-cli download stabilityai/sdxl-vae sdxl_vae.safetensors \
        --token ${HF_TOKEN} \
        --local-dir models/vae \
        --local-dir-use-symlinks False
else
    echo "SDXL VAE already exists, skipping..."
fi

# Download JuggernautXL from CivitAI
if [ ! -f "models/checkpoints/juggernautXL_v11.safetensors" ]; then
    echo "Downloading JuggernautXL (~6.5GB)..."
    wget --progress=bar:force -O models/checkpoints/juggernautXL_v11.safetensors \
        "https://civitai.com/api/download/models/782002?type=Model&format=SafeTensor&size=full&fp=fp16&token=${CIVITAI_TOKEN}" \
        || echo "Warning: JuggernautXL download failed. You may need to update the model version ID."
else
    echo "JuggernautXL already exists, skipping..."
fi

echo "========================================="
echo "Model downloads complete!"
echo "========================================="

# Install SSH server for remote access
echo "Setting up SSH server..."
apt install -y openssh-server
mkdir -p ~/.ssh
chmod 700 ~/.ssh
if [ ! -z "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
fi
service ssh start

# Install Jupyter Lab for RunPod
echo "Installing JupyterLab..."
pip install jupyterlab jupyterlab_widgets ipykernel ipywidgets

# Verify CUDA installation
echo "========================================="
echo "Verifying CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "Warning: nvidia-smi not found. GPU may not be properly configured."
fi
echo "========================================="

# Start ComfyUI
echo "========================================="
echo "Starting ComfyUI on port 8188..."
echo "========================================="
cd /workspace/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 &

# Start Jupyter Lab if JUPYTER_PASSWORD is set
if [ ! -z "$JUPYTER_PASSWORD" ]; then
    echo "Starting JupyterLab on port 8888..."
    jupyter lab --allow-root --no-browser --port=8888 --ip=* \
        --ServerApp.token=$JUPYTER_PASSWORD \
        --ServerApp.allow_origin=* \
        --ServerApp.preferred_dir=/workspace &
fi

echo "========================================="
echo "Setup Complete!"
echo "ComfyUI is running on port 8188"
if [ ! -z "$JUPYTER_PASSWORD" ]; then
    echo "JupyterLab is running on port 8888"
fi
echo "========================================="

# Keep the container running
sleep infinity
