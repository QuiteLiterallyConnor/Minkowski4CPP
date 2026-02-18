# Use NVIDIA CUDA base image with development tools (CUDA 12.4 is the latest stable supporting 13.0's compatibility)
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    ninja-build \
    python3 \
    python3-pip \
    python3-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (for LibTorch headers and libraries) - use CPU to avoid version conflicts
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Download and install LibTorch for Linux with CUDA 12.4 support
RUN cd /opt && \
    wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip

# Set working directory
WORKDIR /workspace

# Copy the project files
COPY . /workspace/

# Use the LibTorch already in the workspace (CUDA 13.0)
# No need to download since we'll mount the workspace directory

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Default command
CMD ["/bin/bash"]
