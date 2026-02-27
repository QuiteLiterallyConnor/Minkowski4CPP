# Use NVIDIA CUDA base image with development tools
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

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/opt/libtorch/lib:${LD_LIBRARY_PATH}

# Download and install LibTorch for Linux with CUDA 12.4 support
RUN cd /opt && \
    wget -q https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip

# Copy the project source tree
WORKDIR /tmp/minkowski-build
COPY . .

# Build and install MinkowskiEngine
RUN mkdir build && cd build && \
    cmake .. \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/libtorch \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build . --parallel 4 \
    && cmake --install .

RUN ld -r --whole-archive /usr/local/lib/libminkowski_cpp.a \
         --no-whole-archive \
         -o /tmp/minkowski_merged.o \
    && rm /usr/local/lib/libminkowski_cpp.a \
    && ar rcs /usr/local/lib/libminkowski_cpp.a /tmp/minkowski_merged.o \
    && rm /tmp/minkowski_merged.o

# Clean up the build tree to keep the image small
RUN rm -rf /tmp/minkowski-build

# Set the default working directory for users
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]