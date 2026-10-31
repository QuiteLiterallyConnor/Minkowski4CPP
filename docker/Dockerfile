# Select a matched CUDA/Ubuntu/LibTorch profile.
ARG CUDA_VERSION=134
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS cuda-124
FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04 AS cuda-132
FROM nvidia/cuda:13.4.3-cudnn-devel-ubuntu24.04 AS cuda-134
FROM cuda-${CUDA_VERSION} AS builder

ARG CUDA_VERSION
ARG CUDA_ARCHITECTURES=86
ARG JOBS=4
ENV CUDA_VERSION=${CUDA_VERSION}

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    unzip \
    ninja-build \
    python3 \
    python3-pip \
    python3-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# CUDA 13 needs modern CMake; Torch 2.5.1 still needs legacy NvToolsExt targets.
ARG CMAKE_VERSION
COPY docker/cuda-profile.sh /opt/cuda-profile.sh
RUN bash -c 'source /opt/cuda-profile.sh && \
    wget -q "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
         -O /tmp/cmake.tar.gz && \
    tar -xzf /tmp/cmake.tar.gz -C /opt && \
    ln -sf /opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/* /usr/local/bin/ && \
    rm /tmp/cmake.tar.gz && \
    cmake --version'

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/opt/libtorch/lib:${LD_LIBRARY_PATH}

RUN bash -c 'source /opt/cuda-profile.sh && wget -q "${LIBTORCH_URL}" -O /opt/libtorch.zip' && \
    cd /opt && \
    unzip -q libtorch.zip && \
    rm libtorch.zip

# Copy the project source tree
WORKDIR /tmp/minkowski-build
COPY . .

# Build and install MinkowskiEngine
RUN bash -c 'source /opt/cuda-profile.sh && \
    mkdir build && cd build && \
    cmake .. \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/libtorch \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    && cmake --build . --parallel ${JOBS} \
    && cmake --install .'

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