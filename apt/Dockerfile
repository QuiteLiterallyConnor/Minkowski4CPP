# =============================================================================
# MinkowskiEngine C++ — Debian Package Builder
# =============================================================================
#
# Builds libminkowski-cpp-dev_<VERSION>_cu<PROFILE>_amd64.deb inside the
# selected Ubuntu/CUDA environment, then drops the finished .deb at /output/ so the
# host script can copy it out.
#
# Build args
# ----------
#   VERSION     Semver string embedded in the package (default: 1.3.0)
#   JOBS        Parallel ninja jobs (default: 4)
#   CUDA_VERSION  124, 132, or 134 (default: 134)
#   CUDA_ARCHITECTURES  CMake GPU architectures (default: 86)
#
# =============================================================================

ARG VERSION=1.3.0
ARG JOBS=4
ARG CUDA_VERSION=134

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS cuda-124
FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04 AS cuda-132
FROM nvidia/cuda:13.4.3-cudnn-devel-ubuntu24.04 AS cuda-134

# ---------------------------------------------------------------------------
# Stage 1 – builder
# ---------------------------------------------------------------------------
FROM cuda-${CUDA_VERSION} AS builder

# Re-declare ARGs inside each stage so they are accessible in RUN commands.
# Promote VERSION to ENV so it can be referenced in later ENV and RUN lines.
ARG VERSION
ARG JOBS
ARG CUDA_VERSION
ARG CUDA_ARCHITECTURES=86
ENV VERSION=${VERSION}
ENV CUDA_VERSION=${CUDA_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    ninja-build \
    wget \
    unzip \
    libomp-dev \
    dpkg-dev \
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

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/opt/libtorch/lib:${LD_LIBRARY_PATH}

# The Linux CUDA 13 archives use the CXX11 ABI without the old filename prefix.
RUN bash -c 'source /opt/cuda-profile.sh && wget -q "${LIBTORCH_URL}" -O /opt/libtorch.zip' && \
    cd /opt && \
    unzip -q libtorch.zip && \
    rm libtorch.zip

# Copy project source
WORKDIR /src
COPY . .

# Configure & build
RUN bash -c 'source /opt/cuda-profile.sh && \
    cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/libtorch \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    && cmake --build build --parallel ${JOBS}'

# ---------------------------------------------------------------------------
# Stage 2 – package
# ---------------------------------------------------------------------------
# Stay in the same image; run cmake --install into the deb staging tree, then
# produce the .deb with dpkg-deb.

# Directories
ENV PKG_NAME=libminkowski-cpp-dev
ENV STAGE=/deb-stage
ENV DEB_ROOT=${STAGE}/${PKG_NAME}_${VERSION}_amd64

# cmake --install into the staging tree
RUN cmake --install build --prefix ${DEB_ROOT}/usr/local

# Flatten the static archive so all symbols are in one thin ar member
# (prevents "archive has no index" errors for some linkers)
RUN ld -r --whole-archive ${DEB_ROOT}/usr/local/lib/libminkowski_cpp.a \
           --no-whole-archive \
           -o /tmp/mink_merged.o \
    && rm ${DEB_ROOT}/usr/local/lib/libminkowski_cpp.a \
    && ar rcs ${DEB_ROOT}/usr/local/lib/libminkowski_cpp.a /tmp/mink_merged.o \
    && rm /tmp/mink_merged.o

# Write DEBIAN/control
RUN INSTALLED_SIZE=$(du -sk ${DEB_ROOT}/usr | awk '{print $1}') && \
    mkdir -p ${DEB_ROOT}/DEBIAN && \
    cat > ${DEB_ROOT}/DEBIAN/control <<EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: libdevel
Priority: optional
Architecture: amd64
Installed-Size: ${INSTALLED_SIZE}
Depends: libgomp1, libstdc++6
Recommends: nvidia-cuda-toolkit
Maintainer: Minkowski4CPP <https://github.com/acaicia/minkowski4cpp>
Description: MinkowskiEngine C++ development library
 Built for CUDA cu${CUDA_VERSION}.
 A C++ port of NVIDIA's MinkowskiEngine providing sparse tensor operations
 with full CUDA acceleration.  Installs the static library, public headers,
 and CMake package config files so that downstream projects can consume the
 library via find_package(MinkowskiEngine CONFIG).
 .
 LibTorch is NOT bundled.  Install it separately and point CMake at it:
   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
EOF

# Write DEBIAN/postinst (update linker cache)
RUN cat > ${DEB_ROOT}/DEBIAN/postinst <<'EOF'
#!/bin/sh
set -e
ldconfig
EOF
RUN chmod 0755 ${DEB_ROOT}/DEBIAN/postinst

# Set correct permissions on all installed files
RUN find ${DEB_ROOT} -type d -exec chmod 755 {} \; && \
    find ${DEB_ROOT} -type f ! -name postinst -exec chmod 644 {} \;

# Build the .deb
RUN mkdir -p /output && \
    dpkg-deb --build --root-owner-group ${DEB_ROOT} \
        /output/${PKG_NAME}_${VERSION}_cu${CUDA_VERSION}_amd64.deb

# ---------------------------------------------------------------------------
# Final stage — tiny image that holds only the .deb
# ---------------------------------------------------------------------------
FROM scratch AS artifact
COPY --from=builder /output /
