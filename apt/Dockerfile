# =============================================================================
# MinkowskiEngine C++ — Debian Package Builder
# =============================================================================
#
# Builds libminkowski-cpp-dev_<VERSION>_amd64.deb inside a Ubuntu 22.04 +
# CUDA 12.4 environment, then drops the finished .deb at /output/ so the
# host script can copy it out.
#
# Build args
# ----------
#   VERSION     Semver string embedded in the package (default: 1.0.0)
#   JOBS        Parallel ninja jobs (default: 4)
#
# =============================================================================

ARG VERSION=1.0.0
ARG JOBS=4

# ---------------------------------------------------------------------------
# Stage 1 – builder
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

# Re-declare ARGs inside each stage so they are accessible in RUN commands.
# Promote VERSION to ENV so it can be referenced in later ENV and RUN lines.
ARG VERSION
ARG JOBS
ENV VERSION=${VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    wget \
    unzip \
    libomp-dev \
    dpkg-dev \
    && rm -rf /var/lib/apt/lists/*

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/opt/libtorch/lib:${LD_LIBRARY_PATH}

# Download LibTorch (CUDA 12.4, CXX11 ABI)
RUN cd /opt && \
    wget -q "https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip" \
         -O libtorch.zip && \
    unzip -q libtorch.zip && \
    rm libtorch.zip

# Copy project source
WORKDIR /src
COPY . .

# Configure & build
RUN cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/libtorch \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build build --parallel ${JOBS}

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
        /output/${PKG_NAME}_${VERSION}_amd64.deb

# ---------------------------------------------------------------------------
# Final stage — tiny image that holds only the .deb
# ---------------------------------------------------------------------------
FROM scratch AS artifact
COPY --from=builder /output /
