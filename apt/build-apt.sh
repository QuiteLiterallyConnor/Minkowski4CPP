#!/usr/bin/env bash
# =============================================================================
# MinkowskiEngine C++ — Debian Package Builder (native Linux)
# =============================================================================
#
# Builds libminkowski-cpp-dev_<VERSION>_amd64.deb directly on a Linux host
# without Docker.  Requires a working CUDA toolkit, CMake, Ninja, and a
# LibTorch installation.
#
# Prerequisites
# -------------
#   - CUDA Toolkit 12.4+ (nvcc on PATH)
#   - LibTorch 2.5.1 (CXX11 ABI, CUDA 12.4)
#   - cmake >= 3.18, ninja-build, build-essential, libomp-dev, dpkg-dev
#
# Usage
# -----
#   ./apt/build-apt.sh --libtorch /opt/libtorch
#   ./apt/build-apt.sh --libtorch /opt/libtorch --version 1.2.0 --jobs 8
#
# All flags and their defaults:
#   --libtorch  PATH    Path to LibTorch root (required)
#   --version   STRING  Package version          (default: 1.0.0)
#   --jobs      N       Parallel ninja jobs       (default: nproc)
#   --build-dir PATH    CMake build directory     (default: <project>/build-deb)
#   --output    PATH    Directory for .deb output (default: <script dir>)
#   --keep-build        Keep the CMake build tree after packaging
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
VERSION="1.0.0"
JOBS="$(nproc)"
LIBTORCH_PATH=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR=""
OUTPUT_DIR="${SCRIPT_DIR}"
KEEP_BUILD=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --libtorch)   LIBTORCH_PATH="$2"; shift 2 ;;
        --version)    VERSION="$2";       shift 2 ;;
        --jobs)       JOBS="$2";          shift 2 ;;
        --build-dir)  BUILD_DIR="$2";     shift 2 ;;
        --output)     OUTPUT_DIR="$2";    shift 2 ;;
        --keep-build) KEEP_BUILD=1;       shift   ;;
        -h|--help)
            sed -n '/^# Usage/,/^# ====/p' "$0" | grep -v '^# ====' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1
            ;;
    esac
done

if [[ -z "${LIBTORCH_PATH}" ]]; then
    echo "ERROR: --libtorch <path> is required." >&2
    echo "Example: $0 --libtorch /opt/libtorch" >&2
    exit 1
fi

if [[ -z "${BUILD_DIR}" ]]; then
    BUILD_DIR="${PROJECT_ROOT}/build-deb"
fi

PKG_NAME="libminkowski-cpp-dev"
DEB_FILENAME="${PKG_NAME}_${VERSION}_amd64.deb"
STAGE_DIR="$(mktemp -d /tmp/minkowski-deb-XXXXXX)"
DEB_ROOT="${STAGE_DIR}/${PKG_NAME}_${VERSION}_amd64"

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo -e "\033[0;36m$*\033[0m"; }
ok()    { echo -e "\033[0;32m  $*\033[0m"; }
warn()  { echo -e "\033[0;33m  $*\033[0m"; }
error() { echo -e "\033[0;31mERROR: $*\033[0m" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
echo ""
info "================================================================"
info " MinkowskiEngine C++ - Debian Package Builder (native Linux)"
info "================================================================"
echo "  Version      : ${VERSION}"
echo "  Jobs         : ${JOBS}"
echo "  Project root : ${PROJECT_ROOT}"
echo "  LibTorch     : ${LIBTORCH_PATH}"
echo "  Build dir    : ${BUILD_DIR}"
echo "  Output dir   : ${OUTPUT_DIR}"
echo "  Output file  : ${DEB_FILENAME}"
info "================================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1 – Check prerequisites
# ---------------------------------------------------------------------------
info "[1/4] Checking prerequisites..."

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        error "'$1' not found on PATH. Install it and try again. ($2)"
    fi
    ok "$1 OK"
}

check_cmd cmake    "sudo apt install cmake"
check_cmd ninja    "sudo apt install ninja-build"
check_cmd dpkg-deb "sudo apt install dpkg-dev"
check_cmd ar       "sudo apt install binutils"
check_cmd ld       "sudo apt install binutils"

if [[ ! -d "${LIBTORCH_PATH}" ]]; then
    error "LibTorch directory not found: ${LIBTORCH_PATH}"
fi
ok "LibTorch found at ${LIBTORCH_PATH}"

if ! nvcc --version &>/dev/null; then
    warn "nvcc not on PATH — building without CUDA (CPU_ONLY=ON)."
    warn "Add CUDA bin dir to PATH for a CUDA-enabled build."
    CPU_ONLY=ON
else
    ok "nvcc OK ($(nvcc --version | grep 'release' | awk '{print $6}'))"
    CPU_ONLY=OFF
fi

# ---------------------------------------------------------------------------
# Step 2 – Configure & build
# ---------------------------------------------------------------------------
info ""
info "[2/4] Configuring and building..."

CMAKE_ARGS=(
    -B "${BUILD_DIR}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_PREFIX_PATH="${LIBTORCH_PATH}"
    -DCMAKE_INSTALL_PREFIX="${DEB_ROOT}/usr/local"
)
if [[ "${CPU_ONLY}" == "ON" ]]; then
    CMAKE_ARGS+=(-DCPU_ONLY=ON)
fi

cmake "${CMAKE_ARGS[@]}" "${PROJECT_ROOT}"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

ok "Build complete."

# ---------------------------------------------------------------------------
# Step 3 – cmake --install into staging tree
# ---------------------------------------------------------------------------
info ""
info "[3/4] Installing into staging tree..."

cmake --install "${BUILD_DIR}" --prefix "${DEB_ROOT}/usr/local"

# Flatten the static archive so all symbols are in a single ar member,
# preventing "archive has no index" errors with some linkers.
ARCHIVE="${DEB_ROOT}/usr/local/lib/libminkowski_cpp.a"
if [[ -f "${ARCHIVE}" ]]; then
    info "  Flattening static archive..."
    ld -r --whole-archive "${ARCHIVE}" \
          --no-whole-archive \
          -o /tmp/mink_merged.o
    rm "${ARCHIVE}"
    ar rcs "${ARCHIVE}" /tmp/mink_merged.o
    rm /tmp/mink_merged.o
    ok "Archive flattened."
fi

# ---------------------------------------------------------------------------
# Step 4 – Write DEBIAN metadata and build .deb
# ---------------------------------------------------------------------------
info ""
info "[4/4] Assembling .deb package..."

mkdir -p "${DEB_ROOT}/DEBIAN"

INSTALLED_SIZE=$(du -sk "${DEB_ROOT}/usr" | awk '{print $1}')

cat > "${DEB_ROOT}/DEBIAN/control" <<EOF
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

cat > "${DEB_ROOT}/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e
ldconfig
EOF
chmod 0755 "${DEB_ROOT}/DEBIAN/postinst"

# Fix permissions: directories 755, files 644 (postinst already 755)
find "${DEB_ROOT}" -type d -exec chmod 755 {} \;
find "${DEB_ROOT}" -type f ! -name postinst -exec chmod 644 {} \;

OUTPUT_DEB="${OUTPUT_DIR}/${DEB_FILENAME}"
dpkg-deb --build --root-owner-group "${DEB_ROOT}" "${OUTPUT_DEB}"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
rm -rf "${STAGE_DIR}"

if [[ "${KEEP_BUILD}" -eq 0 ]]; then
    rm -rf "${BUILD_DIR}"
fi

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
SIZE_MB=$(du -sh "${OUTPUT_DEB}" | awk '{print $1}')
echo ""
info "================================================================"
info " Build complete!"
info "================================================================"
echo "  Package : ${DEB_FILENAME}"
echo "  Size    : ${SIZE_MB}"
echo "  Path    : ${OUTPUT_DEB}"
echo ""
echo "  To install:"
echo "    sudo dpkg -i ${DEB_FILENAME}"
echo "    sudo ldconfig"
info "================================================================"
