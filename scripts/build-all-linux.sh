#!/usr/bin/env bash
# MinkowskiEngine C++ - Complete Linux Build Script
# Configures, builds, and optionally installs in one command
set -euo pipefail

# ============================================================================
# Defaults
# ============================================================================
LIBTORCH_PATH="${LIBTORCH_PATH:-/opt/libtorch}"
BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
INSTALL_PREFIX="${INSTALL_PREFIX:-}"
JOBS="${JOBS:-$(nproc)}"
CPU_ONLY="${CPU_ONLY:-OFF}"
CLEAN="${CLEAN:-0}"
INSTALL="${INSTALL:-0}"
VERBOSE="${VERBOSE:-0}"

# ============================================================================
# Usage
# ============================================================================
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --libtorch PATH       Path to LibTorch installation (default: \$LIBTORCH_PATH or /opt/libtorch)
  --build-dir DIR       Build directory (default: build)
  --build-type TYPE     Release|Debug|RelWithDebInfo (default: Release)
  --install-prefix DIR  Installation prefix (default: system default)
  --jobs N              Parallel build jobs (default: nproc)
  --cpu-only            Build without CUDA support
  --clean               Remove build directory before configuring
  --install             Run install step after building
  --verbose             Verbose build output
  -h, --help            Show this help message

Environment variables:
  LIBTORCH_PATH         Path to LibTorch (overridden by --libtorch)
  CUDA_PATH             Path to CUDA toolkit (auto-detected if not set)
  BUILD_DIR, BUILD_TYPE, INSTALL_PREFIX, JOBS, CPU_ONLY, CLEAN, INSTALL, VERBOSE
EOF
    exit 0
}

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --libtorch)      LIBTORCH_PATH="$2"; shift 2 ;;
        --build-dir)     BUILD_DIR="$2"; shift 2 ;;
        --build-type)    BUILD_TYPE="$2"; shift 2 ;;
        --install-prefix) INSTALL_PREFIX="$2"; shift 2 ;;
        --jobs)          JOBS="$2"; shift 2 ;;
        --cpu-only)      CPU_ONLY="ON"; shift ;;
        --clean)         CLEAN=1; shift ;;
        --install)       INSTALL=1; shift ;;
        --verbose)       VERBOSE=1; shift ;;
        -h|--help)       usage ;;
        *)               echo "Unknown option: $1"; usage ;;
    esac
done

# Resolve script and project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "========================================"
echo " MinkowskiEngine C++ - Complete Linux Build"
echo "========================================"
echo ""
echo "  Project dir   : ${PROJECT_DIR}"
echo "  LibTorch      : ${LIBTORCH_PATH}"
echo "  Build dir     : ${BUILD_DIR}"
echo "  Build type    : ${BUILD_TYPE}"
echo "  CPU only      : ${CPU_ONLY}"
echo "  Jobs          : ${JOBS}"
echo "  Install       : ${INSTALL}"
echo ""

# ============================================================================
# Validate LibTorch
# ============================================================================
if [[ ! -d "${LIBTORCH_PATH}" ]]; then
    echo "ERROR: LibTorch not found at ${LIBTORCH_PATH}"
    echo "Set LIBTORCH_PATH or pass --libtorch <path>"
    exit 1
fi

# ============================================================================
# STEP 1: CONFIGURE
# ============================================================================
echo "========================================"
echo " STEP 1: Configure"
echo "========================================"
echo ""

cd "${PROJECT_DIR}"

if [[ "${CLEAN}" -eq 1 ]] && [[ -d "${BUILD_DIR}" ]]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DCMAKE_PREFIX_PATH="${LIBTORCH_PATH}"
    -DCPU_ONLY="${CPU_ONLY}"
)

if [[ -n "${INSTALL_PREFIX}" ]]; then
    CMAKE_ARGS+=(-DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}")
fi

# Use Ninja if available, otherwise Unix Makefiles
if command -v ninja &>/dev/null; then
    CMAKE_ARGS+=(-G Ninja)
    echo "Using Ninja generator"
else
    echo "Using Unix Makefiles generator"
fi

echo "cmake ${CMAKE_ARGS[*]} -S . -B ${BUILD_DIR}"
echo ""
cmake "${CMAKE_ARGS[@]}" -S . -B "${BUILD_DIR}"

echo ""
echo "[OK] Configuration successful"
echo ""

# ============================================================================
# STEP 2: BUILD
# ============================================================================
echo "========================================"
echo " STEP 2: Build"
echo "========================================"
echo ""

BUILD_ARGS=(--build "${BUILD_DIR}" -- -j"${JOBS}")

if [[ "${VERBOSE}" -eq 1 ]]; then
    BUILD_ARGS=(--build "${BUILD_DIR}" --verbose -- -j"${JOBS}")
fi

echo "cmake ${BUILD_ARGS[*]}"
echo ""
cmake "${BUILD_ARGS[@]}"

echo ""
echo "[OK] Build successful"
echo ""

# ============================================================================
# STEP 3: INSTALL (Optional)
# ============================================================================
if [[ "${INSTALL}" -eq 1 ]]; then
    echo "========================================"
    echo " STEP 3: Install"
    echo "========================================"
    echo ""

    INSTALL_ARGS=(--install "${BUILD_DIR}")

    if [[ -n "${INSTALL_PREFIX}" ]]; then
        INSTALL_ARGS+=(--prefix "${INSTALL_PREFIX}")
    fi

    echo "cmake ${INSTALL_ARGS[*]}"
    echo ""
    cmake "${INSTALL_ARGS[@]}"

    echo ""
    echo "[OK] Installation successful"
    if [[ -n "${INSTALL_PREFIX}" ]]; then
        echo "     Installed to: ${INSTALL_PREFIX}"
    fi
    echo ""
fi

# ============================================================================
# COMPLETE
# ============================================================================
echo "========================================"
echo " BUILD COMPLETE!"
echo "========================================"
echo ""

if [[ "${INSTALL}" -eq 0 ]]; then
    echo "To install, run:"
    echo "  cmake --install ${BUILD_DIR} [--prefix <path>]"
    echo ""
    echo "Or re-run with --install:"
    echo "  $0 --install [--install-prefix <path>]"
fi

echo ""
