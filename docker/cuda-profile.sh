#!/usr/bin/env bash
# Sourced by container and host builders to keep CUDA/LibTorch pairings aligned.
case "${CUDA_VERSION:-134}" in
    124)
        CUDA_VERSION=124
        CUDA_RELEASE=12.4.0
        UBUNTU_VERSION=22.04
        TORCH_VERSION=2.5.1
        LIBTORCH_ARCHIVE=libtorch-cxx11-abi-shared-with-deps
        CMAKE_VERSION="${CMAKE_VERSION:-3.28.6}"
        ;;
    132|134)
        CUDA_VERSION="${CUDA_VERSION:-134}"
        UBUNTU_VERSION=24.04
        TORCH_VERSION=2.15.0
        LIBTORCH_ARCHIVE=libtorch-shared-with-deps
        CMAKE_VERSION="${CMAKE_VERSION:-4.2.3}"
        if [[ "${CUDA_VERSION}" == 132 ]]; then
            CUDA_RELEASE=13.2.1
        else
            CUDA_RELEASE=13.4.3
        fi
        ;;
    *)
        echo "ERROR: CUDA_VERSION must be 124, 132, or 134." >&2
        return 1
        ;;
esac
LIBTORCH_URL="https://download.pytorch.org/libtorch/cu${CUDA_VERSION}/${LIBTORCH_ARCHIVE}-${TORCH_VERSION}%2Bcu${CUDA_VERSION}.zip"

# Torch replaces CMAKE_CUDA_ARCHITECTURES with its own global nvcc flags.
TORCH_CUDA_ARCH_LIST=""
IFS=';' read -r -a architectures <<< "${CUDA_ARCHITECTURES:-86}"
for architecture in "${architectures[@]}"; do
    if [[ ! "${architecture}" =~ ^([0-9]+)([0-9])(a?)$ ]]; then
        echo "ERROR: CUDA architectures must be semicolon-separated SM numbers (e.g. 86;89;90)." >&2
        return 1
    fi
    major="${BASH_REMATCH[1]}"
    minor="${BASH_REMATCH[2]}"
    suffix="${BASH_REMATCH[3]}"
    if [[ "${CUDA_VERSION}" != 124 ]] && (( 10#${major} * 10 + 10#${minor} < 75 )); then
        echo "ERROR: CUDA 13 requires SM 75 or newer; use CUDA 12.4 for SM ${architecture}." >&2
        return 1
    fi
    TORCH_CUDA_ARCH_LIST+="${TORCH_CUDA_ARCH_LIST:+;}${major}.${minor}${suffix}"
done
export TORCH_CUDA_ARCH_LIST
