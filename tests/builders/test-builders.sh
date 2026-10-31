#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_DIR="$(mktemp -d)"
trap 'rm -rf "${TEST_DIR}"' EXIT
export DOCKER_TEST_LOG="${TEST_DIR}/docker.log"

if [[ "${CHECK_CUDA:-0}" == 1 ]]; then
    nvcc -std=c++20 -arch=sm_86 -I "${PROJECT_ROOT}/MinkowskiEngine/src/3rdparty" \
        -c "${PROJECT_ROOT}/tests/builders/test_nvtx.cu" -o "${TEST_DIR}/test_nvtx.o"
fi

docker() {
    printf '%s\n' "$@" >> "${DOCKER_TEST_LOG}"
}
export -f docker

for cuda in 124 132 134; do
    CUDA_VERSION="${cuda}"
    CUDA_ARCHITECTURES='86;89;90'
    unset CMAKE_VERSION
    source "${PROJECT_ROOT}/docker/cuda-profile.sh"
    [[ "${TORCH_CUDA_ARCH_LIST}" == '8.6;8.9;9.0' ]]
    case "${cuda}" in
        124) [[ "${CUDA_RELEASE}" == 12.4.0 && "${UBUNTU_VERSION}" == 22.04 && "${TORCH_VERSION}" == 2.5.1 && "${CMAKE_VERSION}" == 3.28.6 ]] ;;
        132) [[ "${CUDA_RELEASE}" == 13.2.1 && "${UBUNTU_VERSION}" == 24.04 && "${TORCH_VERSION}" == 2.15.0 && "${CMAKE_VERSION}" == 4.2.3 ]] ;;
        134) [[ "${CUDA_RELEASE}" == 13.4.3 && "${UBUNTU_VERSION}" == 24.04 && "${TORCH_VERSION}" == 2.15.0 && "${CMAKE_VERSION}" == 4.2.3 ]] ;;
    esac
    if [[ "${CHECK_DOWNLOADS:-0}" == 1 ]]; then
        wget -q --spider "${LIBTORCH_URL}"
    fi
    : > "${DOCKER_TEST_LOG}"
    bash "${PROJECT_ROOT}/docker/docker-build.sh" --cuda-version "${cuda}" --repo test --user test --push --jobs 2 --cuda-architectures '86;89'
    grep -Fxq "CUDA_VERSION=${cuda}" "${DOCKER_TEST_LOG}"
    grep -Fxq 'CUDA_ARCHITECTURES=86;89' "${DOCKER_TEST_LOG}"
    grep -Fxq 'JOBS=2' "${DOCKER_TEST_LOG}"
    grep -Fxq 'linux/amd64' "${DOCKER_TEST_LOG}"
    grep -Fxq "test/test:cu${cuda}" "${DOCKER_TEST_LOG}"
    if [[ "${cuda}" == 134 ]]; then
        grep -Fxq 'test/test:latest' "${DOCKER_TEST_LOG}"
    else
        ! grep -Fxq 'test/test:latest' "${DOCKER_TEST_LOG}"
    fi
done

: > "${DOCKER_TEST_LOG}"
bash "${PROJECT_ROOT}/docker/docker-build.sh" --repo test --user test
grep -Fxq 'CUDA_VERSION=134' "${DOCKER_TEST_LOG}"
grep -Fxq 'test/test:cu134' "${DOCKER_TEST_LOG}"

if bash "${PROJECT_ROOT}/docker/docker-build.sh" --cuda-version 999; then
    echo "ERROR: Invalid CUDA profile was accepted." >&2
    exit 1
fi
if bash "${PROJECT_ROOT}/docker/docker-build.sh" --cuda-version 134 --cuda-architectures 60; then
    echo "ERROR: CUDA 13 accepted a removed GPU architecture." >&2
    exit 1
fi
if bash "${PROJECT_ROOT}/apt/build-apt.sh" --libtorch /unused --cuda-version 999; then
    echo "ERROR: Native builder accepted an invalid CUDA profile." >&2
    exit 1
fi

mkdir -p "${TEST_DIR}/bin" "${TEST_DIR}/libtorch"
export NATIVE_TEST_LOG="${TEST_DIR}/native.log"
cat > "${TEST_DIR}/bin/mock-tool" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
tool="$(basename "$0")"
printf '%s\n' "${tool}" "$@" >> "${NATIVE_TEST_LOG}"
case "${tool}" in
    nvcc) echo "Cuda compilation tools, release ${MOCK_CUDA_RELEASE}, V${MOCK_CUDA_RELEASE}.0" ;;
    cmake)
        if [[ "$1" == --install ]]; then
            mkdir -p "$4/lib"
            touch "$4/lib/libminkowski_cpp.a"
        fi
        ;;
    ld) touch "${@: -1}" ;;
    ar) touch "$2" ;;
    dpkg-deb) touch "${@: -1}" ;;
esac
EOF
chmod +x "${TEST_DIR}/bin/mock-tool"
for tool in nvcc cmake ninja ld ar dpkg-deb; do
    cp "${TEST_DIR}/bin/mock-tool" "${TEST_DIR}/bin/${tool}"
done
export PATH="${TEST_DIR}/bin:${PATH}"

for cuda in 124 132 134; do
    case "${cuda}" in
        124) export MOCK_CUDA_RELEASE=12.4 ;;
        132) export MOCK_CUDA_RELEASE=13.2 ;;
        134) export MOCK_CUDA_RELEASE=13.4 ;;
    esac
    : > "${NATIVE_TEST_LOG}"
    # An explicit build directory keeps test cleanup confined to the fixture.
    bash "${PROJECT_ROOT}/apt/build-apt.sh" --libtorch "${TEST_DIR}/libtorch" \
        --cuda-version "${cuda}" --cuda-architectures '86;89' \
        --build-dir "${TEST_DIR}/build-cu${cuda}" --output "${TEST_DIR}/output" --jobs 2
    grep -Fxq -- '-DCPU_ONLY=OFF' "${NATIVE_TEST_LOG}"
    grep -Fxq -- '-DCMAKE_CUDA_ARCHITECTURES=86;89' "${NATIVE_TEST_LOG}"
    [[ -f "${TEST_DIR}/output/libminkowski-cpp-dev_1.0.0_cu${cuda}_amd64.deb" ]]
done
export MOCK_CUDA_RELEASE=13.4
bash "${PROJECT_ROOT}/apt/build-apt.sh" --libtorch "${TEST_DIR}/libtorch" \
    --build-dir "${TEST_DIR}/build-detected" --output "${TEST_DIR}/detected" --jobs 2
[[ -f "${TEST_DIR}/detected/libminkowski-cpp-dev_1.0.0_cu134_amd64.deb" ]]

export MOCK_CUDA_RELEASE=12.4
if bash "${PROJECT_ROOT}/apt/build-apt.sh" --libtorch "${TEST_DIR}/libtorch" --cuda-version 134; then
    echo "ERROR: Native builder accepted a mismatched CUDA toolkit." >&2
    exit 1
fi

command() {
    if [[ "$*" == '-v nvcc' ]]; then return 1; fi
    builtin command "$@"
}
export -f command
: > "${NATIVE_TEST_LOG}"
bash "${PROJECT_ROOT}/apt/build-apt.sh" --libtorch "${TEST_DIR}/libtorch" \
    --build-dir "${TEST_DIR}/build-cpu" --output "${TEST_DIR}/cpu" --jobs 2
grep -Fxq -- '-DCPU_ONLY=ON' "${NATIVE_TEST_LOG}"
[[ -f "${TEST_DIR}/cpu/libminkowski-cpp-dev_1.0.0_cpu_amd64.deb" ]]
if bash "${PROJECT_ROOT}/apt/build-apt.sh" --libtorch "${TEST_DIR}/libtorch" --cuda-version 134; then
    echo "ERROR: Native builder accepted an explicit profile without nvcc." >&2
    exit 1
fi
unset -f command

if [[ "${CHECK_DOWNLOADS:-0}" == 1 ]]; then
    wget -q --spider https://github.com/Kitware/CMake/releases/download/v4.2.3/cmake-4.2.3-linux-x86_64.tar.gz
    wget -q --spider https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-linux-x86_64.tar.gz
fi
echo "Bash builder tests passed."
