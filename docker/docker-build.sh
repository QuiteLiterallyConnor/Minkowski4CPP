#!/usr/bin/env bash
#
# Build and push the Minkowski4CPP Docker image to Docker Hub.
#
# Usage:
#   ./docker-build.sh                       # build only
#   ./docker-build.sh --push                # build and push
#   ./docker-build.sh --push --tag 1.0.0    # build and push with custom tag
#
# Environment variables (override defaults):
#   DOCKER_REPO   – Docker Hub repository  (default: minkowski4cpp)
#   DOCKER_USER   – Docker Hub username     (default: read from `docker info`)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
DOCKER_REPO="${DOCKER_REPO:-minkowski4cpp}"
DOCKER_USER="${DOCKER_USER:-}"
TAG=""
PUSH=false
CUDA_VERSION=134
CUDA_ARCHITECTURES=86
JOBS=4

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --push)   PUSH=true; shift ;;
        --tag)    TAG="$2"; shift 2 ;;
        --user)   DOCKER_USER="$2"; shift 2 ;;
        --repo)   DOCKER_REPO="$2"; shift 2 ;;
        --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
        --cuda-architectures) CUDA_ARCHITECTURES="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--push] [--tag TAG] [--user DOCKER_USER] [--repo DOCKER_REPO] [--cuda-version 124|132|134] [--cuda-architectures LIST] [--jobs N]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
source "${SCRIPT_DIR}/cuda-profile.sh"
TAG="${TAG:-cu${CUDA_VERSION}}"
if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --jobs must be a positive integer." >&2
    exit 1
fi

# ── Resolve image name ───────────────────────────────────────────────────────
if [[ -z "${DOCKER_USER}" ]]; then
    IMAGE_NAME="${DOCKER_REPO}:${TAG}"
else
    IMAGE_NAME="${DOCKER_USER}/${DOCKER_REPO}:${TAG}"
fi

echo "============================================"
echo " Minkowski4CPP Docker Build"
echo "============================================"
echo " Image:   ${IMAGE_NAME}"
echo " Context: ${PROJECT_ROOT}"
echo " Push:    ${PUSH}"
echo " CUDA:    cu${CUDA_VERSION}"
echo "============================================"

# ── Build ─────────────────────────────────────────────────────────────────────
docker build \
    -t "${IMAGE_NAME}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    --platform linux/amd64 \
    --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
    --build-arg "CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    --build-arg "JOBS=${JOBS}" \
    "${PROJECT_ROOT}"

echo ""
echo "Successfully built: ${IMAGE_NAME}"

# ── Push (optional) ──────────────────────────────────────────────────────────
if [[ "${PUSH}" == true ]]; then
    echo "Pushing ${IMAGE_NAME} to Docker Hub..."
    docker push "${IMAGE_NAME}"

    # If a version tag was given, also tag and push as 'latest'
    if [[ "${CUDA_VERSION}" == 134 && "${TAG}" != "latest" ]]; then
        LATEST_NAME="${IMAGE_NAME%:*}:latest"
        docker tag "${IMAGE_NAME}" "${LATEST_NAME}"
        docker push "${LATEST_NAME}"
        echo "Also pushed: ${LATEST_NAME}"
    fi

    echo "Push complete."
fi
