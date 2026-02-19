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
TAG="latest"
PUSH=false

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --push)   PUSH=true; shift ;;
        --tag)    TAG="$2"; shift 2 ;;
        --user)   DOCKER_USER="$2"; shift 2 ;;
        --repo)   DOCKER_REPO="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--push] [--tag TAG] [--user DOCKER_USER] [--repo DOCKER_REPO]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

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
echo "============================================"

# ── Build ─────────────────────────────────────────────────────────────────────
docker build \
    -t "${IMAGE_NAME}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${PROJECT_ROOT}"

echo ""
echo "Successfully built: ${IMAGE_NAME}"

# ── Push (optional) ──────────────────────────────────────────────────────────
if [[ "${PUSH}" == true ]]; then
    echo "Pushing ${IMAGE_NAME} to Docker Hub..."
    docker push "${IMAGE_NAME}"

    # If a version tag was given, also tag and push as 'latest'
    if [[ "${TAG}" != "latest" ]]; then
        LATEST_NAME="${IMAGE_NAME%%:*}:latest"
        docker tag "${IMAGE_NAME}" "${LATEST_NAME}"
        docker push "${LATEST_NAME}"
        echo "Also pushed: ${LATEST_NAME}"
    fi

    echo "Push complete."
fi
