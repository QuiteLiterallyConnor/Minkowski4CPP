#!/usr/bin/env bash
set -euo pipefail

USER=${1:-yourhubuser}
TAG=${2:-latest}

IMAGE_NAME=${USER}/minkowskiengine:${TAG}

echo "Building ${IMAGE_NAME}..."
docker build -t ${IMAGE_NAME} -f docker/Dockerfile .

echo "Pushing ${IMAGE_NAME} to Docker Hub..."
docker push ${IMAGE_NAME}

echo "Done."
