#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/create-release-tarball.sh <version> [staging-dir]
# Creates a tar.gz containing include/, lib/, and lib/cmake/MinkowskiEngine

VERSION=${1:-v1.0.0}
STAGING_DIR=${2:-$(mktemp -d)}
INSTALL_PREFIX=/usr/local

echo "Version: ${VERSION}"
echo "Staging dir: ${STAGING_DIR}"

mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
cmake --build build -- -j

# Use CMake install with --destdir to stage files
cmake --install build --prefix ${INSTALL_PREFIX} --destdir "${STAGING_DIR}"

TARBALL="minkowskiengine-${VERSION}-linux.tar.gz"
tar -czf "${TARBALL}" -C "${STAGING_DIR}" .

echo "Created ${TARBALL}"
echo "Staged contents:"
tar -tzf "${TARBALL}" | sed -n '1,50p'
