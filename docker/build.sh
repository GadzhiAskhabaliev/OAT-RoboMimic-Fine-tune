#!/bin/bash
set -e

ARCH=$(uname -m)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${ARCH}/oat-robomimic:latest"

docker build "${PROJECT_ROOT}" \
    -f "${SCRIPT_DIR}/Dockerfile.${ARCH}" \
    --build-arg USER="${USER}" \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -t "${IMAGE_NAME}"
