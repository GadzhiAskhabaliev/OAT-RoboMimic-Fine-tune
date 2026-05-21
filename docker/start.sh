#!/bin/bash
set -e

ARCH=$(uname -m)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_NAME=oat_robomimic_${USER}
IMAGE=${ARCH}/oat-robomimic:latest

if [ -n "$(docker ps -q -f "name=^${CONTAINER_NAME}$")" ]; then
    echo "Container ${CONTAINER_NAME} is already running."
elif [ -n "$(docker ps -aq -f "name=^${CONTAINER_NAME}$")" ]; then
    echo "Starting existing container ${CONTAINER_NAME}."
    docker start "${CONTAINER_NAME}"
else
    echo "Creating new container ${CONTAINER_NAME}."
    docker run -it -d \
        --name "${CONTAINER_NAME}" \
        --gpus all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
        -e MUJOCO_GL=egl \
        --ipc host \
        -v "${PROJECT_ROOT}":/workspace/OAT-RoboMimic-Fine-tune \
        -w /workspace/OAT-RoboMimic-Fine-tune \
        "${IMAGE}"
fi
