#!/bin/bash
set -e

CONTAINER_NAME=oat_robomimic_${USER}

if [ -z "$(docker ps -q -f "name=^${CONTAINER_NAME}$")" ]; then
    echo "Container is not running. Start it first:"
    echo "bash docker/start.sh"
    exit 1
fi

docker exec -it "${CONTAINER_NAME}" bash
