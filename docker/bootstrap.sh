#!/bin/bash
set -e

cd /workspace/OAT-RoboMimic-Fine-tune

python -m pip install --upgrade pip
python -m pip install -e .

echo "Bootstrap completed."
echo "If LIBERO submodule is needed, run:"
echo "  git submodule update --init --recursive"
