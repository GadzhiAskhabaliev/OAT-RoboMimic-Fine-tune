#!/usr/bin/env bash
set -euo pipefail

# Two-stage OAT training on RoboMimic Lift:
#   stage1: train tokenizer from scratch
#   stage2: train policy with stage1 tokenizer checkpoint
#
# Example:
#   bash scripts/run_robomimic_lift_two_stage.sh stage1
#   bash scripts/run_robomimic_lift_two_stage.sh stage2 output/2026.../checkpoints/ep-xxxx_mse-xxx.ckpt

STAGE="${1:-}"
TOKENIZER_CKPT="${2:-}"

if [[ -z "${STAGE}" ]]; then
  echo "Usage:"
  echo "  $0 stage1"
  echo "  $0 stage2 <path/to/stage1_tokenizer.ckpt>"
  exit 1
fi

NUM_MACHINES="${NUM_MACHINES:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_DEMO="${NUM_DEMO:-200}"

case "${STAGE}" in
  stage1)
    HYDRA_FULL_ERROR=1 uv run accelerate launch \
      --num_machines "${NUM_MACHINES}" \
      --multi_gpu \
      --num_processes "${NUM_PROCESSES}" \
      scripts/run_workspace.py \
      --config-name=train_oattok \
      task/tokenizer=robomimic/lift \
      training.num_demo="${NUM_DEMO}" \
      training.checkpoint_every=250 \
      checkpoint.topk.k=1 \
      logging.mode=disabled
    ;;
  stage2)
    if [[ -z "${TOKENIZER_CKPT}" ]]; then
      echo "stage2 requires tokenizer checkpoint path"
      echo "Usage: $0 stage2 <path/to/stage1_tokenizer.ckpt>"
      exit 1
    fi
    HYDRA_FULL_ERROR=1 MUJOCO_GL=egl uv run accelerate launch \
      --num_machines "${NUM_MACHINES}" \
      --multi_gpu \
      --num_processes "${NUM_PROCESSES}" \
      scripts/run_workspace.py \
      --config-name=train_oatpolicy \
      task/policy=robomimic/lift \
      policy.action_tokenizer.checkpoint="${TOKENIZER_CKPT}" \
      training.num_demo="${NUM_DEMO}" \
      training.checkpoint_every=250 \
      checkpoint.topk.k=1 \
      logging.mode=disabled
    ;;
  *)
    echo "Unknown stage: ${STAGE}"
    echo "Use stage1 or stage2"
    exit 1
    ;;
esac
