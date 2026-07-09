#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash script/dw/train_dw05.sh <nproc_per_node> [dataclass_overrides...]}"
shift

if [[ -n "${DW05_MODEL_BASE_PATH:-}" && -z "${DIFFSYNTH_MODEL_BASE_PATH:-}" ]]; then
  export DIFFSYNTH_MODEL_BASE_PATH="${DW05_MODEL_BASE_PATH}"
fi
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"

accelerate launch \
  --config_file script/dw/accelerate_zero1_ds.yaml \
  --num_processes "${NPROC_PER_NODE}" \
  playground/example_dw_exp.py \
  --task train \
  "$@"
