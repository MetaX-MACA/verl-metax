#!/usr/bin/env bash
# Shared Megatron environment validated for verl-metax 0.9.0, MACA 3.8.0,
# and Megatron-LM 0.18.0. Override MEGATRON_PATH for compatible dev checkouts.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/metax.sh"

export MEGATRON_PATH="${MEGATRON_PATH:-/workspace/Megatron-LM-0.18.0}"
export PYTHONPATH="${MEGATRON_PATH}:${PYTHONPATH:-}"
