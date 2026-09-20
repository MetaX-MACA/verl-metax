#!/usr/bin/env bash
# Load the vllm-metax 0.23.0 sleep-mode replacement modules before Python starts.

VLLM_METAX_ENV_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_METAX_ROOT=$(cd -- "${VLLM_METAX_ENV_DIR}/../.." && pwd)
export VLLM_METAX_PATCH_DIR="${VLLM_METAX_PATCH_DIR:-${VERL_METAX_ROOT}/patches/vllm_metax_0_23_0}"

if [[ ! -f "${VLLM_METAX_PATCH_DIR}/sitecustomize.py" ]]; then
    echo "Missing vllm-metax sleep-mode patch: ${VLLM_METAX_PATCH_DIR}/sitecustomize.py" >&2
    return 1 2>/dev/null || exit 1
fi

export PYTHONPATH="${VLLM_METAX_PATCH_DIR}:${PYTHONPATH:-}"

