#!/usr/bin/env bash
# Shared environment validated for verl-metax 0.9.0 on MACA 3.8.0.
# Site-specific network interfaces and device visibility remain caller-controlled.
set -euo pipefail

export MACA_PATH="${MACA_PATH:-/opt/maca}"
export MACA_CLANG_PATH="${MACA_CLANG_PATH:-${MACA_PATH}/mxgpu_llvm/bin}"
export MACA_CLANG="${MACA_CLANG:-${MACA_PATH}/mxgpu_llvm}"
export DEVINFO_ROOT="${DEVINFO_ROOT:-${MACA_PATH}}"
export CUCC_PATH="${CUCC_PATH:-${MACA_PATH}/tools/cu-bridge}"
export CUDA_PATH="${CUDA_PATH:-${CUCC_PATH}}"
export PATH="${CUCC_PATH}:${MACA_PATH}/bin:${MACA_CLANG}/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"

export MACA_SMALL_PAGESIZE_ENABLE="${MACA_SMALL_PAGESIZE_ENABLE:-1}"
export MACA_MPS_MODE="${MACA_MPS_MODE:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export SET_DEVICE_NUMA_PREFERRED="${SET_DEVICE_NUMA_PREFERRED:-1}"
export MCPYTORCH_DISABLE_PRINT="${MCPYTORCH_DISABLE_PRINT:-1}"
export PYTORCH_ENABLE_SAME_RAND_A100="${PYTORCH_ENABLE_SAME_RAND_A100:-1}"
export MCCL_MAX_NCHANNELS="${MCCL_MAX_NCHANNELS:-16}"
export NVTE_FLASH_ATTN="${NVTE_FLASH_ATTN:-1}"
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export PYTHONUNBUFFERED=1

# The ASM workaround scopes these variables to actor operations.
unset MHA_USE_BLAS
unset MHA_BWD_NO_ATOMIC_F64

# Training actors must let Ray assign one visible accelerator per rank. Rollout
# server actors that need full visibility set this flag in their own runtime_env.
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES

# verl-metax automatically activates its version-gated vllm-metax sleep-mode
# compatibility package. Set VERL_METAX_SLEEP_PATCH=off to disable it.

# Configure these at the deployment layer when needed:
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export GLOO_SOCKET_IFNAME=bond0
# export MCCL_SOCKET_IFNAME=bond0
# export MCCL_IB_HCA=mlx5_0,mlx5_2
