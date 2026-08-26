#!/usr/bin/env bash
# Actor-only Megatron-Native MoE smoke (no Ray DAPO, no rollout).
# Usage:
#   bash run_actor_smoke.sh alltoall
#   bash run_actor_smoke.sh mori
set -euo pipefail

MODE="${1:-alltoall}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_ROOT="${RL_ROOT:-/workspace/Lumen-RL}"
MEGATRON_PATH="${MEGATRON_PATH:-/workspace/Megatron_17/Megatron-LM}"
TE_PATH="${TE_PATH:-/workspace/TransformerEngine}"
LUMEN_PATH="${LUMEN_PATH:-/workspace/Lumen}"
AITER_PATH="${AITER_PATH:-/workspace/Lumen/third_party/aiter}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
MASTER_PORT="${MASTER_PORT:-29569}"

export PYTHONPATH="${MEGATRON_PATH}:${TE_PATH}:${RL_ROOT}:${LUMEN_PATH}:${AITER_PATH}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT=1800
export LUMENRL_MEM_DIAG="${LUMENRL_MEM_DIAG:-1}"
export TORCHDYNAMO_DISABLE=1
unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN

# RCCL/NCCL baseline matching Megatron examples/qwen3/train_qwen3.sh.
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
export NCCL_CHECKS_DISABLE="${NCCL_CHECKS_DISABLE:-1}"
# NCCL_PROTO left to RCCL auto-selection: the 600 MB ddp-bucket-size + comm
# overlap below already eliminate the pathological coalesced param all-gather
# (verified: forcing Simple gives the same ~10 ms gather and 2.63 s/step). Set
# NCCL_PROTO explicitly to override.
if [ -n "${NCCL_PROTO:-}" ] && [ "${NCCL_PROTO}" != "none" ]; then
  export NCCL_PROTO
else
  unset NCCL_PROTO
fi
export RCCL_MSCCL_ENABLE="${RCCL_MSCCL_ENABLE:-0}"
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"

# Distributed-optimizer comm overlap + 600 MB buckets, as in train_qwen3.sh.
export DDP_BUCKET_SIZE="${DDP_BUCKET_SIZE:-629145600}"
export OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-1}"
export OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-1}"
# Balanced routing (train_qwen3.sh uses aux_loss 1e-3) keeps expert load even so
# dispatch/combine don't stall on stragglers.
export MOE_AUX_LOSS="${MOE_AUX_LOSS:-1e-3}"
# Fair A2A-vs-MORI: keep A2A on the same 8-visible topology as MORI.
export KEEP_ALL_GPUS="${KEEP_ALL_GPUS:-1}"

export MODEL_PATH="${MODEL_PATH:-${DATA_ROOT}/models/Qwen3-30B-A3B-Base}"
# Example 7 (dapo_qwen3moe_a3b_ray_megatron_verlref_longrun): prompt 2048,
# pack budget 8192 tokens/GPU (not the 22528 worst-case singleton row).
export N_STEPS="${N_STEPS:-20}"
export PROFILE_STEP="${PROFILE_STEP:-12}"
export SEQ_LEN="${SEQ_LEN:-8192}"
export N_ROWS="${N_ROWS:-1}"
export PROMPT_LEN="${PROMPT_LEN:-2048}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"

case "${MODE}" in
  alltoall|a2a)
    export MOE_DISPATCHER=alltoall
    export TRACE_TAG="${TRACE_TAG:-a2a}"
    export PROFILE_DIR="${PROFILE_DIR:-${DATA_ROOT}/logs/actor_profile_a2a}"
    unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
    ;;
  mori|flex)
    export MOE_DISPATCHER=flex
    export MOE_FLEX_BACKEND="${MOE_FLEX_BACKEND:-mori}"
    export TRACE_TAG="${TRACE_TAG:-mori}"
    export PROFILE_DIR="${PROFILE_DIR:-${DATA_ROOT}/logs/actor_profile_mori}"
    export MORI_SHMEM_LOG_LEVEL="${MORI_SHMEM_LOG_LEVEL:-INFO}"
    export MORI_GPU_ARCHS="${MORI_GPU_ARCHS:-gfx942}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
    ;;
  *)
    echo "usage: $0 {alltoall|mori}" >&2
    exit 2
    ;;
esac

mkdir -p "${PROFILE_DIR}"
cd "${RL_ROOT}"
echo "dispatcher=${MOE_DISPATCHER} profile_dir=${PROFILE_DIR} steps=${N_STEPS} profile_step=${PROFILE_STEP}"
exec python3 -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/actor_smoke.py"
