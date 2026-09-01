#!/usr/bin/env bash
# Actor-only Megatron-Native MoE smoke (no Ray DAPO, no rollout).
#
# 1 node (8 GPUs, EP=8):
#   NNODES=1 bash run_actor_smoke.sh alltoall
#   NNODES=1 bash run_actor_smoke.sh mori
#
# 2 nodes (8 GPUs/node, world=16, default EP=16). Run the same command on
# every node; only NODE_RANK differs. MASTER_ADDR is node 0's IP.
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0-ip> bash run_actor_smoke.sh alltoall
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0-ip> bash run_actor_smoke.sh alltoall
# Override with EP=8 for two-node DP=2 instead of full expert-parallel.
set -euo pipefail

MODE="${1:-alltoall}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_ROOT="${RL_ROOT:-/workspace/Lumen-RL}"
MEGATRON_PATH="${MEGATRON_PATH:-/workspace/Megatron_17/Megatron-LM}"
TE_PATH="${TE_PATH:-/workspace/TransformerEngine}"
LUMEN_PATH="${LUMEN_PATH:-/workspace/Lumen}"
AITER_PATH="${AITER_PATH:-/workspace/Lumen/third_party/aiter}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-29569}"
MASTER_ADDR="${MASTER_ADDR:-}"

if [ -z "${MASTER_ADDR}" ]; then
  if [ "${NNODES}" -gt 1 ]; then
    echo "NNODES=${NNODES} requires MASTER_ADDR=<node0-ip> (and NODE_RANK=0|1 on each host)." >&2
    echo "  NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0-ip> $0 ${MODE}" >&2
    echo "  NNODES=2 NODE_RANK=1 MASTER_ADDR=<node0-ip> $0 ${MODE}" >&2
    echo "Single node: NNODES=1 $0 ${MODE}" >&2
    exit 2
  fi
  MASTER_ADDR="127.0.0.1"
fi

export PYTHONPATH="${MEGATRON_PATH}:${TE_PATH}:${RL_ROOT}:${LUMEN_PATH}:${AITER_PATH}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT=1800
export LUMENRL_MEM_DIAG="${LUMENRL_MEM_DIAG:-1}"
export TORCHDYNAMO_DISABLE=1
export MORI_SHMEM_HEAP_SIZE=4G
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

# Multi-node RCCL (same defaults as Megatron examples/qwen3/train_qwen3.sh).
if [ "${NNODES}" -gt 1 ]; then
  # Prefer ACTIVE InfiniBand HCAs from /sys/class/infiniband (stable mlx5_*
  # names). Skip Ethernet/RoCE ports (e.g. mlx5_* bound to a socket NIC).
  # Fall back to `rdma` and then to
  # up ibp* netdevs for images that only expose IPoIB-style names.
  if [ -z "${NCCL_IB_HCA:-}" ]; then
    NCCL_IB_HCA="$(python3 -c "
import os
names = []
root = '/sys/class/infiniband'
if os.path.isdir(root):
    for d in sorted(os.listdir(root)):
        port1 = os.path.join(root, d, 'ports', '1')
        try:
            ll = open(os.path.join(port1, 'link_layer')).read().strip()
            st = open(os.path.join(port1, 'state')).read().strip()
        except OSError:
            continue
        if ll == 'InfiniBand' and 'ACTIVE' in st:
            names.append(d)
        if len(names) >= 8:
            break
print(','.join(names))
")"
  fi
  if [ -z "${NCCL_IB_HCA:-}" ]; then
    if command -v rdma >/dev/null 2>&1; then
      NCCL_IB_HCA="$(rdma link -j 2>/dev/null | python3 -c "import json, sys
try:
    links = json.load(sys.stdin)
    print(*[links[i]['ifname'] for i in range(min(8, len(links)))], sep=',')
except Exception:
    pass" || true)"
    fi
  fi
  if [ -z "${NCCL_IB_HCA:-}" ]; then
    NCCL_IB_HCA="$(python3 -c "
import glob, os
names = []
for path in sorted(glob.glob('/sys/class/net/ibp*')):
    name = os.path.basename(path)
    try:
        state = open(os.path.join(path, 'operstate')).read().strip()
    except OSError:
        continue
    if state == 'up':
        names.append(name)
    if len(names) >= 8:
        break
print(','.join(names))
")"
  fi
  if [ -z "${NCCL_IB_HCA:-}" ]; then
    echo "NNODES=${NNODES}: NCCL_IB_HCA is empty (no ACTIVE InfiniBand HCAs). Set it on this host before launch." >&2
    echo "  ls /sys/class/infiniband ; cat /sys/class/infiniband/*/ports/1/state" >&2
    exit 2
  fi
  export NCCL_IB_HCA
  # GID index depends on the fabric: InfiniBand uses LID routing and only GID
  # index 0 is populated; RoCE uses the RoCEv2 GID (index 3). Forcing index 3 on
  # an IB fabric selects an all-zero GID and hangs cross-node QP setup at Init
  # START, so pick the index from the first HCA's link layer.
  if [ -z "${NCCL_IB_GID_INDEX:-}" ]; then
    _first_hca="${NCCL_IB_HCA%%,*}"
    _link_layer="$(cat "/sys/class/infiniband/${_first_hca}/ports/1/link_layer" 2>/dev/null || echo Ethernet)"
    if [ "${_link_layer}" = "InfiniBand" ]; then
      NCCL_IB_GID_INDEX=0
    else
      NCCL_IB_GID_INDEX=3
    fi
  fi
  export NCCL_IB_GID_INDEX
  export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
  # train_qwen3.sh hardcodes ens51np0; some clusters use ens14np0.
  _sock_if="${NCCL_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME:-}}"
  if [ -z "${_sock_if}" ]; then
    for _cand in ens51np0 ens14np0; do
      if [ -d "/sys/class/net/${_cand}" ]; then
        _sock_if="${_cand}"
        break
      fi
    done
  fi
  if [ -n "${_sock_if}" ]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${_sock_if}}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${_sock_if}}"
    export MORI_SOCKET_IFNAME="${MORI_SOCKET_IFNAME:-${_sock_if}}"
  else
    echo "NNODES=${NNODES}: set NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME (no ens51np0/ens14np0)." >&2
  fi
fi

# Distributed-optimizer comm overlap + 600 MB buckets, as in train_qwen3.sh.
export DDP_BUCKET_SIZE="${DDP_BUCKET_SIZE:-629145600}"
export OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-1}"
export OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-1}"
# Balanced routing (train_qwen3.sh uses aux_loss 1e-3) keeps expert load even so
# dispatch/combine don't stall on stragglers.
export MOE_AUX_LOSS="${MOE_AUX_LOSS:-1e-3}"
# Fair A2A-vs-MORI: keep A2A on the same 8-visible topology as MORI.
export KEEP_ALL_GPUS="${KEEP_ALL_GPUS:-1}"
# Qwen3-30B-A3B has 128 experts; EP must divide 128. Default EP = world size
# (8 on 1 node, 16 on 2 nodes) so 2-node A2A/MORI is fully expert-parallel.
export EP="${EP:-$((GPUS_PER_NODE * NNODES))}"

export MODEL_PATH="${MODEL_PATH:-${DATA_ROOT}/models/Qwen3-30B-A3B-Base}"
if [ ! -f "${MODEL_PATH}/config.json" ]; then
  echo "Missing ${MODEL_PATH}/config.json." >&2
  echo "  hf download Qwen/Qwen3-30B-A3B-Base --local-dir ${MODEL_PATH}" >&2
  exit 2
fi
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
echo "dispatcher=${MOE_DISPATCHER} nnodes=${NNODES} node_rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT} nproc=${GPUS_PER_NODE} ep=${EP} nccl_ib_hca=${NCCL_IB_HCA:-unset} gloo_if=${GLOO_SOCKET_IFNAME:-unset} profile_dir=${PROFILE_DIR} steps=${N_STEPS} profile_step=${PROFILE_STEP}"
exec python3 -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/actor_smoke.py"
