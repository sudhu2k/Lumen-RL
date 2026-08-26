# Recreate: Megatron-Native MoE A2A vs MORI (actor-only)

Actor-only comparison of Megatron expert-parallel token dispatch on **8× MI300X (gfx942)**:

- **A2A**: `moe_token_dispatcher_type=alltoall` (NCCL `all_to_all`)
- **MORI**: `moe_token_dispatcher_type=flex` + `moe_flex_dispatcher_backend=mori` (ROCm MORI v1.2.1)

This is **not** a full DAPO/Ray run. There is no rollout, no parquet, no vLLM. Each rank constructs a dummy DAPO batch, calls `compute_log_probs` once, then `update_policy` for `N_STEPS` iterations. Torch profiler captures **one** step (default step 12).

Verified on 2026-08-24. Chrome traces from that run:

- A2A: `/workspace/data/logs/actor_profile_a2a/a2a_step12_actor_rank{0,1}.json`
- MORI: `/workspace/data/logs/actor_profile_mori/mori_step12_actor_rank{0,1}.json`

## What it measures

| Setting | Value |
|---|---|
| Model | Qwen3-30B-A3B-Base (HF weights) |
| Backend | LumenRL `training_backend=megatron_native` (`megatron.core` from Megatron-LM) |
| Parallelism | TP=PP=CP=1, **EP=8**, DP=8, SequentialMLP (`moe_grouped_gemm=false`) |
| Optimizer | distributed Adam, BF16 |
| Rows / rank | `N_ROWS=2` |
| Sequence length | `SEQ_LEN=512` (attention all ones) |
| Response tokens | second half of the sequence → **256 / row**, **512 / rank** (`ppo_kl_tok`) |
| Global PPO tokens | `batch_num_tokens = 512 × 8 = 4096` |
| Forward seq tokens | 2 × 512 = **1024 / rank**, **8192** global |
| `max_tokens_per_gpu` | 2048 (one microbatch) |
| Steps | 20 (`0..19`), profiler on step **12** only |

Dummy `input_ids` are random; this is a comm/kernel microbenchmark, not a quality run.

## Layout (verified machine)

```text
/workspace/Lumen-RL          # this repo
/workspace/Lumen             # Lumen (AITER + patches); not used as megatron.core
/workspace/Megatron_17/Megatron-LM
/workspace/TransformerEngine
/workspace/mori              # ROCm/mori.git tag v1.2.1 (not Lumen third_party/mori)
/workspace/data/models/Qwen3-30B-A3B-Base
```

Override with `RL_ROOT`, `MEGATRON_PATH`, `TE_PATH`, `LUMEN_PATH`, `AITER_PATH`, `DATA_ROOT`, `MODEL_PATH`.

`PYTHONPATH` must put **Megatron-LM and TransformerEngine ahead of Lumen-RL**. Actor training imports `megatron.core` directly. Do not `pip install` Lumen and expect that to provide Megatron.

## 1. Install MORI v1.2.1

Skip this for A2A-only.

```bash
git clone --branch v1.2.1 --recurse-submodules https://github.com/ROCm/mori.git /workspace/mori
cd /workspace/mori

# setuptools 75 rejects PEP 639 `license = "MIT"`. If pip fails on project.license:
#   change pyproject.toml to: license = {text = "MIT"}

# UMBP needs gRPC headers. EP/SHMEM do not.
export PYTORCH_ROCM_ARCH=gfx942 GPU_ARCHS=gfx942 MORI_GPU_ARCHS=gfx942 BUILD_UMBP=OFF
python3 -m pip install -e . --no-build-isolation --no-deps

python3 -c "import mori, mori.ops, mori.shmem; print(mori.__file__)"
```

Expect `mori.__file__` under `/workspace/mori/python/mori`. Kernels JIT on first use into `~/.mori/jit/` (`ep_intranode` for gfx942). First MORI `compute_log_probs` can take ~40s; later runs use the cache.

Do **not** put Lumen's `third_party/mori` on `PYTHONPATH` (that tree has Python without `libmori_pybinds.so`).

## 2. Common environment

```bash
export PYTHONPATH="/workspace/Megatron_17/Megatron-LM:/workspace/TransformerEngine:/workspace/Lumen-RL:/workspace/Lumen:/workspace/Lumen/third_party/aiter"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT=1800
export TORCHDYNAMO_DISABLE=1
unset NVTE_FLASH_ATTN NVTE_FUSED_ATTN NVTE_UNFUSED_ATTN
export MODEL_PATH=/workspace/data/models/Qwen3-30B-A3B-Base
export N_STEPS=20 PROFILE_STEP=12 SEQ_LEN=512 N_ROWS=2
```

**Must unset** `NVTE_FLASH_ATTN` / `NVTE_FUSED_ATTN` / `NVTE_UNFUSED_ATTN`. Forcing `NVTE_FLASH_ATTN=0` breaks Megatron's auto attention backend on this stack.

Flex preprocess is `@jit_fuser` → `torch.compile`. This image's Triton has no `triton_key`; the smoke calls `disable_jit_fuser()` and sets `TORCHDYNAMO_DISABLE=1`.

Cards should be idle (~300 MB VRAM) before launch.

## 3. Run A2A

Each process must see **one** GPU (`CUDA_VISIBLE_DEVICES=$LOCAL_RANK`). The smoke script does that when `MOE_DISPATCHER=alltoall`.

```bash
cd /workspace/Lumen-RL
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export MOE_DISPATCHER=alltoall TRACE_TAG=a2a
export PROFILE_DIR=/workspace/data/logs/actor_profile_a2a
mkdir -p "$PROFILE_DIR"
python3 -m torch.distributed.run --nproc_per_node=8 --master_port=29569 \
  examples/megatron_moe_dispatcher/actor_smoke.py
```

Or: `bash examples/megatron_moe_dispatcher/run_actor_smoke.sh alltoall`

## 4. Run MORI

Each process must see **all eight** GPUs. MORI-SHMEM calls `hipDeviceCanAccessPeer` with physical device ids. Pinning one GPU per process fails with `invalid device ordinal`.

```bash
cd /workspace/Lumen-RL
export MOE_DISPATCHER=flex MOE_FLEX_BACKEND=mori TRACE_TAG=mori
export PROFILE_DIR=/workspace/data/logs/actor_profile_mori
export MORI_SHMEM_LOG_LEVEL=INFO MORI_GPU_ARCHS=gfx942
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
mkdir -p "$PROFILE_DIR"
python3 -m torch.distributed.run --nproc_per_node=8 --master_port=29568 \
  examples/megatron_moe_dispatcher/actor_smoke.py
```

Or: `bash examples/megatron_moe_dispatcher/run_actor_smoke.sh mori`

Leave `CUDA_VISIBLE_DEVICES` set to `0,1,...,7` **before** `import lumenrl.workers.actor_worker`. If it is unset, `base_worker` calls Ray to discover GPUs and eight ranks each start a local Ray cluster.

## 5. Outputs

| Artifact | Path |
|---|---|
| Chrome trace (ranks 0 and 1) | `$PROFILE_DIR/${TRACE_TAG}_step${PROFILE_STEP}_actor_rank{0,1}.json` |
| Profiler table | same prefix `.txt` |
| Log line | `step 12 … profiled` — wall time includes profiler overhead |

Open the JSON in `chrome://tracing` or Perfetto. Compare `nccl:all_to_all` (A2A) vs `EpCombineIntraNodeKernel` / `MoriDispatch*` (MORI).

## Expected results (this machine, 2026-08-24)

Wall-clock **steady-state** `update_policy` (not the profiled step): ~**1.13–1.17 s** A2A vs ~**1.14–1.16 s** MORI. Step 12 with profiler on is ~3 s on both; ignore that for speed.

**Step 12 CUDA, rank 0** (one `update_policy`, Self CUDA ~0.9 s):

| | A2A | MORI v1.2.1 |
|---|---|---|
| Self CUDA | 951 ms | 906 ms |
| Token dispatch | `nccl:all_to_all` 494 ms (**52%**), 288 calls | `EpCombineIntraNodeKernel_bf16_nop2p` 346 ms (38%) + `MoriDispatchBackward` 317 ms (35%) |
| `nccl:all_to_all` | yes | none |

Remaining NCCL (`reduce_scatter` / `all_gather` / `all_reduce`) is the distributed optimizer, not expert dispatch.

Peak reserved ~80–85 GiB of 192 GiB/GPU.

## Full DAPO (optional)

Same Megatron flags go through `run_dapo.sh`:

```bash
ENABLE_MORI=true MEGATRON_PATH=/workspace/Megatron_17/Megatron-LM \
  bash examples/DAPO/run_dapo.sh
```

That path needs rollout + dataset and is outside this actor microbenchmark.

## Pitfalls

1. `NVTE_FLASH_ATTN=0` → Megatron attention backend error. Unset the NVTE attn vars.
2. MORI + one visible GPU → `hipDeviceCanAccessPeer` / invalid device ordinal.
3. MORI + unset `CUDA_VISIBLE_DEVICES` → eight Ray head processes, hang before model load.
4. Lumen in-tree MORI submodule is not a substitute for ROCm/mori v1.2.1.
5. `BUILD_UMBP=ON` (setup.py default) fails cmake without gRPC; use `BUILD_UMBP=OFF`.
6. Do not run A2A and MORI at the same time on the same eight cards.
