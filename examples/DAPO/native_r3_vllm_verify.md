# Native Megatron R3 vs vLLM: 1-step KL smoke

This is the procedure used to measure whether Megatron-native hard-assignment
R3 (replay of vLLM `routed_experts`) closes the same-weight train vs rollout
log-prob gap. Metric of interest: `r3_verify/frac_abs_gt_0.1`.

Same-weight train vs rollout log-prob gap under **Megatron native** hard-assignment
R3. Native R3 refuses activation recompute. Rollout can be vLLM (`routed_experts`)
or ATOM (ModelRunner capture of `FusedMoE.select_experts` → the same field).

## What is being measured

`RLTrainer._r3_verify_old_vs_rollout` compares, **before the optimizer step**:

- `old_log_probs` — actor forward on the rollout sequences (train engine)
- `rollout_log_probs` — sampler log-probs (vLLM)

Signed `r3_verify/kl` can cancel expert flips. Watch:

| key | meaning |
| --- | --- |
| `r3_verify/tokens` | masked response tokens |
| `r3_verify/kl` | mean (`rollout − train`) |
| `r3_verify/abs_diff` | mean absolute Δ log-prob |
| `r3_verify/frac_abs_gt_0.1` | fraction with \|Δ\| > 0.1 |

A packing-correct, kernel-matched run should drive `frac_abs_gt_0.1` toward 0
and `abs_diff` toward ~1e-3. Residual after R3-on is leftover train/infer
kernel mismatch, not proof that replay never ran.

## Prerequisites

- 8 GPUs, idle (stop other `--network=host` Ray jobs first)
- Host trees:

```text
/home/sugovind/Lumen-RL          # bind-mounted source (has r3_verify + smoke yaml)
/data/rl_data/models/Qwen3-30B-A3B-Base
/data/rl_data/data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet
/data/rl_data/data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet
```

- Image with **both** Megatron (`RouterReplay`) and vLLM 0.23
  (`enable_return_routed_experts` / `CompletionOutput.routed_experts`):

```text
zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260910
```

Do **not** use `ato-rl:gfx950-vllm0.23` (vLLM only, no Megatron) or the IFU
ATOM image (`sudhu_megatron_ifu` / `megatron:ifu18`) for this A/B.

## 1. Stop competing Ray

Shared `--network=host` means a leftover cluster on 6379 will steal the job.

```bash
# on the box that still owns the old cluster, e.g.
docker exec <old-container> ray stop --force
```

Confirm GPUs are free before starting the new container.

## 2. Start the DAPO image with host Lumen-RL

```bash
mkdir -p /data/rl_data/logs /data/rl_data/ckpts

docker run -d --name lumenrl-r3-vllm \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --shm-size 64G \
  -v /data/rl_data:/data/rl_data \
  -v /home/sugovind/Lumen-RL:/opt/lumenrl/Lumen-RL \
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260910 \
  sleep infinity
```

`run_dapo.sh` expects `RL_ROOT=/opt/lumenrl` (so `$RL_ROOT/Lumen-RL` is the
bind mount). `DATA_ROOT=/data/rl_data`.

## 3. Preflight (inside the container)

```bash
docker exec lumenrl-r3-vllm bash -lc '
python3 -c "import vllm; from vllm.outputs import CompletionOutput; print(vllm.__version__, \"routed_experts\" in CompletionOutput.__annotations__)"
python3 -c "from megatron.core.transformer.moe.router_replay import RouterReplay; print(\"RouterReplay ok\")"
test -f /opt/lumenrl/Lumen-RL/examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_r3_smoke.yaml
test -d /data/rl_data/models/Qwen3-30B-A3B-Base
'
```

Expect `0.23.0 True` and `RouterReplay ok`.

## 4. Smoke config

File: `examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_r3_smoke.yaml`

Hard constraints that were required for this path:

- `policy.training_backend: megatron_native`
- `policy.generation_backend: vllm`
- `moe.r3.enabled: true`, `replay_mode: hard_assignment`,
  `record_router_logits: false` (vLLM IDs, not ATOM logits)
- no activation recompute (`enable_dynamic_batch: false`; native R3 fail-closes
  if recompute is on)
- `calculate_log_probs: true`
- 1 step, `filter_groups.enable: false`, `wandb_enabled: false`
- EP=8, TP=PP=CP=1, seq 1024 / 256, batch 64

## 5. R3-on (1 step)

```bash
docker exec -d lumenrl-r3-vllm bash -lc '
export RL_ROOT=/opt/lumenrl
export DATA_ROOT=/data/rl_data
export MODE=bf16
export TRAIN_FP8=0
export STEPS=1
export CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_r3_smoke.yaml
export MODEL_PATH=/data/rl_data/models/Qwen3-30B-A3B-Base
export LOG=/data/rl_data/logs/r3_verify_vllm_on.log
export PYTHONUNBUFFERED=1
bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh
'
```

Wait until the log shows `RLTrainer.train (ray-controller) finished after 1 steps`.
On this box that was ~2–3 minutes after worker init (weight load + first vLLM
JIT). `run_dapo.sh` itself runs `ray stop --force` before `python -m lumenrl.trainer.main`.

Confirm native R3 actually installed:

```text
[MegatronNativeEngine] ... r3=True
LumenActorWorker: initialized megatron_native engine
```

Routes packed if you see `torch.as_tensor(raw_routes)` in `rl_trainer.py`
(non-writable numpy warning). Missing `routed_experts` would fail earlier in
the trainer pack path.

Grep:

```bash
rg "r3_verify |r3=True|initialized megatron_native|finished after" \
  /data/rl_data/logs/r3_verify_vllm_on.log
```

## 6. R3-off A/B (same image, same config, same seed)

After the on-run fully exits (cleanup printed `LumenRL finished`), launch the
matched off run. Same yaml, Hydra override only:

```bash
docker exec -d lumenrl-r3-vllm bash -lc '
export RL_ROOT=/opt/lumenrl
export DATA_ROOT=/data/rl_data
export MODE=bf16
export TRAIN_FP8=0
export STEPS=1
export CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_r3_smoke.yaml
export MODEL_PATH=/data/rl_data/models/Qwen3-30B-A3B-Base
export LOG=/data/rl_data/logs/r3_verify_vllm_off.log
export PYTHONUNBUFFERED=1
export EXTRA_OVERRIDE=moe.r3.enabled=false
bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh
'
```

Confirm `r3=False` in the Megatron spec line.

## 7. Numbers from this machine (2026-09-16 / 2026-09-17)

Qwen3-30B-A3B-Base, DAPO math parquet, seed 10086, 1 step.

| run | `r3` | tokens | kl | abs_diff | **frac_abs_gt_0.1** |
| --- | --- | --- | --- | --- | --- |
| vLLM + native | off | 15623 | 0.00222 | 0.0240 | **0.081** |
| vLLM + native | on | 15533 | 0.000835 | 0.0135 | **0.039** |
| ATOM native `enable_return_routed_experts` | off | 16087 | 0.00196 | 0.0256 | **0.087** |
| ATOM native `enable_return_routed_experts` | on | 16240 | 0.000645 | 0.0133 | **0.041** |

Older Lumen-RL wrapper capture (eager `select_experts`, 2026-09-16) was ATOM-off
0.087 / on 0.057, then fused Adam OOM at `gpu_memory_utilization: 0.90`. Signed
`kl` rose on that on-run; native capture does not.

Logs:

- `/data/rl_data/logs/r3_verify_vllm_on.log`
- `/data/rl_data/logs/r3_verify_vllm_off.log`
- `/data/rl_data/logs/r3_verify_atom_on.log`
- `/data/rl_data/logs/r3_verify_atom_off.log`

vLLM R3-on cut the tail ~2× (0.081 → 0.039). Native ATOM R3-on matches that
(~2.1×, 0.087 → 0.041) and brings `abs_diff` / signed `kl` in line with vLLM-on.
Neither goes to ~0; leftover is train/infer kernel mismatch (fused MoE / RoPE).

Native ATOM smokes finished the optimizer step at `gpu_memory_utilization: 0.50`
with `MODE=atombf16` sleep_level=2. `0.35` left `available_for_kv=0`.

## 8. ATOM A/B (native capture, MODE=atombf16)

Mount a host ATOM tree that has `enable_return_routed_experts` over
`/opt/lumenrl/ATOM`. The image copy does not have the flag. Config:
`examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_r3_atom_smoke.yaml`.
`MODE=atombf16` enables CUDA graphs (`compilation_config.level=3`); native
capture is graph-safe, so do not force eager.

```bash
docker run -d --name lumenrl-r3-atom \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --shm-size 64G \
  -v /data/rl_data:/data/rl_data \
  -v /home/sugovind/Lumen-RL:/opt/lumenrl/Lumen-RL \
  -v /home/sugovind/ATOM:/opt/lumenrl/ATOM \
  zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260910 \
  sleep infinity

docker exec -d lumenrl-r3-atom bash -lc '
export RL_ROOT=/opt/lumenrl
export DATA_ROOT=/data/rl_data
export MODE=atombf16
export TRAIN_FP8=0
export STEPS=1
export CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_r3_atom_smoke.yaml
export MODEL_PATH=/data/rl_data/models/Qwen3-30B-A3B-Base
export LOG=/data/rl_data/logs/r3_verify_atom_on.log
export PYTHONUNBUFFERED=1
bash /opt/lumenrl/Lumen-RL/examples/DAPO/run_dapo.sh
'
```

Confirm native replay packed routes:

```text
[MegatronNativeEngine] ... r3=True
routes = torch.as_tensor(raw_routes)
```

R3-off: same command with `LOG=.../r3_verify_atom_off.log` and
`EXTRA_OVERRIDE=moe.r3.enabled=false`. Expect `r3=False`.

## Pitfalls that already burned a run

- **`gpu_memory_utilization: 0.35` colocated**: ATOM warmup succeeds then
  `available_for_kv=0`. Use ~0.50 plus sleep_level=2.
- **Older ATOM-only IFU path**: before capture, native Megatron saw no expert
  ids (`frac` ~0.14). Ray ATOM + `moe.r3.enabled` now records the same
  `routed_experts` field as vLLM.
- **vLLM-only image**: cannot import Megatron `RouterReplay`.
- **Host `python - <<'PY'`** while intending container: runs on the host. Use
  `docker exec` (or `docker cp` + `python3 /tmp/...`).
- **`--network=host` Ray collision**: stop the previous cluster before
  `docker run`.
- **Activation recompute** with native R3: engine raises; keep it off in the
  smoke yaml.
- **`@staticmethod` on `_r3_verify_old_vs_rollout`**: a container copy of
  `rl_trainer.py` that dropped the decorator dies with
  `takes 3 positional arguments but 4 were given`. Bind-mount the host tree
  so the verify helper is the one on disk.
