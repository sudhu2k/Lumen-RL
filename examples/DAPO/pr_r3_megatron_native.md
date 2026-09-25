# PR: Native Megatron R3 (vLLM hard-assignment)

**Suggested title:** `feat(r3): replay vLLM expert IDs in Megatron-native training`

Compare: [ZhangDanyang-AMD/main…sudhu2k:sudhu/R3_megatron_native](https://github.com/ZhangDanyang-AMD/Lumen-RL/compare/main...sudhu2k:Lumen-RL:sudhu/R3_megatron_native?expand=1)

Paste the body below into the GitHub PR.

---

## Summary

- Replay vLLM rollout expert IDs (`routed_experts`) through Megatron-Core `RouterReplay` on native Megatron log-prob and update forwards, instead of the old router-logit hook path.
- Fail closed unless R3 is **vLLM + `hard_assignment`**. ATOM backends and `replay_mode=distribution` raise when `moe.r3.enabled=true` (check runs only if R3 is enabled).
- Add `r3_verify/*` metrics (same-weight train vs rollout log-probs before the optimizer step) so routing mismatch is visible as `frac_abs_gt_0.1`, not only signed KL.

## Why fail-closed to vLLM hard-assignment

R3 in the paper ([arXiv:2510.11370](https://arxiv.org/abs/2510.11370)) replays the **experts the rollout actually selected**, not the full router softmax. vLLM implements that as `enable_return_routed_experts` → `CompletionOutput.routed_experts`: `int16 [seq_len-1, num_layers, top_k]` IDs ([vllm#28284](https://github.com/vllm-project/vllm/pull/28284)). Megatron-Core `RouterReplay` consumes those IDs. There is no vLLM field for full-width router logits, so `replay_mode=distribution` cannot be filled from rollout and must not silently fall back to training-time routing.

ATOM on `ROCm/ATOM` main still has no native `generate()` `routed_experts` contract. Capture for ATOM is in-flight as [ROCm/ATOM#2274](https://github.com/ROCm/ATOM/pull/2274) (open); until that lands in the image this trainer uses, `generation_backend=atom` + `moe.r3.enabled` would look enabled and then produce no routes. Even that ATOM PR matches vLLM IDs only — it does not return distributions either.

Native Megatron already has `RouterReplay`; this PR wires LumenRL packing (ragged `rollout_routing` / dense `rollout_routed_experts`, PP layer map, CP token split) into that API.

## What changed

- [`lumenrl/engine/training/megatron_r3_replay.py`](lumenrl/engine/training/megatron_r3_replay.py) — extract, pack, install per-local-layer IDs.
- [`lumenrl/engine/training/megatron_native_engine.py`](lumenrl/engine/training/megatron_native_engine.py) — install replay on MoE; refuse activation recompute while R3 is on (checkpointed forwards would replay the last microbatch’s IDs).
- [`lumenrl/moe/r3_scope.py`](lumenrl/moe/r3_scope.py) — supported combo is `generation_backend=vllm` and `replay_mode=hard_assignment`.
- [`lumenrl/trainer/rl_trainer.py`](lumenrl/trainer/rl_trainer.py) — `r3_verify/tokens`, `r3_verify/kl`, `r3_verify/abs_diff`, `r3_verify/frac_abs_gt_0.1`.
- `R3Config` defaults: `enabled=false`, `record_router_logits=false`, `replay_mode=hard_assignment`. Example YAMLs aligned (ATOM examples keep R3 off).

`record_router_logits` still only gates the old in-process logit recorder (`R3Manager.record_phase`). The vLLM ID path does not use it.

## Test plan

- [ ] `pytest tests/unit/engine/test_megatron_native_r3.py tests/unit/moe/test_r3_scope.py tests/unit/test_config.py`
- [ ] Config load: ATOM + `moe.r3.enabled=true` raises; ATOM with R3 off loads; `replay_mode=distribution` with R3 on raises.
- [ ] 1-step DAPO smoke, native Megatron EP=8 + vLLM 0.23, Qwen3-30B-A3B, `dapo_qwen3moe_a3b_ray_megatron_r3_smoke.yaml` (or equivalent): log shows `[MegatronNativeEngine] ... r3=True` and `initialized megatron_native engine`.
- [ ] Matched A/B on the same image: R3 on vs `EXTRA_OVERRIDE=moe.r3.enabled=false`. Watch `r3_verify/frac_abs_gt_0.1` (R3-on should drop it; residual kernel mismatch can remain).
- [ ] Confirm activation recompute + R3 fails closed.

## Smoke numbers (vLLM, 2026-09-16)

Image `zhangdanyangamd/lumen-rl:dapo-gfx950-rocm7.2.3-260910`, Qwen3-30B-A3B-Base, native Megatron EP=8, vLLM 0.23, 1 DAPO step, seed 10086.

On the response mask, `delta = rollout_log_probs − old_log_probs` at the **same weights, before the optimizer**. `tokens` is the masked count; `kl` is mean `delta` (signed, so expert flips can cancel); `abs_diff` is mean `|delta|`; `frac_abs_gt_0.1` is the fraction with `|delta| > 0.1`. `timing/gen_s` is vLLM generate wall time.

| r3 | tokens | kl | abs_diff | **frac_abs_gt_0.1** | timing/gen_s |
| --- | --- | --- | --- | --- | --- |
| off | 15623 | 0.00222 | 0.0240 | **0.081** | 7.37 |
| on | 15533 | 0.000835 | 0.0135 | **0.039** | 9.73 |

Hard-assignment cuts the large-mismatch tail ~2×. Residual is train/infer kernel mismatch (vLLM fused MoE vs Megatron), not missing routes.
