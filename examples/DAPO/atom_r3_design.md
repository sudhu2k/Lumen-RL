# ATOM R3: design and implementation

**Status:** production R3 is `hard_assignment` from vLLM or ATOM.
`assert_supported_r3` fail-closes `replay_mode=distribution`.

This document describes the ATOM R3 path that records the experts selected
during ATOM rollout and replays those same expert assignments during the
training forward.

The implementation deliberately uses the same `routed_experts` contract as
vLLM. Once rollout results reach `RLTrainer`, the training path does not need to
know which inference engine produced them.

## Problem

An MoE router selects a small top-k set of experts for each token. ATOM and the
training engine can select different experts even with identical model weights
because their attention, normalization, router, and MoE kernels are not
numerically identical.

This is more serious than a small floating-point difference. Top-k selection is
discrete: a small router-logit change can send a token through a different
expert and cause a much larger token log-probability difference.

R3 removes that source of mismatch:

1. Record the expert IDs ATOM actually selected.
2. Transfer the IDs with the generated sequence.
3. Replay those IDs during the Megatron or FSDP training forward.

R3 does not make all train and inference kernels identical. Residual log-prob
differences from attention, RoPE, fused MoE, and other kernels can remain.

## What existed before

Lumen-RL already contained a logit-oriented R3 abstraction:

- `R3Manager`
- `RouterRecorder`
- `RouterReplayer`
- `DataProto.router_distributions`
- `AtomRolloutWorker.prepare_r3_recording()`

That design expected a live PyTorch `nn.Module` in the same process. It installed
forward hooks, captured layer outputs as router logits, transferred those
logits, and replaced outputs on another `nn.Module`.

It was not connected to production Ray ATOM:

- Production DAPO/GRPO rollout uses `ATOMReplicaManager` and `ATOMRayServer`,
  not `AtomRolloutWorker`.
- The real ATOM model executes in an EngineCore/ModelRunner process. The
  controller and Ray server do not own the live `FusedMoE` modules.
- `AtomEngine` does not expose an `_model` for `R3Manager.record_phase`.
- Real `FusedMoE` blocks return hidden states, not router logits. Generic
  forward-output replacement would target the wrong tensor.
- Native Megatron replay consumes top-k expert IDs, not
  `DataProto.router_distributions`.

Consequently, the old abstraction demonstrated record/transfer/replay with
simple in-process test modules, but it did not supply usable routes in the
production ATOM Ray path.

## Chosen contract: hard expert assignments

The added path records top-k expert IDs rather than full router distributions.
Each completion carries:

```text
routed_experts: int16[sequence_length - 1, num_layers, top_k]
```

The axes mean:

- `sequence_length - 1`: every causal next-token prediction position used by
  the loss
- `num_layers`: global transformer layer index
- `top_k`: global IDs of the experts ATOM selected

This is the existing MILES/vLLM contract. `RLTrainer` packs it into
`rollout_routing` / `rollout_routed_experts`, and the training engine performs
hard-assignment replay.

Only the discrete expert choice is substituted during training. The training
router's live probabilities are still used for expert weights, preserving the
router's gradient path.

## Why this design

### Reuse the vLLM boundary

Both rollout engines now return the same completion field:

```text
vLLM CompletionOutput.routed_experts ─┐
                                      ├─> RLTrainer ─> train replay
ATOM FusedMoE expert-ID capture ──────┘
```

This keeps route validation, ragged row selection, left-padding alignment,
DataProto packing, and train replay independent of the rollout backend.

### Capture at the actual selection point

ATOM does not expose selected experts through its public completion API.
`FusedMoE.select_experts` is the point where router logits become the exact
weights and IDs used for dispatch. Recording there avoids trying to reproduce
ATOM's top-k decision outside ATOM.

### Run capture inside ModelRunner

The live `FusedMoE` objects are in ModelRunner, not in `RLTrainer` or the
`ATOMRayServer` Ray actor. Capture is therefore installed in
`NoCustomARModelRunner` after ATOM warmup has populated
`compilation_config.static_forward_context`.

### Fail closed

R3 must not silently train without replay. The implementation raises on:

- no `select_experts` calls
- no discoverable `FusedMoE` layers
- non-consecutive global layer indices
- inconsistent token counts between layers
- invalid route rank or sequence length
- missing ModelRunner IPC support

## End-to-end flow

```text
RLTrainer._setup_ray_atom_rollout
    sets lumenrl_r3_capture
        |
        v
ATOMReplicaManager
    sets LUMENRL_ATOM_R3=1
    selects NoCustomARModelRunner
        |
        v
ModelRunner._maybe_warmup
    installs AtomExpertIdCapture
        |
        v
FusedMoE.forward_impl
    establishes the global layer index
FusedMoE.select_experts
    records [tokens, top_k] expert IDs
        |
        v
EngineCore utility command: get_r3_routes
        |
        v
ATOMRayServer
    trims and attaches completion["routed_experts"]
        |
        v
RLTrainer
    validates and packs routes
        |
        v
Megatron RouterReplay / FSDP RoutingReplayContext
    substitutes expert selections during training
```

## Implementation components

### `lumenrl/engine/inference/atom_r3_capture.py`

`AtomExpertIdCapture` temporarily wraps:

- each `FusedMoE.forward_impl`, to identify the active global layer
- `FusedMoE.select_experts`, to collect the exact selected IDs

ATOM invokes every MoE layer for prefill and then again for decode steps.
`assemble_routed_experts()` recognizes the start of another forward when a
layer index repeats, concatenates token rows per layer, and stacks them into
`[tokens, layers, top_k]`.

Routes are moved to CPU and stored as `int16`. Qwen3-MoE expert IDs fit in this
type and it reduces transfer size.

`trim_routed_experts()` accepts the exact causal length or drops one trailing
route produced by engine versions that include the final loss-unused token.
Other length mismatches raise.

The same module registers an EngineCore utility handler for
`get_r3_routes`. That handler calls `ModelRunner.get_r3_routes()` and returns
the drained NumPy tensor to the Ray server.

### `lumenrl/engine/inference/model_runner_nocustomar.py`

When `LUMENRL_ATOM_R3=1`, the patched runner installs capture after warmup.
Installation after warmup is required because ATOM's static forward context and
`FusedMoE` modules are not ready earlier.

`get_r3_routes()` drains the current capture buffer so routes from one request
cannot leak into the next request.

### `examples/DAPO/atom_aiter_shim/sitecustomize.py`

EngineCore imports ATOM in a different process. The shim installs an import
finder when `LUMENRL_ATOM_R3=1` and patches ATOM's
`EngineUtilityHandler` as its module loads.

This keeps the change in Lumen-RL and avoids requiring an ATOM source fork for
one additional utility command.

### `lumenrl/engine/inference/atom_ray_server.py`

`ATOMRayServer` removes `lumenrl_r3_capture` before constructing ATOM because it
is a Lumen-RL option, not an ATOM config field.

With capture enabled:

1. Generate one sequence.
2. Request `get_r3_routes` from EngineCore.
3. Validate/trim the route length.
4. Add `routed_experts` to the completion dictionary.

An async lock serializes generate plus drain. Batch generation also falls back
to one sequence at a time. Without serialization, concurrent requests would
write into the same ModelRunner capture buffer and routes could be attributed
to the wrong completion.

### `lumenrl/trainer/rl_trainer.py`

For ATOM rollout, `moe.r3.enabled` sets `lumenrl_r3_capture=true`. For every
completion, the trainer:

- requires `routed_experts`
- checks `[prompt + response - 1, layers, top_k]`
- converts routes to CPU `int16`
- carries them with their variable-length row through filtering and DP splits
- packs them into the existing train-engine replay payload

There is no ATOM-specific branch after the completion field is read.

## Configuration

The native hard-assignment path uses:

```yaml
policy:
  generation_backend: atom

moe:
  r3:
    enabled: true
    record_router_logits: false
    replay_mode: hard_assignment
```

`record_router_logits` belongs to the older `R3Manager` distribution path. It is
not needed for expert-ID capture.

Production capture is ATOM native `enable_return_routed_experts` (CUDA-graph-safe
slot buffer). `MODE=atombf16` may keep `compilation_config.level=3`.

## Concurrency and performance trade-off

The capture buffer is process-global to a ModelRunner and does not tag calls
with request IDs. The safe initial implementation serializes requests per ATOM
replica.

This favors correctness and fail-closed behavior over peak rollout throughput.
A future ATOM-native API could preserve batching by returning routes as part of
each request output or tagging selection records with request/sequence IDs.

## Compatibility and current limitations

- Capture requires a MoE model using ATOM `FusedMoE` (including `LazyMoEWrapper`).
- Native capture is CUDA-graph-safe; DP-attention is fail-closed in ATOM.
- Layer names must resolve to consecutive global indices starting at zero.
- The verified Qwen3-MoE setup uses ATOM tensor parallel size 1.
- This implementation records ATOM decisions; it does not modify ATOM routing.
  Hard replacement happens only in the training engine.
- The older `R3Manager`/`router_distributions` API remains present but is a
  separate distribution-replay abstraction.

## Tests and verification

CPU tests in `tests/unit/engine/test_atom_r3_capture.py` cover:

- prefill plus decode assembly
- rejection of empty capture
- rejection of layer gaps
- trailing-token trimming
- completion field export and pass-through

The production path was also tested with one-step, same-weight Qwen3-30B-A3B
rollouts. The primary indicator is the fraction of response tokens where the
absolute rollout-vs-train log-prob difference exceeds `0.1`.

```text
rollout  R3   abs_diff   frac_abs_gt_0.1
vLLM     off  0.0240     0.081
vLLM     on   0.0135     0.039
ATOM     off  0.0256     0.087
ATOM     on   0.0133     0.041
```

Native ATOM hard replay cut the large-mismatch tail from `0.087` to `0.041`
(~2×, matching vLLM). That confirms real routes reached training. The remaining
gap is train/inference kernel difference, not missing replay.

The exact launch procedure and full metrics are documented in
`examples/DAPO/native_r3_vllm_verify.md`. The matched ATOM smoke config is
`examples/DAPO/configs/dapo_qwen3moe_a3b_ray_megatron_r3_atom_smoke.yaml`.

## Alternatives considered

### Repair the old `R3Manager` logit path

This would still require reaching inside ModelRunner, identifying the exact
router tensor, transferring much larger float distributions, and adding a
separate train-engine consumer. Replacing router logits can also sever or alter
the live training router's gradient semantics.

The expert-ID contract was already implemented and validated on the training
side, making it smaller and safer.

### Recompute ATOM top-k in the controller

The controller does not have ATOM's per-layer hidden states or exact fused
router behavior. Recomputing top-k elsewhere would record an approximation,
not the experts used for rollout.

### Patch ATOM upstream directly

ATOM now exposes `enable_return_routed_experts` on `LLM.generate()`. Lumen-RL
sets that flag from `moe.r3.enabled` and forwards `routed_experts` through
`ATOMRayServer`.
