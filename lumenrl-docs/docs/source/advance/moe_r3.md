# MoE Rollout Routing Replay (R3)

MoE reinforcement learning couples two different code paths: high-throughput
inference routers (vLLM or ATOM) and training-time routers inside Megatron/Lumen.
Production R3 is **hard_assignment** of expert ids from ``vllm`` or ``atom``.
``assert_supported_r3`` fail-closes ``replay_mode=distribution``.

Replay consumes MILES expert ids ``[seq_len-1, layers, top_k]`` on
``rollout_routing`` / ``rollout_routed_experts``. vLLM fills that from
``CompletionOutput.routed_experts``. ATOM fills it from native
``LLMEngine.generate()`` when ``enable_return_routed_experts`` is on.

## Problem statement

During on-policy or near on-policy training we expect the training forward to evaluate the same routing distribution as the rollout engine for each token. In practice:

- Inference stacks may fuse softmax, top-k, and dispatch differently than training.
- FP8 rollouts change logits slightly, which alters argmax routes even when BF16 training is “close.”
- Expert parallel sharding reorders reductions, amplifying bitwise drift.

The result is **router inconsistency**: the policy gradient is computed against a different routing process than the one that produced the data, which breaks the stationarity assumptions underlying PPO-style surrogates.

## R3 solution (record / transfer / replay)

Production Ray RL (native Megatron) uses **hard-assignment expert ids**:

1. **Record** — vLLM or ATOM `enable_return_routed_experts`.
2. **Transfer** — `RLTrainer` packs per-completion `routed_experts` into
   ragged `rollout_routing`.
3. **Replay** — `MegatronNativeEngine` injects those ids via Megatron
   `RouterReplay`.

A second, older contract still exists for logit-level replay (unit tests /
`AtomRolloutWorker`): `RouterRecorder` → `DataProto.router_distributions` →
`RouterReplayer`. Native Megatron `RouterReplay` does **not** consume that
payload.

## Configuration

```yaml
moe:
  r3:
    enabled: true
    record_router_logits: false   # expert-id path; true is the logit recorder
    replay_mode: hard_assignment  # native Megatron RouterReplay
```

| Field | Purpose |
| --- | --- |
| `enabled` | Master switch; Ray rollout attaches `routed_experts` when true |
| `record_router_logits` | ATOM `RouterRecorder` logit hooks (`AtomRolloutWorker`); unused by native expert-id replay |
| `replay_mode` | Must be `hard_assignment`. `distribution` fail-closes until vLLM returns router logits. |

Structured types are `MoEConfig` → `R3Config` in {doc}`/api/config`.

## Monitoring

Use KL and utilization side channels alongside standard RL metrics:

- **Policy–reference KL** — spikes often precede MoE collapse when routers disagree.
- **Expert utilization** — compute from captured `router_logits` with `check_expert_utilization`.
- **Router entropy** — `compute_router_entropy` tracks how decisive routing is over time.

```python
from lumenrl.moe.moe_utils import compute_router_entropy, check_expert_utilization

util = check_expert_utilization(router_logits, num_experts=128)
ent = compute_router_entropy(router_logits)
```

## Best practices

- Enable R3 for **all MoE RL runs** unless you are explicitly isolating a baseline.
- Keep `record_router_logits=true` until you verify hooks are lightweight in your deployment.
- When debugging, log histograms of `router_dist_layer_*` norms per layer to spot missing tensors after `DataProto.merge`.
- Pair R3 with FP8 correction ({doc}`/advance/fp8_quantization`) when rollouts are quantized—both address distinct mismatch sources.

API entry points: {doc}`/api/moe` and {doc}`/api/protocol`.
