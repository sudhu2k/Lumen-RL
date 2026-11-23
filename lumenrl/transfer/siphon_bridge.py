"""Lumen-RL glue for ``weight_sync.backend=siphon``.

Calls into the ``siphon`` package (editable install of rocm-siphon). No copy of
siphon's transport or plan code lives here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def require_siphon():
    try:
        import siphon  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "weight_sync.backend=siphon needs the rocm-siphon package on PYTHONPATH "
            "(pip install -e /path/to/rocm-siphon)"
        ) from e


class LumenDispatcher:
    """siphon ``Dispatcher`` over Lumen-RL's worker group and vLLM Ray servers."""

    def __init__(self, actor_wg, servers) -> None:
        self.actor_wg = actor_wg
        self.servers = list(servers)

    async def trainers(self, fn: str, **kw) -> list:
        return await _gather(self.actor_wg.execute_all_async("siphon_call", fn, kw))

    async def rollouts(self, fn: str, per_replica: list[dict]) -> list:
        return await _gather([
            s.collective_rpc.remote("siphon_call", kwargs={"fn": fn, "kw": kw})
            for s, kw in zip(self.servers, per_replica)
        ])


async def _gather(refs) -> list:
    import asyncio

    # Wait for every replica before raising (siphon design-gaps G1): do not
    # abandon in-flight receives on other engines when the first one fails.
    rows = await asyncio.gather(*refs, return_exceptions=True)
    errors = [r for r in rows if isinstance(r, BaseException)]
    if errors:
        raise errors[0]
    return list(rows)


def vllm_engine_kwargs_from_config(config) -> dict:
    """Weight-relevant vLLM engine kwargs from a LumenRL config object."""
    vcfg = config.policy.generation.vllm_cfg
    out = {
        "tensor_parallel_size": int(getattr(vcfg, "tensor_parallel_size", 1) or 1),
        "dtype": str(getattr(vcfg, "dtype", None) or "bfloat16"),
    }
    if bool(getattr(vcfg, "enable_expert_parallel", False)):
        out["enable_expert_parallel"] = True
    quant = getattr(vcfg, "quantization", None)
    if quant:
        out["quantization"] = quant
    # Optional AITER / backend knobs that change loaded weights
    for key in ("attention_backend", "moe_backend", "linear_backend"):
        val = getattr(vcfg, key, None)
        if val is not None:
            out[key] = val
    return out


def megatron_engine_config_from_config(config) -> dict:
    """Subset of megatron_cfg that shapes the trainer's loaded weights."""
    meg = getattr(config.policy.training, "megatron_cfg", None) or {}
    if hasattr(meg, "__dict__") and not isinstance(meg, dict):
        from dataclasses import asdict, is_dataclass
        meg = asdict(meg) if is_dataclass(meg) else dict(vars(meg))
    keys = (
        "tensor_model_parallel_size", "tensor_parallel_size",
        "pipeline_model_parallel_size", "pipeline_parallel_size",
        "context_parallel_size", "expert_model_parallel_size",
        "expert_parallel_size", "expert_tensor_parallel_size",
        "sequence_parallel", "data_parallel", "num_experts",
        "moe_grouped_gemm", "seed", "moe_token_dispatcher_type",
        "moe_flex_dispatcher_backend",
    )
    out = {}
    for k in keys:
        if k in meg and meg[k] is not None:
            out[k] = meg[k]
    # Normalize aliases
    if "tensor_model_parallel_size" not in out and "tensor_parallel_size" in out:
        out["tensor_model_parallel_size"] = out.pop("tensor_parallel_size")
    if "pipeline_model_parallel_size" not in out and "pipeline_parallel_size" in out:
        out["pipeline_model_parallel_size"] = out.pop("pipeline_parallel_size")
    if "expert_model_parallel_size" not in out and "expert_parallel_size" in out:
        out["expert_model_parallel_size"] = out.pop("expert_parallel_size")
    out.setdefault("tensor_model_parallel_size", 1)
    out.setdefault("pipeline_model_parallel_size", 1)
    out.setdefault("context_parallel_size", 1)
    out.setdefault("expert_model_parallel_size", 1)
    out.setdefault("sequence_parallel", False)
    out.setdefault("data_parallel", "ddp")
    out.setdefault("seed", 42)
    out["build_optimizer"] = False
    return out
