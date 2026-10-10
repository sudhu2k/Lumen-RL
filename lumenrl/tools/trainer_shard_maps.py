"""torchrun worker: the ``megatron_native`` trainer's per-rank source maps, on CPU.

One process per trainer rank, gloo only. Each rank builds its Megatron shard on
CPU (``use_cpu_initialization``), loads every id pass the way the engine does
(``_shard_hf_for_moe`` / ``hf_to_megatron`` / ``_shard_hf_for_mp``, then
``load_state_dict``) and decodes its own parameters. ``--real`` instead loads the
real checkpoint the same way and saves the rank's parameters, for verification.

The TransformerConfig below restates the engine's layout-relevant fields. A
mismatch shows up as a ``load_state_dict`` shape error, not as a wrong map.

    torchrun --nproc-per-node 4 -m lumenrl.tools.trainer_shard_maps \\
        --model /data/rl_data/models/Qwen3-8B-Base --out /tmp/maps --tp 2 --pp 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lumenrl.transfer.weight_source_map import CheckpointIndex, capture


class LazyIdState(Mapping):
    """Pass ``k`` of the id checkpoint, generated per tensor on first access.

    A rank's conversion only materializes what it reads (its experts, the tensors
    it slices), not the whole model.
    """

    def __init__(self, index: CheckpointIndex, k) -> None:
        self._index, self._k = index, k
        self._names = [t.name for t in index.tensors]
        self._known = set(self._names)
        self._cache: dict[str, torch.Tensor] = {}

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self._known:
            raise KeyError(name)
        t = self._cache.get(name)
        if t is None:
            t = self._cache[name] = self._index.id_tensor(name, self._k)
        return t

    def __contains__(self, name) -> bool:
        return name in self._known

    def __iter__(self):
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


def _build_model(hf: dict, spec, tp: int, pp: int, ep: int, etp: int, grouped_gemm: bool = True):
    """The engine's GPTModel for this rank, on CPU (layout-relevant fields only)."""
    from megatron.core import parallel_state as mpu
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_decoder_block_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    from lumenrl.engine.training.megatron_base_engine import moe_dispatcher_kwargs
    from lumenrl.engine.training.model_registry import hf_num_experts

    moe = spec.name == "gpt_moe"
    head_dim = hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"])
    kw: dict = {}
    if moe:
        kw = dict(
            num_moe_experts=int(hf_num_experts(hf)),
            moe_ffn_hidden_size=spec.build_dims(hf).moe_ffn,
            moe_router_topk=int(hf.get("num_experts_per_tok") or 2),
            moe_grouped_gemm=grouped_gemm,
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0.0,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=etp,
            moe_router_pre_softmax=False,
        )
        shared = int(hf.get("shared_expert_intermediate_size", 0) or 0)
        if shared:
            kw["moe_shared_expert_intermediate_size"] = shared
    tfcfg = TransformerConfig(
        num_layers=hf["num_hidden_layers"], hidden_size=hf["hidden_size"],
        num_attention_heads=hf["num_attention_heads"],
        num_query_groups=hf["num_key_value_heads"], kv_channels=head_dim,
        ffn_hidden_size=hf["intermediate_size"], gated_linear_unit=True,
        activation_func=F.silu, add_bias_linear=False,
        add_qkv_bias=bool(hf.get("attention_bias", False)),
        normalization="RMSNorm", layernorm_epsilon=hf.get("rms_norm_eps", 1e-6),
        qk_layernorm=True, hidden_dropout=0.0, attention_dropout=0.0,
        bf16=True, params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp,
        sequence_parallel=moe and tp > 1, use_cpu_initialization=True,
        variable_seq_lengths=pp > 1,
        **moe_dispatcher_kwargs({}, tp=tp, cp=1, sp=moe and tp > 1), **kw,
    )
    layer_spec = (get_gpt_decoder_block_spec(tfcfg, use_transformer_engine=True) if moe
                  else get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True))
    return GPTModel(
        config=tfcfg, transformer_layer_spec=layer_spec, vocab_size=hf["vocab_size"],
        max_sequence_length=hf.get("max_position_embeddings", 32768),
        pre_process=mpu.is_pipeline_first_stage(), post_process=mpu.is_pipeline_last_stage(),
        position_embedding_type="rope", rotary_base=hf.get("rope_theta", 1000000.0),
        share_embeddings_and_output_weights=bool(hf.get("tie_word_embeddings", False)),
        parallel_output=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--etp", type=int, default=1)
    ap.add_argument("--grouped-gemm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--real", action="store_true", help="save real-checkpoint params instead")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    import lumenrl.engine.training.model_specs  # noqa: F401  (registers the specs)
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from lumenrl.engine.training.bridges.core import load_hf_safetensors
    from lumenrl.engine.training.bridges.gpt import hf_to_megatron
    from lumenrl.engine.training.megatron_native_engine import MegatronNativeEngine
    from lumenrl.engine.training.model_registry import MODEL_REGISTRY

    dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=args.tp, pipeline_model_parallel_size=args.pp,
        expert_model_parallel_size=args.ep, expert_tensor_parallel_size=args.etp,
        create_gloo_process_groups=False,
    )
    model_parallel_cuda_manual_seed(0)

    hf_cfg = json.load(open(os.path.join(args.model, "config.json")))
    spec = MODEL_REGISTRY.resolve(hf_cfg, {})
    if spec.name not in ("gpt_dense", "gpt_moe"):
        raise NotImplementedError(spec.name)
    moe = spec.name == "gpt_moe"
    model = _build_model(hf_cfg, spec, args.tp, args.pp, args.ep, args.etp, args.grouped_gemm)
    stub = MegatronNativeEngine.__new__(MegatronNativeEngine)
    stub._dims = spec.build_dims(hf_cfg)

    def load(hf) -> dict[str, torch.Tensor]:
        # The engine's branch order (megatron_native_engine, HF load path).
        if moe:
            meg = stub._shard_hf_for_moe(model, hf)
        elif args.tp == 1 and args.pp == 1:
            meg = hf_to_megatron(hf, stub._dims, te=True)
        else:
            meg = stub._shard_hf_for_mp(model, hf)
        missing = model.load_state_dict(meg, strict=False)
        real_missing = [k for k in missing.missing_keys if "_extra_state" not in k]
        if real_missing:
            raise RuntimeError(f"rank {rank}: missing keys {real_missing[:6]}")
        return {n: p.detach() for n, p in model.named_parameters()}

    coords = {"rank": rank, "world": world,
              "tp_rank": mpu.get_tensor_model_parallel_rank(),
              "pp_rank": mpu.get_pipeline_model_parallel_rank(),
              "ep_rank": mpu.get_expert_model_parallel_rank() if moe else 0,
              "dp_rank": mpu.get_data_parallel_rank()}
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    if args.real:
        params = load(load_hf_safetensors(args.model))
        torch.save({"coords": coords, "params": {n: p.clone() for n, p in params.items()}},
                   os.path.join(args.out, f"real_rank{rank}.pt"))
    else:
        index = CheckpointIndex.from_dir(args.model)
        log = (lambda s: print(f"[rank {rank}] {s}", flush=True)) if rank == 0 else (lambda s: None)
        maps, _ = capture(index, lambda k: {"loader": load(LazyIdState(index, k))}, log=log)
        m = maps["loader"]
        m.meta["coords"] = coords
        m.save(os.path.join(args.out, f"trainer_rank{rank}.pt"))
        print(f"[rank {rank}] {coords} {m.stats()} {time.time() - t0:.1f}s", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
