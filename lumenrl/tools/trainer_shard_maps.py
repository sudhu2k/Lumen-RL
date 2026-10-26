"""torchrun worker: the ``megatron_native`` trainer's per-rank source maps, on CPU.

One process per trainer rank, gloo only. Each rank builds its shard with the engine's
own code (``MegatronNativeEngine._init_parallel`` / ``_build_model`` /
``_hf_state_for_rank``, then ``load_state_dict``) from the run's resolved engine
config, so every layout option the engine reads is the one the plan sees. It then
loads every id pass and decodes its own parameters. ``--real`` instead loads the
real checkpoint the same way and saves the rank's parameters, for verification.

Data-parallel copies are built once. A rank builds only if it is the first rank of
its data-parallel group (context parallel included) or of its expert data-parallel
group, or shares a pipeline group with one that does (model construction can
communicate within it). The others write their replica leaders and stop;
``--all-ranks`` builds every rank instead, to check that the copies really match.

    torchrun --nproc-per-node 4 -m lumenrl.tools.trainer_shard_maps \\
        --model /data/rl_data/models/Qwen3-8B-Base --out /tmp/maps \\
        --engine-config /tmp/maps/trainer_engine_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping

import torch
import torch.distributed as dist

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


def _first_rank(group) -> int:
    return min(dist.get_process_group_ranks(group))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-config", required=True,
                    help="JSON engine config, as the actor worker passes it to the engine")
    ap.add_argument("--real", action="store_true", help="save real-checkpoint params instead")
    ap.add_argument("--all-ranks", action="store_true", help="build data-parallel copies too")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from megatron.core import parallel_state as mpu

    from lumenrl.engine.training.bridges.core import load_hf_safetensors
    from lumenrl.engine.training.megatron_native_engine import MegatronNativeEngine

    with open(args.engine_config) as f:
        engine_config = json.load(f)
    dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    engine = MegatronNativeEngine(model_config={}, engine_config=engine_config,
                                  optimizer_config={}, model_name=args.model)
    engine._init_parallel(create_gloo_process_groups=False)

    dense_leader = _first_rank(mpu.get_data_parallel_group(with_context_parallel=True))
    expert_leader = _first_rank(mpu.get_expert_data_parallel_group())
    builds = torch.tensor([int(args.all_ranks or rank in (dense_leader, expert_leader))])
    dist.all_reduce(builds, op=dist.ReduceOp.MAX, group=mpu.get_pipeline_model_parallel_group())
    coords = {"rank": rank, "world": world, "built": bool(builds.item()),
              "dense_leader": dense_leader, "expert_leader": expert_leader,
              "tp_rank": mpu.get_tensor_model_parallel_rank(),
              "pp_rank": mpu.get_pipeline_model_parallel_rank(),
              "ep_rank": mpu.get_expert_model_parallel_rank(),
              "dp_rank": mpu.get_data_parallel_rank()}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"trainer_rank{rank}.json"), "w") as f:
        json.dump(coords, f)

    if coords["built"]:
        model = engine._build_model()
        if not engine._caps.supports_hf_bridge:
            raise NotImplementedError(f"{engine._spec.name}: no HF bridge for this family")

        def load(hf) -> dict[str, torch.Tensor]:
            missing = model.load_state_dict(engine._hf_state_for_rank(model, hf), strict=False)
            real_missing = [k for k in missing.missing_keys if "_extra_state" not in k]
            if real_missing:
                raise RuntimeError(f"rank {rank}: missing keys {real_missing[:6]}")
            return {n: p.detach() for n, p in model.named_parameters()}

        t0 = time.time()
        if args.real:
            params = load(load_hf_safetensors(args.model))
            torch.save({"coords": coords, "params": {n: p.clone() for n, p in params.items()}},
                       os.path.join(args.out, f"real_rank{rank}.pt"))
        else:
            index = CheckpointIndex.from_dir(args.model)
            log = ((lambda s: print(f"[rank {rank}] {s}", flush=True)) if rank == 0
                   else (lambda s: None))
            maps, _ = capture(index, lambda k: {"loader": load(LazyIdState(index, k))}, log=log)
            m = maps["loader"]
            m.meta["coords"] = coords
            m.save(os.path.join(args.out, f"trainer_rank{rank}.pt"))
            print(f"[rank {rank}] {coords} {m.stats()} {time.time() - t0:.1f}s", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
