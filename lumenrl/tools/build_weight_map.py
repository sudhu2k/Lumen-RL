"""Build the MORI weight-sync maps and plan offline, on CPU.

Runs the id checkpoint through both sides' own loading code (the trainer's
``megatron_native`` conversion and vLLM's ``load_weights`` plus post-processing on
a CPU-built model), decodes the source maps, joins them through the checkpoint and
plans the reads. ``--verify`` then loads the real checkpoint on both sides and
executes the plan with local copies: every rollout parameter must come out
bitwise equal to what vLLM's own loader produces.

Supported so far: dense and Qwen3-MoE models. The trainer side takes TP, PP, EP and
DP (DP ranks are replicas). The vLLM side takes TP, PP, DP, prefill context parallel
and expert parallel (``--vllm-ep``); ranks whose loaded weights differ each get their
own plan, and identical ranks share one. Decode context parallel subdivides TP.

Run inside the release image, e.g.::

    python -m lumenrl.tools.build_weight_map --model /data/rl_data/models/Qwen3-8B-Base \\
        --out /data/rl_data/weight_maps/qwen3-8b --n-trainer 4 --n-vllm 4 --verify
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from typing import Optional

import torch

from lumenrl.transfer.weight_plan import apply_plan, correspond, plan_rounds
from lumenrl.transfer.weight_source_map import CheckpointIndex, SourceMap, capture

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- trainer side

class TrainerLayout:
    """The ``megatron_native`` engine's weights at TP=1 PP=1 as a source catalog.

    Every distinct trainer tensor appears once. Non-expert tensors are replicated
    on all ranks; routed experts (grouped-GEMM names) are named by their *global*
    index and held only by the ranks of their expert-parallel group, where they
    sit under the local name (``local_name``). The tensors are what
    ``_shard_hf_for_moe`` / ``hf_to_megatron`` load at ETP=1.
    """

    def __init__(self, model_dir: str, n_ranks: int, ep: int = 1,
                 engine_config: Optional[dict] = None) -> None:
        import lumenrl.engine.training.model_specs  # noqa: F401  (registers the specs)
        from lumenrl.engine.training.model_registry import MODEL_REGISTRY

        hf_cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.spec = MODEL_REGISTRY.resolve(hf_cfg, engine_config or {})
        if self.spec.name not in ("gpt_dense", "gpt_moe"):
            raise NotImplementedError(f"trainer side supports gpt_dense and gpt_moe, "
                                      f"not {self.spec.name}")
        self.moe = self.spec.name == "gpt_moe"
        self.dims = self.spec.build_dims(hf_cfg)
        if n_ranks % ep or (self.moe and self.dims.num_experts % ep) or (not self.moe and ep > 1):
            raise ValueError(f"bad layout: ranks={n_ranks} ep={ep} moe={self.moe}")
        self.n, self.ep = n_ranks, ep
        self.num_local = self.dims.num_experts // ep if self.moe else 0

    def catalog(self, hf: dict) -> dict[str, torch.Tensor]:
        from lumenrl.engine.training.bridges import gpt

        if not self.moe:
            return gpt.hf_to_megatron(hf, self.dims, te=True)
        d = self.dims
        m = gpt.non_expert_hf_to_megatron(hf, d)
        for L in range(d.num_layers):
            for e in range(d.num_experts):
                p = f"decoder.layers.{L}.mlp.experts."
                m[p + f"linear_fc1.weight{e}"] = gpt.hf_expert_fc1(hf, d, L, e)
                m[p + f"linear_fc2.weight{e}"] = gpt.hf_expert_fc2(hf, d, L, e)
        return m

    def holders(self, names) -> dict[str, list[int]]:
        from lumenrl.engine.training.bridges.core import expert_local_index

        every = list(range(self.n))
        out = {}
        for n in names:
            exp = expert_local_index(n)
            if exp is None:
                out[n] = every
            else:
                g = exp[0] // self.num_local
                out[n] = [r for r in every if r % self.ep == g]
        return out

    def local_name(self, name: str) -> str:
        from lumenrl.engine.training.bridges.core import expert_local_index, relabel_expert_index

        exp = expert_local_index(name)
        return name if exp is None else relabel_expert_index(name, exp[0] % self.num_local)

    def rank_states(self, catalog: dict[str, torch.Tensor]) -> dict[int, dict[str, torch.Tensor]]:
        """Each rank's tensors under catalog names (only what that rank holds)."""
        h = self.holders(catalog)
        return {r: {n: t for n, t in catalog.items() if r in h[n]} for r in range(self.n)}

    def key(self) -> dict:
        return {"engine": "megatron_native", "spec": self.spec.name, "tp": 1, "pp": 1,
                "ep": self.ep, "ranks": self.n, "grouped_gemm": True}


class SimulatedTrainer:
    """Trainer ranks simulated by ``trainer_shard_maps`` (torchrun, gloo, CPU).

    Each rank's map comes from the engine's own sharding and ``load_state_dict``.
    The catalog keeps one entry per distinct (name, map) pair across ranks, held by
    every rank whose map is identical (data-parallel replicas, TP-replicated norms).
    """

    def __init__(self, model_dir: str, out: str, tp: int, pp: int, ep: int, world: int) -> None:
        self.model, self.out = model_dir, out
        self.tp, self.pp, self.ep, self.n = tp, pp, ep, world
        self.local: dict[str, str] = {}
        self._holders: dict[str, list[int]] = {}

    def _run(self, real: bool) -> None:
        import random
        import subprocess

        cmd = ["torchrun", "--nproc-per-node", str(self.n),
               "--master-port", str(29700 + random.randrange(200)),
               "-m", "lumenrl.tools.trainer_shard_maps", "--model", self.model,
               "--out", os.path.join(self.out, "trainer_ranks"), "--tp", str(self.tp),
               "--pp", str(self.pp), "--ep", str(self.ep),
               "--threads", str(max(1, torch.get_num_threads() // self.n))]
        cmd += ["--real"] if real else []
        subprocess.run(cmd, check=True)

    def capture(self) -> SourceMap:
        from lumenrl.transfer.weight_source_map import ParamMap

        self._run(real=False)
        params: dict[str, ParamMap] = {}
        seen: dict[tuple[str, str], str] = {}
        meta: dict = {}
        for r in range(self.n):
            m = SourceMap.load(os.path.join(self.out, "trainer_ranks", f"trainer_rank{r}.pt"))
            meta = {k: v for k, v in m.meta.items() if k != "coords"}
            for name, p in m.params.items():
                key = (name, _map_fingerprint(p))
                cname = seen.get(key)
                if cname is None:
                    cname = seen[key] = f"{name}@{r}"
                    params[cname] = ParamMap(cname, p.shape, p.dtype, p.runs, p.raw, p.valid)
                    self._holders[cname] = []
                    self.local[cname] = name
                self._holders[cname].append(r)
        return SourceMap(params, meta)

    def holders(self, names) -> dict[str, list[int]]:
        return {n: self._holders[n] for n in names}

    def local_name(self, name: str) -> str:
        return self.local[name]

    def real_states(self) -> dict[int, dict[str, torch.Tensor]]:
        self._run(real=True)
        out: dict[int, dict[str, torch.Tensor]] = {r: {} for r in range(self.n)}
        for r in range(self.n):
            real = torch.load(os.path.join(self.out, "trainer_ranks", f"real_rank{r}.pt"),
                              weights_only=False)["params"]
            for cname, hs in self._holders.items():
                if r in hs:
                    out[r][cname] = real[self.local[cname]]
        return out

    def key(self) -> dict:
        return {"engine": "megatron_native", "simulated": True, "tp": self.tp, "pp": self.pp,
                "ep": self.ep, "ranks": self.n, "grouped_gemm": True}


def _map_fingerprint(p) -> str:
    h = hashlib.sha1(f"{tuple(p.shape)}|{p.dtype}|{p.valid}".encode())
    if p.runs is not None:
        for t in p.runs.to_dict().values():
            h.update(t.contiguous().numpy().tobytes())
    if p.raw is not None:
        h.update(p.raw.contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- vLLM side

class CpuVllm:
    """A vLLM model built on CPU, loaded with the engine's own loader and post-processing.

    ``world > 1`` builds this process's shard. vLLM's world is
    ``dp x pp x prefill_cp x tp``; expert parallel is a flag that splits whole experts
    across that group instead of tensor-sharding each one. Every rank passes
    ``local_rank=0`` because the parameters stay on CPU.
    """

    def __init__(self, model_dir: str, tp: int = 1, pp: int = 1, dp: int = 1, pcp: int = 1,
                 dcp: int = 1, expert_parallel: bool = False, rank: int = 0,
                 world: int = 1) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.distributed import init_distributed_environment, initialize_model_parallel
        from vllm.engine.arg_utils import EngineArgs

        self.tp, self.pp, self.dp, self.pcp, self.dcp = tp, pp, dp, pcp, dcp
        self.expert_parallel = expert_parallel
        args = dict(model=model_dir, dtype="bfloat16", enforce_eager=True, max_model_len=2048)
        if world > 1:
            args.update(tensor_parallel_size=tp, pipeline_parallel_size=pp,
                        data_parallel_size=dp, prefill_context_parallel_size=pcp,
                        decode_context_parallel_size=dcp, enable_expert_parallel=expert_parallel,
                        distributed_executor_backend="external_launcher")
        self.cfg = EngineArgs(**args).create_engine_config()
        self.cfg.load_config.device = "cpu"
        self._ctx = set_current_vllm_config(self.cfg)
        self._ctx.__enter__()
        from datetime import timedelta

        # A torchrun worker joins the store torchrun already started. Opening a second
        # store on WEIGHT_MAP_PORT leaves every rank a client of a port nobody listens
        # on. The single-process capture has no torchrun, so it opens its own.
        # One GPU is visible and the weights stay on CPU, so every rank reports
        # local rank 0 and the model-parallel groups do not open an NCCL communicator.
        if world > 1:
            import vllm.distributed.parallel_state as parallel_state

            method = "env://"
            original = parallel_state.init_model_parallel_group

            def groups_without_nccl(*args, **kwargs):
                kwargs["use_device_communicator"] = False
                return original(*args, **kwargs)

            parallel_state.init_model_parallel_group = groups_without_nccl
        else:
            port = os.environ.get("WEIGHT_MAP_PORT", "29591")
            method = f"tcp://127.0.0.1:{port}"
        init_distributed_environment(world_size=world, rank=rank, local_rank=0, backend="gloo",
                                     distributed_init_method=method, timeout=timedelta(seconds=120))
        initialize_model_parallel(tp, pp, prefill_context_model_parallel_size=pcp,
                                  decode_context_model_parallel_size=dcp)

    def load(self, weights, keep_loader_stage: bool):
        from vllm.model_executor.model_loader.utils import (
            initialize_model,
            process_weights_after_loading,
        )
        from vllm.utils.torch_utils import set_default_torch_dtype

        self.cfg.compilation_config.static_forward_context.clear()
        with set_default_torch_dtype(torch.bfloat16), torch.device("cpu"):
            model = initialize_model(vllm_config=self.cfg, model_config=self.cfg.model_config)
        model.load_weights(weights(model) if callable(weights) else weights)
        loader = ({n: p.detach().clone() for n, p in model.named_parameters()}
                  if keep_loader_stage else None)
        process_weights_after_loading(model, self.cfg.model_config, torch.device("cpu"))
        final = {n: p.detach() for n, p in model.named_parameters()}
        return loader, final

    def real_weights(self):
        """The real checkpoint through vLLM's own weight iterator, bound to the model."""
        from vllm.model_executor.model_loader import get_model_loader

        loader = get_model_loader(self.cfg.load_config)
        return lambda model: loader.get_all_weights(self.cfg.model_config, model)

    def key(self) -> dict:
        import vllm

        out = {"vllm": vllm.__version__,
               "aiter": {k: os.environ.get(k) for k in sorted(os.environ)
                         if k.startswith("VLLM_ROCM_USE_AITER")},
               "dtype": "bfloat16", "tp": self.tp}
        if (self.pp, self.dp, self.pcp, self.dcp, self.expert_parallel) != (1, 1, 1, 1, False):
            out.update(pp=self.pp, dp=self.dp, pcp=self.pcp, dcp=self.dcp,
                       expert_parallel=self.expert_parallel)
        return out


class VllmRanks:
    """One vLLM process per rank (``vllm_shard_maps``), grouped by identical weights.

    Tensor, pipeline and expert parallel put different weights on different ranks, so
    each distinct rank gets its own map and plan. Data-parallel and context-parallel
    ranks that loaded the same bytes share one.
    """

    def __init__(self, model_dir: str, out: str, tp: int, pp: int, dp: int, pcp: int,
                 dcp: int, expert_parallel: bool, threads: int) -> None:
        self.model, self.out = model_dir, out
        self.tp, self.pp, self.dp, self.pcp, self.dcp = tp, pp, dp, pcp, dcp
        self.expert_parallel, self.threads = expert_parallel, threads
        self.world = tp * pp * dp * pcp
        self.groups: list[dict] = []

    def _run(self, real: bool) -> None:
        import random
        import subprocess

        cmd = ["torchrun", "--nproc-per-node", str(self.world),
               "--master-port", str(29700 + random.randrange(200)),
               "-m", "lumenrl.tools.vllm_shard_maps", "--model", self.model,
               "--out", os.path.join(self.out, "vllm_ranks"), "--tp", str(self.tp),
               "--pp", str(self.pp), "--dp", str(self.dp), "--pcp", str(self.pcp),
               "--dcp", str(self.dcp), "--threads", str(max(1, self.threads // self.world))]
        if self.expert_parallel:
            cmd.append("--ep")
        if real:
            cmd.append("--real")
        subprocess.run(cmd, check=True)

    def capture(self) -> list[dict]:
        self._run(real=False)
        seen: dict[str, int] = {}
        for r in range(self.world):
            m = SourceMap.load(os.path.join(self.out, "vllm_ranks", f"vllm_rank{r}.pt"))
            fp = _rank_fingerprint(m)
            gi = seen.get(fp)
            if gi is None:
                seen[fp] = gi = len(self.groups)
                self.groups.append({"ranks": [], "map": m, "classes": m.meta.get("classes", {})})
            self.groups[gi]["ranks"].append(r)
        return self.groups

    def real_params(self, rank: int) -> dict:
        return torch.load(os.path.join(self.out, "vllm_ranks", f"real_rank{rank}.pt"),
                          weights_only=False)["params"]

    def key(self) -> dict:
        import vllm

        return {"vllm": vllm.__version__,
                "aiter": {k: os.environ.get(k) for k in sorted(os.environ)
                          if k.startswith("VLLM_ROCM_USE_AITER")},
                "dtype": "bfloat16", "tp": self.tp, "pp": self.pp, "dp": self.dp,
                "pcp": self.pcp, "dcp": self.dcp, "expert_parallel": self.expert_parallel}


def _rank_fingerprint(m: SourceMap) -> str:
    h = hashlib.sha1()
    for name in sorted(m.params):
        h.update(name.encode())
        h.update(_map_fingerprint(m.params[name]).encode())
    h.update(json.dumps(m.meta.get("classes", {}), sort_keys=True).encode())
    return h.hexdigest()


# ---------------------------------------------------------------- main

def cache_key(index: CheckpointIndex, trainer: dict, vllm: dict, plan: dict) -> str:
    blob = json.dumps({"checkpoint": index.fingerprint(), "trainer": trainer, "vllm": vllm,
                       "plan": plan}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-trainer", type=int, default=4, help="trainer ranks (sources)")
    ap.add_argument("--trainer-ep", type=int, default=1, help="trainer expert parallelism")
    ap.add_argument("--trainer-tp", type=int, default=1, help="trainer tensor parallelism")
    ap.add_argument("--trainer-pp", type=int, default=1, help="trainer pipeline parallelism")
    ap.add_argument("--simulate-trainer", action="store_true",
                    help="build the Megatron shards per rank (implied by TP or PP > 1)")
    ap.add_argument("--n-vllm", type=int, default=4,
                    help="identical vLLM replicas, used when no vLLM parallel size is set")
    ap.add_argument("--vllm-tp", type=int, default=1)
    ap.add_argument("--vllm-pp", type=int, default=1)
    ap.add_argument("--vllm-dp", type=int, default=1)
    ap.add_argument("--vllm-pcp", type=int, default=1, help="vLLM prefill context parallel")
    ap.add_argument("--vllm-dcp", type=int, default=1, help="vLLM decode context parallel")
    ap.add_argument("--vllm-ep", action="store_true", help="vLLM enable_expert_parallel")
    ap.add_argument("--round-mb", type=int, default=512)
    ap.add_argument("--max-piece-mb", type=int, default=32)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--threads", type=int, default=64)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    report: dict = {"model": args.model}

    index = CheckpointIndex.from_dir(args.model)
    report["checkpoint"] = {"tensors": len(index.tensors), "elements": index.total,
                            "passes": index.passes}
    log(f"checkpoint: {len(index.tensors)} tensors, {index.total} elements, "
        f"{index.passes} passes")

    simulate = args.simulate_trainer or args.trainer_tp > 1 or args.trainer_pp > 1
    t0 = time.time()
    if simulate:
        trainer = SimulatedTrainer(args.model, args.out, args.trainer_tp, args.trainer_pp,
                                   args.trainer_ep, args.n_trainer)
        tmaps = {"loader": trainer.capture()}
    else:
        trainer = TrainerLayout(args.model, args.n_trainer, args.trainer_ep)
        tmaps, _ = capture(index, lambda k: {"loader": trainer.catalog(index.id_state(k))},
                           log=lambda s: log("trainer " + s))
    report["trainer"] = {"build_s": round(time.time() - t0, 1), "layout": trainer.key(),
                         **tmaps["loader"].stats()}
    tmaps["loader"].save(os.path.join(args.out, "trainer_map.pt"))
    log(f"trainer map: {report['trainer']}")

    vllm_sharded = (args.vllm_tp, args.vllm_pp, args.vllm_dp, args.vllm_pcp) != (1, 1, 1, 1) \
        or args.vllm_dcp > 1 or args.vllm_ep
    if vllm_sharded:
        ranks = VllmRanks(args.model, args.out, args.vllm_tp, args.vllm_pp, args.vllm_dp,
                          args.vllm_pcp, args.vllm_dcp, args.vllm_ep, args.threads)
        t0 = time.time()
        groups = ranks.capture()
        report["vllm"] = {"build_s": round(time.time() - t0, 1), "world": ranks.world,
                          "groups": [{"ranks": g["ranks"], **g["map"].stats()} for g in groups]}
        log(f"vllm maps: {len(groups)} distinct of {ranks.world} ranks, {report['vllm']['build_s']}s")
        vllm_key = ranks.key()
    else:
        vm = CpuVllm(args.model)

        def vllm_pass(k):
            loader, final = vm.load(index.id_tensors(k), keep_loader_stage=True)
            return {"loader": loader, "final": final}

        t0 = time.time()
        vmaps, classes = capture(index, vllm_pass, compare=("loader", "final"),
                                 log=lambda s: log("vllm " + s))
        kinds: dict[str, int] = {}
        for c in classes.values():
            kinds[c] = kinds.get(c, 0) + 1
        report["vllm"] = {"build_s": round(time.time() - t0, 1), "classes": kinds,
                          "non_identity": sorted(n for n, c in classes.items() if c != "identity"),
                          **vmaps["loader"].stats()}
        vmaps["loader"].save(os.path.join(args.out, "vllm_map.pt"))
        log(f"vllm map: {report['vllm']}")
        groups = [{"ranks": list(range(args.n_vllm)), "map": vmaps["loader"], "classes": classes,
                   "single_ref": True}]
        vllm_key = vm.key()
        ranks = None

    plans = []
    t0 = time.time()
    for gi, g in enumerate(groups):
        classes = g["classes"]
        identity = [n for n, c in classes.items()
                    if c == "identity" and g["map"].params[n].valid]
        copies = correspond(tmaps["loader"], g["map"], sorted(identity))
        if copies.report["overlapping_src_runs"]:
            raise RuntimeError(f"{copies.report['overlapping_src_runs']} trainer runs overlap")
        plan = plan_rounds(copies, args.n_trainer, len(g["ranks"]), args.round_mb << 20,
                           args.max_piece_mb << 20, holders=trainer.holders(copies.src_names))
        plan.meta["src_local_names"] = {n: trainer.local_name(n) for n in plan.src_names}
        plan.meta["vllm_ranks"] = g["ranks"]
        plans.append((plan, copies))
        name = "plan.pt" if len(groups) == 1 else f"plan_group{gi}.pt"
        tr = plan.meta["traffic"]
        log(f"plan {name}: ranks {g['ranks']}, {copies.report['copies']} copies "
            f"({copies.report['line_copies']} as lines), {tr['transport_reads']} transport reads")
    report["plan"] = {"build_s": round(time.time() - t0, 1),
                      "groups": [{"ranks": g["ranks"], "correspond": c.report,
                                  "traffic": p.meta["traffic"]}
                                 for (p, c), g in zip(plans, groups)]}
    plan_key = {"n_vllm": args.n_vllm, "round_mb": args.round_mb, "max_piece_mb": args.max_piece_mb}
    if vllm_sharded:
        plan_key["vllm_parallel"] = {k: getattr(args, k) for k in
                                     ("vllm_tp", "vllm_pp", "vllm_dp", "vllm_pcp", "vllm_dcp")}
        plan_key["vllm_ep"] = args.vllm_ep
    key = cache_key(index, trainer.key(), vllm_key, plan_key)
    report["key"] = key
    for gi, (plan, _) in enumerate(plans):
        plan.meta["key"] = key
        plan.save(os.path.join(args.out, "plan.pt" if len(plans) == 1 else f"plan_group{gi}.pt"))
    log(f"key {key}")

    if args.verify:
        del tmaps
        if not vllm_sharded:
            del vmaps
        for g in groups:
            g.pop("map", None)
        gc.collect()
        from lumenrl.engine.training.bridges.core import load_hf_safetensors

        t0 = time.time()
        src = (trainer.real_states() if simulate else
               trainer.rank_states(trainer.catalog(load_hf_safetensors(args.model))))
        if ranks is not None:
            ranks._run(real=True)
        else:
            _, single = vm.load(vm.real_weights(), keep_loader_stage=False)
        res = {"params": 0, "equal": 0, "mismatch": [], "coverage_short": [], "not_planned": []}
        for (plan, _), g in zip(plans, groups):
            for d, r in enumerate(g["ranks"]):
                ref = single if ranks is None else ranks.real_params(r)
                got = {n: torch.zeros_like(ref[n]) for n in plan.dst_names}
                written = apply_plan(plan, d, src, got)
                for n in plan.dst_names:
                    res["params"] += 1
                    if torch.equal(got[n], ref[n]):
                        res["equal"] += 1
                    else:
                        res["mismatch"].append((r, n))
                    if written.get(n, 0) != ref[n].numel() * ref[n].element_size():
                        res["coverage_short"].append((r, n, written.get(n, 0)))
                res["not_planned"] = sorted(set(res["not_planned"]) | (set(ref) - set(plan.dst_names)))
                del got
                if ranks is not None:
                    del ref
        res["mismatch"], res["coverage_short"] = res["mismatch"][:20], res["coverage_short"][:20]
        res["verify_s"] = round(time.time() - t0, 1)
        report["verify"] = res
        log(f"verify: {res['equal']}/{res['params']} equal, not planned {res['not_planned']}")

    report["total_s"] = round(time.time() - _T0, 1)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
