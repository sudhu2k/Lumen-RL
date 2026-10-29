"""Build the MORI weight-sync maps and plan offline, on CPU.

Runs the id checkpoint through both sides' own loading code (the trainer's
``megatron_native`` conversion and vLLM's ``load_weights`` plus post-processing on
a CPU-built model), decodes the source maps, joins them through the checkpoint and
plans the reads. ``--verify`` then loads the real checkpoint on both sides and
executes the plan with local copies: every rollout parameter must come out
bitwise equal to what vLLM's own loader produces.

Supported so far: dense and Qwen3-MoE models. Both sides come from the run's own
config (``--config`` plus overrides), resolved by the code the run uses: the trainer
is the ``megatron_native`` engine under its engine config, so it takes whatever layout
that engine takes (data-parallel copies are built once); vLLM gets the rollout's
engine arguments (``vllm_engine_kwargs``: parallel sizes, dtype, backends...), and
its ranks whose loaded weights differ each get their own plan. The rank counts
default to the run's actor and rollout workers. The trainer passes the resolved
configs as JSON instead (``--trainer-engine-config``, ``--vllm-engine-kwargs``).

Run inside the release image, e.g.::

    python -m lumenrl.tools.build_weight_map --model /data/rl_data/models/Qwen3-8B-Base \\
        --out /data/rl_data/weight_maps/qwen3-8b --verify --config run.yaml \\
        policy.training.megatron_cfg.tensor_model_parallel_size=2 \\
        policy.generation.vllm_cfg.tensor_parallel_size=4
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

from lumenrl.transfer.weight_plan import apply_plan, correspond, merge_plans, plan_rounds
from lumenrl.transfer.weight_source_map import CheckpointIndex, SourceMap, apply_view, capture

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def free_port() -> int:
    """A TCP port nothing listens on now, so concurrent builds don't share a store."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------- trainer side

def engine_config_of(config: dict) -> dict:
    """The engine config the actor worker passes to ``MegatronNativeEngine`` for a run
    (``policy.training.megatron_cfg`` resolved by ``LumenActorWorker._build_engine_config``),
    from the run's config as a plain dict."""
    from types import SimpleNamespace

    from lumenrl.workers.actor_worker import LumenActorWorker

    policy = config.get("policy") or {}
    return LumenActorWorker._build_engine_config(
        SimpleNamespace(config=config), "megatron_native", policy.get("training") or {}, policy)


def run_config(config_path: Optional[str], overrides: list[str]):
    """The run's ``LumenRLConfig`` from its YAML plus ``overrides``; without a file, the
    defaults with the ``megatron_native`` trainer plus ``overrides``."""
    from omegaconf import OmegaConf

    from lumenrl.core.config import LumenRLConfig

    if config_path:
        cfg = LumenRLConfig.from_yaml(config_path, overrides=overrides)
    else:
        dotlist = ["policy.training_backend=megatron_native", *overrides]
        cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(LumenRLConfig),
                                                  OmegaConf.from_dotlist(dotlist)))
    backend = str(cfg.policy.training_backend).lower().replace("-", "_")
    if backend != "megatron_native":
        raise NotImplementedError(f"trainer maps cover megatron_native, not {backend}")
    return cfg


class SimulatedTrainer:
    """Trainer ranks simulated by ``trainer_shard_maps`` (torchrun, gloo, CPU).

    Each rank's map comes from the engine's own model construction, sharding and
    ``load_state_dict``, under the run's engine config. Data-parallel copies are not
    built: they take their leaders' maps (dense tensors from the data-parallel leader,
    routed experts from the expert data-parallel leader). The catalog keeps one entry
    per distinct (name, map) pair across ranks, held by every rank whose map is
    identical (data-parallel replicas, TP-replicated norms).
    """

    def __init__(self, model_dir: str, out: str, engine_config: dict, world: int,
                 all_ranks: bool = False) -> None:
        self.model, self.out, self.n, self.all_ranks = model_dir, out, world, all_ranks
        self.engine_config = engine_config
        self.ranks_dir = os.path.join(out, "trainer_ranks")
        os.makedirs(self.ranks_dir, exist_ok=True)
        self.config_path = os.path.join(out, "trainer_engine_config.json")
        with open(self.config_path, "w") as f:
            json.dump(engine_config, f, indent=1, sort_keys=True)
        self.local: dict[str, str] = {}
        self._holders: dict[str, list[int]] = {}
        self.coords: list[dict] = []

    def _run(self, real: bool) -> None:
        import subprocess

        cmd = ["torchrun", "--nproc-per-node", str(self.n), "--master-port", str(free_port()),
               "-m", "lumenrl.tools.trainer_shard_maps", "--model", self.model,
               "--out", self.ranks_dir, "--engine-config", self.config_path,
               "--threads", str(max(1, torch.get_num_threads() // self.n))]
        cmd += ["--real"] if real else []
        cmd += ["--all-ranks"] if self.all_ranks else []
        subprocess.run(cmd, check=True)

    def _source_rank(self, r: int, local_name: str) -> int:
        """The rank whose map (and real tensor) rank ``r`` shares for this parameter."""
        from lumenrl.engine.training.bridges.core import expert_local_index

        c = self.coords[r]
        if c["built"]:
            return r
        return c["expert_leader"] if expert_local_index(local_name) else c["dense_leader"]

    def capture(self) -> SourceMap:
        from lumenrl.transfer.weight_source_map import ParamMap

        self._run(real=False)
        self.coords = [json.load(open(os.path.join(self.ranks_dir, f"trainer_rank{r}.json")))
                       for r in range(self.n)]
        params: dict[str, ParamMap] = {}
        seen: dict[tuple[str, str], str] = {}
        held: dict[int, list[str]] = {}
        meta: dict = {}
        for r in range(self.n):
            if not self.coords[r]["built"]:
                continue
            m = SourceMap.load(os.path.join(self.ranks_dir, f"trainer_rank{r}.pt"))
            meta = {k: v for k, v in m.meta.items() if k != "coords"}
            held[r] = []
            for name, p in m.params.items():
                key = (name, _map_fingerprint(p))
                cname = seen.get(key)
                if cname is None:
                    cname = seen[key] = f"{name}@{r}"
                    params[cname] = ParamMap(cname, p.shape, p.dtype, p.runs, p.raw, p.valid)
                    self._holders[cname] = []
                    self.local[cname] = name
                self._holders[cname].append(r)
                held[r].append(cname)
        for r in range(self.n):
            if self.coords[r]["built"]:
                continue
            leaders = {self.coords[r]["dense_leader"], self.coords[r]["expert_leader"]}
            for cname in {c for s in leaders for c in held[s]}:
                if self._source_rank(r, self.local[cname]) in self._holders[cname]:
                    self._holders[cname].append(r)
        for hs in self._holders.values():
            hs.sort()
        return SourceMap(params, meta)

    def holders(self, names) -> dict[str, list[int]]:
        return {n: self._holders[n] for n in names}

    def local_name(self, name: str) -> str:
        return self.local[name]

    def real_states(self) -> dict[int, dict[str, torch.Tensor]]:
        self._run(real=True)
        files: dict[int, dict[str, torch.Tensor]] = {}
        out: dict[int, dict[str, torch.Tensor]] = {r: {} for r in range(self.n)}
        for cname, hs in self._holders.items():
            for r in hs:
                s = self._source_rank(r, self.local[cname])
                if s not in files:
                    files[s] = torch.load(os.path.join(self.ranks_dir, f"real_rank{s}.pt"),
                                          weights_only=False)["params"]
                out[r][cname] = files[s][self.local[cname]]
        return out

    def key(self) -> dict:
        return trainer_key(self.engine_config, self.n)


def _map_fingerprint(p) -> str:
    h = hashlib.sha1(f"{tuple(p.shape)}|{p.dtype}|{p.valid}".encode())
    if p.runs is not None:
        for t in p.runs.to_dict().values():
            h.update(t.contiguous().numpy().tobytes())
    if p.raw is not None:
        h.update(p.raw.contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- vLLM side

# ---------------------------------------------------------------- vLLM side

# Rollout engine arguments that change serving but not the loaded weights.
VLLM_RUNTIME_ARGS = frozenset({
    "gpu_memory_utilization", "enforce_eager", "disable_custom_all_reduce",
    "enable_chunked_prefill", "enable_prefix_caching", "max_num_batched_tokens",
    "max_num_seqs", "enable_sleep_mode", "disable_log_stats", "max_model_len",
    "enable_return_routed_experts", "seed",
})
VLLM_PARALLEL_ARGS = {"tensor_parallel_size": "tp", "pipeline_parallel_size": "pp",
                      "data_parallel_size": "dp", "prefill_context_parallel_size": "pcp",
                      "decode_context_parallel_size": "dcp",
                      "enable_expert_parallel": "expert_parallel"}


def split_vllm_args(engine_kwargs: dict) -> tuple[dict, dict]:
    """A rollout's vLLM engine arguments as (parallel sizes keyed by ``CpuVllm``'s
    names, the other arguments that can change the loaded weights)."""
    parallel = {a: engine_kwargs[k] for k, a in VLLM_PARALLEL_ARGS.items() if k in engine_kwargs}
    rest = {k: v for k, v in engine_kwargs.items()
            if k not in VLLM_PARALLEL_ARGS and k not in VLLM_RUNTIME_ARGS}
    return parallel, rest


class CpuVllm:
    """A vLLM model built on CPU, loaded with the engine's own loader and post-processing.

    ``world > 1`` builds this process's shard. vLLM's world is
    ``dp x pp x prefill_cp x tp``; expert parallel is a flag that splits whole experts
    across that group instead of tensor-sharding each one. Every rank passes
    ``local_rank=0`` because the parameters stay on CPU. ``engine_args`` are the
    rollout's weight-relevant engine arguments (``split_vllm_args``).
    """

    def __init__(self, model_dir: str, tp: int = 1, pp: int = 1, dp: int = 1, pcp: int = 1,
                 dcp: int = 1, expert_parallel: bool = False, rank: int = 0,
                 world: int = 1, engine_args: Optional[dict] = None) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.distributed import init_distributed_environment, initialize_model_parallel
        from vllm.engine.arg_utils import EngineArgs

        self.tp, self.pp, self.dp, self.pcp, self.dcp = tp, pp, dp, pcp, dcp
        self.expert_parallel = expert_parallel
        self.engine_args = {"dtype": "bfloat16", **(engine_args or {})}
        args = dict(model=model_dir, **self.engine_args, enforce_eager=True, max_model_len=2048)
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
            port = os.environ.get("WEIGHT_MAP_PORT") or str(free_port())
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
        with set_default_torch_dtype(self.cfg.model_config.dtype), torch.device("cpu"):
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
        return vllm_key(self.tp, self.pp, self.dp, self.pcp, self.dcp, self.expert_parallel,
                        self.engine_args)


class VllmRanks:
    """One vLLM process per rank (``vllm_shard_maps``), grouped by identical weights.

    Tensor, pipeline and expert parallel put different weights on different ranks, so
    each distinct rank gets its own map and plan. Data-parallel and context-parallel
    ranks that loaded the same bytes share one.
    """

    def __init__(self, model_dir: str, out: str, tp: int, pp: int, dp: int, pcp: int,
                 dcp: int, expert_parallel: bool, threads: int,
                 engine_args: Optional[dict] = None) -> None:
        self.model, self.out = model_dir, out
        self.engine_args = {"dtype": "bfloat16", **(engine_args or {})}
        self.tp, self.pp, self.dp, self.pcp, self.dcp = tp, pp, dp, pcp, dcp
        self.expert_parallel, self.threads = expert_parallel, threads
        self.world = tp * pp * dp * pcp
        self.groups: list[dict] = []

    def _run(self, real: bool) -> None:
        import subprocess

        cmd = ["torchrun", "--nproc-per-node", str(self.world), "--master-port", str(free_port()),
               "-m", "lumenrl.tools.vllm_shard_maps", "--model", self.model,
               "--out", os.path.join(self.out, "vllm_ranks"), "--tp", str(self.tp),
               "--pp", str(self.pp), "--dp", str(self.dp), "--pcp", str(self.pcp),
               "--dcp", str(self.dcp), "--engine-args", json.dumps(self.engine_args),
               "--threads", str(max(1, self.threads // self.world))]
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
        return vllm_key(self.tp, self.pp, self.dp, self.pcp, self.dcp, self.expert_parallel,
                        self.engine_args)


def _rank_fingerprint(m: SourceMap) -> str:
    h = hashlib.sha1()
    for name in sorted(m.params):
        h.update(name.encode())
        h.update(_map_fingerprint(m.params[name]).encode())
    h.update(json.dumps(m.meta.get("classes", {}), sort_keys=True).encode())
    h.update(json.dumps(m.meta.get("permutations", {}), sort_keys=True).encode())
    return h.hexdigest()


def view_summary(m: SourceMap) -> list[dict]:
    """The distinct ``permutation`` views in a vLLM map and how many parameters use each."""
    count: dict[str, int] = {}
    for view in m.meta.get("permutations", {}).values():
        k = json.dumps(view, sort_keys=True)
        count[k] = count.get(k, 0) + 1
    return [{**json.loads(k), "params": c} for k, c in sorted(count.items())]


# ---------------------------------------------------------------- main

def trainer_key(engine_config: dict, n_trainer: int) -> dict:
    return {"engine": "megatron_native", "ranks": n_trainer, "engine_config": engine_config}


def vllm_key(tp: int = 1, pp: int = 1, dp: int = 1, pcp: int = 1, dcp: int = 1,
             expert_parallel: bool = False, engine_args: Optional[dict] = None) -> dict:
    import vllm

    return {"vllm": vllm.__version__,
            "aiter": {k: os.environ.get(k) for k in sorted(os.environ)
                      if k.startswith("VLLM_ROCM_USE_AITER")},
            "tp": tp, "pp": pp, "dp": dp, "pcp": pcp, "dcp": dcp,
            "expert_parallel": expert_parallel,
            "engine_args": {"dtype": "bfloat16", **(engine_args or {})}}


def checkpoint_dtype(model_dir: str) -> Optional[str]:
    """The ``torch_dtype`` (or ``dtype``) the model's ``config.json`` declares."""
    with open(os.path.join(model_dir, "config.json")) as f:
        hf = json.load(f)
    for c in (hf, hf.get("text_config") or {}):
        for k in ("torch_dtype", "dtype"):
            if c.get(k):
                return str(c[k]).removeprefix("torch.")
    return None


def check_engine_args(engine_kwargs: dict, model_dir: str) -> None:
    """Reject vLLM settings the maps can't serve: the plan copies BF16 bytes into vLLM's
    parameters as loaded, so they must be BF16 and unquantized (computed parameters,
    design 2.4, are not implemented)."""
    if engine_kwargs.get("quantization"):
        raise NotImplementedError(
            f"MORI weight sync copies BF16 weights; vLLM quantization="
            f"{engine_kwargs['quantization']!r} is not supported yet")
    dtype = str(engine_kwargs.get("dtype") or "auto")
    if dtype == "auto":
        dtype = checkpoint_dtype(model_dir) or "unknown"
    if dtype not in ("bfloat16", "bf16"):
        raise NotImplementedError(f"MORI weight sync copies BF16 weights; vLLM dtype is {dtype}")


_PLAN_CODE = ("lumenrl/engine/training", "lumenrl/transfer/weight_source_map.py",
              "lumenrl/transfer/weight_plan.py", "lumenrl/tools/build_weight_map.py",
              "lumenrl/tools/trainer_shard_maps.py", "lumenrl/tools/vllm_shard_maps.py")
_PLAN_DISTS = {"torch": ("torch",), "megatron_core": ("megatron-core", "megatron_core"),
               "transformer_engine": ("transformer-engine", "transformer_engine")}


def code_versions() -> dict:
    """What decides the trainer side's layout besides the config: a hash of Lumen-RL's
    engine, transfer and tool sources (edits count, unrelated files don't), and the
    installed torch, Megatron-core and TransformerEngine versions."""
    from importlib import metadata

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    paths = []
    for sub in _PLAN_CODE:
        top = os.path.join(root, sub)
        if os.path.isfile(top):
            paths.append(top)
        for d, dirs, files in os.walk(top):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            paths += [os.path.join(d, f) for f in files if f.endswith(".py")]
    h = hashlib.sha1()
    for path in sorted(paths):
        h.update(os.path.relpath(path, root).encode())
        with open(path, "rb") as fh:
            h.update(fh.read())
    out = {"lumenrl_sources": h.hexdigest()}
    for key, names in _PLAN_DISTS.items():
        out[key] = None
        for name in names:
            try:
                out[key] = metadata.version(name)
                break
            except metadata.PackageNotFoundError:
                pass
    return out


def cache_key(index: CheckpointIndex, model_dir: str, trainer: dict, vllm: dict,
              plan: dict) -> str:
    with open(os.path.join(model_dir, "config.json"), "rb") as f:
        model_config = hashlib.sha1(f.read()).hexdigest()
    blob = json.dumps({"checkpoint": index.fingerprint(), "model_config": model_config,
                       "code": code_versions(), "trainer": trainer, "vllm": vllm,
                       "plan": plan}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def build(args, engine_config: dict, index: CheckpointIndex, key: str, out: str) -> None:
    """Maps, plan (``out/plan.pt``) and ``out/report.json``; with ``--verify`` also the
    bitwise check against both engines' own loading."""
    os.makedirs(out, exist_ok=True)
    report: dict = {"model": args.model, "key": key}
    report["checkpoint"] = {"tensors": len(index.tensors), "elements": index.total,
                            "passes": index.passes}
    log(f"checkpoint: {len(index.tensors)} tensors, {index.total} elements, "
        f"{index.passes} passes")

    t0 = time.time()
    trainer = SimulatedTrainer(args.model, out, engine_config, args.n_trainer,
                               args.all_trainer_ranks)
    tmaps = {"loader": trainer.capture()}
    report["trainer"] = {"build_s": round(time.time() - t0, 1), "layout": trainer.key(),
                         "built_ranks": [c["rank"] for c in trainer.coords if c["built"]],
                         **tmaps["loader"].stats()}
    tmaps["loader"].save(os.path.join(out, "trainer_map.pt"))
    log(f"trainer map: {report['trainer']}")

    vllm_sharded = (args.vllm_tp, args.vllm_pp, args.vllm_dp, args.vllm_pcp) != (1, 1, 1, 1) \
        or args.vllm_dcp > 1 or args.vllm_ep
    if vllm_sharded:
        ranks = VllmRanks(args.model, out, args.vllm_tp, args.vllm_pp, args.vllm_dp,
                          args.vllm_pcp, args.vllm_dcp, args.vllm_ep, args.threads,
                          args.vllm_engine_args)
        t0 = time.time()
        groups = ranks.capture()
        report["vllm"] = {"build_s": round(time.time() - t0, 1), "world": ranks.world,
                          "groups": [{"ranks": g["ranks"], **g["map"].stats()} for g in groups]}
        log(f"vllm maps: {len(groups)} distinct of {ranks.world} ranks, {report['vllm']['build_s']}s")
    else:
        vm = CpuVllm(args.model, engine_args=args.vllm_engine_args)

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
                          "permutations": view_summary(vmaps["loader"]),
                          **vmaps["loader"].stats()}
        vmaps["loader"].save(os.path.join(out, "vllm_map.pt"))
        log(f"vllm map: {report['vllm']}")
        groups = [{"ranks": [0], "map": vmaps["loader"], "classes": classes}]
        ranks = None
    world = ranks.world if ranks is not None else 1

    plans = []
    t0 = time.time()
    for g in groups:
        classes = g["classes"]
        views = g["map"].meta.get("permutations", {})
        # Permuted parameters are read in their loader-stage layout (the map is the
        # loader stage's) and moved into place by their view on the rollout GPU.
        synced = sorted(n for n, c in classes.items()
                        if c in ("identity", "permutation") and g["map"].params[n].valid)
        copies = correspond(tmaps["loader"], g["map"], synced)
        if copies.report["overlapping_src_runs"]:
            raise RuntimeError(f"{copies.report['overlapping_src_runs']} trainer runs overlap")
        plan = plan_rounds(copies, args.n_trainer, len(g["ranks"]) * args.n_vllm,
                           args.round_mb << 20, args.max_piece_mb << 20,
                           holders=trainer.holders(copies.src_names), interleave=args.interleave)
        plan.meta["src_local_names"] = {n: trainer.local_name(n) for n in plan.src_names}
        plan.meta["views"] = {n: views[n] for n in synced if classes[n] == "permutation"}
        plan.meta["not_synced"] = sorted(set(classes) - set(synced))
        plans.append((plan, copies))
        tr = plan.meta["traffic"]
        log(f"plan: instance ranks {g['ranks']} x {args.n_vllm} replicas, "
            f"{copies.report['copies']} copies ({copies.report['line_copies']} as lines), "
            f"{tr['transport_reads']} transport reads")
    plan = merge_plans([(p, g["ranks"]) for (p, _), g in zip(plans, groups)], world, args.n_vllm)
    report["plan"] = {"build_s": round(time.time() - t0, 1), "world": world,
                      "replicas": args.n_vllm, "traffic": plan.meta["traffic"],
                      "groups": [{"ranks": g["ranks"], "correspond": c.report,
                                  "traffic": p.meta["traffic"],
                                  "permutations": view_summary(g["map"])}
                                 for (p, c), g in zip(plans, groups)]}
    plan.meta["key"] = key
    plan.save(os.path.join(out, "plan.pt"))
    log(f"plan.pt: {plan.meta['n_dst']} rollout ranks ({args.n_vllm} x {world}), "
        f"{plan.meta['traffic']['transport_reads']} transport reads; key {key}")

    if args.verify:
        del tmaps
        if not vllm_sharded:
            del vmaps
        for g in groups:
            g.pop("map", None)
        gc.collect()
        t0 = time.time()
        src = trainer.real_states()
        if ranks is not None:
            ranks._run(real=True)
        else:
            _, single = vm.load(vm.real_weights(), keep_loader_stage=False)
        res = {"params": 0, "equal": 0, "mismatch": [], "coverage_short": [], "not_planned": [],
               "views": 0, "views_equal": 0}
        dst_col = plan.col("dst_rank")
        for d in range(int(plan.meta["n_dst"])):
            r = d % world
            ref = single if ranks is None else ranks.real_params(r)
            mine = [plan.dst_names[q] for q in sorted(set(plan.col("dst_param")[dst_col == d].tolist()))]
            got = {n: torch.zeros_like(ref[n]) for n in mine}
            written = apply_plan(plan, d, src, got)
            for n in mine:
                res["params"] += 1
                view = plan.meta["views"].get(n)
                if view is not None:
                    res["views"] += 1
                    got[n] = apply_view(view, got[n].view(-1), torch.empty_like(got[n]))
                if torch.equal(got[n], ref[n]):
                    res["views_equal"] += view is not None
                    res["equal"] += 1
                else:
                    res["mismatch"].append((d, n))
                if written.get(n, 0) != ref[n].numel() * ref[n].element_size():
                    res["coverage_short"].append((d, n, written.get(n, 0)))
            res["not_planned"] = sorted(set(res["not_planned"]) | (set(ref) - set(mine)))
            del got
            if ranks is not None:
                del ref
        res["mismatch"], res["coverage_short"] = res["mismatch"][:20], res["coverage_short"][:20]
        res["verify_s"] = round(time.time() - t0, 1)
        report["verify"] = res
        log(f"verify: {res['equal']}/{res['params']} equal, not planned {res['not_planned']}; "
            f"views {res['views_equal']}/{res['views']} equal")

    report["total_s"] = round(time.time() - _T0, 1)
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    bad = report.get("verify", {})
    if bad.get("mismatch") or bad.get("coverage_short"):
        raise RuntimeError(f"verify failed: {bad['mismatch'][:5]} {bad['coverage_short'][:5]}")
    log("done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--out", help="write the maps and plan here")
    where.add_argument("--cache-dir", help="use <cache-dir>/<key>/plan.pt, building it if absent")
    ap.add_argument("--result", help="write {plan, key, cached} as JSON here")
    ap.add_argument("--config", help="the run's LumenRL YAML: the trainer layout is its "
                                     "policy.training.megatron_cfg, the vLLM layout its "
                                     "policy.generation.vllm_cfg")
    ap.add_argument("--n-trainer", type=int,
                    help="trainer ranks (sources); default: the run's actor workers")
    ap.add_argument("--n-vllm", type=int,
                    help="vLLM replicas; default: the run's rollout workers / vLLM world")
    ap.add_argument("--trainer-engine-config",
                    help="the trainer's resolved engine config as JSON, instead of the config")
    ap.add_argument("--vllm-engine-kwargs",
                    help="the rollout's vLLM engine arguments as JSON, instead of the config")
    ap.add_argument("--all-trainer-ranks", action="store_true",
                    help="build data-parallel copies too, to check they match their leaders")
    ap.add_argument("--round-mb", type=int, default=512)
    ap.add_argument("--max-piece-mb", type=int, default=32)
    ap.add_argument("--interleave", action="store_true",
                    help="mix every source's pieces into each round (needs MORI's byte-split "
                         "scatter-gather path)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("overrides", nargs="*",
                    help="config overrides, e.g. policy.training.megatron_cfg."
                         "expert_model_parallel_size=4 "
                         "policy.generation.vllm_cfg.tensor_parallel_size=4")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from dataclasses import asdict

    from lumenrl.trainer.rl_trainer import ray_worker_counts, vllm_engine_kwargs

    given = (args.trainer_engine_config, args.vllm_engine_kwargs, args.n_trainer, args.n_vllm)
    cfg = None if all(x is not None for x in given) else run_config(args.config, args.overrides)
    if args.trainer_engine_config:
        with open(args.trainer_engine_config) as f:
            engine_config = json.load(f)
    else:
        engine_config = engine_config_of(asdict(cfg))
    if args.vllm_engine_kwargs:
        with open(args.vllm_engine_kwargs) as f:
            vllm_kwargs = json.load(f)
    else:
        vllm_kwargs = vllm_engine_kwargs(cfg)
    check_engine_args(vllm_kwargs, args.model)
    parallel, args.vllm_engine_args = split_vllm_args(vllm_kwargs)
    for name in ("tp", "pp", "dp", "pcp", "dcp"):
        setattr(args, f"vllm_{name}", int(parallel.get(name, 1)))
    args.vllm_ep = bool(parallel.get("expert_parallel", False))
    if args.n_trainer is None or args.n_vllm is None:
        actors, rollout = ray_worker_counts(cfg)
        world = args.vllm_tp * args.vllm_pp * args.vllm_dp * args.vllm_pcp
        if args.n_vllm is None and rollout % world:
            raise SystemExit(f"{rollout} rollout workers are not a multiple of the vLLM world {world}")
        args.n_trainer = actors if args.n_trainer is None else args.n_trainer
        args.n_vllm = rollout // world if args.n_vllm is None else args.n_vllm
    log(f"layout: {args.n_trainer} trainer ranks; {args.n_vllm} vLLM replicas of "
        f"TP={args.vllm_tp} PP={args.vllm_pp} DP={args.vllm_dp} PCP={args.vllm_pcp} "
        f"DCP={args.vllm_dcp} EP={args.vllm_ep}; vLLM engine args {args.vllm_engine_args}")
    index = CheckpointIndex.from_dir(args.model)
    plan_params = {"n_vllm": args.n_vllm, "round_mb": args.round_mb,
                   "max_piece_mb": args.max_piece_mb, "interleave": args.interleave}
    key = cache_key(index, args.model, trainer_key(engine_config, args.n_trainer),
                    vllm_key(args.vllm_tp, args.vllm_pp, args.vllm_dp, args.vllm_pcp,
                             args.vllm_dcp, args.vllm_ep, args.vllm_engine_args), plan_params)

    cached = False
    if args.out:
        out = args.out
        build(args, engine_config, index, key, out)
    else:
        out = os.path.join(args.cache_dir, key)
        cached = os.path.exists(os.path.join(out, "plan.pt"))
        if cached:
            log(f"cached plan {out}/plan.pt")
        else:
            tmp = f"{out}.tmp{os.getpid()}"
            build(args, engine_config, index, key, tmp)
            try:
                os.rename(tmp, out)
            except OSError:
                if not os.path.exists(os.path.join(out, "plan.pt")):
                    raise
                log(f"another build finished {out} first")
    if args.result:
        with open(args.result, "w") as f:
            json.dump({"plan": os.path.join(out, "plan.pt"), "key": key, "cached": cached}, f)


if __name__ == "__main__":
    main()
