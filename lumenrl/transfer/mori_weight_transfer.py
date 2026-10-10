"""Pull weight sync over MORI-IO: rollout ranks read a plan's regions from trainer params.

Each trainer rank registers the allocations holding its parameters (Megatron's
distributed-optimizer buffers: one per dtype, dense and expert apart) and exports
where every plan source lives in them. Each rollout rank registers one span over
its own parameters, checks every read of its share of the plan against both
sides' sizes, and turns each round into one ``IOEngine.batch_read`` across all
(source rank, registration) pairs. Nothing runs on the trainer per sync.

Verification hashes every plan read on both sides (``read_hashes``): a
position-weighted sum of the read's 16-bit words, so a shifted, swapped or stale
region changes it.
"""

from __future__ import annotations

import itertools
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from lumenrl.transfer.weight_plan import Plan

__all__ = ["MoriWeightSource", "MoriWeightReader", "SourceExport", "read_entries",
           "check_bounds", "read_hashes", "compare_hashes", "load_plan"]

_ROUND, _SRC, _DP, _DO, _SP, _SO, _RB, _NR, _DS, _SS = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
_engine_ids = itertools.count()


def _new_engine(role: str, num_streams: int):
    from mori.io import BackendType, IOEngine, IOEngineConfig, XgmiBackendConfig

    name = f"lumen-mori-{role}-{os.getpid()}-{next(_engine_ids)}"
    eng = IOEngine(name, IOEngineConfig(host="127.0.0.1", port=0))
    eng.create_backend(BackendType.XGMI, XgmiBackendConfig(num_streams=num_streams,
                                                           num_events=num_streams))
    return eng


def _register(eng, ptr: int, nbytes: int, device: int):
    from mori.io import MemoryLocationType

    return eng.register_memory(int(ptr), int(nbytes), int(device), MemoryLocationType.GPU)


def _allocation_of(ptr: int, segments: list[tuple[int, int]]) -> tuple[int, int]:
    """(base, size) of the caching-allocator segment holding ``ptr``.

    Remote registrations are exported by IPC handle, which names a whole allocation;
    MORI releases before ROCm/mori#416 drop the offset of an interior pointer, so
    sources are registered at their allocation's base.
    """
    for base, size in segments:
        if base <= ptr < base + size:
            return base, size
    raise RuntimeError(f"pointer {ptr:#x} is not in a caching-allocator segment; MORI "
                       "sources must be torch-allocated (no expandable segments)")


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.detach().view(-1).view(torch.uint8)


@dataclass
class SourceExport:
    """What a trainer rank sends to the rollout ranks once, at setup."""

    src_rank: int
    engine: bytes                                # packed EngineDesc
    memories: list[bytes]                        # packed MemoryDesc per allocation
    params: dict[int, tuple[int, int, int]]      # plan src index -> (memory, offset, nbytes)

    def to_wire(self) -> dict[str, Any]:
        """Plain types only: vLLM's RPC path to its workers is msgpack."""
        return {"src_rank": self.src_rank, "engine": bytes(self.engine),
                "memories": [bytes(m) for m in self.memories],
                "params": [[q, *v] for q, v in self.params.items()]}

    @classmethod
    def from_wire(cls, d: Any) -> "SourceExport":
        if isinstance(d, SourceExport):
            return d
        return cls(int(d["src_rank"]), bytes(d["engine"]), [bytes(m) for m in d["memories"]],
                   {int(q): (int(m), int(o), int(n)) for q, m, o, n in d["params"]})


class MoriWeightSource:
    """Trainer rank ``src_rank``: registers the allocations under the plan's sources.

    ``params`` maps the trainer's local parameter names to its tensors; the plan
    names sources by catalog name and ``meta["src_local_names"]`` maps those to
    local names (expert and pipeline-stage indices differ between the two).
    """

    def __init__(self, plan: Plan, src_rank: int, params: Mapping[str, torch.Tensor],
                 num_streams: int = 64) -> None:
        self.plan, self.src_rank = plan, int(src_rank)
        local = plan.meta.get("src_local_names") or {}
        needed = sorted(set(plan.col("src_param")[plan.col("src_rank") == self.src_rank].tolist()))
        missing = [plan.src_names[q] for q in needed
                   if local.get(plan.src_names[q], plan.src_names[q]) not in params]
        if missing:
            raise KeyError(f"trainer rank {src_rank} lacks {len(missing)} plan sources, "
                           f"e.g. {missing[:5]}")
        self.tensors = {q: params[local.get(plan.src_names[q], plan.src_names[q])]
                        for q in needed}
        self.device = next(iter(self.tensors.values())).device.index if self.tensors else 0
        self.eng = _new_engine(f"src{self.src_rank}", num_streams)
        self._ptrs = self._snapshot_ptrs()
        segments = [(s["address"], s["total_size"]) for s in torch.cuda.memory_snapshot()
                    if s.get("device", self.device) == self.device]
        bases: dict[int, int] = {}
        self.memories, self.params = [], {}
        for q, t in self.tensors.items():
            base, size = _allocation_of(t.data_ptr(), segments)
            if base not in bases:
                bases[base] = len(self.memories)
                self.memories.append(_register(self.eng, base, size, self.device))
            self.params[q] = (bases[base], t.data_ptr() - base, t.numel() * t.element_size())

    def _snapshot_ptrs(self) -> dict[int, int]:
        return {q: t.data_ptr() for q, t in self.tensors.items()}

    def export(self) -> SourceExport:
        return SourceExport(self.src_rank, self.eng.get_engine_desc().pack(),
                            [m.pack() for m in self.memories], dict(self.params))

    def prepare(self) -> None:
        """Before rollout ranks read: trainer GPU work done and sources where they were."""
        torch.cuda.synchronize(self.device)
        moved = [self.plan.src_names[q] for q, p in self._snapshot_ptrs().items()
                 if p != self._ptrs[q]]
        if moved:
            raise RuntimeError(f"{len(moved)} MORI weight sources moved since registration "
                               f"(e.g. {moved[:3]}); parameter offload is not supported")

    def hashes(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Per rollout rank: (plan row indices read from this rank, their hashes)."""
        rows = self.plan.rows
        out = {}
        flat = {q: _flat(t) for q, t in self.tensors.items()}
        for d in sorted(set(rows[:, 0].tolist())):
            idx = torch.nonzero((rows[:, 0] == d) & (rows[:, _SRC] == self.src_rank)).flatten()
            out[d] = (idx, read_hashes(rows[idx], flat, _SP, _SO, _SS))
        return out


def read_entries(rows: torch.Tensor) -> list[tuple[int, int, int, int, int, int, int]]:
    """One rollout rank's plan rows as transport reads ``(round, src rank, dst param,
    dst off, src param, src off, bytes)``; a multi-row read with a stride on either
    side becomes one read per row."""
    out = []
    for r in rows.tolist():
        rb, nr, ds, ss = r[_RB], r[_NR], r[_DS], r[_SS]
        if nr == 1 or (ds == rb and ss == rb):
            out.append((r[_ROUND], r[_SRC], r[_DP], r[_DO], r[_SP], r[_SO], rb * nr))
        else:
            out += [(r[_ROUND], r[_SRC], r[_DP], r[_DO] + i * ds, r[_SP], r[_SO] + i * ss, rb)
                    for i in range(nr)]
    return out


def check_bounds(rows: torch.Tensor, dst_nbytes: Mapping[int, int],
                 src_nbytes: Mapping[tuple[int, int], int], plan: Plan) -> None:
    """Every read lies inside its parameter on both sides; span registrations
    give no protection of their own, so this runs before the first read."""
    bad = []
    for r in rows.tolist():
        last = (r[_NR] - 1)
        d_end = r[_DO] + last * r[_DS] + r[_RB]
        s_end = r[_SO] + last * r[_SS] + r[_RB]
        dn = dst_nbytes.get(r[_DP])
        sn = src_nbytes.get((r[_SRC], r[_SP]))
        if dn is None or sn is None or r[_DO] < 0 or r[_SO] < 0 or d_end > dn or s_end > sn:
            bad.append(f"{plan.dst_names[r[_DP]]}[{r[_DO]}:{d_end}] of {dn} <- rank {r[_SRC]} "
                       f"{plan.src_names[r[_SP]]}[{r[_SO]}:{s_end}] of {sn}")
    if bad:
        raise ValueError(f"{len(bad)} plan reads out of bounds, e.g. {bad[:3]}")


_WEIGHTS: dict[torch.device, torch.Tensor] = {}


def _weights(n: int, device: torch.device) -> torch.Tensor:
    w = _WEIGHTS.get(device)
    if w is None or w.numel() < n:
        w = torch.arange(max(n, 1 << 20), device=device, dtype=torch.int64) % 65521 + 1
        _WEIGHTS[device] = w
    return w[:n]


@torch.no_grad()
def read_hashes(rows: torch.Tensor, flat: Mapping[int, torch.Tensor], col_param: int,
                col_off: int, col_stride: int) -> torch.Tensor:
    """Hash of each read's bytes on one side (the param, offset and row stride
    columns pick the side), as an int64 tensor on the side's device."""
    out = []
    for r in rows.tolist():
        f = flat[r[col_param]]
        v = f.as_strided((r[_NR], r[_RB]), (r[col_stride], 1), f.storage_offset() + r[col_off])
        v = v.contiguous().view(-1)
        if v.numel() % 2 == 0:
            v = v.view(torch.int16)
        out.append((v.to(torch.int64) * _weights(v.numel(), v.device)).sum())
    if not out:
        return torch.zeros(0, dtype=torch.int64)
    return torch.stack(out).cpu()


class MoriWeightReader:
    """Rollout rank ``dst_rank``: reads its share of ``plan`` into ``params``.

    ``params`` maps the rollout engine's parameter names to its tensors (vLLM's
    ``named_parameters()``); ``sources`` holds every trainer rank's export.
    """

    def __init__(self, plan: Plan, dst_rank: int, params: Mapping[str, torch.Tensor],
                 sources: list, window: int = 2, num_streams: int = 64) -> None:
        from mori.io import EngineDesc, MemoryDesc

        sources = [SourceExport.from_wire(s) for s in sources]
        self.plan, self.dst_rank, self.window = plan, int(dst_rank), max(1, int(window))
        self.rows = plan.rows[plan.col("dst_rank") == self.dst_rank]
        if self.rows.numel() == 0:
            raise ValueError(f"plan has no reads for rollout rank {dst_rank}")
        used = sorted(set(self.rows[:, _DP].tolist()))
        missing = [plan.dst_names[q] for q in used if plan.dst_names[q] not in params]
        if missing:
            raise KeyError(f"rollout engine lacks {len(missing)} plan params, e.g. {missing[:5]}")
        self.tensors = {q: params[plan.dst_names[q]] for q in used}
        by_rank = {s.src_rank: s for s in sources}
        check_bounds(self.rows, {q: t.numel() * t.element_size() for q, t in self.tensors.items()},
                     {(s.src_rank, q): v[2] for s in sources for q, v in s.params.items()}, plan)

        t0 = time.perf_counter()
        self.device = next(iter(self.tensors.values())).device.index
        self.eng = _new_engine(f"dst{self.dst_rank}", num_streams)
        for s in sources:
            self.eng.register_remote_engine(EngineDesc.unpack(s.engine))
        remote = {s.src_rank: [MemoryDesc.unpack(m) for m in s.memories] for s in sources}
        lo = min(t.data_ptr() for t in self.tensors.values())
        hi = max(t.data_ptr() + t.numel() * t.element_size() for t in self.tensors.values())
        self.local = _register(self.eng, lo, hi - lo, self.device)
        at = {q: t.data_ptr() - lo for q, t in self.tensors.items()}

        n_rounds = int(self.rows[:, _ROUND].max()) + 1
        groups: list[dict[tuple[int, int], tuple[list, list, list]]] = [{} for _ in range(n_rounds)]
        self.entries = read_entries(self.rows)
        for rnd, s, dp, do, sp, so, n in self.entries:
            mem, base, _ = by_rank[s].params[sp]
            g = groups[rnd].setdefault((s, mem), ([], [], []))
            g[0].append(at[dp] + do)
            g[1].append(base + so)
            g[2].append(n)
        self.calls = []
        for g in groups:
            keys = list(g)
            if keys:
                self.calls.append(([self.local] * len(keys), [g[k][0] for k in keys],
                                   [remote[k[0]][k[1]] for k in keys], [g[k][1] for k in keys],
                                   [g[k][2] for k in keys]))
        self.bytes = int(sum(e[6] for e in self.entries))
        self.setup_s = time.perf_counter() - t0

    def sync(self) -> dict[str, Any]:
        """Read every round, ``window`` rounds in flight; raises on any failed read."""
        t0 = time.perf_counter()
        pending: list = []
        failures: list[str] = []

        def drain(statuses) -> None:
            for st in statuses:
                st.Wait()
                if not st.Succeeded() and len(failures) < 4:
                    failures.append(f"code={st.Code()} {st.Message()}")
                elif not st.Succeeded():
                    failures.append("")

        for ld, lo, rd, ro, sz in self.calls:
            while len(pending) >= self.window:
                drain(pending.pop(0))
            pending.append(self.eng.batch_read(ld, lo, rd, ro, sz,
                                               [self.eng.allocate_transfer_uid() for _ in sz]))
        for statuses in pending:
            drain(statuses)
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - t0
        if failures:
            raise RuntimeError(f"MORI weight sync: {len(failures)} batch reads failed on rollout "
                               f"rank {self.dst_rank}: {[f for f in failures if f][:4]}")
        return {"seconds": seconds, "bytes": float(self.bytes), "rounds": float(len(self.calls)),
                "reads": float(len(self.entries)),
                "gbps": self.bytes / seconds / 1e9 if seconds > 0 else 0.0}

    def hashes(self) -> torch.Tensor:
        """Hash of every plan read for this rank, in plan row order."""
        flat = {q: _flat(t) for q, t in self.tensors.items()}
        return read_hashes(self.rows, flat, _DP, _DO, _DS)


def compare_hashes(plan: Plan, dst_rank: int, dst: torch.Tensor,
                   src: list[dict[int, tuple[torch.Tensor, torch.Tensor]]],
                   limit: int = 5) -> list[str]:
    """Plan reads whose hash differs between the rollout rank and its trainer source."""
    idx = torch.nonzero(plan.col("dst_rank") == dst_rank).flatten()
    want = torch.full((plan.rows.shape[0],), -1, dtype=torch.int64)
    seen = torch.zeros(plan.rows.shape[0], dtype=torch.bool)
    for per_rank in src:
        if dst_rank in per_rank:
            i, h = per_rank[dst_rank]
            want[i] = h
            seen[i] = True
    want, seen = want[idx], seen[idx]
    bad = torch.nonzero(~seen | (want != dst)).flatten().tolist()
    out = []
    for k in bad[:limit]:
        r = plan.rows[idx[k]].tolist()
        out.append(f"{plan.dst_names[r[_DP]]}@{r[_DO]} <- rank {r[_SRC]} "
                   f"{plan.src_names[r[_SP]]}@{r[_SO]} ({r[_RB] * r[_NR]} B)"
                   + ("" if seen[k] else " no source hash"))
    if len(bad) > limit:
        out.append(f"... {len(bad) - limit} more")
    return out


def load_plan(path: str, n_src: int, n_dst: int) -> Plan:
    """The cached plan, checked against the live layout."""
    plan = Plan.load(path)
    meta = plan.meta
    if int(meta.get("n_src", -1)) != n_src or int(meta.get("n_dst", -1)) != n_dst:
        raise ValueError(f"MORI weight plan {path} is for {meta.get('n_src')} trainer ranks and "
                         f"{meta.get('n_dst')} rollout ranks; this job has {n_src} and {n_dst}")
    return plan
