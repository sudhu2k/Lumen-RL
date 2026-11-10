"""Pull weight sync over MORI-IO: rollout ranks read a plan's regions from trainer params.

Each trainer rank registers the allocations holding its parameters (Megatron's
distributed-optimizer buffers: one per dtype, dense and expert apart) and exports
where every plan source lives in them. Each rollout rank registers one span over
its own parameters, checks every read of its share of the plan against both
sides' sizes, and turns each round into one ``IOEngine.batch_read`` across all
(source rank, registration) pairs. Nothing runs on the trainer per sync.
Parameters the engine permutes after loading (AITER's MoE shuffle) are read in
their loader-stage layout into a staging slot, one tensor at a time, and moved
into place by the plan's strided view. Engines that rebuild their parameters from
the checkpoint each sync (vLLM online FP8) get a plan into checkpoint tensors
instead: ``MoriCheckpointReader`` stages them and hands them to the engine's loader.

Verification hashes every plan read on both sides (``read_hashes``): a
position-weighted sum of the read's 16-bit words, so a shifted, swapped or stale
region changes it.
"""

from __future__ import annotations

import itertools
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from lumenrl.transfer.rail_topology import gpu_rail
from lumenrl.transfer.weight_plan import Plan
from lumenrl.transfer.weight_source_map import apply_view, unapply_view

__all__ = ["MoriWeightSource", "MoriWeightReader", "MoriCheckpointReader", "SourceExport",
           "read_entries", "check_bounds", "read_hashes", "tensor_digest", "compare_hashes",
           "load_plan"]

_ROUND, _SRC, _DP, _DO, _SP, _SO, _RB, _NR, _DS, _SS = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
_engine_ids = itertools.count()


def _new_engine(role: str, num_streams: int, rdma_host: str | None = None):
    """An XGMI engine, or with ``rdma_host`` (this node's address on the network the
    NICs serve) an RDMA one. Below 4 QPs per transfer one Pollara NIC reaches about
    half its rate."""
    from mori.io import BackendType, IOEngine, IOEngineConfig

    name = f"lumen-mori-{role}-{os.getpid()}-{next(_engine_ids)}"
    if rdma_host:
        from mori.io import RdmaBackendConfig

        eng = IOEngine(name, IOEngineConfig(host=str(rdma_host), port=0))
        cfg = RdmaBackendConfig(qp_per_transfer=int(os.environ.get("LUMEN_MORI_RDMA_QPS", 4)),
                                enable_notification=False)
        # Unchunked multi-hundred-MB READs on ionic can complete before their data lands.
        chunk_mb = int(os.environ.get("LUMEN_MORI_RDMA_CHUNK_MB", 16))
        if chunk_mb > 0:
            cfg.enable_transfer_chunking = True
            cfg.chunk_bytes = chunk_mb << 20
        eng.create_backend(BackendType.RDMA, cfg)
        return eng
    from mori.io import XgmiBackendConfig

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
    # The same allocations behind an RDMA engine, for readers on other nodes.
    rdma_engine: bytes | None = None
    rdma_memories: list[bytes] | None = None
    host: str = ""                               # its node's address
    rail: int = -1                               # its GPU's own NIC's rail (rail_topology)

    def to_wire(self) -> dict[str, Any]:
        """Plain types only: vLLM's RPC path to its workers is msgpack."""
        out = {"src_rank": self.src_rank, "engine": bytes(self.engine),
               "memories": [bytes(m) for m in self.memories],
               "params": [[q, *v] for q, v in self.params.items()],
               "host": self.host, "rail": self.rail}
        if self.rdma_engine is not None:
            out["rdma_engine"] = bytes(self.rdma_engine)
            out["rdma_memories"] = [bytes(m) for m in self.rdma_memories or ()]
        return out

    @classmethod
    def from_wire(cls, d: Any) -> "SourceExport":
        if isinstance(d, SourceExport):
            return d
        rdma = d.get("rdma_engine")
        return cls(int(d["src_rank"]), bytes(d["engine"]), [bytes(m) for m in d["memories"]],
                   {int(q): (int(m), int(o), int(n)) for q, m, o, n in d["params"]},
                   bytes(rdma) if rdma is not None else None,
                   [bytes(m) for m in d["rdma_memories"]] if rdma is not None else None,
                   str(d.get("host", "")), int(d.get("rail", -1)))

    def over_rdma(self) -> "SourceExport":
        """This export with its RDMA engine and memories in the XGMI ones' place."""
        if self.rdma_engine is None:
            raise ValueError(f"trainer rank {self.src_rank} has no RDMA engine; cross-node "
                             "MORI weight sync needs the sources registered with rdma_host")
        return SourceExport(self.src_rank, self.rdma_engine, list(self.rdma_memories or ()),
                            dict(self.params), host=self.host, rail=self.rail)


class MoriWeightSource:
    """Trainer rank ``src_rank``: registers the allocations under the plan's sources.

    ``params`` maps the trainer's local parameter names to its tensors; the plan
    names sources by catalog name and ``meta["src_local_names"]`` maps those to
    local names (expert and pipeline-stage indices differ between the two).
    """

    def __init__(self, plan: Plan, src_rank: int, params: Mapping[str, torch.Tensor],
                 num_streams: int = 64, rdma_host: str | None = None) -> None:
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
        self.host = str(rdma_host or "")
        self.rdma_eng, self.rdma_memories = None, []
        if rdma_host:
            self.rdma_eng = _new_engine(f"src{self.src_rank}-rdma", num_streams, rdma_host)
            sizes = {b: s for b, s in segments}
            self.rdma_memories = [_register(self.rdma_eng, b, sizes[b], self.device)
                                  for b in sorted(bases, key=bases.__getitem__)]

    def _snapshot_ptrs(self) -> dict[int, int]:
        return {q: t.data_ptr() for q, t in self.tensors.items()}

    def export(self) -> SourceExport:
        rdma = self.rdma_eng
        topo = gpu_rail(self.device) if rdma is not None else None
        return SourceExport(self.src_rank, self.eng.get_engine_desc().pack(),
                            [m.pack() for m in self.memories], dict(self.params),
                            rdma.get_engine_desc().pack() if rdma is not None else None,
                            [m.pack() for m in self.rdma_memories] if rdma is not None else None,
                            self.host, rail=topo.own_rail if topo is not None else -1)

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


@torch.no_grad()
def tensor_digest(t: torch.Tensor, chunk: int = 65521 << 10) -> int:
    """``read_hashes``' position-weighted word sum over a whole tensor, in chunks
    (a multiple of the weights' period, so chunking doesn't change it)."""
    f = _flat(t.contiguous())
    v = f.view(torch.int16) if f.numel() % 2 == 0 else f
    total = torch.zeros((), dtype=torch.int64, device=v.device)
    for i in range(0, v.numel(), chunk):
        x = v[i:i + chunk]
        total += (x.to(torch.int64) * _weights(x.numel(), x.device)).sum()
    return int(total)


def _entries(rows: torch.Tensor, strided: bool) -> list[tuple]:
    """Plan rows as reads ``(round, src rank, dst param, dst off, src param, src off,
    row bytes, rows, dst pitch, src pitch)``: one per row with ``strided`` (MORI's
    ``batch_read_strided``), else one contiguous read per row of each."""
    if strided:
        return [(r[_ROUND], r[_SRC], r[_DP], r[_DO], r[_SP], r[_SO], r[_RB], r[_NR],
                 r[_DS], r[_SS]) for r in rows.tolist()]
    return [(*e, 1, e[6], e[6]) for e in read_entries(rows)]


def _read_calls(entries, by_rank, remote, local, offset, local_of=None) -> list[tuple]:
    """``entries`` as one batch per round, one read list per (source, memory, local
    memory); ``offset(dst param)`` is where the param starts in ``local``, or in
    ``local_of(dst param)`` when given."""
    by_round: dict[int, dict[tuple[int, int, int], tuple]] = {}
    for rnd, s, dp, do, sp, so, n, nr, ds, ss in entries:
        mem, base, _ = by_rank[s].params[sp]
        lm = local if local_of is None else local_of(dp)
        g = by_round.setdefault(rnd, {}).setdefault((s, mem, id(lm)),
                                                    (lm, [], [], [], [], [], []))
        for col, v in zip(g[1:], (offset(dp) + do, base + so, n, nr, ds, ss)):
            col.append(v)
    out = []
    for rnd in sorted(by_round):
        g = by_round[rnd]
        keys = list(g)
        out.append(([g[k][0] for k in keys], [g[k][1] for k in keys],
                    [remote[k[0]][k[1]] for k in keys],
                    *([g[k][c] for k in keys] for c in range(2, 7))))
    return out


def _row_views(r: list, dst: torch.Tensor, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Plan row ``r``'s (destination, source) as strided byte views of two flat tensors."""
    shape = (r[_NR], r[_RB])
    return (dst.as_strided(shape, (r[_DS], 1), dst.storage_offset() + r[_DO]),
            src.as_strided(shape, (r[_SS], 1), src.storage_offset() + r[_SO]))


def _issue(eng, strided: bool, call):
    if call is None:
        return []
    ld, lo, rd, ro, sz, nr, lp, rp = call
    uids = [eng.allocate_transfer_uid() for _ in sz]
    if strided:
        return eng.batch_read_strided(ld, lo, rd, ro, sz, nr, lp, rp, uids)
    return eng.batch_read(ld, lo, rd, ro, sz, uids)


def _drain(statuses, failures: list[str]) -> None:
    for st in statuses:
        st.Wait()
        if not st.Succeeded() and len(failures) < 4:
            failures.append(f"code={st.Code()} {st.Message()}")
        elif not st.Succeeded():
            failures.append("")


def _connect(sources: list, role: str, num_streams: int, rdma_host: str | None = None):
    """A local engine that knows every trainer rank's engine and memories."""
    from mori.io import EngineDesc, MemoryDesc

    eng = _new_engine(role, num_streams, rdma_host)
    for s in sources:
        eng.register_remote_engine(EngineDesc.unpack(s.engine))
    remote = {s.src_rank: [MemoryDesc.unpack(m) for m in s.memories] for s in sources}
    return eng, remote


class MoriWeightReader:
    """Rollout rank ``dst_rank``: reads its share of ``plan`` into ``params``.

    ``params`` maps the rollout engine's parameter names to its tensors (vLLM's
    ``named_parameters()``); ``sources`` holds every trainer rank's export.
    ``local_sources`` (source rank -> plan src index -> bytes) are sources on this
    GPU, in this process: their rows are local strided copies, since a process
    cannot open its own IPC handles. ``in_place`` (one flag per rank row, in plan
    order) marks rows the caller writes into ``params`` itself: hashed, never read.
    ``fetched`` marks rows of permuted parameters the caller reads into a staging
    slot: ``fetch(param, slot)`` issues every read of that param into ``slots[slot]``
    (at its loader-layout offsets) and returns their statuses; these params go
    first, before any round. ``num_slots``: staging slots, so up to that many
    permuted params are read at once.
    """

    def __init__(self, plan: Plan, dst_rank: int, params: Mapping[str, torch.Tensor],
                 sources: list, window: int = 2, num_streams: int = 64,
                 strided: bool = True,
                 local_sources: Mapping[int, Mapping[int, torch.Tensor]] | None = None,
                 in_place: torch.Tensor | None = None, fetched: torch.Tensor | None = None,
                 fetch: Callable[[int, int], list] | None = None, num_slots: int = 2) -> None:
        sources = [SourceExport.from_wire(s) for s in sources]
        local_sources = {int(s): dict(m) for s, m in (local_sources or {}).items()}
        self.plan, self.dst_rank, self.window = plan, int(dst_rank), max(1, int(window))
        self.rows = plan.rows[plan.col("dst_rank") == self.dst_rank]
        if self.rows.numel() == 0:
            raise ValueError(f"plan has no reads for rollout rank {dst_rank}")
        used = sorted(set(self.rows[:, _DP].tolist()))
        missing = [plan.dst_names[q] for q in used if plan.dst_names[q] not in params]
        if missing:
            raise KeyError(f"rollout engine lacks {len(missing)} plan params, e.g. {missing[:5]}")
        names = {plan.dst_names[q] for q in used}
        stale = [n for n in plan.meta.get("not_synced") or () if n in params and n not in names]
        if stale:
            raise ValueError(f"MORI weight plan cannot sync {len(stale)} rollout parameters "
                             f"(neither copies nor permutations of trainer weights), e.g. "
                             f"{stale[:5]}")
        self.tensors = {q: params[plan.dst_names[q]] for q in used}
        views = plan.meta.get("views") or {}
        # Permuted parameters: read in their loader-stage layout into a staging slot,
        # then moved into place by one strided copy (``apply_view``).
        self.views = {q: views[plan.dst_names[q]] for q in used if plan.dst_names[q] in views}
        none = torch.zeros(self.rows.shape[0], dtype=torch.bool)
        mask = none if in_place is None else in_place.to(torch.bool)
        ext = none if fetched is None else fetched.to(torch.bool)
        read = self.rows[~(mask | ext)]
        if any(q in self.views for q in self.rows[mask, _DP].tolist()):
            raise ValueError("rows of permuted parameters cannot be written in place")
        self.fetched = sorted(set(self.rows[ext, _DP].tolist()))
        if any(q not in self.views for q in self.fetched) or (self.fetched and fetch is None):
            raise ValueError("fetched rows must be of permuted parameters, with a fetch")
        if set(read[:, _DP].tolist()) & set(self.fetched):
            raise ValueError("a permuted parameter's rows are all fetched or all read")
        self.fetch = fetch
        by_rank = {s.src_rank: s for s in sources}
        src_nbytes = {(s.src_rank, q): v[2] for s in sources for q, v in s.params.items()}
        src_nbytes.update({(s, q): t.numel() for s, m in local_sources.items()
                           for q, t in m.items()})
        check_bounds(read, {q: t.numel() * t.element_size() for q, t in self.tensors.items()},
                     src_nbytes, plan)

        t0 = time.perf_counter()
        self.device = next(iter(self.tensors.values())).device.index
        self.eng, remote = _connect(sources, f"dst{self.dst_rank}", num_streams)
        is_local = torch.isin(read[:, _SRC],
                              torch.tensor(sorted(local_sources), dtype=torch.int64))
        self.local_rows = [(r, local_sources[r[_SRC]][r[_SP]])
                           for r in read[is_local].tolist()]
        direct = [t for q, t in self.tensors.items() if q not in self.views]
        lo = min((t.data_ptr() for t in direct), default=0)
        hi = max((t.data_ptr() + t.numel() * t.element_size() for t in direct), default=0)
        self.local = _register(self.eng, lo, hi - lo, self.device) if direct else None
        at = {q: t.data_ptr() - lo for q, t in self.tensors.items()}
        # Slots: one tensor's reads land in one while another is moved into place.
        slot_bytes = max((self.tensors[q].numel() * self.tensors[q].element_size()
                          for q in self.views), default=0)
        self.slots = [torch.empty(slot_bytes, dtype=torch.uint8, device=self.device)
                      for _ in range(min(max(1, int(num_slots)), len(self.views)))]
        self.slot_mems = [_register(self.eng, s.data_ptr(), slot_bytes, self.device)
                          for s in self.slots]
        self.staging_bytes = slot_bytes * len(self.slots)

        self.strided = bool(strided) and hasattr(self.eng, "batch_read_strided")
        self.entries = _entries(read[~is_local], self.strided)

        direct_entries = [e for e in self.entries if e[2] not in self.views]
        self.calls = _read_calls(direct_entries, by_rank, remote, self.local, at.__getitem__)
        self.call_rounds = sorted({e[0] for e in direct_entries})
        # Per permuted tensor and slot: all its reads as one batch (rounds merged).
        self.staged = [(q, [(_read_calls([(0, *e[1:]) for e in self.entries if e[2] == q],
                                         by_rank, remote, mem, lambda _: 0) or [None])[0]
                            for mem in self.slot_mems])
                       for q in sorted(self.views) if q not in self.fetched]
        self.staged_rounds: dict[int, int] = {}
        for rnd, q in [(e[0], e[2]) for e in self.entries] + [(r[_ROUND], r[_DP])
                                                              for r, _ in self.local_rows]:
            if q in self.views:
                self.staged_rounds[q] = max(self.staged_rounds.get(q, 0), rnd)
        self.direct_copies: dict[int, list] = defaultdict(list)
        for r, src in self.local_rows:
            if r[_DP] not in self.views:
                self.direct_copies[r[_ROUND]].append(_row_views(r, _flat(self.tensors[r[_DP]]),
                                                                src))
        self.bytes = int(self.plan.nbytes()[plan.col("dst_rank") == self.dst_rank].sum())
        self.setup_s = time.perf_counter() - t0

    def _local_copies(self, q: int, slot: torch.Tensor) -> None:
        """Permuted tensor ``q``'s rows from local sources into its staging ``slot``."""
        for r, src in self.local_rows:
            if r[_DP] == q:
                d, s = _row_views(r, slot, src)
                d.copy_(s)

    def sync(self, gate: Callable[[int], None] | None = None) -> dict[str, Any]:
        """Read every round, ``window`` rounds in flight; raises on any failed read.
        ``gate(round)``, when given, returns once the sources' bytes for that round
        are in place; it runs before the round's reads and local copies are issued."""
        t0 = time.perf_counter()
        pending: list = []
        failures: list[str] = []
        calls = dict(zip(self.call_rounds, self.calls))

        def drain(statuses) -> None:
            _drain(statuses, failures)

        def issue(call):
            return _issue(self.eng, self.strided, call)

        # Permuted tensors: tensor i reads into slot i % slots once the copy out of that
        # slot is done; the next tensors' reads overlap tensor i's copy.
        done: list = [None] * len(self.slots)
        turn = [0]

        def place(q, slot, statuses) -> None:
            drain(statuses)
            t = self.tensors[q]
            self._local_copies(q, self.slots[slot])
            src = self.slots[slot][:t.numel() * t.element_size()].view(t.dtype)
            apply_view(self.views[q], src, t)
            done[slot] = torch.cuda.Event()
            done[slot].record()

        def stage(items) -> None:
            staged: list = []
            for q, start, rnd in items:
                slot = turn[0] % len(self.slots)
                turn[0] += 1
                if done[slot] is not None:
                    done[slot].synchronize()
                if gate is not None and rnd is not None:
                    gate(rnd)
                staged.append((q, slot, start(slot)))
                if len(staged) == len(self.slots):
                    place(*staged.pop(0))
            for args in staged:
                place(*args)

        stage([(q, lambda slot, q=q: self.fetch(q, slot), None) for q in self.fetched])
        for rnd in sorted(set(calls) | set(self.direct_copies)):
            if gate is not None:
                gate(rnd)
            for d, s in self.direct_copies.get(rnd, ()):
                d.copy_(s)
            if rnd in calls:
                while len(pending) >= self.window:
                    drain(pending.pop(0))
                pending.append(issue(calls[rnd]))
        for statuses in pending:
            drain(statuses)
        t_direct = time.perf_counter() - t0
        stage([(q, lambda slot, c=c: issue(c[slot]), self.staged_rounds[q])
               for q, c in self.staged])
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - t0
        if failures:
            raise RuntimeError(f"MORI weight sync: {len(failures)} batch reads failed on rollout "
                               f"rank {self.dst_rank}: {[f for f in failures if f][:4]}")
        return {"seconds": seconds, "bytes": float(self.bytes), "rounds": float(len(self.calls)),
                "reads": float(len(self.entries)), "direct_s": t_direct,
                "permuted": float(len(self.staged) + len(self.fetched)),
                "gbps": self.bytes / seconds / 1e9 if seconds > 0 else 0.0}

    def hashes(self) -> torch.Tensor:
        """Hash of every plan read for this rank, in plan row order. A permuted tensor's
        reads are in its loader-stage layout, rebuilt from the tensor into a slot."""
        flat = {q: _flat(t) for q, t in self.tensors.items() if q not in self.views}
        dp = self.rows[:, _DP]
        out = torch.zeros(self.rows.shape[0], dtype=torch.int64)
        direct = torch.ones_like(dp, dtype=torch.bool)
        for q, view in self.views.items():
            mine = dp == q
            direct &= ~mine
            t = self.tensors[q]
            nbytes = t.numel() * t.element_size()
            loader = self.slots[0][:nbytes].view(t.dtype)
            unapply_view(view, t, loader)
            out[mine] = read_hashes(self.rows[mine], {q: loader.view(torch.uint8)}, _DP, _DO, _DS)
        out[direct] = read_hashes(self.rows[direct], flat, _DP, _DO, _DS)
        return out


@dataclass
class _Batch:
    tensors: list[tuple[int, int, int]]          # (plan dst param, slot offset, bytes)
    rows: torch.Tensor                           # this rank's plan rows it reads
    calls: list                                  # its reads, one batch per slot
    copies: list = field(default_factory=list)   # (row, local source, slot offset)
    round: int | None = None                     # gate before its reads and copies


class MoriCheckpointReader:
    """Rollout rank ``dst_rank`` of a plan whose destinations are checkpoint tensors
    (``meta["staged_checkpoint"]``): an engine that rebuilds its parameters from the
    checkpoint each sync, such as vLLM's online FP8 (design 2.4).

    Whole checkpoint tensors, in checkpoint order, are packed into batches of one
    staging slot each; only the elements the rank's loader reads are transferred.
    ``sync(load)`` reads batch ``i + 1`` into one slot while batch ``i`` goes to
    ``load``. Each tensor is handed over as a clone: vLLM's layerwise reload keeps
    the tensors it is given until their layer is complete, after the slot is reused.
    ``local_sources`` are as for ``MoriWeightReader``. ``fetched`` marks rows the
    caller reads into the slots: ``fetch(batch, slot)`` issues batch ``batch``'s
    fetched rows into ``slots[slot]`` (at its tensors' offsets, ``batches[batch]``)
    and returns their statuses; it runs before the batch's own reads.
    """

    def __init__(self, plan: Plan, dst_rank: int, sources: list, device: torch.device | int,
                 num_streams: int = 64, strided: bool = True,
                 slot_bytes: int | None = None,
                 local_sources: Mapping[int, Mapping[int, torch.Tensor]] | None = None,
                 fetched: torch.Tensor | None = None,
                 fetch: Callable[[int, int], list] | None = None) -> None:
        sources = [SourceExport.from_wire(s) for s in sources]
        local_sources = {int(s): dict(m) for s, m in (local_sources or {}).items()}
        self.plan, self.dst_rank, self.device = plan, int(dst_rank), torch.device(device)
        staged = plan.meta["staged_checkpoint"]
        self.rows = plan.rows[plan.col("dst_rank") == self.dst_rank]
        if self.rows.numel() == 0:
            raise ValueError(f"plan has no reads for rollout rank {dst_rank}")
        ext = (torch.zeros(self.rows.shape[0], dtype=torch.bool) if fetched is None
               else fetched.to(torch.bool))
        if bool(ext.any()) and fetch is None:
            raise ValueError("fetched rows need a fetch")
        self.fetch = fetch
        read = self.rows[~ext]
        at = {n: q for q, n in enumerate(plan.dst_names)}
        used = set(self.rows[:, _DP].tolist())
        self.specs = {at[n]: (n, getattr(torch, s["dtype"]), tuple(s["shape"]))
                      for n, s in staged.items() if at.get(n) in used}
        if len(self.specs) != len(used):
            raise ValueError("plan reads into tensors meta['staged_checkpoint'] lacks")
        nbytes = {q: torch.empty((), dtype=d).element_size() * math.prod(shape)
                  for q, (_, d, shape) in self.specs.items()}
        by_rank = {s.src_rank: s for s in sources}
        src_nbytes = {(s.src_rank, q): v[2] for s in sources for q, v in s.params.items()}
        src_nbytes.update({(s, q): t.numel() for s, m in local_sources.items()
                           for q, t in m.items()})
        check_bounds(read, nbytes, src_nbytes, plan)

        t0 = time.perf_counter()
        self.eng, remote = _connect(sources, f"dst{self.dst_rank}", num_streams)
        self.strided = bool(strided) and hasattr(self.eng, "batch_read_strided")
        is_local = torch.isin(read[:, _SRC],
                              torch.tensor(sorted(local_sources), dtype=torch.int64))
        local_rows = [(r, local_sources[r[_SRC]][r[_SP]]) for r in read[is_local].tolist()]
        self.entries = _entries(read[~is_local], self.strided)
        cap = max(max(nbytes.values()), int(slot_bytes or plan.meta.get("round_bytes", 512 << 20)))
        groups: list[list[tuple[int, int, int]]] = [[]]
        fill = 0
        for q in self.specs:
            if fill + nbytes[q] > cap and groups[-1]:
                groups.append([])
                fill = 0
            groups[-1].append((q, fill, nbytes[q]))
            fill += -(-nbytes[q] // 256) * 256
        n_slots = min(2, len(groups))
        self.slots = [torch.empty(cap, dtype=torch.uint8, device=self.device)
                      for _ in range(n_slots)]
        self.slot_mems = [_register(self.eng, s.data_ptr(), cap, self.device.index)
                          for s in self.slots]
        self.staging_bytes = cap * n_slots
        dp = self.rows[:, _DP]
        self.batches = []
        for g in groups:
            off = {q: o for q, o, _ in g}
            mine = [e for e in self.entries if e[2] in off]
            copies = [r for r, _ in local_rows if r[_DP] in off]
            rounds = [e[0] for e in mine] + [r[_ROUND] for r in copies]
            self.batches.append(_Batch(
                g, torch.nonzero(torch.isin(dp, torch.tensor(list(off)))).flatten(),
                [(_read_calls([(0, *e[1:]) for e in mine], by_rank, remote, mem,
                              off.__getitem__) or [None])[0] for mem in self.slot_mems],
                [(r, s, off[r[_DP]]) for r, s in local_rows if r[_DP] in off],
                max(rounds) if rounds else None))
        self.bytes = int(self.plan.nbytes()[plan.col("dst_rank") == self.dst_rank].sum())
        self._hashes: torch.Tensor | None = None
        self.setup_s = time.perf_counter() - t0

    def sync(self, load: Callable[[list[tuple[str, torch.Tensor]]], Any],
             verify: bool = False,
             gate: Callable[[int], None] | None = None) -> dict[str, Any]:
        """Read every batch and pass its tensors to ``load``; with ``verify`` also hash
        every read as it lands (``hashes``). Raises on any failed read. ``gate(round)``
        returns once the sources' bytes up to ``round`` are in place; it runs before a
        batch's own reads and local copies, after its fetch."""
        t0 = time.perf_counter()
        failures: list[str] = []
        done: list = [None] * len(self.slots)
        pending: list = []
        hashes = torch.zeros(self.rows.shape[0], dtype=torch.int64) if verify else None
        wait_s = load_s = 0.0

        def land(b: _Batch, slot: int, statuses) -> None:
            nonlocal wait_s, load_s
            t = time.perf_counter()
            _drain(statuses, failures)
            wait_s += time.perf_counter() - t
            if failures:
                raise RuntimeError(f"MORI weight sync: {len(failures)} batch reads failed on "
                                   f"rollout rank {self.dst_rank}: {[f for f in failures if f][:4]}")
            buf = self.slots[slot]
            if hashes is not None:
                hashes[b.rows] = read_hashes(self.rows[b.rows],
                                             {q: buf[o:o + n] for q, o, n in b.tensors},
                                             _DP, _DO, _DS)
            weights = []
            for q, o, n in b.tensors:
                name, dtype, shape = self.specs[q]
                weights.append((name, buf[o:o + n].view(dtype).view(shape).clone()))
            done[slot] = torch.cuda.Event()
            done[slot].record()
            t = time.perf_counter()
            load(weights)
            load_s += time.perf_counter() - t

        for i, b in enumerate(self.batches):
            slot = i % len(self.slots)
            if done[slot] is not None:
                done[slot].synchronize()
            statuses = list(self.fetch(i, slot)) if self.fetch is not None else []
            if gate is not None and b.round is not None:
                gate(b.round)
            for r, src, o in b.copies:
                d, s = _row_views(r, self.slots[slot][o:], src)
                d.copy_(s)
            statuses += list(_issue(self.eng, self.strided, b.calls[slot]))
            pending.append((b, slot, statuses))
            if len(pending) == len(self.slots):
                land(*pending.pop(0))
        for args in pending:
            land(*args)
        torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - t0
        if hashes is not None:
            self._hashes = hashes
        return {"seconds": seconds, "bytes": float(self.bytes), "rounds": float(len(self.batches)),
                "reads": float(len(self.entries)), "read_wait_s": wait_s, "load_s": load_s,
                "gbps": self.bytes / seconds / 1e9 if seconds > 0 else 0.0}

    def hashes(self) -> torch.Tensor:
        """Every plan read's hash for this rank, in plan row order, as staged by the last
        ``sync(verify=True)``: the staging is gone afterwards."""
        if self._hashes is None:
            raise RuntimeError("no verified sync yet: call sync(load, verify=True)")
        return self._hashes


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
