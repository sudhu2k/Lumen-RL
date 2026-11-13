"""Cross-node MORI weight sync: mirror the trainer GPUs on the rollout node over their
rails, then run the single-node reader against the mirrors (design 5.2.8).

``plan_node``: for the plan ranks on one rollout node, every plan source tensor gets
one trainer holder, on the rail of one of the node's GPUs where possible (among
several, the one whose rail and trainer GPU carry the fewest bytes so far, counting
what the trainer GPU sends to the instances planned before). The GPU on that rail
keeps a mirror of the holder: the byte ranges the node's ranks read, at the
trainer's own offsets, fetched once over the network. A holder with no node GPU on
its rail is mirrored off-rail by the consumer that lands the fewest bytes. A source
whose only reader on the node is its mirror's GPU skips the mirror and is fetched
straight into that GPU's parameters, or into the staging slot it is read through
(permuted parameters, online-FP8 checkpoint tensors) as the reader works through
them, so mirrors hold only what other GPUs read and what needs a local copy. The
plan rows are unchanged except for their source rank (the holder) and round (the
network batch that lands their bytes).

``MoriNodeReader`` runs one rank's part: it fetches the mirrors it hosts and the
parameters it lands in place in batches, flagging each landed batch in a per-node
shared-memory area, while
``MoriWeightReader`` reads the node plan from every mirror as if the trainers were
on this node, over XGMI and by local copies, each round once its batches have
landed. Strided reads, permuted parameters and per-row hashes are the single-node
reader's. A mirror is fetched again only after every consumer has released the
previous version. There is no barrier.
"""

from __future__ import annotations

import bisect
import logging
import math
import mmap
import os
import time
import weakref
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from lumenrl.transfer.mori_weight_transfer import (
    MoriCheckpointReader, MoriWeightReader, SourceExport, _allocation_of, _connect, _drain, _issue, _new_engine,
    _read_calls, _register, _DO, _DP, _DS, _NR, _RB, _ROUND, _SO, _SP, _SRC, _SS, check_bounds,
)
from lumenrl.transfer.rail_topology import gpu_rail
from lumenrl.transfer.weight_plan import Plan

logger = logging.getLogger(__name__)

__all__ = ["plan_node", "NodePlan", "MoriNodeReader", "NodeExchange"]

_ALIGN = 256
# LUMEN_MORI_NODE_DEBUG=1: mirrors and packing scratch filled with these bytes before
# every sync, mirror hashes checked per 1 MB chunk when each network batch completes
# and again LUMEN_MORI_NODE_DEBUG_DELAY_MS later, before packing.
_DEBUG_CHUNK = 1 << 20
_MIRROR_FILL, _SCRATCH_FILL = 0xA5, 0x5A


def small_row_bytes() -> int:
    return int(os.environ.get("LUMEN_MORI_SMALL_ROW_KB", 64)) << 10


def batch_bytes() -> int:
    return int(os.environ.get("LUMEN_MORI_RDMA_BATCH_MB", 256)) << 20


def in_place_enabled() -> bool:
    return os.environ.get("LUMEN_MORI_RDMA_IN_PLACE", "1") != "0"


def pack_enabled() -> bool:
    return os.environ.get("LUMEN_MORI_NODE_PACK", "1") != "0"


def _rails(sources: list, exchange: Callable[[Any], list], device: int):
    """(trainer rank -> its GPU's own rail, the rail each rank of the group reads
    through), or (None, None) when any is unknown (``rail_topology``)."""
    src_rail = {s.src_rank: s.rail for s in sources}
    topo = gpu_rail(device)
    dst_rail = [int(d) for d in exchange(topo.rail if topo is not None else -1)]
    if all(r >= 0 for r in [*src_rail.values(), *dst_rail]):
        return src_rail, dst_rail
    return None, None


class NodeExchange:
    """``exchange(obj)`` over the ranks of one rollout node, which may belong to
    several vLLM instances with no process group in common: a ``torch`` FileStore
    in ``/dev/shm`` named by ``key`` (the same on every rank of the node, unique per
    job), ``size`` ranks, this one at position ``me``."""

    def __init__(self, key: str, size: int, me: int, timeout_s: float = 300.0) -> None:
        from datetime import timedelta

        import torch.distributed as dist

        self.store = dist.FileStore(f"/dev/shm/lumen-mori-x-{key}", int(size))
        self.store.set_timeout(timedelta(seconds=float(timeout_s)))
        self.size, self.me, self.n = int(size), int(me), 0

    def __call__(self, obj: Any) -> list:
        import pickle

        k, self.n = self.n, self.n + 1
        self.store.set(f"{k}/{self.me}", pickle.dumps(obj))
        return [pickle.loads(self.store.get(f"{k}/{i}")) for i in range(self.size)]


def _aligned(n: int) -> int:
    return -(-n // _ALIGN) * _ALIGN


@dataclass
class NodePlan:
    """One rollout node's mirrors; positions index ``group``.

    ``rows``: the group's plan rows, in plan order, each reading from its source's
    chosen holder (``src_rank``) in the round of the network batch that lands its
    bytes. ``in_place``: rows whose bytes the network batches write straight into
    the rollout parameter. ``host``: trainer rank -> position keeping its mirror.
    ``offsets``: (trainer rank, src param) -> where the param's bytes
    ``[0, extent)`` sit in the host's mirror buffer, for sources not landed in
    place. ``fetch[L]``: position ``L``'s network batches, as (src rank, src param,
    src offset, bytes, dst param, dst offset), dst param -1 for the mirror; ranges
    other positions read first. ``land_bytes[L]``: the bytes it lands in place.
    ``staged``: rows the reader fetches over the network straight into its staging
    slots (permuted params, checkpoint tensors), in none of the batches;
    ``stage_bytes[L]``: their bytes. ``pack[L]``: network batch -> the runs position
    ``L`` packs in its mirror once that batch has landed (``_plan_packing``);
    ``view_extent``: the bytes exported per mirrored source, past ``extent`` where a
    run's packed rows now read from it.
    """

    group: list[int]
    rows: torch.Tensor
    host: dict[int, int]
    offsets: dict[tuple[int, int], int]
    extent: dict[tuple[int, int], int]
    mirror_bytes: list[int]
    fetch: list[list[list[tuple[int, ...]]]]
    rail_bytes: dict[int, int] = field(default_factory=dict)
    offrail_bytes: int = 0
    in_place: torch.Tensor | None = None
    land_bytes: list[int] = field(default_factory=list)
    staged: torch.Tensor | None = None
    stage_bytes: list[int] = field(default_factory=list)
    pack: list[dict[int, list["PackRun"]]] = field(default_factory=list)
    view_extent: dict[tuple[int, int], int] = field(default_factory=dict)


@dataclass
class PackRun:
    """One packing step in a mirror, in units of ``elem`` bytes: each copy (scratch
    offset, mirror offset, shape, strides) gathers into the scratch buffer, which
    then goes back to the mirror at ``lo`` (``nbytes`` in all)."""

    lo: int
    nbytes: int
    elem: int
    copies: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]]


def plan_node(plan: Plan, sources: list, group: list[int], dst_rail: list[int],
              small_row: int, groups: list[list[int]] | None = None,
              rails: Mapping[int, int] | None = None,
              batch: int | None = None, land: list[bool] | None = None,
              stage: list[bool] | None = None) -> NodePlan:
    """The node plan for the plan ranks ``group`` on one rollout node, ``dst_rail``
    the rail each reads through (module docstring). Deterministic: every rank of the
    node computes the same plan from the same inputs.

    ``groups`` lists every vLLM instance reading over RDMA, ``rails`` the rail of
    each of their ranks. Instances choose holders in that order, each seeing the
    bytes every trainer GPU already sends to the ones before it, so instances spread
    over the trainer copies (each trainer node's, when every node holds the model).

    A multi-row read with a source stride and rows of at least ``small_row`` bytes
    is fetched row by row; any other read as the whole range it spans. Network
    batches hold up to ``batch`` bytes (``LUMEN_MORI_RDMA_BATCH_MB``).

    A source read only by its mirror's position, by rows that are contiguous or at
    least ``small_row`` bytes each, skips the mirror: rows into parameters it can
    land in (``land[L]``, default all) are fetched straight into them; rows into a
    staging slot (a permuted param or, for ``meta["staged_checkpoint"]``, any
    checkpoint tensor; ``stage[L]``, default all) straight into the slot. A
    permuted param is staged so only if every row into it is.
    """
    sources = [SourceExport.from_wire(s) for s in sources]
    holders: dict[int, list[int]] = {}
    for s in sources:
        for q in s.params:
            holders.setdefault(q, []).append(s.src_rank)
    src_rail = {s.src_rank: s.rail for s in sources}
    by_rank = {s.src_rank: s for s in sources}
    sent: dict[int, int] = defaultdict(int)
    for g in groups or []:
        if list(g) == list(group):
            break
        g_rail = [int((rails or {}).get(d, dst_rail[i % len(dst_rail)])) for i, d in enumerate(g)]
        _choose(plan, holders, src_rail, list(g), g_rail, small_row, sent)
    need, spans, choice, load, offrail = _choose(plan, holders, src_rail, list(group),
                                                 [int(r) for r in dst_rail], small_row, sent)

    host = {s: at for s, at in choice.values()}
    sel = torch.isin(plan.rows[:, 0], torch.tensor(list(group), dtype=torch.int64))
    rows = plan.rows[sel].clone()
    rl = rows.tolist()
    pos = {d: i for i, d in enumerate(group)}
    lands = [True] * len(group) if land is None else [bool(x) for x in land]
    stages = [True] * len(group) if stage is None else [bool(x) for x in stage]
    views = plan.meta.get("views") or {}
    fp8 = bool(plan.meta.get("staged_checkpoint"))

    def into_slot(dp: int) -> bool:
        return fp8 or plan.dst_names[dp] in views

    fits: dict[int, bool] = {}
    feeds: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for r in rl:
        slot = into_slot(r[_DP])
        ok = ((stages if slot else lands)[pos[r[0]]]
              and (r[_NR] == 1 or r[_SS] == r[_RB] == r[_DS] or r[_RB] >= small_row))
        fits[r[_SP]] = fits.get(r[_SP], True) and ok
        if slot and not fp8:
            feeds[r[_SP]].add((r[0], r[_DP]))
    direct_q = {q for q in need if fits[q] and {e[0] for e in need[q]} == {choice[q][1]}}
    # A permuted param is staged from the network only if all its rows are.
    while True:
        mixed = {(r[0], r[_DP]) for r in rl if r[_SP] not in direct_q}
        keep = {q for q in direct_q if not feeds[q] & mixed}
        if keep == direct_q:
            break
        direct_q = keep

    extent: dict[tuple[int, int], int] = {}
    for q, rs in need.items():
        if q not in direct_q:
            extent[(choice[q][0], q)] = max(so + (nr - 1) * ss + rb
                                            for _, _, _, nr, rb, _, so, ss in rs)

    def place(k):
        mem, base, _ = by_rank[k[0]].params[k[1]]
        return (k[0], mem, base)

    offsets: dict[tuple[int, int], int] = {}
    mirror_bytes = [0] * len(group)
    for k in sorted(extent, key=place):
        at = host[k[0]]
        offsets[k] = mirror_bytes[at]
        mirror_bytes[at] += _aligned(extent[k])

    # (piece, read first, plan rows it lands); a piece is (src rank, src param, src
    # off, bytes, dst param, dst off), dst param -1 for the mirror.
    pieces: list[list[tuple[tuple[int, ...], bool, list[int]]]] = [[] for _ in group]
    for q in sorted(set(need) - direct_q):
        s, at = choice[q]
        users = [(so, so + (nr - 1) * ss + rb, p) for p, _, _, nr, rb, _, so, ss in need[q]]
        for lo, hi in spans[q]:
            foreign = any(p != at and a < hi and b > lo for a, b, p in users)
            pieces[at].append(((s, q, lo, hi - lo, -1, 0), foreign, []))
    direct: list[list[tuple[tuple[int, ...], list[int]]]] = [[] for _ in group]
    for i, r in enumerate(rl):
        q = r[_SP]
        if q in direct_q and not into_slot(r[_DP]):
            direct[choice[q][1]] += [(k, [i]) for k in row_pieces(r, choice[q][0])]
    for at, ks in enumerate(direct):
        pieces[at] += [(k, False, idx) for k, idx in merge_pieces(ks)]

    cap = int(batch or batch_bytes())
    fetch: list[list[list[tuple[int, ...]]]] = []
    batch_of: dict[tuple[int, int], int] = {}
    row_round = [0] * len(rl)
    for ks in pieces:
        ks.sort(key=lambda x: (not x[1], *place(x[0]), x[0][2]))
        out: list[list[tuple[int, ...]]] = []
        fill = cap
        for k, _, idx in ks:
            if fill + k[3] > cap:
                out.append([])
                fill = 0
            out[-1].append(k)
            fill += _aligned(k[3])
            if k[4] < 0:
                batch_of[(k[1], k[2])] = len(out) - 1
            for i in idx:
                row_round[i] = max(row_round[i], len(out) - 1)
        fetch.append(out)

    slotted = torch.tensor([into_slot(r[_DP]) for r in rl], dtype=torch.bool)
    direct_rows = torch.tensor([r[_SP] in direct_q for r in rl], dtype=torch.bool)
    in_place, staged = direct_rows & ~slotted, direct_rows & slotted
    for i, r in enumerate(rl):
        q, so = r[_SP], r[_SO]
        rnd = row_round[i]
        if q not in direct_q:
            end = so + (r[_NR] - 1) * r[_SS] + r[_RB]
            iv = spans[q]
            j = max(bisect.bisect_right([lo for lo, _ in iv], so) - 1, 0)
            while j < len(iv) and iv[j][0] < end:
                if iv[j][1] > so:
                    rnd = max(rnd, batch_of[(q, iv[j][0])])
                j += 1
        rows[i, _SRC], rows[i, _ROUND] = choice[q][0], rnd
    land_bytes = [sum(k[3] for b in f for k in b if k[4] >= 0) for f in fetch]
    stage_bytes = [0] * len(group)
    for r in rows[staged].tolist():
        stage_bytes[pos[r[0]]] += r[_RB] * r[_NR]
    pack: list[dict[int, list[PackRun]]] = [{} for _ in group]
    view_extent = dict(extent)
    if pack_enabled() and not fp8:
        last = {q: max(batch_of[(q, lo)] for lo, _ in spans[q]) for q in set(need) - direct_q}
        pack = _plan_packing(rows, ~direct_rows, choice, offsets, extent, last, pos,
                             view_extent)
    return NodePlan(list(group), rows, host, offsets, extent, mirror_bytes, fetch,
                    dict(load), offrail, in_place, land_bytes, staged, stage_bytes,
                    pack, view_extent)


def row_pieces(r: list, s: int) -> list[tuple[int, ...]]:
    """Plan row ``r`` as network reads from trainer rank ``s``, (src rank, src param,
    src off, bytes, dst param, dst off): one when contiguous, else one per row."""
    q, rb, nr, ds, ss = r[_SP], r[_RB], r[_NR], r[_DS], r[_SS]
    if nr == 1 or ss == rb == ds:
        return [(s, q, r[_SO], rb * nr, r[_DP], r[_DO])]
    return [(s, q, r[_SO] + k * ss, rb, r[_DP], r[_DO] + k * ds) for k in range(nr)]


def merge_pieces(ks: list) -> list[tuple[tuple[int, ...], list[int]]]:
    """(piece, rows) pairs with adjacent pieces merged where both sides are
    contiguous."""
    merged: list[tuple[tuple[int, ...], list[int]]] = []
    for k, idx in sorted(ks, key=lambda x: (x[0][0], x[0][1], x[0][4], x[0][2])):
        if merged:
            p, pidx = merged[-1]
            if p[:2] == k[:2] and p[4] == k[4] and p[2] + p[3] == k[2] and p[5] + p[3] == k[5]:
                merged[-1] = ((*p[:3], p[3] + k[3], p[4], p[5]), pidx + idx)
                continue
        merged.append((k, list(idx)))
    return merged


def _plan_packing(rows: torch.Tensor, mirrored: torch.Tensor, choice: Mapping[int, tuple[int, int]],
                  offsets: Mapping[tuple[int, int], int], extent: Mapping[tuple[int, int], int],
                  last: Mapping[int, int], pos: Mapping[int, int],
                  view_extent: dict[tuple[int, int], int]) -> list[dict[int, list[PackRun]]]:
    """Packing of the mirrors, so that the node plan's strided reads from them become
    contiguous ones; rewrites ``rows`` (and ``view_extent``) to read the packed bytes.

    A mirrored source is packable if every read of it either is the only one touching
    its bytes or shares them with reads of the same row count and stride whose row
    pieces are column ranges of that stride, identical or disjoint (a TP split), with
    at least one strided read whose destination rows are contiguous. Runs of
    packable sources adjacent in one mirror, from one trainer rank and landed by the
    same network batch, are packed together: each distinct read's bytes are laid
    out contiguously from the run's start, ordered by (destination param, first
    reader, destination offset), so a reader's reads of one param from the run are
    one range. Every read of a run then reads from its first source, landing with
    that batch; the host packs the run once the batch has landed."""
    import numpy as np

    rl = rows.tolist()
    flags = mirrored.tolist()
    by_q: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rl):
        if flags[i]:
            by_q[r[_SP]].append(i)
    pats: dict[int, dict[tuple[int, int, int, int], list[int]]] = {}
    for q, idx in by_q.items():
        ps: dict[tuple[int, int, int, int], list[int]] = {}
        good = True
        for i in idx:
            r = rl[i]
            nr, rb, ss = r[_NR], r[_RB], r[_SS]
            if nr == 1 or ss == rb:
                key = (r[_SO], 1, rb * nr, rb * nr)
            elif r[_DS] != rb or ss < rb:
                good = False
                break
            else:
                key = (r[_SO], nr, rb, ss)
            ps.setdefault(key, []).append(i)
        if not good or all(k[1] == 1 for k in ps):
            continue
        groups: list[list[tuple[int, int, int, int]]] = []
        end = -1
        for k in sorted(ps):
            so, nr, rb, ss = k
            hi = so + (nr - 1) * ss + rb
            if groups and so < end:
                groups[-1].append(k)
                end = max(end, hi)
            else:
                groups.append([k])
                end = hi
        for g in groups:
            if len(g) == 1:
                continue
            nr, ss = g[0][1], g[0][3]
            if any(k[1] != nr or k[3] != ss or k[1] == 1 for k in g):
                good = False
                break
            b = g[0][0]
            iv = sorted((k[0] - b, k[0] - b + k[2]) for k in g)
            if iv[-1][1] > ss or any(a[1] > c[0] for a, c in zip(iv, iv[1:])):
                good = False
                break
        if good:
            pats[q] = ps

    out: list[dict[int, list[PackRun]]] = [{} for _ in range(max(pos.values(), default=-1) + 1)]
    runs: list[list] = []
    cur: list | None = None
    for k in sorted(extent, key=lambda k: (choice[k[1]][1], offsets[k])):
        s, q = k
        if q not in pats:
            cur = None
            continue
        tag = (choice[q][1], s, last[q])
        if cur is not None and cur[0] == tag and cur[2] == offsets[k]:
            cur[1].append(q)
        else:
            cur = [tag, [q], 0]
            runs.append(cur)
        cur[2] = offsets[k] + _aligned(extent[k])

    for (at, s, rnd), qs, _ in runs:
        anchor = offsets[(s, qs[0])]
        items = []
        for q in qs:
            base = offsets[(s, q)]
            for (so, nr, rb, ss), idx in pats[q].items():
                f = min(idx, key=lambda i: (pos[rl[i][0]], rl[i][_DP], rl[i][_DO]))
                items.append(((rl[f][_DP], pos[rl[f][0]], rl[f][_DO]), base + so, nr, rb, ss, idx))
        items.sort(key=lambda x: x[0])
        packed, fill = [], 0
        for _, x, nr, rb, ss, idx in items:
            packed.append(fill)
            for i in idx:
                rows[i, _SP], rows[i, _SO], rows[i, _ROUND] = qs[0], fill, rnd
                rows[i, _SS] = rows[i, _RB]
            fill += nr * rb
        # Consecutive reads of one shape whose mirror offsets form a grid: one copy each.
        copies = []
        j = 0
        while j < len(items):
            nr, rb, ss = items[j][2:5]
            k = j
            while k < len(items) and items[k][2:5] == (nr, rb, ss):
                k += 1
            xs = np.array([it[1] for it in items[j:k]], dtype=np.int64)
            for x0, dims, n in _factor(xs):
                copies.append((packed[j], x0, tuple(d for d, _ in dims) + (nr, rb),
                               tuple(st for _, st in dims) + (ss, 1)))
                j += n
        g = math.gcd(anchor, fill)
        for so, x0, sh, st in copies:
            g = math.gcd(g, so, x0, sh[-1], *st[:-1])
        elem = next(e for e in (8, 4, 2, 1) if g % e == 0)
        copies = [(so // elem, x0 // elem, sh[:-1] + (sh[-1] // elem,),
                   tuple(v // elem for v in st[:-1]) + (1,)) for so, x0, sh, st in copies]
        out[at].setdefault(rnd, []).append(PackRun(anchor // elem, fill // elem, elem, copies))
        view_extent[(s, qs[0])] = max(view_extent[(s, qs[0])], fill)
    return out


def apply_pack(mirror: torch.Tensor, scratch: torch.Tensor, runs: list[PackRun]) -> None:
    """Pack ``runs`` in ``mirror`` (flat uint8) through ``scratch``, on the current stream."""
    for p in runs:
        dt = _ELEM_DTYPE[p.elem]
        m, sc = mirror.view(dt), scratch.view(dt)
        for so, x0, shape, strides in p.copies:
            sc[so:so + math.prod(shape)].view(shape).copy_(m.as_strided(shape, strides, x0))
        m[p.lo:p.lo + p.nbytes].copy_(sc[:p.nbytes])


def _factor(xs) -> list[tuple[int, list[tuple[int, int]], int]]:
    """``xs`` (offsets) in order as blocks (first offset, dims, count): each block's
    offsets are ``first + sum(i_d * stride_d)`` over its dims (size, stride), outermost
    first, in row-major order; strides are positive."""
    import numpy as np

    out = []
    i = 0
    while i < len(xs):
        x = xs[i:]
        if len(x) == 1 or int(x[1] - x[0]) <= 0:
            out.append((int(x[0]), [], 1))
            i += 1
            continue
        d = int(x[1] - x[0])
        off = (x[1:] - x[:-1]) != d
        n1 = int(off.argmax()) + 1 if off.any() else len(x)
        if n1 == len(x) or len(x) % n1:
            out.append((int(x[0]), [(n1, d)], n1))
            i += n1
            continue
        m = x.reshape(-1, n1)
        if not ((m - m[:, :1]) == np.arange(n1, dtype=np.int64) * d).all():
            out.append((int(x[0]), [(n1, d)], n1))
            i += n1
            continue
        for x0, dims, n in _factor(m[:, 0]):
            out.append((x0, dims + [(n1, d)], n * n1))
        i += len(x)
    return out


def _choose(plan: Plan, holders: Mapping[int, list[int]], src_rail: Mapping[int, int],
            group: list[int], dst_rail: list[int], small_row: int, sent: dict[int, int]):
    """Holder and mirror position of every plan source the ranks ``group`` read, and
    what each reads; ``sent`` (bytes per trainer rank) is updated.

    Largest tensors first. A tensor goes to a holder on one of the node's rails,
    minimizing the busier of that rail and that trainer GPU, then the rail's bytes,
    a mirror GPU that consumes it, the trainer GPU's bytes; with none on the node's
    rails, to the holder sending the fewest, mirrored off-rail by the consumer
    landing the fewest bytes (the same position for every tensor of that holder)."""
    pos = {d: i for i, d in enumerate(group)}
    land_on: dict[int, int] = {}
    for i, r in enumerate(dst_rail):
        land_on.setdefault(int(r), i)

    sel = torch.isin(plan.rows[:, 0], torch.tensor(list(group), dtype=torch.int64))
    need: dict[int, list[tuple[int, ...]]] = {}
    for r in plan.rows[sel].tolist():
        q = r[_SP]
        if q not in holders:
            raise KeyError(f"no trainer rank exports plan source {plan.src_names[q]}")
        p, rb, nr = pos[r[0]], r[_RB], r[_NR]
        if nr > 1 and r[_SS] != rb and rb >= small_row:
            need.setdefault(q, []).extend(
                (p, r[_DP], r[_DO] + i * r[_DS], 1, rb, rb, r[_SO] + i * r[_SS], rb)
                for i in range(nr))
        else:
            need.setdefault(q, []).append(
                (p, r[_DP], r[_DO], nr, rb, r[_DS], r[_SO], r[_SS]))

    spans: dict[int, list[list[int]]] = {}
    for q, rs in need.items():
        merged: list[list[int]] = []
        for lo, hi in sorted((so, so + (nr - 1) * ss + rb) for _, _, _, nr, rb, _, so, ss in rs):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        spans[q] = merged

    load: dict[int, int] = defaultdict(int)
    landed = [0] * len(group)
    off_host: dict[int, int] = {}
    choice: dict[int, tuple[int, int]] = {}
    offrail = 0
    for q in sorted(need, key=lambda q: (-sum(h - lo for lo, h in spans[q]), q)):
        b = sum(h - lo for lo, h in spans[q])
        users = {x[0] for x in need[q]}
        near = [s for s in sorted(holders[q]) if src_rail[s] in land_on]
        if near:
            s = min(near, key=lambda s: (max(load[src_rail[s]], sent[s]), load[src_rail[s]],
                                         land_on[src_rail[s]] not in users, sent[s], s))
            at = land_on[src_rail[s]]
        else:
            s = min(sorted(holders[q]), key=lambda s: (sent[s], s))
            at = off_host.setdefault(s, min(sorted(users), key=lambda p: (landed[p], p)))
            offrail += b
        load[int(dst_rail[at])] += b
        landed[at] += b
        sent[s] += b
        choice[q] = (s, at)
    return need, spans, choice, load, offrail


_FLAG_BITS = 24
_FLAG_MASK = (1 << _FLAG_BITS) - 1
_ELEM_DTYPE = {8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.uint8}


class _Signals:
    """Per-node flags in shared memory, one writer each: ``landed[L]``, the sync
    position ``L`` is fetching (high bits) and how many of its mirror batches have
    landed (low ``_FLAG_BITS``); ``released[L, c]``, the last sync consumer ``c``
    has finished reading ``L``'s mirror for."""

    def __init__(self, path: str, n: int, create: bool) -> None:
        import numpy as np

        size = 8 * (n + n * n)
        if create:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(fd, size)
            weakref.finalize(self, _unlink, path)
        else:
            fd = os.open(path, os.O_RDWR)
        try:
            self._mm = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        a = np.frombuffer(self._mm, dtype=np.int64)
        self.landed, self.released = a[:n], a[n:].reshape(n, n)


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


class MoriNodeReader:
    """Rollout rank ``dst_rank`` of the plan ranks ``group`` on its node (every vLLM
    instance's ranks there), on another node than the trainers (module docstring).
    ``exchange(obj)`` gathers one object from every rank of ``group`` in order
    (``NodeExchange`` when the group spans several instances);
    ``signal_key`` names the node's shared-memory flags, the same on every rank of
    the group and unique per job. Every rank of the group must call ``sync`` for
    every version: a rank waits on the others' mirrors. ``groups`` and ``rails``:
    every instance reading over RDMA and its ranks' rails (``plan_node``).
    """

    def __init__(self, plan: Plan, dst_rank: int, params: Mapping[str, torch.Tensor],
                 sources: list, group: list[int], host: str,
                 exchange: Callable[[Any], list], signal_key: str, num_streams: int = 64,
                 small_row: int | None = None, window: int | None = None,
                 batch: int | None = None, timeout_s: float = 60.0,
                 groups: list[list[int]] | None = None,
                 rails: Mapping[int, int] | None = None, strided: bool = True) -> None:
        t0 = time.perf_counter()
        sources = [SourceExport.from_wire(s).over_rdma() for s in sources]
        self.dst_rank, self.timeout_s = int(dst_rank), float(timeout_s)
        self.device = torch.cuda.current_device()
        group = list(group)
        self.me = me = group.index(self.dst_rank)
        rows = plan.rows[plan.col("dst_rank") == self.dst_rank]
        if rows.numel() == 0:
            raise ValueError(f"plan has no reads for rollout rank {dst_rank}")
        used = sorted(set(rows[:, _DP].tolist()))
        ckpt = plan.meta.get("staged_checkpoint")
        self.checkpoint = bool(ckpt)
        if ckpt:
            spec = {plan.dst_names[q]: ckpt.get(plan.dst_names[q]) for q in used}
            if any(s is None for s in spec.values()):
                raise ValueError("plan reads into tensors meta['staged_checkpoint'] lacks")
            dst_nbytes = {q: torch.empty((), dtype=getattr(torch, spec[plan.dst_names[q]]["dtype"]))
                          .element_size() * math.prod(spec[plan.dst_names[q]]["shape"])
                          for q in used}
        else:
            missing = [plan.dst_names[q] for q in used if plan.dst_names[q] not in params]
            if missing:
                raise KeyError(f"rollout engine lacks {len(missing)} plan params, "
                               f"e.g. {missing[:5]}")
            dst_nbytes = {q: params[plan.dst_names[q]].numel()
                          * params[plan.dst_names[q]].element_size() for q in used}
        # Against the trainers' sizes; the reader checks the mirrors'.
        check_bounds(rows, dst_nbytes,
                     {(s.src_rank, q): v[2] for s in sources for q, v in s.params.items()}, plan)
        src_rail, dst_rail = _rails(sources, exchange, self.device)
        if src_rail is None:
            raise ValueError("the node reader needs every trainer and rollout GPU's rail "
                             "(rail_topology found no RDMA NICs or GPUs in sysfs)")
        self.rail = dst_rail[me]
        # Parameters this rank can land in place: registered per caching-allocator
        # segment, since an RDMA registration must not span unmapped addresses.
        views = plan.meta.get("views") or {}
        segments = [(s["address"], s["total_size"]) for s in torch.cuda.memory_snapshot()
                    if s.get("device", self.device) == self.device]
        direct = in_place_enabled()
        try:
            seg = {q: _allocation_of(params[plan.dst_names[q]].data_ptr(), segments)
                   for q in ([] if ckpt else used) if plan.dst_names[q] not in views}
            ok = direct and not ckpt
        except RuntimeError:
            seg, ok = {}, False
        flags = exchange((ok, direct))
        node = plan_node(plan, sources, group, dst_rail, int(small_row or small_row_bytes()),
                         groups, {**(rails or {}), **dict(zip(group, dst_rail))}, batch,
                         [bool(f[0]) for f in flags], [bool(f[1]) for f in flags])
        self.node = node
        path = f"/dev/shm/lumen-mori-node-{signal_key}"
        if me == 0:
            self.signals = _Signals(path, len(group), create=True)

        # Network: the mirrors this position hosts and the parameters it lands in
        # place, fetched in source order so MORI merges adjacent reads.
        self.debug = os.environ.get("LUMEN_MORI_NODE_DEBUG", "0") == "1"
        size = max(_ALIGN, node.mirror_bytes[me])
        if self.debug:
            size = -(-size // _DEBUG_CHUNK) * _DEBUG_CHUNK
        self.mirror = torch.empty(size, dtype=torch.uint8, device=self.device)
        self.eng, remote = _connect(sources, f"node{self.dst_rank}", num_streams, host)
        self.net_mem = _register(self.eng, self.mirror.data_ptr(), self.mirror.numel(),
                                 self.device)
        self.land_mems: dict[int, Any] = {}
        land_mem, land_at = {}, {}
        for dp in sorted({k[4] for b in node.fetch[me] for k in b if k[4] >= 0}):
            base, size = seg[dp]
            if base not in self.land_mems:
                self.land_mems[base] = _register(self.eng, base, size, self.device)
            land_mem[dp] = self.land_mems[base]
            land_at[dp] = params[plan.dst_names[dp]].data_ptr() - base
        by_rank = {s.src_rank: s for s in sources}

        def entry(k):
            s, q, so, n, dp, do = k
            at = do if dp >= 0 else node.offsets[(s, q)] + so
            return (0, s, dp, at, q, so, n, 1, n, n)

        self.calls = [_read_calls([entry(k) for k in b], by_rank, remote, self.net_mem,
                                  lambda dp: land_at.get(dp, 0),
                                  lambda dp: land_mem.get(dp, self.net_mem))[0]
                      for b in node.fetch[me]]
        self.entries = [k for b in node.fetch[me] for k in b]
        self.net_bytes = int(sum(k[3] for k in self.entries))
        self.land_bytes = int(node.land_bytes[me])
        self.rail_local_bytes = int(sum(k[3] for k in self.entries if src_rail[k[0]] == self.rail))
        self.window = max(1, int(window or int(os.environ.get("LUMEN_MORI_RDMA_WINDOW", 3))))

        # Node: the mirrors exported over XGMI as the trainer ranks they mirror.
        self.xgmi = _new_engine(f"node{self.dst_rank}-xgmi", num_streams)
        segments = [(s["address"], s["total_size"]) for s in torch.cuda.memory_snapshot()
                    if s.get("device", self.device) == self.device]
        base, size = _allocation_of(self.mirror.data_ptr(), segments)
        self.mirror_mem = _register(self.xgmi, base, size, self.device)
        at = self.mirror.data_ptr() - base
        hosted = sorted(s for s, h in node.host.items() if h == me)
        mine: dict[int, dict[int, int]] = defaultdict(dict)
        for (s, q), off in node.offsets.items():
            mine[s][q] = off
        desc, mem = bytes(self.xgmi.get_engine_desc().pack()), bytes(self.mirror_mem.pack())
        ext = node.view_extent or node.extent
        peers = exchange([SourceExport(s, desc, [mem], {q: (0, at + o, ext[(s, q)])
                                                        for q, o in mine[s].items()}).to_wire()
                          for s in hosted])
        if me != 0:
            self.signals = _Signals(path, len(group), create=False)
        local = {s: {q: self.mirror[o:o + ext[(s, q)]] for q, o in mine[s].items()}
                 for s in hosted}
        # Packing: one scratch buffer for the largest run, on a stream of its own.
        self.pack = node.pack[me] if node.pack else {}
        runs = [p for ps in self.pack.values() for p in ps]
        self.pack_bytes = int(sum(p.nbytes * p.elem for p in runs))
        self.scratch = torch.empty(_aligned(max((p.nbytes * p.elem for p in runs), default=0)),
                                   dtype=torch.uint8, device=self.device)
        self.pack_stream = torch.cuda.Stream(self.device) if runs else None
        mine_rows = node.rows[:, 0] == self.dst_rank
        node_plan = Plan(plan.dst_names, plan.src_names, node.rows, plan.meta)
        others = [e for i, es in enumerate(peers) if i != me for e in es]
        self.stage_calls: dict[int, list] = {}
        if ckpt:
            self.reader = MoriCheckpointReader(
                node_plan, self.dst_rank, others, self.device, num_streams=num_streams,
                strided=strided, local_sources=local, fetched=node.staged[mine_rows],
                fetch=self._fetch)
            units = {i: {q: o for q, o, _ in b.tensors} for i, b in enumerate(self.reader.batches)}
        else:
            self.reader = MoriWeightReader(
                node_plan, self.dst_rank, params, others, num_streams=num_streams,
                strided=strided, local_sources=local, in_place=node.in_place[mine_rows],
                fetched=node.staged[mine_rows], fetch=self._fetch, num_slots=self.window + 1)
            units = {q: {q: 0} for q in self.reader.fetched}
        # Staged rows: fetched over the network into the reader's slots, one batch per
        # unit (a permuted param, or a checkpoint batch) and slot.
        staged = node.rows[mine_rows & node.staged].tolist()
        self.slot_mems = [_register(self.eng, s.data_ptr(), s.numel(), self.device)
                          for s in self.reader.slots] if staged else []
        for u, off in units.items():
            ks = [(k, []) for r in staged if r[_DP] in off for k in row_pieces(r, r[_SRC])]
            if not ks:
                continue
            es = [(0, s, dp, off[dp] + do, q, so, n, 1, n, n)
                  for (s, q, so, n, dp, do), _ in merge_pieces(ks)]
            self.stage_calls[u] = [_read_calls(es, by_rank, remote, m, lambda _: 0)[0]
                                   for m in self.slot_mems]
        self.stage_bytes = int(node.stage_bytes[me])
        self.net_bytes += self.stage_bytes
        self.rail_local_bytes += int(sum(r[_RB] * r[_NR] for r in staged
                                         if src_rail[r[_SRC]] == self.rail))

        nb = node.rows[:, _RB] * node.rows[:, _NR]
        at_host = torch.tensor([node.host[s] for s in node.rows[:, _SRC].tolist()])
        pos = torch.tensor([group.index(d) for d in node.rows[:, 0].tolist()])
        self.out_bytes = int(nb[(at_host == me) & (pos != me)].sum())
        self.in_bytes = int(nb[(at_host != me) & (pos == me)].sum())
        self.consumers = sorted(set(pos[(at_host == me) & (pos != me)].tolist()))
        self.hosts = sorted(set(at_host[(at_host != me) & (pos == me)].tolist()))
        self.waits_on: dict[int, set[int]] = defaultdict(set)
        read = (pos == me) & ~node.in_place & ~node.staged
        for r, h in zip(node.rows[read].tolist(), at_host[read].tolist()):
            self.waits_on[r[_ROUND]].add(h)
        # Round r's gate: every host's batches up to its last round <= r that this
        # rank reads, so a gate also covers the rounds before it.
        self.gate_rounds = sorted(self.waits_on)
        self.gate_need: list[dict[int, int]] = []
        upto: dict[int, int] = {}
        for rnd in self.gate_rounds:
            upto.update({h: rnd for h in self.waits_on[rnd]})
            self.gate_need.append(dict(upto))
        if os.environ.get("LUMEN_MORI_NODE_OVERLAP", "1") == "0" and self.gate_need:
            self.gate_need = [dict(upto) for _ in self.gate_need]
        self.version = 0
        self.bytes, self.strided = self.reader.bytes, self.reader.strided
        self.views = getattr(self.reader, "views", {})
        self.staging_bytes = self.mirror.numel() + self.reader.staging_bytes
        if self.debug:
            self._debug_setup(plan, params, mine_rows)
        self.setup_s = time.perf_counter() - t0

    def _debug_setup(self, plan: Plan, params: Mapping[str, torch.Tensor],
                     mine_rows: torch.Tensor) -> None:
        """Per network batch, the mirror chunks no later batch writes (its ranges, and
        whether every byte of each chunk is fetched); the rows read from mirrors into
        vLLM parameters, for the sentinel count after a sync."""
        node, me = self.node, self.me
        self.debug_delay = float(os.environ.get("LUMEN_MORI_NODE_DEBUG_DELAY_MS", 20)) / 1e3
        self.names = (plan.src_names, plan.dst_names)
        n = self.mirror.numel() // _DEBUG_CHUNK
        owner, covered, first = [-1] * n, [0] * n, {}
        for i, b in enumerate(node.fetch[me]):
            for s, q, so, nb, dp, _ in b:
                if dp >= 0:
                    continue
                at = node.offsets[(s, q)] + so
                for c in range(at // _DEBUG_CHUNK, (at + nb - 1) // _DEBUG_CHUNK + 1):
                    owner[c] = max(owner[c], i)
                    covered[c] += min(at + nb, (c + 1) * _DEBUG_CHUNK) - max(at, c * _DEBUG_CHUNK)
                    first.setdefault(c, (s, q))
        self.debug_chunks: list[list[int]] = [[] for _ in node.fetch[me]]
        for c, i in enumerate(owner):
            if i >= 0:
                self.debug_chunks[i].append(c)
        self.debug_full = [[covered[c] == _DEBUG_CHUNK for c in cs] for cs in self.debug_chunks]
        self.debug_first = first
        rows = node.rows[mine_rows & ~node.in_place & ~node.staged]
        self.debug_rows = [] if self.checkpoint else [
            r for r in rows.tolist() if r[_DP] not in self.views
            and r[_DO] % 2 == 0 and r[_RB] % 2 == 0 and (r[_NR] == 1 or r[_DS] % 2 == 0)]
        self.debug_dst = {q: params[plan.dst_names[q]].detach().view(-1).view(torch.int16)
                          for q in {r[_DP] for r in self.debug_rows}}

    def _debug_hash(self, chunks: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Word sum and mirror-sentinel word count of each of ``chunks``."""
        words = self.mirror.view(torch.int32).view(-1, _DEBUG_CHUNK // 4)
        fill = int.from_bytes(bytes([_MIRROR_FILL]) * 4, "little", signed=True)
        sums, sent = [], []
        j = 0
        while j < len(chunks):
            k = j + 1
            while k < len(chunks) and chunks[k] == chunks[k - 1] + 1:
                k += 1
            x = words[chunks[j]:chunks[k - 1] + 1]
            sums.append(x.sum(dim=1, dtype=torch.int64))
            sent.append((x == fill).sum(dim=1))
            j = k
        return torch.cat(sums).cpu(), torch.cat(sent).cpu()

    def _debug_batch(self, i: int) -> None:
        """Hash batch ``i``'s mirror chunks as it completes and ``debug_delay`` later: a
        chunk that changed was written after MORI reported the batch done."""
        chunks = self.debug_chunks[i]
        if not chunks:
            return
        torch.cuda.synchronize(self.device)
        s1, n1 = self._debug_hash(chunks)
        if self.debug_delay > 0:
            time.sleep(self.debug_delay)
        s2, n2 = self._debug_hash(chunks)
        s = self._sync
        for j in torch.nonzero((s1 != s2) | (n1 != n2)).flatten().tolist():
            s["debug_changed"].append((i, chunks[j], int(n1[j]), int(n2[j])))
        for j in torch.nonzero(n2 > 0).flatten().tolist():
            if self.debug_full[i][j]:
                s["debug_missing"].append((i, chunks[j], int(n2[j])))

    def _debug_dest(self) -> torch.Tensor:
        """Per debug row: mirror- and scratch-sentinel half-words in its vLLM destination."""
        a = int.from_bytes(bytes([_MIRROR_FILL]) * 2, "little", signed=True)
        b = int.from_bytes(bytes([_SCRATCH_FILL]) * 2, "little", signed=True)
        out = []
        for r in self.debug_rows:
            v = self.debug_dst[r[_DP]].as_strided((r[_NR], r[_RB] // 2),
                                                  (max(r[_DS] // 2, 1), 1), r[_DO] // 2)
            out.append(torch.stack([(v == a).sum(), (v == b).sum()]))
        return torch.stack(out).cpu() if out else torch.zeros(0, 2, dtype=torch.int64)

    def _debug_report(self, before: torch.Tensor) -> None:
        s, (src_names, dst_names) = self._sync, self.names
        after = self._debug_dest()
        gained = torch.nonzero((after > before).any(dim=1)).flatten().tolist()

        def chunk(i, c):
            sr, q = self.debug_first.get(c, (-1, -1))
            return f"batch {i} chunk {c} (trainer rank {sr} {src_names[q] if q >= 0 else '?'})"

        changed = "; ".join(f"{chunk(i, c)} sentinel words {a}->{b}"
                            for i, c, a, b in s["debug_changed"][:4])
        missing = "; ".join(f"{chunk(i, c)} {w} sentinel words"
                            for i, c, w in s["debug_missing"][:4])
        rows = "; ".join(
            f"{dst_names[r[_DP]]} at {r[_DO]} <- trainer rank {r[_SRC]} {src_names[r[_SP]]} "
            f"round {r[_ROUND]}: {int(after[j, 0] - before[j, 0])} mirror and "
            f"{int(after[j, 1] - before[j, 1])} scratch sentinel half-words"
            for j in gained[:4] for r in [self.debug_rows[j]])
        logger.warning(
            "MORI node debug on rollout rank %d (position %d, rail %d), version %d: %d "
            "batches rechecked after %.0f ms; %d mirror chunks changed after their batch "
            "completed%s; %d fully fetched chunks still held mirror sentinel words%s; %d of "
            "%d reads from mirrors left sentinel words in vLLM's weights%s",
            self.dst_rank, self.me, self.rail, s["v"],
            sum(1 for c in self.debug_chunks if c), self.debug_delay * 1e3,
            len(s["debug_changed"]), f" ({changed})" if changed else "",
            len(s["debug_missing"]), f" ({missing})" if missing else "",
            len(gained), len(self.debug_rows), f" ({rows})" if rows else "")

    def _wait(self, ready: Callable[[], bool], what: str, t0: float,
              need: Mapping[int, int] | None = None) -> None:
        while not ready():
            if time.perf_counter() - t0 > self.timeout_s:
                late = ""
                if need is not None:
                    v = self._sync["v"]
                    late = "; late: " + ", ".join(
                        f"position {h} has {self._landed_of(h, v)} of {r + 1}"
                        for h, r in sorted(need.items()) if self._landed_of(h, v) <= r)
                    logger.warning("%s", self.report("timed out"))
                raise TimeoutError(f"MORI node sync: rollout rank {self.dst_rank} waited "
                                   f"{self.timeout_s:.0f} s for {what}{late}")
            time.sleep(20e-6)

    def _landed_of(self, h: int, v: int) -> int:
        """Batches position ``h`` has flagged for version ``v`` (0 if still on an older one)."""
        x = int(self.signals.landed[h])
        return x & _FLAG_MASK if x >> _FLAG_BITS == v else 0

    def report(self, why: str) -> str:
        """This rank's network fetch in the current sync: each batch's issue and landing
        time, its trainer ranks and bytes, and the longest gap between polls."""
        s = self._sync
        now = time.perf_counter() - s["t0"]
        rows = []
        for i, b in enumerate(self.node.fetch[self.me]):
            if i >= len(s["issued"]):
                break
            done = s["ready"][i] if i < len(s["ready"]) else None
            rows.append(f"{i}: {s['issued'][i]:.3f}-" + (f"{done:.3f}" if done is not None else "...")
                        + f" src {sorted({k[0] for k in b})} {sum(k[3] for k in b) / 1e6:.0f}MB")
        waits = sorted(s["gate_waits"], key=lambda w: -w[0])
        gates = "; ".join(f"{dt:.3f}s from {at:.3f}s for round {rnd} behind "
                          + ",".join(f"{h}@{n}" for h, n in sorted(b.items()))
                          for dt, rnd, at, b in waits[:4])
        return (f"MORI node fetch on rollout rank {self.dst_rank} (position {self.me}, rail "
                f"{self.rail}) {why} at {now:.3f}s, started at wall {s['wall0']:.3f}, "
                f"reader {'done' if s.get('read_end') is not None else 'not done'}"
                + (f" at {s['read_end']:.3f}s" if s.get('read_end') is not None else "")
                + f": {len(s['ready'])} of {len(self.calls)} "
                f"batches landed, {len(s['inflight'])} in flight; longest gap between polls "
                f"{s['max_gap']:.3f}s at {s['gap_at']:.3f}s; {s['gate_calls']} gate calls, "
                f"{len(waits)} waited, {sum(w[0] for w in waits):.3f}s, longest (position@landed): "
                f"{gates or 'none'}; batches (issued-landed): " + "; ".join(rows))

    def _advance(self, block: bool) -> None:
        """Issue network batches up to ``window`` in flight and flag the ones landed, in
        order; ``block``: wait for at least one."""
        s = self._sync
        t = time.perf_counter()
        if t - s["polled"] > s["max_gap"]:
            s["max_gap"], s["gap_at"] = t - s["polled"], s["polled"] - s["t0"]
        try:
            self._advance_inner(block)
        finally:
            s["polled"] = time.perf_counter()

    def _advance_inner(self, block: bool) -> None:
        s = self._sync
        while True:
            while s["next"] < len(self.calls) and len(s["inflight"]) < self.window:
                s["issued"].append(time.perf_counter() - s["t0"])
                s["inflight"].append(_issue(self.eng, False, self.calls[s["next"]]))
                s["next"] += 1
            if not s["inflight"]:
                return
            head = s["inflight"][0]
            if not block and any(st.InProgress() for st in head):
                return
            t = time.perf_counter()
            _drain(s["inflight"].popleft(), s["failures"])
            s["wait_s"] += time.perf_counter() - t
            if s["failures"]:
                raise RuntimeError(f"MORI weight sync: {len(s['failures'])} RDMA reads failed "
                                   f"on rollout rank {self.dst_rank}: "
                                   f"{[f for f in s['failures'] if f][:4]}")
            if self.debug:
                self._debug_batch(len(s["ready"]))
            self._pack(len(s["ready"]))
            s["ready"].append(time.perf_counter() - s["t0"])
            self.signals.landed[self.me] = s["v"] << _FLAG_BITS | len(s["ready"])
            block = False

    def _pack(self, batch: int) -> None:
        """Pack the mirror runs network batch ``batch`` completes, before it is flagged."""
        runs = self.pack.get(batch)
        if not runs:
            return
        t = time.perf_counter()
        with torch.cuda.stream(self.pack_stream):
            apply_pack(self.mirror, self.scratch, runs)
        self.pack_stream.synchronize()
        self._sync["pack_s"] += time.perf_counter() - t

    def _fetch(self, unit: int, slot: int) -> list:
        """Issue staging unit ``unit``'s network reads into reader slot ``slot``."""
        self._advance(block=False)
        calls = self.stage_calls.get(unit)
        return list(_issue(self.eng, False, calls[slot])) if calls else []

    def _gate(self, rnd: int) -> None:
        """Returns once every mirror this rank reads in rounds up to ``rnd`` has landed
        them."""
        s, sig = self._sync, self.signals
        i = bisect.bisect_right(self.gate_rounds, rnd) - 1
        need = self.gate_need[i] if i >= 0 else {}
        t = time.perf_counter()
        behind = {h: self._landed_of(h, s["v"]) for h, r in need.items()
                  if self._landed_of(h, s["v"]) <= r}

        def ready() -> bool:
            self._advance(block=False)
            return all(int(sig.landed[h]) >> _FLAG_BITS == s["v"]
                       and int(sig.landed[h]) & _FLAG_MASK > r for h, r in need.items())

        self._wait(ready, f"mirror batches up to {rnd} of positions {sorted(need)}", s["t0"],
                   need)
        dt = time.perf_counter() - t
        s["gate_s"] += dt
        s["gate_calls"] += 1
        if behind:
            s["gate_waits"].append((dt, rnd, t - s["t0"], behind))

    def sync(self, load: Callable[[list[tuple[str, torch.Tensor]]], Any] | None = None,
             verify: bool = False) -> dict[str, Any]:
        """Fetch this rank's mirrors, flagging each landed batch, while the node plan
        reads every round from the mirrors once its batches have landed. A checkpoint
        plan passes each staged batch to ``load`` (``MoriCheckpointReader.sync``)."""
        t0 = time.perf_counter()
        v = self.version + 1
        sig, me = self.signals, self.me
        self._wait(lambda: all(sig.released[me, c] >= v - 1 for c in self.consumers),
                   f"its mirror's release of version {v - 1}", t0)
        release_s = time.perf_counter() - t0
        if self.debug:
            self.mirror.fill_(_MIRROR_FILL)
            self.scratch.fill_(_SCRATCH_FILL)
            before = self._debug_dest()
            torch.cuda.synchronize(self.device)
        sig.landed[me] = v << _FLAG_BITS
        self._sync = s = {"v": v, "t0": t0, "next": 0, "inflight": deque(), "issued": [],
                          "ready": [], "failures": [], "wait_s": 0.0, "gate_s": 0.0,
                          "pack_s": 0.0, "polled": time.perf_counter(), "max_gap": 0.0,
                          "gap_at": 0.0, "wall0": time.time() - (time.perf_counter() - t0),
                          "gate_calls": 0, "gate_waits": [], "debug_changed": [],
                          "debug_missing": []}
        self._advance(block=False)
        if self.checkpoint:
            st = self.reader.sync(load, verify=verify, gate=self._gate)
        else:
            st = self.reader.sync(gate=self._gate)
        s["read_end"] = time.perf_counter() - t0
        while s["next"] < len(self.calls) or s["inflight"]:
            self._advance(block=True)
        net_s = s["ready"][-1] if s["ready"] else 0.0
        if self.debug:
            torch.cuda.synchronize(self.device)
            self._debug_report(before)
        for h in self.hosts:
            sig.released[h, me] = v
        seconds = time.perf_counter() - t0
        self.version = v
        self.timeline = list(zip(s["issued"], s["ready"]))
        if seconds > float(os.environ.get("LUMEN_MORI_SLOW_SYNC_S", 2.0)):
            logger.warning("%s", self.report(f"slow sync ({seconds:.3f}s, gate wait "
                                             f"{s['gate_s']:.3f}s)"))
        return {"seconds": seconds, "bytes": float(self.bytes), "rounds": float(len(self.calls)),
                "reads": float(len(self.entries)), "read_wait_s": s["wait_s"],
                "net_bytes": float(self.net_bytes),
                "land_bytes": float(self.land_bytes), "stage_bytes": float(self.stage_bytes),
                "mirror_bytes": float(self.mirror.numel()), "out_bytes": float(self.out_bytes),
                "in_bytes": float(self.in_bytes), "release_s": release_s, "net_s": net_s,
                "gate_wait_s": s["gate_s"], "direct_s": st.get("direct_s", 0.0),
                "pack_s": s["pack_s"], "pack_bytes": float(self.pack_bytes),
                "poll_gap_s": s["max_gap"],
                "permuted": st.get("permuted", 0.0),
                **({"load_s": st["load_s"], "slot_wait_s": st["read_wait_s"]}
                   if self.checkpoint else {}),
                "rail_local_bytes": float(self.rail_local_bytes), "rail": float(self.rail),
                "gbps": self.bytes / seconds / 1e9 if seconds > 0 else 0.0,
                "net_gbps": self.net_bytes / seconds / 1e9 if seconds > 0 else 0.0}

    def hashes(self) -> torch.Tensor:
        """Hash of every plan read for this rank, in plan row order."""
        return self.reader.hashes()
