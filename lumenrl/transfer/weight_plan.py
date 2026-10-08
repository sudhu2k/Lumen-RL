"""Transfer plans from two source maps: which bytes each rollout rank reads from where.

Both maps say which checkpoint element each parameter element holds
(``weight_source_map``), so the checkpoint is the pivot: intersecting a rollout
run's id range with the trainer runs holding those ids gives contiguous copies
``(dst param, dst offset) <- (src param, src offset)``. The planner then splits
copies into pieces, cuts them into rounds of a fixed byte budget per rollout rank,
and assigns each piece to one trainer replica, balancing bytes per (trainer,
rollout) pair. Every rollout rank's read list per round and source is what a
batched MORI read takes.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Mapping, Optional

import torch

from lumenrl.transfer.weight_source_map import SourceMap

__all__ = ["Copies", "correspond", "Plan", "plan_rounds", "apply_plan"]

_ELEM = {"bfloat16": 2, "float16": 2, "float32": 4, "uint8": 1, "float8_e4m3fn": 1}


@dataclass
class Copies:
    """Element copies of ``rows`` rows of ``n`` elements: row ``i`` is
    ``dst[dst_off + i*dst_stride :][:n] <- src[src_off + i*src_stride :][:n]``.

    ``rows`` and the strides are None for single-row copies (before ``_rectangles``).
    """

    dst_names: list[str]
    src_names: list[str]
    dst: torch.Tensor
    dst_off: torch.Tensor
    src: torch.Tensor
    src_off: torch.Tensor
    n: torch.Tensor
    elem: torch.Tensor
    report: dict = field(default_factory=dict)
    rows: Optional[torch.Tensor] = None
    dst_stride: Optional[torch.Tensor] = None
    src_stride: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return int(self.n.numel())

    def nrows(self) -> torch.Tensor:
        return self.rows if self.rows is not None else torch.ones_like(self.n)

    def total_bytes(self) -> int:
        return int((self.n * self.elem * self.nrows()).sum())


def correspond(src: SourceMap, dst: SourceMap, dst_params: Optional[list[str]] = None) -> Copies:
    """Element copies that fill ``dst``'s written elements from ``src``.

    Args:
        src: The trainer map (one replica).
        dst: The rollout map.
        dst_params: Destination parameters to cover; default all valid ones.

    Only unit-step runs are joined (slice copies); 2-D runs are joined row by row and
    the resulting copies merged back into rectangles. Anything else, and any element
    no source holds, is listed in ``report`` instead.
    """
    if src.meta.get("checkpoint") != dst.meta.get("checkpoint"):
        raise ValueError("source maps were built from different checkpoints")
    src_names = sorted(src.params)
    s_ckpt, s_len, s_param, s_off = [], [], [], []
    unsupported_src = []
    for i, name in enumerate(src_names):
        p = src.params[name]
        r = p.runs.as_lines() if p.valid and p.runs is not None else None
        if r is None:
            unsupported_src.append(name)
            continue
        length = r.length
        unit = (r.ckpt_strides[:, 0] == 1) | (length == 1)
        if not bool(unit.all()):
            unsupported_src.append(name)
        s_ckpt.append(r.ckpt_offset[unit]), s_len.append(length[unit])
        s_off.append(r.param_offset[unit])
        s_param.append(torch.full((int(unit.sum()),), i, dtype=torch.int64))
    ckpt0 = torch.cat(s_ckpt)
    order = torch.argsort(ckpt0)
    ckpt0, length = ckpt0[order], torch.cat(s_len)[order]
    sparam, soff = torch.cat(s_param)[order], torch.cat(s_off)[order]
    end = ckpt0 + length
    dup = int((ckpt0[1:] < end[:-1]).sum()) if ckpt0.numel() > 1 else 0
    ckpt0_l, end_l = ckpt0.tolist(), end.tolist()
    sparam_l, soff_l = sparam.tolist(), soff.tolist()

    names = dst_params if dst_params is not None else sorted(
        n for n, p in dst.params.items() if p.valid)
    out = {k: [] for k in ("dst", "dst_off", "src", "src_off", "n", "elem")}
    missing: dict[str, int] = {}
    unsupported_dst, dtype_mismatch = [], []
    for di, name in enumerate(names):
        p = dst.params[name]
        r = p.runs.as_lines() if p.runs is not None else None
        if r is None:
            unsupported_dst.append(name)
            continue
        esize = _ELEM[p.dtype]
        for off, ln, a, st in zip(r.param_offset.tolist(), r.length.tolist(),
                                  r.ckpt_offset.tolist(), r.ckpt_strides[:, 0].tolist()):
            if ln > 1 and st != 1:
                unsupported_dst.append(name)
                continue
            b = a + ln
            i = max(bisect.bisect_right(ckpt0_l, a) - 1, 0)
            covered = 0
            while i < len(ckpt0_l) and ckpt0_l[i] < b:
                lo, hi = max(a, ckpt0_l[i]), min(b, end_l[i])
                if lo < hi:
                    si = sparam_l[i]
                    if src.params[src_names[si]].dtype != p.dtype:
                        dtype_mismatch.append(name)
                    out["dst"].append(di), out["dst_off"].append(off + lo - a)
                    out["src"].append(si), out["src_off"].append(soff_l[i] + lo - ckpt0_l[i])
                    out["n"].append(hi - lo), out["elem"].append(esize)
                    covered += hi - lo
                i += 1
            if covered != ln:
                missing[name] = missing.get(name, 0) + ln - covered
    t = {k: torch.tensor(v, dtype=torch.int64) for k, v in out.items()}
    copies = Copies(names, src_names, t["dst"], t["dst_off"], t["src"], t["src_off"], t["n"],
                    t["elem"])
    copies = _coalesce(copies)
    line_copies = len(copies)
    copies = _rectangles(copies)
    copies.report = {
        "copies": len(copies), "line_copies": line_copies, "bytes": copies.total_bytes(),
        "missing_elements": missing, "unsupported_src": sorted(set(unsupported_src)),
        "unsupported_dst": sorted(set(unsupported_dst)),
        "dtype_mismatch": sorted(set(dtype_mismatch)), "overlapping_src_runs": dup,
    }
    return copies


def _coalesce(c: Copies) -> Copies:
    """Merge copies that continue each other on both sides."""
    if len(c) < 2:
        return c
    order = torch.argsort(c.dst * (1 << 40) + c.dst_off)
    f = {k: getattr(c, k)[order] for k in ("dst", "dst_off", "src", "src_off", "n", "elem")}
    cont = torch.zeros(len(c), dtype=torch.bool)
    cont[1:] = ((f["dst"][1:] == f["dst"][:-1]) & (f["src"][1:] == f["src"][:-1])
                & (f["dst_off"][1:] == f["dst_off"][:-1] + f["n"][:-1])
                & (f["src_off"][1:] == f["src_off"][:-1] + f["n"][:-1]))
    head = torch.cumsum((~cont).to(torch.int64), 0) - 1
    keep = ~cont
    n = torch.zeros(int(keep.sum()), dtype=torch.int64).index_add_(0, head, f["n"])
    return Copies(c.dst_names, c.src_names, f["dst"][keep], f["dst_off"][keep], f["src"][keep],
                  f["src_off"][keep], n, f["elem"][keep], c.report)


def _rectangles(c: Copies) -> Copies:
    """Merge single-row copies of equal length between the same two tensors whose
    offsets advance by constant amounts on both sides into multi-row copies.

    A row-parallel TP shard (half of every row of the rollout tensor) becomes one copy
    instead of one per row.
    """
    if len(c) < 2:
        return c
    order = torch.argsort(c.dst_off, stable=True)
    for k in (c.n, c.src, c.dst):
        order = order[torch.argsort(k[order], stable=True)]
    dst, doff, src, soff, n, elem = (getattr(c, k)[order].tolist()
                                     for k in ("dst", "dst_off", "src", "src_off", "n", "elem"))
    out = {k: [] for k in ("dst", "dst_off", "src", "src_off", "n", "elem", "rows", "ds", "ss")}
    m, i = len(dst), 0

    def same(a: int, b: int) -> bool:
        return dst[a] == dst[b] and src[a] == src[b] and n[a] == n[b] and elem[a] == elem[b]

    while i < m:
        j = i
        if i + 1 < m and same(i, i + 1):
            dd, ds = doff[i + 1] - doff[i], soff[i + 1] - soff[i]
            j = i + 1
            while (j + 1 < m and same(i, j + 1) and doff[j + 1] - doff[j] == dd
                   and soff[j + 1] - soff[j] == ds):
                j += 1
        rows = j - i + 1
        for k, v in (("dst", dst[i]), ("dst_off", doff[i]), ("src", src[i]),
                     ("src_off", soff[i]), ("n", n[i]), ("elem", elem[i]), ("rows", rows),
                     ("ds", doff[i + 1] - doff[i] if rows > 1 else n[i]),
                     ("ss", soff[i + 1] - soff[i] if rows > 1 else n[i])):
            out[k].append(v)
        i = j + 1
    t = {k: torch.tensor(v, dtype=torch.int64) for k, v in out.items()}
    back = torch.argsort(t["dst"] * (1 << 40) + t["dst_off"])  # destination order
    t = {k: v[back] for k, v in t.items()}
    return Copies(c.dst_names, c.src_names, t["dst"], t["dst_off"], t["src"], t["src_off"],
                  t["n"], t["elem"], c.report, t["rows"], t["ds"], t["ss"])


@dataclass
class Plan:
    """Reads, one row each: ``[dst_rank, round, src_rank, dst_param, dst_off, src_param,
    src_off, row_bytes, rows, dst_row_stride, src_row_stride]`` (bytes).

    Row ``i`` of a read copies ``row_bytes`` from ``src_off + i * src_row_stride`` to
    ``dst_off + i * dst_row_stride``. With ``src_row_stride == row_bytes`` the source is
    one contiguous range (one transport read); a strided destination is a local scatter.
    """

    dst_names: list[str]
    src_names: list[str]
    rows: torch.Tensor
    meta: dict = field(default_factory=dict)

    COLS = ("dst_rank", "round", "src_rank", "dst_param", "dst_off", "src_param", "src_off",
            "row_bytes", "rows", "dst_row_stride", "src_row_stride")

    def col(self, name: str) -> torch.Tensor:
        return self.rows[:, self.COLS.index(name)]

    def save(self, path: str) -> None:
        torch.save({"dst_names": self.dst_names, "src_names": self.src_names,
                    "rows": self.rows, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str) -> "Plan":
        b = torch.load(path, weights_only=False)
        rows = b["rows"]
        if rows.shape[1] == 8:  # plans saved before multi-row reads: one row each
            nb = rows[:, 7:8]
            rows = torch.cat([rows, torch.ones_like(nb), nb, nb], 1)
        return cls(b["dst_names"], b["src_names"], rows, b["meta"])

    def nbytes(self) -> torch.Tensor:
        return self.col("row_bytes") * self.col("rows")

    def traffic(self) -> dict:
        """Bytes per source, per destination and per (source, destination) pair, and
        how many transport reads and local scatters the plan needs."""
        nb, s, d = self.nbytes(), self.col("src_rank"), self.col("dst_rank")
        rb, nr = self.col("row_bytes"), self.col("rows")
        src_contig = (nr == 1) | (self.col("src_row_stride") == rb)
        dst_contig = (nr == 1) | (self.col("dst_row_stride") == rb)
        ns, nd = int(s.max()) + 1, int(d.max()) + 1
        pair = torch.zeros(ns * nd, dtype=torch.int64).index_add_(0, s * nd + d, nb).view(ns, nd)
        r = self.col("round")
        n_rounds = int(r.max()) + 1
        per_round = torch.zeros(n_rounds * ns, dtype=torch.int64).index_add_(
            0, r * ns + s, nb).view(n_rounds, ns)
        mean = per_round.sum(1).double() / ns
        return {"per_src": pair.sum(1).tolist(), "per_dst": pair.sum(0).tolist(),
                "pair": pair.tolist(), "rounds": n_rounds, "reads": int(self.rows.shape[0]),
                # Contiguous source: one read; strided source: one read per row.
                "transport_reads": int(src_contig.sum() + nr[~src_contig].sum()),
                "strided_dst_reads": int((~dst_contig).sum()),
                # Busiest source's bytes over the mean, per round: 1.0 is perfectly spread.
                "round_src_skew_max": round(float((per_round.max(1).values / mean).max()), 3),
                "round_src_skew_mean": round(float((per_round.max(1).values / mean).mean()), 3)}


def plan_rounds(copies: Copies, n_src: int, n_dst: int, round_bytes: int = 512 << 20,
                max_piece_bytes: int = 32 << 20,
                holders: Optional[Mapping[str, list[int]]] = None) -> Plan:
    """Rounds and source assignment for ``n_dst`` identical rollout ranks.

    Every rollout rank needs all of ``copies`` (TP=1 replicas). Copies are cut into
    pieces of at most ``max_piece_bytes``, in destination order, then into rounds
    of at most ``round_bytes`` per rollout rank. Within a round, pieces go to the
    least-loaded source among those holding the piece's source tensor
    (``holders``, by source name; default every source), starting from a different
    source on each rollout rank, so the ``n_src x n_dst`` links carry about the
    same bytes.
    """
    every = list(range(n_src))
    can = [list(holders[n]) if holders is not None else every for n in copies.src_names]
    elem_l = copies.elem.tolist()
    row_b = (copies.n * copies.elem).tolist()
    nrows = copies.nrows().tolist()
    d_b = (copies.dst_off * copies.elem).tolist()
    s_b = (copies.src_off * copies.elem).tolist()
    ds_b = ((copies.dst_stride if copies.dst_stride is not None else copies.n)
            * copies.elem).tolist()
    ss_b = ((copies.src_stride if copies.src_stride is not None else copies.n)
            * copies.elem).tolist()
    # Piece: (copy, first row, rows, byte offset in row, bytes per row). Multi-row
    # copies split by whole rows; a row longer than a piece splits like a 1-D copy.
    pieces = []
    for i in range(len(copies)):
        rb, R, e = row_b[i], nrows[i], elem_l[i]
        if R > 1 and rb <= max_piece_bytes:
            per = max(1, max_piece_bytes // rb)
            for r0 in range(0, R, per):
                pieces.append((i, r0, min(per, R - r0), 0, rb))
            continue
        step = max(e, (max_piece_bytes // e) * e)
        for r in range(R):
            for off in range(0, rb, step):
                pieces.append((i, r, 1, off, min(step, rb - off)))
    rounds, cur, cur_b = [], [], 0
    for p in pieces:
        b = p[2] * p[4]
        if cur and cur_b + b > round_bytes:
            rounds.append(cur)
            cur, cur_b = [], 0
        cur.append(p)
        cur_b += b
    if cur:
        rounds.append(cur)
    dst_l, src_l = copies.dst.tolist(), copies.src.tolist()
    rows = []
    for d in range(n_dst):
        # Each rollout rank starts at a different round: under EP a round's experts
        # sit on one trainer rank, which all rollout ranks would otherwise hit at once.
        k = d * len(rounds) // n_dst
        for ri, rnd in enumerate(rounds[k:] + rounds[:k]):
            load = [0] * n_src
            # Pieces with fewer holders first, so replicated ones fill the gaps.
            for ci, r0, nr, off, nb in sorted(
                    rnd, key=lambda p: (len(can[src_l[p[0]]]), -p[2] * p[4])):
                s = min(can[src_l[ci]], key=lambda q: (load[q], (q - d) % n_src))
                load[s] += nr * nb
                dstr, sstr = (ds_b[ci], ss_b[ci]) if nr > 1 else (nb, nb)
                rows.append((d, ri, s, dst_l[ci], d_b[ci] + r0 * ds_b[ci] + off, src_l[ci],
                             s_b[ci] + r0 * ss_b[ci] + off, nb, nr, dstr, sstr))
    plan = Plan(copies.dst_names, copies.src_names, torch.tensor(rows, dtype=torch.int64),
                {"n_src": n_src, "n_dst": n_dst, "round_bytes": round_bytes,
                 "max_piece_bytes": max_piece_bytes})
    total = copies.total_bytes()
    plan.meta["model_bytes"] = total
    plan.meta["traffic"] = plan.traffic()
    plan.meta["ideal_per_src"] = total * n_dst / n_src
    return plan


def apply_plan(plan: Plan, dst_rank: int, src: Mapping[int, Mapping[str, torch.Tensor]],
               dst: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """Execute one rollout rank's reads with local copies; returns bytes written per param.

    A multi-row read is done as the transport would: the source rows (one contiguous
    range when ``src_row_stride == row_bytes``) are staged, then scattered with a
    strided copy into the destination.
    """
    rows = plan.rows[plan.col("dst_rank") == dst_rank]
    flat_dst = {n: dst[n].view(-1).view(torch.uint8) for n in plan.dst_names if n in dst}
    flat_src: dict[tuple[int, str], torch.Tensor] = {}
    written: dict[str, int] = {}
    for _, _, s, dp, do, sp, so, rb, nr, dstr, sstr in rows.tolist():
        dn, sn = plan.dst_names[dp], plan.src_names[sp]
        key = (s, sn)
        if key not in flat_src:
            flat_src[key] = src[s][sn].contiguous().view(-1).view(torch.uint8)
        fs, fd = flat_src[key], flat_dst[dn]
        if nr == 1:
            fd[do:do + rb].copy_(fs[so:so + rb])
        else:
            staged = fs.as_strided((nr, rb), (sstr, 1), fs.storage_offset() + so)
            fd.as_strided((nr, rb), (dstr, 1), fd.storage_offset() + do).copy_(staged)
        written[dn] = written.get(dn, 0) + nr * rb
    return written
