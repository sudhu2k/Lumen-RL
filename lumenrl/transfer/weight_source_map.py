"""Black-box source maps: where each checkpoint element lands in a side's parameters.

Both the trainer and the rollout engine already have a function that loads an HF
checkpoint into their own parameters. Running it on an *id checkpoint*, where every
element holds its own global index, and reading the parameters back says which
checkpoint element each parameter element holds, without reading the loader's code.
The checkpoint is then the pivot between the two sides (``weight_plan``).

BF16 holds the integers 0-256 exactly, so the id goes through one byte per pass:
pass ``k`` fills every element with ``(id >> 8k) & 255``. An extra pass of ones marks
the elements a loader writes at all (padding reads 0 in every pass, like id 0).

Ids are never stored per element. After each pass a parameter's readback is
encoded as runs: ``length`` parameter elements from ``param_offset`` holding
``ckpt_offset + ckpt_stride * j`` modulo the bits seen so far (``Runs``). The next pass rebuilds each element's low bits from the runs, adds
its new byte and re-segments. A slice copy is one run per contiguous stretch, so
memory follows the number of runs, not elements. A parameter whose runs explode
(a fine-grained permutation) keeps its partial ids raw, bounded by its own size.
Look-alike parameters (layers, experts) are checked against one decoded member
instead of decoded each (see the templates section).
"""

from __future__ import annotations

import bisect
import json
import os
import struct
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Optional

import torch

__all__ = [
    "CheckpointTensor",
    "CheckpointIndex",
    "Runs",
    "ParamMap",
    "SourceMap",
    "capture",
    "merge_rectangles",
]

_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

# Above this many runs a parameter is kept raw: a run per few elements costs more
# than the int64 ids themselves, and only permutations get there.
DEFAULT_RUN_LIMIT = 1 << 20


@dataclass(frozen=True)
class CheckpointTensor:
    """One checkpoint tensor and the global id of its first element."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    start: int
    numel: int


class CheckpointIndex:
    """The checkpoint's tensors in safetensors file order, numbered globally.

    Only the headers are read; the id checkpoint is generated, never loaded.
    """

    def __init__(self, tensors: list[CheckpointTensor]) -> None:
        self.tensors = tensors
        self.total = max((t.start + t.numel for t in tensors), default=0)
        self._by_name = {t.name: t for t in tensors}

    @classmethod
    def from_dir(cls, model_dir: str) -> "CheckpointIndex":
        files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors under {model_dir}")
        tensors, start = [], 0
        for f in files:
            with open(os.path.join(model_dir, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            header.pop("__metadata__", None)
            for name, info in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
                shape = tuple(int(s) for s in info["shape"])
                numel = 1
                for s in shape:
                    numel *= s
                tensors.append(CheckpointTensor(name, shape, info["dtype"], start, numel))
                start += numel
        return cls(tensors)

    @property
    def passes(self) -> int:
        """Byte passes needed for the largest id."""
        return max(1, ((self.total - 1).bit_length() + 7) // 8)

    def fingerprint(self) -> str:
        """Names, shapes and dtypes in order: what the ids depend on."""
        import hashlib

        h = hashlib.sha1()
        for t in self.tensors:
            h.update(f"{t.name}|{t.shape}|{t.dtype};".encode())
        return h.hexdigest()

    def id_tensor(self, name: str, k: Optional[int]) -> torch.Tensor:
        """Pass ``k`` of one tensor's ids, or all ones for ``k=None``."""
        t = self._by_name[name]
        dtype = _ST_DTYPES.get(t.dtype)
        if dtype is None:
            raise TypeError(f"{name}: id passes support {sorted(_ST_DTYPES)}, not {t.dtype}")
        if k is None:
            return torch.ones(t.shape, dtype=dtype)
        ids = torch.arange(t.start, t.start + t.numel, dtype=torch.int64)
        return ((ids >> (8 * k)) & 255).to(dtype).view(t.shape)

    def id_tensors(self, k: Optional[int]) -> Iterator[tuple[str, torch.Tensor]]:
        """The whole id checkpoint for pass ``k``, one tensor at a time."""
        for t in self.tensors:
            yield t.name, self.id_tensor(t.name, k)

    def id_state(self, k: Optional[int]) -> dict[str, torch.Tensor]:
        """The whole id checkpoint for pass ``k`` as a dict, for loaders that want one."""
        return dict(self.id_tensors(k))


@dataclass
class Runs:
    """Strided views pairing parameter elements with checkpoint ids, one per run.

    Run ``r`` covers the index grid ``shape[r]``. Its element ``(i_0, .., i_{D-1})``
    sits at ``param_offset[r] + sum_d i_d * param_strides[r, d]`` in the flat
    parameter and holds checkpoint id ``ckpt_offset[r] + sum_d i_d * ckpt_strides[r, d]``.
    ``shape`` and both strides are ``[runs, D]``. The decoder emits lines: ``D = 1``
    with ``param_strides`` 1 (``Runs.lines``).
    """

    param_offset: torch.Tensor
    ckpt_offset: torch.Tensor
    shape: torch.Tensor
    param_strides: torch.Tensor
    ckpt_strides: torch.Tensor

    _FIELDS = ("param_offset", "ckpt_offset", "shape", "param_strides", "ckpt_strides")

    @classmethod
    def lines(cls, param_offset: torch.Tensor, length: torch.Tensor,
              ckpt_offset: torch.Tensor, ckpt_stride: torch.Tensor) -> "Runs":
        """1-D runs: ``length[r]`` consecutive parameter elements, ids step ``ckpt_stride[r]``."""
        return cls(param_offset, ckpt_offset, length.view(-1, 1),
                   torch.ones_like(length).view(-1, 1), ckpt_stride.view(-1, 1))

    def __len__(self) -> int:
        return int(self.param_offset.numel())

    @property
    def ndim(self) -> int:
        return int(self.shape.shape[1]) if self.shape.dim() == 2 else 1

    def is_lines(self) -> bool:
        """1-D with contiguous parameter elements: what the decoder and the line join use."""
        return self.ndim == 1 and bool((self.param_strides == 1).all())

    @property
    def length(self) -> torch.Tensor:
        """Elements per run."""
        return self.shape.prod(1)

    def as_lines(self) -> Optional["Runs"]:
        """The same elements as 1-D runs (one per rectangle row), or None if not 1-D/2-D
        with contiguous rows on the parameter side."""
        if self.is_lines():
            return self
        if self.ndim != 2 or not bool((self.param_strides[:, 1] == 1).all()):
            return None
        rep, i = _run_offsets(self.shape[:, 0])
        return Runs.lines(self.param_offset[rep] + i * self.param_strides[rep, 0],
                          self.shape[rep, 1],
                          self.ckpt_offset[rep] + i * self.ckpt_strides[rep, 0],
                          self.ckpt_strides[rep, 1])

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {k: getattr(self, k) for k in self._FIELDS}

    @classmethod
    def from_dict(cls, d: Mapping[str, torch.Tensor]) -> "Runs":
        if "pos" in d:  # maps saved before the rename: 1-D (pos, length, id0, step)
            return cls.lines(d["pos"], d["length"], d["id0"], d["step"])
        return cls(*(d[k] for k in cls._FIELDS))

    def expand(self, numel: int) -> torch.Tensor:
        """Per-element ids, -1 where nothing was written. For tests and fallbacks."""
        out = torch.full((numel,), -1, dtype=torch.int64)
        if len(self):
            rep, j = _run_offsets(self.length)
            pos, ids = self.param_offset[rep].clone(), self.ckpt_offset[rep].clone()
            for d in reversed(range(self.ndim)):
                n = self.shape[rep, d]
                i = j % n
                j = j // n
                pos += i * self.param_strides[rep, d]
                ids += i * self.ckpt_strides[rep, d]
            out[pos] = ids
        return out


@dataclass
class ParamMap:
    """One parameter's map at one capture point."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    runs: Optional[Runs] = None
    raw: Optional[torch.Tensor] = None
    valid: bool = True

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def ids(self) -> torch.Tensor:
        """Per-element ids, -1 where nothing was written."""
        if self.raw is not None:
            return self.raw
        assert self.runs is not None
        return self.runs.expand(self.numel)


@dataclass
class SourceMap:
    """Every parameter of one side, at one capture point."""

    params: dict[str, ParamMap]
    meta: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        blob = {"meta": self.meta, "params": {}}
        for n, p in self.params.items():
            blob["params"][n] = {
                "shape": list(p.shape), "dtype": p.dtype, "valid": p.valid,
                "runs": p.runs.to_dict() if p.runs is not None else None,
                "raw": p.raw,
            }
        torch.save(blob, path)

    @classmethod
    def load(cls, path: str) -> "SourceMap":
        blob = torch.load(path, weights_only=False)
        params = {}
        for n, d in blob["params"].items():
            params[n] = ParamMap(
                n, tuple(d["shape"]), d["dtype"],
                runs=Runs.from_dict(d["runs"]) if d["runs"] is not None else None,
                raw=d["raw"], valid=d["valid"],
            )
        return cls(params, blob["meta"])

    def stats(self) -> dict:
        valid = [p for p in self.params.values() if p.valid]
        return {
            "params": len(self.params),
            "invalid": len(self.params) - len(valid),
            "raw": sum(p.raw is not None for p in valid),
            "runs": sum(len(p.runs) for p in valid if p.runs is not None),
            "elements": sum(p.numel for p in self.params.values()),
        }


# ---------------------------------------------------------------- decoding

def _run_offsets(length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For runs of ``length``: each element's run index and offset within its run."""
    rep = torch.repeat_interleave(torch.arange(length.numel()), length)
    first = torch.cumsum(length, 0) - length
    return rep, torch.arange(rep.numel()) - first[rep]


def _mul_mod(s: torch.Tensor, j: torch.Tensor, bits: int) -> torch.Tensor:
    """``(s * j) mod 2**bits`` for ``0 <= s < 2**bits``, ``0 <= j < 2**31``, without int64 overflow."""
    if bits <= 32:
        return torch.remainder(s * j, 1 << bits)
    lo = s & ((1 << 20) - 1)
    hi = s >> 20
    return torch.remainder((torch.remainder(hi * j, 1 << (bits - 20)) << 20) + lo * j, 1 << bits)


def _blocks(written: Optional[torch.Tensor], n: int) -> list[tuple[int, int]]:
    """Maximal ``[start, end)`` stretches of written elements."""
    if written is None or bool(written.all()):
        return [(0, n)]
    w = written.to(torch.int8)
    edges = torch.nonzero(w[1:] != w[:-1]).squeeze(1) + 1
    bounds = [0, *edges.tolist(), n]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if bool(written[a])]


def _segment(x: torch.Tensor, written: Optional[torch.Tensor], modulus: int,
             limit: int) -> Optional[Runs]:
    """Greedy maximal runs of constant step (mod ``modulus``) over written elements."""
    n = x.numel()
    blocks = _blocks(written, n)
    if n > 1:
        dx = torch.remainder(x[1:] - x[:-1], modulus)
        if written is not None:
            dx[~(written[1:] & written[:-1])] = -1
        change = torch.ones(n - 1, dtype=torch.bool)
        change[1:] = dx[1:] != dx[:-1]
        starts_t = torch.nonzero(change).squeeze(1)
        if starts_t.numel() > 2 * limit:
            return None
        starts = starts_t.tolist()
        ends = [s - 1 for s in starts[1:]] + [n - 2]
        vals = dx[starts_t].tolist()
    offset, length, stride = [], [], []
    for b0, b1 in blocks:
        p = b0
        while p < b1:
            if p == b1 - 1:
                offset.append(p), length.append(1), stride.append(0)
                break
            r = bisect.bisect_right(starts, p) - 1
            e = min(ends[r], b1 - 2)
            offset.append(p), length.append(e + 2 - p), stride.append(vals[r])
            p = e + 2
            if len(offset) > limit:
                return None
    offset_t = torch.tensor(offset, dtype=torch.int64)
    return Runs.lines(offset_t, torch.tensor(length, dtype=torch.int64),
                      torch.remainder(x[offset_t], modulus),
                      torch.tensor(stride, dtype=torch.int64))


def merge_rectangles(runs: Runs) -> Runs:
    """Merge consecutive lines of equal length and step whose parameter and checkpoint
    offsets advance by constant amounts into 2-D runs, one row per original line.

    Lossless (``as_lines`` gives the input back). A row-parallel TP shard, half of every
    checkpoint row, becomes one run instead of one per row. Returns ``runs`` unchanged
    when nothing merges.
    """
    if not runs.is_lines() or len(runs) < 2:
        return runs
    po, co = runs.param_offset.tolist(), runs.ckpt_offset.tolist()
    ln, st = runs.length.tolist(), runs.ckpt_strides[:, 0].tolist()
    out = {k: [] for k in ("po", "co", "rows", "len", "ps", "cs", "st")}
    n, i = len(po), 0
    while i < n:
        j = i
        if i + 1 < n and ln[i + 1] == ln[i] and st[i + 1] == st[i] and po[i + 1] - po[i] >= ln[i]:
            dp, dc = po[i + 1] - po[i], co[i + 1] - co[i]
            j = i + 1
            while (j + 1 < n and ln[j + 1] == ln[i] and st[j + 1] == st[i]
                   and po[j + 1] - po[j] == dp and co[j + 1] - co[j] == dc):
                j += 1
        rows = j - i + 1
        out["po"].append(po[i]), out["co"].append(co[i]), out["rows"].append(rows)
        out["len"].append(ln[i]), out["st"].append(st[i])
        out["ps"].append(po[i + 1] - po[i] if rows > 1 else ln[i])
        out["cs"].append(co[i + 1] - co[i] if rows > 1 else 0)
        i = j + 1
    if len(out["po"]) == n:
        return runs
    t = {k: torch.tensor(v, dtype=torch.int64) for k, v in out.items()}
    return Runs(t["po"], t["co"], torch.stack([t["rows"], t["len"]], 1),
                torch.stack([t["ps"], torch.ones_like(t["ps"])], 1),
                torch.stack([t["cs"], t["st"]], 1))


class _ParamDecoder:
    """Accumulates one parameter's passes into runs (or raw ids)."""

    def __init__(self, name: str, shape: tuple[int, ...], dtype: str, limit: int) -> None:
        self.map = ParamMap(name, shape, dtype)
        self.written: Optional[torch.Tensor] = None
        self.limit = limit

    def ones(self, v: torch.Tensor) -> None:
        if not bool(((v == 0) | (v == 1)).all()):
            self.map.valid = False
            return
        self.written = None if bool((v == 1).all()) else (v == 1)

    def observe(self, k: int, b: torch.Tensor) -> None:
        m = self.map
        if not m.valid:
            return
        b = b.to(torch.int64) << (8 * k)
        if m.raw is not None:
            m.raw |= b
            return
        x = b if k == 0 else self._low_bits(k) | b
        modulus = 1 << (8 * (k + 1))
        runs = _segment(x, self.written, modulus, self.limit)
        if runs is None:
            m.raw, m.runs = x, None
            if self.written is not None:
                m.raw[~self.written] = -1
        else:
            m.runs = runs

    def _low_bits(self, k: int) -> torch.Tensor:
        """Each element's id modulo 256**k, rebuilt from the current runs."""
        runs = self.map.runs
        assert runs is not None
        out = torch.zeros(self.map.numel, dtype=torch.int64)
        if len(runs):
            assert runs.is_lines()
            rep, j = _run_offsets(runs.length)
            bits = 8 * k
            prod = _mul_mod(runs.ckpt_strides[rep, 0], j, bits)
            out[runs.param_offset[rep] + j] = torch.remainder(runs.ckpt_offset[rep] + prod,
                                                              1 << bits)
        return out

    def finish(self, bits: int) -> ParamMap:
        m = self.map
        if m.runs is not None and len(m.runs):
            half = 1 << (bits - 1)
            s = m.runs.ckpt_strides
            m.runs.ckpt_strides = torch.where(s >= half, s - (1 << bits), s)
            m.runs = merge_rectangles(m.runs)
        return m


_ROUND_TRIP = (torch.bfloat16, torch.float16, torch.float32, torch.float64)


def _as_bytes(t: torch.Tensor, ones: bool) -> Optional[torch.Tensor]:
    """Flat uint8 values if every element is a valid id byte (or 0/1), else None.

    A round trip through uint8 can only reproduce a value that is a whole number in
    0..255, whatever the cast does with NaN, negatives or values above 255, so it
    rejects exactly what a range-and-integer check rejects.
    """
    f = t.detach().reshape(-1)
    if f.dtype == torch.uint8:
        u = f
    elif f.dtype in _ROUND_TRIP:
        u = f.to(torch.uint8)
        if not torch.equal(u.to(f.dtype), f):
            return None
    else:
        g = f.to(torch.float32)
        if not bool(((g == g.round()) & (g >= 0) & (g <= 255)).all()):
            return None
        u = g.to(torch.uint8)
    if ones and bool((u > 1).any()):
        return None
    return u


# ---------------------------------------------------------------- templates
#
# Parameters that go through the same code (the layers of a dense model, the experts
# of a MoE) have the same run geometry and differ only in where their ids start.
# The first parameter of each group (same shape, dtype and name with digits
# removed) is decoded; the others only record each pass compactly. After the last
# pass each is predicted from a decoded member, with run offsets read from its own
# bytes at the run starts, and the prediction is checked against every pass. A
# mismatch replays the recorded passes through the decoder, so a wrong prediction
# costs time, never correctness.

@dataclass
class _Packed:
    """One pass's bytes for one parameter, run-length coded on their differences mod 256.

    Packing is deterministic, so two byte arrays are equal exactly when their packed
    forms are. Step-1 runs give a constant difference except at carries; anything
    with more changes than that is kept raw.
    """

    n: int
    starts: Optional[torch.Tensor] = None
    vals: Optional[torch.Tensor] = None
    raw: Optional[torch.Tensor] = None

    def same(self, other: "_Packed") -> bool:
        if self.n != other.n or (self.raw is None) != (other.raw is None):
            return False
        if self.raw is not None:
            return torch.equal(self.raw, other.raw)
        return torch.equal(self.starts, other.starts) and torch.equal(self.vals, other.vals)

    def bytes(self) -> torch.Tensor:
        if self.raw is not None:
            return self.raw
        lens = torch.diff(self.starts, append=torch.tensor([self.n]))
        return torch.cumsum(torch.repeat_interleave(self.vals, lens), 0, dtype=torch.uint8)

    def at(self, pos: torch.Tensor) -> torch.Tensor:
        """The bytes at ``pos``, as int64."""
        if self.raw is not None:
            return self.raw[pos].to(torch.int64)
        lens = torch.diff(self.starts, append=torch.tensor([self.n]))
        v = self.vals.to(torch.int64)
        before = torch.cumsum(v * lens, 0) - v * lens
        seg = torch.searchsorted(self.starts, pos, right=True) - 1
        return (before[seg] + v[seg] * (pos - self.starts[seg] + 1)) & 255


def _pack(u: torch.Tensor) -> _Packed:
    n = u.numel()
    d = u.clone()
    d[1:] -= u[:-1]
    change = torch.nonzero(d[1:] != d[:-1]).squeeze(1) + 1
    if 9 * (change.numel() + 1) > n:
        return _Packed(n, raw=u.contiguous().clone())
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), change])
    return _Packed(n, starts=starts, vals=d[starts])


def _split_at_tensors(lines: Runs, bounds: torch.Tensor) -> Runs:
    """``lines`` cut wherever their ids cross into another checkpoint tensor.

    Look-alikes keep the geometry within each checkpoint tensor, but the tensors
    themselves can sit in a different order in the file (and so coalesce
    differently), e.g. an expert's gate and up adjacent in one layer and apart in
    the next.
    """
    co, n, st = lines.ckpt_offset, lines.length, lines.ckpt_strides[:, 0]
    last = co + (n - 1) * st
    lo = torch.searchsorted(bounds, torch.minimum(co, last), right=True)
    hi = torch.searchsorted(bounds, torch.maximum(co, last), right=True)
    cross = set(torch.nonzero(lo != hi).squeeze(1).tolist())
    if not cross:
        return lines
    po_l, n_l, co_l, st_l = (lines.param_offset.tolist(), n.tolist(), co.tolist(), st.tolist())
    out_po, out_n, out_co, out_st = [], [], [], []
    for r in range(len(po_l)):
        cuts = [0, n_l[r]]
        if r in cross:
            s = st_l[r]
            for b in bounds[int(lo[r]):int(hi[r])].tolist():
                cuts.append(-((co_l[r] - b) // s) if s > 0 else (co_l[r] - b) // -s + 1)
            cuts = sorted({c for c in cuts if 0 <= c <= n_l[r]})
        for a, b in zip(cuts[:-1], cuts[1:]):
            out_po.append(po_l[r] + a), out_n.append(b - a)
            out_co.append(co_l[r] + a * st_l[r]), out_st.append(st_l[r] if b - a > 1 else 0)
    t = lambda v: torch.tensor(v, dtype=torch.int64)  # noqa: E731
    return Runs.lines(t(out_po), t(out_n), t(out_co), t(out_st))


def _join_lines(lines: Runs) -> Runs:
    """Join consecutive lines that continue each other (adjacent elements, ids on one step)."""
    po, n, co, st = (lines.param_offset.tolist(), lines.length.tolist(),
                     lines.ckpt_offset.tolist(), lines.ckpt_strides[:, 0].tolist())
    out = [[po[0], n[0], co[0], st[0]]] if po else []
    for r in range(1, len(po)):
        p = out[-1]
        if po[r] == p[0] + p[1]:
            s = p[3] if p[1] > 1 else co[r] - p[2]
            if (n[r] == 1 or st[r] == s) and co[r] == p[2] + p[1] * s:
                p[1] += n[r]
                p[3] = s
                continue
        out.append([po[r], n[r], co[r], st[r]])
    if len(out) == len(po):
        return lines
    t = torch.tensor(out, dtype=torch.int64).view(-1, 4)
    return Runs.lines(t[:, 0], t[:, 1], t[:, 2], t[:, 3])


def _template_key(name: str, shape: tuple[int, ...], dtype: str) -> tuple:
    import re

    return (shape, dtype, re.sub(r"\d+", "#", name))


def _run_ends(runs: Runs) -> tuple[torch.Tensor, torch.Tensor]:
    """Each run's last parameter position and last checkpoint id."""
    span = runs.shape - 1
    return (runs.param_offset + (span * runs.param_strides).sum(1),
            runs.ckpt_offset + (span * runs.ckpt_strides).sum(1))


def _fill(runs: Runs, numel: int) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Per-element ids (0 where nothing is written) and the written mask (None: all)."""
    full = int(runs.length.sum()) == numel
    if len(runs) > 4096:
        ids = runs.expand(numel)
        written = None if full else ids >= 0
        return ids.clamp_(min=0), written
    ids = torch.zeros(numel, dtype=torch.int64)
    written = None if full else torch.zeros(numel, dtype=torch.bool)
    po, co = runs.param_offset.tolist(), runs.ckpt_offset.tolist()
    shape, ps, cs = runs.shape.tolist(), runs.param_strides.tolist(), runs.ckpt_strides.tolist()
    for r in range(len(po)):
        view = ids.as_strided(shape[r], ps[r], po[r])
        n, st = shape[r][-1], cs[r][-1]
        if n == 1 or st == 0:
            row = torch.full((n,), co[r], dtype=torch.int64)
        else:
            row = torch.arange(co[r], co[r] + n * st, st, dtype=torch.int64)
        if len(shape[r]) == 1:
            view.copy_(row)
        else:
            v = row
            for d in reversed(range(len(shape[r]) - 1)):
                v = v.unsqueeze(0) + (torch.arange(shape[r][d], dtype=torch.int64)
                                      * cs[r][d]).view(-1, *([1] * v.dim()))
            view.copy_(v)
        if written is not None:
            written.as_strided(shape[r], ps[r], po[r]).fill_(True)
    return ids, written


def _predict(template: Runs, obs: dict, numel: int, total: int) -> Optional[Runs]:
    """``template``'s geometry with offsets read from ``obs``, if it reproduces every pass."""
    passes = [k for k in obs if k is not None]

    def ids_at(pos: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(pos)
        for k in passes:
            out |= obs[k].at(pos) << (8 * k)
        return out

    runs = Runs(template.param_offset, ids_at(template.param_offset), template.shape,
                template.param_strides, template.ckpt_strides)
    span = (runs.shape - 1) * runs.ckpt_strides
    lo = runs.ckpt_offset + span.clamp(max=0).sum(1)
    hi = runs.ckpt_offset + span.clamp(min=0).sum(1)
    if bool((lo < 0).any()) or bool((hi >= total).any()):
        return None
    last_pos, last_id = _run_ends(runs)
    if not torch.equal(ids_at(last_pos), last_id):
        return None
    ids, written = _fill(runs, numel)
    ones = torch.ones(numel, dtype=torch.uint8) if written is None else written.to(torch.uint8)
    if not _pack(ones).same(obs[None]):
        return None
    b = ids.view(torch.uint8)
    for k in passes:
        if not _pack(b[k::8]).same(obs[k]):
            return None
    return runs


class _Deferred:
    """A parameter that only records its passes until a template can be checked."""

    def __init__(self, name: str, shape: tuple[int, ...], dtype: str, limit: int) -> None:
        self.map = ParamMap(name, shape, dtype)
        self.obs: dict[Optional[int], _Packed] = {}
        self.limit = limit

    def record(self, k: Optional[int], b: torch.Tensor) -> None:
        if self.map.valid:
            self.obs[k] = _pack(b)

    def replay(self, passes: list[Optional[int]], bits: int) -> ParamMap:
        dec = _ParamDecoder(self.map.name, self.map.shape, self.map.dtype, self.limit)
        for k in passes:
            b = self.obs.pop(k).bytes()
            dec.ones(b) if k is None else dec.observe(k, b)
        return dec.finish(bits)


ReadBack = Mapping[str, Mapping[str, torch.Tensor]]


def capture(
    index: CheckpointIndex,
    run_pass: Callable[[Optional[int]], ReadBack],
    decode_stages: tuple[str, ...] = ("loader",),
    compare: Optional[tuple[str, str]] = None,
    limit: int = DEFAULT_RUN_LIMIT,
    log: Callable[[str], None] = print,
    templates: bool = True,
) -> tuple[dict[str, SourceMap], dict[str, str]]:
    """Run every id pass through a side's loader and decode its parameters.

    Args:
        index: The checkpoint.
        run_pass: Loads pass ``k`` (``None`` = the ones pass) with the side's own
            code and returns ``{stage: {param_name: tensor}}``, e.g. the vLLM
            parameters after ``load_weights`` (``"loader"``) and after
            post-processing (``"final"``).
        decode_stages: Stages to decode into maps.
        compare: ``(a, b)``: classify each parameter by comparing stage ``b``'s
            bytes with stage ``a``'s on every pass.
        limit: Runs per parameter above which its ids are kept raw.
        templates: Decode one parameter per group of look-alikes and check the
            others against it (see ``_Packed``) instead of decoding each.

    Returns:
        ``({stage: SourceMap}, {param: class})``, class one of ``identity``
        (b equals a), ``changed`` (b holds valid ids but differs: a permutation
        candidate) or ``computed`` (b's values are not ids).
    """
    import time

    decoders: dict[str, dict[str, "_ParamDecoder | _Deferred"]] = {s: {} for s in decode_stages}
    classes: dict[str, str] = {}
    passes: list[Optional[int]] = [None, *range(index.passes)]
    for k in passes:
        t0 = time.time()
        readback = run_pass(k)
        for stage in decode_stages:
            decs = decoders[stage]
            seen: set[tuple] = set()
            for name, t in readback[stage].items():
                dec = decs.get(name)
                if dec is None:
                    shape, dtype = tuple(t.shape), str(t.dtype).removeprefix("torch.")
                    key = _template_key(name, shape, dtype)
                    cls = _Deferred if templates and key in seen else _ParamDecoder
                    seen.add(key)
                    dec = decs[name] = cls(name, shape, dtype, limit)
                b = _as_bytes(t, ones=k is None)
                if b is None:
                    dec.map.valid = False
                elif isinstance(dec, _Deferred):
                    dec.record(k, b)
                elif k is None:
                    dec.ones(b)
                else:
                    dec.observe(k, b)
        if compare is not None:
            a, bstage = compare
            for name, tb in readback[bstage].items():
                ta = readback[a].get(name)
                prev = classes.get(name, "identity")
                if prev == "computed":
                    continue
                same = (ta is not None and ta.shape == tb.shape and ta.dtype == tb.dtype
                        and torch.equal(ta, tb))
                a_dec = decoders.get(a, {}).get(name)
                if same and a_dec is not None and a_dec.map.valid:
                    # Byte-equal to a stage that just passed the check: valid too.
                    classes[name] = prev
                elif _as_bytes(tb, ones=k is None) is None:
                    classes[name] = "computed"
                elif not same:
                    classes[name] = "changed"
                else:
                    classes[name] = prev
        del readback
        log(f"pass {'ones' if k is None else k}: {time.time() - t0:.1f}s")
    bits = 8 * index.passes
    bounds = torch.tensor([t.start for t in index.tensors], dtype=torch.int64)
    maps = {}
    for stage, decs in decoders.items():
        t0 = time.time()
        known: dict[tuple, list[Runs]] = {}
        params: dict[str, ParamMap] = {}
        n_pred, replayed = 0, []
        for name, dec in decs.items():
            m = dec.map
            key = _template_key(name, m.shape, m.dtype)
            cands = known.setdefault(key, [])
            decoded = False
            if isinstance(dec, _Deferred) and m.valid:
                for i, t in enumerate(cands):
                    runs = _predict(t, dec.obs, m.numel, index.total)
                    if runs is not None:
                        m.runs = merge_rectangles(_join_lines(runs))
                        cands.insert(0, cands.pop(i))
                        n_pred += 1
                        break
                else:
                    m, decoded = dec.replay(passes, bits), True
                    replayed.append(name)
                dec.obs.clear()
            elif isinstance(dec, _ParamDecoder):
                m, decoded = dec.finish(bits), True
            if decoded and m.valid and m.runs is not None and len(cands) < 8:
                lines = m.runs.as_lines()
                if lines is not None:
                    cands.append(_split_at_tensors(lines, bounds))
            params[name] = m
        meta = {"checkpoint": index.fingerprint(), "elements": index.total,
                "passes": index.passes, "stage": stage,
                "templates": {"predicted": n_pred, "replayed": len(replayed),
                              "replayed_names": replayed[:20]}}
        maps[stage] = SourceMap(params, meta)
        log(f"{stage} finish: {time.time() - t0:.1f}s, {n_pred} predicted from a template, "
            f"{len(replayed)} replayed {replayed[:4]}")
    return maps, classes
