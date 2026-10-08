"""Black-box source maps: where each checkpoint element lands in a side's parameters.

Both the trainer and the rollout engine already have a function that loads an HF
checkpoint into their own parameters. Running it on an *id checkpoint*, where every
element holds its own global index, and reading the parameters back says which
checkpoint element each parameter element holds, without reading the loader's code.
The checkpoint is then the pivot between the two sides (``weight_plan``).

BF16 holds the integers 0-256 exactly, so the id goes through one byte per pass:
pass ``k`` fills every element with ``(id >> 8k) & 255``. An extra pass of ones marks
the elements a loader writes at all (padding reads 0 in every pass, like id 0).

Ids are never stored per element during the passes. Each parameter records its
bytes compactly (``_Packed``). After the last pass the bytes are assembled into
ids once and segmented into runs (``Runs``): ``length`` parameter elements from
``param_offset`` holding ``ckpt_offset + ckpt_stride * j``. A slice copy is one run
per contiguous stretch, so the saved map follows the number of runs, not elements.
A parameter whose runs explode (a fine-grained permutation) keeps its ids raw,
bounded by its own size.
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
    """Records one parameter's passes, then segments the assembled ids once."""

    def __init__(self, name: str, shape: tuple[int, ...], dtype: str, limit: int) -> None:
        self.map = ParamMap(name, shape, dtype)
        self.obs: dict[Optional[int], _Packed] = {}
        self.limit = limit

    def record(self, k: Optional[int], b: torch.Tensor) -> None:
        if self.map.valid:
            self.obs[k] = _pack(b)

    def finish(self, passes: list[Optional[int]], bits: int) -> ParamMap:
        m = self.map
        if m.valid and self.obs:
            ones = self.obs[None]
            written = None if ones.fill == 1 else (ones.bytes() == 1)
            ids = torch.zeros(ones.n, dtype=torch.int64)
            # Byte k of an int64 id occupies lane k. This host is little-endian.
            assert torch.tensor([1], dtype=torch.int64).view(torch.uint8)[0] == 1
            view = ids.view(torch.uint8)
            for k in passes:
                if k is None:
                    continue
                p = self.obs[k]
                if p.fill is None:
                    view[k::8] = p.bytes()
                elif p.fill != 0:
                    view[k::8] = p.fill
            runs = _segment(ids, written, 1 << bits, self.limit)
            if runs is None:
                if written is not None:
                    ids[~written] = -1
                m.raw = ids
            else:
                half = 1 << (bits - 1)
                s = runs.ckpt_strides
                runs.ckpt_strides = torch.where(s >= half, s - (1 << bits), s)
                m.runs = merge_rectangles(runs)
        self.obs.clear()
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


# ---------------------------------------------------------------- recorded passes
#
# Each pass is stored until the last one, run-length coded on its byte differences
# (a slice copy is a few bytes per carry). ``_ParamDecoder.finish`` assembles the
# ids and segments once.

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
    # Set when every byte has this value: the ones pass is usually all 1 and the
    # high id bytes all 0, and finish uses them without expanding n bytes.
    fill: Optional[int] = None

    def same(self, other: "_Packed") -> bool:
        if self.n != other.n or (self.raw is None) != (other.raw is None):
            return False
        if self.raw is not None:
            return torch.equal(self.raw, other.raw)
        return torch.equal(self.starts, other.starts) and torch.equal(self.vals, other.vals)

    def bytes(self) -> torch.Tensor:
        if self.raw is not None:
            return self.raw
        d = torch.zeros(self.n, dtype=torch.uint8)
        v = self.vals
        d[self.starts] = v - torch.cat([v.new_zeros(1), v[:-1]])
        return _cumsum_u8(_cumsum_u8(d))

    def at(self, pos: torch.Tensor) -> torch.Tensor:
        """The bytes at ``pos``, as int64."""
        if self.raw is not None:
            return self.raw[pos].to(torch.int64)
        lens = torch.diff(self.starts, append=torch.tensor([self.n]))
        v = self.vals.to(torch.int64)
        before = torch.cumsum(v * lens, 0) - v * lens
        seg = torch.searchsorted(self.starts, pos, right=True) - 1
        return (before[seg] + v[seg] * (pos - self.starts[seg] + 1)) & 255


def _cumsum_u8(d: torch.Tensor, chunks: int = 1024) -> torch.Tensor:
    """``cumsum`` mod 256. A 1-D cumsum runs on one thread; scanning rows of a 2-D
    view runs them in parallel, and each row then adds the total of the rows before it."""
    n = d.numel()
    m = n - n % chunks
    out = torch.empty_like(d)
    carry = 0
    if m:
        h = torch.cumsum(d[:m].view(chunks, -1), 1, dtype=torch.uint8)
        totals = torch.cumsum(h[:, -1], 0, dtype=torch.uint8)
        h[1:] += totals[:-1, None]
        out[:m] = h.view(-1)
        carry = totals[-1]
    if m < n:
        out[m:] = torch.cumsum(d[m:], 0, dtype=torch.uint8) + carry
    return out


def _pack(u: torch.Tensor) -> _Packed:
    n = u.numel()
    fill = int(u[0]) if n and bool((u == u[0]).all()) else None
    d = u.clone()
    d[1:] -= u[:-1]
    change = torch.nonzero(d[1:] != d[:-1]).squeeze(1) + 1
    if 9 * (change.numel() + 1) > n:
        return _Packed(n, raw=u.contiguous().clone(), fill=fill)
    starts = torch.cat([torch.zeros(1, dtype=torch.int64), change])
    return _Packed(n, starts=starts, vals=d[starts], fill=fill)


ReadBack = Mapping[str, Mapping[str, torch.Tensor]]


def capture(
    index: CheckpointIndex,
    run_pass: Callable[[Optional[int]], ReadBack],
    decode_stages: tuple[str, ...] = ("loader",),
    compare: Optional[tuple[str, str]] = None,
    limit: int = DEFAULT_RUN_LIMIT,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, SourceMap], dict[str, str]]:
    """Run every id pass through a side's loader and decode its parameters.

    Each pass is only recorded. After the last pass every parameter's bytes are
    assembled into ids and segmented once.

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

    Returns:
        ``({stage: SourceMap}, {param: class})``, class one of ``identity``
        (b equals a), ``changed`` (b holds valid ids but differs: a permutation
        candidate) or ``computed`` (b's values are not ids).
    """
    import time

    decoders: dict[str, dict[str, _ParamDecoder]] = {s: {} for s in decode_stages}
    classes: dict[str, str] = {}
    passes: list[Optional[int]] = [None, *range(index.passes)]
    for k in passes:
        t0 = time.time()
        readback = run_pass(k)
        for stage in decode_stages:
            decs = decoders[stage]
            for name, t in readback[stage].items():
                dec = decs.get(name)
                if dec is None:
                    dec = decs[name] = _ParamDecoder(
                        name, tuple(t.shape), str(t.dtype).removeprefix("torch."), limit)
                b = _as_bytes(t, ones=k is None)
                if b is None:
                    dec.map.valid = False
                else:
                    dec.record(k, b)
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
    maps = {}
    for stage, decs in decoders.items():
        t0 = time.time()
        params = {n: d.finish(passes, bits) for n, d in decs.items()}
        meta = {"checkpoint": index.fingerprint(), "elements": index.total,
                "passes": index.passes, "stage": stage}
        maps[stage] = SourceMap(params, meta)
        log(f"{stage} finish: {time.time() - t0:.1f}s")
    return maps, classes
