"""MORI weight sync pieces that need no transport: reads, bounds, hashes, export wire."""

import pytest
import torch

from lumenrl.transfer.mori_weight_transfer import (SourceExport, check_bounds, compare_hashes,
                                                   read_entries, read_hashes)
from lumenrl.transfer.weight_plan import Plan, apply_plan, correspond, plan_rounds
from lumenrl.transfer.weight_source_map import CheckpointIndex, CheckpointTensor, capture

_SHAPES = {"a": (64, 32), "b": (16, 32)}


def _plan(n_src=2, n_dst=2) -> Plan:
    tensors, off = [], 0
    for name, shape in _SHAPES.items():
        tensors.append(CheckpointTensor(name, shape, "BF16", off, shape[0] * shape[1]))
        off += shape[0] * shape[1]
    index = CheckpointIndex(tensors)
    tmap, _ = capture(index, lambda k: {"loader": dict(index.id_state(k))}, log=lambda s: None)
    # The rollout side fuses both tensors into one.
    vmap, _ = capture(index, lambda k: {"loader": {"w": torch.cat(
        [index.id_state(k)["a"], index.id_state(k)["b"]], 0)}}, log=lambda s: None)
    return plan_rounds(correspond(tmap["loader"], vmap["loader"]), n_src=n_src, n_dst=n_dst,
                       round_bytes=2048, max_piece_bytes=512)


def _tensors():
    torch.manual_seed(0)
    src = {n: torch.randn(s).to(torch.bfloat16) for n, s in _SHAPES.items()}
    dst = {"w": torch.cat([src["a"], src["b"]], 0)}
    return src, dst


def _flat(t):
    return t.view(-1).view(torch.uint8)


def test_read_entries_cover_every_byte_once():
    plan = _plan()
    for d in range(2):
        rows = plan.rows[plan.col("dst_rank") == d]
        seen = torch.zeros(sum(s[0] * s[1] for s in _SHAPES.values()) * 2, dtype=torch.int32)
        for _, s, dp, do, sp, so, n in read_entries(rows):
            seen[do:do + n] += 1
        assert bool((seen == 1).all())


def test_read_entries_split_strided_rows():
    # rows=3 with a destination stride: one read per row.
    rows = torch.tensor([[0, 0, 1, 0, 100, 0, 8, 16, 3, 32, 16]])
    out = read_entries(rows)
    assert [(e[3], e[5], e[6]) for e in out] == [(100, 8, 16), (132, 24, 16), (164, 40, 16)]
    rows = torch.tensor([[0, 0, 1, 0, 100, 0, 8, 16, 3, 16, 16]])
    assert [(e[3], e[5], e[6]) for e in read_entries(rows)] == [(100, 8, 48)]


def test_check_bounds_names_the_bad_read():
    plan = _plan()
    rows = plan.rows[plan.col("dst_rank") == 0]
    dst = {0: (64 + 16) * 32 * 2}
    src = {(s, q): _SHAPES[plan.src_names[q]][0] * _SHAPES[plan.src_names[q]][1] * 2
           for s in range(2) for q in range(len(plan.src_names))}
    check_bounds(rows, dst, src, plan)
    with pytest.raises(ValueError, match="out of bounds"):
        check_bounds(rows, {0: dst[0] - 2}, src, plan)
    small = dict(src)
    small[(0, 0)] -= 2
    small[(1, 0)] -= 2
    with pytest.raises(ValueError, match="out of bounds"):
        check_bounds(rows, dst, small, plan)


def test_hashes_match_after_apply_and_catch_one_flipped_byte():
    plan = _plan()
    src, _ = _tensors()
    dst = {"w": torch.zeros(80, 32, dtype=torch.bfloat16)}
    apply_plan(plan, 0, {0: src, 1: src}, dst)
    src_flat = {q: _flat(src[n]) for q, n in enumerate(plan.src_names)}
    per_rank = []
    for s in range(2):
        idx = torch.nonzero((plan.col("dst_rank") == 0) & (plan.col("src_rank") == s)).flatten()
        per_rank.append({0: (idx, read_hashes(plan.rows[idx], src_flat, 5, 6, 10))})
    rows = plan.rows[plan.col("dst_rank") == 0]
    dst_flat = {0: _flat(dst["w"])}
    assert compare_hashes(plan, 0, read_hashes(rows, dst_flat, 3, 4, 9), per_rank) == []
    _flat(dst["w"])[1000] ^= 1
    bad = compare_hashes(plan, 0, read_hashes(rows, dst_flat, 3, 4, 9), per_rank)
    assert len(bad) == 1 and bad[0].startswith("w@")


def test_hash_sees_swapped_words():
    v = torch.arange(64, dtype=torch.uint8)
    w = v.clone()
    w[[0, 1, 2, 3]] = w[[2, 3, 0, 1]]
    rows = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 64, 1, 64, 64]])
    assert read_hashes(rows, {0: v}, 3, 4, 9) != read_hashes(rows, {0: w}, 3, 4, 9)


def test_source_export_wire_round_trip():
    e = SourceExport(3, b"eng", [b"m0", b"m1"], {7: (1, 4096, 128), 2: (0, 0, 64)})
    w = e.to_wire()
    assert all(isinstance(x, list) for x in w["params"])
    assert SourceExport.from_wire(w) == e
