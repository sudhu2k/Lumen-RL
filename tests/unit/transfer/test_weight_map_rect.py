"""Row-parallel shards: rectangle runs and multi-row reads instead of one copy per row."""

import torch

from lumenrl.transfer.weight_plan import apply_plan, correspond, plan_rounds
from lumenrl.transfer.weight_source_map import (
    CheckpointIndex,
    CheckpointTensor,
    capture,
    merge_rectangles,
    Runs,
)

R, C = 64, 48


def _index() -> CheckpointIndex:
    return CheckpointIndex([CheckpointTensor("norm", (8,), "BF16", 0, 8),
                            CheckpointTensor("w", (R, C), "BF16", 8, R * C)])


def _trainer(hf):
    # Row-parallel TP=2: each rank holds half of every row, contiguous.
    return {"w.tp0": hf["w"][:, :C // 2].contiguous(), "w.tp1": hf["w"][:, C // 2:].contiguous(),
            "norm": hf["norm"].clone()}


def _rollout(hf):
    return {"w": hf["w"].clone(), "norm": hf["norm"].clone()}


def test_merge_rectangles_is_lossless():
    lines = Runs.lines(torch.tensor([0, 24, 48, 72, 100]), torch.tensor([24, 24, 24, 24, 3]),
                       torch.tensor([8, 56, 104, 152, 0]), torch.tensor([1, 1, 1, 1, 1]))
    rect = merge_rectangles(lines)
    assert len(rect) == 2 and rect.ndim == 2
    assert rect.shape.tolist() == [[4, 24], [1, 3]]
    assert torch.equal(rect.expand(103), lines.expand(103))
    back = rect.as_lines()
    assert torch.equal(back.expand(103), lines.expand(103)) and len(back) == 5


def test_row_parallel_shard_is_one_copy_per_shard():
    index = _index()
    tmap, _ = capture(index, lambda k: {"loader": _trainer(index.id_state(k))}, log=lambda s: None)
    vmap, _ = capture(index, lambda k: {"loader": _rollout(index.id_state(k))}, log=lambda s: None)
    t = tmap["loader"].params["w.tp0"].runs
    assert len(t) == 1 and t.shape.tolist() == [[R, C // 2]]
    assert t.param_strides.tolist() == [[C // 2, 1]] and t.ckpt_strides.tolist() == [[C, 1]]

    copies = correspond(tmap["loader"], vmap["loader"])
    rep = copies.report
    assert not rep["missing_elements"] and not rep["unsupported_dst"] and not rep["unsupported_src"]
    assert rep["line_copies"] == 2 * R + 1 and rep["copies"] == 3
    assert rep["bytes"] == (R * C + 8) * 2

    plan = plan_rounds(copies, n_src=2, n_dst=2, round_bytes=1 << 20, max_piece_bytes=1 << 20)
    tr = plan.traffic()
    assert tr["transport_reads"] == 2 * 3 and tr["strided_dst_reads"] == 2 * 2

    torch.manual_seed(0)
    hf = {"norm": torch.randn(8).to(torch.bfloat16),
          "w": torch.randn(R, C).to(torch.bfloat16)}
    src = {s: _trainer(hf) for s in range(2)}
    ref = _rollout(hf)
    for d in range(2):
        got = {n: torch.zeros_like(v) for n, v in ref.items()}
        written = apply_plan(plan, d, src, got)
        for n in ref:
            assert torch.equal(got[n], ref[n]), n
            assert written[n] == ref[n].numel() * 2


def test_pieces_split_by_rows():
    index = _index()
    tmap, _ = capture(index, lambda k: {"loader": _trainer(index.id_state(k))}, log=lambda s: None)
    vmap, _ = capture(index, lambda k: {"loader": _rollout(index.id_state(k))}, log=lambda s: None)
    copies = correspond(tmap["loader"], vmap["loader"])
    # 48-byte rows, 256-byte pieces: 5 rows per piece.
    plan = plan_rounds(copies, n_src=2, n_dst=1, round_bytes=1024, max_piece_bytes=256)
    rb, nr = plan.col("row_bytes"), plan.col("rows")
    assert int((rb * nr).max()) <= 256
    assert int(plan.nbytes().sum()) == copies.total_bytes()
    torch.manual_seed(1)
    hf = {"norm": torch.randn(8).to(torch.bfloat16), "w": torch.randn(R, C).to(torch.bfloat16)}
    got = {n: torch.zeros_like(v) for n, v in _rollout(hf).items()}
    apply_plan(plan, 0, {s: _trainer(hf) for s in range(2)}, got)
    assert torch.equal(got["w"], hf["w"])
