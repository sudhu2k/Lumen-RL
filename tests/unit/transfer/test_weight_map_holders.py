"""Planning when trainer ranks hold different tensors (expert parallelism)."""

import torch

from lumenrl.transfer.weight_plan import apply_plan, correspond, plan_rounds
from lumenrl.transfer.weight_source_map import CheckpointIndex, CheckpointTensor, capture

# Two "expert" tensors held by one rank each, one replicated tensor.
_SHAPES = {"e0": (64, 32), "e1": (64, 32), "shared": (16, 32)}
_HOLDERS = {"e0": [0, 2], "e1": [1, 3], "shared": [0, 1, 2, 3]}


def _index() -> CheckpointIndex:
    tensors, off = [], 0
    for name, shape in _SHAPES.items():
        n = shape[0] * shape[1]
        tensors.append(CheckpointTensor(name, shape, "BF16", off, n))
        off += n
    return CheckpointIndex(tensors)


def _rollout(hf):
    return {"w": torch.cat([hf["e0"], hf["e1"]], 0), "s": hf["shared"].clone()}


def _plan(interleave: bool = False):
    index = _index()
    tmap, _ = capture(index, lambda k: {"loader": dict(index.id_state(k))}, log=lambda s: None)
    vmap, _ = capture(index, lambda k: {"loader": _rollout(index.id_state(k))}, log=lambda s: None)
    copies = correspond(tmap["loader"], vmap["loader"])
    return plan_rounds(copies, n_src=4, n_dst=2, round_bytes=2048, max_piece_bytes=512,
                       holders=_HOLDERS, interleave=interleave)


def _check_parity(plan) -> None:
    torch.manual_seed(0)
    hf = {n: torch.randn(s).to(torch.bfloat16) for n, s in _SHAPES.items()}
    src = {r: {n: t for n, t in hf.items() if r in _HOLDERS[n]} for r in range(4)}
    ref = _rollout(hf)
    for d in range(2):
        got = {n: torch.zeros_like(t) for n, t in ref.items()}
        apply_plan(plan, d, src, got)
        for n in ref:
            assert torch.equal(got[n], ref[n]), n


def test_plan_reads_only_from_holders():
    plan = _plan()
    for s, sp in zip(plan.col("src_rank").tolist(), plan.col("src_param").tolist()):
        assert s in _HOLDERS[plan.src_names[sp]]
    tr = plan.traffic()
    assert sum(tr["per_src"]) == 2 * plan.meta["model_bytes"]
    assert max(tr["per_src"]) - min(tr["per_src"]) <= 2 * 512
    _check_parity(plan)


def test_interleave_mixes_holders_into_every_round():
    base, mixed = _plan(), _plan(interleave=True)
    assert mixed.meta["interleave"] and not base.meta["interleave"]
    assert sum(mixed.traffic()["per_src"]) == sum(base.traffic()["per_src"])
    assert mixed.traffic()["round_src_skew_mean"] < base.traffic()["round_src_skew_mean"]
    _check_parity(mixed)
