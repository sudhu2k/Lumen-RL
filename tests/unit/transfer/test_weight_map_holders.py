"""Planning when trainer ranks hold different tensors (expert parallelism)."""

import torch

from lumenrl.transfer.weight_plan import apply_plan, correspond, merge_plans, plan_rounds
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


def _tp_rollout(hf, tp: int):
    """Rank ``tp`` of a TP=2 instance: column halves, so every row is a strided read."""
    full = _rollout(hf)
    return {n: t[:, tp * 16:(tp + 1) * 16].contiguous() for n, t in full.items()}


def test_merge_plans_single_rank_keeps_rows():
    plan = _plan()
    merged = merge_plans([(plan, [0])], world=1, replicas=2)
    assert torch.equal(merged.rows, plan.rows)
    assert merged.meta["n_dst"] == 2 and merged.meta["vllm_world"] == 1
    assert merged.meta["model_bytes"] == plan.meta["model_bytes"]
    _check_parity(merged)


def test_merge_plans_tp_ranks_get_global_rollout_ranks():
    index = _index()
    tmap, _ = capture(index, lambda k: {"loader": dict(index.id_state(k))}, log=lambda s: None)
    groups = []
    for tp in range(2):
        vmap, _ = capture(index, lambda k: {"loader": _tp_rollout(index.id_state(k), tp)},
                          log=lambda s: None)
        groups.append((plan_rounds(correspond(tmap["loader"], vmap["loader"]), n_src=4,
                                   n_dst=2, round_bytes=2048, max_piece_bytes=512,
                                   holders=_HOLDERS), [tp]))
    merged = merge_plans(groups, world=2, replicas=2)
    assert merged.meta["n_dst"] == 4 and merged.meta["vllm_replicas"] == 2
    assert sorted(set(merged.col("dst_rank").tolist())) == [0, 1, 2, 3]
    model_bytes = 2 * sum(r * c for r, c in _SHAPES.values())
    assert merged.meta["model_bytes"] == model_bytes

    torch.manual_seed(0)
    hf = {n: torch.randn(s).to(torch.bfloat16) for n, s in _SHAPES.items()}
    src = {r: {n: t for n, t in hf.items() if r in _HOLDERS[n]} for r in range(4)}
    for d in range(4):
        ref = _tp_rollout(hf, d % 2)
        got = {n: torch.zeros_like(t) for n, t in ref.items()}
        apply_plan(merged, d, src, got)
        for n in ref:
            assert torch.equal(got[n], ref[n]), (d, n)
