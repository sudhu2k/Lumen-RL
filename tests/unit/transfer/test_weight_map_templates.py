"""Template checks: look-alike parameters predicted from one decoded member (CPU only)."""

import pytest
import torch

from lumenrl.transfer.weight_source_map import (
    CheckpointIndex,
    CheckpointTensor,
    _pack,
    capture,
)


def _index(base: int, layers: int, experts: int, rows: int = 6, cols: int = 40,
           order=lambda L: ("gate", "up", "down")) -> CheckpointIndex:
    tensors, start = [], base
    for L in range(layers):
        for e in range(experts):
            for p in order(L):
                shape = (cols, rows) if p == "down" else (rows, cols)
                tensors.append(CheckpointTensor(f"model.layers.{L}.mlp.experts.{e}.{p}_proj.weight",
                                                shape, "BF16", start, rows * cols))
                start += rows * cols
        tensors.append(CheckpointTensor(f"model.layers.{L}.norm.weight", (cols,), "BF16", start, cols))
        start += cols + 3
    return CheckpointIndex(tensors)


def _moe_loader(layers, experts, odd=()):
    """Stacked per-layer w13 [gate;up] and transposed w2; ``odd`` layers load w13 up-first."""
    def load(hf):
        out = {}
        for L in range(layers):
            p = f"model.layers.{L}.mlp.experts."
            parts = ("up", "gate") if L in odd else ("gate", "up")
            out[p + "w13"] = torch.stack([torch.cat([hf[p + f"{e}.{x}_proj.weight"] for x in parts])
                                          for e in range(experts)])
            out[p + "w2"] = torch.stack([hf[p + f"{e}.down_proj.weight"].t()
                                         for e in range(experts)]).contiguous()
            out[f"model.layers.{L}.norm"] = hf[f"model.layers.{L}.norm.weight"].clone()
            pad = torch.zeros(48, dtype=hf[f"model.layers.{L}.norm.weight"].dtype)
            pad[:40 - (L % 2)] = hf[f"model.layers.{L}.norm.weight"][:40 - (L % 2)]
            out[f"model.layers.{L}.padded"] = pad
        return out
    return load


def _expert_loader(layers, experts):
    """One parameter per expert, the trainer's grouped-GEMM naming."""
    def load(hf):
        out = {}
        for L in range(layers):
            for e in range(experts):
                p = f"model.layers.{L}.mlp.experts.{e}."
                out[f"decoder.layers.{L}.mlp.experts.linear_fc1.weight{e}"] = torch.cat(
                    [hf[p + "gate_proj.weight"], hf[p + "up_proj.weight"]])
                out[f"decoder.layers.{L}.mlp.experts.linear_fc2.weight{e}"] = hf[p + "down_proj.weight"].clone()
        return out
    return load


def _truth(index, loader):
    hf = {t.name: torch.arange(t.start, t.start + t.numel).view(t.shape) for t in index.tensors}
    return {n: t.reshape(-1) for n, t in loader(hf).items()}


def _check(index, loader):
    quiet = lambda s: None  # noqa: E731
    on, _ = capture(index, lambda k: {"loader": loader(index.id_state(k))}, log=quiet)
    off, _ = capture(index, lambda k: {"loader": loader(index.id_state(k))}, log=quiet,
                     templates=False)
    truth = _truth(index, loader)
    for name, t in truth.items():
        ids = on["loader"].params[name].ids()
        w = ids >= 0
        assert torch.equal(ids[w], t[w]), name
        assert torch.equal(ids, off["loader"].params[name].ids()), name
    return on["loader"].meta["templates"]


def test_pack_is_exact_and_reads_back():
    torch.manual_seed(0)
    cases = [
        (torch.arange(1000) & 255).to(torch.uint8),
        ((torch.arange(1000) + 77) >> 8 & 255).to(torch.uint8),
        torch.randint(0, 256, (1000,), dtype=torch.uint8),
        torch.zeros(5, dtype=torch.uint8),
        torch.ones(1, dtype=torch.uint8),
    ]
    for u in cases:
        p = _pack(u)
        assert torch.equal(p.bytes(), u)
        pos = torch.randint(0, u.numel(), (50,))
        assert torch.equal(p.at(pos), u[pos].to(torch.int64))
        assert p.same(_pack(u.clone()))
        v = u.clone()
        v[u.numel() // 2] ^= 1
        assert not p.same(_pack(v))
    assert _pack(cases[0]).raw is None and _pack(cases[2]).raw is not None


@pytest.mark.parametrize("base", [0, 250, (1 << 33) - 1000])
def test_stacked_experts_predicted_and_exact(base):
    index = _index(base, layers=4, experts=3)
    stats = _check(index, _moe_loader(4, 3))
    assert stats["predicted"] >= 6


def test_per_expert_params_predicted_across_carries():
    index = _index(200, layers=2, experts=24)
    stats = _check(index, _expert_loader(2, 24))
    assert stats["predicted"] >= 2 * 2 * 24 - 4
    assert stats["replayed"] == 0


def test_checkpoint_order_differing_between_layers_is_predicted():
    # Even layers: gate and up adjacent in the file, so each expert's [gate; up] is one
    # run. Odd layers: down sits between them, so it is two.
    index = _index(1000, layers=4, experts=3,
                   order=lambda L: ("gate", "up", "down") if L % 2 == 0 else ("gate", "down", "up"))
    stats = _check(index, _moe_loader(4, 3))
    assert stats["replayed_names"] == ["model.layers.1.padded"]  # written differently by design


def test_look_alike_with_different_layout_is_replayed():
    index = _index(1 << 16, layers=4, experts=2)
    stats = _check(index, _moe_loader(4, 2, odd=(2,)))
    assert stats["replayed"] >= 1


def test_look_alike_with_different_padding_is_exact():
    index = _index(300, layers=3, experts=1)
    _check(index, _moe_loader(3, 1))
