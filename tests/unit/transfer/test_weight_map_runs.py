"""``Runs`` as strided views: N-D expansion and maps saved before the rename."""

import torch

from lumenrl.transfer.weight_source_map import Runs


def test_lines_expand():
    r = Runs.lines(torch.tensor([0, 4]), torch.tensor([3, 2]), torch.tensor([10, 50]),
                   torch.tensor([1, -2]))
    assert r.is_lines() and r.ndim == 1
    assert r.expand(7).tolist() == [10, 11, 12, -1, 50, 48, -1]


def test_two_dim_expand():
    # One run: a 3 x 2 block stored contiguously, read from checkpoint rows of 10.
    r = Runs(torch.tensor([1]), torch.tensor([100]), torch.tensor([[3, 2]]),
             torch.tensor([[2, 1]]), torch.tensor([[10, 1]]))
    assert not r.is_lines() and r.ndim == 2 and r.length.tolist() == [6]
    assert r.expand(8).tolist() == [-1, 100, 101, 110, 111, 120, 121, -1]


def test_strided_param_side():
    # Destination strided: half-rows of a 2 x 4 parameter (columns 2-3).
    r = Runs(torch.tensor([2]), torch.tensor([0]), torch.tensor([[2, 2]]),
             torch.tensor([[4, 1]]), torch.tensor([[2, 1]]))
    assert r.expand(8).tolist() == [-1, -1, 0, 1, -1, -1, 2, 3]


def test_legacy_dict():
    legacy = {"pos": torch.tensor([0, 5]), "length": torch.tensor([5, 1]),
              "id0": torch.tensor([7, 3]), "step": torch.tensor([1, 0])}
    r = Runs.from_dict(legacy)
    assert r.is_lines()
    assert r.expand(6).tolist() == [7, 8, 9, 10, 11, 3]
    assert Runs.from_dict(r.to_dict()).expand(6).tolist() == r.expand(6).tolist()
