"""The uint8 round-trip byte check accepts exactly what the range-and-integer check did."""

import pytest
import torch

from lumenrl.transfer.weight_source_map import _as_bytes


def _reference(t: torch.Tensor, ones: bool):
    f = t.reshape(-1).to(torch.float32)
    top = 1 if ones else 255
    ok = (f == f.round()) & (f >= 0) & (f <= top)
    return ok, f


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("ones", [False, True])
def test_every_bit_pattern(dtype, ones):
    allbits = torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32).to(torch.int16).view(dtype)
    ok, f = _reference(allbits, ones)
    # Element by element: each value alone must be accepted iff the reference accepts it.
    for i in range(0, allbits.numel(), 4096):
        chunk = allbits[i:i + 4096]
        got = torch.tensor([_as_bytes(v.reshape(1), ones) is not None for v in chunk])
        assert torch.equal(got, ok[i:i + 4096]), dtype
    valid = allbits[ok]
    u = _as_bytes(valid, ones)
    assert u is not None and u.dtype == torch.uint8
    assert torch.equal(u.to(torch.float32), f[ok])
    assert _as_bytes(allbits, ones) is None


def test_other_dtypes():
    x = torch.tensor([0.0, 1.0, 255.0, 17.0])
    assert torch.equal(_as_bytes(x, False), torch.tensor([0, 1, 255, 17], dtype=torch.uint8))
    assert _as_bytes(torch.tensor([256.0]), False) is None
    assert _as_bytes(torch.tensor([0.5]), False) is None
    assert _as_bytes(torch.tensor([-1.0]), False) is None
    assert _as_bytes(torch.tensor([float("nan")]), False) is None
    assert _as_bytes(torch.tensor([2.0]), True) is None
    i = torch.tensor([3, 300], dtype=torch.int64)
    assert _as_bytes(i, False) is None
    assert torch.equal(_as_bytes(i[:1], False), torch.tensor([3], dtype=torch.uint8))
