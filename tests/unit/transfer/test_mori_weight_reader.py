"""MoriWeightReader's reads: strided when MORI has batch_read_strided, else one per row."""

import sys
import types

import pytest
import torch

from lumenrl.transfer import mori_weight_transfer as mwt
from lumenrl.transfer.mori_weight_transfer import MoriWeightReader, SourceExport
from lumenrl.transfer.weight_plan import Plan

# [dst_rank, round, src_rank, dst_param, dst_off, src_param, src_off, row_bytes, rows,
#  dst_stride, src_stride]: three 16 B rows, 32 B apart locally, then one contiguous read.
_ROWS = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 16, 3, 32, 16],
                      [0, 0, 0, 0, 96, 0, 64, 32, 1, 32, 32]])


class _Status:
    def Wait(self):
        pass

    def Succeeded(self):
        return True


class _Engine:
    def __init__(self, name, cfg):
        self.calls = []

    def create_backend(self, kind, cfg):
        pass

    def register_remote_engine(self, desc):
        pass

    def allocate_transfer_uid(self):
        return 0

    def batch_read(self, ld, lo, rd, ro, sz, uids):
        self.calls.append(("plain", lo, ro, sz))
        return [_Status() for _ in sz]


class _StridedEngine(_Engine):
    def batch_read_strided(self, ld, lo, rd, ro, sz, nr, lp, rp, uids):
        self.calls.append(("strided", lo, ro, sz, nr, lp, rp))
        return [_Status() for _ in sz]


@pytest.fixture
def mori(monkeypatch):
    def install(engine_cls):
        io = types.ModuleType("mori.io")
        io.IOEngine = engine_cls
        io.IOEngineConfig = io.XgmiBackendConfig = lambda **kw: kw
        io.BackendType = types.SimpleNamespace(XGMI="xgmi")
        io.MemoryLocationType = types.SimpleNamespace(GPU="gpu")
        io.EngineDesc = io.MemoryDesc = types.SimpleNamespace(unpack=lambda b: b)
        monkeypatch.setitem(sys.modules, "mori", types.ModuleType("mori"))
        monkeypatch.setitem(sys.modules, "mori.io", io)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
        monkeypatch.setattr(mwt, "_register", lambda eng, ptr, nbytes, device: ("local", ptr))

    return install


def _reader(**kw):
    plan = Plan(["w"], ["a"], _ROWS.clone())
    w = torch.zeros(128, dtype=torch.uint8)
    src = SourceExport(0, b"eng", [b"mem"], {0: (0, 0, 96)})
    return MoriWeightReader(plan, 0, {"w": w}, [src], **kw)


def test_strided_reads_one_per_plan_row(mori):
    mori(_StridedEngine)
    r = _reader()
    assert r.strided and len(r.entries) == 2
    r.sync()
    (kind, lo, ro, sz, nr, lp, rp), = r.eng.calls
    assert kind == "strided"
    assert (lo[0], ro[0], sz[0], nr[0], lp[0], rp[0]) == ([0, 96], [0, 64], [16, 32], [3, 1],
                                                          [32, 32], [16, 32])
    assert r.bytes == 3 * 16 + 32


_SRC = torch.arange(64, dtype=torch.int16).view(torch.uint8)  # source memory b"mem"


class _CopyEngine(_StridedEngine):
    """Does the reads: local memories are uint8 tensors, b"mem" is ``_SRC``."""

    def batch_read_strided(self, ld, lo, rd, ro, sz, nr, lp, rp, uids):
        super().batch_read_strided(ld, lo, rd, ro, sz, nr, lp, rp, uids)
        for dst, los, _, ros, szs, nrs, lps, rps in zip(ld, lo, rd, ro, sz, nr, lp, rp):
            for a, b, n, k, pa, pb in zip(los, ros, szs, nrs, lps, rps):
                for i in range(k):
                    dst[a + i * pa:a + i * pa + n] = _SRC[b + i * pb:b + i * pb + n]
        return [_Status() for _ in sz]


def _checkpoint_reader(monkeypatch, slot_bytes):
    """Checkpoint tensors a (4x4), b (8,), c (4x4) in BF16; this rank's loader reads
    a's first two columns, all of b and c's first row. A 32 B slot holds one tensor
    at a time, so the third batch reuses the first slot."""
    import ctypes

    monkeypatch.setattr(mwt, "_register", lambda eng, ptr, nbytes, device: torch.frombuffer(
        (ctypes.c_uint8 * nbytes).from_address(ptr), dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "Event", lambda: types.SimpleNamespace(
        record=lambda: None, synchronize=lambda: None))
    # Bytes; words are BF16 elements: a[:, :2] <- words 0-7, b <- 16-23, c[0] <- 32-35.
    rows = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 4, 4, 8, 4],
                         [0, 1, 0, 1, 0, 0, 32, 16, 1, 16, 16],
                         [0, 2, 0, 2, 0, 0, 64, 8, 1, 8, 8]])
    staged = {"a": {"shape": [4, 4], "dtype": "bfloat16"}, "b": {"shape": [8], "dtype": "bfloat16"},
              "c": {"shape": [4, 4], "dtype": "bfloat16"}}
    plan = Plan(["a", "b", "c"], ["s"], rows, {"staged_checkpoint": staged})
    src = SourceExport(0, b"eng", [b"mem"], {0: (0, 0, _SRC.numel())})
    return mwt.MoriCheckpointReader(plan, 0, [src], device="cpu", slot_bytes=slot_bytes)


@pytest.mark.parametrize("slot_bytes", [32, 1 << 20])
def test_checkpoint_reader_stages_whole_tensors_and_hands_over_clones(mori, monkeypatch,
                                                                      slot_bytes):
    mori(_CopyEngine)
    r = _checkpoint_reader(monkeypatch, slot_bytes)
    assert len(r.batches) == (3 if slot_bytes == 32 else 1) and len(r.slots) == min(2, len(r.batches))
    loaded = []
    stats = r.sync(loaded.extend, verify=True)
    got = dict(loaded)
    words = _SRC.view(torch.int16)
    assert [n for n, _ in loaded] == ["a", "b", "c"]
    assert torch.equal(got["a"][:, :2].contiguous().view(torch.int16).view(-1), words[0:8])
    assert torch.equal(got["b"].view(torch.int16), words[16:24])
    assert torch.equal(got["c"][0].view(torch.int16), words[32:36])
    want = mwt.read_hashes(r.rows, {0: _SRC}, mwt._SP, mwt._SO, mwt._SS)
    assert torch.equal(r.hashes(), want)
    assert stats["bytes"] == 16 + 16 + 8


@pytest.mark.parametrize("n", [1, 7, 65521, 65521 * 3 + 5])
def test_tensor_digest_independent_of_chunking(n):
    t = torch.randn(n, dtype=torch.float32).to(torch.bfloat16)
    assert mwt.tensor_digest(t, chunk=65521) == mwt.tensor_digest(t, chunk=65521 << 4)
    flat = t.view(torch.int16).to(torch.int64)
    assert mwt.tensor_digest(t) == int((flat * (torch.arange(n) % 65521 + 1)).sum())


@pytest.mark.parametrize("engine, strided", [(_Engine, True), (_StridedEngine, False)])
def test_plain_reads_without_strided_api_or_when_off(mori, engine, strided):
    mori(engine)
    r = _reader(strided=strided)
    assert not r.strided and len(r.entries) == 4
    r.sync()
    (kind, lo, ro, sz), = r.eng.calls
    assert kind == "plain"
    assert (lo[0], ro[0], sz[0]) == ([0, 32, 64, 96], [0, 16, 32, 64], [16, 16, 16, 32])
    assert r.bytes == 3 * 16 + 32
