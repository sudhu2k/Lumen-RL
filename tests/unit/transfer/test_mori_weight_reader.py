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
