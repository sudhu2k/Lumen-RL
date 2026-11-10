"""The cross-node reader: the node planner, and a node's ranks syncing against a fake
MORI engine that copies bytes between registrations, one thread per rank since each
waits on the others' mirrors."""

import itertools
import os
import sys
import threading
import types

import pytest
import torch

from lumenrl.transfer import mori_node as mn
from lumenrl.transfer import mori_weight_transfer as mwt
from lumenrl.transfer.mori_node import MoriNodeReader, plan_node
from lumenrl.transfer.mori_weight_transfer import SourceExport, _flat, read_hashes
from lumenrl.transfer.weight_plan import Plan

_MEMS: dict = {}   # registration -> flat uint8 tensor
_BY_ID: dict = {}  # packed registration -> registration
_ids = itertools.count()
_SMALL = 64 << 10
_BIG = 128 << 10
_SRC_NB = {0: 4096, 1: 64 * 64, 2: 4 * _BIG, 3: 2048, 4: 128 * 16}
_HOLDERS = {0: [0], 1: [0, 1], 2: [0, 1], 3: [0], 4: [1]}
# [dst_rank, round, src_rank, dst_param, dst_off, src_param, src_off, row_bytes, rows,
#  dst_stride, src_stride]. Trainer rank s is on GPU s.
_ROWS = torch.tensor([
    [0, 0, 0, 0, 0, 0, 0, 4096, 1, 4096, 4096],
    [0, 0, 1, 1, 0, 1, 0, 32 * 64, 1, 32 * 64, 32 * 64],
    [1, 0, 0, 2, 0, 1, 32, 32, 64, 32, 64],                           # small pieces
    [1, 0, 1, 3, 0, 2, _BIG // 2, _BIG // 2, 4, _BIG // 2, _BIG],     # large pieces
    [0, 0, 1, 4, 0, 2, 0, _BIG // 2, 4, _BIG // 2, _BIG],
    [1, 0, 0, 5, 0, 3, 0, 2048, 1, 2048, 2048],
    [0, 0, 0, 7, 0, 3, 0, 2048, 1, 2048, 2048],                       # replicated
    [0, 0, 1, 6, 0, 4, 0, 64, 16, 64, 128],                           # small pieces
])
_DST_NB = {0: 4096, 1: 32 * 64, 2: 32 * 64, 3: 2 * _BIG, 4: 2 * _BIG, 5: 2048, 6: 64 * 16,
           7: 2048}


def _plan():
    return Plan([f"d{i}" for i in range(8)], [f"s{q}" for q in range(5)], _ROWS.clone())


def _exports():
    out = []
    for s in (0, 1):
        mine = [q for q in _SRC_NB if s in _HOLDERS[q]]
        mems = [(s, q) for q in mine]
        out.append(SourceExport(s, b"xgmi", mems, {q: (i, 0, _SRC_NB[q]) for i, q in enumerate(mine)},
                                rdma_engine=b"rdma", rdma_memories=mems, host="T", rail=s))
    return out


def _fetched(node):
    return [k[:4] for batches in node.fetch for b in batches for k in b]


def test_plan_node_mirrors_each_range_once_on_rail():
    plan = _plan()
    node = plan_node(plan, _exports(), [0, 1], [0, 1], _SMALL)
    fetched = _fetched(node)
    assert len(fetched) == len(set(fetched))
    assert node.offrail_bytes == 0
    assert node.host == {0: 0, 1: 1}                 # trainer rank s mirrored on rail s
    for p, batches in enumerate(node.fetch):
        ks = [k for b in batches for k in b]
        assert all(node.host[k[0]] == p for k in ks)
        # Ranges another position reads come first.
        foreign = [any(r[0] != node.group[p] and r[5] == k[1] and node.host[r[2]] == p
                       for r in node.rows.tolist()) for k in ks]
        assert foreign == sorted(foreign, reverse=True)
    # Source 3 is fetched once for both ranks, from the only holder.
    assert [k for k in fetched if k[1] == 3] == [(0, 3, 0, 2048)]
    # Large strided rows are fetched row by row; here the two ranks' halves of the
    # rows tile the tensor, so they merge into one read. Small ones as their span.
    assert [k[3] for k in fetched if k[1] == 2] == [4 * _BIG]
    assert [k[3] for k in fetched if k[1] == 4] == [15 * 128 + 64]
    # Rows are the plan's, reading from the chosen holder; nothing else changes.
    keep = [c for c in range(11) if c not in (1, 2)]
    assert torch.equal(node.rows[:, keep], plan.rows[:, keep])
    for r in node.rows.tolist():
        assert r[5] in {k[1] for k in fetched if k[0] == r[2]}


def _landed(node):
    return [k for batches in node.fetch for b in batches for k in b if k[4] >= 0]


def test_plan_node_lands_sources_only_their_mirror_gpu_reads_in_place():
    plan = _plan()
    node = plan_node(plan, _exports(), [0, 1], [0, 1], _SMALL)
    # Source 0 is read only by rank 0, the GPU on its holder's rail: no mirror.
    assert _landed(node) == [(0, 0, 0, 4096, 0, 0)]
    assert node.in_place.tolist() == [r[5] == 0 for r in node.rows.tolist()]
    assert (0, 0) not in node.offsets and node.land_bytes == [4096, 0]
    # Sources other GPUs also read stay mirrored, with their mirrors unchanged.
    every = plan_node(plan, _exports(), [0, 1], [0, 1], _SMALL, land=[False, False])
    assert not _landed(every) and not bool(every.in_place.any())
    assert every.mirror_bytes[0] == node.mirror_bytes[0] + 4096
    assert node.mirror_bytes[1] == every.mirror_bytes[1]
    assert not bool(node.staged.any()) and node.stage_bytes == [0, 0]
    # vLLM permutes d0 after loading: fetched straight into a staging slot instead.
    plan.meta["views"] = {"d0": {"sizes": [64, 64], "strides": [1, 64], "offset": 0}}
    staged = plan_node(plan, _exports(), [0, 1], [0, 1], _SMALL)
    assert not _landed(staged) and not bool(staged.in_place.any())
    assert staged.staged.tolist() == node.in_place.tolist()
    assert (0, 0) not in staged.offsets and staged.stage_bytes == [4096, 0]
    assert staged.mirror_bytes == node.mirror_bytes
    mirrored = plan_node(plan, _exports(), [0, 1], [0, 1], _SMALL, stage=[False, False])
    assert not bool(mirrored.staged.any()) and mirrored.mirror_bytes == every.mirror_bytes


def test_plan_node_stages_a_permuted_param_only_if_every_row_into_it_can_be():
    """d3 (permuted) reads own-only source 0 and source 3, which rank 1 also reads:
    d3 goes through the mirror, so source 0 is mirrored for d0 as well. A checkpoint
    plan may mix the two in one batch: source 0's rows are staged."""
    rows = torch.tensor([
        [0, 0, 0, 3, 0, 0, 0, 2048, 1, 2048, 2048],
        [0, 0, 0, 3, 2048, 3, 0, 2048, 1, 2048, 2048],
        [1, 0, 0, 5, 0, 3, 0, 2048, 1, 2048, 2048],
        [0, 0, 0, 0, 0, 0, 2048, 2048, 1, 2048, 2048],
    ])
    plan = Plan(_plan().dst_names, _plan().src_names, rows)
    assert plan_node(plan, _exports(), [0, 1], [0, 1], _SMALL).in_place.tolist() == [
        True, False, False, True]
    plan.meta["views"] = {"d3": {"sizes": [64, 64], "strides": [1, 64], "offset": 0}}
    node = plan_node(plan, _exports(), [0, 1], [0, 1], _SMALL)
    assert not bool(node.in_place.any()) and not bool(node.staged.any())
    assert set(node.offsets) == {(0, 0), (0, 3)}
    plan.meta = {"staged_checkpoint": {n: {"dtype": "uint8", "shape": [2 * _BIG]}
                                       for n in plan.dst_names}}
    node = plan_node(plan, _exports(), [0, 1], [0, 1], _SMALL)
    assert node.staged.tolist() == [True, False, False, True]
    assert not bool(node.in_place.any()) and set(node.offsets) == {(0, 3)}


def test_plan_node_lands_large_strided_rows_in_place_row_by_row():
    """Rank 0 alone reads a strided slice of source 2 (large rows) and a contiguous
    run of source 3 split over two rows; small strided rows (source 4) keep the
    whole source mirrored."""
    rows = torch.tensor([
        [0, 0, 0, 4, 0, 2, 0, _BIG // 2, 4, _BIG // 2, _BIG],
        [0, 0, 0, 5, 0, 3, 0, 1024, 1, 1024, 1024],
        [0, 0, 0, 5, 1024, 3, 1024, 1024, 1, 1024, 1024],
        [0, 0, 1, 6, 0, 4, 0, 64, 16, 64, 128],
    ])
    plan = Plan(_plan().dst_names, _plan().src_names, rows)
    node = plan_node(plan, _exports(), [0, 1], [0, 1], _SMALL)
    landed = _landed(node)
    assert sorted(k for k in landed if k[1] == 2) == [
        (0, 2, i * _BIG, _BIG // 2, 4, i * _BIG // 2) for i in range(4)]
    assert [k for k in landed if k[1] == 3] == [(0, 3, 0, 2048, 5, 0)]  # merged
    assert node.in_place.tolist() == [True, True, True, False]
    assert set(node.offsets) == {(1, 4)}


def test_plan_node_rounds_follow_the_mirror_batches():
    """With one range per batch, a row's round is the batch that lands its bytes."""
    node = plan_node(_plan(), _exports(), [0, 1], [0, 1], _SMALL, batch=1)
    where = {k[1]: i for batches in node.fetch for i, b in enumerate(batches) for k in b}
    assert max(len(b) for batches in node.fetch for b in batches) == 1
    assert [r[1] for r in node.rows.tolist()] == [where[r[5]] for r in node.rows.tolist()]


def test_plan_node_reads_off_rail_without_a_gpu_on_the_holders_rail():
    node = plan_node(_plan(), _exports(), [0, 1], [0, 5], _SMALL)
    fetched = _fetched(node)
    assert {k[0] for k in fetched if k[1] in (1, 2)} == {0}   # the holder on rail 0
    assert node.offrail_bytes == 15 * 128 + 64                # source 4, held on rail 1 only
    assert [k[0] for k in fetched if k[1] == 4] == [1]
    assert node.host[1] == 0                                  # by its only consumer


def _two_trainer_nodes():
    """Trainer nodes T1 (ranks 0, 1) and T2 (ranks 2, 3), each holding the model as
    ``_exports`` does, rank s on rail s % 2; two rollout instances [0, 1], [2, 3]."""
    out = []
    for s in range(4):
        mine = [q for q in _SRC_NB if s % 2 in _HOLDERS[q]]
        mems = [(s, q) for q in mine]
        out.append(SourceExport(s, b"xgmi", mems, {q: (i, 0, _SRC_NB[q]) for i, q in enumerate(mine)},
                                rdma_engine=b"rdma", rdma_memories=mems,
                                host=f"T{s // 2 + 1}", rail=s % 2))
    rows = torch.cat([_ROWS, _ROWS + torch.tensor([2] + [0] * 10)])
    return Plan([f"d{i}" for i in range(8)], [f"s{q}" for q in range(5)], rows), out


def _sent_by_host(plans):
    out = {"T1": 0, "T2": 0}
    for node in plans:
        for k in _fetched(node):
            out[f"T{k[0] // 2 + 1}"] += k[3]
    return out


_GROUPS, _RAILS = [[0, 1], [2, 3]], {0: 0, 1: 1, 2: 0, 3: 1}


def _one_tensor():
    """Each instance's rank on rail 0 reads only source 0, held by ranks 0 (T1) and 2 (T2)."""
    plan, sources = _two_trainer_nodes()
    rows = torch.tensor([[d, 0, 0, 0, 0, 0, 0, 4096, 1, 4096, 4096] for d in (0, 2)])
    return Plan(plan.dst_names, plan.src_names, rows), sources


def test_plan_node_spreads_instances_over_trainer_nodes():
    plan, sources = _one_tensor()
    alone = [plan_node(plan, sources, g, [0, 1], _SMALL) for g in _GROUPS]
    assert [_fetched(node)[0][0] for node in alone] == [0, 0]
    joint = [plan_node(plan, sources, g, [0, 1], _SMALL, _GROUPS, _RAILS) for g in _GROUPS]
    assert [_fetched(node)[0][0] for node in joint] == [0, 2]

    plan, sources = _two_trainer_nodes()
    joint = [plan_node(plan, sources, g, [0, 1], _SMALL, _GROUPS, _RAILS) for g in _GROUPS]
    sent = _sent_by_host(joint)
    assert abs(sent["T1"] - sent["T2"]) <= 4 * _BIG  # within the largest fetch
    for node in joint:
        assert node.offrail_bytes == 0
        assert all(node.host[k[0]] == k[0] % 2 for k in _fetched(node))


_TENSORS: list = []  # device tensors a registration's address can fall in


def test_plan_node_mirrors_two_trainer_nodes_holding_different_layers():
    """Trainer PP across nodes: T1 (ranks 0, 1) holds sources 0-2, T2 (ranks 2, 3) holds
    3-4, rank s on rail s % 2. Each rollout GPU mirrors the trainer GPU on its rail on
    both nodes, so both trainer nodes send and every read stays on-rail."""
    stage = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
    sources = []
    for s in range(4):
        mine = [q for q in _SRC_NB if stage[q] == s // 2 and s % 2 in _HOLDERS[q]]
        mems = [(s, q) for q in mine]
        sources.append(SourceExport(s, b"xgmi", mems,
                                    {q: (i, 0, _SRC_NB[q]) for i, q in enumerate(mine)},
                                    rdma_engine=b"rdma", rdma_memories=mems,
                                    host=f"T{s // 2 + 1}", rail=s % 2))
    node = plan_node(_plan(), sources, [0, 1], [0, 1], _SMALL)
    fetched = _fetched(node)
    assert node.offrail_bytes == 0
    assert {k[0] // 2 for k in fetched} == {0, 1}            # both trainer nodes send
    assert all(node.host[k[0]] == k[0] % 2 for k in fetched)  # on-rail
    assert {h for s, h in node.host.items() if s // 2 == 0} == {0, 1}
    assert {h for s, h in node.host.items() if s // 2 == 1} <= {0, 1}
    for r in node.rows.tolist():
        assert stage[r[5]] == r[2] // 2                      # read from its own stage


class _Reg:
    """A registration at a device address; packed, an id that unpacks to it."""

    def __init__(self, ptr):
        self.ptr = int(ptr)
        self.id = next(_ids).to_bytes(8, "little")
        _BY_ID[self.id] = self

    def pack(self):
        return self.id


def _at(mem, off, n):
    """``n`` bytes at ``off`` in a registration or a trainer memory (``_MEMS``)."""
    if not isinstance(mem, _Reg):
        return _MEMS[mem][off:off + n]
    addr = mem.ptr + off
    for t in _TENSORS:
        p = t.data_ptr()
        if p <= addr and addr + n <= p + t.numel() * t.element_size():
            return _flat(t)[addr - p:addr - p + n]
    raise AssertionError(f"no tensor holds {n} bytes at {addr:#x}")


class _Status:
    def Wait(self):
        pass

    def Succeeded(self):
        return True

    def InProgress(self):
        return False


class _Engine:
    def __init__(self, name, cfg):
        pass

    def create_backend(self, kind, cfg):
        pass

    def register_remote_engine(self, desc):
        pass

    def allocate_transfer_uid(self):
        return 0

    def get_engine_desc(self):
        return types.SimpleNamespace(pack=lambda: b"eng")

    def batch_read(self, ld, lo, rd, ro, sz, uids):
        for l, los, r, ros, sizes in zip(ld, lo, rd, ro, sz):
            for a, b, n in zip(los, ros, sizes):
                _at(l, a, n).copy_(_at(r, b, n))
        return [_Status() for _ in sz]


class _StridedEngine(_Engine):
    def batch_read_strided(self, ld, lo, rd, ro, sz, nr, lp, rp, uids):
        for l, los, r, ros, sizes, nrs, lps, rps in zip(ld, lo, rd, ro, sz, nr, lp, rp):
            for a, b, n, k, pa, pb in zip(los, ros, sizes, nrs, lps, rps):
                for i in range(k):
                    _at(l, a + i * pa, n).copy_(_at(r, b + i * pb, n))
        return [_Status() for _ in sz]


@pytest.fixture(params=[_Engine, _StridedEngine], ids=["plain", "strided"])
def mori(monkeypatch, request):
    io = types.ModuleType("mori.io")
    io.IOEngine = request.param
    io.IOEngineConfig = io.RdmaBackendConfig = io.XgmiBackendConfig = (
        lambda **kw: types.SimpleNamespace(**kw))
    io.BackendType = types.SimpleNamespace(RDMA="rdma", XGMI="xgmi")
    io.EngineDesc = types.SimpleNamespace(unpack=lambda b: b)
    io.MemoryDesc = types.SimpleNamespace(unpack=lambda b: _BY_ID.get(b, b))
    monkeypatch.setitem(sys.modules, "mori", types.ModuleType("mori"))
    monkeypatch.setitem(sys.modules, "mori.io", io)
    for m in (mwt, mn):
        monkeypatch.setattr(m, "_register", lambda eng, ptr, nbytes, device: _Reg(ptr))
    monkeypatch.setattr(mn, "_allocation_of", lambda ptr, segments: (ptr, 0))
    yield request.param is _StridedEngine
    _MEMS.clear()
    _BY_ID.clear()
    _TENSORS.clear()


class _Group:
    """``exchange`` for ranks running in threads of one process."""

    def __init__(self, n):
        self.barrier, self.vals = threading.Barrier(n, timeout=30), [None] * n

    def exchange(self, i):
        def ex(obj):
            self.vals[i] = obj
            self.barrier.wait()
            out = list(self.vals)
            self.barrier.wait()
            return out
        return ex


def test_node_exchange_gathers_across_processes_without_a_group():
    key, n, out, errors = f"test-{os.getpid()}-{next(_ids)}", 3, [None] * 3, []

    def rank(i):
        try:
            ex = mn.NodeExchange(key, n, i, timeout_s=30)
            out[i] = (ex({"rank": i}), ex(i * 10))
        except BaseException as e:  # noqa: BLE001 - re-raised in the test thread
            errors.append(e)

    threads = [threading.Thread(target=rank, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert out == [([{"rank": 0}, {"rank": 1}, {"rank": 2}], [0, 10, 20])] * n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("in_place", [True, False], ids=["in-place", "mirrors-only"])
@pytest.mark.parametrize("permuted", [False, True], ids=["as-loaded", "permuted"])
@pytest.mark.parametrize("rails", [[0, 1], [0, 5]])
def test_node_sync_reads_every_row(mori, monkeypatch, rails, permuted, in_place):
    """With ``permuted``, vLLM transposes ``d0`` (64 x 64 bytes) after loading: the
    reader stages it and moves it into place, as on one node, its bytes fetched
    straight into the staging slot. Otherwise source 0 lands straight in ``d0``.
    With in-place landing off, both go through the mirror."""
    monkeypatch.setattr(mn, "_rails", lambda sources, exchange, device: (
        {s.src_rank: s.rail for s in sources}, list(rails)))
    monkeypatch.setenv("LUMEN_MORI_RDMA_IN_PLACE", "1" if in_place else "0")
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: [])
    gen = torch.Generator().manual_seed(0)
    src = {q: torch.randint(0, 255, (n,), dtype=torch.uint8, generator=gen).cuda()
           for q, n in _SRC_NB.items()}
    for s, q in ((s, q) for s in (0, 1) for q in _SRC_NB if s in _HOLDERS[q]):
        _MEMS[(s, q)] = src[q]
    plan, exports = _plan(), _exports()
    view = {"sizes": [64, 64], "strides": [1, 64], "offset": 0}
    if permuted:
        plan.meta["views"] = {"d0": view}
    group, key = _Group(2), f"test-{os.getpid()}-{next(_ids)}"
    readers, params, errors = [None, None], [None, None], []

    def rank(d):
        try:
            torch.cuda.set_device(0)
            rows = plan.rows[plan.rows[:, 0] == d]
            params[d] = {f"d{q}": torch.zeros(_DST_NB[q], dtype=torch.uint8, device="cuda")
                         for q in rows[:, 3].tolist()}
            _TENSORS.extend(params[d].values())
            r = MoriNodeReader(plan, d, params[d], exports, group=[0, 1], host="h",
                               exchange=group.exchange(d), signal_key=key, window=2,
                               batch=_BIG)
            _TENSORS.extend([r.mirror, *r.reader.slots])
            readers[d] = r
            for _ in range(2):  # trainer weights change between syncs
                group.barrier.wait()
                if d == 0:
                    for t in src.values():
                        t.add_(1)
                    torch.cuda.synchronize()
                group.barrier.wait()
                r.sync()
                group.barrier.wait()
                for row in rows.tolist():
                    if permuted and row[3] == 0:
                        want = src[0].view(64, 64).t().reshape(-1)
                        assert torch.equal(params[d]["d0"], want)
                        continue
                    want = src[row[5]].as_strided((row[8], row[7]), (row[10], 1), row[6])
                    got = params[d][f"d{row[3]}"].as_strided((row[8], row[7]), (row[9], 1),
                                                            row[4])
                    assert torch.equal(got, want), row
            ref = read_hashes(rows, {q: _flat(t) for q, t in src.items()}, 5, 6, 10)
            assert torch.equal(r.hashes(), ref)
        except BaseException as e:  # noqa: BLE001 - re-raised in the test thread
            errors.append(e)
            group.barrier.abort()

    threads = [threading.Thread(target=rank, args=(d,)) for d in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    if errors:
        raise errors[0]
    assert readers[1].in_bytes > 0
    assert sum(r.out_bytes for r in readers) == sum(r.in_bytes for r in readers)
    for r in readers:  # what the vLLM worker logs for every reader
        assert r.strided == mori and r.staging_bytes > 0
    assert set(readers[0].views) == ({0} if permuted else set())
    assert readers[0].land_bytes == (4096 if in_place and not permuted else 0)
    assert readers[0].stage_bytes == (4096 if in_place and permuted else 0)
    assert readers[1].land_bytes == readers[1].stage_bytes == 0
    assert sum(len(r.calls) for r in readers) > 2    # several mirror batches, gated


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("direct", [True, False], ids=["staged", "mirrors-only"])
def test_node_sync_stages_checkpoint_tensors_for_load(mori, monkeypatch, direct):
    """Online FP8: the plan's destinations are checkpoint tensors, staged in slot
    batches and handed to ``load``. Source 0's bytes are fetched straight into the
    slot; the rest come from the mirrors, each batch gated on its rounds."""
    monkeypatch.setattr(mn, "_rails", lambda sources, exchange, device: (
        {s.src_rank: s.rail for s in sources}, [0, 1]))
    monkeypatch.setenv("LUMEN_MORI_RDMA_IN_PLACE", "1" if direct else "0")
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: [])
    gen = torch.Generator().manual_seed(1)
    src = {q: torch.randint(0, 255, (n,), dtype=torch.uint8, generator=gen).cuda()
           for q, n in _SRC_NB.items()}
    for s, q in ((s, q) for s in (0, 1) for q in _SRC_NB if s in _HOLDERS[q]):
        _MEMS[(s, q)] = src[q]
    plan, exports = _plan(), _exports()
    plan.meta = {"round_bytes": 4096, "staged_checkpoint": {
        f"d{q}": {"dtype": "uint8", "shape": [n]} for q, n in _DST_NB.items()}}
    group, key = _Group(2), f"test-{os.getpid()}-{next(_ids)}"
    readers, errors = [None, None], []

    def rank(d):
        try:
            torch.cuda.set_device(0)
            rows = plan.rows[plan.rows[:, 0] == d]
            r = MoriNodeReader(plan, d, {}, exports, group=[0, 1], host="h",
                               exchange=group.exchange(d), signal_key=key, window=2,
                               batch=_BIG)
            _TENSORS.extend([r.mirror, *r.reader.slots])
            readers[d] = r
            assert len(r.reader.batches) > 1
            for _ in range(2):
                group.barrier.wait()
                if d == 0:
                    for t in src.values():
                        t.add_(1)
                    torch.cuda.synchronize()
                group.barrier.wait()
                loaded: dict[str, torch.Tensor] = {}
                st = r.sync(lambda ws: loaded.update(ws), verify=True)
                group.barrier.wait()
                assert st["load_s"] >= 0 and sorted(loaded) == sorted(
                    f"d{q}" for q in set(rows[:, 3].tolist()))
                for row in rows.tolist():
                    want = src[row[5]].as_strided((row[8], row[7]), (row[10], 1), row[6])
                    got = loaded[f"d{row[3]}"].as_strided((row[8], row[7]), (row[9], 1),
                                                          row[4])
                    assert torch.equal(got, want), row
                ref = read_hashes(rows, {q: _flat(t) for q, t in src.items()}, 5, 6, 10)
                assert torch.equal(r.hashes(), ref)
        except BaseException as e:  # noqa: BLE001 - re-raised in the test thread
            errors.append(e)
            group.barrier.abort()

    threads = [threading.Thread(target=rank, args=(d,)) for d in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    if errors:
        raise errors[0]
    assert readers[0].checkpoint and readers[0].land_bytes == 0
    assert readers[0].stage_bytes == (4096 if direct else 0)
    assert readers[1].in_bytes > 0
