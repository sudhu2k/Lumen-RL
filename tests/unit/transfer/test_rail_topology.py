import pytest

from lumenrl.transfer import rail_topology as rt
from lumenrl.transfer.rail_topology import Nic, match_nics, rails_of


def _flat():
    """A VM that shows each NUMA node's 4 GPUs and 4 NICs under one root, plus a slower
    management NIC: the hop counts tie, so MORI's match falls back to NIC names."""
    gpus = [f"000{d}:00:0{i}.0" for d in (2, 3) for i in (1, 2, 3, 4)]
    nics = [Nic(f"ionic_{k}", f"000{2 + k // 4}:00:0{'9abc'[k % 4]}.0", 400.0, k // 4)
            for k in range(8)] + [Nic("mlx5_0", "0000:00:03.0", 100.0, 0)]
    paths = {b: (f"pci{b[:7]}", b) for b in gpus + [n.bus for n in nics]}
    numa = {b: 0 if b.startswith(("0002", "0000")) else 1 for b in paths}
    return gpus, nics, paths, numa


def _switched():
    """Bare metal: GPU k and NIC k behind their own PCIe switch."""
    gpus = [f"0000:{k:x}5:00.0" for k in range(8)]
    nics = [Nic(f"ionic_{k}", f"0000:{k:x}6:00.0", 400.0, k // 4) for k in range(8)]
    paths = {}
    for k in range(8):
        root, sw = f"pci0000:{k:x}0", f"0000:{k:x}1:00.0"
        paths[gpus[k]] = (root, sw, gpus[k])
        paths[nics[k].bus] = (root, sw, nics[k].bus)
    numa = {b: 0 if int(b[5], 16) < 4 else 1 for b in paths}
    return gpus, nics, paths, numa


def test_bus_order_gives_every_gpu_its_own_nic_on_a_flat_vm():
    gpus, nics, paths, numa = _flat()
    assert match_nics(gpus, nics, paths, numa) == [f"ionic_{k}" for k in range(8)]


def test_mori_matches_by_visible_position_when_hops_tie():
    gpus, nics, paths, numa = _flat()
    order = [0, 2, 1, 3, 7, 4, 5, 6]                    # vLLM's devices in a DAPO run
    visible = [gpus[i] for i in order]
    assert match_nics(visible, nics, paths, numa) == [f"ionic_{k}" for k in range(8)]
    r = rails_of(nics, gpus[2], visible, gpus, paths, numa)   # rank 1, on GPU 2
    assert (r.nic, r.rail, r.own_nic, r.own_rail) == ("ionic_1", 1, "ionic_2", 2)


def test_hops_pick_the_own_nic_in_any_order_on_bare_metal():
    gpus, nics, paths, numa = _switched()
    visible = [gpus[i] for i in (0, 2, 1, 3, 7, 4, 5, 6)]
    for g in visible:
        r = rails_of(nics, g, visible, gpus, paths, numa)
        assert r.nic == r.own_nic == f"ionic_{gpus.index(g)}" and r.rail == r.own_rail


def test_a_process_with_one_gpu_takes_the_best_nic_and_its_own_is_known():
    gpus, nics, paths, numa = _flat()
    r = rails_of(nics, gpus[5], [gpus[5]], gpus, paths, numa)
    assert (r.nic, r.own_nic) == ("ionic_4", "ionic_5")      # first numa-1 NIC by name


def test_a_mori_matching_over_the_host_gives_every_process_its_own_nic(monkeypatch):
    monkeypatch.setenv("LUMEN_MORI_NIC_MATCH", "node")
    gpus, nics, paths, numa = _flat()
    for visible in ([gpus[5]], [gpus[2], gpus[3]], [gpus[i] for i in (0, 2, 1, 3, 7, 4, 5, 6)]):
        for g in visible:
            r = rails_of(nics, g, visible, gpus, paths, numa)
            assert r.nic == r.own_nic == f"ionic_{gpus.index(g)}" and r.rail == r.own_rail


def test_rails_skip_nics_without_an_active_port():
    gpus, nics, paths, numa = _flat()
    nics = [Nic(n.name, n.bus, 0.0 if n.name == "ionic_0" else n.gbps, n.numa) for n in nics]
    r = rails_of(nics, gpus[0], gpus, gpus, paths, numa)
    assert (r.nic, r.rail) == ("ionic_1", 0)


def _sysfs(root, gpus, nics):
    def dev(bus, **files):
        d = root / "sys/devices" / f"pci{bus[:7]}" / bus
        d.mkdir(parents=True)
        for k, v in files.items():
            (d / k).write_text(v + "\n")
        link = root / "sys/bus/pci/devices" / bus
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(d)
        return d

    for bus, numa in gpus:
        dev(bus, vendor="0x1002", **{"class": "0x120000", "numa_node": str(numa)})
    dev("0002:00:05.0", vendor="0x1344", **{"class": "0x010802", "numa_node": "0"})  # NVMe
    for name, bus, numa, state in nics:
        d = dev(bus, vendor="0x1dd8", **{"class": "0x020000", "numa_node": str(numa)})
        ib = root / "sys/class/infiniband" / name
        (ib / "ports/1").mkdir(parents=True)
        (ib / "device").symlink_to(d)
        (ib / "ports/1/state").write_text(state + "\n")
        (ib / "ports/1/rate").write_text("400 Gb/sec (4X NDR)\n")


def test_gpu_rail_reads_sysfs(tmp_path, monkeypatch):
    gpus = [("0002:00:01.0", 0), ("0002:00:02.0", 0)]
    _sysfs(tmp_path, gpus, [("ionic_0", "0002:00:09.0", 0, "4: ACTIVE"),
                            ("ionic_1", "0002:00:0a.0", 0, "4: ACTIVE"),
                            ("ionic_2", "0002:00:0b.0", 0, "1: DOWN")])
    root = str(tmp_path)
    assert rt._node_gpus(root) == ["0002:00:01.0", "0002:00:02.0"]
    assert [(n.name, n.gbps) for n in rt._nics(root)] == [
        ("ionic_0", 400.0), ("ionic_1", 400.0), ("ionic_2", 0.0)]
    assert rt._pci_path("0002:00:0a.0", root) == ("pci0002:00", "0002:00:0a.0")
    monkeypatch.setattr(rt, "_visible_gpus", lambda: ["0002:00:02.0", "0002:00:01.0"])
    rt.gpu_rail.cache_clear()
    r = rt.gpu_rail(0, root)
    assert (r.bus, r.nic, r.rail, r.own_nic, r.own_rail) == (
        "0002:00:02.0", "ionic_0", 0, "ionic_1", 1)
    rt.gpu_rail.cache_clear()


@pytest.mark.parametrize("spec,expect", [
    ("ionic_5", ["ionic_5"]),
    ("ionic_5, ionic_6", ["ionic_5", "ionic_6"]),
    ("^ionic_0,mlx5_0", [f"ionic_{k}" for k in range(1, 8)]),
    ("", [f"ionic_{k}" for k in range(8)] + ["mlx5_0"]),
    (None, [f"ionic_{k}" for k in range(8)] + ["mlx5_0"]),
])
def test_mori_allowed_follows_mori_rdma_devices(spec, expect):
    _, nics, _, _ = _flat()
    assert [n.name for n in rt.mori_allowed(nics, spec)] == expect


def test_mori_rdma_devices_gives_a_one_gpu_process_its_own_nic_and_keeps_rails():
    gpus, nics, paths, numa = _flat()
    r = rails_of(nics, gpus[5], [gpus[5]], gpus, paths, numa, "ionic_5")
    assert (r.nic, r.rail, r.own_nic, r.own_rail) == ("ionic_5", 5, "ionic_5", 5)


def test_gpu_rail_is_none_without_rdma_nics(tmp_path):
    rt.gpu_rail.cache_clear()
    assert rt.gpu_rail(0, str(tmp_path)) is None


@pytest.mark.parametrize("var", ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"])
def test_visible_order_names_the_variable(monkeypatch, var):
    for v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(var, "0,2,1")
    assert rt.visible_order() == f"{var}=0,2,1"
