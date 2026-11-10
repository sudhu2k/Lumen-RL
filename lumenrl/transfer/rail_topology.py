"""Which NIC MORI's RDMA backend reads a GPU's buffers through, from the PCIe topology.

A rail is a NIC position: on a rail-only fabric NIC k of one node reaches only NIC k
of the others. MORI picks the NIC of a GPU buffer as ``TopoSystem::MatchAllGpusAndNics``
does: it walks the process's visible GPUs in order and gives each the best NIC no
earlier GPU took (fastest, then on the GPU's NUMA node, then fewest PCIe hops, then by
name). A reader's NIC therefore follows its device's position in ``*_VISIBLE_DEVICES``,
not which physical GPU it is; with ``MORI_IO_RAIL_AFFINITY=1`` the passive side answers
on the NIC with the same index. A GPU's own NIC is the one that match gives it with
every GPU of the node visible in bus order; a NIC reading another GPU's memory crosses
the host's PCIe fabric, which hypervisors may hide (all devices under one root).

MORI only considers the NICs ``MORI_RDMA_DEVICES`` allows (a comma list, or ``^`` and a
list to exclude). ``pin_own_nic`` narrows it to a reader's own NIC when the match above
would give it another one, as for a process that sees one GPU of the node.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

__all__ = ["Nic", "GpuRail", "match_nics", "mori_allowed", "rails_of", "gpu_rail",
           "pin_own_nic", "visible_order"]

_GPU_CLASSES = ("0x03", "0x12")  # display, processing accelerator
_AMD = "0x1002"


@dataclass(frozen=True)
class Nic:
    name: str
    bus: str
    gbps: float  # over its active ports; 0 when none is active
    numa: int


@dataclass(frozen=True)
class GpuRail:
    """A process's GPU: its PCI address, the NIC MORI reads its buffers through and
    that NIC's rail, and the GPU's own NIC and rail."""

    bus: str
    nic: str
    rail: int
    own_nic: str
    own_rail: int

    def to_wire(self) -> dict:
        return asdict(self)


def _hops(a: Sequence[str], b: Sequence[str]) -> int:
    """PCIe hops between two devices given their paths from the root: the devices on
    the path through their lowest common ancestor, less the two ends."""
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    if not common:
        return len(a) + len(b)
    return max(len(a) + len(b) - 2 * common + 1 - 2, 0)


def match_nics(gpus: Sequence[str], nics: Sequence[Nic], paths: Mapping[str, Sequence[str]],
               numa: Mapping[str, int]) -> list[str]:
    """The NIC MORI gives each of ``gpus`` (PCI addresses, in visible-device order);
    ``paths`` maps a PCI address to its path from the root, ``numa`` to its node."""
    taken: set[str] = set()
    out = []
    for g in gpus:
        cands = sorted((n for n in nics if n.bus in paths and g in paths),
                       key=lambda n: (-n.gbps, n.numa != numa.get(g, -1),
                                      _hops(paths[g], paths[n.bus]), n.name))
        free = [n.name for n in cands if n.name not in taken]
        if free:
            taken.add(free[0])
            out.append(free[0])
        else:
            out.append(cands[len(out) % len(cands)].name if cands else "")
    return out


def _read(path: str, default: str = "") -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def _pci_path(bus: str, root: str) -> tuple[str, ...]:
    """``bus``'s ancestors and itself from the root complex, from its sysfs device path."""
    base = os.path.realpath(os.path.join(root, "sys/devices"))
    real = os.path.realpath(os.path.join(root, "sys/bus/pci/devices", bus))
    if not real.startswith(base + os.sep):
        return ()
    return tuple(os.path.relpath(real, base).split(os.sep))


def _nics(root: str) -> list[Nic]:
    base = os.path.join(root, "sys/class/infiniband")
    out = []
    for name in sorted(os.listdir(base)) if os.path.isdir(base) else ():
        dev = os.path.realpath(os.path.join(base, name, "device"))
        bus = os.path.basename(dev)
        gbps = 0.0
        ports = os.path.join(base, name, "ports")
        for p in sorted(os.listdir(ports)) if os.path.isdir(ports) else ():
            if "ACTIVE" in _read(os.path.join(ports, p, "state")):
                rate = _read(os.path.join(ports, p, "rate"), "0").split()
                gbps += float(rate[0]) if rate else 0.0
        out.append(Nic(name, bus, gbps, int(_read(os.path.join(dev, "numa_node"), "-1"))))
    return out


def _node_gpus(root: str) -> list[str]:
    base = os.path.join(root, "sys/bus/pci/devices")
    return sorted(b for b in (os.listdir(base) if os.path.isdir(base) else ())
                  if _read(os.path.join(base, b, "vendor")) == _AMD
                  and _read(os.path.join(base, b, "class")).startswith(_GPU_CLASSES))


def _visible_gpus() -> list[str]:
    """PCI addresses of this process's GPUs, in device order."""
    import torch

    hip = ctypes.CDLL("libamdhip64.so")
    out = []
    for i in range(torch.cuda.device_count()):
        buf = ctypes.create_string_buffer(32)
        err = hip.hipDeviceGetPCIBusId(buf, ctypes.c_int(len(buf)), ctypes.c_int(i))
        if err != 0:
            raise OSError(f"hipDeviceGetPCIBusId({i}) failed: hipError {err}")
        out.append(buf.value.decode().lower())
    return out


def mori_allowed(nics: Sequence[Nic], spec: str | None) -> list[Nic]:
    """The NICs MORI's RDMA backend sees under ``MORI_RDMA_DEVICES=spec``."""
    names = {s.strip() for s in (spec or "").lstrip("^").split(",") if s.strip()}
    if not names:
        return list(nics)
    exclude = (spec or "").startswith("^")
    return [n for n in nics if (n.name in names) != exclude]


def rails_of(nics: Sequence[Nic], gpu: str, visible: Sequence[str], node: Sequence[str],
             paths: Mapping[str, Sequence[str]], numa: Mapping[str, int],
             mori_devices: str | None = None) -> GpuRail:
    """``gpu``'s NICs and rails, ``visible`` the process's GPUs in device order and
    ``node`` every GPU of the node; ``mori_devices`` is ``MORI_RDMA_DEVICES``. Rails
    number the NICs with an active port by name, the same on every node of the same
    build."""
    active = sorted(n.name for n in nics if n.gbps > 0)
    seen = mori_allowed(nics, mori_devices)
    nic = match_nics(visible, seen, paths, numa)[list(visible).index(gpu)]
    own = match_nics(node, nics, paths, numa)[list(node).index(gpu)] if gpu in node else ""
    return GpuRail(gpu, nic, active.index(nic) if nic in active else -1,
                   own, active.index(own) if own in active else -1)


@functools.lru_cache(maxsize=None)
def gpu_rail(device: int, root: str = "/") -> GpuRail | None:
    """This process's GPU ``device``'s NICs and rails; None when the GPUs or NICs cannot
    be read (no RDMA devices, no sysfs)."""
    try:
        nics = _nics(root)
        if not nics:
            return None
        visible, node = _visible_gpus(), _node_gpus(root)
        buses = {*visible, *node, *(n.bus for n in nics)}
        paths = {b: _pci_path(b, root) for b in buses}
        paths = {b: p for b, p in paths.items() if p}
        numa = {b: int(_read(os.path.join(root, "sys/bus/pci/devices", b, "numa_node"), "-1"))
                for b in buses}
        return rails_of(nics, visible[int(device)], visible, node, paths, numa,
                        os.environ.get("MORI_RDMA_DEVICES"))
    except (OSError, ValueError, IndexError) as e:
        logger.warning("rail topology unavailable for GPU %s: %s", device, e)
        return None


def pin_own_nic(device: int, root: str = "/") -> str | None:
    """Limit this process's MORI RDMA backend to GPU ``device``'s own NIC when MORI would
    read its buffers through another one; returns the NIC pinned, else None. Only for
    readers: a passive source answers on the reader's rail and needs every NIC. Must run
    before MORI's first RDMA engine in the process, which reads the variable once. A
    ``MORI_RDMA_DEVICES`` already set is narrowed to the own NIC if it allows it, else
    left alone; ``LUMEN_MORI_PIN_NIC=0`` turns this off."""
    if os.environ.get("LUMEN_MORI_PIN_NIC") == "0":
        return None
    topo = gpu_rail(device, root)
    if topo is None or not topo.own_nic or topo.nic == topo.own_nic:
        return None
    spec = os.environ.get("MORI_RDMA_DEVICES")
    if spec and topo.own_nic not in {n.name for n in mori_allowed(_nics(root), spec)}:
        logger.warning("GPU %s's own NIC %s is excluded by MORI_RDMA_DEVICES=%s; MORI reads "
                       "its buffers through %s", topo.bus, topo.own_nic, spec, topo.nic)
        return None
    os.environ["MORI_RDMA_DEVICES"] = topo.own_nic
    gpu_rail.cache_clear()
    logger.info("MORI RDMA limited to %s, GPU %s's own NIC, instead of %s (%s)",
                topo.own_nic, topo.bus, topo.nic, visible_order())
    return topo.own_nic


def visible_order() -> str:
    """The visible-device list this process was started with, for messages."""
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if os.environ.get(var, "").strip():
            return f"{var}={os.environ[var].strip()}"
    return "all GPUs visible"
