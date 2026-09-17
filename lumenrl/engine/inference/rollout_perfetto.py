"""Opt-in Chrome/Perfetto traces of Ray rollout generate().

Set ``LUMENRL_ROLLOUT_PERFETTO_DIR`` to an absolute directory. ATOM writes
``*.pt.trace.json.gz`` via ``torch_profiler_dir`` + ``LLMEngine.start_profile``.
vLLM writes torch profiler dumps via ``profiler_config`` +
``AsyncLLM.start_profile``. Open the gzipped chrome traces in
https://ui.perfetto.dev.

``LUMENRL_ROLLOUT_PERFETTO_REPLICA`` selects which replica traces (default
``0``). Use ``all`` for every replica — that multiplies disk by DP.
"""

from __future__ import annotations

import os
from pathlib import Path


def rollout_perfetto_dir() -> str | None:
    raw = os.environ.get("LUMENRL_ROLLOUT_PERFETTO_DIR", "").strip()
    return raw or None


def rollout_perfetto_enabled() -> bool:
    return rollout_perfetto_dir() is not None


def replica_should_trace(replica_rank: int) -> bool:
    if not rollout_perfetto_enabled():
        return False
    sel = os.environ.get("LUMENRL_ROLLOUT_PERFETTO_REPLICA", "0").strip().lower()
    if sel in {"all", "*", "-1"}:
        return True
    try:
        return int(sel) == int(replica_rank)
    except ValueError:
        return int(replica_rank) == 0


def replica_trace_dir(backend: str, replica_rank: int) -> str | None:
    root = rollout_perfetto_dir()
    if root is None:
        return None
    path = Path(root) / backend / f"replica{int(replica_rank)}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
