from lumenrl.engine.inference.rollout_perfetto import (
    replica_should_trace,
    replica_trace_dir,
    rollout_perfetto_dir,
    rollout_perfetto_enabled,
)


def test_perfetto_off_by_default(monkeypatch):
    monkeypatch.delenv("LUMENRL_ROLLOUT_PERFETTO_DIR", raising=False)
    assert rollout_perfetto_dir() is None
    assert not rollout_perfetto_enabled()
    assert not replica_should_trace(0)
    assert replica_trace_dir("atom", 0) is None


def test_perfetto_replica_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("LUMENRL_ROLLOUT_PERFETTO_DIR", str(tmp_path))
    monkeypatch.delenv("LUMENRL_ROLLOUT_PERFETTO_REPLICA", raising=False)
    assert rollout_perfetto_enabled()
    assert replica_should_trace(0)
    assert not replica_should_trace(1)
    path = replica_trace_dir("atom", 0)
    assert path.endswith("/atom/replica0")


def test_perfetto_all_replicas(monkeypatch, tmp_path):
    monkeypatch.setenv("LUMENRL_ROLLOUT_PERFETTO_DIR", str(tmp_path))
    monkeypatch.setenv("LUMENRL_ROLLOUT_PERFETTO_REPLICA", "all")
    assert replica_should_trace(0)
    assert replica_should_trace(7)
