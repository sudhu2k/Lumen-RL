from __future__ import annotations

from pathlib import Path

import pytest

from lumenrl.core.config import LumenRLConfig
from lumenrl.moe.r3_scope import (
    SUPPORTED_R3_GENERATION_BACKEND,
    SUPPORTED_R3_REPLAY_MODE,
    assert_supported_r3,
    r3_fields_from_config,
)


def test_supported_constants() -> None:
    assert SUPPORTED_R3_GENERATION_BACKEND == "vllm"
    assert SUPPORTED_R3_REPLAY_MODE == "hard_assignment"


def test_enabled_vllm_hard_assignment_ok() -> None:
    assert_supported_r3(
        replay_mode="hard_assignment",
        generation_backend="vllm",
    )


def test_enabled_atom_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="generation_backend=vllm"):
        assert_supported_r3(
            replay_mode="hard_assignment",
            generation_backend="atom",
        )


def test_enabled_distribution_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="hard_assignment"):
        assert_supported_r3(
            replay_mode="distribution",
            generation_backend="vllm",
        )


def test_r3_fields_from_nested_mapping() -> None:
    enabled, mode, backend = r3_fields_from_config(
        {
            "policy": {"generation_backend": "vllm"},
            "moe": {"r3": {"enabled": True, "replay_mode": "hard_assignment"}},
        }
    )
    assert (enabled, mode, backend) == (True, "hard_assignment", "vllm")


def test_from_yaml_atom_plus_r3_fail_closed() -> None:
    yaml_path = Path(__file__).resolve().parents[3] / "configs" / "grpo_dense_bf16.yaml"
    with pytest.raises(RuntimeError, match="generation_backend=vllm"):
        LumenRLConfig.from_yaml(yaml_path, overrides=["moe.r3.enabled=true"])


def test_from_yaml_atom_without_r3_loads() -> None:
    yaml_path = Path(__file__).resolve().parents[3] / "configs" / "grpo_dense_bf16.yaml"
    cfg = LumenRLConfig.from_yaml(yaml_path)
    assert cfg.moe.r3.enabled is False
    assert cfg.policy.generation_backend == "atom"


def test_default_lumenrl_config_atom_without_r3() -> None:
    cfg = LumenRLConfig()
    assert cfg.policy.generation_backend == "atom"
    assert cfg.moe.r3.enabled is False
