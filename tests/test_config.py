"""Schema and path-handling tests for core/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.config import Config


def write_cfg(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(body), encoding="utf-8")
    return p


def test_minimal_config_loads(tmp_path: Path) -> None:
    cfg = Config.from_yaml(
        write_cfg(tmp_path, {"topic": "test topic", "title": "test-title"})
    )
    assert cfg.topic == "test topic"
    assert cfg.title == "test-title"
    assert cfg.provider.name == "codex"
    assert cfg.engine.framework == "langgraph"
    assert cfg.execution.sandbox == "venv"
    assert cfg.knowledge.enabled is True
    assert cfg.output.kinds == ["paper_md", "paper_pdf"]


def test_tilde_expansion_in_output_dir(tmp_path: Path) -> None:
    cfg = Config.from_yaml(
        write_cfg(
            tmp_path,
            {
                "topic": "t",
                "output": {"output_dir": "~/some-output-dir"},
            },
        )
    )
    assert "~" not in str(cfg.output.output_dir)
    assert cfg.output.output_dir.is_absolute() or str(cfg.output.output_dir).startswith(("/", "C:", "~")) is False


def test_axon_config_inline_dict(tmp_path: Path) -> None:
    cfg = Config.from_yaml(
        write_cfg(
            tmp_path,
            {
                "topic": "t",
                "knowledge": {
                    "enabled": True,
                    "axon_config": {"embedding": {"provider": "ollama"}},
                    "top_k": 7,
                },
            },
        )
    )
    assert isinstance(cfg.knowledge.axon_config, dict)
    assert cfg.knowledge.top_k == 7


def test_invalid_provider_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        Config.from_yaml(
            write_cfg(tmp_path, {"topic": "t", "provider": {"name": "not-a-provider"}})
        )


def test_invalid_sandbox_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        Config.from_yaml(
            write_cfg(tmp_path, {"topic": "t", "execution": {"sandbox": "wasm"}})
        )
