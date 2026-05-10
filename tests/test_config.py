"""Schema and path-handling tests for core/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.config import Config, KnowledgeConfig, OutputConfig


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


@pytest.mark.parametrize(
    "patch",
    [
        # Each entry exercises a Literal alphabet that must stay in lock-step
        # with code elsewhere in the repo (see id for the cross-reference).
        pytest.param(
            {"provider": {"name": "not-a-provider"}}, id="provider-vs-provider.py",
        ),
        pytest.param(
            {"execution": {"sandbox": "wasm"}}, id="sandbox-vs-make_executor",
        ),
        pytest.param(
            {"engine": {"framework": "autogen"}}, id="framework-vs-engine.py",
        ),
        pytest.param(
            {"output": {"paper_format": "acm"}}, id="paper_format-vs-templates",
        ),
        pytest.param(
            {"output": {"kinds": ["paper_md", "video"]}},
            id="kinds-vs-generation",
        ),
    ],
)
def test_invalid_literal_rejected(tmp_path: Path, patch: dict) -> None:
    body = {"topic": "t", **patch}
    with pytest.raises(Exception):
        Config.from_yaml(write_cfg(tmp_path, body))


def test_axon_config_path_string_expanded(tmp_path: Path) -> None:
    """When `axon_config` is a string, the validator coerces to Path and
    expands `~`. The Knowledge layer's `_build_brain` dispatches on
    `isinstance(..., Path)`, so this branch must hit Path."""
    cfg = Config.from_yaml(
        write_cfg(
            tmp_path,
            {
                "topic": "t",
                "knowledge": {"axon_config": "~/axon-config.yaml"},
            },
        )
    )
    assert isinstance(cfg.knowledge.axon_config, Path)
    assert "~" not in str(cfg.knowledge.axon_config)


def test_axon_config_path_object_expanded() -> None:
    """Direct Python construction with a `Path` containing `~` — the
    validator should still expand it (mirrors the output_dir behaviour)."""
    kc = KnowledgeConfig(axon_config=Path("~/axon.yaml"))
    assert isinstance(kc.axon_config, Path)
    assert "~" not in str(kc.axon_config)


def test_axon_config_none_default() -> None:
    """Default `axon_config` is None — no validator coercion."""
    kc = KnowledgeConfig()
    assert kc.axon_config is None
    assert kc.enabled is True
    assert kc.top_k == 5
    assert kc.write_back_quests is True


def test_output_config_defaults_roundtrip() -> None:
    """OutputConfig() defaults should survive a model_dump → model_validate
    round trip without changing field values."""
    oc = OutputConfig()
    dumped = oc.model_dump()
    reconstructed = OutputConfig.model_validate(dumped)
    assert reconstructed.kinds == oc.kinds == ["paper_md", "paper_pdf"]
    assert reconstructed.paper_format == oc.paper_format == "generic"
    assert reconstructed.output_dir == oc.output_dir == Path("./outputs")


def test_full_config_yaml_roundtrip_stable(tmp_path: Path) -> None:
    """Round-trip through YAML → model → dump → model again should be a
    fixed point on the second pass (the first pass may add defaults)."""
    cfg1 = Config.from_yaml(
        write_cfg(
            tmp_path,
            {
                "topic": "round-trip topic",
                "title": "rt-title",
                "provider": {"name": "ollama", "model": "qwen2.5"},
                "execution": {"sandbox": "venv", "timeout_s": 900},
                "knowledge": {"top_k": 3},
                "output": {
                    "kinds": ["paper_md", "slides"],
                    "paper_format": "neurips",
                    "output_dir": "./out",
                },
            },
        )
    )
    dumped1 = cfg1.model_dump(mode="json")
    cfg2 = Config.model_validate(dumped1)
    dumped2 = cfg2.model_dump(mode="json")
    assert dumped1 == dumped2
    # Field-level cross-checks to catch silent coercion drift.
    assert cfg2.provider.name == "ollama"
    assert cfg2.execution.sandbox == "venv"
    assert cfg2.execution.timeout_s == 900
    assert cfg2.output.paper_format == "neurips"
    assert cfg2.output.kinds == ["paper_md", "slides"]
    assert cfg2.knowledge.top_k == 3
