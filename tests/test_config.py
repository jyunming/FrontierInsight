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


def test_knowledge_offline_defaults_off(tmp_path: Path) -> None:
    """Default: offline knobs are off / unset so normal machines keep
    fetching models from huggingface.co."""
    cfg = KnowledgeConfig()
    assert cfg.offline is False
    assert cfg.models_dir is None


def test_knowledge_offline_from_yaml(tmp_path: Path) -> None:
    """YAML can pin offline + a local models dir (with ~ expansion)."""
    cfg = KnowledgeConfig.model_validate(
        {"offline": True, "models_dir": "~/fi-models"}
    )
    assert cfg.offline is True
    assert cfg.models_dir == Path("~/fi-models").expanduser()


def test_knowledge_offline_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """``FI_OFFLINE`` / ``FI_MODELS_DIR`` seed the defaults so a machine
    can be configured offline without per-quest YAML."""
    monkeypatch.setenv("FI_OFFLINE", "yes")
    monkeypatch.setenv("FI_MODELS_DIR", str(Path.home() / "m"))
    cfg = KnowledgeConfig()
    assert cfg.offline is True
    assert cfg.models_dir == (Path.home() / "m")


def test_knowledge_yaml_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit ``offline: false`` in input wins over a truthy env."""
    monkeypatch.setenv("FI_OFFLINE", "1")
    cfg = KnowledgeConfig.model_validate({"offline": False})
    assert cfg.offline is False


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
    # top_k bumped 5 → 8 when the Axon vs external caps were split:
    # dense embedding hits are precise, 8 is the new sweet spot.
    assert kc.top_k == 8
    # New independent cap on external (web) search results.
    assert kc.external_top_k == 20
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


def test_top_k_rejects_zero_and_negative(tmp_path: Path) -> None:
    """`top_k` < 1 produces surprising slicing behavior in
    `Knowledge.asearch` (e.g., local-paper slice returns empty even
    though papers are pinned). Pydantic must reject these up-front."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        KnowledgeConfig(top_k=0)
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        KnowledgeConfig(top_k=-3)
    # Boundary: 1 is the minimum.
    assert KnowledgeConfig(top_k=1).top_k == 1


def test_ingest_help_text_does_not_reference_knowledge_enabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The `--ingest` mode bypasses `KnowledgeConfig.enabled` entirely
    (it goes through a one-off Knowledge instance). Earlier the help
    text said `Requires knowledge.enabled: true in --axon-config`,
    which is wrong on two counts: --axon-config is an AxonConfig YAML
    (no `knowledge.*` namespace), and the flag bypasses that check
    anyway. Pin the corrected wording."""
    from launch import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--help"])
    help_text = capsys.readouterr().out
    # Locate the --ingest section.
    assert "--ingest" in help_text
    assert "knowledge.enabled" not in help_text
    assert "axon" in help_text.lower()


def test_pauses_namespace_legacy_mapping_and_norway() -> None:
    """The unified `pauses:` namespace: legacy flags map in, value vocab is
    translated, and the YAML 1.1 'Norway problem' (unquoted `off` → bool
    False) is coerced back so hand-edited blocks don't need quotes."""
    from core.config import Config

    # Legacy scattered flags fold into pauses with vocab translation.
    legacy = Config.model_validate({
        "topic": "t",
        "engine": {"clarify_mode": "interactive", "human_feedback_gate": "after_review",
                   "pause_for_user_input": "after_design"},
        "knowledge": {"pause_for_user_papers": True},
    })
    assert legacy.pauses.clarify == "ask"
    assert legacy.pauses.review == "ask"
    assert legacy.pauses.supply == "before_build"
    assert legacy.pauses.papers is True

    # Unquoted `off` arrives as boolean False from YAML; coerce to "off".
    norway = Config.model_validate({
        "topic": "t", "pauses": {"clarify": False, "review": False},
    })
    assert norway.pauses.clarify == "off"
    assert norway.pauses.review == "off"

    # An explicit pauses.* value always wins over a legacy flag.
    conflict = Config.model_validate({
        "topic": "t", "engine": {"clarify_mode": "auto"},
        "pauses": {"clarify": "ask"},
    })
    assert conflict.pauses.clarify == "ask"
