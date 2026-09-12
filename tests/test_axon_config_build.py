"""FI must build an ``AxonConfig`` through Axon's real API.

``AxonConfig`` is a flat **dataclass**, not a pydantic model. FI called
``AxonConfig.model_validate(...)`` for an inline mapping and
``AxonConfig.from_yaml(...)`` for a path — neither method exists — so *every*
quest that set ``knowledge.axon_config`` died at engine construction with
``AttributeError: type object 'AxonConfig' has no attribute 'model_validate'``.
Only the ``axon_config is None`` branch ever worked, which is why it survived:
the shipped ``examples/integrator_bakeoff/config.yaml`` sets the key and could
never have run.

Two layers of cover, because they fail for different reasons:

* The stub tests pin the *API surface* FI is allowed to use. They run
  everywhere, including CI, which installs ``.[dev,docker]`` and therefore has
  no Axon at all — the environment where a real-Axon test silently skips and
  proves nothing.
* The real-Axon test checks the nested→flat mapping actually happens, which a
  stub cannot tell us.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import core.knowledge as K


class _StubAxonConfig:
    """Mirrors the real class's surface: constructed from a *file* via
    ``load``. Deliberately defines neither ``model_validate`` nor
    ``from_yaml``, so reintroducing either call fails here."""

    calls: list[dict[str, Any]] = []
    paths: list[str] = []

    @classmethod
    def load(cls, path: str | None = None) -> "_StubAxonConfig":
        assert path is not None, "FI should never fall back to Axon's user config"
        cls.paths.append(path)
        cls.calls.append(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
        return cls()


@pytest.fixture
def _stub(monkeypatch: pytest.MonkeyPatch) -> None:
    _StubAxonConfig.calls = []
    _StubAxonConfig.paths = []
    monkeypatch.setattr(K, "AxonConfig", _StubAxonConfig)


def test_inline_mapping_is_handed_to_load_with_nesting_intact(_stub: None) -> None:
    """Axon's YAML is nested; its dataclass is flat. The nested→flat mapping
    lives inside ``load``, so FI must pass the nesting through rather than
    splat the mapping into the constructor."""
    spec = {
        "embedding": {"provider": "ollama", "model": "nomic-embed-text"},
        "llm": {"provider": "ollama", "model": "gemma4:31b-cloud"},
    }
    K._axon_config_from(spec)
    assert len(_StubAxonConfig.calls) == 1
    assert _StubAxonConfig.calls[0] == spec, "nesting must survive the round-trip"


def test_temp_file_is_cleaned_up(_stub: None) -> None:
    K._axon_config_from({"embedding": {"provider": "ollama"}})
    tmp = Path(_StubAxonConfig.paths[0])
    assert not tmp.exists(), f"left a temp config behind at {tmp}"


def test_a_path_is_passed_straight_through(_stub: None, tmp_path: Path) -> None:
    """A path needs no round-trip — it is already the file ``load`` wants."""
    cfg = tmp_path / "axon.yaml"
    cfg.write_text("embedding:\n  provider: ollama\n", encoding="utf-8")
    K._axon_config_from(cfg)
    assert _StubAxonConfig.paths == [str(cfg)]


def test_a_string_path_is_also_accepted(_stub: None, tmp_path: Path) -> None:
    cfg = tmp_path / "axon.yaml"
    cfg.write_text("llm:\n  provider: ollama\n", encoding="utf-8")
    K._axon_config_from(str(cfg))
    assert _StubAxonConfig.paths == [str(cfg)]


def test_unicode_survives_the_round_trip(_stub: None) -> None:
    """``yaml.safe_dump`` defaults to ASCII escaping; a path or model name
    with non-ASCII characters must not be mangled on the way to Axon."""
    spec = {"llm": {"provider": "ollama", "model": "模型-α"}}
    K._axon_config_from(spec)
    assert _StubAxonConfig.calls[0]["llm"]["model"] == "模型-α"


@pytest.mark.parametrize("missing", ["model_validate", "from_yaml"])
def test_fi_does_not_depend_on_pydantic_style_constructors(_stub: None, missing: str) -> None:
    """The exact regression: these are the methods FI used to call. If the
    real Axon ever grows them this test still holds — FI should not rely on
    them, because the installed Axon is a dataclass."""
    assert not hasattr(_StubAxonConfig, missing)
    K._axon_config_from({"embedding": {"provider": "ollama"}})  # must not raise


# --- the part a stub cannot prove -----------------------------------------

def test_real_axon_maps_nested_yaml_onto_flat_fields() -> None:
    """Skipped where Axon is absent (CI). Locally this is the check that the
    stub tests cannot make: that ``load`` really does turn
    ``embedding: {provider: ...}`` into ``embedding_provider``."""
    pytest.importorskip("axon", reason="Axon is an optional extra, absent in CI")
    from core.knowledge import _axon_config_from

    ac = _axon_config_from({
        "embedding": {"provider": "ollama", "model": "nomic-embed-text"},
        "llm": {"provider": "ollama", "model": "gemma4:31b-cloud"},
    })
    assert ac.embedding_provider == "ollama"
    assert ac.embedding_model == "nomic-embed-text"
    assert ac.llm_provider == "ollama"
    assert ac.llm_model == "gemma4:31b-cloud"
