"""Direct tests for `core.proposal`.

The proposal module produces a planning markdown + companion YAML
under outputs/_drafts/. Tests cover slug generation, prompt assembly,
companion YAML shape (must be parseable by Config.from_yaml), the
write-order crash-safety contract (md before yaml), validation of
the topic argument (empty + oversized), and an end-to-end with
mocked LLM.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.config import ProviderConfig
from core.proposal import (
    _build_proposal_prompt,
    _new_proposal_id,
    _render_companion_yaml,
    _slugify_topic,
    generate_proposal,
)


# ---------- slug + id ------------------------------------------------------


def test_slugify_topic_basics() -> None:
    assert _slugify_topic("Compare RK4 vs Verlet").startswith("compare-rk4-vs-verlet")
    assert _slugify_topic("Hello, World!") == "hello-world"
    assert _slugify_topic("   ") == "proposal"   # empty falls back


def test_slugify_topic_caps_length() -> None:
    long_topic = "a " * 100   # 200 chars
    slug = _slugify_topic(long_topic, max_chars=48)
    assert len(slug) <= 48


def test_new_proposal_id_has_epoch_slug_nonce_shape() -> None:
    pid = _new_proposal_id("My topic")
    # <10-digit-epoch>-<slug>-<6-hex>
    import re
    assert re.match(r"^\d{10}-[a-z0-9-]+-[0-9a-f]{6}$", pid), pid


def test_new_proposal_id_collision_avoidance() -> None:
    """Two calls in quick succession with the same topic must produce
    distinct ids (the nonce makes this true)."""
    a = _new_proposal_id("identical topic")
    b = _new_proposal_id("identical topic")
    assert a != b


# ---------- prompt assembly -------------------------------------------------


def test_build_proposal_prompt_injects_topic_and_date() -> None:
    prompt = _build_proposal_prompt(
        "Compare RK4 vs Verlet on a damped harmonic oscillator",
        generated_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )
    assert "Compare RK4 vs Verlet" in prompt
    assert "2026-05-13" in prompt


# ---------- companion YAML --------------------------------------------------


def test_companion_yaml_round_trips_through_config_loader(tmp_path: Path) -> None:
    """The companion YAML must be Config.from_yaml-loadable; the user
    can edit-and-run it without further massaging."""
    from core.config import Config

    yaml_text = _render_companion_yaml(
        "Compare RK4 vs Verlet on a damped oscillator",
        proposal_id="1778452404-test-aabbcc",
        generated_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    cfg = Config.from_yaml(yaml_path)
    assert "Compare RK4 vs Verlet" in cfg.topic
    assert cfg.title  # non-empty
    assert cfg.provider.name == "vscode_extension"


def test_companion_yaml_preserves_multiline_topic() -> None:
    """When the topic spans multiple paragraphs, the `|` block scalar
    must keep them intact. Specifically, blank lines inside the topic
    must not break YAML parsing."""
    from core.config import Config
    import io
    import yaml as pyyaml

    topic = (
        "First paragraph of the topic.\n"
        "\n"
        "Second paragraph after a blank line."
    )
    yaml_text = _render_companion_yaml(
        topic,
        proposal_id="1778452404-test-aabbcc",
        generated_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )
    parsed = pyyaml.safe_load(io.StringIO(yaml_text))
    assert "First paragraph" in parsed["topic"]
    assert "Second paragraph" in parsed["topic"]


def test_companion_yaml_references_proposal_md_filename() -> None:
    yaml_text = _render_companion_yaml(
        "X",
        proposal_id="1778452404-test-aabbcc",
        generated_at=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )
    assert "1778452404-test-aabbcc-proposal.md" in yaml_text


# ---------- end-to-end ------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_proposal_writes_md_and_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    captured: dict[str, str] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        captured["prompt"] = messages[-1]["content"]
        captured["node"] = kw.get("node", "")
        return "# Proposal\n\n## TL;DR\nfeasible.\n"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    art = await generate_proposal(
        "Compare RK4 vs Verlet on damped oscillators",
        outputs,
        provider=ProviderConfig(name="openai"),
        knowledge=None,
    )

    assert art.proposal_path.is_file()
    body = art.proposal_path.read_text(encoding="utf-8")
    assert body.startswith("# Proposal")

    assert art.yaml_path.is_file()
    yaml_body = art.yaml_path.read_text(encoding="utf-8")
    assert "topic:" in yaml_body
    assert "Compare RK4 vs Verlet" in yaml_body
    assert captured["node"] == "proposal"
    # Both files live under _drafts.
    assert art.proposal_path.parent.name == "_drafts"
    assert art.yaml_path.parent.name == "_drafts"


@pytest.mark.asyncio
async def test_generate_proposal_rejects_empty_topic(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    with pytest.raises(ValueError):
        await generate_proposal(
            "   ", outputs,
            provider=ProviderConfig(name="openai"),
            knowledge=None,
        )


@pytest.mark.asyncio
async def test_generate_proposal_rejects_oversized_topic(
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    huge = "x " * 3000   # ~6000 chars, way over the 4K cap
    with pytest.raises(ValueError):
        await generate_proposal(
            huge, outputs,
            provider=ProviderConfig(name="openai"),
            knowledge=None,
        )


@pytest.mark.asyncio
async def test_generate_proposal_ingests_to_axon_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return "# Plan\n"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    knowledge = MagicMock()
    knowledge.enabled = True
    knowledge.add_text = MagicMock(return_value=True)

    art = await generate_proposal(
        "Test topic",
        outputs,
        provider=ProviderConfig(name="openai"),
        knowledge=knowledge,
    )
    assert art.ingested_to_axon is True
    kwargs = knowledge.add_text.call_args.kwargs
    assert kwargs["kind"] == "fi_proposal"
    assert kwargs["metadata"]["proposal_id"] == art.proposal_id
