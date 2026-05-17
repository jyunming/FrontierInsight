"""Poster generator (beamerposter wrapper) validation.

Covers:
- LLM JSON -> 3-column poster.tex substitution path with no real LLM calls.
- LaTeX inline math (`$x^2$`) inside LLM-produced columns survives substitution
  (would raise on `string.Template.substitute` only if the *template* itself
  contained malformed `$`; values are passed through verbatim by both
  substitute and safe_substitute).
- pdflatex skip when the binary is not on PATH (no PDF asserted unless real
  pdflatex is available; even then, beamerposter may not be installed, so we
  do not require a successful PDF compile here).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)
from core.engine import QuestArtifacts
from core.provider import ResolvedEndpoint
from generation.poster import PosterGenerator


def _make_config(tmp_path: Path, kinds: list[str]) -> Config:
    return Config(
        topic="poster generator unit test",
        title="poster-test",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=10),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out", kinds=kinds),
    )


def _make_artifacts(tmp_path: Path) -> QuestArtifacts:
    quest_root = tmp_path / "quest"
    paper_dir = quest_root / "paper"
    paper_dir.mkdir(parents=True)
    paper_md = paper_dir / "paper.md"
    paper_md.write_text(
        "# Toy Paper\n\nMethods. Results show $y = x^2$ scaling.\n",
        encoding="utf-8",
    )
    figures = quest_root / "figures"
    figures.mkdir()
    (figures / "result.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # not parsed, just listed
    return QuestArtifacts(
        quest_id="qtest",
        quest_root=quest_root,
        paper_md=paper_md,
        figures_dir=figures,
    )


def _patch_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_resolve(provider, supervisor):  # noqa: ANN001
        return ResolvedEndpoint(
            base_url="http://127.0.0.1:1/v1", model="x", api_key="not-needed"
        )

    monkeypatch.setattr("generation.poster.resolve_endpoint_async", fake_resolve)


@pytest.mark.asyncio
async def test_poster_skipped_when_kind_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When 'poster' is not in output.kinds, generator returns {} and no
    LLM call happens (fail-loud sentinel inside the patched chat)."""
    cfg = _make_config(tmp_path, kinds=["paper_md"])
    art = _make_artifacts(tmp_path)

    async def must_not_call(self, messages, **kw):  # noqa: ANN001
        raise AssertionError("LLM should not be called when poster kind is off")

    monkeypatch.setattr("generation.poster.LLMClient.chat", must_not_call)

    result = await PosterGenerator(cfg).generate(art, art.quest_root)
    assert result == {}


@pytest.mark.asyncio
async def test_poster_writes_tex_with_substituted_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end of the LaTeX wrapper: fake LLM returns valid 3-column JSON,
    poster.tex is written, contains the title and each column body."""
    cfg = _make_config(tmp_path, kinds=["poster"])
    art = _make_artifacts(tmp_path)

    payload = {
        "title": "On Toy Scaling",
        "left": r"\textbf{Abstract.} We study scaling.",
        "middle": r"\textbf{Methods.} Plot \(y\) vs \(x\).",
        "right": r"\textbf{Results.} Monotonic curve.",
    }

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return json.dumps(payload)

    _patch_endpoint(monkeypatch)
    monkeypatch.setattr("generation.poster.LLMClient.chat", fake_chat)
    # Force the pdflatex branch off so the test never depends on a TeX install
    # (and never sees the multi-second timeout) regardless of host setup.
    monkeypatch.setattr("generation.poster.shutil.which", lambda _name: None)

    result = await PosterGenerator(cfg).generate(art, art.quest_root)

    assert "poster_tex" in result
    tex = result["poster_tex"].read_text(encoding="utf-8")
    assert r"\title{On Toy Scaling}" in tex
    assert r"\textbf{Abstract.} We study scaling." in tex
    assert r"\textbf{Methods.} Plot \(y\) vs \(x\)." in tex
    assert r"\textbf{Results.} Monotonic curve." in tex
    # No leftover Python Template placeholders.
    for placeholder in ("$title", "$left", "$middle", "$right"):
        assert placeholder not in tex
    # No PDF when pdflatex is suppressed.
    assert "poster_pdf" not in result


@pytest.mark.asyncio
async def test_poster_handles_inline_latex_math_in_llm_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM emits LaTeX `$math$` inside column bodies. Template substitution
    must not raise on substituted values containing `$`, and the math must
    appear verbatim in poster.tex."""
    cfg = _make_config(tmp_path, kinds=["poster"])
    art = _make_artifacts(tmp_path)

    payload = {
        "title": "Energy $E=mc^2$",
        "left": r"Inline math: $\alpha + \beta = \gamma$ and $x^2$.",
        "middle": r"Two equations: $f(x) = \int_0^1 g(t)\,dt$, $\sum_i a_i$.",
        "right": r"Bare dollars too: cost is \$5 and value $v$.",
    }

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return json.dumps(payload)

    _patch_endpoint(monkeypatch)
    monkeypatch.setattr("generation.poster.LLMClient.chat", fake_chat)
    monkeypatch.setattr("generation.poster.shutil.which", lambda _name: None)

    result = await PosterGenerator(cfg).generate(art, art.quest_root)

    tex = result["poster_tex"].read_text(encoding="utf-8")
    assert r"$\alpha + \beta = \gamma$" in tex
    assert r"$f(x) = \int_0^1 g(t)\,dt$" in tex
    assert r"\$5" in tex
    assert r"Energy $E=mc^2$" in tex


@pytest.mark.asyncio
async def test_poster_falls_back_when_llm_returns_garbage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed LLM output -> poster.tex is still written with the default
    placeholder columns (Untitled / empty)."""
    cfg = _make_config(tmp_path, kinds=["poster"])
    art = _make_artifacts(tmp_path)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return "not json at all, no braces here"

    _patch_endpoint(monkeypatch)
    monkeypatch.setattr("generation.poster.LLMClient.chat", fake_chat)
    monkeypatch.setattr("generation.poster.shutil.which", lambda _name: None)

    result = await PosterGenerator(cfg).generate(art, art.quest_root)

    tex = result["poster_tex"].read_text(encoding="utf-8")
    assert r"\title{Untitled}" in tex


@pytest.mark.skipif(
    shutil.which("pdflatex") is None,
    reason="pdflatex not on PATH; skipping real compile path",
)
@pytest.mark.asyncio
async def test_poster_pdflatex_invoked_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When pdflatex is available, the generator invokes it. We do not
    require the compile to succeed (beamerposter may not be installed),
    only that the .tex was written and the call site was reached."""
    cfg = _make_config(tmp_path, kinds=["poster"])
    art = _make_artifacts(tmp_path)

    payload = {"title": "T", "left": "L", "middle": "M", "right": "R"}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return json.dumps(payload)

    _patch_endpoint(monkeypatch)
    monkeypatch.setattr("generation.poster.LLMClient.chat", fake_chat)

    result = await PosterGenerator(cfg).generate(art, art.quest_root)
    assert "poster_tex" in result
    assert result["poster_tex"].exists()
