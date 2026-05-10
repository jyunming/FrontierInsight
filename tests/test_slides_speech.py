"""Direct tests for `generation/slides.py` and `generation/speech.py`.

Both generators perform a single LLM call and write a markdown file to
`out_dir`. Slides additionally shells out to `marp` for html/pdf when the
CLI is on PATH; those steps are gated behind a skipif.

LLM calls are intercepted by monkeypatching `core.provider.LLMClient.chat`
(the class is re-exported into `generation.slides` / `generation.speech`,
so patching the source binds everywhere).
"""

from __future__ import annotations

import shutil
import subprocess
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
from generation.slides import SlideGenerator, _strip_outer_fence
from generation.speech import SpeechGenerator


_FENCED_MARP = """```markdown
---
marp: true
theme: default
---

# Toy Title

Body text.

---

## Results
![bg right](figures/result.png)
```"""


_PLAIN_TALK = """# Talk: Toy Title

[slide: 1] Welcome. We'll walk through the toy result.

[slide: 2] Notice that the curve is monotonic.
"""


def _make_config(tmp_path: Path, kinds: list[str]) -> Config:
    return Config(
        topic="slides+speech direct test",
        title="slides-speech-test",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(kinds=kinds, output_dir=tmp_path / "outputs"),
    )


def _make_artifacts(tmp_path: Path, *, with_figure: bool) -> QuestArtifacts:
    quest_root = tmp_path / "quest"
    (quest_root / "paper").mkdir(parents=True, exist_ok=True)
    paper_md = quest_root / "paper" / "paper.md"
    paper_md.write_text(
        "# Toy Title\n\nMethods. Results. ![r](figures/result.png)\n",
        encoding="utf-8",
    )
    figures_dir: Path | None = None
    if with_figure:
        figures_dir = quest_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        (figures_dir / "result.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return QuestArtifacts(
        quest_id="qid-test",
        quest_root=quest_root,
        paper_md=paper_md,
        figures_dir=figures_dir,
    )


# ---------- _strip_outer_fence ----------


def test_strip_outer_fence_with_language_hint() -> None:
    raw = "```markdown\nhello\nworld\n```"
    assert _strip_outer_fence(raw) == "hello\nworld"


def test_strip_outer_fence_plain_fence() -> None:
    raw = "```\nhello\n```"
    assert _strip_outer_fence(raw) == "hello"


def test_strip_outer_fence_no_fence_passthrough() -> None:
    raw = "no fences here\nstill raw"
    assert _strip_outer_fence(raw) == "no fences here\nstill raw"


def test_strip_outer_fence_handles_trailing_whitespace() -> None:
    raw = "  ```markdown\nhello\n```\n  "
    assert _strip_outer_fence(raw) == "hello"


# ---------- SlideGenerator ----------


@pytest.mark.asyncio
async def test_slides_writes_md_with_fence_stripped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    art = _make_artifacts(tmp_path, with_figure=True)
    cfg = _make_config(tmp_path, kinds=["slides"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)

    captured: dict[str, str] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        captured["prompt"] = messages[-1]["content"]
        return _FENCED_MARP

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    result = await SlideGenerator(cfg).generate(art, out_dir)
    slides_md = out_dir / "slides.md"
    assert slides_md.exists()
    assert "slides_md" in result and result["slides_md"] == slides_md
    body = slides_md.read_text(encoding="utf-8")
    assert not body.startswith("```")
    assert body.startswith("---\nmarp: true")
    assert "figures/result.png" in body
    assert "$paper_md" not in captured["prompt"]
    assert "$figure_list" not in captured["prompt"]
    assert "- figures/result.png" in captured["prompt"]


@pytest.mark.asyncio
async def test_slides_skipped_when_kind_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["paper_md"])  # no "slides"

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        raise AssertionError("LLM must not be called when slides kind is absent")

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    result = await SlideGenerator(cfg).generate(art, art.quest_root)
    assert result == {}


@pytest.mark.asyncio
async def test_slides_no_figures_dir_uses_none_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `figures_dir` is missing the prompt must still substitute cleanly."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["slides"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)

    captured: dict[str, str] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        captured["prompt"] = messages[-1]["content"]
        return "---\nmarp: true\n---\n\n# Hi\n"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    await SlideGenerator(cfg).generate(art, out_dir)
    assert "(none)" in captured["prompt"]


@pytest.mark.skipif(shutil.which("marp") is None, reason="marp CLI not on PATH")
@pytest.mark.asyncio
async def test_slides_marp_renders_html_and_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    art = _make_artifacts(tmp_path, with_figure=True)
    cfg = _make_config(tmp_path, kinds=["slides"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return "---\nmarp: true\ntheme: default\n---\n\n# Hi\n\n---\n\n## Bye\n"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    result = await SlideGenerator(cfg).generate(art, out_dir)
    assert "slides_html" in result
    assert result["slides_html"].exists()
    assert "slides_pdf" in result
    assert result["slides_pdf"].exists()


# ---------- SpeechGenerator ----------


@pytest.mark.asyncio
async def test_speech_writes_talk_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["speech"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)

    captured: dict[str, str] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        captured["prompt"] = messages[-1]["content"]
        return _PLAIN_TALK

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    result = await SpeechGenerator(cfg).generate(art, out_dir)
    talk = out_dir / "talk.md"
    assert talk.exists()
    assert result["speech_md"] == talk
    body = talk.read_text(encoding="utf-8")
    assert body.startswith("# Talk:")
    assert "[slide: 1]" in body
    assert body.endswith("\n")
    assert "(no slide deck available)" in captured["prompt"]
    assert "$paper_md" not in captured["prompt"]
    assert "$slides_outline" not in captured["prompt"]


@pytest.mark.asyncio
async def test_speech_picks_up_existing_slides_outline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["speech"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)
    slides_md = out_dir / "slides.md"
    slides_md.write_text(
        "---\nmarp: true\n---\n\n# Slide 1\n\nMARKER_OUTLINE_TOKEN\n",
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        captured["prompt"] = messages[-1]["content"]
        return _PLAIN_TALK

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    await SpeechGenerator(cfg).generate(art, out_dir)
    assert "MARKER_OUTLINE_TOKEN" in captured["prompt"]
    assert "(no slide deck available)" not in captured["prompt"]


@pytest.mark.asyncio
async def test_speech_truncates_large_slides_outline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """slides_outline is hard-capped at 4000 chars before substitution."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["speech"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)
    big = "A" * 5000 + "TAIL_SHOULD_BE_TRUNCATED"
    (out_dir / "slides.md").write_text(big, encoding="utf-8")

    captured: dict[str, str] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        captured["prompt"] = messages[-1]["content"]
        return _PLAIN_TALK

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    await SpeechGenerator(cfg).generate(art, out_dir)
    assert "TAIL_SHOULD_BE_TRUNCATED" not in captured["prompt"]
    # The retained chunk is exactly the first 4000 chars (all 'A').
    assert "A" * 4000 in captured["prompt"]


@pytest.mark.asyncio
async def test_speech_skipped_when_kind_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["paper_md"])  # no "speech"

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        raise AssertionError("LLM must not be called when speech kind is absent")

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    result = await SpeechGenerator(cfg).generate(art, art.quest_root)
    assert result == {}
