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

[slide: 1] Welcome everyone, and thank you for being here. Today
we'll walk through the toy result that motivated this whole quest.
The setup is simple: we sample a synthetic curve, fit a parametric
form to it, and report the residuals across three random seeds.

[slide: 2] Notice that the curve is monotonic, which is exactly the
property we wanted to demonstrate. If you look at the right panel
you can see the fitted line lies inside the 95 percent envelope
across the full input range. We'll come back to the implications
for downstream applications in the discussion section.
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
    # Pin "neither render CLI on PATH" so this test stays focused on
    # slides.md production. On dev boxes where marp / pandoc are
    # installed, omitting this lets the generator spawn the real
    # binaries — which on Windows fails for marp.ps1 (can't exec a
    # PowerShell script directly via create_subprocess_exec).
    monkeypatch.setattr("generation.slides.shutil.which", lambda _n: None)

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
    # Same rationale as test_slides_writes_md_with_fence_stripped above:
    # pin "no render CLIs" so this test isolates the prompt-substitution
    # path.
    monkeypatch.setattr("generation.slides.shutil.which", lambda _n: None)
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


# ---------- pandoc → pptx ----------


@pytest.mark.asyncio
async def test_slides_invokes_pandoc_for_pptx_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User feedback: 'the generated slide is not really a slide yet,
    it's a md file. can we make it really a pptx?' — pin the new
    pandoc invocation: argv includes pandoc + slides.md + --slide-level=2
    + an output path ending in .pptx."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["slides"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _FENCED_MARP

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    # Pretend marp is NOT on PATH (we're isolating the pandoc branch)
    # and pandoc IS available.
    def fake_which(name: str) -> str | None:
        return "/fake/pandoc" if name == "pandoc" else None
    monkeypatch.setattr("generation.slides.shutil.which", fake_which)

    captured_argv: list[list[str]] = []

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def fake_exec(*argv: str, **_kw):  # noqa: ANN001
        captured_argv.append(list(argv))
        # Create the destination so the result-dict assertion below holds.
        try:
            o_idx = argv.index("-o")
            Path(argv[o_idx + 1]).touch()
        except ValueError:
            pass
        return _FakeProc()

    monkeypatch.setattr(
        "generation.slides.asyncio.create_subprocess_exec", fake_exec,
    )

    result = await SlideGenerator(cfg).generate(art, out_dir)

    assert any(
        a[0].endswith("pandoc") and "--slide-level=2" in a and a[-1].endswith("slides.pptx")
        for a in captured_argv
    ), f"expected pandoc invocation; got {captured_argv!r}"
    assert "slides_pptx" in result
    assert result["slides_pptx"] == out_dir / "slides.pptx"


@pytest.mark.asyncio
async def test_slides_spawn_failure_does_not_abort_other_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `create_subprocess_exec` raise (FileNotFoundError,
    PermissionError, OSError — possible when the resolved path
    becomes invalid between `shutil.which` and spawn) must NOT
    propagate out of `_run_cli`. The slide generator is
    contractually best-effort across its 3 targets."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["slides"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _FENCED_MARP

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    # Both render CLIs "available" so both branches try to spawn.
    monkeypatch.setattr("generation.slides.shutil.which",
                        lambda name: f"/fake/{name}")

    async def fake_exec(*_argv, **_kw):  # noqa: ANN001
        raise FileNotFoundError("path vanished between which() and spawn()")

    monkeypatch.setattr(
        "generation.slides.asyncio.create_subprocess_exec", fake_exec,
    )

    # Must NOT raise. result contains slides_md but no rendered targets.
    result = await SlideGenerator(cfg).generate(art, out_dir)
    assert "slides_md" in result
    assert "slides_html" not in result
    assert "slides_pdf" not in result
    assert "slides_pptx" not in result


@pytest.mark.asyncio
async def test_slides_skips_pptx_when_pandoc_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pandoc absent: the pptx branch must skip cleanly without erroring;
    slides.md is still produced."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["slides"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _FENCED_MARP

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    monkeypatch.setattr("generation.slides.shutil.which", lambda _n: None)

    result = await SlideGenerator(cfg).generate(art, out_dir)
    assert "slides_md" in result
    assert "slides_pptx" not in result


# ---------- Wave 3: slides_skipped.md diagnostic (Pattern C) ----------


@pytest.mark.asyncio
async def test_slides_writes_skip_diagnostic_when_marp_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Marp CLI absent on PATH: a single ``slides_skipped.md`` diagnostic
    must be written next to ``slides.md`` with ``reason_code=no_marp``
    so the user discovers the skip without grepping run.log. Mirrors
    the ``paper_pdf_skipped.md`` and ``poster_pdf_skipped.md`` UX."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["slides"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _FENCED_MARP

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    monkeypatch.setattr("generation.slides.shutil.which", lambda _n: None)

    result = await SlideGenerator(cfg).generate(art, out_dir)
    diag = out_dir / "slides_skipped.md"
    assert diag.exists(), "slides_skipped.md must be written when marp is absent"
    body = diag.read_text(encoding="utf-8")
    # Display name is scoped to the actual failure surface
    # (slides.html / slides.pdf — what marp produces); slides.md
    # typically lands even when marp fails so a blanket "slides was
    # requested but not produced" mis-describes the state.
    assert "slides.html / slides.pdf was requested but not produced" in body
    assert "no_marp" in body
    assert "marp-team/marp-cli" in body  # the install recipe
    assert result.get("slides_skipped") == diag
    # slides.md should still be produced — the LLM call doesn't depend
    # on marp.
    assert "slides_md" in result


@pytest.mark.asyncio
async def test_slides_writes_skip_diagnostic_when_marp_returns_nonzero_rc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Marp present but exits with non-zero rc: the diagnostic carries
    a ``marp_rc_<N>`` reason code so the operator can filter logs by
    failure mode."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["slides"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _FENCED_MARP

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    # Marp "available", pandoc absent (we're isolating the marp-failure
    # branch).
    def fake_which(name: str) -> str | None:
        return "/fake/marp" if name == "marp" else None
    monkeypatch.setattr("generation.slides.shutil.which", fake_which)

    class _FakeProc:
        returncode = 7

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b"some unrelated render error"

    async def fake_exec(*_argv, **_kw):  # noqa: ANN001
        return _FakeProc()

    monkeypatch.setattr(
        "generation.slides.asyncio.create_subprocess_exec", fake_exec,
    )

    result = await SlideGenerator(cfg).generate(art, out_dir)
    diag = out_dir / "slides_skipped.md"
    assert diag.exists()
    body = diag.read_text(encoding="utf-8")
    assert "marp_rc_7" in body
    # The render-target stderr is surfaced so the user can read the
    # actual marp error without re-running.
    assert "some unrelated render error" in body
    assert result.get("slides_skipped") == diag
    assert "slides_html" not in result
    assert "slides_pdf" not in result


@pytest.mark.asyncio
async def test_slides_skip_diagnostic_detects_chromium_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When marp's stderr contains the chromium-missing signature, the
    reason code MUST be the specific ``chromium_missing`` so the
    how-to-fix text points the user at the puppeteer/Chromium issue
    instead of the generic "marp errored" advice."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["slides"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _FENCED_MARP

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    def fake_which(name: str) -> str | None:
        return "/fake/marp" if name == "marp" else None
    monkeypatch.setattr("generation.slides.shutil.which", fake_which)

    class _FakeProc:
        returncode = 1

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b"Error: Could not find Chromium (rev. 1095492)."

    async def fake_exec(*_argv, **_kw):  # noqa: ANN001
        return _FakeProc()

    monkeypatch.setattr(
        "generation.slides.asyncio.create_subprocess_exec", fake_exec,
    )

    await SlideGenerator(cfg).generate(art, out_dir)
    diag = out_dir / "slides_skipped.md"
    assert diag.exists()
    body = diag.read_text(encoding="utf-8")
    assert "chromium_missing" in body
    assert "PUPPETEER_EXECUTABLE_PATH" in body


@pytest.mark.asyncio
async def test_slides_success_removes_stale_skip_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a prior failed run left ``slides_skipped.md`` on disk and the
    current run renders successfully, the stale diagnostic MUST be
    deleted. Mirrors the same contract poster.py and paper.py
    enforce on their skip files."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["slides"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "slides_skipped.md"
    stale.write_text("stale from previous run", encoding="utf-8")

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _FENCED_MARP

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    def fake_which(name: str) -> str | None:
        return f"/fake/{name}" if name == "marp" else None
    monkeypatch.setattr("generation.slides.shutil.which", fake_which)

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def fake_exec(*argv: str, **_kw):  # noqa: ANN001
        # Touch the requested -o output so the success path records it.
        try:
            o_idx = list(argv).index("-o")
            Path(argv[o_idx + 1]).touch()
        except ValueError:
            pass
        return _FakeProc()

    monkeypatch.setattr(
        "generation.slides.asyncio.create_subprocess_exec", fake_exec,
    )

    result = await SlideGenerator(cfg).generate(art, out_dir)
    assert "slides_html" in result and "slides_pdf" in result
    assert not stale.exists(), "stale slides_skipped.md must be cleaned up on success"
    assert "slides_skipped" not in result


# ---------- Wave 3: speech_skipped.md diagnostic ----------


@pytest.mark.asyncio
async def test_speech_writes_skip_diagnostic_on_empty_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 50-char LLM response is well below the 200-char floor for a
    usable 10-minute talk script. The generator must NOT ship a
    broken ``talk.md``; it must write ``speech_skipped.md`` instead
    with reason code ``llm_refused_or_empty``."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["speech"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        # 50 chars — well under the 200-char threshold.
        return "Too short to be a useful talk script, sorry. Bye."

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    result = await SpeechGenerator(cfg).generate(art, out_dir)
    assert not (out_dir / "talk.md").exists(), (
        "talk.md must NOT be written when the LLM response is too short"
    )
    diag = out_dir / "speech_skipped.md"
    assert diag.exists()
    body = diag.read_text(encoding="utf-8")
    # Display name names the actual file the user would have
    # gotten (talk.md) so the H1 reads correctly to a reader.
    assert "speech (talk.md) was requested but not produced" in body
    assert "llm_refused_or_empty" in body
    # Raw response excerpt embedded for debugging.
    assert "Too short to be a useful talk script" in body
    assert "speech_md" not in result
    assert result.get("speech_skipped") == diag


@pytest.mark.asyncio
async def test_speech_writes_skip_diagnostic_on_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM refusal: a content-policy decline like ``"I'm sorry, I can't
    help with that."`` (padded to clear the length floor) must
    trigger a skip diagnostic, NOT a shipped ``talk.md`` containing
    the refusal."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["speech"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)

    refusal_response = (
        "I'm sorry, I can't help with that. " * 10
        + "The topic appears to be outside my safety guidelines."
    )

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return refusal_response

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    result = await SpeechGenerator(cfg).generate(art, out_dir)
    assert not (out_dir / "talk.md").exists(), (
        "talk.md must NOT be written when the LLM refuses"
    )
    diag = out_dir / "speech_skipped.md"
    assert diag.exists()
    body = diag.read_text(encoding="utf-8")
    assert "llm_refused_or_empty" in body
    # The matched refusal phrase is surfaced for debugging.
    assert "sorry, i can't" in body.lower()
    assert "speech_md" not in result
    assert result.get("speech_skipped") == diag


@pytest.mark.asyncio
async def test_speech_success_removes_stale_skip_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prior run left ``speech_skipped.md`` on disk; this run's LLM
    response is fine. The stale diagnostic must be removed so the
    quest dir doesn't show both ``talk.md`` AND ``speech_skipped.md``."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["speech"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "speech_skipped.md"
    stale.write_text("stale from previous run", encoding="utf-8")

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return _PLAIN_TALK

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    result = await SpeechGenerator(cfg).generate(art, out_dir)
    assert (out_dir / "talk.md").exists()
    assert not stale.exists(), (
        "stale speech_skipped.md must be cleaned up on success"
    )
    assert result.get("speech_md") == out_dir / "talk.md"
    assert "speech_skipped" not in result


@pytest.mark.asyncio
async def test_speech_removes_stale_diagnostic_when_kind_dropped(
    tmp_path: Path,
) -> None:
    """User had ``speech`` in output.kinds; this run drops it. The
    stale ``speech_skipped.md`` from a prior failed run must be
    cleaned up — otherwise it persists indefinitely after the user
    opts out. Mirrors PaperGenerator's cleanup-on-opt-out pattern."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["paper_md"])  # NO "speech"
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "speech_skipped.md"
    stale.write_text("stale from previous failed run", encoding="utf-8")

    result = await SpeechGenerator(cfg).generate(art, out_dir)

    assert result == {}, "no artifacts when speech is not in kinds"
    assert not stale.exists(), (
        "stale speech_skipped.md must be removed when speech is "
        "dropped from output.kinds"
    )


@pytest.mark.asyncio
async def test_slides_removes_stale_diagnostic_when_kind_dropped(
    tmp_path: Path,
) -> None:
    """Same cleanup-on-opt-out contract for slides."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["paper_md"])  # NO "slides"
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "slides_skipped.md"
    stale.write_text("stale from previous failed run", encoding="utf-8")

    result = await SlideGenerator(cfg).generate(art, out_dir)

    assert result == {}, "no artifacts when slides is not in kinds"
    assert not stale.exists(), (
        "stale slides_skipped.md must be removed when slides is "
        "dropped from output.kinds"
    )


@pytest.mark.asyncio
async def test_speech_diagnostic_escapes_backtick_fences_in_llm_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM response embedded in the diagnostic might itself contain
    ``` (e.g. a refusal mentioning code, or an aborted code block).
    Default ``` fence would break the markdown structure. The
    helper must pick a longer fence to escape correctly."""
    art = _make_artifacts(tmp_path, with_figure=False)
    cfg = _make_config(tmp_path, kinds=["speech"])
    out_dir = art.quest_root
    out_dir.mkdir(parents=True, exist_ok=True)

    # 50 chars (below threshold) so the rejection fires; contains
    # a literal ``` that would have broken the fence.
    bad_response = "Sorry, I can't help.\n```python\nprint(1)\n```"

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return bad_response

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    await SpeechGenerator(cfg).generate(art, out_dir)
    body = (out_dir / "speech_skipped.md").read_text(encoding="utf-8")
    # The fence MUST be longer than the longest backtick run in the
    # embedded preview, so the markdown structure stays intact.
    # 3 backticks in the response → fence must be at least 4.
    assert "````" in body, (
        "diagnostic must use a 4+-backtick fence when the preview "
        "contains ``` so the embedded raw response doesn't escape "
        "its container"
    )
    # The full preview content still lands inside the longer fence.
    assert "print(1)" in body
