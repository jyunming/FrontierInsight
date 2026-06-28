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
- Tectonic fallback: when pdflatex is absent but tectonic is on PATH, the
  poster generator picks tectonic — same 3-tier discovery the paper
  generator uses (audit BLOCK #5).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

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
from generation import poster as poster_mod
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
    assert r"\textbf{Results.} Monotonic curve." in tex
    # No leftover Python Template placeholders (portrait template: 2 columns).
    for placeholder in ("$title", "$left", "$right", "$references"):
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
        "right": r"Two equations: $f(x) = \int_0^1 g(t)\,dt$, $\sum_i a_i$. Bare dollars too: cost is \$5 and value $v$.",
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


# ---- Audit BLOCK #5: poster shares paper's 3-tier engine discovery ---------


@pytest.mark.asyncio
async def test_poster_falls_back_to_tectonic_when_pdflatex_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit BLOCK #5 regression: a user who followed the documented
    no-admin install (``python launch.py --install-tectonic``) gets
    ``paper.pdf`` but the poster used to silently skip because the
    previous gate was a bare ``shutil.which('pdflatex')`` check. After
    the fix, poster.py must consult the same 3-tier discovery the
    paper generator uses (pdflatex -> system tectonic -> repo-local
    ``tools/tectonic[.exe]``).

    Mock pdflatex as absent and tectonic as present on PATH; assert
    the engine binary the subprocess actually receives is tectonic.
    """
    cfg = _make_config(tmp_path, kinds=["poster"])
    art = _make_artifacts(tmp_path)

    payload = {"title": "T", "left": "L", "middle": "M", "right": "R"}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return json.dumps(payload)

    _patch_endpoint(monkeypatch)
    monkeypatch.setattr("generation.poster.LLMClient.chat", fake_chat)

    def fake_which(name):  # type: ignore[no-untyped-def]
        if name == "pdflatex":
            return None
        if name == "tectonic":
            return "/fake/tectonic.exe"
        return None
    # Patch the shutil module used by both poster.py and _pdf_engine.py
    # (same module object — both ``import shutil``).
    monkeypatch.setattr(poster_mod.shutil, "which", fake_which)

    captured_cmd: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        captured_cmd[:] = list(cmd)
        # Pretend the engine produced poster.pdf so the success branch
        # records it in the result dict and the test can assert on it.
        cwd = Path(_kwargs.get("cwd") or ".")
        (cwd / "poster.pdf").write_bytes(b"%PDF-fake\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(poster_mod.subprocess, "run", fake_run)

    result = await PosterGenerator(cfg).generate(art, art.quest_root)

    # The subprocess invocation used the tectonic binary, NOT pdflatex —
    # the regression we're guarding against is the old gate passing the
    # literal string "pdflatex" here regardless of fallback.
    assert captured_cmd, "subprocess.run was not invoked"
    assert captured_cmd[0] == "/fake/tectonic.exe", (
        f"Expected poster to invoke tectonic when pdflatex is missing; "
        f"got argv0={captured_cmd[0]!r}"
    )
    # Same flag contract works for both engines.
    assert "-interaction=nonstopmode" in captured_cmd
    assert "-halt-on-error" in captured_cmd
    assert "poster_pdf" in result
    # On success the diagnostic file MUST NOT be left behind.
    assert "poster_pdf_skipped" not in result
    assert not (art.quest_root / "poster_pdf_skipped.md").exists()


@pytest.mark.asyncio
async def test_poster_writes_skip_diagnostic_when_no_engine_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When neither pdflatex nor tectonic is reachable (and the repo-local
    fallback at ``tools/tectonic[.exe]`` is also missing), the poster
    generator must write ``poster_pdf_skipped.md`` next to ``poster.tex``
    — mirroring the ``paper_pdf_skipped.md`` contract so the user
    discovers the skip without grepping ``run.log``."""
    cfg = _make_config(tmp_path, kinds=["poster"])
    art = _make_artifacts(tmp_path)

    payload = {"title": "T", "left": "L", "middle": "M", "right": "R"}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return json.dumps(payload)

    _patch_endpoint(monkeypatch)
    monkeypatch.setattr("generation.poster.LLMClient.chat", fake_chat)
    # PATH lookup misses everything.
    monkeypatch.setattr(poster_mod.shutil, "which", lambda _name: None)
    # Steer the repo-local probe at a clean tmp dir so a dev box with
    # ``tools/tectonic.exe`` already present doesn't accidentally
    # rescue the test.
    from generation import _pdf_engine
    monkeypatch.setattr(_pdf_engine, "_DEFAULT_REPO_ROOT", tmp_path)

    result = await PosterGenerator(cfg).generate(art, art.quest_root)

    # poster.tex still produced.
    assert "poster_tex" in result
    # No PDF.
    assert "poster_pdf" not in result
    # Diagnostic file IS produced.
    diag = art.quest_root / "poster_pdf_skipped.md"
    assert diag.exists(), "poster_pdf_skipped.md was not written"
    body = diag.read_text(encoding="utf-8")
    assert "poster.pdf was requested but not produced" in body
    assert "no_latex_engine" in body  # reason code
    assert "--install-tectonic" in body  # how-to-fix recipe
    # result dict surfaces the diagnostic so callers (launch.py / VSCode
    # bridge) can include it in their "your quest is done" messages.
    assert result.get("poster_pdf_skipped") == diag


@pytest.mark.asyncio
async def test_poster_writes_skip_diagnostic_when_pdf_missing_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine returned rc=0 but ``poster.pdf`` isn't on disk —
    silent partial success. Without a diagnostic the caller sees a
    missing ``poster_pdf`` in the result dict with no breadcrumb.

    Mirrors the analogous branch in ``PaperGenerator._compile_pdf``
    so the user gets the same skip-diagnostic shape across both
    output kinds. Reason code is distinct
    (``output_missing_after_success``) so the operator can tell this
    apart from a real engine failure (rc != 0) when filtering logs.
    """
    cfg = _make_config(tmp_path, kinds=["poster"])
    art = _make_artifacts(tmp_path)

    payload = {"title": "T", "left": "L", "middle": "M", "right": "R"}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return json.dumps(payload)

    _patch_endpoint(monkeypatch)
    monkeypatch.setattr("generation.poster.LLMClient.chat", fake_chat)
    # Engine is present.
    monkeypatch.setattr(
        poster_mod.shutil, "which",
        lambda name: f"/fake/{name}.exe" if name == "pdflatex" else None,
    )

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        # Subprocess "succeeds" without producing the output file.
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(poster_mod.subprocess, "run", fake_run)

    result = await PosterGenerator(cfg).generate(art, art.quest_root)

    # No PDF.
    assert "poster_pdf" not in result
    # Diagnostic IS produced — the gap this test exists to plug.
    diag = art.quest_root / "poster_pdf_skipped.md"
    assert diag.exists()
    body = diag.read_text(encoding="utf-8")
    assert "output_missing_after_success" in body
    assert result.get("poster_pdf_skipped") == diag


@pytest.mark.asyncio
async def test_poster_uses_same_engine_discovery_as_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever ``generation._pdf_engine.find_pdf_engine`` returns is
    what the poster generator MUST use — the audit fix is exactly the
    "use the shared helper" deduplication. Verify the contract by
    swapping the helper for a sentinel and asserting the sentinel's
    binary path lands in the subprocess argv."""
    cfg = _make_config(tmp_path, kinds=["poster"])
    art = _make_artifacts(tmp_path)

    payload = {"title": "T", "left": "L", "middle": "M", "right": "R"}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return json.dumps(payload)

    _patch_endpoint(monkeypatch)
    monkeypatch.setattr("generation.poster.LLMClient.chat", fake_chat)

    sentinel = ("tectonic", "/sentinel/tectonic.exe")
    monkeypatch.setattr(poster_mod, "find_pdf_engine", lambda: sentinel)

    captured_cmd: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        # Record only the FIRST subprocess call — the LaTeX compile. The
        # auto-fit overflow detector may make a SECOND call (pdftoppm) to
        # rasterise the result, which must not clobber the assertion.
        if not captured_cmd:
            captured_cmd[:] = list(cmd)
        cwd = Path(_kwargs.get("cwd") or ".")
        (cwd / "poster.pdf").write_bytes(b"%PDF-fake\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(poster_mod.subprocess, "run", fake_run)

    await PosterGenerator(cfg).generate(art, art.quest_root)

    assert captured_cmd[0] == sentinel[1], (
        "Poster MUST invoke whichever binary find_pdf_engine returned; "
        f"got {captured_cmd[0]!r} expected {sentinel[1]!r}"
    )


def test_escape_latex_text_specials() -> None:
    """Bare &, %, # in LLM column text are escaped (a literal ampersand is a
    fatal pdflatex 'Misplaced alignment tab' otherwise); already-escaped
    specials and math/macros are left alone."""
    from generation.poster import _escape_latex_text_specials as esc
    assert esc("Microlensing & Timing") == r"Microlensing \& Timing"
    assert esc("50% done, item #1") == r"50\% done, item \#1"
    assert esc(r"keep \& and \% intact") == r"keep \& and \% intact"
    assert esc(r"$x^2$ \textbf{bold}") == r"$x^2$ \textbf{bold}"
    assert esc("") == ""


def test_cleanup_poster_artifacts_success_keeps_pdf_and_tex(tmp_path: Path) -> None:
    from generation.poster import _cleanup_poster_artifacts
    for name in ("poster.pdf", "poster.tex", "poster.aux", "poster.log",
                 "poster.out", "poster.nav", "poster.toc"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    _cleanup_poster_artifacts(tmp_path, keep_log=False)
    assert sorted(p.name for p in tmp_path.glob("poster.*")) == [
        "poster.pdf", "poster.tex",
    ]


def test_cleanup_poster_artifacts_failure_keeps_log(tmp_path: Path) -> None:
    from generation.poster import _cleanup_poster_artifacts
    for name in ("poster.tex", "poster.log", "poster.aux", "poster.out"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "poster_pdf_skipped.md").write_text("x", encoding="utf-8")
    _cleanup_poster_artifacts(tmp_path, keep_log=True)
    remaining = sorted(p.name for p in tmp_path.glob("poster*"))
    assert "poster.tex" in remaining            # source kept
    assert "poster.log" in remaining            # error log kept for debugging
    assert "poster_pdf_skipped.md" in remaining  # diagnostic kept
    assert "poster.aux" not in remaining         # scratch dropped
    assert "poster.out" not in remaining
