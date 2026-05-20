"""Tests for generation/paper.py.

These tests must NOT require pandoc/pdflatex. Every external invocation is
mocked via monkeypatch.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import Config
from core.engine import QuestArtifacts
from generation import paper as paper_mod
from generation.paper import PaperGenerator, _tighten_inline_math


# ---------------------------------------------------------------------------
# _tighten_inline_math — rescue LLM-emitted "$ \beta $" math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        # The exact pattern that broke quest 1779131901 — spaces around \beta.
        ("Align so the reported $ \\beta $ matches.",
         "Align so the reported $\\beta$ matches."),
        # Greek letter alone with trailing space only.
        ("Use $\\sigma $ as the scale.",
         "Use $\\sigma$ as the scale."),
        # Compound math expression with leading + trailing space.
        ("Set $ \\dfrac{a}{b} $ to one.",
         "Set $\\dfrac{a}{b}$ to one."),
    ],
)
def test_tighten_inline_math_strips_inner_whitespace(raw: str, expected: str) -> None:
    """Inline math containing a LaTeX command gets surrounding
    whitespace stripped so pandoc parses it as math instead of
    treating the dollars as literal characters."""
    assert _tighten_inline_math(raw) == expected


def test_tighten_leaves_already_tight_math_unchanged() -> None:
    """``$\\beta$`` (no spaces) — already correct, must not be mangled.
    Including the cross-span case where two tight math expressions
    sit on the same line with prose between them; the regex must NOT
    bridge them into one ``$ ... $`` span."""
    raw = "Use $\\beta$ here and $\\sigma$ there."
    assert _tighten_inline_math(raw) == raw


def test_tighten_does_not_touch_currency_dollars() -> None:
    """``$ 100`` has no backslash command — left untouched."""
    raw = "It costs $ 100 to $ 200 dollars."
    assert _tighten_inline_math(raw) == raw


def test_tighten_does_not_match_across_lines() -> None:
    """Inline math is single-line. A ``$ ... \n ... $`` pattern is
    almost certainly two separate dollars, not a math span."""
    raw = "Cost is $ 50.\nProfit is $ 10 high."
    assert _tighten_inline_math(raw) == raw


def test_tighten_handles_mixed_tight_and_loose_math() -> None:
    """A line with one tight + one loose math span tightens only the
    loose one."""
    raw = "Tight $\\alpha$ then loose $ \\beta $ end."
    expected = "Tight $\\alpha$ then loose $\\beta$ end."
    assert _tighten_inline_math(raw) == expected


def test_tighten_skips_inline_code_spans() -> None:
    """A literal ``$ \\beta $`` inside inline backticks is a code
    example, not math — pandoc would render it as code. The tightener
    must leave it alone so paper-about-LaTeX prose stays accurate."""
    raw = "In prose: $ \\beta $ should tighten. But in code: ``$ \\beta $`` should NOT change."
    out = _tighten_inline_math(raw)
    assert "$\\beta$ should tighten" in out
    assert "``$ \\beta $``" in out


def test_tighten_skips_fenced_code_blocks() -> None:
    """A fenced code block documenting a LaTeX example must survive
    untouched — even when the surrounding prose has a loose math span."""
    raw = (
        "Loose prose math: $ \\sigma $\n\n"
        "```latex\n"
        "Example: $ \\beta $ stays exactly as written.\n"
        "```\n"
        "After fence: $ \\gamma $ tightens."
    )
    out = _tighten_inline_math(raw)
    assert "$\\sigma$" in out
    assert "$\\gamma$" in out
    assert "Example: $ \\beta $ stays exactly as written." in out


def _make_config(tmp_path: Path, kinds: list[str], paper_format: str = "generic") -> Config:
    return Config.model_validate(
        {
            "topic": "t",
            "title": "test-title",
            "output": {
                "kinds": kinds,
                "paper_format": paper_format,
                "output_dir": str(tmp_path / "outputs"),
            },
        }
    )


def _make_artifacts(tmp_path: Path, *, with_figures: bool = False, with_manifest: bool = False) -> QuestArtifacts:
    quest_root = tmp_path / "quest"
    quest_root.mkdir(parents=True, exist_ok=True)
    paper_md = quest_root / "paper.md"
    paper_md.write_text("# Title\n\nbody\n", encoding="utf-8")

    figures_dir: Path | None = None
    if with_figures:
        figures_dir = quest_root / "figures"
        figures_dir.mkdir()
        (figures_dir / "fig1.png").write_bytes(b"\x89PNG\r\n")

    bundle_manifest: Path | None = None
    if with_manifest:
        bundle_manifest = quest_root / "bundle.json"
        bundle_manifest.write_text("{}", encoding="utf-8")

    return QuestArtifacts(
        quest_id="q1",
        quest_root=quest_root,
        paper_md=paper_md,
        figures_dir=figures_dir,
        bundle_manifest=bundle_manifest,
    )


def test_pandoc_missing_only_paper_md(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: None)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    out_dir = tmp_path / "out"

    result = PaperGenerator(cfg).generate(art, out_dir)

    assert "paper_md" in result
    assert result["paper_md"].exists()
    assert result["paper_md"].read_text(encoding="utf-8").startswith("# Title")
    assert "paper_pdf" not in result
    assert not (out_dir / "paper.pdf").exists()


def test_pandoc_rc_nonzero_no_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: "/fake/pandoc")

    def fake_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(returncode=1, stdout="", stderr="LaTeX error: missing.sty")

    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    out_dir = tmp_path / "out"

    result = PaperGenerator(cfg).generate(art, out_dir)
    assert "paper_pdf" not in result
    assert "paper_md" in result


def test_pandoc_happy_path_emits_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Return per-binary mocks so the test can assert that BOTH the
    # pandoc and the pdflatex paths flow into the subprocess command
    # (regression for: pandoc + MiKTeX both installed but pdflatex
    # missing from the child process's PATH).
    def fake_which(name):  # type: ignore[no-untyped-def]
        if name == "pandoc":
            return "/fake/pandoc.exe"
        if name == "pdflatex":
            return "/fake/pdflatex.exe"
        return None
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)

    out_dir = tmp_path / "out"
    captured_cmd: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        captured_cmd[:] = list(cmd)
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_bytes(b"%PDF-1.4 fake\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)

    result = PaperGenerator(cfg).generate(art, out_dir)
    assert "paper_pdf" in result
    assert result["paper_pdf"].name == "paper.pdf"
    assert result["paper_pdf"].exists()
    # Resolved binary paths flow into argv (no bare names that would
    # rely on the child process's PATH inheriting MiKTeX / pandoc dirs).
    assert captured_cmd[0] == "/fake/pandoc.exe"
    assert "--pdf-engine=/fake/pdflatex.exe" in captured_cmd


def test_missing_template_falls_back_to_pandoc_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: "/fake/pandoc")
    # Repoint TEMPLATES_DIR somewhere empty so no template.tex exists for any
    # paper_format. The PaperGenerator must still call pandoc — without
    # `--template` — and produce a PDF.
    empty_templates = tmp_path / "no-such-templates"
    monkeypatch.setattr(paper_mod, "TEMPLATES_DIR", empty_templates)

    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = list(cmd)
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_bytes(b"%PDF\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"], paper_format="neurips")
    art = _make_artifacts(tmp_path)
    out_dir = tmp_path / "out"

    result = PaperGenerator(cfg).generate(art, out_dir)
    assert "paper_pdf" in result
    assert "--template" not in captured["cmd"]


def test_pandoc_filenotfound_after_which_truthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`shutil.which` can succeed and `subprocess.run` still raise FNF if PATH
    changed between the two calls. Generator should swallow and skip PDF."""
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: "/fake/pandoc")

    def fake_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError(2, "No such file", "pandoc")

    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    out_dir = tmp_path / "out"

    result = PaperGenerator(cfg).generate(art, out_dir)
    assert "paper_pdf" not in result
    assert "paper_md" in result


def test_pandoc_timeout_skips_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: "/fake/pandoc")

    def fake_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(cmd="pandoc", timeout=120)

    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    out_dir = tmp_path / "out"

    result = PaperGenerator(cfg).generate(art, out_dir)
    assert "paper_pdf" not in result


def test_figures_copied_when_src_differs_from_dst(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: None)

    cfg = _make_config(tmp_path, ["paper_md"])
    art = _make_artifacts(tmp_path, with_figures=True)
    out_dir = tmp_path / "out"

    result = PaperGenerator(cfg).generate(art, out_dir)
    assert "figures_dir" in result
    assert (result["figures_dir"] / "fig1.png").exists()


def test_figure_copy_is_noop_when_src_equals_dst(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `out_dir/figures` resolves to the same path as `art.figures_dir`, the
    generator must NOT call rmtree+copytree (which would destroy the source)."""
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: None)

    # Set up: figures_dir IS literally `out_dir/figures`.
    out_dir = tmp_path / "shared"
    out_dir.mkdir()
    figures = out_dir / "figures"
    figures.mkdir()
    (figures / "fig1.png").write_bytes(b"\x89PNG\r\n")

    paper_md = tmp_path / "paper.md"
    paper_md.write_text("# t\n", encoding="utf-8")

    art = QuestArtifacts(
        quest_id="q1",
        quest_root=tmp_path,
        paper_md=paper_md,
        figures_dir=figures,
    )

    rmtree_calls: list[Path] = []
    copytree_calls: list[tuple[Path, Path]] = []
    real_rmtree = paper_mod.shutil.rmtree
    real_copytree = paper_mod.shutil.copytree

    def spy_rmtree(p, *a, **kw):  # type: ignore[no-untyped-def]
        rmtree_calls.append(Path(p))
        return real_rmtree(p, *a, **kw)

    def spy_copytree(s, d, *a, **kw):  # type: ignore[no-untyped-def]
        copytree_calls.append((Path(s), Path(d)))
        return real_copytree(s, d, *a, **kw)

    monkeypatch.setattr(paper_mod.shutil, "rmtree", spy_rmtree)
    monkeypatch.setattr(paper_mod.shutil, "copytree", spy_copytree)

    cfg = _make_config(tmp_path, ["paper_md"])
    result = PaperGenerator(cfg).generate(art, out_dir)

    assert "figures_dir" in result
    assert (figures / "fig1.png").exists(), "source figures must not be deleted"
    assert rmtree_calls == [], f"rmtree should NOT be called, got {rmtree_calls}"
    assert copytree_calls == [], f"copytree should NOT be called, got {copytree_calls}"


def test_figure_copy_noop_when_paths_differ_in_spelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The figure-copy guard uses `.resolve()` so two Path spellings that
    point at the same physical directory (e.g. one absolute, one relative
    via `..`) short-circuit and don't rmtree the source."""
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: None)

    out_dir = (tmp_path / "shared").resolve()
    out_dir.mkdir()
    figures_real = out_dir / "figures"
    figures_real.mkdir()
    (figures_real / "fig1.png").write_bytes(b"\x89PNG\r\n")

    # Alternate spelling of the same directory: go up and back down.
    figures_alt = out_dir / "figures" / ".." / "figures"
    assert figures_real != figures_alt  # naive equality differs
    assert figures_real.resolve() == figures_alt.resolve()  # resolve matches

    paper_md = tmp_path / "paper.md"
    paper_md.write_text("# t\n", encoding="utf-8")
    art = QuestArtifacts(
        quest_id="q1",
        quest_root=tmp_path,
        paper_md=paper_md,
        figures_dir=figures_alt,
    )

    rmtree_calls: list[Path] = []
    monkeypatch.setattr(
        paper_mod.shutil, "rmtree", lambda p, *a, **kw: rmtree_calls.append(Path(p))
    )
    copytree_calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        paper_mod.shutil,
        "copytree",
        lambda s, d, *a, **kw: copytree_calls.append((Path(s), Path(d))),
    )

    cfg = _make_config(tmp_path, ["paper_md"])
    PaperGenerator(cfg).generate(art, out_dir)

    assert (figures_real / "fig1.png").exists()
    assert rmtree_calls == []
    assert copytree_calls == []


def test_bundle_manifest_copied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: None)

    cfg = _make_config(tmp_path, ["paper_md"])
    art = _make_artifacts(tmp_path, with_manifest=True)
    out_dir = tmp_path / "out"

    result = PaperGenerator(cfg).generate(art, out_dir)
    assert "bundle_manifest" in result
    assert result["bundle_manifest"].name == "paper_bundle_manifest.json"
    assert result["bundle_manifest"].read_text(encoding="utf-8") == "{}"


def test_paper_pdf_not_in_kinds_skips_compile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `paper_pdf` is not requested, _compile_pdf must not be called even
    if pandoc is on PATH."""
    called = {"n": 0}

    def fake_which(_c):  # type: ignore[no-untyped-def]
        called["n"] += 1
        return "/fake/pandoc"

    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)

    cfg = _make_config(tmp_path, ["paper_md"])  # no paper_pdf
    art = _make_artifacts(tmp_path)
    out_dir = tmp_path / "out"

    result = PaperGenerator(cfg).generate(art, out_dir)
    assert "paper_pdf" not in result
    assert called["n"] == 0  # which() never invoked because compile path skipped


def test_no_pdf_engine_skips_pdf_with_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pandoc present but NEITHER pdflatex NOR tectonic available:
    skip the PDF cleanly (no subprocess invocation, no traceback)."""
    def fake_which(name):  # type: ignore[no-untyped-def]
        return "/fake/pandoc.exe" if name == "pandoc" else None
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)
    # Repoint REPO_ROOT to an empty dir so no tools/tectonic.exe is found.
    monkeypatch.setattr(paper_mod, "REPO_ROOT", tmp_path)

    fake_run_called = {"n": 0}

    def fake_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        fake_run_called["n"] += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    result = PaperGenerator(cfg).generate(art, tmp_path / "out")
    assert "paper_pdf" not in result
    # No engine = no pandoc spawn at all (vs. spawning and failing
    # mid-run, which would waste time on a bigger paper).
    assert fake_run_called["n"] == 0


def test_tectonic_fallback_when_pdflatex_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The corporate-laptop case: pdflatex isn't on PATH, but tectonic
    is. The generator picks tectonic and passes its full path to
    pandoc via `--pdf-engine`."""
    def fake_which(name):  # type: ignore[no-untyped-def]
        if name == "pandoc":
            return "/fake/pandoc.exe"
        if name == "pdflatex":
            return None
        if name == "tectonic":
            return "/fake/tectonic.exe"
        return None
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)

    captured_cmd: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        captured_cmd[:] = list(cmd)
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_bytes(b"%PDF\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    result = PaperGenerator(cfg).generate(art, tmp_path / "out")
    assert "paper_pdf" in result
    assert "--pdf-engine=/fake/tectonic.exe" in captured_cmd


def test_pdflatex_preferred_over_tectonic_when_both_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordering contract: pdflatex wins when both are on PATH.
    Warm MiKTeX caches compile in 1-3 s; tectonic's first-run
    package fetch is slower. Flipping the order would penalize the
    common case to help the corporate-laptop minority."""
    def fake_which(name):  # type: ignore[no-untyped-def]
        if name == "pandoc":
            return "/fake/pandoc.exe"
        if name == "pdflatex":
            return "/fake/pdflatex.exe"
        if name == "tectonic":
            return "/fake/tectonic.exe"
        return None
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)

    captured_cmd: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        captured_cmd[:] = list(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"%PDF\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    PaperGenerator(
        _make_config(tmp_path, ["paper_md", "paper_pdf"])
    ).generate(_make_artifacts(tmp_path), tmp_path / "out")
    assert "--pdf-engine=/fake/pdflatex.exe" in captured_cmd
    assert "--pdf-engine=/fake/tectonic.exe" not in captured_cmd


def test_repo_local_tectonic_picked_when_nothing_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User ran `python launch.py --install-tectonic` — the binary
    lives at `<repo_root>/tools/tectonic.exe` (or `/tectonic` on
    POSIX) and PATH is bare. Generator must find the repo-local copy."""
    monkeypatch.setattr(paper_mod, "REPO_ROOT", tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    # ``_find_pdf_engine`` checks ONLY the platform-appropriate
    # extension (``.exe`` on Windows, bare on POSIX) — see
    # generation/paper.py near "platform-appropriate filename". The
    # test must mirror that so it works on both Linux CI and Windows
    # CI; checking only ``.exe`` made the assertion silently fail on
    # the ubuntu-latest matrix leg because the engine returned None
    # and pandoc never ran.
    tectonic_name = "tectonic.exe" if sys.platform == "win32" else "tectonic"
    repo_tectonic = tools / tectonic_name
    repo_tectonic.write_bytes(b"#!/bin/sh\necho fake\n")

    def fake_which(name):  # type: ignore[no-untyped-def]
        return "/fake/pandoc.exe" if name == "pandoc" else None
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)

    captured_cmd: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        captured_cmd[:] = list(cmd)
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"%PDF\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    PaperGenerator(
        _make_config(tmp_path, ["paper_md", "paper_pdf"])
    ).generate(_make_artifacts(tmp_path), tmp_path / "out")
    # The repo-local path got picked.
    assert f"--pdf-engine={repo_tectonic}" in captured_cmd


# ---- paper_pdf_skipped.md diagnostic file -----------------------------------


def test_paper_pdf_skipped_md_written_when_pandoc_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the user requested ``paper_pdf`` but pandoc is missing, the
    generator must write ``paper_pdf_skipped.md`` next to ``paper.md``
    so the user discovers the failure without grepping ``run.log``.

    A host with none of pandoc / pdflatex / tectonic installed would
    otherwise produce zero indication that the PDF was requested but
    skipped — the marker file makes that visible from the quest
    output without grepping run.log."""
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: None)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    out_dir = tmp_path / "out"

    result = PaperGenerator(cfg).generate(art, out_dir)

    # PDF wasn't produced.
    assert "paper_pdf" not in result
    assert not (out_dir / "paper.pdf").exists()

    # The diagnostic file WAS produced.
    diag = out_dir / "paper_pdf_skipped.md"
    assert diag.exists(), "paper_pdf_skipped.md was not written"
    body = diag.read_text(encoding="utf-8")

    # The user needs to know three things:
    assert "paper.pdf was requested but not produced" in body
    assert "no_pandoc" in body                 # reason code
    assert "pandoc not on PATH" in body        # what happened
    assert "winget" in body or "brew" in body or "pandoc.org" in body  # how to fix

    # And the result dict surfaces the diagnostic path so callers
    # (launch.py / the VSCode bridge) can include it in their
    # "your quest is done" messages.
    assert "paper_pdf_skipped" in result
    assert result["paper_pdf_skipped"] == diag


def test_paper_pdf_skipped_md_written_when_no_latex_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different skip reason: pandoc is installed but no LaTeX engine
    is on PATH. The diagnostic should name THAT failure mode and
    point at ``--install-tectonic`` as the no-admin fix."""

    def fake_which(name):  # type: ignore[no-untyped-def]
        return "/fake/pandoc" if name == "pandoc" else None
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)
    # ``_find_pdf_engine`` also checks REPO_ROOT/tools/tectonic.exe
    # as a fallback (the path populated by `--install-tectonic`).
    # If a real tectonic happens to be sitting there on the dev box,
    # the test thinks pandoc-only is fine and skips the diagnostic.
    # Repoint REPO_ROOT at a clean tmp dir so the lookup misses.
    monkeypatch.setattr(paper_mod, "REPO_ROOT", tmp_path)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    out_dir = tmp_path / "out"

    PaperGenerator(cfg).generate(art, out_dir)

    diag = out_dir / "paper_pdf_skipped.md"
    assert diag.exists()
    body = diag.read_text(encoding="utf-8")
    assert "no_latex_engine" in body
    assert "--install-tectonic" in body


def test_paper_pdf_skipped_md_carries_engine_stderr_on_compile_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When pandoc+engine are present but the compile fails, the
    diagnostic must include the engine's stderr tail so the user can
    actually diagnose the LaTeX issue without re-running."""

    def fake_which(name):  # type: ignore[no-untyped-def]
        if name == "pandoc":
            return "/fake/pandoc"
        if name == "pdflatex":
            return "/fake/pdflatex"
        return None
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)

    def fake_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            returncode=1, stdout="",
            stderr="! LaTeX Error: File `missing.sty' not found.\n",
        )
    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    PaperGenerator(cfg).generate(_make_artifacts(tmp_path), tmp_path / "out")

    diag = tmp_path / "out" / "paper_pdf_skipped.md"
    assert diag.exists()
    body = diag.read_text(encoding="utf-8")
    assert "pdflatex_rc_1" in body
    assert "missing.sty" in body, (
        "stderr tail must be embedded so the user can fix LaTeX errors"
    )


def test_paper_pdf_skipped_md_not_written_when_pdf_not_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``paper_pdf`` is not in ``output.kinds``, no skip happened —
    the absence of paper.pdf is intentional. Don't pollute the dir
    with a misleading diagnostic file."""
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: None)

    cfg = _make_config(tmp_path, ["paper_md"])    # no paper_pdf
    PaperGenerator(cfg).generate(_make_artifacts(tmp_path), tmp_path / "out")

    diag = tmp_path / "out" / "paper_pdf_skipped.md"
    assert not diag.exists(), (
        "skip diagnostic was written even though paper_pdf was not requested"
    )


def test_paper_pdf_skipped_md_cleaned_up_on_subsequent_md_only_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a prior run wrote ``paper_pdf_skipped.md`` and the user then
    edited the config to remove ``paper_pdf`` from kinds, the stale
    diagnostic should be cleaned up — otherwise the file lingers and
    confuses readers."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "paper_pdf_skipped.md"
    stale.write_text("(stale from prior run)\n", encoding="utf-8")

    # No pandoc, but also no paper_pdf in kinds — generator should
    # take the cleanup branch.
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: None)
    cfg = _make_config(tmp_path, ["paper_md"])
    PaperGenerator(cfg).generate(_make_artifacts(tmp_path), out_dir)
    assert not stale.exists(), "stale skip diagnostic was not cleaned up"


def test_paper_pdf_skipped_md_removed_when_subsequent_run_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a previous run wrote ``paper_pdf_skipped.md`` (engine
    missing) and a later run succeeds (engine now installed), the
    stale diagnostic must be removed so the quest dir doesn't
    simultaneously report a paper.pdf AND a 'skipped' marker."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "paper_pdf_skipped.md"
    stale.write_text("(stale from prior failed run)\n", encoding="utf-8")

    # Now: pandoc + pdflatex both available, compile succeeds.
    def fake_which(name):  # type: ignore[no-untyped-def]
        return f"/fake/{name}" if name in ("pandoc", "pdflatex") else None
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_bytes(b"%PDF-1.4 fake\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    result = PaperGenerator(cfg).generate(_make_artifacts(tmp_path), out_dir)

    # PDF was produced.
    assert "paper_pdf" in result
    assert (out_dir / "paper.pdf").exists()
    # Stale skip diagnostic from the prior failed run is GONE.
    assert not stale.exists(), (
        "stale paper_pdf_skipped.md was not removed after success"
    )


def test_paper_pdf_skipped_md_overwritten_on_subsequent_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second run with a DIFFERENT skip reason must overwrite the
    diagnostic, not append. The latest attempt is the user-relevant
    one."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: None)
    # Repoint the repo-local tectonic lookup at a clean dir so a real
    # `tools/tectonic.exe` on the dev box can't shadow the test.
    monkeypatch.setattr(paper_mod, "REPO_ROOT", tmp_path)
    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])

    # Run 1: no pandoc → no_pandoc reason.
    PaperGenerator(cfg).generate(_make_artifacts(tmp_path), out_dir)
    assert "no_pandoc" in (out_dir / "paper_pdf_skipped.md").read_text(encoding="utf-8")

    # Run 2: pandoc now found, but no LaTeX engine → no_latex_engine
    # reason should fully REPLACE the prior file's content.
    def fake_which(name):  # type: ignore[no-untyped-def]
        return "/fake/pandoc" if name == "pandoc" else None
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)
    PaperGenerator(cfg).generate(_make_artifacts(tmp_path), out_dir)
    body = (out_dir / "paper_pdf_skipped.md").read_text(encoding="utf-8")
    assert "no_latex_engine" in body
    assert "no_pandoc" not in body, "prior skip reason was not overwritten"


# ---- output.require_pdf strict mode (compile-path enforcement) -----------
#
# These tests cover the SECOND half of the require_pdf contract: the
# pre-flight in core/engine.py catches missing prerequisites BEFORE the
# LLM runs (test_engine_helpers.py owns that). PaperGenerator.generate
# is responsible for the AFTER-LLM half: when require_pdf=True and the
# PDF still can't be compiled (timeout, nonzero LaTeX rc, missing
# output despite rc=0), raise RuntimeError instead of silently writing
# only paper_pdf_skipped.md.


def _make_config_strict(tmp_path: Path) -> Config:
    """Same as _make_config but with require_pdf=True."""
    return Config.model_validate(
        {
            "topic": "t",
            "title": "test-title",
            "output": {
                "kinds": ["paper_md", "paper_pdf"],
                "paper_format": "generic",
                "output_dir": str(tmp_path / "outputs"),
                "require_pdf": True,
            },
        }
    )


def test_require_pdf_raises_when_pandoc_invocation_fails_post_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``shutil.which`` resolved pandoc (preflight would pass), but
    ``subprocess.run`` raises ``FileNotFoundError`` at compile time
    (e.g. PATH disagreement between resolver and child process). With
    require_pdf=True the generator must raise — it must NOT silently
    write paper_pdf_skipped.md and let the quest report success."""
    def fake_which(name):  # type: ignore[no-untyped-def]
        return "/fake/pandoc" if name == "pandoc" else "/fake/pdflatex"
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)

    def boom(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("pandoc")
    monkeypatch.setattr(paper_mod.subprocess, "run", boom)

    cfg = _make_config_strict(tmp_path)
    art = _make_artifacts(tmp_path)
    out_dir = tmp_path / "out"

    with pytest.raises(RuntimeError) as ei:
        PaperGenerator(cfg).generate(art, out_dir)
    assert "output.require_pdf=True" in str(ei.value)
    assert "pandoc_invocation_failed" in str(ei.value)
    # Diagnostic file must still be written for the user to read after
    # the exception is caught upstream — the RuntimeError points at it.
    assert (out_dir / "paper_pdf_skipped.md").exists()


def test_require_pdf_raises_on_compile_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tectonic / pdflatex compile exceeded the timeout. With strict
    mode, the user gets a hard failure — not a "completed" quest
    pointing at a paper_pdf_skipped.md."""
    def fake_which(name):  # type: ignore[no-untyped-def]
        return "/fake/pandoc" if name == "pandoc" else "/fake/pdflatex"
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)

    def fake_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(cmd="pandoc", timeout=300)
    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    cfg = _make_config_strict(tmp_path)
    art = _make_artifacts(tmp_path)
    out_dir = tmp_path / "out"

    with pytest.raises(RuntimeError) as ei:
        PaperGenerator(cfg).generate(art, out_dir)
    assert "output.require_pdf=True" in str(ei.value)
    assert "timeout" in str(ei.value).lower()


def test_require_pdf_raises_on_nonzero_latex_rc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LaTeX compile failed with rc=1 (e.g. malformed template, missing
    CTAN package). Strict mode surfaces it as a quest-level failure."""
    def fake_which(name):  # type: ignore[no-untyped-def]
        return "/fake/pandoc" if name == "pandoc" else "/fake/pdflatex"
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)
    monkeypatch.setattr(
        paper_mod.subprocess, "run",
        lambda *_a, **_kw: SimpleNamespace(
            returncode=1, stdout="", stderr="! Undefined control sequence.",
        ),
    )

    cfg = _make_config_strict(tmp_path)
    art = _make_artifacts(tmp_path)
    out_dir = tmp_path / "out"

    with pytest.raises(RuntimeError) as ei:
        PaperGenerator(cfg).generate(art, out_dir)
    assert "output.require_pdf=True" in str(ei.value)
    assert "rc_1" in str(ei.value) or "rc=1" in str(ei.value)


def test_require_pdf_false_keeps_silent_skip_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default mode (require_pdf=False) MUST keep the existing
    graceful-skip-with-diagnostic behavior. Regression guard for the
    contract: strict mode is OPT-IN, not the new default."""
    def fake_which(name):  # type: ignore[no-untyped-def]
        return "/fake/pandoc" if name == "pandoc" else "/fake/pdflatex"
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)
    monkeypatch.setattr(
        paper_mod.subprocess, "run",
        lambda *_a, **_kw: SimpleNamespace(
            returncode=1, stdout="", stderr="! LaTeX error.",
        ),
    )

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])  # require_pdf default False
    art = _make_artifacts(tmp_path)
    out_dir = tmp_path / "out"

    # No raise — graceful skip with diagnostic. Quest can finish.
    result = PaperGenerator(cfg).generate(art, out_dir)
    assert "paper_pdf" not in result
    assert "paper_pdf_skipped" in result
    assert (out_dir / "paper_pdf_skipped.md").exists()


# ---- Template-substitution-leak regression -------------------------------
#
# Two bugs we guard against: (a) paper.pdf compile failed with
# "Environment Shaded undefined" + "\\tightlist undefined" because
# pandoc emits those commands but our custom templates didn't define
# them; (b) poster.tex compile failed because the template's
# top-of-file comment ``% Substitutions (Python string.Template):
# $title, $left, ...`` was itself substituted by ``safe_substitute``,
# and the multi-line ``$left`` content injected into the comment
# broke the comment boundary, pushing later lines (including
# ``\documentclass``) past raw LaTeX that pdflatex couldn't parse.


def test_paper_templates_declare_pandocbounded_for_image_bounding() -> None:
    """Pins that every shipped paper template carries a complete
    ``\\providecommand{\\pandocbounded}`` definition, body included.

    Background: ``\\pandocbounded`` is emitted by pandoc to constrain
    inline images / float environments to the text width. The
    standard pandoc latex template defines it; when FI overrides
    with a custom template, we have to re-declare or pdflatex
    errors with ``! Undefined control sequence. l.NNN
    \\pandocbounded``.

    Bot-review-strengthened: assert the FULL body
    ``\\noindent\\resizebox{\\textwidth}{!}{#1}`` (not just the
    command name) so a corrupted body — e.g. via shell-escape
    collapsing ``\\noindent`` into a literal newline + ``oindent``
    — fails the test. That's the exact regression we'd ship if
    someone re-applied the change via a brittle ``python -c``
    one-liner."""
    repo = Path(__file__).resolve().parent.parent
    working = ["generic", "neurips", "essay", "report", "policy_brief", "whitepaper"]
    expected = (
        r"\providecommand{\pandocbounded}[1]"
        r"{\noindent\resizebox{\textwidth}{!}{#1}}"
    )
    for fmt in working:
        path = repo / "templates" / "paper" / fmt / "template.tex"
        assert path.exists(), f"working template {fmt!r} is missing"
        body = path.read_text(encoding="utf-8")
        non_comment = "\n".join(
            line for line in body.splitlines() if not line.lstrip().startswith("%")
        )
        assert expected in non_comment, (
            f"templates/paper/{fmt}/template.tex must contain the EXACT "
            f"\\providecommand{{\\pandocbounded}} body — including the "
            f"\\noindent / \\resizebox / \\textwidth control sequences. "
            f"A name-only check would pass even when the body is "
            f"corrupted into literal text (newline + 'oindent' etc.) "
            f"and pdflatex would silently emit a PDF with stray text "
            f"and unbounded images."
        )


def test_paper_templates_declare_pandoc_compatibility_macros() -> None:
    """Every working paper template must include \\providecommand for
    \\tightlist and \\passthrough plus the pandoc highlighting-macros
    expansion. Without these, paper.md containing fenced code blocks
    or compact lists fails pdflatex compile (the exact regression
    that broke a user-reported quest pre-hotfix)."""
    repo = Path(__file__).resolve().parent.parent
    working = ["generic", "neurips", "essay", "report", "policy_brief", "whitepaper"]
    # Pandoc's variable expansion that pulls in the Shaded env + token
    # color commands. The literal directive we want pandoc to see is
    # the bare-name form sandwiched between two ``$`` sigils; we build
    # it here from a placeholder concat so this assertion line itself
    # is never confused with a pandoc directive by editors/linters.
    sigil = "$"
    directive = f"{sigil}highlighting-macros{sigil}"
    for fmt in working:
        path = repo / "templates" / "paper" / fmt / "template.tex"
        assert path.exists(), f"working template {fmt!r} is missing"
        body = path.read_text(encoding="utf-8")
        assert "\\providecommand{\\tightlist}" in body, (
            f"templates/paper/{fmt}/template.tex must \\providecommand "
            f"\\tightlist — pandoc emits it inside compact lists and "
            f"the default-template definition is lost when we override."
        )
        assert "\\providecommand{\\passthrough}" in body, (
            f"templates/paper/{fmt}/template.tex must \\providecommand "
            f"\\passthrough — pandoc emits it around inline raw-blocks."
        )
        # Strip ``%``-comments before checking for the actual template
        # directive — the templates have inline comments that mention
        # ``highlighting-macros`` by name, so a bare substring search
        # would pass even if the real ``$highlighting-macros$`` line
        # got deleted. That is the exact regression this test guards.
        non_comment = "\n".join(
            line for line in body.splitlines() if not line.lstrip().startswith("%")
        )
        assert directive in non_comment, (
            f"templates/paper/{fmt}/template.tex must include the real "
            f"pandoc {directive} template directive on a non-comment "
            f"line (Shaded env + token colors). Found only in comments "
            f"or not at all."
        )


def test_paper_templates_dont_spell_pandoc_vars_in_comments() -> None:
    """Pandoc substitutes ``$var$`` tokens everywhere in the template
    — including inside ``%``-comments. When the substituted value
    line-wraps (e.g. a long paper title spilling onto a second line),
    the leading ``%`` only comments the first line, and every later
    line lands as raw LaTeX. Most of the time that raw LaTeX appears
    BEFORE ``\\begin{document}`` and pdflatex dies with
    "Missing \\begin{document}".

    This test pins the contract: NO comment line in any paper
    template may spell out the pandoc variables we know wrap onto
    multiple lines. The matching note inside each template already
    warns future editors; this assertion makes the warning
    enforceable."""
    repo = Path(__file__).resolve().parent.parent
    venues = (
        "generic", "neurips", "iclr", "ieee_access", "nature_mi",
        "essay", "report", "policy_brief", "whitepaper",
    )
    # `$body$` and `$title$` are the two that reliably wrap.
    # `$abstract$` is by definition a paragraph that wraps. The poster
    # template has its own trap but that is covered separately. We
    # keep this list explicit (not a regex over all $...$) so a
    # legitimate substitution like `$if(abstract)$` in a comment —
    # which is fine because the body it gates is short — doesn't
    # trigger a false positive.
    forbidden = ("$title$", "$body$", "$abstract$")
    for fmt in venues:
        path = repo / "templates" / "paper" / fmt / "template.tex"
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.lstrip()
            if not stripped.startswith("%"):
                continue
            for tok in forbidden:
                assert tok not in stripped, (
                    f"templates/paper/{fmt}/template.tex line {lineno} "
                    f"spells {tok!r} inside a `%` comment. Pandoc will "
                    f"substitute the value, and if it wraps onto a "
                    f"second line, pdflatex dies with "
                    f"'Missing \\\\begin{{document}}'. Refer to the "
                    f"variable indirectly in the comment instead."
                )


def test_paper_venue_templates_are_real_not_stubs() -> None:
    """The iclr/ieee_access/nature_mi templates must be real,
    minimal venue-flavored templates — 1-line LaTeX-comment stubs
    pass through pandoc but pdflatex rejects them at compile.
    Each template must contain the placeholders pandoc needs
    (``$title$``, ``$body$``) and at least one ``\\documentclass``
    line so it actually compiles. Future stub regressions (a 1-line
    comment-only template, an empty file) fail here loudly instead
    of silently producing an unreadable PDF."""
    repo = Path(__file__).resolve().parent.parent
    for fmt in ("iclr", "ieee_access", "nature_mi"):
        path = repo / "templates" / "paper" / fmt / "template.tex"
        assert path.exists(), (
            f"templates/paper/{fmt}/template.tex must exist as a real "
            f"venue template — `paper_format: {fmt}` quests fall back "
            f"to pandoc's generic default when the file is missing."
        )
        body = path.read_text(encoding="utf-8")
        # Strip LaTeX comments so a stub like "% TODO" can't pass.
        non_comment = "\n".join(
            line for line in body.splitlines() if not line.lstrip().startswith("%")
        )
        for needle in ("\\documentclass", "$title$", "$body$"):
            assert needle in non_comment, (
                f"templates/paper/{fmt}/template.tex is missing "
                f"{needle!r} on a non-comment line — pandoc cannot "
                f"render a paper without all three. A pure-comment "
                f"file is the failure mode this guard prevents."
            )


def test_poster_template_substitution_does_not_leak_into_preamble() -> None:
    """Regression guard. The poster template's top
    comment used to read
    ``% Substitutions (Python string.Template): $title, $left, ...``
    — and ``string.Template.safe_substitute`` happily expanded the
    bare dollar names inside the comment. When the LLM's ``$left``
    content was multi-line (a column wrapper with its own
    ``\\begin{column}`` ... ``\\end{column}``), the second+ lines
    landed outside the ``%`` comment and became raw LaTeX BEFORE
    ``\\documentclass`` — pdflatex then errored on the first
    ``\\textbf`` it saw with "Undefined control sequence".

    This test pins the fix: after substitution with a multi-line
    ``left`` value, ``\\documentclass`` must still appear AFTER all
    the comment lines, AND the rendered file must not contain raw
    LaTeX commands before ``\\documentclass``."""
    import string

    repo = Path(__file__).resolve().parent.parent
    tmpl = (repo / "templates" / "poster" / "poster.tex").read_text(encoding="utf-8")

    out = string.Template(tmpl).safe_substitute(
        title="Cross-Country Achievement",
        left="\\begin{column}{0.32\\textwidth}\nLEFT CONTENT\n\\end{column}",
        middle="middle",
        right="right",
    )
    # \documentclass must appear in the output and must be preceded
    # only by comment lines (so the preamble is intact).
    doc_idx = out.find("\\documentclass")
    assert doc_idx > 0, "rendered poster.tex must contain \\documentclass"
    before = out[:doc_idx]
    for line in before.splitlines():
        stripped = line.lstrip()
        if not stripped:
            continue
        assert stripped.startswith("%"), (
            f"rendered poster.tex has non-comment raw LaTeX before "
            f"\\documentclass: {line!r} — the comment-substitution "
            f"leak has regressed."
        )


# ---- Unicode sanitization regression -------------------------------------
#
# User-reported quest 1778878424-euv-stochastics-c339a6 failed pdflatex
# with "Unicode character ≈ (U+2248) not set up for use with LaTeX".
# The paper.md contained ×, —, −, ≈, ≥ — all common LLM emissions that
# pdflatex + inputenc utf8 don't handle without explicit mappings. The
# fix pre-processes paper.md before pandoc reads it, rewriting each
# glyph to its \\ensuremath-wrapped LaTeX command.


def test_sanitize_unicode_replaces_common_math_glyphs() -> None:
    """The five glyphs from the user's failing quest must all rewrite
    to LaTeX-safe forms after the sanitizer runs."""
    src = "FNR ≈ 0.24, sample × 32, threshold ≥ 0.5, 10⁴ periods — note: −5°C."
    out = paper_mod._sanitize_unicode_for_latex(src)
    # Each problematic glyph rewritten.
    for ch in ("×", "—", "−", "≈", "≥"):
        assert ch not in out, (
            f"{ch!r} still present after sanitize — pdflatex would error"
        )
    # \ensuremath is the wrapper we picked so it works in both text
    # and math contexts. Spot-check a couple.
    assert r"\ensuremath{\approx}" in out
    assert r"\ensuremath{\geq}" in out
    # Em dash → triple-hyphen (standard LaTeX text-mode em dash).
    assert "---" in out


def test_sanitize_unicode_covers_greek_letters_and_superscripts() -> None:
    """LLMs use these bare in prose like ``α = 0.05`` and ``10⁴``."""
    src = "α = 0.05, σ² = 0.1, λ = 2.3, sample 10⁴, Δx = 0.01"
    out = paper_mod._sanitize_unicode_for_latex(src)
    for ch in ("α", "σ", "²", "λ", "⁴", "Δ"):
        assert ch not in out, f"{ch!r} not rewritten"
    assert r"\ensuremath{\alpha}" in out
    assert r"\ensuremath{\sigma}" in out
    assert r"\ensuremath{\Delta}" in out
    assert r"\ensuremath{^{2}}" in out
    assert r"\ensuremath{^{4}}" in out


def test_sanitize_unicode_outside_code_helper_preserves_code_blocks() -> None:
    """The ``_outside_code_blocks`` helper exists as a documented
    option for callers that legitimately need prose-only rewriting.
    The PDF pipeline does NOT use it (sanitizes everything — see
    the ``compile_pdf`` test below) because the templates lack a
    ``\\DeclareUnicodeCharacter`` safety net. This test just pins
    the helper's prose-only contract for any future consumer."""
    src = (
        "Text with ≈ outside.\n"
        "```\n"
        "print('inside ≈ still raw')\n"
        "```\n"
        "More text with ≥.\n"
    )
    out = paper_mod._sanitize_unicode_outside_code_blocks(src)
    # Outside-code occurrences rewritten.
    assert r"\ensuremath{\approx}" in out
    assert r"\ensuremath{\geq}" in out
    # Inside-code occurrence preserved.
    assert "'inside ≈ still raw'" in out


def test_sanitize_unicode_preserves_inline_code() -> None:
    """`backtick code` is also preserved verbatim — same reasoning."""
    src = "Use `f(x) ≈ x²` to compute the result."
    out = paper_mod._sanitize_unicode_outside_code_blocks(src)
    assert "`f(x) ≈ x²`" in out, "inline code must not be sanitized"


def test_sanitize_unicode_is_idempotent() -> None:
    """Running the sanitizer twice produces the same string the first
    pass did — important because the user's saved paper.md may itself
    have been generated by an earlier run of this code."""
    src = "FNR ≈ 0.24, ≥ 0.5"
    once = paper_mod._sanitize_unicode_for_latex(src)
    twice = paper_mod._sanitize_unicode_for_latex(once)
    assert once == twice


def test_compile_pdf_writes_sanitized_md_alongside_paper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: when paper.md contains the user's reported glyphs,
    ``_compile_pdf`` produces a ``paper_pdf_source.md`` next to the
    PDF carrying the sanitized content, and pandoc is invoked against
    THAT file — not the original."""
    def fake_which(name):  # type: ignore[no-untyped-def]
        return "/fake/pandoc" if name == "pandoc" else "/fake/pdflatex" if name == "pdflatex" else None
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)

    captured_input: list[str] = []

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        # The pandoc input arg is the second element (cmd[1]).
        captured_input.append(cmd[1])
        # Write a fake PDF at the -o target.
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_bytes(b"%PDF\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    art.paper_md.write_text(
        "Result: FNR ≈ 0.24, sample × 32 — see Δx.\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    PaperGenerator(cfg).generate(art, out_dir)

    # The pandoc input was the sanitized copy, not the original.
    assert captured_input, "subprocess.run was never called"
    assert "paper_pdf_source.md" in captured_input[0], (
        f"pandoc must consume the sanitized copy; got {captured_input[0]!r}"
    )
    sanitized = (out_dir / "paper_pdf_source.md").read_text(encoding="utf-8")
    assert "≈" not in sanitized
    assert "×" not in sanitized
    assert "Δ" not in sanitized
    assert r"\ensuremath{\approx}" in sanitized
    # Original paper.md (the user-facing archive copy) is unchanged.
    orig = (out_dir / "paper.md").read_text(encoding="utf-8")
    assert "≈" in orig, "user-facing paper.md must keep the original glyphs"


def test_count_sanitized_glyphs_returns_source_count() -> None:
    """Honest counter: the INFO log reports source-glyph occurrences,
    not length-deltas. Deriving the count from length deltas would
    go negative because LaTeX replacements (~20 chars) are longer
    than the source glyphs (1 char each)."""
    assert paper_mod._count_sanitized_glyphs("plain ASCII only") == 0
    # 3 distinct glyphs, each appearing once → 3.
    assert paper_mod._count_sanitized_glyphs("FNR ≈ x, ≥ y, × z") == 3
    # 3 ≈ occurrences → 3 (not 1).
    assert paper_mod._count_sanitized_glyphs("≈ a ≈ b ≈ c") == 3


def test_compile_pdf_sanitizes_unicode_inside_code_blocks_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: an earlier version sanitized only prose, leaving
    raw glyphs inside fenced/inline code. pdflatex still errored on
    those because the templates don't have a
    ``\\DeclareUnicodeCharacter`` safety net. The PDF pipeline now
    sanitizes everything — what lands in ``paper_pdf_source.md``
    has zero leftover unicode from the source paper.md, including
    inside code blocks."""
    def fake_which(name):  # type: ignore[no-untyped-def]
        return "/fake/pandoc" if name == "pandoc" else "/fake/pdflatex" if name == "pdflatex" else None
    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_bytes(b"%PDF\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    art.paper_md.write_text(
        "Text ≈ outside.\n"
        "```\n"
        "print('inside ≈ would have killed pdflatex')\n"
        "```\n"
        "Inline `f(x) ≥ y` too.\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    PaperGenerator(cfg).generate(art, out_dir)

    sanitized = (out_dir / "paper_pdf_source.md").read_text(encoding="utf-8")
    assert "≈" not in sanitized, (
        "raw ≈ inside code blocks must also be rewritten — "
        "templates lack a \\DeclareUnicodeCharacter safety net"
    )
    assert "≥" not in sanitized


def test_compile_pdf_clears_stale_source_md_on_early_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-existing ``paper_pdf_source.md`` from a prior SUCCESSFUL
    run must be cleared at the START of ``_compile_pdf`` so an
    early-out skip (e.g. pandoc missing) doesn't leave a stale file
    masquerading as the source of the just-failed attempt. THIS run
    didn't sanitize anything, so the only correct state for
    ``paper_pdf_source.md`` is "absent"."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "paper_pdf_source.md"
    stale.write_text("from a previous successful run", encoding="utf-8")

    # pandoc missing → early-out skip path (returns BEFORE sanitization).
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: None)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    PaperGenerator(cfg).generate(art, out_dir)
    assert not stale.exists()


def test_compile_pdf_preserves_source_md_on_post_sanitize_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When pandoc runs but exits non-zero (e.g. a LaTeX template bug),
    ``paper_pdf_source.md`` must survive — it's the exact markdown
    pandoc consumed and choked on, the single most useful artifact
    for post-mortem. Regression test for the deletion-on-skip fix
    surfaced by the 1779224125 quest (missing \\usepackage{calc})."""
    monkeypatch.setattr(paper_mod.shutil, "which", lambda c: f"/fake/{c}.exe")
    monkeypatch.setattr(
        PaperGenerator, "_find_pdf_engine",
        lambda self: ("pdflatex", "/fake/pdflatex.exe"),
    )

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        # Simulate pdflatex rc!=0 — pandoc invoked, sanitization
        # completed, paper_pdf_source.md is on disk, but no PDF.
        return SimpleNamespace(
            returncode=43,
            stdout="",
            stderr="! Missing number, treated as zero.\nl.417 \\begin",
        )
    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    out_dir = tmp_path / "out"
    PaperGenerator(cfg).generate(art, out_dir)
    sanitized = out_dir / "paper_pdf_source.md"
    assert sanitized.exists(), (
        "paper_pdf_source.md must be preserved on post-sanitize compile "
        "failure — it's the markdown pandoc just consumed."
    )
    assert (out_dir / "paper_pdf_skipped.md").exists()


def test_generate_cleans_paper_pdf_source_when_pdf_not_in_kinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the user removes ``paper_pdf`` from ``output.kinds``, the
    quest dir should not retain a stale ``paper_pdf_source.md`` from
    a previous run that DID request it. Mirrors the existing cleanup
    for ``paper_pdf_skipped.md``."""
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: None)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "paper_pdf_source.md"
    stale.write_text("from when paper_pdf was on", encoding="utf-8")

    cfg = _make_config(tmp_path, ["paper_md"])  # no paper_pdf
    art = _make_artifacts(tmp_path)
    PaperGenerator(cfg).generate(art, out_dir)
    assert not stale.exists()


# ----------------------------------------------------------------------
# PDF preprocessor — title lift, abstract lift, list extension, heading
# shift, and ref-line dedupe. Tests live here (not just unit tests on
# the helpers in isolation) so the full _compile_pdf code path is
# exercised end-to-end with a fake pandoc and the contract of "what
# pandoc consumes" is pinned against future regressions.
# ----------------------------------------------------------------------


def _capture_pandoc_call(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Wire up a fake `pandoc` + `pdflatex` and return a dict whose
    ``cmd`` / ``input_path`` keys get populated when _compile_pdf
    invokes the pandoc subprocess. The fake writes a stub PDF so the
    generator's "did pandoc emit the output file?" check still passes."""
    state: dict = {}

    def fake_which(name: str) -> str | None:
        if name == "pandoc":
            return "/fake/pandoc"
        if name == "pdflatex":
            return "/fake/pdflatex"
        return None

    monkeypatch.setattr(paper_mod.shutil, "which", fake_which)

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        state["cmd"] = list(cmd)
        state["input_path"] = cmd[1]
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_bytes(b"%PDF\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)
    return state


def test_preprocessor_lifts_h1_title_into_yaml_frontmatter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first `# Title` in paper.md must land in the
    ``paper_pdf_source.md`` as ``---\\ntitle: "..."\\n---`` and must
    NOT remain as an H1 in the body, because pandoc would otherwise
    emit a numbered ``\\section{}`` for it (which is the
    user-reported "1 A Lightweight ..." double-stacked header)."""
    state = _capture_pandoc_call(monkeypatch)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    art.paper_md.write_text(
        "# A Lightweight EUV Simulator\n\n"
        "## Introduction\nbody text\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    PaperGenerator(cfg).generate(art, out_dir)

    sanitized = (out_dir / "paper_pdf_source.md").read_text(encoding="utf-8")
    assert sanitized.startswith("---\n"), "frontmatter must be first"
    assert 'title: "A Lightweight EUV Simulator"' in sanitized
    assert "# A Lightweight EUV Simulator" not in sanitized, (
        "the original H1 must be stripped from the body — otherwise "
        "pandoc emits it as a numbered \\section{}"
    )


def test_preprocessor_lifts_abstract_into_yaml_frontmatter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``## Abstract`` section's body lands in the frontmatter as
    ``abstract: |`` so the template can render it inside a real
    ``\\begin{abstract}`` env, not as a numbered ``\\section{Abstract}``.
    The Abstract heading + body must be removed from the body."""
    state = _capture_pandoc_call(monkeypatch)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    art.paper_md.write_text(
        "# Title\n\n"
        "## Abstract\nFirst sentence. Second sentence.\n\n"
        "## Introduction\nBody.\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    PaperGenerator(cfg).generate(art, out_dir)

    sanitized = (out_dir / "paper_pdf_source.md").read_text(encoding="utf-8")
    assert "abstract: |" in sanitized
    assert "First sentence. Second sentence." in sanitized
    # The original heading is gone from the body — only the YAML copy
    # of the abstract text remains (and that copy is indented by 2
    # spaces as a literal block scalar).
    body_after_frontmatter = sanitized.split("---\n", 2)[-1]
    assert "## Abstract" not in body_after_frontmatter
    assert "## Introduction" in body_after_frontmatter, (
        "non-abstract H2s must be preserved"
    )


def test_preprocessor_enables_list_extension_and_shifts_headings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two pandoc CLI contracts that depend on whether a title was
    lifted: ``--from=markdown+lists_without_preceding_blankline`` is
    ALWAYS appended (the LLM routinely drops lists immediately after
    a paragraph with no blank line), and
    ``--shift-heading-level-by=-1`` is appended ONLY when a title
    was lifted (otherwise the H2 sections are still the user's
    top-level structure and shifting would mangle them)."""
    state = _capture_pandoc_call(monkeypatch)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    art.paper_md.write_text(
        "# Title\n## Introduction\nbody\n",
        encoding="utf-8",
    )
    PaperGenerator(cfg).generate(art, tmp_path / "out")

    cmd = state["cmd"]
    assert "--from=markdown+lists_without_preceding_blankline" in cmd, (
        "list-after-paragraph rendering depends on this extension"
    )
    assert "--shift-heading-level-by=-1" in cmd, (
        "title was lifted, so H2s become \\section — without the shift "
        "they would render as 0.1 / 0.2 subsections of an implicit "
        "section 0"
    )


def test_preprocessor_skips_heading_shift_when_no_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the markdown has NO ``# H1``, we must NOT pass
    ``--shift-heading-level-by=-1`` — that would shift the author's
    intended H2 sections up to H1 (which renders correctly but
    re-numbers everything), and could even shift H1s the author
    intentionally wrote into the metadata title space, where pandoc
    silently drops them.

    The list extension is unaffected: it's safe to enable in either
    case."""
    state = _capture_pandoc_call(monkeypatch)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    # Note: no `# H1` line — body starts straight at H2.
    art.paper_md.write_text(
        "## Section one\nbody\n",
        encoding="utf-8",
    )
    PaperGenerator(cfg).generate(art, tmp_path / "out")

    cmd = state["cmd"]
    assert "--from=markdown+lists_without_preceding_blankline" in cmd
    assert "--shift-heading-level-by=-1" not in cmd, (
        "with no title to lift, shifting headings would mangle the "
        "author's intended hierarchy"
    )


def test_preprocessor_dedupes_duplicated_reference_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The writer LLM emits ``"1. Title. Title."`` when the
    prior-work excerpt starts with the title (a structural artifact
    of the engine's old _format_lit format). The preprocessor
    collapses the duplication so the rendered References section
    reads cleanly."""
    state = _capture_pandoc_call(monkeypatch)

    cfg = _make_config(tmp_path, ["paper_md", "paper_pdf"])
    art = _make_artifacts(tmp_path)
    art.paper_md.write_text(
        "# Title\n\n"
        "## References\n"
        "1. Stratonovich-type integral with respect to a general "
        "stochastic measure. Stratonovich-type integral with respect "
        "to a general stochastic measure.\n",
        encoding="utf-8",
    )
    PaperGenerator(cfg).generate(art, tmp_path / "out")

    sanitized = (tmp_path / "out" / "paper_pdf_source.md").read_text(encoding="utf-8")
    # The dup pattern is gone — the title appears once per ref line.
    assert sanitized.count(
        "Stratonovich-type integral with respect to a general stochastic measure"
    ) == 1, sanitized
