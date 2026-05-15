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
from generation.paper import PaperGenerator


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

    Repro: outputs/1778751621-belgium-culture-vs-taiwan-cultur-32f2ff/
    on 2026-05-14 — host had no pandoc / pdflatex / tectonic, and the
    skip was completely invisible from the quest output."""
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
    simultaneously report a paper.pdf AND a 'skipped' marker.

    Regression for PR #55 bot comment."""
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
# only paper_pdf_skipped.md. Bot comment on PR #58.


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


# ---- Template-substitution-leak regression (Phase Q hotfix) -------------
#
# Two bugs hit the user post-#79: (a) paper.pdf compile failed with
# "Environment Shaded undefined" + "\\tightlist undefined" because
# pandoc emits those commands but our custom templates didn't define
# them; (b) poster.tex compile failed because the template's
# top-of-file comment ``% Substitutions (Python string.Template):
# $title, $left, ...`` was itself substituted by ``safe_substitute``,
# and the multi-line ``$left`` content injected into the comment
# broke the comment boundary, pushing later lines (including
# ``\documentclass``) past raw LaTeX that pdflatex couldn't parse.


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


def test_paper_stub_templates_were_deleted() -> None:
    """The iclr/ieee_access/nature_mi templates shipped as 1-line
    LaTeX-comment stubs that pandoc accepted as templates and
    pdflatex rejected at compile. Audit #05 flagged this; the
    Phase-Q hotfix deletes them so ``_find_pdf_engine`` /
    ``template.exists()`` returns False and pandoc's built-in
    default template runs — which actually compiles."""
    repo = Path(__file__).resolve().parent.parent
    for fmt in ("iclr", "ieee_access", "nature_mi"):
        stub = repo / "templates" / "paper" / fmt / "template.tex"
        assert not stub.exists(), (
            f"templates/paper/{fmt}/template.tex must NOT exist — the "
            f"file is a 1-line stub that breaks pdflatex compile. "
            f"Pandoc-default fallback works correctly when no "
            f"template.tex is present."
        )


def test_poster_template_substitution_does_not_leak_into_preamble() -> None:
    """Phase-Q hotfix regression guard. The poster template's top
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
