"""Tests for generation/paper.py.

These tests must NOT require pandoc/pdflatex. Every external invocation is
mocked via monkeypatch.
"""

from __future__ import annotations

import subprocess
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
    monkeypatch.setattr(paper_mod.shutil, "which", lambda _c: "/fake/pandoc")

    out_dir = tmp_path / "out"

    def fake_run(cmd, **_kwargs):  # type: ignore[no-untyped-def]
        # Simulate pandoc creating the output PDF.
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
