"""How paper.pdf was made reaches run.log in one plain line, and QUALITY.md is written again
after the paper exists, so its PDF section is there on the first run and current on a rerun."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from generation.paper import render_line


def test_the_line_for_a_typeset_paper() -> None:
    assert render_line({"engine": "xelatex", "typeset": True, "pages": 4}) == "paper.pdf: typeset with XeLaTeX, 4 pages"
    assert render_line({"engine": "pdflatex", "typeset": True, "pages": 1}) == "paper.pdf: typeset with pdfLaTeX, 1 page"


def test_the_line_for_a_paper_the_browser_made() -> None:
    line = render_line({"engine": "html_fallback", "typeset": False, "pages": 5, "reason": "no LaTeX engine found"})
    assert line == "paper.pdf: NOT typeset (no LaTeX engine found); made by the browser, 5 pages"
    assert render_line({"engine": "html_briefing", "typeset": True, "pages": 3}) == (
        "paper.pdf: made by the browser in the briefing style, as configured, 3 pages"
    )


def test_the_line_for_no_paper() -> None:
    assert render_line({"engine": "none", "typeset": False, "pages": None, "reason": "xelatex timeout"}) == (
        "paper.pdf: not made (xelatex timeout)"
    )
    assert render_line(None) == "paper.pdf: not made (no record of a render)"


@pytest.mark.asyncio
async def test_finishing_the_outputs_logs_the_paper_line_and_writes_quality_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import launch
    from core.config import Config
    from core.engine import QuestArtifacts
    from core.provider import ProxySupervisor

    launch._import_runtime()
    root = tmp_path / "q"
    (root / ".fi").mkdir(parents=True)
    (root / "paper").mkdir()
    (root / "paper" / "paper.md").write_text("# A paper\n", encoding="utf-8")
    # QUALITY.md as the engine left it at the end of its run, before the paper existed.
    (root / "QUALITY.md").write_text("# old report\n", encoding="utf-8")

    async def fake_generators(cfg, art, **_kw):  # noqa: ANN001, ANN003
        (root / "paper.pdf").write_bytes(b"%PDF-1.4\n")
        (root / ".fi" / "paper_render.json").write_text(json.dumps({
            "engine": "html_fallback", "typeset": False, "rc": 43, "pages": 5, "format": "generic",
            "reason": "pdflatex errored (rc=43)", "glyphs_replaced": [], "glyphs_not_set": [],
        }), encoding="utf-8")
        return {"paper_pdf": root / "paper.pdf"}

    monkeypatch.setattr(launch, "_run_generators", fake_generators)
    cfg = Config.model_validate({"topic": "t", "title": "t", "output": {
        "output_dir": str(tmp_path), "kinds": ["paper_md", "paper_pdf"], "visual_check": False,
    }})
    art = QuestArtifacts(quest_id="render-line-q", quest_root=root, paper_md=root / "paper" / "paper.md")
    await launch._finish_outputs(cfg, art, supervisor=ProxySupervisor())

    log = (root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "paper.pdf: NOT typeset (pdflatex errored (rc=43)); made by the browser, 5 pages" in log
    quality = (root / "QUALITY.md").read_text(encoding="utf-8")
    assert "## The PDF" in quality and "not typeset" in quality
