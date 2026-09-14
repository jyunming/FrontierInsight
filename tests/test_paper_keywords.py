"""The keywords a writer gives a paper: read back out of paper.md, printed
under the abstract in the PDF, styled in the HTML render, and carried into
the knowledge layer's index card (``generation/_keywords.py``)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import Config, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from core.knowledge import Knowledge, _render_paper_spine
from generation import _html_pdf
from generation._html_pdf import render_paper_html_pdf
from generation._keywords import extract_keywords, keywords_block, paper_keywords, split_keywords

REPO = Path(__file__).resolve().parent.parent
PAPER = (
    "# Symplectic Integrators on a Damped Oscillator\n\n"
    "## Abstract\n"
    "We compare three integrators. RK4 keeps the energy error below 1e-8.\n"
    "**Keywords:** symplectic integrator, Runge-Kutta, energy drift, damped oscillator\n\n"
    "## Introduction\n"
    "Keywords: this line is body text and stays.\n"
)
REPORT = (
    "# Integrator Choice for Long Simulations\n"
    "<!-- Keywords: symplectic integrator; energy drift; simulation cost -->\n\n"
    "## Executive Summary\nUse RK4.\n"
)
VENUES = ("generic", "neurips", "iclr", "ieee_access", "nature_mi")
PERSONAS = ("report", "policy_brief", "essay", "whitepaper")


@pytest.mark.parametrize("line", [
    "**Keywords:** a, b, c",
    "**Keywords**: a, b, c",
    "*Keywords:* a; b; c.",
    "Keywords: a, b, c",
    "KEYWORDS: a, B, b, c",
])
def test_the_keywords_line_is_read_in_its_usual_forms(line: str) -> None:
    words, rest = extract_keywords(f"# T\n\n## Abstract\nText.\n{line}\n\n## Introduction\nBody.\n")
    assert [w.lower() for w in words] == ["a", "b", "c"]
    assert rest == "# T\n\n## Abstract\nText.\n\n## Introduction\nBody.\n"


def test_a_report_keeps_its_keywords_in_a_comment() -> None:
    words, rest = extract_keywords(REPORT)
    assert words == ["symplectic integrator", "energy drift", "simulation cost"]
    assert rest == "# Integrator Choice for Long Simulations\n\n## Executive Summary\nUse RK4.\n"


def test_only_the_title_and_abstract_are_read() -> None:
    words, rest = extract_keywords(PAPER)
    assert words == ["symplectic integrator", "Runge-Kutta", "energy drift", "damped oscillator"]
    assert "Keywords: this line is body text and stays." in rest
    assert paper_keywords("# T\n\n## Introduction\nKeywords: not these.\n") == []


def test_keywords_lose_emphasis_quotes_full_stops_and_repeats() -> None:
    assert split_keywords(" `a`, *b*; 'c'. ,, A ") == ["a", "b", "c"]


def _html_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, markdown: str) -> tuple[str, list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd, **_kw):  # noqa: ANN001
        calls.append(list(cmd))
        raise OSError("stop at pandoc")

    monkeypatch.setattr(_html_pdf.subprocess, "run", fake_run)
    pmd = tmp_path / "paper.md"
    pmd.write_text(markdown, encoding="utf-8")
    render_paper_html_pdf(pmd, tmp_path / "paper.pdf", pandoc_path="pandoc", browser=("msedge", "edge"))
    return (tmp_path / "paper_html_body.md").read_text(encoding="utf-8"), calls[0]


def test_the_html_render_styles_the_keywords_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body, cmd = _html_render(tmp_path, monkeypatch, PAPER)
    assert (
        "RK4 keeps the energy error below 1e-8.\n\n::: keywords\n"
        "**Keywords:** symplectic integrator, Runge-Kutta, energy drift, damped oscillator\n:::\n\n## Introduction"
    ) in body
    assert "keywords=symplectic integrator, Runge-Kutta, energy drift, damped oscillator" in cmd
    # A report's comment stays hidden, and its keywords are not shown anywhere.
    body, cmd = _html_render(tmp_path, monkeypatch, REPORT)
    assert "<!-- Keywords:" in body and "::: keywords" not in body
    assert not any(arg.startswith("keywords=") for arg in cmd)
    assert keywords_block(REPORT) == ([], REPORT)


def test_the_themes_style_the_abstract_by_its_id_and_the_keywords_block() -> None:
    html = REPO / "templates" / "paper" / "_html"
    latexlike = (html / "latexlike.css").read_text(encoding="utf-8")
    # "The first section" centred and unnumbered a paper's Introduction when it had no abstract.
    assert "first-of-type" not in latexlike
    assert "h2#abstract {" in latexlike and "h2#abstract::before { content: none; }" in latexlike
    for theme in ("latexlike.css", "briefing.css"):
        assert ".keywords p {" in (html / theme).read_text(encoding="utf-8"), theme


def test_the_venue_templates_print_the_keywords_under_the_abstract_and_in_the_pdf() -> None:
    for name in VENUES:
        tex = (REPO / "templates" / "paper" / name / "template.tex").read_text(encoding="utf-8")
        assert "pdfkeywords={$for(keywords)$$keywords$$sep$, $endfor$}" in tex, name
        line = r"\textbf{Keywords:} $for(keywords)$$keywords$$sep$, $endfor$"
        assert line in tex, name
        assert tex.index(line) > tex.rindex(r"\end{abstract}"), name
    # The formats without an abstract show none.
    for name in PERSONAS:
        tex = (REPO / "templates" / "paper" / name / "template.tex").read_text(encoding="utf-8")
        assert "keywords" not in tex.lower(), name


def test_the_writer_is_asked_for_an_abstract_and_keywords() -> None:
    text = (REPO / "agents" / "write.md").read_text(encoding="utf-8")
    assert "open with `## Abstract`" in text
    assert "`**Keywords:** <keyword>, <keyword>, <keyword>, <keyword>`" in text
    assert "`<!-- Keywords: <keyword>, <keyword>, <keyword>, <keyword> -->`" in text


def test_the_index_card_carries_the_keywords(tmp_path: Path) -> None:
    text = _render_paper_spine({"title": "T", "topic": "t"}, keywords=["symplectic integrator", "energy drift"])
    assert "KEYWORDS: symplectic integrator, energy drift" in text

    class Brain:
        def __init__(self) -> None:
            self.docs: list[dict] = []

        def ingest(self, documents: list[dict]) -> None:
            self.docs.extend(documents)

        def finalize_ingest(self) -> None:
            pass

    brain = Brain()
    k = Knowledge(KnowledgeConfig(enabled=False))
    k.enabled, k.cfg, k._brain = True, KnowledgeConfig(enabled=True, write_back_quests=True), brain
    paper = tmp_path / "paper.md"
    paper.write_text(PAPER, encoding="utf-8")
    assert k.add_quest_artifacts(
        quest_id="q1", paper_md_path=paper, summary="s", metadata={"title": "T", "keywords": ["energy drift"]},
    )
    (spine,) = [d for d in brain.docs if d["metadata"]["kind"] == "fi_paper_spine"]
    assert "KEYWORDS: energy drift" in spine["text"]


def test_the_write_back_reads_the_keywords_from_the_paper(tmp_path: Path) -> None:
    cfg = Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        knowledge=KnowledgeConfig(enabled=False, write_back_quests=True),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    engine = Engine(cfg)
    captured: dict = {}
    engine.knowledge.enabled = True
    engine.knowledge.add_quest_artifacts = lambda **kw: captured.update(kw) or True
    root = tmp_path / "quest"
    (root / "paper").mkdir(parents=True)
    paper = root / "paper" / "paper.md"
    paper.write_text(REPORT, encoding="utf-8")
    engine._write_back_knowledge(
        SimpleNamespace(paper_md=paper, quest_root=root),
        {"title": "t", "topic": "t", "review": {"verdict": "accept"}},
    )
    assert captured["metadata"]["keywords"] == ["symplectic integrator", "energy drift", "simulation cost"]
