"""Web pages are Further reading; papers are the numbered References.

A web page is kept for the text the writer quotes, and it is listed apart
from the scholarly sources a claim is cited to. The split runs through every
output: the writer's prior-work block labels web pages [W1], [W2]... while
papers keep [1], [2]...; the paper gets a ``## Further reading`` section the
engine appends itself; claim grounding accepts a [W<k>] source; the poster's
Sources band, the slides and the BibTeX / CSL-JSON export carry the two lists
separately. All of them number from one de-duplicated list, so [2] names the
same source in the prompt, the paper, the claim ledger and the bib.
"""
from __future__ import annotations

import asyncio
import json
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
from core.engine import (
    Engine,
    QuestArtifacts,
    _append_further_reading,
    _format_lit_from_state,
    build_further_reading,
    build_references,
    render_further_reading_marp_slide,
    render_poster_references_latex,
    render_references_marp_slide,
)
from core.provider import ResolvedEndpoint
from generation.paper import PaperGenerator
from generation.poster import PosterGenerator
from generation.slides import SlideGenerator

AGENTS = Path(__file__).resolve().parent.parent / "agents"
TOPIC = "How did action figures shape children's play in the 1980s?"


def _web(title: str, url: str) -> dict:
    return {"content": f"{title} page text", "metadata": {
        "source": "web_search", "kind": "web_page", "title": title, "url": url,
        "site": url.split("/")[2]}}


def _paper(title: str, doi: str) -> dict:
    return {"content": f"{title}. An abstract.", "metadata": {
        "source": "crossref", "title": title, "doi": doi,
        "authors": ["A. Author"], "year": 2020, "venue": "J. Toys"}}


LIT = [
    _web("A collector's history of action figures", "https://collectors.example/history"),
    _paper("Action figures and children's play", "10.1/a"),
    _paper("Action Figures and Children's Play", "10.1/a-copy"),  # the same work again
    {"content": "memory", "metadata": {"kind": "fi_paper_spine", "title": "internal"}},
    _web("Museum of Play: the 1980s", "https://museum.example/1980s"),
    _paper("Toys, television and the 1980s", "10.1/b"),
]


def _config(tmp_path: Path, kinds: list[str] | None = None) -> Config:
    return Config(
        topic=TOPIC, title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, claim_grounding=True),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out", kinds=kinds or ["paper_md"]),
    )


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(_config(tmp_path))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    (tmp_path / "paper").mkdir(parents=True, exist_ok=True)
    return eng


def _artifacts(tmp_path: Path) -> QuestArtifacts:
    quest_root = tmp_path / "quest"
    (quest_root / "paper").mkdir(parents=True)
    md = quest_root / "paper" / "paper.md"
    md.write_text("# Play and Plastic\n\nBody.\n", encoding="utf-8")
    art = QuestArtifacts(quest_id="q1", quest_root=quest_root, paper_md=md)
    art.raw_state = {"literature": LIT}
    return art


def _fake_model(monkeypatch: pytest.MonkeyPatch, module: str, reply: str) -> None:
    async def resolve(provider, supervisor):  # noqa: ANN001
        return ResolvedEndpoint(base_url="http://127.0.0.1:1/v1", model="x", api_key="none")

    async def chat(self, messages, **kw):  # noqa: ANN001
        return reply

    monkeypatch.setattr(f"generation.{module}.resolve_endpoint_async", resolve)
    monkeypatch.setattr(f"generation.{module}.LLMClient.chat", chat)


# --- one list, two parts ------------------------------------------------------

def test_papers_are_numbered_and_web_pages_are_further_reading() -> None:
    refs = build_references(LIT, audience="external")
    further = build_further_reading(LIT, audience="external")
    assert [(r["n"], r["doi"]) for r in refs] == [(1, "10.1/a"), (2, "10.1/b")]
    assert [(w["label"], w["url"]) for w in further] == [
        ("W1", "https://collectors.example/history"),
        ("W2", "https://museum.example/1980s"),
    ]


def test_the_writer_sees_the_labels_the_reference_lists_use() -> None:
    block = _format_lit_from_state({"literature": LIT})
    assert "[1] A. Author (2020). Action figures and children's play" in block
    assert "[2] A. Author (2020). Toys, television and the 1980s" in block
    assert "[W1] A collector's history of action figures" in block
    assert "[W2] Museum of Play: the 1980s" in block
    assert "[3]" not in block, "the second copy of work 1 is not a new source"


# --- the paper ----------------------------------------------------------------

def test_the_paper_gets_a_further_reading_section_once() -> None:
    further = build_further_reading(LIT)
    md = "# Title\n\nBody.\n\n## References\n1. A. Author (2020). Action figures.\n"
    out = _append_further_reading(md, further)
    assert out.index("## References") < out.index("## Further reading")
    assert ("- [W1] A collector's history of action figures. "
            "https://collectors.example/history") in out
    assert _append_further_reading(out, further) == out, "never appended twice"
    written = md + "\n## Further Reading\n- a page\n"
    assert _append_further_reading(written, further) == written, "the writer's own heading counts"
    assert _append_further_reading(md, []) == md


def test_the_written_paper_ends_with_further_reading(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    seen: dict = {}

    async def chat(prompt, *, node=""):
        seen["prompt"] = prompt
        return ("# Play and Plastic\n\nBody [1], quoting [W1].\n\n## References\n"
                "1. A. Author (2020). Action figures and children's play. J. Toys. DOI: 10.1/a\n")

    eng._chat = chat  # type: ignore[method-assign]
    out = asyncio.run(eng._node_write({"topic": TOPIC, "literature": LIT}))  # type: ignore[arg-type]
    text = Path(out["paper_md"]).read_text(encoding="utf-8")
    assert text.index("## References") < text.index("## Further reading")
    assert "[W2] Museum of Play: the 1980s" in text
    assert "[W1] A collector's history of action figures" in seen["prompt"]


# --- claim grounding -----------------------------------------------------------

def test_a_claim_can_rest_on_a_further_reading_source(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    paper = tmp_path / "paper" / "paper.md"
    paper.write_text("# P\n\nClaims.\n", encoding="utf-8")
    seen: dict = {}

    async def chat(prompt, *, node=""):
        seen["prompt"] = prompt
        return json.dumps({"claims": [
            {"claim": "Figures sold in millions", "basis": "citation",
             "citation_index": "W1", "evidence": "the collector page"},
            {"claim": "Television drove toy lines", "basis": "citation",
             "citation_index": 2, "evidence": "the paper"},
            {"claim": "Museums kept the boxes", "basis": "citation",
             "citation_index": "[w2]", "evidence": ""},
            {"claim": "A page that does not exist", "basis": "citation",
             "citation_index": "W9", "evidence": ""},
        ], "summary": ""})

    eng._chat = chat  # type: ignore[method-assign]
    state = {"topic": TOPIC, "paper_md": str(paper), "literature": LIT}
    g = asyncio.run(eng._node_claim_check(state))["claim_grounding"]  # type: ignore[arg-type]
    assert [c["citation_index"] for c in g["claims"]] == ["W1", 2, "W2", None]
    assert g["unsupported"] == ["A page that does not exist"]
    assert "[W1] A collector's history of action figures" in seen["prompt"]
    assert "[2] Toys, television and the 1980s" in seen["prompt"]
    assert "cite [W1]" in (tmp_path / "paper" / "CLAIMS.md").read_text(encoding="utf-8")


# --- poster, slides, bib --------------------------------------------------------

def test_the_poster_band_lists_further_reading_after_the_sources() -> None:
    band = render_poster_references_latex(build_references(LIT), build_further_reading(LIT))
    assert band.index("\\textbf{Sources:}") < band.index("\\textbf{Further reading:}")
    assert "[1]~" in band and "[W1]~" in band
    web_only = render_poster_references_latex([], build_further_reading(LIT))
    assert "Further reading:" in web_only and "Sources:" not in web_only


@pytest.mark.asyncio
async def test_the_poster_carries_further_reading(tmp_path: Path, monkeypatch) -> None:
    art = _artifacts(tmp_path)
    _fake_model(monkeypatch, "poster", json.dumps(
        {"title": "x", "left": r"\textbf{Background.} Toys.", "right": r"\textbf{Findings.} Play."}))
    monkeypatch.setattr("generation.poster.shutil.which", lambda _name: None)
    result = await PosterGenerator(_config(tmp_path, ["poster"])).generate(art, art.quest_root)
    tex = result["poster_tex"].read_text(encoding="utf-8")
    assert "Further reading:" in tex and "[W1]~" in tex
    assert tex.index("Sources:") < tex.index("Further reading:")


def test_a_web_only_band_is_still_branded_at_its_first_line() -> None:
    """The poster's corner mark goes at the start of the band's first line.
    With only web pages that line is Further reading, not Sources."""
    from generation.poster import _POSTER_ICON_TEX, _brand_references_band

    band = _brand_references_band(
        render_poster_references_latex([], build_further_reading(LIT)), True)
    assert band.index("\\hrule") < band.index(_POSTER_ICON_TEX) < band.index("Further reading:")


def test_slides_get_a_separate_further_reading_slide() -> None:
    slide = render_further_reading_marp_slide(build_further_reading(LIT))
    assert slide.startswith("---") and "## Further reading" in slide
    assert "- [W2] Museum of Play: the 1980s. https://museum.example/1980s" in slide
    assert render_further_reading_marp_slide([]) == ""
    assert "museum.example" not in render_references_marp_slide(build_references(LIT))


@pytest.mark.parametrize("deck,further_slides", [
    ("---\nmarp: true\n---\n\n# Play\n", 1),
    ("---\nmarp: true\n---\n\n# Play\n\n---\n\n## Further reading\n\n- a page\n", 1),
])
@pytest.mark.asyncio
async def test_the_deck_ends_with_references_then_further_reading(
    tmp_path: Path, monkeypatch, deck: str, further_slides: int,
) -> None:
    art = _artifacts(tmp_path)
    _fake_model(monkeypatch, "slides", deck)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    slides_md = await SlideGenerator(_config(tmp_path, ["slides"]))._author_marp(
        art, out_dir, supervisor=None)
    text = slides_md.read_text(encoding="utf-8")
    assert text.count("## Further reading") == further_slides, "never a second copy"
    assert "## References" in text
    assert text.index("## References") < text.rindex("## Further reading") or "a page" in text


def test_the_bib_export_splits_papers_and_web_pages(tmp_path: Path) -> None:
    cfg = Config.model_validate({"topic": "t", "title": "tt", "output": {
        "kinds": ["paper_md"], "output_dir": str(tmp_path / "outputs")}})
    art = _artifacts(tmp_path)
    out = tmp_path / "out"
    result = PaperGenerator(cfg).generate(art, out)
    refs_bib = (out / "paper" / "references.bib").read_text(encoding="utf-8")
    further_bib = (out / "paper" / "further_reading.bib").read_text(encoding="utf-8")
    assert "10.1/a" in refs_bib and "collectors.example" not in refs_bib
    assert "@misc{" in further_bib and "https://museum.example/1980s" in further_bib
    assert "10.1/a" not in further_bib
    assert result["further_reading_bib"] == out / "paper" / "further_reading.bib"
    items = json.loads((out / "paper" / "further_reading.csl.json").read_text(encoding="utf-8"))
    assert [i["type"] for i in items] == ["webpage", "webpage"]
    assert result["further_reading_csl_json"] == out / "paper" / "further_reading.csl.json"


# --- the depth rule -------------------------------------------------------------

def test_the_prompts_count_further_reading_toward_depth() -> None:
    review = (AGENTS / "review.md").read_text(encoding="utf-8")
    write = (AGENTS / "write.md").read_text(encoding="utf-8")
    assert "References or Further reading" in review
    assert "References or Further reading" in write
    assert "[W" in write and "## Further reading" in write
