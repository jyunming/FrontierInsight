"""Web pages are Further reading; papers are the numbered References.

A web page is kept for the text the writer quotes, and it is listed apart
from the scholarly sources a claim is cited to. The split runs through every
output: the writer's prior-work block labels web pages [W1], [W2]... while
papers keep [1], [2]...; the paper gets a ``## Further reading`` section the
engine appends itself; claim grounding accepts a [W<k>] source; the poster's
reference band, the slides and the BibTeX / CSL-JSON export carry the two lists
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
    md.write_text("# Play and Plastic\n\nBody [1], and more [2].\n", encoding="utf-8")
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


def test_a_bot_check_page_is_never_further_reading() -> None:
    """A real SIR paper listed "[W7] Checking your browser - reCAPTCHA": a web hit
    saved under a challenge page's title. It gets no label anywhere — not in the
    writer's block, not in the Further reading the engine writes."""
    from core.engine import _finalize_paper_sources
    wall = _web("Checking your browser - reCAPTCHA",
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC6002090/")
    lit = [LIT[0], wall, *LIT[1:]]
    assert [w["url"] for w in build_further_reading(lit)] == [
        "https://collectors.example/history", "https://museum.example/1980s",
    ]
    assert "reCAPTCHA" not in _format_lit_from_state({"literature": lit})
    body, _ordered, _dropped = _finalize_paper_sources("# T\n\nBody [1].\n", lit, "external")
    assert "## Further reading" in body and "[W2] Museum of Play: the 1980s" in body
    assert "reCAPTCHA" not in body and "[W3]" not in body


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
             "citation_index": "W1", "evidence": "the collector page",
             "quote": "A collector's history of action figures"},
            {"claim": "Television drove toy lines", "basis": "citation",
             "citation_index": 2, "evidence": "the paper", "quote": "Toys, television and the 1980s"},
            {"claim": "Museums kept the boxes", "basis": "citation",
             "citation_index": "[w2]", "evidence": "", "quote": "Museum of Play: the 1980s page text"},
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

def test_the_poster_band_lists_cited_papers_before_cited_web_pages() -> None:
    from generation.poster import _Block, _band_entry, _band_latex, _cited_labels

    refs, further = build_references(LIT), build_further_reading(LIT)
    sources = {str(r["n"]): r for r in refs} | {w["label"]: w for w in further}
    cited = _cited_labels([_Block("text", text="Play changed [W2], as shown [2] and [W1].")], set(sources))
    assert cited == ["2", "W1", "W2"]
    band = _band_latex([_band_entry(sources[label], label) for label in cited], "References", 2, True)
    assert band.index("[2] A. Author (2020). Toys, television and the 1980s. J. Toys. doi:10.1/b.") < (
        band.index("[W1] A collector's history of action figures. collectors.example.")
    )
    assert "museum.example" in band and "https://" not in band


@pytest.mark.asyncio
async def test_the_poster_cites_papers_and_web_pages_from_one_list(tmp_path: Path, monkeypatch) -> None:
    art = _artifacts(tmp_path)
    _fake_model(monkeypatch, "poster", json.dumps({"headline": "Action figures turned play into stories", "blocks": [
        {"type": "heading", "text": "Background"},
        {"type": "text", "text": "Collectors tell the story [W1]."},
        {"type": "heading", "text": "What it means"},
        {"type": "text", "text": "Television sold the toys [2]."},
    ]}))
    monkeypatch.setattr("generation.poster.find_pdf_engine", lambda: None)
    result = await PosterGenerator(_config(tmp_path, ["poster"])).generate(art, art.quest_root)
    band = result["poster_tex"].read_text(encoding="utf-8").split(r"{\bfseries References}", 1)[1]
    assert band.index("[2] A. Author (2020). Toys, television and the 1980s.") < (
        band.index("[W1] A collector's history of action figures. collectors.example.")
    )
    assert "Museum of Play" not in band  # listed only when cited


def test_the_band_carries_the_brand_mark_on_its_heading_line() -> None:
    from generation.poster import _band_entry, _band_latex

    further = build_further_reading(LIT)
    band = _band_latex([_band_entry(w, w["label"]) for w in further], "References", 2, True)
    assert band.index("References") < band.index("fi_icon.png") < band.index("[W1]")
    assert "fi_icon.png" in _band_latex([], "References", 2, True)


def test_the_source_slide_lists_papers_but_not_web_pages() -> None:
    slide = render_references_marp_slide(build_references(LIT))
    assert slide.startswith("---") and "## References" in slide
    assert "museum.example" not in slide
    assert render_references_marp_slide([]) == ""


def _long_refs(count: int) -> list[dict]:
    return [
        {
            "n": k, "title": f"A long study title number {k} about damped oscillators and symplectic integration",
            "authors": ["A. Author", "B. Author", "C. Author"], "year": 2000 + k,
            "venue": "Journal of Computational Physics", "doi": f"10.1016/j.jcp.2014.{k:02d}.008",
        }
        for k in range(1, count + 1)
    ]


def _entries(deck: str) -> list[int]:
    return [int(line[3:line.index("]")]) for line in deck.splitlines() if line.startswith("- [")]


def test_the_source_slide_lists_the_most_cited_references_on_one_slide() -> None:
    """The validation quest's nine-slide talk ended on three References and
    three Further reading slides, and no slide cited any of them."""
    from core import engine

    paper = (
        "Intro [4]. Method [2, 4]. Results [4], [7] and [9-11], with a page [W1].\n\n"
        "## References\n\n1. Not counted [3] [3] [3].\n"
    )
    deck = render_references_marp_slide(_long_refs(12), paper_md=paper)
    assert deck.count("---") == 1 and deck.count("## References") == 1 and "(continued)" not in deck
    numbers = _entries(deck)
    # Chosen by how often the paper cites them, listed in reference order.
    assert numbers[:2] == [2, 4] and numbers == sorted(numbers) and set(numbers) <= {2, 4, 7, 9, 10, 11}
    entries = [line for line in deck.splitlines() if line.startswith("- [")]
    assert sum(-(-len(e) // engine._SOURCE_SLIDE_CHARS_PER_LINE) + 0.5 for e in entries) <= engine._SOURCE_SLIDE_LINES
    assert deck.rstrip().endswith(f"_({12 - len(numbers)} more sources in the paper)_")


def test_the_source_slide_budget_is_the_size_of_the_list_it_is_set_at() -> None:
    """At 0.96em in two columns the list holds about 35 characters a line and 27 lines: six
    references of about 110 characters cost 4.5 lines each and all fit, while a stored quest's
    six (150 to 270 characters, see test_slide_figure_fit) fit four."""
    from core import engine

    assert (engine._SOURCE_SLIDE_CHARS_PER_LINE, engine._SOURCE_SLIDE_LINES, engine._SOURCE_SLIDE_MAX) == (35, 27, 6)
    shorter = [
        {"n": k, "title": f"A study of extinction thresholds number {k}", "authors": ["A. Author", "B. Author"],
         "year": 2000 + k, "venue": "Journal of Dynamics", "doi": f"10.1000/j.{k}"}
        for k in range(1, 7)
    ]
    assert _entries(render_references_marp_slide(shorter, paper_md="Cites [1] [2] [3] [4] [5] [6].")) == [1, 2, 3, 4, 5, 6]
    longer = [
        {"n": k, "title": "A long study title about damped oscillators and symplectic integration " * 2,
         "authors": ["A. Author", "B. Author"], "year": 2000 + k, "venue": "Journal of Computational Physics"}
        for k in range(1, 7)
    ]
    # About 210 characters each: 7.5 lines apiece, three of them.
    assert len(_entries(render_references_marp_slide(longer))) == 3


def test_a_short_source_list_is_listed_whole() -> None:
    short = [{"n": k, "title": f"Study {k}", "authors": ["A. Author"], "year": 2000 + k} for k in range(1, 9)]
    assert _entries(render_references_marp_slide(short[:3])) == [1, 2, 3]
    assert "in the paper" not in render_references_marp_slide(short[:3])
    # Nothing cited: the first ones, up to the cap.
    uncited = render_references_marp_slide(short)
    assert _entries(uncited) == [1, 2, 3, 4, 5, 6] and uncited.rstrip().endswith("_(2 more sources in the paper)_")


def test_the_deck_theme_styles_the_source_slide() -> None:
    css = (Path(__file__).resolve().parent.parent / "templates" / "slides" / "fi.css").read_text(encoding="utf-8")
    assert 'section:has(h2[id^="references"]) ul' in css
    assert 'section:has(h2[id^="further-reading"]) ul' in css
    # The "(N more sources in the paper)" line under the list.
    assert 'section:has(h2[id^="references"]) :is(ul, ol) + p' in css
    # 0.96em is 18 pt on the slide; the list, and the line under it, are set at that size and
    # the engine's budget (_SOURCE_SLIDE_*) counts lines at it.
    section = css[css.index("References slide"):]
    assert section.count("font-size: 0.96em;") == 2 and "0.6em" not in section


@pytest.mark.parametrize("deck,further_slides", [
    ("---\nmarp: true\n---\n\n# Play\n", 0),
    ("---\nmarp: true\n---\n\n# Play\n\n---\n\n## Further reading\n\n- a page\n", 1),
])
@pytest.mark.asyncio
async def test_the_deck_ends_on_one_references_slide(
    tmp_path: Path, monkeypatch, deck: str, further_slides: int,
) -> None:
    art = _artifacts(tmp_path)
    # Cites the second of the two papers only.
    art.paper_md.write_text("# Play and Plastic\n\nBody [2].\n", encoding="utf-8")
    _fake_model(monkeypatch, "slides", deck)
    import generation.slides as slides_mod

    seen: dict = {}
    render = slides_mod.render_references_marp_slide

    def recording(refs, **kw):  # noqa: ANN001
        seen.update(kw, numbers=[r["n"] for r in refs])
        return render(refs, **kw)

    monkeypatch.setattr(slides_mod, "render_references_marp_slide", recording)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    slides_md = await SlideGenerator(_config(tmp_path, ["slides"]))._author_marp(
        art, out_dir, supervisor=None)
    text = slides_md.read_text(encoding="utf-8")
    # The slide is chosen by what the whole paper cites, not the 8000 characters the deck author saw.
    assert seen["paper_md"] == art.paper_md.read_text(encoding="utf-8")
    # Like the paper's References, the slide offers only the papers the text cites.
    assert seen["numbers"] == [2]
    assert text.count("## References") == 1
    # Web pages stay in the paper; a Further reading slide the deck wrote itself stays too.
    assert text.count("## Further reading") == further_slides


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
