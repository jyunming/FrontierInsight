"""Web-research feature: relevance guard, web-derived plots, and the
references that flow into poster + slides.

The general-web search adapters (Brave / DuckDuckGo), HTML extraction, and
the always-parallel ``asearch`` merge live in ``test_knowledge.py``. This
file covers the engine-side wiring: the no-simulation collector's relevance
guard, the ``web_plots`` node, and the citation rendering shared by the
poster/slides generators.
"""

from __future__ import annotations

import asyncio
import json
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
from core.engine import (
    Engine,
    QuestArtifacts,
    build_references,
    render_poster_references_latex,
    render_references_marp_slide,
)
from core.knowledge import RetrievedDoc


def _engine(tmp_path: Path, **engine_kw) -> Engine:
    cfg = Config(
        topic="t",
        title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(no_simulation=True, clarify_mode="off", **engine_kw),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    eng = Engine(cfg)
    qr = tmp_path / "quest"
    qr.mkdir(exist_ok=True)
    eng.quest_root = qr
    return eng


# ---------------------------------------------------------------------------
# Reference rendering (shared by paper / poster / slides)
# ---------------------------------------------------------------------------


def test_build_references_includes_web_drops_internal() -> None:
    lit = [
        {"content": "p", "metadata": {
            "source": "web_search", "kind": "web_page",
            "title": "SpaceX 2023 revenue", "url": "https://payloadspace.com/x",
            "site": "payloadspace.com"}},
        {"content": "p", "metadata": {
            "title": "Reusable economics", "authors": ["A. Smith"],
            "year": 2022, "venue": "Acta", "doi": "10.1/x"}},
        {"content": "p", "metadata": {"kind": "fi_paper_spine", "title": "memory"}},
        {"content": "p", "metadata": {"url": "https://no-title"}},  # not citable
    ]
    refs = build_references(lit, audience="external")
    assert len(refs) == 2
    assert refs[0]["url"] == "https://payloadspace.com/x"
    assert refs[1]["doi"] == "10.1/x"


def test_poster_references_latex_escapes_urls() -> None:
    refs = build_references([
        {"content": "", "metadata": {
            "source": "web_search", "title": "Q1 & Q2 report",
            "url": "https://e.com/a?x=1&y=2"}},
    ])
    band = render_poster_references_latex(refs)
    assert "\\textbf{Sources:}" in band
    assert "\\&" in band          # & escaped for LaTeX
    assert "Q1 \\& Q2 report" in band


def test_marp_references_slide_built() -> None:
    refs = build_references([
        {"content": "", "metadata": {
            "source": "web_search", "title": "T", "url": "https://e.com/a"}},
    ])
    slide = render_references_marp_slide(refs)
    assert slide.startswith("---")
    assert "## References" in slide
    assert "https://e.com/a" in slide


# ---------------------------------------------------------------------------
# Relevance guard
# ---------------------------------------------------------------------------


def test_relevance_guard_keeps_only_ontopic(tmp_path: Path) -> None:
    eng = _engine(tmp_path)  # relevance_guard defaults True
    docs = [
        RetrievedDoc("SpaceX booked $8B", {"title": "SpaceX revenue 2023"}),
        RetrievedDoc("B-meson CP violation", {"title": "LHCb B-meson decays"}),
    ]

    async def fake_chat(prompt, *, node=None):
        assert node == "relevance_guard"
        return '{"relevant_indices": [0]}'
    eng._chat = fake_chat

    kept = asyncio.run(eng._filter_relevant_docs("SpaceX revenue", docs))
    assert [d.metadata["title"] for d in kept] == ["SpaceX revenue 2023"]


def test_relevance_guard_fails_open_on_bad_response(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    docs = [RetrievedDoc("x", {"title": "a"}), RetrievedDoc("y", {"title": "b"})]

    async def fake_chat(prompt, *, node=None):
        return "I am not JSON"
    eng._chat = fake_chat

    kept = asyncio.run(eng._filter_relevant_docs("topic", docs))
    assert kept == docs  # fail-open: never drop the whole corpus on a hiccup


def test_relevance_guard_disabled_is_passthrough(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.config.knowledge.relevance_guard = False
    docs = [RetrievedDoc("x", {"title": "a"})]

    async def must_not_call(prompt, *, node=None):
        raise AssertionError("guard disabled — no LLM call expected")
    eng._chat = must_not_call

    kept = asyncio.run(eng._filter_relevant_docs("topic", docs))
    assert kept == docs


# ---------------------------------------------------------------------------
# web_plots node
# ---------------------------------------------------------------------------


class _FakeExecutor:
    """Stands in for the real sandbox: install is a no-op, execute
    'runs' the plot script by dropping a PNG into figures/."""

    def __init__(self, quest_root: Path, *, make_fig: bool = True) -> None:
        self.quest_root = quest_root
        self.make_fig = make_fig
        self.executed = False

    async def install(self, packages, *, quest_root):
        return SimpleNamespace(returncode=0, stdout="", stderr="",
                               timed_out=False, duration_s=0.1)

    def python_path(self, quest_root):
        return Path("python")

    async def execute(self, argv, *, cwd, timeout_s):
        self.executed = True
        if self.make_fig:
            figs = self.quest_root / "figures"
            figs.mkdir(parents=True, exist_ok=True)
            (figs / "chart.png").write_bytes(b"\x89PNG\r\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="",
                               timed_out=False, duration_s=0.2)


_NOSIM_STATE = {
    "no_simulation_resolved": True,
    "topic": "spacex revenue",
    "result_json": {},
    "literature": [{
        "content": "Revenue: 2020: 2.0B, 2021: 4.0B, 2022: 8.0B",
        "metadata": {"source": "web_search", "kind": "web_page",
                     "title": "SpaceX revenue", "url": "https://e.com/x"},
    }],
}


def test_web_plots_generates_figures(tmp_path: Path) -> None:
    eng = _engine(tmp_path, web_derived_plots=True)
    eng.executor = _FakeExecutor(eng.quest_root)

    async def fake_chat(prompt, *, node=None):
        assert "spacex" in prompt.lower()
        return (
            "import matplotlib\nimport matplotlib.pyplot as plt\n"
            "plt.plot([2,4,8]); plt.savefig('figures/chart.png')\n"
        )
    eng._chat = fake_chat

    out = asyncio.run(eng._node_web_plots(dict(_NOSIM_STATE)))
    assert eng.executor.executed is True
    assert "chart.png" in (out.get("figures") or [])
    assert (eng.quest_root / "code" / "web_plots.py").is_file()


def test_web_plots_no_plot_sentinel_skips(tmp_path: Path) -> None:
    eng = _engine(tmp_path, web_derived_plots=True)
    eng.executor = _FakeExecutor(eng.quest_root)

    async def fake_chat(prompt, *, node=None):
        return "NO_PLOT"
    eng._chat = fake_chat

    out = asyncio.run(eng._node_web_plots(dict(_NOSIM_STATE)))
    assert out == {}
    assert eng.executor.executed is False  # never ran the sandbox


def test_web_plots_times_out_gracefully(tmp_path: Path) -> None:
    """web_plots must never hang the quest: if the render (LLM + run)
    exceeds the budget, the node skips plots and returns cleanly."""
    eng = _engine(tmp_path, web_derived_plots=True)
    eng.config.engine.web_plots_timeout_s = 0.05

    async def slow_render(state, sources_text):
        await asyncio.sleep(1.0)  # far past the 0.05s budget
        return {"figures": ["never.png"]}
    eng._web_plots_render = slow_render

    out = asyncio.run(eng._node_web_plots(dict(_NOSIM_STATE)))
    assert out == {}  # timed out → skipped, no hang


def test_web_plots_passthrough_when_disabled(tmp_path: Path) -> None:
    eng = _engine(tmp_path, web_derived_plots=False)

    async def must_not_call(prompt, *, node=None):
        raise AssertionError("disabled — no LLM call expected")
    eng._chat = must_not_call

    out = asyncio.run(eng._node_web_plots(dict(_NOSIM_STATE)))
    assert out == {}


def test_web_plots_passthrough_in_simulation_mode(tmp_path: Path) -> None:
    eng = _engine(tmp_path, web_derived_plots=True)

    async def must_not_call(prompt, *, node=None):
        raise AssertionError("simulation path makes its own figures")
    eng._chat = must_not_call

    sim_state = dict(_NOSIM_STATE)
    sim_state["no_simulation_resolved"] = False
    out = asyncio.run(eng._node_web_plots(sim_state))
    assert out == {}


# ---------------------------------------------------------------------------
# No-sim collector reuses already-retrieved literature (no web re-query)
# ---------------------------------------------------------------------------


def test_literature_seed_counts_without_duplicating(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    state = {"literature": [
        {"content": "EV outlook body", "metadata": {
            "source": "web_search", "kind": "web_page",
            "title": "IEA Global EV Outlook", "url": "https://iea.org/ev"}},
        {"content": "internal", "metadata": {"kind": "fi_paper_spine", "title": "mem"}},
        {"content": "x", "metadata": {}},  # no title/url → skipped
    ]}
    n = eng._literature_seed_step(state)
    assert n == 1   # one citable web source counted (internal + untitled skipped)
    # No duplicate copy is written: the sources already live in
    # data/literature/ (written by the literature node) and data_load walks
    # the whole data/ tree, so a second copy would only double-feed them.
    assert not (eng.quest_root / "data" / "auto_collected").exists()


def test_auto_collect_reuses_literature_without_requery(tmp_path: Path) -> None:
    """The fix for the live-run failure: when the literature node already
    fetched sources, the no-sim collector writes those to disk and does NOT
    re-query the web (which can be rate-limited and come back empty)."""
    eng = _engine(tmp_path)
    called = {"asearch": False}

    async def fake_asearch(*a, **k):
        called["asearch"] = True
        return []
    eng.knowledge.asearch = fake_asearch
    eng.knowledge.enabled = True

    state = {
        "no_simulation_resolved": True, "topic": "ev market share",
        "literature": [{
            "content": "2018: 2.0M, 2024: 17M EVs sold",
            "metadata": {"source": "web_search", "kind": "web_page",
                         "title": "IEA EV Outlook", "url": "https://iea.org/ev"}},
        ],
    }
    out = asyncio.run(eng._node_auto_collect_data(state))
    assert out["auto_collected_count"] >= 1
    assert called["asearch"] is False  # reused literature; no redundant web query
    # The reuse path counts the already-downloaded sources but writes NO
    # duplicate copy — they already live in data/literature/.
    assert not (eng.quest_root / "data" / "auto_collected").exists()


def test_write_literature_files_persists_citable_corpus(tmp_path: Path) -> None:
    """The shared writer drops one lit_NNN_slug.md per citable source and
    skips entries with no title/url and FI-internal artifacts."""
    eng = _engine(tmp_path)
    out_dir = eng.quest_root / "data" / "literature"
    lit = [
        {"content": "RPK recovered to 103% of 2019 by 2024", "metadata": {
            "source": "web_search", "kind": "web_page",
            "title": "IATA Global Outlook", "url": "https://iata.org/x"}},
        {"content": "internal", "metadata": {"kind": "fi_paper_spine", "title": "mem"}},
        {"content": "z", "metadata": {}},  # no title/url → skipped
    ]
    n = eng._write_literature_files(lit, out_dir)
    assert n == 1
    files = list(out_dir.glob("lit_*.md"))
    assert len(files) == 1
    txt = files[0].read_text(encoding="utf-8")
    assert "https://iata.org/x" in txt
    assert "RPK recovered" in txt


def test_write_literature_files_no_citable_leaves_no_dir(tmp_path: Path) -> None:
    """Lazy mkdir: a corpus with nothing citable leaves no empty dir."""
    eng = _engine(tmp_path)
    out_dir = eng.quest_root / "data" / "literature"
    n = eng._write_literature_files([{"content": "x", "metadata": {}}], out_dir)
    assert n == 0
    assert not out_dir.exists()


def test_node_literature_downloads_corpus_for_every_quest(tmp_path: Path) -> None:
    """The literature node downloads its retrieved sources to
    data/literature/ for EVERY quest — including simulation quests, which
    never reach auto_collect_data. Builds a simulation engine
    (no_simulation=False) and asserts the corpus lands on disk."""
    from core.knowledge import RetrievedDoc

    cfg = Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(no_simulation=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    eng = Engine(cfg)
    qr = tmp_path / "quest"
    qr.mkdir(exist_ok=True)
    eng.quest_root = qr

    docs = [
        RetrievedDoc(content="IATA reports RPK back to 103% by 2024",
                     metadata={"source": "web_search", "kind": "web_page",
                               "title": "IATA Global Outlook",
                               "url": "https://iata.org/outlook"}),
        RetrievedDoc(content="x", metadata={}),  # no title/url → skipped
    ]

    async def fake_asearch(*a, **k):
        return docs
    eng.knowledge.asearch = fake_asearch  # type: ignore[method-assign]

    out = asyncio.run(eng._node_literature({
        "topic": "air travel recovery 2019-2024",
        "chosen_idea": {"title": "Air travel recovery"},
    }))
    lit_dir = qr / "data" / "literature"
    assert lit_dir.is_dir(), "simulation quest must still download data/literature/"
    files = list(lit_dir.glob("lit_*.md"))
    assert len(files) == 1  # only the citable source is written to disk
    assert "https://iata.org/outlook" in files[0].read_text(encoding="utf-8")
    # The node still returns BOTH retrieved docs for downstream nodes
    # (the non-citable one is kept in state but not downloaded).
    assert len(out["literature"]) == 2


# ---------------------------------------------------------------------------
# Literature relevance floor (deterministic embedding filter)
# ---------------------------------------------------------------------------


def _lit_docs(*titles: str) -> list:
    return [{"content": t, "metadata": {"title": t}} for t in titles]


def test_relevance_floor_drops_off_topic(monkeypatch, tmp_path: Path) -> None:
    """Docs scoring below relevance_min_score are dropped; on-topic kept."""
    import core.passages as pmod
    eng = _engine(tmp_path)
    eng.config.knowledge.relevance_min_score = 0.2
    eng.config.knowledge.relevance_min_keep = 1
    docs = _lit_docs("History of Sculpture", "Change-Point Detection Math")
    monkeypatch.setattr(pmod, "_embed_scores", lambda blobs, q: [0.6, 0.05])
    kept = eng._filter_docs_by_relevance("sculpture history", docs)
    assert [d["metadata"]["title"] for d in kept] == ["History of Sculpture"]


def test_relevance_floor_retention_never_starves(monkeypatch, tmp_path: Path) -> None:
    """When every doc is below the floor, retain the top min_keep by score
    (never empty the corpus)."""
    import core.passages as pmod
    eng = _engine(tmp_path)
    eng.config.knowledge.relevance_min_score = 0.5
    eng.config.knowledge.relevance_min_keep = 2
    docs = _lit_docs("a", "b", "c")
    monkeypatch.setattr(pmod, "_embed_scores", lambda blobs, q: [0.10, 0.30, 0.05])
    kept = {d["metadata"]["title"] for d in eng._filter_docs_by_relevance("t", docs)}
    assert kept == {"a", "b"}  # top-2 by score, though both below 0.5


def test_relevance_floor_fail_open_without_embeddings(monkeypatch, tmp_path: Path) -> None:
    """No embeddings available (FI_OFFLINE / no model) → never filter blind."""
    import core.passages as pmod
    eng = _engine(tmp_path)
    eng.config.knowledge.relevance_min_score = 0.2
    docs = _lit_docs("a", "b")
    monkeypatch.setattr(pmod, "_embed_scores", lambda blobs, q: None)
    assert eng._filter_docs_by_relevance("t", docs) == docs


def test_relevance_floor_disabled_at_zero(monkeypatch, tmp_path: Path) -> None:
    """relevance_min_score=0 is a passthrough that never even embeds."""
    import core.passages as pmod
    eng = _engine(tmp_path)
    eng.config.knowledge.relevance_min_score = 0.0
    called = []
    monkeypatch.setattr(pmod, "_embed_scores", lambda blobs, q: called.append(1) or [0.0])
    docs = _lit_docs("a", "b")
    assert eng._filter_docs_by_relevance("t", docs) == docs
    assert not called  # short-circuited before embedding


def test_node_literature_applies_relevance_floor(monkeypatch, tmp_path: Path) -> None:
    """End-to-end: the literature node drops off-topic retrieved docs so only
    on-topic sources land in data/literature/ and in state."""
    from core.knowledge import RetrievedDoc
    import core.passages as pmod
    eng = _engine(tmp_path)  # the floor is mode-independent (runs in _node_literature)
    eng.config.knowledge.relevance_min_score = 0.2
    eng.config.knowledge.relevance_min_keep = 1
    docs = [
        RetrievedDoc(content="movie character figurines history",
                     metadata={"title": "Sculpture in Pop Culture",
                               "url": "https://ex.com/sculpt", "kind": "web_page"}),
        RetrievedDoc(content="kernel change-point detection",
                     metadata={"title": "Change-Point Algorithm",
                               "url": "https://arxiv.org/abs/x", "kind": "web_page"}),
    ]

    async def fake_asearch(*a, **k):
        return docs
    eng.knowledge.asearch = fake_asearch  # type: ignore[method-assign]
    # On-topic doc scores high, off-topic low.
    monkeypatch.setattr(pmod, "_embed_scores", lambda blobs, q: [0.55, 0.04])

    out = asyncio.run(eng._node_literature({
        "topic": "the evolution of sculpture in pop culture",
        "chosen_idea": {"title": "pop-culture sculpture"},
    }))
    titles = [d["metadata"].get("title") for d in out["literature"]]
    assert titles == ["Sculpture in Pop Culture"]  # off-topic dropped
    files = list((eng.quest_root / "data" / "literature").glob("lit_*.md"))
    assert len(files) == 1
    assert "Change-Point" not in files[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# References land in poster + slides (end-to-end through the generators)
# ---------------------------------------------------------------------------


def _artifacts_with_web_refs(tmp_path: Path) -> QuestArtifacts:
    quest_root = tmp_path / "quest_gen"
    (quest_root / "paper").mkdir(parents=True)
    paper_md = quest_root / "paper" / "paper.md"
    paper_md.write_text("# Paper\n\nBody.\n", encoding="utf-8")
    return QuestArtifacts(
        quest_id="q",
        quest_root=quest_root,
        paper_md=paper_md,
        raw_state={"literature": [{
            "content": "SpaceX page",
            "metadata": {"source": "web_search", "kind": "web_page",
                         "title": "SpaceX 2023 revenue", "site": "payloadspace.com",
                         "url": "https://payloadspace.com/spacex-2023"},
        }]},
    )


def _gen_config(tmp_path: Path, kind: str) -> Config:
    return Config(
        topic="t", title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=10),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out", kinds=[kind]),
    )


@pytest.mark.asyncio
async def test_poster_injects_web_sources_band(tmp_path, monkeypatch) -> None:
    from core.provider import ResolvedEndpoint
    from generation.poster import PosterGenerator

    cfg = _gen_config(tmp_path, "poster")
    art = _artifacts_with_web_refs(tmp_path)

    async def fake_resolve(provider, supervisor):
        return ResolvedEndpoint(base_url="http://127.0.0.1:1/v1", model="x", api_key="n")

    async def fake_chat(self, messages, **kw):
        return json.dumps({"title": "T", "left": "L", "middle": "M", "right": "R"})

    monkeypatch.setattr("generation.poster.resolve_endpoint_async", fake_resolve)
    monkeypatch.setattr("generation.poster.LLMClient.chat", fake_chat)
    monkeypatch.setattr("generation.poster.shutil.which", lambda _n: None)

    result = await PosterGenerator(cfg).generate(art, art.quest_root)
    tex = result["poster_tex"].read_text(encoding="utf-8")
    # Sources band injected by the template (not the LLM) with the web URL.
    assert "\\textbf{Sources:}" in tex
    assert "https://payloadspace.com/spacex-2023" in tex
    assert "$references" not in tex  # placeholder fully substituted


@pytest.mark.asyncio
async def test_slides_appends_references_slide(tmp_path, monkeypatch) -> None:
    from core.provider import ResolvedEndpoint
    from generation.slides import SlideGenerator

    cfg = _gen_config(tmp_path, "slides")
    art = _artifacts_with_web_refs(tmp_path)

    async def fake_resolve(provider, supervisor):
        return ResolvedEndpoint(base_url="http://127.0.0.1:1/v1", model="x", api_key="n")

    async def fake_chat(self, messages, **kw):
        return "---\nmarp: true\n---\n\n# Deck\n\n---\n\n## Findings\n\nBody."

    monkeypatch.setattr("generation.slides.resolve_endpoint_async", fake_resolve)
    monkeypatch.setattr("generation.slides.LLMClient.chat", fake_chat)
    # No marp / pandoc on PATH → only slides.md is produced (which is what
    # we assert on); render targets skip cleanly.
    monkeypatch.setattr("generation.slides.shutil.which", lambda _n: None)

    result = await SlideGenerator(cfg).generate(art, art.quest_root)
    md = result["slides_md"].read_text(encoding="utf-8")
    assert "## References" in md
    assert "https://payloadspace.com/spacex-2023" in md


# ---------------------------------------------------------------------------
# Web-plot house style — every web_plots script is prefixed with the FI
# brand style so the figures match the paper / poster / slides identity.
# ---------------------------------------------------------------------------

def test_fi_mpl_preamble_applies_brand_style() -> None:
    """The injected house-style preamble compiles and brands matplotlib:
    teal leads the colour cycle, the font is serif, top/right spines off."""
    pytest.importorskip("matplotlib")
    import matplotlib as mpl
    from core.engine import _FI_MPL_PREAMBLE

    saved = mpl.rcParams.copy()
    try:
        ns: dict = {}
        exec(compile(_FI_MPL_PREAMBLE, "<fi_preamble>", "exec"), ns)
        assert ns["FI_TEAL"] == "#0E6E6B"
        colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        assert colors[0] == "#0E6E6B"               # teal leads the cycle
        assert mpl.rcParams["font.family"] == ["serif"]
        assert mpl.rcParams["axes.spines.top"] is False
        assert mpl.rcParams["axes.spines.right"] is False
    finally:
        mpl.rcParams.update(saved)


# ---------------------------------------------------------------------------
# web_figures node — license-clean illustrative figures with attribution
# ---------------------------------------------------------------------------

def test_web_figures_node_embeds_and_credits(monkeypatch, tmp_path: Path) -> None:
    import core.figure_sources as fsmod
    from core.figure_sources import WebFigure

    eng = _engine(tmp_path, web_derived_plots=True)
    eng.config.knowledge.fetch_web_figures = True
    eng.config.knowledge.web_figures_max = 1
    figdir = eng.quest_root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    a = figdir / "webfig_arxiv_x.png"; a.write_bytes(b"\x89PNG\r\n")
    b = figdir / "webfig_commons_y.png"; b.write_bytes(b"\x89PNG\r\n")
    cands = [
        WebFigure(a, "Detection methods overview", "https://arxiv.org/abs/x",
                  "CC BY 4.0", "arXiv:x", "oa_paper"),
        WebFigure(b, "Off-topic image", "https://commons.example/y",
                  "CC0", "Someone", "commons"),
    ]
    monkeypatch.setattr(fsmod, "collect_topic_figures", lambda *a, **k: cands)
    monkeypatch.setattr("core.engine._stamp_figure_credit", lambda *a, **k: None)

    async def fake_chat(prompt, *, node=None):
        assert node == "web_figures"
        return "[0]"  # keep only the arXiv (cited-paper) figure

    eng._chat = fake_chat
    state = {"no_simulation_resolved": True, "topic": "exoplanet detection",
             "literature": [], "figures": ["existing.png"]}
    out = asyncio.run(eng._node_web_figures(state))
    assert out["figures"] == ["existing.png", "webfig_arxiv_x.png"]
    assert [c["license"] for c in out["figure_credits"]] == ["CC BY 4.0"]
    assert a.exists()          # selected, kept
    assert not b.exists()      # not selected → removed


def test_web_figures_node_off_by_default(tmp_path: Path) -> None:
    eng = _engine(tmp_path, web_derived_plots=True)  # fetch_web_figures defaults False
    state = {"no_simulation_resolved": True, "topic": "t", "literature": [], "figures": []}
    assert asyncio.run(eng._node_web_figures(state)) == {}


def test_web_figures_forced_on_in_survey_mode(monkeypatch, tmp_path: Path) -> None:
    """Survey mode is text-only (no experiment, no data plots), so illustrative
    images are the paper's only visuals — web_figures runs even though
    ``knowledge.fetch_web_figures`` is off by default."""
    import core.figure_sources as fsmod
    from core.figure_sources import WebFigure

    eng = _engine(tmp_path, web_derived_plots=True)
    assert eng.config.knowledge.fetch_web_figures is False  # default off
    figdir = eng.quest_root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    a = figdir / "webfig_arxiv_x.png"; a.write_bytes(b"\x89PNG\r\n")
    cands = [WebFigure(a, "A diagram", "https://arxiv.org/abs/x",
                       "CC BY 4.0", "arXiv:x", "oa_paper")]
    monkeypatch.setattr(fsmod, "collect_topic_figures", lambda *a, **k: cands)
    monkeypatch.setattr("core.engine._stamp_figure_credit", lambda *a, **k: None)

    async def fake_chat(prompt, *, node=None):
        return "[0]"
    eng._chat = fake_chat
    # survey implies no_simulation; both set as the engine would at resolve time.
    state = {"no_simulation_resolved": True, "survey_mode_resolved": True,
             "topic": "history of sculpture", "literature": [], "figures": []}
    out = asyncio.run(eng._node_web_figures(state))
    assert out["figures"] == ["webfig_arxiv_x.png"]


def test_figure_list_for_prompt_annotates_credits() -> None:
    from core.engine import _figure_list_for_prompt
    state = {
        "figures": ["chart.png", "webfig_arxiv_x.png"],
        "figure_credits": [{
            "file": "webfig_arxiv_x.png", "caption": "Detection methods tree",
            "source_url": "https://arxiv.org/abs/x", "license": "CC BY 4.0",
        }],
    }
    block = _figure_list_for_prompt(state)
    assert "- figures/chart.png" in block            # plain chart unannotated
    assert "ILLUSTRATIVE" in block and "Detection methods tree" in block
    assert "CC BY 4.0" in block and "arxiv.org/abs/x" in block


def test_arxiv_ids_from_literature_old_and_new_style(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    state = {"literature": [
        {"content": "", "metadata": {"url": "https://arxiv.org/abs/2404.09143"}},
        {"content": "", "metadata": {"url": "https://arxiv.org/pdf/hep-th/9701001v2"}},
        {"content": "", "metadata": {"arxiv_id": "2210.04940"}},
        {"content": "", "metadata": {"url": "https://example.com/not-arxiv"}},
    ]}
    ids = eng._arxiv_ids_from_literature(state)
    assert "2404.09143" in ids          # new-style from URL
    assert "hep-th/9701001" in ids      # old-style from URL (Copilot fix)
    assert "2210.04940" in ids          # from metadata
