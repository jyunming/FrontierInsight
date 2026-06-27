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


def test_literature_seed_writes_files_with_source_urls(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    auto_dir = eng.quest_root / "data" / "auto_collected"
    state = {"literature": [
        {"content": "EV outlook body", "metadata": {
            "source": "web_search", "kind": "web_page",
            "title": "IEA Global EV Outlook", "url": "https://iea.org/ev"}},
        {"content": "internal", "metadata": {"kind": "fi_paper_spine", "title": "mem"}},
        {"content": "x", "metadata": {}},  # no title/url → skipped
    ]}
    n = eng._literature_seed_step(state, auto_dir)
    assert n == 1
    files = list(auto_dir.glob("*.md"))
    assert len(files) == 1
    txt = files[0].read_text(encoding="utf-8")
    assert "https://iea.org/ev" in txt   # source URL preserved in front matter
    assert "EV outlook body" in txt


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
    assert (eng.quest_root / "data" / "auto_collected").is_dir()


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
