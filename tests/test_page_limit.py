"""A page limit, set in the config or stated in the topic.

Graded SIR papers ran to 5 and 6 pages against the topic's "(<= 4 pages)".
With a limit, the paper templates switch to a tighter layout, the writer gets
a word budget, and the review renders every draft the way paper.pdf is
rendered and counts its pages: a draft over the limit goes back to be
shortened, at most twice, without using ``engine.max_iterations``. Without a
limit nothing changes.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig,
    ProviderConfig, page_limit_from_text, resolve_page_limit,
)
from core.engine import (
    _PAGE_LIMIT_REWRITES,
    Engine,
    _format_review_for_writer,
    _hits_need_only_a_rewrite,
    _only_page_limit_hits,
    _page_limit_hit,
    _page_limit_note,
    _words_to_cut,
)
from generation import paper as paper_mod
from generation.paper import PaperGenerator, small_source_lists

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "templates" / "paper"
FORMATS = sorted(p.parent.name for p in TEMPLATES.glob("*/template.tex"))
ONE_INCH_MARGIN = {"generic", "iclr", "neurips", "report", "whitepaper"}

# The topic of the graded SIR trend runs, word for word.
SIR_TOPIC = """TOPIC: Compare a deterministic SIR epidemic model (ODE) with its stochastic
counterpart (Gillespie direct method) in a closed, well-mixed population.

GOALS:
1. Implement the SIR ODE and a Gillespie stochastic simulation in Python
   (numpy; scipy allowed for the ODE).
2. For R0 in {0.9, 1.5, 3.0} and population N in {100, 1000, 5000}, with one
   initial infective, estimate the probability of a major outbreak and the
   final epidemic size from 300 stochastic runs per setting.
3. Compare the stochastic final sizes with the deterministic final-size
   relation, and the outbreak probability with the branching-process
   estimate 1 - 1/R0.
4. Cross-reference results with the literature on stochastic epidemic
   models and early extinction.
5. Produce a short markdown paper (<= 4 pages, IMRAD) with three figures.
6. Cite at least three primary references.
"""

PAPER = """# Outbreaks in a Small Population

## Abstract

One paragraph.

## Introduction

Text [1].

## References

1. Kermack, W. O. and McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics.

## Further reading

- [W1] Stochastic SIR model. https://example.org/sir
"""


def _topic(path: str) -> str:
    return yaml.safe_load((REPO / path).read_text(encoding="utf-8"))["topic"]


# --- reading the limit ---------------------------------------------------------

@pytest.mark.parametrize("path", [
    "examples/integrator_bakeoff/config.yaml",
    "examples/bernstein_vazirani_noise/config.yaml",
    "examples/euv_mor_shot_noise/config.yaml",
])
def test_the_example_topics_state_four_pages(path: str) -> None:
    assert page_limit_from_text(_topic(path)) == 4


def test_the_sir_trend_topic_states_four_pages() -> None:
    assert page_limit_from_text(SIR_TOPIC) == 4


def test_no_other_example_topic_states_a_limit() -> None:
    stated = {"integrator_bakeoff", "bernstein_vazirani_noise", "euv_mor_shot_noise"}
    others = [
        p for p in sorted((REPO / "examples").glob("*/*.yaml"))
        if p.parent.name not in stated and "topic" in (yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    ]
    # The format_validation set: a brief preprint and journal-length papers
    # in nine formats, none with a page limit.
    assert len(others) >= 9
    for path in others:
        assert page_limit_from_text(_topic(str(path.relative_to(REPO)))) is None, path


@pytest.mark.parametrize("text, limit", [
    ("(≤ 4 pages, IMRAD)", 4),
    ("(<= 4 pages, IMRAD)", 4),
    ("≤4 pages", 4),
    ("at most four pages", 4),
    ("no more than 6 pages", 6),
    ("a 4-page paper", 4),
    ("3 pages max", 3),
    ("4 pages or fewer", 4),
    ("up to 5 pages", 5),
    ("a maximum of 2 pages", 2),
    ("page limit: 8", 8),
    # A bound in words wins over the "N-page" form, which can name a part.
    ("a 2-page appendix, and the paper at most 6 pages", 6),
    ("at most 6 pages, ideally no more than 5 pages", 5),
])
def test_a_stated_limit_is_read(text: str, limit: int) -> None:
    assert page_limit_from_text(text) == limit


@pytest.mark.parametrize("text", [
    "4–8 pages",
    "4-8 pages",
    "4 to 8 pages",
    "a 4–8-page paper",
    "no more than 4 or 5 pages",
    "journal-length — 4–8 pages. Aim for ~1500–2500 words.",
    "cite at most 3 web pages",
    "summarise 10 pages of notes",
    "at most 0 pages",
    "",
])
def test_a_length_that_is_not_a_limit_is_not_read(text: str) -> None:
    assert page_limit_from_text(text) is None


def test_the_writers_own_length_guide_states_no_limit() -> None:
    assert page_limit_from_text((REPO / "agents" / "write.md").read_text(encoding="utf-8")) is None


def test_the_config_value_wins_over_the_topic() -> None:
    assert resolve_page_limit(Config(topic=SIR_TOPIC, output=OutputConfig(page_limit=6))) == 6
    assert resolve_page_limit(Config(topic=SIR_TOPIC)) == 4
    assert resolve_page_limit(Config(topic="Extinction times in a stochastic SIR model")) is None
    assert resolve_page_limit(SimpleNamespace()) is None
    with pytest.raises(ValidationError):
        OutputConfig(page_limit=0)


def test_page_limit_is_a_known_output_key(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("topic: t\noutput:\n  page_limit: 3\n", encoding="utf-8")
    assert Config.from_yaml(config).output.page_limit == 3


# --- the templates -------------------------------------------------------------

def _code_lines(fmt: str) -> list[str]:
    text = (TEMPLATES / fmt / "template.tex").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if not line.lstrip().startswith("%")]


def _switch(tight: str, default: str) -> list[str]:
    return ["$if(fi-tight)$", tight, "$else$", default, "$endif$"]


def _contains_run(lines: list[str], run: list[str]) -> bool:
    return any(lines[i:i + len(run)] == run for i in range(len(lines)))


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_template_switches_its_figure_cap_on_fi_tight(fmt: str) -> None:
    cap = r"  \Gscale@div\@tempa{%s\textheight}{\dimexpr\ht\FI@figbox+\dp\FI@figbox\relax}%%"
    assert _contains_run(_code_lines(fmt), _switch(cap % "0.33", cap % "0.45")), fmt


@pytest.mark.parametrize("fmt", FORMATS)
def test_templates_with_one_inch_margins_switch_to_two_centimetres(fmt: str) -> None:
    lines = _code_lines(fmt)
    switched = _contains_run(
        lines, _switch(r"\usepackage[margin=2cm]{geometry}", r"\usepackage[margin=1in]{geometry}"),
    )
    assert switched == (fmt in ONE_INCH_MARGIN), fmt
    # A template with its own margins keeps them: ieee_access's 0.75 in is
    # already tighter than 2 cm.
    if fmt not in ONE_INCH_MARGIN:
        assert sum("geometry}" in line for line in lines) == 1, fmt


@pytest.mark.parametrize("fmt", FORMATS)
def test_pandoc_renders_the_default_layout_without_the_flag_and_the_tight_one_with_it(
    tmp_path: Path, fmt: str,
) -> None:
    from generation._pandoc import find_pandoc

    pandoc = find_pandoc(REPO)
    if pandoc is None:
        pytest.skip("needs pandoc")
    md = tmp_path / "t.md"
    md.write_text("---\ntitle: T\n---\n\n# Body\n\nText.\n", encoding="utf-8")

    def render(*extra: str) -> str:
        return subprocess.run(
            [pandoc, str(md), "--from=markdown", "-t", "latex", "--standalone",
             "--template", str(TEMPLATES / fmt / "template.tex"), *extra],
            capture_output=True, text=True, encoding="utf-8", check=True,
        ).stdout

    default, tight = render(), render("-V", "fi-tight=true")
    assert r"{0.45\textheight}" in default and r"{0.33\textheight}" not in default
    assert r"{0.33\textheight}" in tight and r"{0.45\textheight}" not in tight
    if fmt in ONE_INCH_MARGIN:
        assert r"\usepackage[margin=1in]{geometry}" in default and "margin=2cm" not in default
        assert r"\usepackage[margin=2cm]{geometry}" in tight and "margin=1in" not in tight
    for latex in (default, tight):
        body = latex.split(r"\providecommand{\pandocbounded}[1]{%", 1)[1].split(r"\usebox{\FI@figbox}}}", 1)[0]
        assert "\n\n" not in body, "a blank line in the macro body would end a paragraph inside it"


# --- the PDF render --------------------------------------------------------------

def _config(tmp_path: Path, topic: str = "t", **output: Any) -> Config:
    return Config.model_validate({
        "topic": topic,
        "title": "t",
        "output": {
            "kinds": ["paper_md", "paper_pdf"],
            "output_dir": str(tmp_path / "outputs"),
            "html_pdf_fallback": False,
            **output,
        },
    })


def _fake_toolchain(monkeypatch) -> list[list[str]]:  # noqa: ANN001
    monkeypatch.setattr(paper_mod, "find_pandoc", lambda _root: "/fake/pandoc")
    monkeypatch.setattr(PaperGenerator, "_find_pdf_engine", lambda self: ("pdflatex", "/fake/pdflatex"))
    commands: list[list[str]] = []

    def fake_run(cmd, **_kw):  # noqa: ANN001
        commands.append(list(cmd))
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"%PDF-fake\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(paper_mod.subprocess, "run", fake_run)
    return commands


def _paper(tmp_path: Path, name: str = "paper.md") -> Path:
    md = tmp_path / name
    md.write_text(PAPER, encoding="utf-8")
    return md


def test_the_tight_layout_reaches_the_template_only_with_a_limit(tmp_path: Path, monkeypatch) -> None:
    commands = _fake_toolchain(monkeypatch)
    md = _paper(tmp_path)
    for index, cfg in enumerate([
        _config(tmp_path),
        _config(tmp_path, topic=SIR_TOPIC),
        _config(tmp_path, page_limit=3),
    ]):
        out = tmp_path / f"out{index}"
        out.mkdir()
        PaperGenerator(cfg)._compile_pdf(md, out)
    assert "fi-tight=true" not in commands[0]
    for cmd in commands[1:]:
        assert cmd[cmd.index("fi-tight=true") - 1] == "-V"


def test_source_lists_are_set_small_in_the_pdf_source_only_and_only_with_a_limit(
    tmp_path: Path, monkeypatch,
) -> None:
    _fake_toolchain(monkeypatch)
    md = _paper(tmp_path)
    before = md.read_bytes()
    plain, tight = tmp_path / "plain", tmp_path / "tight"
    plain.mkdir()
    tight.mkdir()
    PaperGenerator(_config(tmp_path))._compile_pdf(md, plain)
    PaperGenerator(_config(tmp_path, page_limit=4))._compile_pdf(md, tight)
    assert md.read_bytes() == before, "paper.md itself is never changed"
    assert "begingroup" not in (plain / "paper_pdf_source.md").read_text(encoding="utf-8")
    source = (tight / "paper_pdf_source.md").read_text(encoding="utf-8")
    assert source.count("\\begingroup\\small") == 2 and source.count("\\endgroup") == 2
    references = source.index("## References")
    further = source.index("## Further reading")
    assert references < source.index("\\begingroup\\small", references) < source.index("1. Kermack") < further
    assert further < source.index("\\begingroup\\small", further) < source.index("- [W1]")


def test_small_source_lists_wraps_the_list_bodies_and_nothing_else() -> None:
    md = PAPER + "\n## Appendix\n\nA table.\n"
    out = small_source_lists(md)
    begin, end = "```{=latex}\n\\begingroup\\small\n```", "```{=latex}\n\\endgroup\n```"
    assert out.startswith(PAPER.split("## References")[0])
    assert "## References\n\n" + begin + "\n\n1. Kermack" in out
    assert "https://example.org/sir\n\n" + end + "\n\n## Appendix\n\nA table.\n" in out
    assert small_source_lists("# T\n\n## Results\n\nText.\n") == "# T\n\n## Results\n\nText.\n"
    assert small_source_lists("# T\n\n## References\n") == "# T\n\n## References\n"


def test_raw_latex_passes_through_pandoc_to_the_latex(tmp_path: Path) -> None:
    from generation._pandoc import find_pandoc

    pandoc = find_pandoc(REPO)
    if pandoc is None:
        pytest.skip("needs pandoc")
    md = tmp_path / "t.md"
    md.write_text(small_source_lists(PAPER), encoding="utf-8")
    latex = subprocess.run(
        [pandoc, str(md), f"--from={paper_mod.MARKDOWN_READER}", "-t", "latex"],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    assert re.search(r"\\begingroup\\small\s+\\begin\{enumerate\}", latex)
    assert re.search(r"\\end\{itemize\}\s+\\endgroup", latex)
    assert "{=latex}" not in latex


def test_the_html_render_never_gets_the_raw_latex(tmp_path: Path, monkeypatch) -> None:
    import generation._html_pdf as html_pdf

    monkeypatch.setattr(paper_mod, "find_pandoc", lambda _root: "/fake/pandoc")
    monkeypatch.setattr(html_pdf, "find_html_browser", lambda: ("edge", "/fake/edge"))
    seen: list[str] = []

    def fake_render(paper_md, out_pdf, **_kw):  # noqa: ANN001
        seen.append(Path(paper_md).read_text(encoding="utf-8"))
        Path(out_pdf).write_bytes(b"%PDF-fake\n")
        return Path(out_pdf), ""

    monkeypatch.setattr(html_pdf, "render_paper_html_pdf", fake_render)
    out = tmp_path / "out"
    out.mkdir()
    pdf, _skip = PaperGenerator(_config(tmp_path, page_limit=4, paper_style="briefing"))._compile_pdf(_paper(tmp_path), out)
    assert pdf is not None and seen and "begingroup" not in seen[0]


# --- the review ----------------------------------------------------------------

def _engine(tmp_path: Path, *, topic: str = SIR_TOPIC, max_iterations: int = 2, panel: bool = False) -> Engine:
    eng = Engine(Config(
        topic=topic, title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=max_iterations, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        pauses=PausesConfig(review="off"),
        output=OutputConfig(output_dir=tmp_path / "out", kinds=["paper_md"]),
    ))
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    (tmp_path / "paper").mkdir(parents=True, exist_ok=True)
    (tmp_path / "paper" / "paper.md").write_text(PAPER, encoding="utf-8")
    if panel:
        eng.config.engine.review_panel = ["methodologist", "statistician"]

    async def accepting(prompt, *, node=None):  # noqa: ANN001, ARG001
        if node == "review_moderator":
            return json.dumps({"rationale": "both accept"})
        return json.dumps({"verdict": "accept", "score": 5, "suggestions": [], "must_flag_hits": []})

    eng._chat = accepting  # type: ignore[assignment,method-assign]
    return eng


def _measured(monkeypatch, pages: int | None, *, last_page_empty: float = 0.5) -> list[Path]:  # noqa: ANN001
    calls: list[Path] = []

    async def fake(self, paper_path, state):  # noqa: ANN001, ARG001
        calls.append(Path(paper_path))
        if pages is None:
            return None
        return {"pages": pages, "words": 1200, "last_page_lines": 20, "last_page_empty": last_page_empty}

    monkeypatch.setattr(Engine, "_measure_draft_pages", fake)
    return calls


def _review(eng: Engine, tmp_path: Path, **state: Any) -> dict[str, Any]:
    return asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": eng.config.topic, "iteration": 0, "review": {},
        "paper_md": str(tmp_path / "paper" / "paper.md"), **state,
    }))


@pytest.mark.parametrize("panel", [False, True])
def test_a_draft_over_the_limit_is_sent_back_without_using_an_iteration(
    tmp_path: Path, monkeypatch, panel: bool,
) -> None:
    eng = _engine(tmp_path, panel=panel)
    calls = _measured(monkeypatch, 5)
    # The iterations are already spent: the shortening still happens.
    patch = _review(eng, tmp_path, iteration=2, page_limit_rewrites=0)
    assert calls == [tmp_path / "paper" / "paper.md"]
    hits = patch["review"]["must_flag_hits"]
    assert hits == [_page_limit_hit(5, 4, 250)]
    assert patch["review"]["page_limit"] == {"pages": 5, "limit": 4, "rewrites": 0, "words_to_cut": 250}
    assert "iteration" not in patch
    assert patch["page_limit_rewrites"] == 1
    state = {"review": patch["review"], "iteration": 2, "page_limit_rewrites": 1}
    assert eng._route_after_review(state) == "rewrite"  # type: ignore[arg-type]


@pytest.mark.parametrize("panel", [False, True])
def test_after_two_shortening_rewrites_the_overrun_is_recorded_not_forced(
    tmp_path: Path, monkeypatch, panel: bool,
) -> None:
    eng = _engine(tmp_path, panel=panel)
    _measured(monkeypatch, 5)
    patch = _review(eng, tmp_path, iteration=0, page_limit_rewrites=_PAGE_LIMIT_REWRITES)
    assert patch["review"]["must_flag_hits"] == []
    assert patch["review"]["page_limit"] == {
        "pages": 5, "limit": 4, "rewrites": 2, "exceeded_after_rewrites": True,
    }
    assert "page_limit_rewrites" not in patch and "iteration" not in patch
    state = {"review": patch["review"], "iteration": 0, "page_limit_rewrites": 2}
    assert eng._route_after_review(state) == "done"  # type: ignore[arg-type]


@pytest.mark.parametrize("panel", [False, True])
def test_other_hits_take_the_normal_path_and_the_shortening_still_counts(
    tmp_path: Path, monkeypatch, panel: bool,
) -> None:
    eng = _engine(tmp_path, panel=panel)
    _measured(monkeypatch, 6, last_page_empty=0.2)
    patch = _review(eng, tmp_path, iteration=0, claim_check_failed="No capacity")
    hits = patch["review"]["must_flag_hits"]
    assert len(hits) == 2 and hits[0].startswith("citations_unchecked") and hits[1] == _page_limit_hit(6, 4, 800)
    assert patch["iteration"] == 1 and patch["page_limit_rewrites"] == 1
    assert eng._route_after_review({"review": patch["review"], "iteration": 1, "page_limit_rewrites": 1}) == "rewrite"  # type: ignore[arg-type]
    # With the iterations spent, other hits do not get past the budget.
    assert eng._route_after_review({"review": patch["review"], "iteration": 2, "page_limit_rewrites": 1}) == "done"  # type: ignore[arg-type]


@pytest.mark.parametrize("panel", [False, True])
def test_a_draft_within_the_limit_is_recorded_and_not_forced(tmp_path: Path, monkeypatch, panel: bool) -> None:
    eng = _engine(tmp_path, panel=panel)
    _measured(monkeypatch, 4)
    patch = _review(eng, tmp_path)
    assert patch["review"]["must_flag_hits"] == []
    assert patch["review"]["page_limit"] == {"pages": 4, "limit": 4, "rewrites": 0}
    assert "page_limit_rewrites" not in patch


@pytest.mark.parametrize("panel", [False, True])
def test_without_a_limit_the_draft_is_never_rendered(tmp_path: Path, monkeypatch, panel: bool) -> None:
    eng = _engine(tmp_path, topic="Extinction times in a stochastic SIR model", panel=panel)

    async def boom(self, paper_path, state):  # noqa: ANN001, ARG001
        raise AssertionError("rendered without a page limit")

    monkeypatch.setattr(Engine, "_measure_draft_pages", boom)
    patch = _review(eng, tmp_path)
    assert "page_limit" not in patch["review"] and "page_limit_rewrites" not in patch
    assert patch["review"]["must_flag_hits"] == []


def test_a_draft_that_cannot_be_measured_is_not_forced(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    _measured(monkeypatch, None)
    patch = _review(eng, tmp_path)
    assert patch["review"]["must_flag_hits"] == [] and "page_limit" not in patch["review"]


@pytest.mark.parametrize("panel", [False, True])
@pytest.mark.parametrize("topic", [SIR_TOPIC, "Extinction times in a stochastic SIR model"])
def test_an_over_page_limit_hit_the_reviewer_writes_is_dropped(
    tmp_path: Path, monkeypatch, panel: bool, topic: str,
) -> None:
    """Only the measured page count forces the hit. A reviewer's own
    ``over_page_limit`` never moves the shortening counter, so kept, it would
    send the paper back to be rewritten on every review, without end."""
    eng = _engine(tmp_path, topic=topic, panel=panel)
    _measured(monkeypatch, 4)  # within the limit, when there is one

    async def flagging(prompt, *, node=None):  # noqa: ANN001, ARG001
        if node == "review_moderator":
            return json.dumps({"rationale": "all accept"})
        return json.dumps({
            "verdict": "accept", "score": 4, "suggestions": [],
            "must_flag_hits": ["over_page_limit: the paper looks longer than four pages"],
        })

    eng._chat = flagging  # type: ignore[assignment,method-assign]
    patch = _review(eng, tmp_path)
    assert patch["review"]["must_flag_hits"] == []
    assert "page_limit_rewrites" not in patch and "iteration" not in patch
    assert eng._route_after_review({"review": patch["review"], "iteration": 0}) == "done"  # type: ignore[arg-type]


@pytest.mark.parametrize("panel", [False, True])
def test_a_measured_overrun_replaces_the_reviewers_own_page_hit(tmp_path: Path, monkeypatch, panel: bool) -> None:
    eng = _engine(tmp_path, panel=panel)
    _measured(monkeypatch, 5)

    async def flagging(prompt, *, node=None):  # noqa: ANN001, ARG001
        if node == "review_moderator":
            return json.dumps({"rationale": "all accept"})
        return json.dumps({
            "verdict": "accept", "score": 4, "suggestions": [], "must_flag_hits": ["over_page_limit"],
        })

    eng._chat = flagging  # type: ignore[assignment,method-assign]
    patch = _review(eng, tmp_path)
    assert patch["review"]["must_flag_hits"] == [_page_limit_hit(5, 4, 250)]
    assert patch["page_limit_rewrites"] == 1


def test_only_page_limit_hits() -> None:
    hit = _page_limit_hit(5, 4, 100)
    assert _only_page_limit_hits([hit]) and _only_page_limit_hits([hit, hit])
    assert not _only_page_limit_hits([]) and not _only_page_limit_hits([hit, "unsupported_claim"])
    assert _hits_need_only_a_rewrite([hit]) and _hits_need_only_a_rewrite([hit, "figure_caption: x"])


def test_the_route_bypasses_the_iteration_budget_only_within_the_shortening_cap(tmp_path: Path) -> None:
    eng = _engine(tmp_path, max_iterations=0)
    review = {"verdict": "accept", "must_flag_hits": [_page_limit_hit(5, 4, 100)]}
    assert eng._route_after_review({"review": review, "iteration": 0, "page_limit_rewrites": 2}) == "rewrite"  # type: ignore[arg-type]
    assert eng._route_after_review({"review": review, "iteration": 0, "page_limit_rewrites": 3}) == "done"  # type: ignore[arg-type]


def test_the_graph_shortens_the_paper_after_the_iterations_are_spent(tmp_path: Path) -> None:
    eng = _engine(tmp_path, max_iterations=2)
    visits: list[str] = []
    reviews = iter([
        {"review": {"verdict": "accept", "must_flag_hits": [_page_limit_hit(5, 4, 250)]},
         "page_limit_rewrites": 1},
        {"review": {"verdict": "accept", "must_flag_hits": []}},
    ])

    def node(name: str, patch: Any = None) -> Any:
        async def run(state: dict[str, Any]) -> dict[str, Any]:
            visits.append(name)
            return patch() if patch else {}
        return run

    for name in ("design", "implement_outline", "implement", "execute", "analyze", "write", "claim_check"):
        setattr(eng, f"_node_{name}", node(name))
    eng._node_review = node("review", lambda: next(reviews))  # type: ignore[method-assign]
    eng._route_after_evidence_gate = lambda state: "write"  # type: ignore[method-assign]
    graph = eng._build_graph().compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "shorten"}}
    graph.update_state(config, {"topic": SIR_TOPIC, "iteration": 2}, as_node="evidence_gate")
    asyncio.run(graph.ainvoke(None, config))
    assert visits == ["write", "claim_check", "review", "write", "claim_check", "review"]
    values = graph.get_state(config).values
    assert values["iteration"] == 2 and values["page_limit_rewrites"] == 1


# --- measuring a draft -----------------------------------------------------------

def _blank_pdf(path: Path, pages: int) -> Path:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(595, 842)
    pdf.save(str(path))
    pdf.close()
    return path


def _quiet_log(eng: Engine) -> list[str]:
    warnings: list[str] = []
    eng._log = SimpleNamespace(  # type: ignore[assignment]
        warning=lambda msg, *a: warnings.append(msg % a if a else msg),
        info=lambda *a, **k: None,
    )
    return warnings


def test_a_draft_is_measured_from_a_render_in_the_quest_fi_folder(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "a.png").write_bytes(b"png")
    seen: dict[str, Any] = {}

    def fake_compile(self, paper_md, out_dir, *, extra_lines=0):  # noqa: ANN001, ARG001
        seen.update(config=self.config, paper_md=Path(paper_md), out_dir=Path(out_dir),
                    figures=sorted(p.name for p in (Path(out_dir) / "figures").iterdir()))
        return _blank_pdf(Path(out_dir) / "paper.pdf", 3), None

    monkeypatch.setattr(PaperGenerator, "_compile_pdf", fake_compile)
    before = sorted(p.name for p in tmp_path.iterdir())
    result = asyncio.run(eng._measure_draft_pages(
        tmp_path / "paper" / "paper.md", {"clarify_answers": {"paper_venue": "ieee_access"}},
    ))
    assert result is not None and result["pages"] == 3
    assert seen["out_dir"] == tmp_path / ".fi" / "page_check"
    assert seen["paper_md"] == tmp_path / "paper" / "paper.md"
    assert seen["figures"] == ["a.png"]
    # The draft is rendered as paper.pdf will be: LaTeX only, in the venue the
    # final render uses, and with the same page limit.
    assert seen["config"].output.html_pdf_fallback is False
    assert seen["config"].output.paper_format == "ieee_access"
    assert resolve_page_limit(seen["config"]) == 4
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted({*before, ".fi"})


def test_a_draft_that_will_render_through_html_is_not_measured(tmp_path: Path, monkeypatch) -> None:
    import generation._html_pdf as html_pdf

    eng = _engine(tmp_path)
    eng.config.output.paper_style = "briefing"
    monkeypatch.setattr(html_pdf, "find_html_browser", lambda: ("edge", "/fake/edge"))

    def never(*_a, **_k):  # noqa: ANN002, ANN003
        raise AssertionError("compiled a briefing paper")

    monkeypatch.setattr(PaperGenerator, "_compile_pdf", never)
    warnings = _quiet_log(eng)
    for _ in range(2):
        assert asyncio.run(eng._measure_draft_pages(tmp_path / "paper" / "paper.md", {})) is None
    assert len(warnings) == 1 and "briefing" in warnings[0]


def test_a_draft_that_does_not_render_is_skipped_with_one_warning(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    reason = SimpleNamespace(summary="no LaTeX engine found")
    monkeypatch.setattr(PaperGenerator, "_compile_pdf", lambda self, md, out, **_k: (None, reason))
    warnings = _quiet_log(eng)
    for _ in range(2):
        assert asyncio.run(eng._measure_draft_pages(tmp_path / "paper" / "paper.md", {})) is None
    assert len(warnings) == 1 and "no LaTeX engine found" in warnings[0]


@pytest.mark.slow
def test_a_real_render_forces_a_long_draft_and_passes_a_short_one(tmp_path: Path) -> None:
    from generation._pandoc import find_pandoc
    from generation._pdf_engine import find_pdf_engine

    if find_pandoc(REPO) is None or find_pdf_engine(REPO) is None:
        pytest.skip("needs pandoc and a LaTeX engine")
    eng = _engine(tmp_path, topic="A short paper (at most 2 pages).")
    paper = tmp_path / "paper" / "paper.md"
    filler = " ".join(f"Sentence {k} reports one more result of the outbreak study." for k in range(260))
    paper.write_text(f"# A Long Draft\n\n## Abstract\n\nShort.\n\n## Results\n\n{filler}\n", encoding="utf-8")
    hits, record = asyncio.run(eng._page_limit_review({}, paper))  # type: ignore[arg-type]
    assert record is not None and record["pages"] > 2, record
    assert len(hits) == 1 and hits[0].startswith("over_page_limit: the rendered PDF is")
    paper.write_text("# A Short Draft\n\n## Abstract\n\nShort.\n\n## Results\n\nOne result.\n", encoding="utf-8")
    hits, record = asyncio.run(eng._page_limit_review({}, paper))  # type: ignore[arg-type]
    assert hits == [] and record == {"pages": 1, "limit": 2, "rewrites": 0}
    assert (tmp_path / ".fi" / "page_check" / "paper.pdf").is_file()
    assert not (tmp_path / "paper.pdf").exists() and not (tmp_path / "paper_pdf_source.md").exists()


# --- the writer ------------------------------------------------------------------

def test_words_to_cut() -> None:
    assert _words_to_cut(5, 4, 0.95) == 100  # a few lines over
    assert _words_to_cut(5, 4, 0.0) == 450  # a full page over
    assert _words_to_cut(6, 4, 0.2) == 800


def test_the_page_limit_note() -> None:
    assert _page_limit_note(None, 3) == ""
    note = _page_limit_note(4, 3)
    assert "**Page limit: the rendered paper must fit in 4 pages.**" in note
    assert "about 1300 words" in note and "each of the 3 figures" in note
    assert "References and Further reading" in note and "`Study depth`" in note
    assert "about 1800 words" in _page_limit_note(4, 0)


def _write_prompt(eng: Engine, state: dict[str, Any]) -> str:
    seen: dict[str, str] = {}

    async def chat(prompt: str, *, node: str = "") -> str:
        seen[node] = prompt
        return "# Outbreaks\n\n## Abstract\nShort.\n"

    eng._chat = chat  # type: ignore[method-assign]
    asyncio.run(eng._node_write({"topic": eng.config.topic, **state}))  # type: ignore[arg-type]
    return seen["write"]


def test_without_a_limit_the_write_prompt_is_as_before(tmp_path: Path) -> None:
    prompt = _write_prompt(_engine(tmp_path, topic="Extinction times"), {"figures": ["a.png"]})
    assert "Page limit" not in prompt and "page_limit_note" not in prompt
    assert "default to **journal-length**.\n\n## Never narrate" in prompt


def test_with_a_limit_the_writer_gets_a_word_budget(tmp_path: Path) -> None:
    prompt = _write_prompt(_engine(tmp_path), {"figures": ["a.png", "b.png", "c.png"]})
    assert _page_limit_note(4, 3) in prompt
    assert "default to **journal-length**." + _page_limit_note(4, 3) + "\n\n## Never narrate" in prompt
    planned = _write_prompt(_engine(tmp_path), {"design": {"figures_planned": ["a.png", "b.png"]}})
    assert _page_limit_note(4, 2) in planned


def test_the_writer_is_told_the_pages_the_limit_and_the_words_to_cut() -> None:
    hit = _page_limit_hit(5, 4, 250)
    text = _format_review_for_writer({"review": {"verdict": "accept", "score": 5, "must_flag_hits": [hit]}})  # type: ignore[typeddict-item]
    assert text.splitlines() == [
        "Verdict: accept (score 5)",
        "Must fix:",
        "  - over_page_limit: the rendered PDF is 5 pages; the limit is 4. Cut about 250 words, "
        "keep every figure and all numbers; shorten background and discussion first",
    ]
