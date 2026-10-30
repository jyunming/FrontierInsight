"""The plan's model is asked what each figure shows when the plan names figures only by file name and FI writes the analysis
(``figures`` part of core/contract_parts.py): FI draws only the figures the contract declares, so a figure named only by file
name was drawn by nobody. The fake model here is a function, never a real one. Neutral made-up study: a pond's algae."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

import pytest

from core import analysis_frame as af
from core import contract, contract_parts as cp, receipts

TEMPLATES = {n: string.Template((Path(__file__).parent.parent / "agents" / f"{n}.md").read_text(encoding="utf-8"))
             for n in ("plan_roles", "plan_names", "plan_comparisons", "plan_formulas", "plan_figures")}

PLANNED = ["pond_peak.png (two panels, nutrient 1 and 4: peak algae against temp)",
           "pond_bloom.png (P(bloom) against nutrient for each temp)",
           "pond_gap.png (the difference in peak between the two temps, with its interval)"]

DESIGN: dict[str, Any] = {
    "hypothesis": "h", "method": "Simulate algae growth in a pond and count the peak and whether it blooms.",
    "figures_planned": PLANNED,
    "protocol": {
        "grid": {"nutrient": [1, 2, 4], "temp": [10, 20]},
        "runs_per_setting": 50,
        "metrics": [
            {"id": "algae_peak", "kind": "mean", "estimand": "peak algae count", "unit": "one pond run"},
            {"id": "bloom", "kind": "proportion", "estimand": "P(bloom)", "unit": "one pond run"},
        ],
    },
}
ROLES = {"results": {"algae_peak": {"role": "result", "depends_on": ["nutrient", "temp"], "returned": "each_run"},
                     "bloom": {"role": "result", "depends_on": ["nutrient"], "returned": "each_run"}}}
COMPARE = {"comparisons": [{"result": "algae_peak", "across": "temp", "values": [10, 20], "how": "difference"}]}
GOOD = {"figures": {
    "pond_peak.png": {"shows": ["algae_peak"], "x": "temp", "series": "nutrient", "panels": 2},
    "pond_bloom.png": {"shows": ["bloom"], "x": "nutrient", "series": "temp", "panels": 1},
    "pond_gap.png": {"shows": ["algae_peak", "bloom"], "x": "nutrient", "series": None, "panels": 1}}}
BAD = {"figures": {**GOOD["figures"], "pond_bloom.png": {"shows": ["salinity"], "x": "nutrient", "panels": 1}}}

MARKERS = (("roles", "What kind of number it is"), ("comparisons", "compares between values"),
           ("figures", "FI draws each figure itself"))


class Fake:
    """Answers each part from a queue of replies (a dict is sent as JSON, a string as it is)."""

    def __init__(self, **replies: list[Any]) -> None:
        self.replies, self.prompts = replies, []

    async def __call__(self, prompt: str) -> str:
        part = next(k for k, marker in MARKERS if marker in prompt)
        self.prompts.append((part, prompt))
        reply = self.replies[part].pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)

    def asked(self, part: str) -> int:
        return sum(1 for p, _ in self.prompts if p == part)


def parse(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


async def run(fake: Fake, design: dict[str, Any] = DESIGN, *, fi_analysis: bool = True) -> cp.Outcome:
    return await cp.ask_parts(design, fake, parse, TEMPLATES, fi_analysis=fi_analysis)


def _copy(**changes: Any) -> dict[str, Any]:
    return {**json.loads(json.dumps(DESIGN)), **changes}


@pytest.mark.asyncio
async def test_figures_named_only_by_file_are_asked_and_declared_and_the_gap_goes() -> None:
    before = contract.from_design(cp.strip_fi_keys(DESIGN), fi_analysis=True)
    assert [f.declared for f in before.figures] == [False] * 3
    assert any("only by file name" in g for g in before.gaps)
    fake = Fake(roles=[ROLES], comparisons=[COMPARE], figures=[GOOD])
    out = await run(fake)
    assert [p for p, _ in fake.prompts] == ["roles", "comparisons", "figures"] and out.form == "contract"
    c = contract.from_design(out.design, fi_analysis=True)
    assert [(f.id, f.file, f.shows, f.x, f.series, f.panels, f.declared) for f in c.figures] == [
        ("fig1", "pond_peak.png", ("algae_peak",), "temp", "nutrient", 2, True),
        ("fig2", "pond_bloom.png", ("bloom",), "nutrient", "temp", 1, True),
        ("fig3", "pond_gap.png", ("algae_peak", "bloom"), "nutrient", None, 1, True)]
    assert not any("only by file name" in g for g in c.gaps), c.gaps
    first = fake.prompts[2][1]  # the plan's own words, what may be shown, the settings and the panel limit
    assert "- pond_gap.png: (the difference in peak" in first and "- algae_peak:" in first and "algae_peak_by_temp" not in first
    assert "nutrient, temp" in first and f"at most {contract.MAX_PANELS} panel" in first


@pytest.mark.asyncio
async def test_the_declared_figures_reach_the_analysis_plan() -> None:
    out = await run(Fake(roles=[ROLES], comparisons=[COMPARE], figures=[GOOD]))
    plan = af.plan_of(out.design)
    assert {"figure_fig1", "figure_fig2", "figure_fig3"} <= set(af.body_names(plan))
    assert plan["figures"]["fig1"]["file"] == "pond_peak.png"
    older = await run(Fake(roles=[ROLES], comparisons=[COMPARE]), fi_analysis=False)
    assert af.body_names(af.plan_of(older.design)) == []  # what the plan did before: no figure body


@pytest.mark.asyncio
async def test_a_bad_answer_is_sent_back_and_the_second_good_one_kept() -> None:
    fake = Fake(roles=[ROLES], comparisons=[COMPARE], figures=[BAD, GOOD])
    out = await run(fake)
    assert fake.asked("figures") == 2 and out.parts["figures"]["valid"] and out.parts["figures"]["asks"] == 2
    assert "salinity" in fake.prompts[-1][1] and "Your last answer could not be used" in fake.prompts[-1][1]
    assert all(f.declared for f in contract.from_design(out.design, fi_analysis=True).figures)


@pytest.mark.asyncio
async def test_each_fault_of_an_answer_is_named() -> None:
    held = cp.strip_fi_keys(DESIGN)
    asked = cp.figure_faults(held)
    assert list(asked) == ["pond_peak.png", "pond_bloom.png", "pond_gap.png"]
    row = {"shows": ["algae_peak"], "x": "temp", "series": None, "panels": 1}
    problems = cp._check_figures({"pond_peak.png": {**row, "x": "depth", "panels": 9}, "pond_bloom.png": {"shows": []},
                                  "extra.png": row}, asked, held)
    text = " | ".join(problems)
    assert "`depth`" in text and "panels" in text and "`pond_bloom.png`: `shows`" in text
    assert "`pond_gap.png` is not answered" in text and "`extra.png` was not asked about" in text


@pytest.mark.asyncio
async def test_a_figure_that_cannot_be_drawn_may_be_dropped_and_is_said() -> None:
    answer = {"figures": {**GOOD["figures"], "pond_gap.png": {"drop": True, "why": "the interval is not reported"}}}
    out = await run(Fake(roles=[ROLES], comparisons=[COMPARE], figures=[answer]))
    c = contract.from_design(out.design, fi_analysis=True)
    assert [f.file for f in c.figures] == ["pond_peak.png", "pond_bloom.png"] and out.parts["figures"]["valid"]
    assert any("pond_gap.png" in n and "the interval is not reported" in n for n in out.notes)


@pytest.mark.asyncio
async def test_a_model_that_never_answers_usably_leaves_the_figures_out_alone_and_nothing_stops() -> None:
    fake = Fake(roles=[ROLES], comparisons=[COMPARE], figures=["no json at all", BAD, {"figures": {}}])
    out = await run(fake)
    assert fake.asked("figures") == cp.MAX_ASKS and not out.parts["figures"]["valid"]
    assert out.form == "contract" and "parts_left_out" not in out.design["protocol"]
    c = contract.from_design(out.design, fi_analysis=True)
    assert [f.declared for f in c.figures] == [False] * 3 and any("only by file name" in g for g in c.gaps)
    assert any("pond_peak.png" in n and "did not say" in n for n in out.notes)
    assert af.body_names(af.plan_of(out.design)) == []


@pytest.mark.asyncio
async def test_without_fis_analysis_no_figure_question_is_asked() -> None:
    fake = Fake(roles=[ROLES], comparisons=[COMPARE])
    out = await run(fake, fi_analysis=False)
    assert fake.asked("figures") == 0 and "figures" not in out.parts and out.form == "contract"
    assert "figures" not in out.design["protocol"]


@pytest.mark.asyncio
async def test_figures_already_declared_are_not_asked_about() -> None:
    declared = [{"file": "a.png", "shows": ["algae_peak"], "x": "temp"}, {"file": "b.png", "shows": ["bloom"], "x": "nutrient"}]
    design = _copy()
    design["protocol"]["figures"] = declared
    fake = Fake(roles=[ROLES], comparisons=[COMPARE])
    out = await run(fake, design)
    assert fake.asked("figures") == 0 and "figures" not in out.parts
    assert [f.declared for f in contract.from_design(out.design, fi_analysis=True).figures] == [True, True]
    none = Fake(roles=[ROLES], comparisons=[COMPARE])
    await run(none, _copy(figures_planned=[]))
    assert none.asked("figures") == 0


@pytest.mark.asyncio
async def test_a_plan_with_some_figures_declared_keeps_them_and_asks_only_about_the_rest() -> None:
    design = _copy()
    design["protocol"]["figures"] = [{"file": "a.png", "shows": ["algae_peak"], "x": "temp"}, "b.png (bloom against nutrient)"]
    answer = {"figures": {"b.png": {"shows": ["bloom"], "x": "nutrient", "series": None, "panels": 1}}}
    fake = Fake(roles=[ROLES], comparisons=[COMPARE], figures=[answer])
    out = await run(fake, design)
    assert "a.png" not in fake.prompts[-1][1].split("The figures, each")[1].split("What a figure may show")[0]
    c = contract.from_design(out.design, fi_analysis=True)
    assert [(f.id, f.file, f.declared) for f in c.figures] == [("fig1", "a.png", True), ("fig2", "b.png", True)]


@pytest.mark.asyncio
async def test_the_audits_verdict_covers_the_design_after_the_figures_are_answered() -> None:
    """The answer is FI's, not the study the audit judged: the design core is the plan's, with the figures as it wrote them,
    and a model's own mark for the answer is dropped."""
    out = await run(Fake(roles=[ROLES], comparisons=[COMPARE], figures=[GOOD]))
    assert "figure_declarations" in out.design["protocol"] and "figures" not in out.design["protocol"]
    assert out.design["figures_planned"] == PLANNED  # the plan's own list is never rewritten
    assert receipts.design_core(out.design) == receipts.design_core(DESIGN)
    both = _copy()
    both["protocol"]["figures"] = ["x.png (shows algae)"]
    again = await run(Fake(roles=[ROLES], comparisons=[COMPARE], figures=[{"figures": {"x.png": GOOD["figures"]["pond_peak.png"]}}]),
                      both)
    assert receipts.design_core(again.design) == receipts.design_core(both)
    assert "figure_declarations" not in cp.strip_fi_keys(out.design)["protocol"]
    assert again.design["protocol"]["figures"] == ["x.png (shows algae)"]


# --- the redesign, the plan file, the file names, the axes ---------------------------------------------------------------------
def _planned_protocol() -> dict[str, Any]:
    """The plan's protocol after the plan step answered the figures of DESIGN (as the plan file holds it)."""
    p = json.loads(json.dumps(DESIGN["protocol"]))
    for m in p["metrics"]:
        m.update(ROLES["results"][m["id"]])
    return {**p, "form": "contract", "figure_declarations": json.loads(json.dumps(GOOD["figures"]))}


def _redesign_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asked: list[str], answer: Any) -> Any:
    from core.config import ExecutionConfig
    from tests.test_contract_parts import _engine

    engine = _engine(tmp_path)
    monkeypatch.setattr("core.plan.load_design", lambda root: ({"hypothesis": "h", "protocol": _planned_protocol()}, None))
    engine.config.execution = ExecutionConfig(sandbox="venv", split_analysis=True, code_skeleton=True)
    engine._client = object()  # type: ignore[assignment]
    engine._prompts = TEMPLATES

    async def chat(prompt: str, **kw: Any) -> str:
        asked.append(prompt)
        return json.dumps(answer)

    monkeypatch.setattr(engine, "_chat", chat)
    return engine


def _redraft(**changes: Any) -> dict[str, Any]:
    """A redesign: its own grid, metrics (with roles) and figures, none of the plan's declarations."""
    protocol = {"grid": {"nutrient": [1, 2, 4], "temp": [10, 20]}, "runs_per_setting": 50,
                "metrics": [{**m, **ROLES["results"][m["id"]]} for m in DESIGN["protocol"]["metrics"]]}
    return {"hypothesis": "h", "method": "m", "protocol": protocol, **changes}


@pytest.mark.asyncio
async def test_a_redesign_that_keeps_the_figures_inherits_their_declarations_and_asks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []
    engine = _redesign_engine(tmp_path, monkeypatch, asked, {})
    state = {"design": {}, "iteration": 1}
    design = engine._inherit_plan_parts(state, _redraft(figures_planned=list(PLANNED)), iteration=1)
    assert set(design["protocol"]["figure_declarations"]) == {"pond_peak.png", "pond_bloom.png", "pond_gap.png"}
    design = await engine._ask_figures_of_a_redesign(state, design)
    assert asked == []
    c = contract.from_design(design, fi_analysis=True)
    assert [f.declared for f in c.figures] == [True] * 3 and not any("only by file name" in g for g in c.gaps)
    assert {"figure_fig1", "figure_fig2", "figure_fig3"} <= set(af.body_names(af.plan_of(design)))


@pytest.mark.asyncio
async def test_a_redesign_with_its_own_figures_has_the_new_ones_asked_and_declared_and_the_gone_ones_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []
    new = {"figures": {"pond_new.png": {"shows": ["bloom"], "x": "temp", "series": "nutrient", "panels": 1}}}
    engine = _redesign_engine(tmp_path, monkeypatch, asked, new)
    state = {"design": {}, "iteration": 1}
    own = ["pond_peak.png (peak against temp)", "pond_new.png (bloom against temp for each nutrient)"]
    design = engine._inherit_plan_parts(state, _redraft(figures_planned=own), iteration=1)
    design = await engine._ask_figures_of_a_redesign(state, design)
    assert len(asked) == 1 and "- pond_new.png:" in asked[0] and "- pond_peak.png:" not in asked[0]
    c = contract.from_design(design, fi_analysis=True)
    assert [(f.id, f.file, f.declared) for f in c.figures] == [("fig1", "pond_peak.png", True), ("fig2", "pond_new.png", True)]
    assert set(design["protocol"]["figure_declarations"]) == {"pond_peak.png", "pond_new.png"}  # the files no longer planned are gone
    assert design["figures_planned"] == own


@pytest.mark.asyncio
async def test_a_redesign_whose_old_declaration_no_longer_fits_has_that_figure_asked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []
    again = {"figures": {"pond_peak.png": {"drop": True, "why": "it needs the setting that is gone"},
                         "pond_bloom.png": {"shows": ["bloom"], "x": "nutrient", "series": None, "panels": 1}}}
    engine = _redesign_engine(tmp_path, monkeypatch, asked, again)
    state = {"design": {}, "iteration": 1}
    redraft = _redraft(figures_planned=list(PLANNED[:2]))
    redraft["protocol"]["grid"] = {"nutrient": [1, 2, 4], "light": [1, 2]}  # `temp` is gone: the figure that used it cannot stand
    design = engine._inherit_plan_parts(state, redraft, iteration=1)
    design = await engine._ask_figures_of_a_redesign(state, design)
    assert len(asked) == 1 and "- pond_peak.png:" in asked[0] and "- pond_bloom.png:" in asked[0]
    c = contract.from_design(design, fi_analysis=True)
    assert [f.file for f in c.figures] == ["pond_bloom.png"] and c.figures[0].declared and c.figures[0].x == "nutrient"


@pytest.mark.asyncio
async def test_a_redesign_is_not_asked_when_fi_does_not_write_the_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.config import ExecutionConfig

    asked: list[str] = []
    engine = _redesign_engine(tmp_path, monkeypatch, asked, {})
    state = {"design": {}, "iteration": 1}
    redraft = _redraft(figures_planned=["pond_new.png (x)"])
    redraft["protocol"]["form"] = "contract"
    engine.config.execution = ExecutionConfig(sandbox="venv", split_analysis=True, code_skeleton=False)
    assert await engine._ask_figures_of_a_redesign(state, redraft) == redraft and asked == []


def test_the_declarations_survive_the_plan_file_and_the_plans_own_figures_are_what_is_drawn() -> None:
    from core import plan

    protocol = _planned_protocol()
    text = plan.render("Algae", {}, {**DESIGN, "protocol": protocol, "plan": {}}, [])
    parsed = plan.parse(text).design
    assert parsed["protocol"]["figure_declarations"] == protocol["figure_declarations"]
    assert parsed["figures_planned"] == PLANNED
    assert [f.declared for f in contract.from_design(parsed, fi_analysis=True).figures] == [True] * 3
    # a person adds a figure and removes two in plan.md: the plan's list decides which are drawn
    edited = {**parsed, "figures_planned": [PLANNED[0], "pond_extra.png (anything)"]}
    c = contract.from_design(edited, fi_analysis=True)
    assert [(f.file, f.declared) for f in c.figures][0] == ("pond_peak.png", True)
    assert c.figures[1].declared is False and len(c.figures) == 2
    assert list(cp.figure_faults(edited)) == ["pond_extra.png"]


def test_a_planned_figure_has_one_file_name_everywhere_it_is_matched() -> None:
    from core import engine as eng

    assert contract.figure_file("pond_peak.png (two panels, nutrient 1 and 4)") == "pond_peak.png"
    assert contract.figure_file("figures/pond_peak.png (two panels)") == "pond_peak.png"
    assert contract.figure_file("figures\\pond_peak.png") == "pond_peak.png"
    assert contract.figure_file("my pond.png (a title with words)") == "my pond.png"
    assert contract.figure_file({"file": "out/a.png"}) == "a.png" and contract.figure_file("") == ""
    state = {"design": {"figures_planned": ["figures/pond_peak.png (two panels)", "pond_gap.png"]},
             "figures": ["figures/pond_peak.png"], "figure_records": {}}
    assert eng._planned_figure_order(state) == ["pond_peak.png"]
    declared = contract.from_design({"protocol": {**_planned_protocol()},
                                     "figures_planned": ["figures/pond_peak.png (two panels)"]}, fi_analysis=True)
    assert [f.file for f in declared.figures] == ["pond_peak.png"]  # saved under its base name, in the figures folder


def test_an_axis_must_be_another_setting_than_x_with_more_than_one_value() -> None:
    held = cp.strip_fi_keys(DESIGN)
    held["protocol"]["grid"]["light"] = [3]  # one value
    asked = {"a.png": "", "b.png": "", "c.png": ""}
    base = {"shows": ["algae_peak"], "panels": 1}
    problems = cp._check_figures({"a.png": {**base, "x": "temp", "series": "temp"}, "b.png": {**base, "x": "light"},
                                  "c.png": {**base, "series": "temp"}}, asked, held)
    text = " | ".join(problems)
    assert "`a.png`: `series` must be another setting than `x`" in text
    assert "`b.png`: `x` is `light`, which has one value" in text
    assert "`c.png`: `x` is the setting the figure is drawn along" in text
    assert cp._check_figures({n: {**base, "x": "temp", "series": "nutrient"} for n in asked}, asked, held) == []


@pytest.mark.asyncio
async def test_figures_are_not_asked_when_the_roles_could_not_be_used() -> None:
    fake = Fake(roles=["no json", "no json", "no json"], comparisons=[COMPARE])
    out = await run(fake)
    assert fake.asked("figures") == 0 and out.form == "older"
