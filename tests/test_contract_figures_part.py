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
    "pond_gap.png": {"shows": ["algae_peak_by_temp"], "x": "nutrient", "series": None, "panels": 1}}}
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
        ("fig3", "pond_gap.png", ("algae_peak_by_temp",), "nutrient", None, 1, True)]
    assert not any("only by file name" in g for g in c.gaps), c.gaps
    first = fake.prompts[2][1]  # the plan's own words, what may be shown, the settings and the panel limit
    assert "- pond_gap.png: (the difference in peak" in first and "- algae_peak_by_temp:" in first
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
    asked = cp._figure_faults(held)
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
    assert "figures" in out.design["protocol"] and "figures_asked" in out.design["protocol"]
    assert receipts.design_core(out.design) == receipts.design_core(DESIGN)
    both = _copy()
    both["protocol"]["figures"] = ["x.png (shows algae)"]
    again = await run(Fake(roles=[ROLES], comparisons=[COMPARE], figures=[{"figures": {"x.png": GOOD["figures"]["pond_peak.png"]}}]),
                      both)
    assert receipts.design_core(again.design) == receipts.design_core(both)
    assert "figures" not in cp.strip_fi_keys(out.design)["protocol"]
