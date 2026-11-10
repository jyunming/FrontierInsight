"""The plan's model is asked which result answers each main quantity of the topic, and at which value of a setting where the topic
asks for one (the ``answers`` part of core/contract_parts.py), and the paper's answers block is built from that map: one line
for each value of a setting the study compares, never one range across them; a quantity pinned to a value is answered at it; a
quantity nothing answers stays "not answered". The fake model is a function. Neutral made-up study: a pond's algae."""

from __future__ import annotations

import asyncio
import json
import string
from pathlib import Path
from typing import Any

from core import contract_parts as cp
from core import paper_facts as pf
from core import paper_values as pv
from core import topic_requirements as tr

TEMPLATES = {"plan_answers": string.Template((Path(__file__).parent.parent / "agents" / "plan_answers.md").read_text(encoding="utf-8"))}
PROTOCOL: dict[str, Any] = {
    "grid": {"nutrient": [1, 2], "temp": [10, 20]},
    "runs_per_setting": 5,
    "metrics": [{"id": "algae_peak", "kind": "mean", "estimand": "peak algae count", "unit": "one pond run"},
                {"id": "bloom", "kind": "proportion", "estimand": "P(bloom)", "unit": "one pond run"}],
    "comparisons": [{"id": "peak_by_temp", "result": "algae_peak", "across": "temp", "values": [10, 20], "how": "difference"}],
}
DESIGN = {"hypothesis": "h", "method": "Simulate algae in a pond.", "protocol": PROTOCOL}
REQ = {"quantities": [
    {"name": "peak algae", "stated": "report the peak", "main": True},
    {"name": "chance of a bloom (at nutrient 1)", "stated": "the bloom chance at the lowest nutrient", "main": True},
    {"name": "water clarity", "stated": "report the clarity", "main": True},
    {"name": "a side quantity", "stated": "x", "main": False},
]}


def _ask(design: dict[str, Any], replies: list[Any], asked: dict[str, dict[str, str]]):
    seen: list[str] = []

    async def ask(prompt: str) -> str:
        seen.append(prompt)
        reply = replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)

    def parse(text: str) -> Any:
        try:
            return json.loads(text)
        except ValueError:
            return None

    out = asyncio.run(cp.ask_parts(design, ask, parse, TEMPLATES, only=("answers",), topic_quantities=asked))
    return out, seen


def test_only_main_quantities_without_a_placed_map_are_asked() -> None:
    asked = tr.quantities_to_place(REQ, PROTOCOL)
    assert list(asked) == ["Q1", "Q2", "Q3"] and asked["Q1"]["by"] == ""
    placed = {**PROTOCOL, "topic_map": [{"item": "Q1", "by": "metric:algae_peak", "at": {}}]}
    got = tr.quantities_to_place(REQ, placed)
    assert list(got) == ["Q2", "Q3"]
    unplaced = {**PROTOCOL, "topic_map": [{"item": "Q1", "by": "metric:algae_peak"}]}
    assert tr.quantities_to_place(REQ, unplaced)["Q1"]["by"] == "metric:algae_peak"  # the plan's own map is kept, `at` is asked


def test_a_valid_answer_is_written_into_the_map_and_an_unmappable_quantity_adds_nothing() -> None:
    asked = tr.quantities_to_place(REQ, PROTOCOL)
    reply = {"answers": {"Q1": {"by": "metric:algae_peak"}, "Q2": {"by": "metric:bloom", "at": {"nutrient": 1}},
                         "Q3": {"by": "none"}}}
    out, seen = _ask(DESIGN, [reply], asked)
    assert out.parts["answers"]["valid"] and len(seen) == 1
    rows = out.design["protocol"]["topic_map"]
    assert {"item": "Q2", "by": "metric:bloom", "at": {"nutrient": 1}} in rows
    assert {"item": "Q1", "by": "metric:algae_peak", "at": {}} in rows
    assert not [r for r in rows if r["item"] == "Q3"]  # nothing answers it: it stays unmapped


def test_an_answer_naming_what_the_study_does_not_have_is_asked_again_with_the_reason() -> None:
    asked = tr.quantities_to_place(REQ, PROTOCOL)
    bad = {"answers": {"Q1": {"by": "metric:salinity"}, "Q2": {"by": "metric:bloom", "at": {"nutrient": 7}},
                       "Q3": {"by": "none", "at": {"depth": 1}}}}
    good = {"answers": {"Q1": {"by": "metric:algae_peak"}, "Q2": {"by": "metric:bloom"}, "Q3": {"by": "none"}}}
    out, seen = _ask(DESIGN, [bad, good], asked)
    assert out.parts["answers"]["valid"] and out.parts["answers"]["asks"] == 2
    assert "metric:salinity" in seen[1] or "`by` must be" in seen[1]
    assert "no value 7" in seen[1] and "not a setting of the study" in seen[1]
    # a plan that never gives a usable answer is left as it was
    out, _seen = _ask(DESIGN, [bad] * cp.MAX_ASKS, asked)
    assert not out.parts["answers"]["valid"] and "topic_map" not in out.design["protocol"]


def test_an_answer_in_a_different_physical_unit_is_refused_even_when_the_model_calls_it_the_right_unit() -> None:
    """The plan's declared physical unit, not the answer's self-described ``by_unit``, decides what a result means."""
    protocol = {**PROTOCOL, "metrics": [
        {"id": "length", "kind": "mean", "estimand": "pond length", "unit": "one pond run", "physical_unit": "m"},
        {"id": "relative_error", "kind": "mean", "estimand": "relative length error", "unit": "one pond run", "physical_unit": "1"},
    ]}
    design = {**DESIGN, "protocol": protocol}
    asked = {"Q1": {"name": "pond length", "stated": "report pond length in metres", "by": ""}}
    bad = {"answers": {"Q1": {"unit": "m", "by": "metric:relative_error", "by_unit": "m"}}}
    good = {"answers": {"Q1": {"unit": "m", "by": "metric:length", "by_unit": "m"}}}
    out, seen = _ask(design, [bad, good], asked)
    assert out.parts["answers"]["valid"] and out.parts["answers"]["asks"] == 2
    assert "metric:relative_error" in seen[1] and "question asks for a quantity in `m`" in seen[1]
    assert "- length: pond length [m]" in seen[0]
    assert {"item": "Q1", "by": "metric:length", "at": {}} in out.design["protocol"]["topic_map"]


KNOWN = {**PROTOCOL, "constants": {"peak_true": 30.0},
         "metrics": [{"id": "algae_peak", "kind": "mean", "estimand": "peak algae count", "unit": "one pond run",
                      "known_value": {"true_value": "peak_true", "estimate": "algae_peak", "standard_error": "peak_se", "level": 0.95}},
                     PROTOCOL["metrics"][1]]}


def test_a_typed_statistic_is_checked_and_written_into_the_map() -> None:
    design = {**DESIGN, "protocol": KNOWN}
    asked = tr.quantities_to_place(REQ, KNOWN)
    bad = {"answers": {"Q1": {"by": "metric:algae_peak", "statistic": "mean_error"}, "Q2": {"by": "metric:bloom", "statistic": "bias"},
                       "Q3": {"by": "none"}}}
    good = {"answers": {"Q1": {"by": "metric:algae_peak", "statistic": "bias"}, "Q2": {"by": "metric:bloom", "statistic": "sd"},
                        "Q3": {"by": "none"}}}
    out, seen = _ask(design, [bad, good], asked)
    assert out.parts["answers"]["valid"] and out.parts["answers"]["asks"] == 2
    assert "`statistic` must be one of value, bias, sd" in seen[1] and "`bias` needs a result with a known true value" in seen[1]
    rows = out.design["protocol"]["topic_map"]
    assert {"item": "Q1", "by": "metric:algae_peak", "at": {}, "statistic": "bias"} in rows
    assert {"item": "Q2", "by": "metric:bloom", "at": {}, "statistic": "sd"} in rows


def test_the_prompt_lists_the_statistics_and_an_answer_without_one_is_the_value() -> None:
    asked = tr.quantities_to_place(REQ, PROTOCOL)
    out, seen = _ask(DESIGN, [{"answers": {"Q1": {"by": "metric:algae_peak"}, "Q2": {"by": "metric:bloom"}, "Q3": {"by": "none"}}}], asked)
    assert "value, bias, sd" in seen[0]
    assert all("statistic" not in r for r in out.design["protocol"]["topic_map"])



def test_an_answer_that_is_the_value_clears_an_earlier_statistic_on_the_entry() -> None:
    p = {**PROTOCOL, "topic_map": [{"item": "peak algae", "by": "metric:algae_peak", "statistic": "bias"}]}
    asked = {"Q1": {"name": "peak algae", "stated": "report the peak"}}
    rows = cp._apply_answers(p, {**DESIGN, "protocol": p}, {"Q1": {"by": "metric:algae_peak", "statistic": "value"}}, asked)["topic_map"]
    assert rows == [{"item": "peak algae", "by": "metric:algae_peak", "at": {}}]
    rows = cp._apply_answers(p, {**DESIGN, "protocol": p}, {"Q1": {"by": "metric:algae_peak", "statistic": "sd"}}, asked)["topic_map"]
    assert rows[0]["statistic"] == "sd"

def _quest(tmp_path: Path, monkeypatch, topic_map: list[dict[str, Any]]) -> Path:
    (tmp_path / "raw").mkdir(parents=True)
    cells = []
    for n in (1, 2):
        for t in (10, 20):
            vals = [n * 10.0 + t + k for k in range(5)]
            cells.append({"cell": {"nutrient": n, "temp": t}, "key": f"nutrient={n},temp={t}", "planned": 5, "ok": 5, "failed": 0,
                          "metrics": {"algae_peak": {"values": vals, "trials": list(range(5)), "count": 5, "total": sum(vals),
                                                     "non_finite": 0},
                                      "bloom": {"values": [0.0, 1.0, 0.0, 1.0, 1.0], "trials": list(range(5)), "count": 5,
                                                "total": 3.0, "non_finite": 0}}})
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / ".fi").mkdir()
    (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps(REQ), encoding="utf-8")
    protocol = {**PROTOCOL, "topic_map": topic_map}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: protocol)
    return tmp_path


def test_a_compared_setting_is_answered_for_each_value_and_an_at_pins_the_quantity(tmp_path, monkeypatch) -> None:
    root = _quest(tmp_path, monkeypatch, [{"item": "Q1", "by": "metric:algae_peak", "at": {}},
                                          {"item": "Q2", "by": "metric:bloom", "at": {"nutrient": 1}}])
    text = pf.answers_text(root)
    peak = [ln for ln in text.splitlines() if ln.startswith("- peak algae")]
    assert len(peak) == 1 and "(measured as: `algae_peak`), by temp: 10: {{v:algae_peak @ nutrient=each, temp=10 | range}}; 20: {{v:algae_peak @ nutrient=each, temp=20 | range}}" in peak[0]
    assert "{{v:algae_peak @ nutrient=each, temp=each | range}}" not in text  # no one range across the compared values
    assert "(when nutrient = 1)" in text and "{{v:bloom:mean_ci @ nutrient=1, temp=10}}" in text
    assert "- water clarity: the plan links no result of the study to it here" in text  # the known positive
    out = pv.fill_values(text, pv.records_of(root, pf.plan_protocol(root)))[0]
    assert "not recorded" not in out


def test_a_map_without_at_or_comparisons_is_read_as_before(tmp_path, monkeypatch) -> None:
    root = _quest(tmp_path, monkeypatch, [{"item": "Q1", "by": "metric:algae_peak"}])
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: {**PROTOCOL, "comparisons": [],
                                                              "topic_map": [{"item": "Q1", "by": "metric:algae_peak"}]})
    assert "- peak algae (measured as: `algae_peak`): {{v:algae_peak @ nutrient=each, temp=each | range}}" in pf.answers_text(root)


def test_a_study_run_through_the_deterministic_entry_says_no_seed_was_passed_otherwise_the_recorded_base_seed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pf._run_facts, "seed_verdict", lambda root: {"verdict": pf._run_facts.UNKNOWN})
    monkeypatch.setattr(pf._disclosure, "recorded_seeds", lambda root: (0, frozenset()))
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: {"seed_policy": "A constant base_seed of 99 is used."})
    (tmp_path / "code").mkdir()
    study = tmp_path / "code" / "study.json"
    study.write_text(json.dumps({"entry": "run_trial", "base_seed": 0}), encoding="utf-8")
    assert pf._seed_text(tmp_path) == ("a seed passed to each trial, derived from base seed 0 "
                                       "(FI could not tell from the code whether the simulation uses it)")  # the plan's prose is not quoted
    study.write_text(json.dumps({"entry": "run_cell", "base_seed": 0}), encoding="utf-8")
    assert pf._seed_text(tmp_path) == "none (each setting ran once with no random numbers; FI passed no seed)"


DERIVED = [{"id": "peak_ratio", "inputs": ["algae_peak"], "role": "result", "label": "ratio of the peak at the high to the low temperature",
            "asked": True}]


def test_a_comparison_of_a_worked_out_quantity_does_not_split_its_setting_into_a_line_for_each_value(tmp_path, monkeypatch) -> None:
    """Before: a comparison whose result is a quantity worked out from the results (here across nutrient) made every value of
    that setting an alternative, one near-identical line each. Now only a comparison of a reported result splits."""
    root = _quest(tmp_path, monkeypatch, [{"item": "Q1", "by": "metric:algae_peak", "at": {}}])
    worked = {**PROTOCOL, "derived": DERIVED, "topic_map": [{"item": "Q1", "by": "metric:algae_peak", "at": {}}],
              "comparisons": [{"id": "r_by_nutrient", "result": "peak_ratio", "across": "nutrient", "values": [1, 2], "how": "ratio"}]}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: worked)
    text = pf.answers_text(root)
    assert "- peak algae (measured as: `algae_peak`): {{v:algae_peak @ nutrient=each, temp=each | range}}" in text
    assert "nutrient = " not in text.split("water clarity")[0]


def test_a_quantity_worked_out_from_the_mapped_result_is_disclosed_and_never_replaces_the_map(tmp_path, monkeypatch) -> None:
    protocol = {**PROTOCOL, "derived": DERIVED, "topic_map": [{"item": "Q1", "by": "metric:algae_peak"}]}
    design = {**DESIGN, "protocol": protocol}
    asked = tr.quantities_to_place(REQ, protocol)
    kept = cp._apply_answers(protocol, design, {"Q1": {"by": "metric:peak_ratio"}, "Q2": {"by": "none"}, "Q3": {"by": "none"}}, asked)
    assert {"item": "Q1", "by": "metric:algae_peak", "at": {}, "other_candidate": "metric:peak_ratio"} in kept["topic_map"]
    # an answer that is not worked out from the mapped result leaves the plan's own map, with nothing to disclose
    other = cp._apply_answers(protocol, design, {"Q1": {"by": "metric:bloom"}, "Q2": {"by": "none"}, "Q3": {"by": "none"}}, asked)
    assert {"item": "Q1", "by": "metric:algae_peak", "at": {}} in other["topic_map"]
    # the paper still answers by the plan's result; the model's proposal is only told to the reader of the quality report
    root = _quest(tmp_path, monkeypatch, kept["topic_map"])
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: kept)
    text = pf.answers_text(root)
    assert "- peak algae (measured as: `algae_peak`)" in text and "{{v:algae_peak" in text and "peak_ratio" not in text
    assert "peak_ratio" in cp._answers_prompt(TEMPLATES, design, asked, [], "")  # the model is shown what it is worked out from
    assert pf.other_candidates(root) == [{"name": "peak algae", "planned": "metric:algae_peak", "candidate": "metric:peak_ratio"}]


def test_a_variance_worked_out_from_a_mean_is_not_printed_as_the_mean() -> None:
    """Planted: the question asks for a mean; the plan maps the mean; the second model call names a variance computed from it.
    The old code replaced the plan's map, so the paper printed the variance as the mean."""
    derived = [{"id": "peak_variance", "inputs": ["algae_peak"], "role": "result", "label": "variance of the peak", "asked": True}]
    protocol = {**PROTOCOL, "derived": derived, "topic_map": [{"item": "Q1", "by": "metric:algae_peak"}]}
    design = {**DESIGN, "protocol": protocol}
    asked = {"Q1": {"name": "mean peak", "stated": "report the mean peak", "by": "metric:algae_peak"}}
    row = cp._apply_answers(protocol, design, {"Q1": {"by": "metric:peak_variance"}}, asked)["topic_map"][0]
    assert row["by"] == "metric:algae_peak" and row["other_candidate"] == "metric:peak_variance"
    again = cp._apply_answers({**protocol, "topic_map": [row]}, design, {"Q1": {"by": "metric:algae_peak"}}, asked)["topic_map"][0]
    assert "other_candidate" not in again  # the entry is rewritten by each answer: an earlier candidate does not stay


def test_a_condition_the_question_already_states_is_not_added_again(tmp_path, monkeypatch) -> None:
    """Before: the question 'chance of a bloom (when nutrient = 1)' with the plan pinning nutrient to 1.0 printed
    '(when nutrient = 1) (when nutrient = 1.0)'. 1 and 1.0 are one value, said once; a pin the words do not state is still added."""
    monkeypatch.setitem(REQ["quantities"][1], "name", "chance of a bloom (when nutrient = 1)")
    root = _quest(tmp_path, monkeypatch, [{"item": "Q1", "by": "metric:algae_peak", "at": {}},
                                          {"item": "Q2", "by": "metric:bloom", "at": {"nutrient": 1.0}}])
    line = next(ln for ln in pf.answers_text(root).splitlines() if ln.startswith("- chance of a bloom"))
    assert line.count("nutrient = 1") == 1 and "nutrient = 1.0" not in line
    monkeypatch.setitem(REQ["quantities"][1], "name", "chance of a bloom (when nutrient = 2)")
    (root / ".fi" / "topic_requirements.json").write_text(json.dumps(REQ), encoding="utf-8")
    line = next(ln for ln in pf.answers_text(root).splitlines() if ln.startswith("- chance of a bloom"))
    assert "nutrient = 2)" in line and "nutrient = 1.0" in line  # a different stated value is not the pin: both are shown


def test_a_stated_condition_is_matched_as_a_whole_name_and_a_whole_number() -> None:
    """Before: 'beta = 2 error' counted as stating 'a = 2' (a name ending in the setting's name), 'dt = 1-3' as 'dt = 1',
    so the pin was dropped and the line lost which setting the number belongs to."""
    class _Rec:
        label = short = staticmethod(lambda a: a)

    pins = lambda name, at: pf._pins_not_in(name, _Rec(), at)  # noqa: E731
    assert pins("beta = 2 error", {"a": 2}) == ["a = 2"]
    assert pins("run = 5 error", {"n": 5}) == ["n = 5"]
    assert pins("error at dt = 1-3", {"dt": 1}) == ["dt = 1"]
    assert pins("error at dt = 1/2", {"dt": 1}) == ["dt = 1"]
    assert pins("error at a = 20", {"a": 2}) == ["a = 2"]
    assert pins("error (when a = 2)", {"a": 2}) == []  # a condition really stated is still said once
    assert pins("error when a = 2.", {"a": 2}) == []
