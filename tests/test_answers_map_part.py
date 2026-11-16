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


# Two different questions the map answers with one number (the same result at the same setting with the same statistic)
REQ_TWO = {"quantities": [
    {"name": "nutrient at which algae pass 100", "stated": "the nutrient where the count passes 100 and 500", "main": True},
    {"name": "nutrient at which algae pass 500", "stated": "the nutrient where the count passes 100 and 500", "main": True},
    {"name": "chance of a bloom", "stated": "report the bloom chance", "main": True},
]}
PLAN_TWO = {**PROTOCOL, "topic_map": [{"item": "Q1", "by": "metric:algae_peak", "at": {}},
                                      {"item": "Q2", "by": "metric:algae_peak", "at": {}},
                                      {"item": "Q3", "by": "metric:bloom", "at": {}}]}


def test_quantities_the_plan_gives_one_number_are_asked_again_and_the_others_are_not() -> None:
    asked = tr.quantities_to_place(REQ_TWO, PLAN_TWO)
    assert list(asked) == ["Q1", "Q2"]
    assert asked["Q1"]["shared"] == ["Q2"] and asked["Q2"]["shared"] == ["Q1"] and asked["Q1"]["by"] == "metric:algae_peak"
    # the same result read at two different settings is two numbers: nothing is asked
    apart = {**PROTOCOL, "topic_map": [{"item": "Q1", "by": "metric:algae_peak", "at": {"temp": 10}},
                                       {"item": "Q2", "by": "metric:algae_peak", "at": {"temp": 20}},
                                       {"item": "Q3", "by": "metric:bloom", "at": {}}]}
    assert tr.quantities_to_place(REQ_TWO, apart) == {}
    # one setting value written two ways is one number: asked again
    written = {**PROTOCOL, "topic_map": [{"item": "Q1", "by": "metric:algae_peak", "at": {"temp": 10}},
                                         {"item": "Q2", "by": "metric:algae_peak", "at": {"temp": 10.0}},
                                         {"item": "Q3", "by": "metric:bloom", "at": {}}]}
    assert list(tr.quantities_to_place(REQ_TWO, written)) == ["Q1", "Q2"]


def test_an_answer_that_still_gives_two_questions_one_number_is_asked_again_and_a_none_leaves_one_unanswered() -> None:
    asked = tr.quantities_to_place(REQ_TWO, PLAN_TWO)
    same = {"answers": {"Q1": {"by": "metric:algae_peak"}, "Q2": {"by": "metric:algae_peak"}}}
    fixed = {"answers": {"Q1": {"by": "metric:algae_peak"}, "Q2": {"by": "none"}}}
    out, seen = _ask({**DESIGN, "protocol": PLAN_TWO}, [same, fixed], asked)
    assert len(seen) == 2 and "the same number as Q2" in seen[0]
    assert "would all be given the same number" in seen[1]
    rows = out.design["protocol"]["topic_map"]
    assert [r["item"] for r in rows if r["by"] == "metric:algae_peak"] == ["Q1"]
    assert not [r for r in rows if r["item"] == "Q2"]  # unanswered in the paper, never the other question's number


def test_a_shared_quantity_may_be_moved_to_its_own_result() -> None:
    asked = tr.quantities_to_place(REQ_TWO, PLAN_TWO)
    reply = {"answers": {"Q1": {"by": "metric:algae_peak"}, "Q2": {"by": "metric:bloom"}}}
    out, _seen = _ask({**DESIGN, "protocol": PLAN_TWO}, [reply], asked)
    rows = {r["item"]: r["by"] for r in out.design["protocol"]["topic_map"]}
    assert rows["Q1"] == "metric:algae_peak" and rows["Q2"] == "metric:bloom"


def test_quantities_still_given_one_number_at_the_last_ask_are_kept_and_recorded() -> None:
    asked = tr.quantities_to_place(REQ_TWO, PLAN_TWO)
    same = {"answers": {"Q1": {"by": "metric:algae_peak"}, "Q2": {"by": "metric:algae_peak"}}}
    out, seen = _ask({**DESIGN, "protocol": PLAN_TWO}, [same] * cp.MAX_ASKS, asked)
    assert len(seen) == cp.MAX_ASKS and out.parts["answers"]["valid"]
    assert out.parts["answers"]["same_number"] == ["Q1", "Q2"]  # the paper gives their number once, saying so
    rows = out.design["protocol"]["topic_map"]
    assert {r["item"] for r in rows if r["by"] == "metric:algae_peak"} == {"Q1", "Q2"}
    assert {"item": "Q3", "by": "metric:bloom", "at": {}} in rows


def test_one_question_one_number_asks_nothing_more() -> None:
    plain = {**PROTOCOL, "topic_map": [{"item": "Q1", "by": "metric:algae_peak", "at": {}},
                                       {"item": "Q2", "by": "metric:bloom", "at": {}}]}
    assert tr.shared_answers(REQ_TWO, plain) == {}


def test_the_paper_never_gives_two_questions_one_number_whatever_map_it_reads(tmp_path, monkeypatch) -> None:
    # A frozen or resumed plan whose map still answers two questions with one number (the answers step never asked, or failed)
    root = _quest(tmp_path, monkeypatch, [{"item": "Q1", "by": "metric:algae_peak", "at": {}},
                                          {"item": "Q3", "by": "metric:algae_peak", "at": {}}])
    text = pf.answers_text(root)
    assert "- peak algae; water clarity (" in text and "(the plan gives these different questions one number, so it can be the answer to only one of them)" in text
    assert not [ln for ln in text.splitlines() if ln.startswith("- water clarity")]
    # the same result read at different settings is two numbers: both answered (the known negative)
    root2 = tmp_path / "two"
    root2.mkdir()
    root2 = _quest(root2, monkeypatch, [{"item": "Q1", "by": "metric:algae_peak", "at": {"nutrient": 1}},
                                        {"item": "Q3", "by": "metric:algae_peak", "at": {"nutrient": 2}}])
    assert "(the plan gives these different questions one number, so it can be the answer to only one of them)" not in pf.answers_text(root2)


def test_a_map_with_several_entries_still_never_gives_two_questions_one_number(tmp_path, monkeypatch) -> None:
    # Q1 has an entry for a result the study lacks and one for algae_peak; the paper reads algae_peak for it, as for Q3
    root = _quest(tmp_path, monkeypatch, [{"item": "Q1", "by": "metric:not_a_result", "at": {}},
                                          {"item": "Q1", "by": "metric:algae_peak", "at": {}},
                                          {"item": "Q3", "by": "metric:algae_peak", "at": {}}])
    text = pf.answers_text(root)
    assert "- peak algae; water clarity (" in text and "(the plan gives these different questions one number, so it can be the answer to only one of them)" in text
    assert not [ln for ln in text.splitlines() if ln.startswith("- water clarity")]


def test_one_number_the_analysis_worked_out_is_not_given_to_two_questions_at_different_settings(tmp_path, monkeypatch) -> None:
    # an analysis answer is one number whatever setting the map pins: two settings in the map still print one number
    root = _quest(tmp_path, monkeypatch, [{"item": "Q1", "by": "metric:clarity", "at": {"nutrient": 1}},
                                          {"item": "Q3", "by": "metric:clarity", "at": {"nutrient": 2}}])
    protocol = {**pf.plan_protocol(root), "metrics": [*PROTOCOL["metrics"], {"id": "clarity", "kind": "mean", "estimand": "clarity"}]}
    monkeypatch.setattr(pf, "plan_protocol", lambda *a, **k: protocol)
    (root / pv.ANSWERS).write_text(json.dumps({"scheduled": 1, "runs": [{"run": "0", "values": {"clarity": 0.5}}]}), encoding="utf-8")
    text = pf.answers_text(root)
    assert "- peak algae; water clarity (" in text and "(the plan gives these different questions one number, so it can be the answer to only one of them)" in text
    assert not [ln for ln in text.splitlines() if ln.startswith("- water clarity")]
    assert text.count("{{v:clarity}}") == 1
    # the known negative: one question on that number is answered with it
    protocol["topic_map"] = [{"item": "Q1", "by": "metric:clarity", "at": {"nutrient": 1}}]
    assert "- peak algae (measured as: `clarity`): {{v:clarity}}" in pf.answers_text(root)


def test_one_place_written_two_ways_is_still_one_number(tmp_path, monkeypatch) -> None:
    # the settings in another order, and 10 written as 10.0, read the same trials
    root = _quest(tmp_path, monkeypatch, [{"item": "Q1", "by": "metric:algae_peak", "at": {"nutrient": 1, "temp": 10}},
                                          {"item": "Q3", "by": "metric:algae_peak", "at": {"temp": 10.0, "nutrient": 1}}])
    text = pf.answers_text(root)
    assert "- peak algae; water clarity (" in text and "(the plan gives these different questions one number, so it can be the answer to only one of them)" in text
    assert not [ln for ln in text.splitlines() if ln.startswith("- water clarity")]


def test_a_setting_the_record_does_not_have_does_not_make_two_numbers(tmp_path, monkeypatch) -> None:
    # the map pins a setting the study never varied: the line reads the result without it, so both print one number
    root = _quest(tmp_path, monkeypatch, [{"item": "Q1", "by": "metric:algae_peak", "at": {"salinity": 3}},
                                          {"item": "Q3", "by": "metric:algae_peak", "at": {"salinity": 4}}])
    text = pf.answers_text(root)
    assert "- peak algae; water clarity (" in text and "(the plan gives these different questions one number, so it can be the answer to only one of them)" in text
    assert not [ln for ln in text.splitlines() if ln.startswith("- water clarity")]


def test_a_question_at_one_setting_of_a_result_another_gives_for_each_setting_keeps_its_answer(tmp_path, monkeypatch) -> None:
    # Q1 is answered for each temp, Q3 at temp 10 only: one of Q1's numbers, not all of them; both keep their answers
    root = _quest(tmp_path, monkeypatch, [{"item": "Q1", "by": "metric:algae_peak", "at": {}},
                                          {"item": "Q3", "by": "metric:algae_peak", "at": {"temp": 10}}])
    text = pf.answers_text(root)
    assert "(the plan gives these different questions one number, so it can be the answer to only one of them)" not in text
    assert "20: {{v:algae_peak @ nutrient=each, temp=20 | range}}" in text and "- water clarity" in text
    assert "{{v:algae_peak @ nutrient=each, temp=10 | range}}" in [ln for ln in text.splitlines() if ln.startswith("- water clarity")][0]


def test_a_shared_question_moved_to_a_result_with_a_true_value_may_ask_for_its_bias() -> None:
    plan = {**PROTOCOL, "constants": {"peak_true": 30.0},
            "metrics": [*PROTOCOL["metrics"], {"id": "peak_hat", "kind": "mean", "known_value": {
                "true_value": "peak_true", "estimate": "peak_hat", "standard_error": "peak_se", "level": 0.95}}],
            "topic_map": [{"item": "Q1", "by": "metric:bloom", "at": {}}, {"item": "Q2", "by": "metric:bloom", "at": {}},
                          {"item": "Q3", "by": "metric:algae_peak", "at": {}}]}
    asked = tr.quantities_to_place(REQ_TWO, plan)
    reply = {"Q1": {"by": "metric:bloom"}, "Q2": {"by": "metric:peak_hat", "statistic": "bias"}}
    assert not [p for p in cp._check_answers(reply, asked, {**DESIGN, "protocol": plan}) if "bias" in p]
    # the known positive: the plan's shared result has no true value, so a bias kept there is refused
    kept = {"Q1": {"by": "metric:bloom"}, "Q2": {"by": "metric:bloom", "statistic": "bias"}}
    assert [p for p in cp._check_answers(kept, asked, {**DESIGN, "protocol": plan}) if "bias" in p]


def test_a_shared_question_moved_to_another_result_is_held_to_that_result() -> None:
    # moved to a result with no known true value, a bias is refused (the plan's result is not what is checked)
    asked = tr.quantities_to_place(REQ_TWO, PLAN_TWO)
    reply = {"answers": {"Q1": {"by": "metric:algae_peak"}, "Q2": {"by": "metric:bloom", "statistic": "bias"}}}
    problems = cp._check_answers(reply["answers"], asked, {**DESIGN, "protocol": PLAN_TWO})
    assert any("`Q2`: `bias` needs a result with a known true value" in p and "`bloom` has none" in p for p in problems)


SPREAD = {**PROTOCOL, "derived": [{"id": "peak_spread", "inputs": ["algae_peak"], "role": "result",
                                   "label": "relative spread of the peak over the runs"}]}


def test_the_spread_of_a_result_worked_out_once_per_setting_is_refused_and_its_value_is_kept() -> None:
    # A derived result is one number per setting: an `sd` of it has no trials to spread over, and the paper then said the
    # quantity was "not answered" because no setting had two trials, though every setting had many.
    design = {**DESIGN, "protocol": SPREAD}
    asked = {"Q1": {"name": "spread of the estimates", "stated": "report the spread", "by": "metric:peak_spread"}}
    bad = {"answers": {"Q1": {"by": "metric:peak_spread", "statistic": "sd"}}}
    problems = cp._check_answers(bad["answers"], asked, design)
    assert any("`Q1`: `sd` is the spread over the trials" in p and "`peak_spread` is worked out once per setting from "
               "`algae_peak`" in p for p in problems)
    good = {"answers": {"Q1": {"by": "metric:peak_spread", "statistic": "value"}}}
    out, seen = _ask(design, [bad, good], asked)
    assert out.parts["answers"]["valid"] and out.parts["answers"]["asks"] == 2
    assert "worked out once per setting" in seen[1]
    assert not [r for r in out.design["protocol"]["topic_map"] if r.get("statistic") == "sd"]


def test_the_spread_of_a_result_each_trial_returns_is_still_accepted() -> None:
    design = {**DESIGN, "protocol": SPREAD}
    asked = {"Q1": {"name": "spread of the peak", "stated": "report the spread", "by": ""}}
    reply = {"answers": {"Q1": {"by": "metric:algae_peak", "statistic": "sd"}}}
    assert cp._check_answers(reply["answers"], asked, design) == []


def test_an_sd_given_for_another_result_is_not_put_on_the_plans_result_worked_out_once_per_setting() -> None:
    # Recorded on a real plan: the plan maps the spread to a derived relative spread; the answer names the estimate with
    # `sd` (the right pair). FI keeps the plan's map, so the `sd` went onto the derived result, which has no spread over
    # trials, and the paper said the spread was not answered. The plan's result now answers with its value.
    design = {**DESIGN, "protocol": SPREAD}
    asked = {"Q1": {"name": "spread of the estimates", "stated": "report the spread", "by": "metric:peak_spread"}}
    reply = {"answers": {"Q1": {"by": "metric:algae_peak", "statistic": "sd"}}}
    assert cp._check_answers(reply["answers"], asked, design) == []
    out, seen = _ask({**design, "protocol": {**SPREAD, "topic_map": [{"item": "Q1", "by": "metric:peak_spread"}]}},
                     [reply], asked)
    assert out.parts["answers"]["valid"] and len(seen) == 1
    row = next(r for r in out.design["protocol"]["topic_map"] if r["item"] == "Q1")
    assert row["by"] == "metric:peak_spread" and "statistic" not in row


def test_a_plan_map_naming_its_result_worked_out_once_per_setting_without_a_prefix_gets_no_sd_either() -> None:
    # A plan's own map may name a result without `metric:` (FI reads `x` and `metric:x` alike); the bare name hid that the
    # kept result is worked out once per setting, and the answer's `sd` went onto it.
    design = {**DESIGN, "protocol": SPREAD}
    asked = {"Q1": {"name": "spread of the estimates", "stated": "report the spread", "by": "peak_spread"}}
    out, seen = _ask({**design, "protocol": {**SPREAD, "topic_map": [{"item": "Q1", "by": "peak_spread"}]}},
                     [{"answers": {"Q1": {"by": "metric:algae_peak", "statistic": "sd"}}}], asked)
    row = next(r for r in out.design["protocol"]["topic_map"] if r["item"] == "Q1")
    assert row["by"] == "peak_spread" and "statistic" not in row
    refused = cp._check_answers({"Q1": {"by": "", "statistic": "sd"}}, asked, design)
    assert refused and "worked out once per setting" in refused[0]
    again = cp._check_answers({"Q1": {"by": "metric:peak_spread", "statistic": "sd"}}, asked, design)
    assert again and "worked out once per setting" in again[0], "the same result named with its prefix is the plan's result"


# A shared result the plan works out once per setting from algae_peak: two thresholds given one number.
CROSS = [{"id": "nutrient_cross", "inputs": ["algae_peak"], "role": "result", "label": "nutrient where algae pass 100 and 500",
          "asked": True}]
PLAN_TWO_D = {**PROTOCOL, "derived": CROSS, "topic_map": [{"item": "Q1", "by": "metric:nutrient_cross", "at": {}},
                                                          {"item": "Q2", "by": "metric:nutrient_cross", "at": {}},
                                                          {"item": "Q3", "by": "metric:bloom", "at": {}}]}


def _new_of(out, ident):
    return next((d for d in out.design["protocol"].get("derived") or [] if d.get("for_question") == ident), None)


def test_a_shared_quantity_with_no_number_of_its_own_gets_a_new_result_worked_out_from_what_the_shared_one_is() -> None:
    # Recorded on a real plan: two thresholds of one curve were one derived result; the answers step could only name an
    # existing result, and the second threshold had none, so the paper could answer only one of the two questions.
    asked = tr.quantities_to_place(REQ_TWO, PLAN_TWO_D)
    reply = {"answers": {"Q1": {"by": "metric:nutrient_cross"},
                         "Q2": {"by": "new", "id": "nutrient_500", "what": "nutrient where algae pass 500",
                                "inputs": ["metric:nutrient_cross"]}}}
    out, seen = _ask({**DESIGN, "protocol": PLAN_TWO_D}, [reply], asked)
    assert out.parts["answers"]["valid"] and len(seen) == 1 and out.parts["answers"]["new_results"] == ["nutrient_500"]
    new = _new_of(out, "Q2")
    assert new["id"] == "nutrient_500" and new["inputs"] == ["algae_peak"], "FI's record of what the shared one is worked from"
    rows = {r["item"]: r["by"] for r in out.design["protocol"]["topic_map"]}
    assert rows == {"Q1": "metric:nutrient_cross", "Q2": "metric:nutrient_500", "Q3": "metric:bloom"}
    assert "same_number" not in out.parts["answers"]
    assert '"by": "new"' in seen[0]


def test_a_new_result_for_a_shared_reported_number_is_worked_out_from_that_number() -> None:
    asked = tr.quantities_to_place(REQ_TWO, PLAN_TWO)
    reply = {"answers": {"Q1": {"by": "metric:algae_peak"}, "Q2": {"by": "new", "id": "nutrient_500", "what": "w"}}}
    out, _seen = _ask({**DESIGN, "protocol": PLAN_TWO}, [reply], asked)
    assert _new_of(out, "Q2")["inputs"] == ["algae_peak"]


def test_new_is_refused_for_a_quantity_not_shared_and_for_a_name_given_twice() -> None:
    asked = tr.quantities_to_place(REQ, PROTOCOL)
    first = next(iter(asked))
    bad = {"answers": {i: {"by": "none"} for i in asked}}
    bad["answers"][first] = {"by": "new", "id": "x_new", "what": "w"}
    good = {"answers": {i: {"by": "none"} for i in asked}}
    out, seen = _ask(DESIGN, [bad, good], asked)
    assert len(seen) == 2 and "`new` is only for a quantity the plan gives the same number as another" in seen[1]
    assert not out.design["protocol"].get("derived")
    asked2 = tr.quantities_to_place(REQ_TWO, PLAN_TWO_D)
    twice = {"answers": {"Q1": {"by": "new", "id": "n_cross", "what": "a"}, "Q2": {"by": "new", "id": "n_cross", "what": "b"}}}
    problems = cp._new_results(twice["answers"], asked2, {**DESIGN, "protocol": PLAN_TWO_D}, False)[2]
    assert problems == ["`Q2`: `n_cross` is given twice; each new result has its own name"]
    taken = cp._new_results({"Q2": {"by": "new", "id": "algae_peak", "what": "w"}}, asked2, {**DESIGN, "protocol": PLAN_TWO_D}, False)[2]
    assert taken and "already a number of the study" in taken[0]


def test_an_unusable_new_at_the_last_ask_keeps_the_plans_map_for_it_and_the_other_answers() -> None:
    asked = tr.quantities_to_place(REQ_TWO, PLAN_TWO)
    reply = {"answers": {"Q1": {"by": "metric:bloom"}, "Q2": {"by": "new", "what": "no name given"}}}
    out, seen = _ask({**DESIGN, "protocol": PLAN_TWO}, [reply] * cp.MAX_ASKS, asked)
    assert len(seen) == cp.MAX_ASKS and out.parts["answers"]["valid"] and out.parts["answers"]["new_left_out"] == ["Q2"]
    rows = {r["item"]: r["by"] for r in out.design["protocol"]["topic_map"]}
    assert rows["Q1"] == "metric:bloom" and rows["Q2"] == "metric:algae_peak"


def test_the_spread_over_trials_of_a_new_result_is_refused_like_any_result_worked_out_once_per_setting() -> None:
    asked = tr.quantities_to_place(REQ_TWO, PLAN_TWO_D)
    reply = {"answers": {"Q1": {"by": "metric:nutrient_cross"},
                         "Q2": {"by": "new", "id": "nutrient_500", "what": "w", "statistic": "sd"}}}
    problems = cp._check_answers(*cp._new_results(reply["answers"], asked, {**DESIGN, "protocol": PLAN_TWO_D}, False)[:1], asked,
                                 {**DESIGN, "protocol": {**PLAN_TWO_D, "derived": [*CROSS, *cp._new_results(
                                     reply["answers"], asked, {**DESIGN, "protocol": PLAN_TWO_D}, False)[1]]}})
    assert any("worked out once per setting" in p for p in problems)


def test_the_plan_step_writes_a_new_result_into_the_plan_with_the_map_that_names_it(tmp_path: Path) -> None:
    # Recorded: the plan step's answers path wrote only the map back to plan.md, so the second question was mapped to a
    # result the plan did not have.
    from unittest.mock import AsyncMock

    from core import dependent_map as dm
    from core import plan
    from tests.test_plan_optimisation import _engine

    eng = _engine(tmp_path, [])
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    path = plan.plan_path(eng.quest_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.render("t", None, {**DESIGN, "protocol": PLAN_TWO_D}), encoding="utf-8")
    reply = {"answers": {"Q1": {"by": "metric:nutrient_cross"}, "Q2": {"by": "new", "id": "nutrient_500", "what": "w"}}}
    eng._client = type("Stub", (), {"chat": AsyncMock(return_value=json.dumps(reply))})()
    design, sha = eng._design_from_plan()
    out, new_sha = asyncio.run(eng._ask_plan_for_the_answers(design, sha, REQ_TWO))
    written = plan.parse(path.read_text(encoding="utf-8")).design["protocol"]
    assert new_sha != sha and out["protocol"] == written
    assert [d["id"] for d in written["derived"] if d.get("for_question") == "Q2"] == ["nutrient_500"]
    ids = {dm._norm(i) for i in dm.declared_ids(written)}
    assert all(dm._norm(m["by"].split(":", 1)[1]) in ids for m in written["topic_map"] if m["by"].startswith("metric:"))


def test_new_on_a_quantity_not_shared_at_the_last_ask_keeps_its_plan_map_and_the_other_answers() -> None:
    # Recorded with the real prompt: one plan's model answered `new` for a question the plan does not give another's number.
    plan_q3 = {**PLAN_TWO, "topic_map": [r for r in PLAN_TWO["topic_map"] if r["item"] != "Q3"]}
    asked = tr.quantities_to_place(REQ_TWO, plan_q3)
    assert sorted(asked) == ["Q1", "Q2", "Q3"] and not asked["Q3"].get("shared")
    reply = {"answers": {"Q1": {"by": "metric:algae_peak"}, "Q2": {"by": "new", "id": "nutrient_500", "what": "w"},
                         "Q3": {"by": "new", "id": "bloom_new", "what": "w"}}}
    out, seen = _ask({**DESIGN, "protocol": plan_q3}, [reply] * cp.MAX_ASKS, asked)
    part = out.parts["answers"]
    assert len(seen) == cp.MAX_ASKS and part["valid"] and part["new_left_out"] == ["Q3"] and part["new_results"] == ["nutrient_500"]
    rows = {r["item"]: r["by"] for r in out.design["protocol"]["topic_map"]}
    assert rows["Q2"] == "metric:nutrient_500" and rows.get("Q3") != "metric:bloom_new"
