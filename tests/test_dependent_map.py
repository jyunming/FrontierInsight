"""The plan's model says what each dependent variable that is no declared result is (core/dependent_map.py). Its answers only add
checks and disclosure. The fake model is a function, never a real one; neutral made-up study (a pond's algae count)."""

from __future__ import annotations

import asyncio
import copy
import json
import string
from pathlib import Path
from typing import Any

from core import code_review, dependent_map, frozen_protocol, quality_report
from core import contract_parts as cp
from tests.test_contract_parts import COMPARE, DESIGN, NAMES, ROLES, TEMPLATES, parse

TEMPL = {**TEMPLATES, "plan_dependents": string.Template(
    (Path(__file__).parent.parent / "agents" / "plan_dependents.md").read_text(encoding="utf-8"))}
ENTRY_PEAK = "peak algae count per pond (read from the cells)"
ENTRY_TEMP = "temp"
ENTRY_NONE = "number of ponds that dried out"


def design(*dependent: str) -> dict[str, Any]:
    d = copy.deepcopy(DESIGN)
    d["variables"] = {"dependent": list(dependent), "independent": ["nutrient"]}
    return d


class Fake:
    """Answers each part; the dependents answers come from a queue (a dict is sent as JSON, a string as it is)."""

    def __init__(self, dependents: list[Any]) -> None:
        self.dependents, self.prompts = dependents, []

    async def __call__(self, prompt: str) -> str:
        if "Its list of dependent variables has entries" in prompt:
            self.prompts.append(prompt)
            reply = self.dependents.pop(0)
        elif "What kind of number it is" in prompt:
            reply = ROLES
        elif "Say what each one is" in prompt and "plan does not list them" in prompt:
            reply = NAMES
        else:
            reply = COMPARE
        return reply if isinstance(reply, str) else json.dumps(reply)


def run(fake: Fake, d: dict[str, Any]) -> cp.Outcome:
    return asyncio.run(cp.ask_parts(d, fake, parse, TEMPL))


GOOD = {"names": {"D1": {"answer": "result", "inputs": ["algae_peak"]}, "D2": {"answer": "setting", "setting": "temp"},
                  "D3": {"answer": "not_reported"}}}


def test_the_part_is_asked_once_for_entries_that_are_no_declared_result_and_kept_beside_the_form() -> None:
    d = design("bloom", ENTRY_PEAK, ENTRY_TEMP, ENTRY_NONE)  # `bloom` is exactly a declared result: not asked
    fake = Fake([GOOD])
    out = run(fake, d)
    assert len(fake.prompts) == 1 and "D3: " + ENTRY_NONE in fake.prompts[0] and "bloom" not in fake.prompts[0].split("The entries")[1]
    assert out.parts["dependents"]["valid"] and out.form == "contract"  # an own part: it never moves the form
    stored = out.design["protocol"]["dependent_map"]
    assert stored[ENTRY_PEAK] == {"answer": "result", "by": ["algae_peak"]}
    assert stored[ENTRY_TEMP] == {"answer": "setting", "setting": "temp"} and stored[ENTRY_NONE] == {"answer": "not_reported"}
    plain = run(Fake([GOOD]), design("bloom"))
    assert "dependents" not in plain.parts and "dependent_map" not in plain.design["protocol"]  # nothing to ask
    assert frozen_protocol.sha256(out.design["protocol"]) != frozen_protocol.sha256(plain.design["protocol"])  # the hash covers it


def test_a_missing_row_is_asked_again_once_and_a_part_still_unusable_is_left_out_alone() -> None:
    d = design(ENTRY_PEAK, ENTRY_TEMP, ENTRY_NONE)
    short = {"names": {"D1": GOOD["names"]["D1"], "D2": GOOD["names"]["D2"]}}
    fake = Fake([short, GOOD])
    out = run(fake, d)
    assert len(fake.prompts) == 2 and "`D3` is not answered" in fake.prompts[1] and out.parts["dependents"]["valid"]
    fake = Fake([short, short, GOOD])
    out = run(fake, d)
    assert len(fake.prompts) == 2 and not out.parts["dependents"]["valid"]  # at most two asks
    assert "dependent_map" not in out.design["protocol"] and out.form == "contract"  # left out alone, the form is kept


def test_an_answer_naming_an_undeclared_result_or_a_setting_the_study_does_not_vary_is_not_used() -> None:
    d = design(ENTRY_PEAK, ENTRY_TEMP)
    entries = dependent_map.asked(d)
    bad = {"names": {"D1": {"answer": "result", "inputs": ["not_a_result"]}, "D2": {"answer": "setting", "setting": "salinity"}}}
    problems = dependent_map.check(bad["names"], entries, d)
    assert any("`D1`" in p for p in problems) and any("salinity" in p for p in problems)
    assert any("must be one of" in p for p in dependent_map.check({"D1": {"answer": "computed"}, "D2": GOOD["names"]["D2"]}, entries, d))
    assert dependent_map.check({"D1": GOOD["names"]["D1"], "D2": GOOD["names"]["D2"], "X": {}}, entries, d) == ["`X` was not asked about"]


ENTRY_SPREAD = "spread of the peak count between ponds"
SPREAD = {"answer": "derived", "id": "algae_peak_sd", "inputs": ["algae_peak"],
          "what": "standard deviation of the peak count across the runs of a setting"}


def test_an_entry_answered_a_new_derived_result_is_declared_by_the_plan_before_it_is_frozen() -> None:
    from core import contract

    d = design(ENTRY_SPREAD, ENTRY_NONE)
    out = run(Fake([{"names": {"D1": SPREAD, "D2": {"answer": "not_reported"}}}]), d)
    assert out.parts["dependents"]["valid"] and out.form == "contract"
    row = next(r for r in out.design["protocol"]["derived"] if r["id"] == "algae_peak_sd")
    assert row == {"id": "algae_peak_sd", "inputs": ["algae_peak"], "role": "result", "label": SPREAD["what"], "asked": True}
    assert out.design["protocol"]["dependent_map"][ENTRY_SPREAD] == {"answer": "derived", "by": ["algae_peak_sd"]}
    # the contract the frame is written from declares it, so the code writes its function as for any derived result
    assert "algae_peak_sd" in [x.id for x in contract.from_design(out.design).derived]
    said = dependent_map.sentence(ENTRY_SPREAD, out.design["protocol"]["dependent_map"][ENTRY_SPREAD])
    assert "new derived result `algae_peak_sd`" in said
    # it is the plan's: the methodology audit's view of the design leaves it out like every row FI asked for
    assert all(r.get("id") != "algae_peak_sd" for r in cp.without_answers(out.design)["protocol"].get("derived") or [])


def test_a_new_derived_result_that_is_a_declared_result_under_another_spelling_is_not_added_twice() -> None:
    d = design(ENTRY_PEAK)
    twice = {"names": {"D1": {"answer": "derived", "id": "AlgaePeak", "inputs": ["algae_peak"], "what": "the peak count"}}}
    fake = Fake([twice, {"names": {"D1": {"answer": "result", "inputs": ["algae_peak"]}}}])
    out = run(fake, d)
    assert len(fake.prompts) == 2 and "already" in fake.prompts[1] and "algae_peak" in fake.prompts[1]
    assert out.parts["dependents"]["valid"] and not out.design["protocol"].get("derived")
    assert out.design["protocol"]["dependent_map"][ENTRY_PEAK] == {"answer": "result", "by": ["algae_peak"]}


def test_a_new_derived_result_needs_a_code_name_of_its_own_and_inputs_the_plan_declares() -> None:
    d = design(ENTRY_SPREAD, ENTRY_NONE)
    entries = dependent_map.asked(d)
    bad = dependent_map.check({"D1": {**SPREAD, "id": "peak sd"}, "D2": {**SPREAD, "inputs": ["salinity"]}}, entries, d)
    assert any("`D1`" in p and "name" in p for p in bad) and any("`D2`" in p and "inputs" in p for p in bad)
    same = dependent_map.check({"D1": SPREAD, "D2": SPREAD}, entries, d)
    assert any("`D2`" in p and "twice" in p for p in same)
    assert any("temp" in p for p in dependent_map.check({"D1": {**SPREAD, "id": "temp"}, "D2": SPREAD}, entries, d))
    # the part is still asked at most twice and left out alone: no stop, the form is kept
    fake = Fake([{"names": {"D1": {**SPREAD, "id": "peak sd"}, "D2": {"answer": "not_reported"}}}] * 3)
    out = run(fake, d)
    assert len(fake.prompts) == 2 and not out.parts["dependents"]["valid"] and out.form == "contract"
    assert not out.design["protocol"].get("derived") and "dependent_map" not in out.design["protocol"]


def test_an_uncertainty_the_plan_names_for_a_result_may_be_declared_as_its_own_derived_result() -> None:
    d = design(ENTRY_NONE)
    d["protocol"]["metrics"][0]["estimand"] = "peak algae count; its uncertainty is the standard error of the fitted peak"
    unc = {"algae_peak": {"what": "standard error of the fitted peak", "answer": "derived", "id": "algae_peak_se",
                          "inputs": ["algae_peak"]}}
    fake = Fake([{"names": {"D1": {"answer": "not_reported"}}, "uncertainties": unc}])
    out = run(fake, d)
    assert "uncertainties" in fake.prompts[0] and out.parts["dependents"]["valid"]
    assert any(r["id"] == "algae_peak_se" and r["inputs"] == ["algae_peak"] for r in out.design["protocol"]["derived"])
    rows = out.design["protocol"]["dependent_map"]
    key = next(k for k in rows if k != ENTRY_NONE)
    assert "algae_peak" in key and "standard error" in key and rows[key] == {"answer": "derived", "by": ["algae_peak_se"]}
    # an uncertainty of something the plan does not report, or one not reported, is held to FI's ids and stored as said
    entries = dependent_map.asked(d)
    assert any("not_a_result" in p for p in dependent_map.check_uncertainties({"not_a_result": unc["algae_peak"]}, d, {}))
    out = run(Fake([{"names": {"D1": {"answer": "not_reported"}},
                     "uncertainties": {"bloom": {"what": "interval of the bloom chance", "answer": "not_reported"}}}]), d)
    assert not out.design["protocol"].get("derived") and len(out.design["protocol"]["dependent_map"]) == 2
    assert entries == {"D1": ENTRY_NONE}
    # uncertainties still unusable at the last ask are left out alone: the dependent variables' answers are kept
    bad_unc = {"names": {"D1": {"answer": "not_reported"}}, "uncertainties": {"not_a_result": unc["algae_peak"]}}
    fake = Fake([bad_unc, bad_unc])
    out = run(fake, d)
    assert len(fake.prompts) == 2 and out.parts["dependents"]["valid"]
    assert out.design["protocol"]["dependent_map"] == {ENTRY_NONE: {"answer": "not_reported"}}
    assert any("not_a_result" in p for p in out.parts["dependents"]["uncertainties_left_out"])


def _protocol(rows: dict[str, Any]) -> dict[str, Any]:
    p = copy.deepcopy(DESIGN["protocol"])
    p["derived"] = [{"id": "bloom_rate", "label": "bloom per day"}]
    p["dependent_map"] = rows
    return p


def test_an_answer_only_adds_to_the_requirements_and_never_removes_one() -> None:
    d = design(ENTRY_PEAK, ENTRY_NONE)
    base = code_review.requirements(d, _protocol({}))
    wrong = {ENTRY_PEAK: {"answer": "result", "by": ["bloom"]}, ENTRY_NONE: {"answer": "not_reported"}}  # a wrong tie, a wrong "none"
    after = code_review.requirements(d, _protocol(wrong))
    assert [r["text"] for r in base if r["kind"] == "dependent variable"] == [r["text"] for r in after if r["kind"] == "dependent variable"]
    assert len(after) >= len(base) and {r["kind"] for r in after} >= {r["kind"] for r in base}
    bloom = next(r for r in after if r["name"] == "bloom")
    assert ENTRY_PEAK in bloom["text"] and "also names it as" in bloom["text"]  # an extra check on the code for that result
    assert ENTRY_PEAK not in next(r for r in base if r["name"] == "bloom")["text"]
    assert any(r["id"] == base_r["id"] and r["kind"] == base_r["kind"] for base_r in base for r in after)  # the ids stay


def test_an_answer_never_changes_the_main_quantity_and_a_derived_result_gets_an_item() -> None:
    d = design(ENTRY_PEAK, "bloom")
    rows = {ENTRY_PEAK: {"answer": "result", "by": ["bloom_rate", "algae_peak"]}}  # the first entry, tied by the plan's model
    assert code_review.headline_names(d, _protocol(rows)) == code_review.headline_names(d, _protocol({}))
    before = {r["name"]: r["headline"] for r in code_review.requirements(d, _protocol({}))}
    after = code_review.requirements(d, _protocol(rows))
    assert all(r["headline"] == before[r["name"]] for r in after if r["name"] in before)  # no item becomes the main quantity
    derived = next(r for r in after if r["name"] == "bloom_rate")  # a derived result: one more item, checked, not the main one
    assert not derived["headline"] and ENTRY_PEAK in derived["text"]
    assert ENTRY_PEAK in next(r for r in after if r["name"] == "algae_peak")["text"]


def test_a_long_first_entry_and_a_short_one_are_treated_alike() -> None:
    long_entry = ("peak algae count per pond, read from the cells of every pond in the sample over the full season of "
                  "observation and averaged over the ponds of each nutrient level")
    for entry in ("peak count", long_entry):
        d = design(entry, "bloom")
        entries = dependent_map.asked(d)
        p = dependent_map.apply(d["protocol"], {k: {"answer": "result", "inputs": ["algae_peak"]} for k in entries}, entries)
        peak = next(r for r in code_review.requirements(d, p) if r["name"] == "algae_peak")
        assert "also names it as" in peak["text"] and not peak["headline"], entry
        assert code_review.headline_names(d, p) == code_review.headline_names(d, d["protocol"]), entry


def test_a_wrong_tie_of_the_first_entry_never_turns_a_fault_that_is_left_out_into_a_stop(tmp_path: Path) -> None:
    from core import shown_reach
    from tests import test_shown_reach as T

    root = tmp_path / "quest"
    for rel, text in T.FILES.items():
        (root / "code" / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / "code" / rel).write_text(text, encoding="utf-8")
    (root / "raw").mkdir()
    metrics = {k: {"values": [1.0], "trials": [0]} for k in ("est_a", "est_b", "se_a", "lo_a", "hi_a")}
    (root / "raw" / "trials.json").write_text(json.dumps({"cells": [{"cell": {"noise": 0.1}, "metrics": metrics}]}), encoding="utf-8")
    derived = {"schema": "fi.derived/v1", "complete": True, "not_computed": [], "bodies": {}, "figures": [],
               "derived": {"corr_ab": {"status": "ok", "by": "derive_corr_ab", "inputs": ["est_a"], "by_setting": {"noise=0.1": 0.4}},
                           "cover": {"status": "ok", "by": "derive_cover", "inputs": [], "by_setting": {"noise=0.1": 0.9}}}}
    (root / "data" / "results").mkdir(parents=True)
    (root / "data" / "results" / "derived.json").write_text(json.dumps(derived), encoding="utf-8")
    entry = "the size of the first effect over the noise levels"

    def effect(dependent: list[str], rows: dict[str, Any] | None, moved: str) -> Any:
        protocol = copy.deepcopy(T.PROTOCOL) | ({"dependent_map": rows} if rows else {})
        d = {"variables": {"dependent": dependent, "independent": ["noise"]}}
        return shown_reach.Reach(root, protocol, code_review.headline_names(d, protocol), []).effect_of_changes([moved])

    wrong = effect([entry, "est_b"], {entry: {"answer": "result", "by": ["corr_ab"]}}, "corr_ab")
    assert not wrong.stop and wrong.left_out == ["corr_ab"]  # left out and said, as without the answer
    untied = effect([entry, "est_b"], None, "corr_ab")
    assert (untied.stop, untied.left_out) == (wrong.stop, wrong.left_out)
    own = effect(["est_a", "est_b"], None, "est_a")  # the plan's own main quantity: a fault there still stops
    assert own.stop and "est_a" in own.stop


def test_an_answer_never_ties_an_entry_for_the_release_of_a_finding() -> None:
    rows = {ENTRY_PEAK: {"answer": "result", "by": ["algae_peak"]}}
    rec = {"read_problems": [{"fid": "F-1", "kind": "dependent variable", "name": ENTRY_PEAK, "status": "missing"}]}
    ids = ["algae_peak", "bloom"]
    assert code_review.untied_dependents(rec, {"id": "F-1"}, ids, []) == [ENTRY_PEAK]  # the answer unlocks nothing
    assert code_review.unplaced_dependents(design(ENTRY_PEAK), _protocol(rows) | {"form": "contract"}) == [ENTRY_PEAK]


def test_the_quality_report_says_what_the_plan_s_model_answered_and_which_model(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / ".fi").mkdir(parents=True)
    (root / "needs").mkdir()
    protocol = {**_protocol({"peak count": {"answer": "result", "by": ["algae_peak"]}, "dried ponds": {"answer": "not_reported"}}),
                "form": "contract"}
    record = {"schema": frozen_protocol.FROZEN_SCHEMA, "protocol": protocol, "sha256": frozen_protocol.sha256(protocol)}
    (root / "needs" / "FROZEN_PROTOCOL.json").write_text(json.dumps(record), encoding="utf-8")
    block = {"hypothesis": "h", "variables": {"dependent": ["peak count", "dried ponds", "unmapped entry"], "independent": ["x"]}}
    (root / "plan.md").write_text("# Plan\n\n## The design (used as written)\n\n```yaml\n" + json.dumps(block) + "\n```\n",
                                  encoding="utf-8")
    (root / ".fi" / "model_calls.jsonl").write_text(json.dumps({"node": "plan_contract", "served_model": "claude-sonnet-5-5",
                                                                 "provider": "claude_cli"}) + "\n", encoding="utf-8")
    rows = quality_report._unplaced_problems(quality_report._Quest(root))
    text = " | ".join(r["text"] for r in rows)
    assert "claude-sonnet-5-5" in text and "the plan's model" in text and "is not reported by the study" in text
    assert "1 time in 20" in text and "unmapped entry" in text and "did not place it on any declared result" in text


def test_the_map_is_not_part_of_the_audited_design_and_a_new_ask_replaces_an_old_map() -> None:
    from core import receipts

    d = design(ENTRY_PEAK, ENTRY_TEMP)
    mapped = copy.deepcopy(d)
    mapped["protocol"]["dependent_map"] = {ENTRY_PEAK: {"answer": "result", "by": ["algae_peak"]}}
    assert receipts.design_core(mapped) == receipts.design_core(d)  # the methodology audit's receipt still covers the design
    answer = {"names": {"D1": {"answer": "not_reported"}, "D2": {"answer": "not_reported"}}}
    out = run(Fake([answer]), mapped)
    assert out.design["protocol"]["dependent_map"] == {ENTRY_PEAK: {"answer": "not_reported"}, ENTRY_TEMP: {"answer": "not_reported"}}
