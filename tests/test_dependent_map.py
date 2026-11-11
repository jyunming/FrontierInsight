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
    assert any("must be one of" in p for p in dependent_map.check({"D1": {"answer": "derived"}, "D2": GOOD["names"]["D2"]}, entries, d))
    assert dependent_map.check({"D1": GOOD["names"]["D1"], "D2": GOOD["names"]["D2"], "X": {}}, entries, d) == ["`X` was not asked about"]


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


def test_the_main_quantity_only_gains_names_and_a_derived_result_gets_an_item() -> None:
    d = design(ENTRY_PEAK, "bloom")
    before = code_review.headline_names(d, _protocol({}))
    rows = {ENTRY_PEAK: {"answer": "result", "by": ["bloom_rate", "algae_peak"]}}
    after = code_review.headline_names(d, _protocol(rows))
    assert set(before) <= set(after) and {"algae_peak", "bloom_rate"} <= set(after)
    assert any(r["name"] == "bloom_rate" for r in code_review.requirements(d, _protocol(rows)))  # a derived result: one more item


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
