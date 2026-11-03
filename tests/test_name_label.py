"""A name the plan's model says is only another word for several declared results is a label, never a result FI freezes
(core/contract.py ``is_family_label``, core/contract_parts.py ``_apply_names``). Neutral made-up study: two counting rules
(`rate_a`, `rate_b`) applied to the same simulated samples. The fake model is a function, never a real one."""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

import pytest

from core import code_review, contract, contract_parts as cp, plan

TEMPLATES = {n: string.Template((Path(__file__).parent.parent / "agents" / f"{n}.md").read_text(encoding="utf-8"))
             for n in ("plan_roles", "plan_names", "plan_comparisons", "plan_formulas")}

DESIGN: dict[str, Any] = {
    "hypothesis": "h", "method": "Simulate samples and apply two counting rules; report `rate` for each rule.",
    "protocol": {
        "grid": {"size": [10, 20], "rule": ["a", "b"]},
        "runs_per_setting": 50,
        "metrics": [
            {"id": "rate_a", "kind": "mean", "estimand": "rate under rule a", "unit": "one sample"},
            {"id": "rate_b", "kind": "mean", "estimand": "rate under rule b", "unit": "one sample"},
        ],
    },
}
ROLES = {"results": {"rate_a": {"role": "result", "depends_on": ["size"], "returned": "each_run"},
                     "rate_b": {"role": "result", "depends_on": ["size"], "returned": "each_run"}}}
LABEL = {"names": {"rate": {"answer": "label", "what": "the rate of each rule", "inputs": ["rate_a", "rate_b"]}}}
# a name computed from two different reported numbers, with no formula: the study's own function works it out
DERIVED = {"names": {"rate": {"answer": "derived", "what": "correlation of the two rates across samples", "inputs": ["rate_a", "rate_b"]}}}


class Fake:
    def __init__(self, **replies: list[Any]) -> None:
        self.replies, self.prompts = replies, []

    async def __call__(self, prompt: str) -> str:
        part = next(k for k, marker in (("roles", "What kind of number it is"), ("names", "Say what each one is"),
                                        ("comparisons", "compares between values")) if marker in prompt)
        self.prompts.append((part, prompt))
        reply = self.replies[part].pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


def _parse(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


async def _run(fake: Fake) -> cp.Outcome:
    return await cp.ask_parts(DESIGN, fake, _parse, TEMPLATES)


def test_the_names_question_offers_the_label_answer() -> None:
    assert "label" in contract.NAME_ANSWERS
    assert '"label"' in TEMPLATES["plan_names"].template


def test_check_part_label_needs_two_or_more_declared_results() -> None:
    assert contract.check_part("names", LABEL["names"], DESIGN) == []
    for inputs in (["rate_a"], ["rate_a", "nothing_declared"], []):
        bad = {"rate": {"answer": "label", "what": "w", "inputs": inputs}}
        assert contract.check_part("names", bad, DESIGN), inputs


@pytest.mark.asyncio
async def test_a_label_is_kept_as_a_name_for_its_results_and_is_never_a_derived_result() -> None:
    out = await _run(Fake(roles=[ROLES], names=[LABEL], comparisons=[{"comparisons": []}]))
    p = out.design["protocol"]
    assert out.form == "contract" and not p.get("derived")
    assert contract.family_labels(p) == {"rate": ["rate_a", "rate_b"]}
    c = contract.from_design(out.design)
    assert [r.id for r in c.results] == ["rate_a", "rate_b"] and not c.derived
    assert [m["id"] for m in p["metrics"]] == ["rate_a", "rate_b"]


@pytest.mark.asyncio
async def test_a_comparison_of_a_label_is_asked_again_naming_its_results() -> None:
    again = {"comparisons": [{"result": "rate_a", "across": "size", "values": [10, 20], "how": "difference"}]}
    fake = Fake(roles=[ROLES], names=[LABEL],
                comparisons=[{"comparisons": [{"result": "rate", "across": "size", "values": [10, 20], "how": "difference"}]}, again])
    out = await _run(fake)
    assert out.parts["comparisons"]["asks"] == 2 and out.form == "contract"
    assert "another name for rate_a, rate_b" in fake.prompts[-1][1]
    assert [c["result"] for c in out.design["protocol"]["comparisons"]] == ["rate_a"]


@pytest.mark.asyncio
async def test_a_derived_result_of_several_results_with_no_formula_is_still_frozen() -> None:
    """The known positive: a quantity the study's own function works out from several different results is not a label."""
    out = await _run(Fake(roles=[ROLES], names=[DERIVED], comparisons=[{"comparisons": []}]))
    p = out.design["protocol"]
    assert [(d["id"], d["inputs"], d["role"]) for d in p["derived"]] == [("rate", ["rate_a", "rate_b"], "result")]
    assert not contract.family_labels(p)
    assert [d.id for d in contract.from_design(out.design).derived] == ["rate"]


def test_a_plans_own_derived_result_with_several_inputs_and_a_formula_is_read_unchanged() -> None:
    design = {**DESIGN, "protocol": {**DESIGN["protocol"], "form": "contract", "derived": [
        {"id": "gap", "inputs": ["rate_a", "rate_b"], "formula": "rate_a - rate_b", "role": "result"}]}}
    d = contract.from_design(design).derived
    assert [(x.id, x.inputs, x.formula) for x in d] == [("gap", ("rate_a", "rate_b"), "rate_a - rate_b")]


def test_a_label_as_the_main_result_is_its_results() -> None:
    ids = ["rate_a", "rate_b", "other"]
    assert code_review.headline_ids(["rate"], ids, {}, {"rate": ["rate_a", "rate_b"]}) == {"rate_a", "rate_b"}
    assert code_review.headline_ids(["rate"], ids, {}, {}) == set()
    assert code_review.headline_ids(["other"], ids, {}, {"rate": ["rate_a", "rate_b"]}) == {"other"}


@pytest.mark.asyncio
async def test_the_plan_says_a_label_computes_nothing() -> None:
    out = await _run(Fake(roles=[ROLES], names=[LABEL], comparisons=[{"comparisons": []}]))
    text = "\n".join(plan._returns_lines(out.design["protocol"]))
    assert "- `rate`: only another name for `rate_a`, `rate_b`" in text and "no number of this name is computed" in text
