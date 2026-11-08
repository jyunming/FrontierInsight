"""The code reader is told that the plan's frozen settings are its definition of those settings.

A made-up topic: a plan freezes the sample positions rounded (``in_each_run``) while its prose says they are derived exactly.
FI hands the frozen values to the code, so code that takes them as given is the plan's code; the reader used to call every
metric computed at those positions "different", and FI left those quantities out of the paper. The rule is in the reader's
prompt (``agents/code_review.md``); no model is called here."""
from __future__ import annotations

import json
import string
from pathlib import Path

from core import code_review as cr

PROMPT = Path(__file__).resolve().parent.parent / "agents" / "code_review.md"
PROTOCOL = {"grid": {"noise": [0.01, 0.05]}, "in_each_run": {"spacing": [1.0, 1.78, 3.16, 5.62, 10.0]},
            "metrics": [{"id": "rate_hat", "estimand": "decay rate read at the five derived positions (10 ** (i / 4))",
                         "kind": "mean", "unit": "1/unit"}]}
DESIGN = {"method": "The positions are derived: p_i = 10 ** (i / 4).", "variables": {"independent": ["noise"]}}


def _rendered() -> str:
    reqs = cr.requirements(DESIGN, PROTOCOL)
    calls = cr.reading_calls(PROMPT.parent, DESIGN, PROTOCOL, reqs=reqs)  # any code folder: only the plan is looked at
    plan = calls[0]["plan"]
    return string.Template(PROMPT.read_text(encoding="utf-8")).substitute(
        topic="a decay rate", requirements=json.dumps([{k: r[k] for k in ("id", "kind", "text")} for r in reqs]), plan=plan,
        open_items="(none)", code="(code)", previous="", small_run="(none)")


def test_the_reader_is_told_the_frozen_settings_are_the_plans_definition() -> None:
    text = _rendered()
    assert "frozen settings are its definition" in text
    # what it must still call different: a value that is not the frozen one
    assert "still \"different\"" in text and "written into the code" in text
    # a disagreement between the plan's prose and its frozen values is reported as the plan's problem
    assert "`plan_conflict` true" in text.split("frozen settings are its definition")[1][:900]


def test_the_rule_sits_with_the_judging_of_requirements_not_after_the_plan() -> None:
    text = _rendered()
    assert text.index("frozen settings are its definition") < text.index("# What to judge about the logic")


def test_the_plan_the_reader_is_shown_holds_the_frozen_values_the_rule_speaks_of() -> None:
    text = _rendered()
    assert '"in_each_run"' in text and "1.78" in text and '"grid"' in text
