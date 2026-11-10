"""The names step records, for a derived result it creates, which question of the topic it answers (``for_question``, from its own
reply, and only for a question the topic asks). The answers part replaces the plan's map for a question only with a derived result
made for that very question; any other derived result stays a disclosed candidate. The fake model is a function. Neutral made-up
study: a pond's algae."""

from __future__ import annotations

import asyncio
import json
import string
from pathlib import Path
from typing import Any

from core import contract_parts as cp
from tests.test_contract_parts import COMPARE, DESIGN, ROLES

NAMES_TEMPLATES = {n: string.Template((Path(__file__).parent.parent / "agents" / f"{n}.md").read_text(encoding="utf-8"))
                   for n in ("plan_roles", "plan_names", "plan_comparisons", "plan_answers")}
ASKED = {"Q1": {"name": "peak algae", "stated": "report the peak", "by": "metric:algae_peak"},
         "Q2": {"name": "bloom chance", "stated": "report the chance", "by": "metric:bloom"}}


def _flow(names_row: dict[str, Any], answer: dict[str, Any]) -> tuple[cp.Outcome, str]:
    design = {**DESIGN, "protocol": {**DESIGN["protocol"],
                                     "topic_map": [{"item": "Q1", "by": "metric:algae_peak"}, {"item": "Q2", "by": "metric:bloom"}]}}
    names_prompt: list[str] = []

    async def ask(prompt: str) -> str:
        if "Say what each one is" in prompt:
            names_prompt.append(prompt)
            return json.dumps({"names": {"growth_theory": names_row}})
        if "What kind of number it is" in prompt:
            return json.dumps(ROLES)
        if "compares between values" in prompt:
            return json.dumps(COMPARE)
        return json.dumps({"answers": answer})

    out = asyncio.run(cp.ask_parts(design, ask, lambda t: json.loads(t), NAMES_TEMPLATES, topic_quantities=ASKED))
    assert out.parts["names"]["valid"] and out.parts["answers"]["valid"]
    return out, names_prompt[0]


def _entry(out: cp.Outcome, item: str) -> dict[str, Any]:
    return next(m for m in out.design["protocol"]["topic_map"] if m["item"] == item)


DERIVED = {"answer": "derived", "what": "the peak after a day", "inputs": ["algae_peak"]}
ANSWER = {"Q1": {"by": "metric:growth_theory"}, "Q2": {"by": "none"}}


def test_a_derived_result_made_for_a_question_replaces_the_plans_map_for_it() -> None:
    out, prompt = _flow({**DERIVED, "question": "Q1"}, ANSWER)
    assert "- Q1: peak algae" in prompt and "- Q2: bloom chance" in prompt  # the step is shown the questions
    assert _entry(out, "Q1")["by"] == "metric:growth_theory" and "other_candidate" not in _entry(out, "Q1")
    assert _entry(out, "Q2")["by"] == "metric:bloom"
    assert next(d for d in out.design["protocol"]["derived"] if d["id"] == "growth_theory")["for_question"] == "Q1"


def test_a_derived_result_made_for_another_question_does_not_replace_the_map() -> None:
    out, _prompt = _flow({**DERIVED, "question": "Q2"}, ANSWER)
    assert _entry(out, "Q1")["by"] == "metric:algae_peak" and _entry(out, "Q1")["other_candidate"] == "metric:growth_theory"


def test_a_derived_result_with_no_link_or_a_link_to_no_question_keeps_the_map_and_discloses_it() -> None:
    for row in (DERIVED, {**DERIVED, "question": "Q9"}, {**DERIVED, "question": ["Q1"]}):
        out, _prompt = _flow(row, ANSWER)
        assert _entry(out, "Q1")["by"] == "metric:algae_peak" and _entry(out, "Q1")["other_candidate"] == "metric:growth_theory"
        assert "for_question" not in next(d for d in out.design["protocol"]["derived"] if d["id"] == "growth_theory")


def test_the_link_is_an_fi_key_a_models_own_word_for_it_is_not_believed() -> None:
    design = {**DESIGN, "protocol": {**DESIGN["protocol"], "derived": [{"id": "d1", "inputs": ["algae_peak"], "role": "result",
                                                                       "asked": True, "for_question": "Q1"}]}}
    assert "for_question" not in cp.strip_fi_keys(design)["protocol"]["derived"][0]
