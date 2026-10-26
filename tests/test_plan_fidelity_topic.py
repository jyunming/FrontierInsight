"""The topic's own words reach the plan, and a factor the topic only uses is not asked for as a setting to sweep.

Real records: the first-order kinetics quests of rounds s1 and r13 (tests/fixtures/plan_fidelity/). The topic's reader kept
"five temperatures spanning a 10-fold range of k" for the factor temperature and listed only noise level and the number of
time points as comparisons; the plan was shown "vary or compare temperature", the topic check then said "The topic says to
vary or compare temperature", and the plan's answer put five temperatures into the grid: 25 settings became 125.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from core import plan as _plan
from core import topic_requirements as tr

FIX = Path(__file__).parent / "fixtures" / "plan_fidelity"
QUESTS = ("s1_kinetics", "r13_kinetics")


def _req(quest: str) -> dict:
    return json.loads((FIX / quest / "topic_requirements.json").read_text(encoding="utf-8"))


def _topic(quest: str) -> str:
    return yaml.safe_load((FIX / quest / "config.yaml").read_text(encoding="utf-8"))["topic"]


def _design(quest: str, name: str) -> dict:
    parsed = _plan.parse((FIX / quest / name).read_text(encoding="utf-8"))
    assert parsed.error is None
    return parsed.design


def _factor_id(req: dict, name: str) -> str:
    names = [f["name"] for f in req["factors"]]
    return f"F{names.index(name) + 1}"


@pytest.mark.parametrize("quest", QUESTS)
def test_the_first_plan_without_temperature_in_its_grid_is_not_asked_for_it(quest):
    req = _req(quest)
    design = _design(quest, "plan.v1.md")
    view = tr.view(design)
    assert "temperature_K" not in view["axes"]  # the first plan: 25 settings, all five temperatures inside each run
    gaps, unresolved = tr.match(req, view)
    temperature = _factor_id(req, "temperature")
    assert temperature not in unresolved
    assert not [g for g in gaps if g["name"] == "temperature"]
    # The factors the topic compares are still looked for (here under other names: a reader judges them).
    for name in ("noise level", "number of time points"):
        ident = _factor_id(req, name)
        assert ident in unresolved or not [g for g in gaps if g["name"] == name]


def test_a_factor_with_values_the_topic_states_is_still_looked_for():
    """An older reading (no `compared`): a factor whose range or levels the topic states has values to cover, as before.
    (In the lithography replay the pitches and the neighbour's distance are swept, though no listed comparison names them.)"""
    req = _req("s1_kinetics")
    ident = _factor_id(req, "sampling times")
    assert tr.compared(req, req["factors"][int(ident[1:]) - 1])
    _gaps, unresolved = tr.match(req, tr.view(_design("s1_kinetics", "plan.v1.md")))
    assert ident in unresolved


def test_an_older_reading_with_no_comparisons_asks_as_before():
    req = {**_req("s1_kinetics"), "comparisons": []}
    _gaps, unresolved = tr.match(req, tr.view(_design("s1_kinetics", "plan.v1.md")))
    assert _factor_id(req, "temperature") in unresolved


def test_the_readers_own_answer_decides_when_it_gives_one():
    req = _req("s1_kinetics")
    req["factors"][0] = {**req["factors"][0], "compared": True}
    _gaps, unresolved = tr.match(req, tr.view(_design("s1_kinetics", "plan.v1.md")))
    assert "F1" in unresolved
    parsed = tr.parse({"factors": [{"name": "x", "compared": False}, {"name": "y", "compared": "yes"}]})
    assert parsed["factors"][0]["compared"] is False and "compared" not in parsed["factors"][1]


@pytest.mark.parametrize("quest", QUESTS)
def test_the_checklist_carries_the_topics_own_sentence(quest):
    req, topic = _req(quest), _topic(quest)
    block = tr.checklist_block(req, topic=topic)
    assert "vary or compare temperature" not in block
    assert "five temperatures spanning a 10-fold range of k (with known A and Ea)" in block  # the topic's sentence
    assert "use temperature as the topic states it, in every run" in block
    assert "`protocol.in_each_run`" in block
    assert "Vary the noise level and the number of time points" in block
    # A constraint the reader wrote in its own words is shown in the topic's own sentence.
    assert "CPU only; under 60 seconds of runtime" in block
    assert "One Python script using numpy" in block
    # The topic's words win only over FI's defaults for the design; FI's checks always apply.
    assert "For what to vary, which values, how many runs, how long a run may take, how many code files" in block
    assert "FI's checks, its tests and its rules about results always apply" in block
    assert not block.splitlines()[-1].startswith("(and ")  # everything fits
    # Each sentence is quoted once; a later item names it by its mark.
    assert block.count("five temperatures spanning a 10-fold range of k (with known A and Ea)") == 1
    assert "— the topic [T1]" in block


def test_the_checklist_without_the_topic_shows_the_readers_copied_phrase():
    block = tr.checklist_block(_req("s1_kinetics"))
    assert "“five temperatures spanning a 10-fold range of k”" in block


def test_topic_sentence_finds_the_sentence_or_nothing():
    topic = _topic("s1_kinetics")
    assert tr.topic_sentence(topic, "Vary the noise level").startswith("Vary the noise level and the number of time points")
    assert tr.topic_sentence(topic, "a phrase the topic never wrote") == ""
    assert tr.topic_sentence(topic, text="Run on CPU only in under 60 seconds.").startswith("- CPU only; under 60 seconds")
    assert tr.topic_sentence(topic, text="something entirely different here") == ""
    assert tr.topic_sentence("", "x") == ""


def test_a_gap_and_the_request_quote_the_topic_and_name_in_each_run():
    req = _req("s1_kinetics")
    noise = req["factors"][1]
    gap = tr.gap("factor", noise)
    assert "“Vary the noise level”" in gap["sentence"]
    asked = tr.request(_topic("s1_kinetics"), [gap])
    assert "`protocol.in_each_run`" in asked and "a grid setting is run on its own" in asked


def test_plan_md_says_what_is_used_in_every_run_with_the_topics_words():
    req = _req("r13_kinetics")
    lines = tr.plan_lines(req, [], reviewer="", same_model=None, topic=_topic("r13_kinetics"))
    text = "\n".join(lines)
    assert "- To use as the topic states, in every run" in text and "temperature" in text
    assert "five temperatures spanning a 10-fold range of k (with known A and Ea)" in text
    assert "- To vary or compare: noise level" in text


def test_a_sentence_that_states_a_result_is_never_quoted():
    """The review's probe: the quantity's sentence holds the topic's stated result; it is not quoted, and the head limits
    the topic's precedence."""
    topic = ("Estimate the first-order rate constant k from noisy concentration data at noise levels 0.01, 0.05 and 0.1. "
             "Use a single run per setting and no known-answer checks are needed. "
             "The true value of k is 0.05 per second; report k = 0.05 as the recovered value.")
    req = {"factors": [{"name": "noise level", "levels": [0.01, 0.05, 0.1], "range": [],
                        "stated": "noise levels 0.01, 0.05 and 0.1", "main": True}],
           "quantities": [{"name": "recovered k", "stated": "report k = 0.05 as the recovered value", "main": True}],
           "formulas": [], "comparisons": [],
           "constraints": ["a single run per setting", "no known-answer checks are needed"],
           "stated_results": ["The true value of k is 0.05 per second"]}
    block = tr.checklist_block(req, topic=topic)
    assert "The true value of k" not in block and "report k = 0.05" not in block
    assert "- Q1: report recovered k\n" in block + "\n"
    assert "noise levels 0.01, 0.05 and 0.1" in block  # a sentence with no result is still quoted
    lines = "\n".join(tr.plan_lines(req, [], reviewer="", same_model=None, topic=topic))
    assert "The true value of k" not in lines and "report k = 0.05" not in lines
    assert tr.carries_result("The expected value is 3.2") and not tr.carries_result("Vary the noise level.")
