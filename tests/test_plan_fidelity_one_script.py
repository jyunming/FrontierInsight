"""A topic that asks for the code in one script is not given FI's default package layout (core/code_layout.py).

The real case (rounds s1 and r13, first-order kinetics): the topic's goal "One Python script using numpy ..." and FI's
default layout (a package, unit tests, an equation list, run.py: code_layout.json shape "package") disagreed; the
topic's own constraints win over FI's defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from core import code_layout
from tests.test_equation_gate import SPRING, _engine, _plan_md

FIX = Path(__file__).parent / "fixtures" / "plan_fidelity"


@pytest.mark.parametrize("quest", ["s1_kinetics", "r13_kinetics"])
def test_the_kinetics_topics_ask_for_one_script(quest):
    topic = yaml.safe_load((FIX / quest / "config.yaml").read_text(encoding="utf-8"))["topic"]
    required = json.loads((FIX / quest / "topic_requirements.json").read_text(encoding="utf-8"))
    said = code_layout.one_file_sentence(topic, required["constraints"])
    assert "one Python script" in said.replace("One", "one")
    assert "One Python script" in code_layout.one_file_sentence(topic)  # from the topic's text alone, too


@pytest.mark.parametrize("text", [
    "Write the code in more than one script.",
    "Save one file per setting under raw/.",
    "Use at least one file of results for each figure.",
    "Simulate a damped spring and report the settling time.",
])
def test_other_mentions_of_one_file_are_not_a_request_for_one_script(text):
    assert code_layout.one_file_sentence(text) == ""


def test_the_topics_request_decides_the_layout_and_says_why():
    decision = code_layout.decide({}, enabled=True, max_extra_lines=10_000, max_extra_calls=2, package="spring",
                                  one_file="One Python script using numpy.")
    assert decision["shape"] == code_layout.SINGLE and decision["topic"] == "One Python script using numpy."
    lines = "\n".join(code_layout.plan_lines(decision))
    assert "the topic asks for “One Python script using numpy.”" in lines and "Raise the limit" not in lines
    assert "the topic asks for" in code_layout.summary(decision)
    assert code_layout.decide({}, enabled=True, max_extra_lines=10_000, max_extra_calls=2,
                              package="spring")["shape"] == code_layout.PACKAGE


def test_at_plan_time_a_one_script_topic_keeps_two_scripts(tmp_path: Path) -> None:
    protocol = {"grid": {"c": [0.5, 1.0]}, "model": SPRING}
    engine = _engine(tmp_path)
    _plan_md(engine, protocol)
    lines = engine._plan_code_layout({"design": {"protocol": protocol},
                                      "topic": "Damped spring. CONSTRAINTS: one Python script; CPU only."})
    assert code_layout.load(engine.quest_root)["shape"] == code_layout.SINGLE
    assert any("the topic asks for" in line for line in lines)
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "the code keeps two scripts" in log and "the topic asks for" in log
