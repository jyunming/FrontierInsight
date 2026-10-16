"""A topic that asks for the code in one script: the topic's reader says so as a number (``code_files``), and the request
is said, never a reason to turn FI's checks of the code off (core/code_layout.py).

The real case (rounds s1 and r13, first-order kinetics): the topic's goal "One Python script using numpy ..." and FI's
default layout (a package, unit tests, an equation list, run.py) disagreed. An earlier reading matched the topic's prose
against a pattern of words, which both missed requests ("one .py file", "a single-file implementation") and read others
that are not requests ("write all results to one file", "do not put everything in one script"), and a match turned off
the step-by-step functions and the unit tests. Now the topic's reader answers ``code_files``; FI keeps its layout and its
checks, and plan.md, run.log and the paper say the code has more files than the topic asked for.

The fixtures' ``code_files: 1`` was added by hand (their topic reading came before the field existed); every topic of
the review's probe is handled the same way below, because nothing reads the prose.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import code_layout, disclosure, paper_facts
from core import topic_requirements as tr
from tests.test_equation_gate import SPRING, _engine, _plan_md

FIX = Path(__file__).parent / "fixtures" / "plan_fidelity"

# The review's probe (scratch_plan/onefile_probe.py): the first seven are not a request for one script, the rest are.
PROBE = [
    "Simulate an SIR epidemic over R0 1.5..3 and write all results to one file (results.csv).",
    "Write one file per setting with the trajectories.",
    "Load the measurements from a single file of 200 rows and fit the model.",
    "Compare one script that uses Euler with another that uses RK4.",
    "Assess a single program of vaccination across age groups.",
    "The input is 1 file of spectra; estimate the peak positions.",
    "Do not put everything in one script; use a package with tests.",
    "One Python script using numpy only, under 60 s.",
    "Deliver one .py file that runs in under a minute.",
    "A single-file implementation is enough.",
    "Write a single self-contained script.",
    "Use no more than one file of code.",
    "Put the code in one module.",
    "Just a short script with numpy, nothing else.",
]
_READING = {"factors": [{"name": "x", "stated": "x"}], "quantities": [], "formulas": [], "comparisons": []}


def test_the_prose_is_never_read_for_a_one_script_request():
    """No word pattern is left: the layout cannot be decided from the topic's text."""
    assert not hasattr(code_layout, "one_file_sentence")
    import inspect

    assert "topic" not in inspect.signature(code_layout.decide).parameters


@pytest.mark.parametrize("topic", PROBE)
def test_only_the_readers_number_counts_for_every_probe_topic(topic):
    """Whatever the topic's wording, only the reader's structured answer is kept, and a reply without it is ``None``."""
    assert tr.code_files(tr.parse({**_READING, "constraints": [topic]})) is None
    assert tr.code_files(tr.parse({**_READING, "constraints": [topic], "code_files": 1})) == 1


@pytest.mark.parametrize("value, kept", [(1, 1), ("1", 1), (3.0, 3), (None, None), ("one", None), (True, None),
                                         (0, None), (-2, None), ([1], None)])
def test_the_readers_answer_is_a_whole_number_or_nothing(value, kept):
    assert tr.code_files(tr.parse({**_READING, "code_files": value})) == kept
    assert ("code_files" in tr.parse({**_READING, "code_files": value})) is (kept is not None)


@pytest.mark.parametrize("quest", ["s1_kinetics", "r13_kinetics"])
def test_the_kinetics_readings_ask_for_one_file(quest):
    required = json.loads((FIX / quest / "topic_requirements.json").read_text(encoding="utf-8"))
    assert tr.code_files(required) == 1


def test_the_reader_is_asked_for_the_number():
    prompt = (Path(__file__).resolve().parent.parent / "agents" / "topic_requirements.md").read_text(encoding="utf-8")
    assert "**code_files**" in prompt and '"code_files": null' in prompt


def test_a_one_file_request_keeps_the_package_and_its_tests_and_is_said():
    decision = code_layout.decide({}, enabled=True, max_extra_lines=10_000, max_extra_calls=2, package="spring",
                                  code_files=1)
    assert decision["shape"] == code_layout.PACKAGE and decision["topic_code_files"] == 1
    note = code_layout.files_note(decision)
    assert "The topic asks for the code in one file; FI keeps more" in note and "unit tests" in note
    assert note in "\n".join(code_layout.plan_lines(decision))
    # Nothing asked, or as many files as FI keeps: nothing to say.
    assert code_layout.files_note(code_layout.decide({}, enabled=True, max_extra_lines=10_000, max_extra_calls=2,
                                                     package="spring")) == ""
    assert code_layout.files_note(code_layout.decide({}, enabled=True, max_extra_lines=10_000, max_extra_calls=2,
                                                     package="spring", code_files=6)) == ""
    # Two scripts (the layout is off): one file asked is still more than FI keeps.
    off = code_layout.decide({}, enabled=False, max_extra_lines=10_000, max_extra_calls=2, package="spring",
                             code_files=1)
    assert off["shape"] == code_layout.SINGLE and "simulate.py) and the analysis" in code_layout.files_note(off)


def test_at_plan_time_a_one_file_topic_keeps_the_package_and_the_paper_must_say_it(tmp_path: Path) -> None:
    protocol = {"grid": {"c": [0.5, 1.0]}, "model": SPRING}
    engine = _engine(tmp_path)
    _plan_md(engine, protocol)
    (engine.fi_dir / "topic_requirements.json").write_text(json.dumps({**_READING, "code_files": 1}), encoding="utf-8")
    lines = engine._plan_code_layout({"design": {"protocol": protocol}, "topic": "Damped spring. One Python script."})
    assert code_layout.load(engine.quest_root)["shape"] == code_layout.PACKAGE
    assert any("The topic asks for the code in one file" in line for line in lines)
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "small research tool" in log and "The topic asks for the code in one file" in log
    items = [i for i in paper_facts.required_items(engine.quest_root) if i.kind == paper_facts.NOT_DONE]
    assert [i.name for i in items] == ["code in the files the topic asked for"]
    assert "not in one file" in disclosure.run_record(engine.quest_root)
    # The reading no longer asks for it (a topic edited and read again): the item goes.
    (engine.fi_dir / "topic_requirements.json").write_text(json.dumps(_READING), encoding="utf-8")
    engine._plan_code_layout({"design": {"protocol": protocol}, "topic": "Damped spring."})
    assert not [i for i in paper_facts.required_items(engine.quest_root) if i.kind == paper_facts.NOT_DONE]
