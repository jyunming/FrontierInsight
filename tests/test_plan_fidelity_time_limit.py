"""The code is written for the topic's own run-time limit when it is shorter than FI's (execution.timeout_s).

The real case (rounds s1 and r13, first-order kinetics): the topic says "under 60 seconds of runtime" and the config's
execution.timeout_s is 900; the code-writing prompts were given 900 s. Sizing the run to the topic's limit is the work of
the run-size change (RunChecksMixin._size_the_run there); this file covers only what the prompts are told.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from core import disclosure
from core.engine import Engine
from tests.test_design_prompt_inputs import _cfg

FIX = Path(__file__).parent / "fixtures" / "plan_fidelity"


def _topic(quest: str) -> tuple[str, list[str]]:
    config = yaml.safe_load((FIX / quest / "config.yaml").read_text(encoding="utf-8"))
    required = json.loads((FIX / quest / "topic_requirements.json").read_text(encoding="utf-8"))
    return config["topic"], [c if isinstance(c, str) else c.get("stated", "") for c in required.get("constraints") or []]


@pytest.mark.parametrize("quest", ["s1_kinetics", "r13_kinetics"])
def test_the_kinetics_topics_state_sixty_seconds(quest):
    topic, constraints = _topic(quest)
    assert disclosure._topic_limit(constraints, topic) == 60.0


def test_the_code_writing_prompts_are_given_the_topics_shorter_limit(tmp_path):
    eng = Engine(_cfg(tmp_path))
    eng.config.execution.timeout_s = 900
    assert eng._planning_limit({"topic": "CPU only; under 60 seconds of runtime."}) == (
        "60 (the topic's own limit, which the code is written for; FI stops a run that takes longer than 900 s, its "
        "execution.timeout_s)")
    assert eng._planning_limit({"topic": "SIR final outbreak size"}) == "900"
    assert eng._planning_limit({"topic": "Run in under 2 hours of runtime."}) == "900"  # FI's own is shorter


def test_the_design_prompt_keeps_fis_limit_beside_the_time_note():
    source = (Path(__file__).resolve().parent.parent / "core" / "engine.py").read_text(encoding="utf-8")
    assert "timeout_s=str(self.config.execution.timeout_s),\n            time_note=self._time_note(state)," in source
    assert source.count("timeout_s=self._planning_limit(state)") == 3  # implement_outline, implement, implement_body
