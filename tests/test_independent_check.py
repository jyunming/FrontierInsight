"""A second implementation must not compare a value with itself (core/independent_check.py, read before the run through
core/static_checks.py). Neutral toy code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import independent_check as ic
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

PROTOCOL = {"grid": {"n": [1, 2]}, "oracles": [
    {"name": "two_ways", "kind": "second_implementation", "check": "the two routes give one total", "expected": 0,
     "tolerance": 1e-9, "reference": "derivation: the same total", "case": {"n": 1}, "measure": "gap"}]}
NO_SECOND = {"grid": {"n": [1, 2]}, "oracles": [
    {"name": "limit", "kind": "special_case", "check": "c", "expected": 1, "tolerance": 0.1, "reference": "r",
     "case": {"n": 1}, "measure": "y"}]}

SAME_REDUCTION = '''\
import numpy as np

def gap(w):
    first = np.sum(w, dtype=np.float64)
    second = np.einsum("ij->", w, dtype=np.float64)
    return abs(first - second) / max(abs(first), 1e-300)
'''
METHOD_FORM = '''\
def gap(w):
    return abs(w.sum() - sum(w))
'''
WRAPPER = '''\
import numpy as np

def total_a(w):
    return np.sum(w)

def total_b(w):
    return total_a(w)

def gap(w):
    return abs(total_a(w) - total_b(w))
'''
COPY_UNDER_ANOTHER_NAME = '''\
def route_a(w, k):
    out = 0.0
    for v in w:
        out += v * k
    return out

def route_b(w, k):
    out = 0.0
    for v in w:
        out += v * k
    return out

def gap(w):
    return abs(route_a(w, 2.0) - route_b(w, 2.0))
'''
INDEPENDENT = '''\
import numpy as np

def by_loop(w):
    out = 0.0
    for v in np.ravel(w):
        out += v
    return out

def by_pairs(w):
    flat = np.ravel(w)
    while flat.size > 1:
        flat = flat[::2] + np.append(flat[1::2], 0.0)[: flat[::2].size]
    return flat[0]

def gap(w):
    return abs(by_loop(w) - by_pairs(w)) / max(abs(by_loop(w)), 1e-300)
'''
CHANGING_NAMES = '''\
def step(x):
    return x * 0.5

def drift(x0):
    x = x0
    start = step(x)
    for _ in range(3):
        x = x * 0.9
    return abs(step(x) - start)
'''
DIFFERENT_ARGS = '''\
import numpy as np

def gap(w):
    return abs(np.sum(w) - np.sum(w * 2.0)) + np.linalg.norm(w[1:] - w[:-1])
'''


def _found(source: str, protocol: Any = PROTOCOL) -> list[ic.Tautology]:
    return ic.problems({"simulate.py": source}, protocol)


@pytest.mark.parametrize("source", [SAME_REDUCTION, METHOD_FORM, WRAPPER, COPY_UNDER_ANOTHER_NAME])
def test_a_comparison_of_one_operation_with_itself_is_caught(source: str) -> None:
    found = _found(source)
    assert len(found) == 1 and found[0].function == "gap" and found[0].script == "simulate.py"
    assert "cannot differ" in found[0].says() and f"line {found[0].line}" in found[0].says()


@pytest.mark.parametrize("source", [INDEPENDENT, CHANGING_NAMES, DIFFERENT_ARGS])
def test_two_different_routes_or_two_different_values_pass(source: str) -> None:
    assert _found(source) == []


def test_a_plan_without_a_second_implementation_check_is_not_read() -> None:
    assert _found(SAME_REDUCTION, NO_SECOND) == [] and _found(SAME_REDUCTION, {}) == []
    assert ic.tautologies({"simulate.py": SAME_REDUCTION})  # the reading itself is the same; the plan decides whether it applies
    assert ic.tautologies({"simulate.py": "def broken(:\n"}) == []


def test_the_directive_and_the_notes_say_it_plainly_without_a_value() -> None:
    found = _found(SAME_REDUCTION)
    text = ic.directive(found, PROTOCOL)
    assert "compares a value with itself" in text and "`two_ways`" in text and "the other method the plan names" in text
    assert text.rstrip().endswith("Change nothing else.")
    assert "not independent validation" in ic.writer_note(found, PROTOCOL)
    assert "not independent validation" in ic.gap(found, PROTOCOL)
    assert ic.record(found, PROTOCOL)["comparisons"][0]["function"] == "gap"


# ---- through the engine -------------------------------------------------------------------------------------------------


class _Client:
    def __init__(self, replies: list[str], asked: list[str]) -> None:
        self.replies, self.asked = replies, asked
        self.last_usage = None
        self.last_model = "test-model"

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        self.asked.append(messages[-1]["content"])
        return self.replies[min(len(self.asked), len(self.replies)) - 1]


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> tuple[Engine, list[str]]:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    asked: list[str] = []
    eng = Engine(Config(
        topic="a neutral topic", title="second", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    eng._client = _Client(replies, asked)  # type: ignore[assignment]
    eng._protocol_block = lambda state: PROTOCOL  # type: ignore[method-assign]
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "simulate.py").write_text(SAME_REDUCTION, encoding="utf-8")
    return eng, asked


@pytest.mark.asyncio
async def test_it_is_sent_back_and_an_independent_rewrite_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, asked = _engine(tmp_path, monkeypatch, [json.dumps({"code": INDEPENDENT, "patch_summary": "own route"})])
    assert await eng._static_gate({}) is None
    assert len(asked) == 1 and "compares a value with itself" in asked[0]
    assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == INDEPENDENT
    assert eng._static_check_gaps() == [] and eng._static_checks_note() == ""


@pytest.mark.asyncio
async def test_one_that_stays_does_not_count_as_validation_and_the_paper_says_so_but_the_quest_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, asked = _engine(tmp_path, monkeypatch, [json.dumps({"code": SAME_REDUCTION, "patch_summary": "no change"})])
    assert await eng._static_gate({}) is None, "a note, not a stop"
    assert len(asked) == 2
    gaps = eng._static_check_gaps()
    assert len(gaps) == 1 and "not independent validation" in gaps[0]
    note = eng._run_checks_note()
    assert "limitations" in note and "not independent validation" in note
    # The same text is not asked about again, and the record stands; once the code is independent it goes.
    assert await eng._static_gate({}) is None and len(asked) == 2 and eng._static_check_gaps() == gaps
    (eng.quest_root / "code" / "simulate.py").write_text(INDEPENDENT, encoding="utf-8")
    assert await eng._static_gate({}) is None and eng._static_check_gaps() == [] and eng._run_checks_note() == ""
