"""Replay of a real quest whose own run-time setting guard overwrote a constant the plan defines
(tests/fixtures/replay/plan_constant_overwritten/). The code read a fixed reference value with a fallback, which is the
number the plan states ("a declared reference ... of 80"); the guard took the read for the swept setting spelt another
way, asked for a repair, and the repair set the reference to the swept setting, so every threshold then followed the swept
value and the response was erased. The guard now tells a fallback for a swept setting (a real fault) from a constant the
plan itself defines (not a fault), and a repair never substitutes the first for the second. No model is called. The
names are read from the fixture, not written here."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from core import grid_keys as gk
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "plan_constant_overwritten"
PROTOCOL = json.loads((FIXTURE / "protocol.json").read_text(encoding="utf-8"))
PLAN_TEXT = (FIXTURE / "plan_excerpt.md").read_text(encoding="utf-8")
BEFORE = (FIXTURE / "simulate_before_guard.py").read_text(encoding="utf-8")
SWEPT = next(iter(PROTOCOL["grid"]))  # the setting the study sweeps
DESIGN = {"_text": PLAN_TEXT}
_READ = gk.reads(BEFORE, "simulate.py")
CONST_KEYS = next(r.keys for r in _READ if r.fallback)  # the names the code reads the plan's constant under
CONST = CONST_KEYS[0]
READ_TEXT = next(line.strip() for line in BEFORE.splitlines() if f'"{CONST}"' in line)  # name = float(cell.get(..., 80.0))
READ_EXPR = READ_TEXT.split("=", 1)[1].strip()
TARGET = READ_TEXT.split("=", 1)[0].strip()
# The guard's own rewrite, as the real code has it: the same name set to the swept setting.
REWRITE_LINE = next(line.strip() for line in (FIXTURE / "simulate_after_guard.py").read_text(encoding="utf-8").splitlines()
                    if line.strip().startswith(f"{TARGET}=float(cell["))
AFTER = BEFORE.replace(READ_TEXT, REWRITE_LINE)


def _flagged_keys(code: str, design: dict[str, Any]) -> list[tuple[str, ...]]:
    return [r.keys for r in gk.problems({"simulate.py": code}, PROTOCOL, design)]


def test_the_evidence_is_what_the_run_log_says() -> None:
    log = (FIXTURE / "run_log_excerpt.txt").read_text(encoding="utf-8")
    assert f"reads the setting `{CONST}` or `{CONST_KEYS[1]}` (and gives 80.0 when it is not there)" in log
    assert "rewrote simulate.py to repair the fault" in log and f"with the declared {SWEPT} setting" in log
    assert re.search(r"declared .* of 80 ", PLAN_TEXT), "the plan itself states the constant"
    assert AFTER != BEFORE and REWRITE_LINE in AFTER


def test_a_constant_the_plan_defines_is_not_a_setting_the_study_lacks() -> None:
    assert _flagged_keys(BEFORE, DESIGN) == []


def test_without_the_plan_stating_the_value_the_same_read_is_still_a_fault() -> None:
    """The check was narrowed, not loosened: with no plan text (or a plan that does not say it) the read is flagged as it was."""
    assert _flagged_keys(BEFORE, {}) == [CONST_KEYS]
    other = {"_text": f"a reference value is chosen; {SWEPT} is swept over its grid"}
    assert _flagged_keys(BEFORE, other) == [CONST_KEYS]


@pytest.mark.parametrize("make", [
    lambda: f'cell.get("{CONST}", 75.0)',            # a number the plan does not give for the constant
    lambda: f'cell.get("{SWEPT[:-1]}", 80.0)',      # the swept setting misspelt: still a collapsed axis, whatever the plan says
    lambda: f'cell.get("{SWEPT}_nm", 80.0)',        # a longer form of the swept setting
    lambda: f'cell.get("{CONST}")',                  # no fallback: it cannot be answered
])
def test_a_fallback_that_is_not_the_plans_constant_is_still_flagged(make: Any) -> None:
    code = BEFORE.replace(READ_EXPR, f"float({make()})")
    assert _flagged_keys(code, DESIGN) != []


def test_a_repair_may_not_replace_the_plans_constant_by_the_swept_setting() -> None:
    assert gk.substituted({"simulate.py": BEFORE}, {"simulate.py": AFTER}, PROTOCOL, DESIGN) != []
    assert "replaced a constant the plan defines" in gk.rejects({"simulate.py": BEFORE}, {"simulate.py": AFTER}, PROTOCOL, DESIGN)
    # Keeping the constant (under its own name, with the plan's value) is not a substitution.
    kept = "PLAN_CONSTANT = 80.0\n" + BEFORE.replace(READ_EXPR, "PLAN_CONSTANT")
    assert gk.rejects({"simulate.py": BEFORE}, {"simulate.py": kept}, PROTOCOL, DESIGN) == ""
    # A real misspelt fallback that is mended by reading the swept setting is the repair that is wanted.
    broken = BEFORE.replace(READ_EXPR, f'float(cell.get("{SWEPT[:-1]}", 80.0))')
    assert gk.rejects({"simulate.py": broken}, {"simulate.py": AFTER}, PROTOCOL, DESIGN) == ""


class _Client:
    def __init__(self, reply: str) -> None:
        self.reply, self.asked = reply, []
        self.last_usage = None
        self.last_model = "test-model"

    async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
        self.asked.append(messages[-1]["content"])
        return self.reply


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str, reply: str) -> Engine:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    eng = Engine(Config(
        topic="a neutral topic", title="keys", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs")))
    eng._client = _Client(reply)  # type: ignore[assignment]
    eng._protocol_block = lambda state: PROTOCOL  # type: ignore[method-assign]
    (eng.quest_root / "plan.md").write_text(PLAN_TEXT, encoding="utf-8")
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "simulate.py").write_text(code, encoding="utf-8")
    return eng


@pytest.mark.asyncio
async def test_the_gate_no_longer_asks_for_a_repair_of_the_plans_constant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng = _engine(tmp_path, monkeypatch, BEFORE, json.dumps({"code": AFTER, "patch_summary": "the guard's own rewrite"}))
    assert await eng._static_gate({}) is None
    assert eng._client.asked == []  # type: ignore[attr-defined]
    assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == BEFORE


@pytest.mark.asyncio
async def test_a_repair_that_also_swaps_the_constant_for_the_swept_setting_is_not_kept(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real fault elsewhere in the file is still repaired; the same rewrite that also overwrites the plan's constant is refused."""
    spare = "    spare = 0.2  # a spare constant: the test turns this line into a bad read\n"
    assert spare in BEFORE
    flagged = BEFORE.replace(spare, '    spare = float(cell.get("unknown_setting_nm"))\n')
    assert 'cell.get("unknown_setting_nm")' in flagged
    fixed = flagged.replace('float(cell.get("unknown_setting_nm"))', "0.2")
    swapped = fixed.replace(READ_TEXT, REWRITE_LINE)
    eng = _engine(tmp_path, monkeypatch, flagged, json.dumps({"code": swapped, "patch_summary": "both"}))
    assert _flagged_keys(flagged, DESIGN) != [], "the other read is a fault"
    stop = await eng._static_gate({})
    assert stop is not None, "the swapped rewrite was refused, so the fault stands"
    assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == flagged
    eng2 = _engine(tmp_path / "ok", monkeypatch, flagged, json.dumps({"code": fixed, "patch_summary": "only the fault"}))
    assert await eng2._static_gate({}) is None
    assert (eng2.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == fixed


@pytest.mark.asyncio
async def test_the_repair_of_a_readers_finding_may_not_overwrite_the_plans_constant_either(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, this_python: None) -> None:
    """The second channel that rewrites code (a finding of the code reader, repaired after a failing test) is held to the
    same rule: the plan's constant is never replaced by the swept setting."""
    from core import code_review as cr
    from core import finding_repair as fr
    from tests.test_code_review import CHANGED_TEST

    eng = _engine(tmp_path, monkeypatch, BEFORE, "")
    line = next(n for n, text in enumerate(BEFORE.splitlines(), 1) if READ_TEXT in text)
    problems = cr.parse_logic({"findings": [{"where": f"simulate.py:{line}", "what": "the reference follows the wrong thing",
                                             "why": "w", "fix": "f", "severity": "high", "quantity": "reference",
                                             "changes_result": True}]}, [])
    record: dict[str, Any] = {"same_model": False, "reviewer": "other-model"}
    cr.note_reading(record, "v", problems, eng.quest_root / "code")
    # The reply is the function with the plan's constant read from the swept setting, as the guard's own rewrite did.
    swapped = "\n".join(BEFORE.splitlines()[3:]).replace(READ_TEXT, REWRITE_LINE) + "\n"
    assert REWRITE_LINE in swapped and READ_TEXT not in swapped

    class Swapping(_Client):
        async def chat(self, messages: list[dict[str, str]], **kw: Any) -> str:
            prompt = messages[-1]["content"]
            self.asked.append(prompt)
            if prompt.lstrip().startswith(fr.TEST_HEAD):
                return "```python\n" + CHANGED_TEST.format(rel="simulate.py") + "```"
            return "```python\n" + swapped + "```"

    eng._client = Swapping("")  # type: ignore[assignment]
    assert await eng._repair_review_findings({}, record, PROTOCOL, "code_review") is False
    assert (eng.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == BEFORE
    (issue,) = record["issues"].values()
    assert issue["demonstrated"] is True and issue["outcome"] == "not repaired"
    assert all(r["outcome"] == "rejected" and "replaced a constant the plan defines" in r["why"] for r in issue["repairs"])


_ALLOWED = set(PROTOCOL["grid"])
_WORDS = " ".join(gk._distinct_words(CONST, _ALLOWED))


@pytest.mark.parametrize("text", [
    f"the {_WORDS} is compared across conditions, with 80 trials per run",    # the number belongs to something else
    f"the {_WORDS} is compared across conditions and 80 trials are run",
    f"a {_WORDS} is chosen. 80 trials are run",                                # a sentence break between them
    f"the {_WORDS} is 75, the other setting is 80",                            # another number between / other value
    f"the {_WORDS} of 75 and 80 runs",                                         # another number in the clause
])
def test_a_number_that_merely_stands_near_the_words_is_not_the_plans_constant(text: str) -> None:
    assert _flagged_keys(BEFORE, {"_text": text}) == [CONST_KEYS]


@pytest.mark.parametrize("text", [
    f"a declared {_WORDS} of 80 is used",
    f"the {_WORDS} = 80",
    f"the {_WORDS} is 80",
    f"the {_WORDS}: 80",
    f"80 units {_WORDS}",
])
def test_a_number_stated_as_the_value_of_the_name_in_one_clause_is_the_plans_constant(text: str) -> None:
    assert _flagged_keys(BEFORE, {"_text": text}) == []


def test_a_structured_field_of_the_plan_that_gives_the_name_a_value_is_the_plans_constant() -> None:
    for protocol in ({**PROTOCOL, "fixed": {CONST: 80}}, {**PROTOCOL, "constants": {CONST.upper(): {"value": 80}}},
                     {**PROTOCOL, "thresholds": [{"name": CONST, "value": 80.0}]}):
        assert [r.keys for r in gk.problems({"simulate.py": BEFORE}, protocol, {})] == []
    other = {**PROTOCOL, "notes": {CONST: 75}}
    assert [r.keys for r in gk.problems({"simulate.py": BEFORE}, other, {})] == [CONST_KEYS]
    elsewhere = {**PROTOCOL, "constants": {"unrelated_setting": 80}}
    assert [r.keys for r in gk.problems({"simulate.py": BEFORE}, elsewhere, {})] == [CONST_KEYS]
