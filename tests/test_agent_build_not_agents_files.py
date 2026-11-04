"""A finding about a file the coding agent may not edit is FI's own finding, not the agent's to fix (core/agent_build.py).

A real quest stopped with no paper because a check failed in every round on a file under ``code/skills/`` (FI's own copy
of a skill): the agent could never clear it, so its rounds "made no progress". The reused fakes are those of
tests/test_agent_build.py: a stand-in agent and a stand-in list of the checks' verdicts, no model.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from core import agent_build as ab
from core import agent_coder as ac
from core import code_checks
from tests.test_agent_build import _copies_in_the_test_folder  # noqa: F401 -- the autouse fixture of that file
from tests.test_agent_build import _Agent, _engine, _first_write, _gate_always_asks, _write

FI_FILE = "skills/plotting/palette.py"       # FI's copy of a skill: the agent may not edit it
OWN_FILE = "simulate.py"                      # the agent's own file


def _table(file_of_t: str, rounds: dict[int, dict[str, bool]]) -> Any:
    """Verdicts for the ``value = <n>`` the code holds: ``typed_constants`` carries the file it found its fault in."""
    async def check(root: Path, *_a: Any, **_k: Any) -> list[code_checks.Verdict]:
        text = (Path(root) / "code" / "experiment.py").read_text(encoding="utf-8")
        value = int(text.split("value = ")[1].split()[0])
        out = []
        for name, ok in rounds[value].items():
            details = {"functions": [{"script": file_of_t, "function": "f", "line": 3}]} if name == "typed_constants" else {}
            out.append(code_checks.Verdict(name, ok, f"{name} {'passed' if ok else 'found a fault'}", details))
        return out
    return check


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file_of_t: str, rounds: dict[int, dict[str, bool]],
         gate_passes_at: int | None) -> tuple[Any, _Agent, Any]:
    agent = _Agent([_write(v) for v in rounds])
    eng = _engine(tmp_path, agent=agent)
    monkeypatch.setattr("core.code_checks.check_code", _table(file_of_t, rounds))
    asks = _gate_always_asks(eng)

    async def gate(state: Any, py: Any, env: Any, seed: Path) -> tuple[Any, Any, bool]:
        value = int((eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8").split("value = ")[1].split()[0])
        if value == gate_passes_at:
            return None, None, False
        return await asks()
    monkeypatch.setattr(eng, "_gate_pass", gate)
    _first_write(eng, agent)
    out = asyncio.run(eng._agent_build({}, "python", {}, eng.quest_root / "code" / "experiment.py"))
    return eng, agent, out


ROUNDS = {1: {"equation_tests": False, "typed_constants": False},
          2: {"equation_tests": True, "typed_constants": False}}


def test_a_finding_in_a_file_the_agent_may_not_edit_is_not_sent_nor_counted_and_is_disclosed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Old code sent ``typed_constants`` to the agent every round and kept it among the failing checks (so it was in the
    record's checks as False and, once it was all that failed, counted as no progress). Now it is FI's own finding."""
    eng, agent, out = _run(tmp_path, monkeypatch, FI_FILE, {**ROUNDS, 3: {"equation_tests": True, "typed_constants": False}}, 3)
    assert out == (None, None, False), "the build went on to the run; it did not stop on FI's own finding"
    assert all("typed_constants found a fault" not in p for p in agent.prompts), "not sent to the agent"
    rounds = ac.AgentCoder(eng.quest_root, agent).load()["rounds"]
    assert all("typed_constants" not in (r.get("checks") or {}) for r in rounds)
    assert "typed_constants found a fault" in rounds[1]["fi_own_findings"][0] and FI_FILE in rounds[1]["fi_own_findings"][0]
    log = (eng.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "not the coding agent's to fix" in log


def test_a_finding_in_the_agents_own_file_is_still_sent_and_a_round_without_progress_still_stops(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Known positive: the same check failing in the agent's own simulate file counts as before."""
    same = {"equation_tests": True, "typed_constants": False}
    eng, agent, (stop, code, _r) = _run(tmp_path, monkeypatch, OWN_FILE, {1: same, 2: same}, None)
    assert code is None and "made no check of FI's pass that failed before" in stop["exec_result"]["cannot_repair"]
    assert any("typed_constants found a fault" in p for p in agent.prompts)
    rounds = ac.AgentCoder(eng.quest_root, agent).load()["rounds"]
    assert rounds[0]["checks"]["typed_constants"] is False and not rounds[0].get("fi_own_findings")


def test_only_findings_that_are_all_in_files_the_agent_may_not_edit_are_set_aside() -> None:
    """A check with one finding in a FI-owned file and one in the agent's file is the agent's."""
    details = {"functions": [{"script": FI_FILE, "line": 1}, {"script": OWN_FILE, "line": 2}]}
    assert ab.verdict_files(details) == {FI_FILE, OWN_FILE}
    assert ab.verdict_files({"settings": ["a"]}) == set(), "a verdict that names no file is the agent's"
