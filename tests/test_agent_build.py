"""Agent mode (``engine.code_writer: agent``, core/agent_build.py): the coding agent writes the code, FI's checks judge it.

A stand-in agent (no model, no command-line tool) writes files in the copy of ``code/`` it is given; FI's checks before the
run are stood in for by a function that asks for a repair the way a real check does (``Engine._chat`` with a code-writing
node), so the report mode itself is what is tested. No model is called.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import pytest

from core import agent_build as ab
from core import agent_coder as ac
from core import code_checks
from core import evidence
from core.agent_build import RepairForAgent
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine


@pytest.fixture(autouse=True)
def _copies_in_the_test_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent's copy of ``code/`` lives in the computer's temporary folder (:func:`core.agent_coder.default_workdir`):
    here, the test's own."""
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))


def _cfg(tmp_path: Path, writer: str) -> Config:
    return Config(
        topic="a neutral topic about a cooling curve", title="agent-build", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off", oracle_check="off",
                            code_writer=writer),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )


class _Agent:
    """A stand-in coding agent: round ``k`` runs ``steps[k]`` on the folder it is given and says ``says[k]``."""

    def __init__(self, steps: list[Callable[[Path], None]], says: list[str] | None = None) -> None:
        self.steps, self.says = steps, says or []
        self.prompts: list[str] = []

    async def run(self, prompt: str, *, cwd: Path, session_id: str | None, timeout_s: float) -> ac.RunnerOutput:
        k = len(self.prompts)
        self.prompts.append(prompt)
        if k < len(self.steps):
            self.steps[k](Path(cwd))
        message = self.says[k] if k < len(self.says) else "done: every check passes"
        return ac.RunnerOutput(exit_code=0, message=message, session_id="s1", duration_s=1.0)


def _write(value: int) -> Callable[[Path], None]:
    def step(code: Path) -> None:
        code.mkdir(parents=True, exist_ok=True)
        (code / "experiment.py").write_text(f"value = {value}\nprint('RESULT_JSON: {{}}')\n", encoding="utf-8")
    return step


def _engine(tmp_path: Path, writer: str = "agent", agent: _Agent | None = None) -> Engine:
    eng = Engine(_cfg(tmp_path, writer))
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    if agent is not None:
        eng._agent_runner = agent  # type: ignore[attr-defined]
    return eng


def _no_code_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    async def none(*_a: Any, **_k: Any) -> list[Any]:
        return []
    monkeypatch.setattr("core.code_checks.check_code", none)


def _gate_wanting(eng: Engine, value_needed: int | Callable[[int], str | None]) -> Callable[..., Any]:
    """A stand-in for the checks before the run: a known-answer check that fails unless ``value = <value_needed>`` (or, for
    a callable, the node the check for the current value asks to repair). It logs what a real check logs (an expected
    value included) and then asks for a repair as a real one does."""
    async def gate(state: Any, py: Any, exec_env: Any, seed_path: Path) -> tuple[Any, Any, bool]:
        text = (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")
        value = int(text.split("value = ")[1].split()[0])
        node = (value_needed(value) if callable(value_needed)
                else (None if value == value_needed else "implement_oracle"))
        if node is None:
            return None, None, False
        eng._log.warning("[oracle] the check 'c1' measured %s, expected 0.62 within 0.01", value)
        await eng._chat("repair the script so the check passes; expected 0.62", node=node)
        raise AssertionError("the chat model was asked to repair the agent's code")
    return gate


# --- report mode ----------------------------------------------------------------------------------------------------------


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def chat(self, messages: list[dict[str, str]], **_k: Any) -> str:
        self.calls.append(messages[-1]["content"])
        return "ok"


def _with_client(eng: Engine, monkeypatch: pytest.MonkeyPatch) -> _Client:
    client = _Client()
    eng._client = client  # type: ignore[assignment]

    async def client_for(_node: Any) -> tuple[Any, Any]:
        return client, None

    async def recorded(_node: str, _messages: Any, call: Callable[[], Any]) -> tuple[str, dict[str, Any]]:
        return await call(), {}

    monkeypatch.setattr(eng, "_client_for", client_for)
    monkeypatch.setattr(eng, "_recorded_call", recorded)
    monkeypatch.setattr(eng, "_log_chat_cost", lambda **_k: None)
    return client


def test_in_agent_mode_a_request_to_repair_code_is_a_report_and_asks_no_model(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    eng = _engine(tmp_path)
    client = _with_client(eng, monkeypatch)
    for node in ("implement_oracle", "implement_protocol", "implement_seed", "implement_figures", "execute_reflect",
                 "implement", "implement.finding_repair", "code_review.repair"):
        with pytest.raises(RepairForAgent) as asked:
            asyncio.run(eng._chat("repair it", node=node))
        assert asked.value.node == node and asked.value.what
    assert client.calls == [], "no model was asked to repair the agent's code"
    # A request that writes no code (a reader, the plan, the outline, a speed-up) still goes to the model.
    for node in ("code_review", "design", "implement_outline", "implement.finding_test", "improve"):
        assert asyncio.run(eng._chat("read it", node=node)) == "ok"
    assert len(client.calls) == 5


def test_in_chat_mode_every_request_goes_to_the_model_as_before(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    eng = _engine(tmp_path, writer="chat")
    client = _with_client(eng, monkeypatch)
    assert asyncio.run(eng._chat("repair it", node="implement_oracle")) == "ok"
    assert client.calls == ["repair it"]
    assert "coding agent" not in str(eng._improve_skip({}))


def test_a_repair_is_not_taken_for_a_failed_model_call() -> None:
    """The checks keep ``except Exception`` for a model call that failed; the report passes through it."""
    async def site() -> str:
        try:
            raise RepairForAgent("implement_oracle")
        except Exception:  # noqa: BLE001 -- what a repair site does with a failed call
            return "swallowed"
    with pytest.raises(RepairForAgent):
        asyncio.run(site())


def test_numbers_are_taken_out_of_what_the_agent_is_shown_and_names_are_kept() -> None:
    line = "[oracle] the check 'c1' at size80 (E1) measured 0.61, expected 6.2e-1 within 0.01 after 3 runs"
    shown = ab.without_numbers(line)
    assert "0.61" not in shown and "6.2e-1" not in shown and "0.01" not in shown and " 3 " not in shown
    assert "size80" in shown and "E1" in shown and "c1" in shown


# --- the loop before the run --------------------------------------------------------------------------------------------


def _first_write(eng: Engine, agent: _Agent) -> None:
    coder = asyncio.run(eng._code_agent())
    assert asyncio.run(coder.write("write the study")).ok
    assert agent.prompts == ["write the study"]


def test_the_agent_fixes_what_a_check_found_and_the_checks_then_pass(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    agent = _Agent([_write(1), _write(2)], says=["done: every check passes", "fixed it"])
    eng = _engine(tmp_path, agent=agent)
    _no_code_checks(monkeypatch)
    monkeypatch.setattr(eng, "_gate_pass", _gate_wanting(eng, 2))
    _first_write(eng, agent)
    out = asyncio.run(eng._agent_build({}, "python", {}, eng.quest_root / "code" / "experiment.py"))
    assert out == (None, None, False), "the code passed FI's checks after one fix"
    assert len(agent.prompts) == 2
    sent = agent.prompts[1]
    assert "known-answer check" in sent and "the check 'c1' measured …, expected … within …" in sent
    assert "0.62" not in sent, "a value a check compares with never reaches the agent"
    record = ac.AgentCoder(eng.quest_root, agent).load()
    first, second = record["rounds"]
    assert first["claimed"] == "done: every check passes" and first["fi_passed"] is False and first["found"]
    assert second["fi_passed"] is True and second["found"] == []


def test_fi_overrules_an_agent_that_says_it_is_done_and_stops_when_a_round_changes_nothing(tmp_path: Path,
                                                                                         monkeypatch) -> None:  # noqa: ANN001
    agent = _Agent([_write(1), lambda _code: None], says=["done: every check passes", "done: every check passes"])
    eng = _engine(tmp_path, agent=agent)
    _no_code_checks(monkeypatch)
    monkeypatch.setattr(eng, "_gate_pass", _gate_wanting(eng, 2))
    _first_write(eng, agent)
    stop, code, repaired = asyncio.run(eng._agent_build({}, "python", {}, eng.quest_root / "code" / "experiment.py"))
    assert code is None and repaired is False
    said = stop["exec_result"]["cannot_repair"]
    assert "made no check of FI's pass that failed before" in said and "known-answer check" in said
    assert stop["exec_give_up_reason"] == said
    assert len(agent.prompts) == 2, "one fix round, then the stop: the agent's 'done' counted for nothing"


def test_the_fix_rounds_before_the_run_are_bounded(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    # Each round fixes one check and breaks another, so every round makes progress and only the budget ends the loop.
    nodes = {1: "implement_oracle", 2: "implement_protocol", 3: "implement_seed", 4: "implement_figures",
             5: "implement_oracle"}
    agent = _Agent([_write(1), _write(2), _write(3), _write(4), _write(5)])
    eng = _engine(tmp_path, agent=agent)
    _no_code_checks(monkeypatch)
    monkeypatch.setattr(eng, "_gate_pass", _gate_wanting(eng, lambda v: nodes.get(v)))
    _first_write(eng, agent)
    stop, _code, _repaired = asyncio.run(eng._agent_build({}, "python", {}, eng.quest_root / "code" / "experiment.py"))
    assert f"its {ab.FIX_ROUNDS} round(s) of fixes before the run are used" in stop["exec_result"]["cannot_repair"]
    assert len(agent.prompts) == 1 + ab.FIX_ROUNDS
    assert asyncio.run(eng._code_agent()).fixes_left() == ab.AFTER_FREEZE_ROUNDS, "one round is kept for after the freeze"


def test_a_stop_that_is_not_the_codes_is_returned_as_it_is(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    agent = _Agent([_write(1)])
    eng = _engine(tmp_path, agent=agent)
    plan_stop = eng._stop_before_run("the plan gives no value for the check", "cannot_repair")

    async def gate(*_a: Any) -> tuple[Any, Any, bool]:
        return plan_stop, None, False

    monkeypatch.setattr(eng, "_gate_pass", gate)
    _first_write(eng, agent)
    assert asyncio.run(eng._agent_build({}, "python", {}, eng.quest_root / "code" / "experiment.py")) == (
        plan_stop, None, False)
    assert len(agent.prompts) == 1, "the agent is not asked to fix what is the plan's"


def test_a_failed_run_of_the_checks_goes_to_the_agent_with_its_traceback(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    agent = _Agent([_write(1), _write(2)])
    eng = _engine(tmp_path, agent=agent)
    _no_code_checks(monkeypatch)

    async def gate(*_a: Any) -> tuple[Any, Any, bool]:
        text = (eng.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")
        if "value = 2" in text:
            return None, None, False
        return eng._code_review_failed_run("Traceback: NameError: name 'rate' is not defined"), None, False

    monkeypatch.setattr(eng, "_gate_pass", gate)
    _first_write(eng, agent)
    assert asyncio.run(eng._agent_build({}, "python", {}, eng.quest_root / "code" / "experiment.py")) == (None, None, False)
    assert "NameError: name 'rate' is not defined" in agent.prompts[1]


# --- the implement step, the run step and the repair step ---------------------------------------------------------------


def test_the_agents_task_holds_no_expected_value_nor_the_quest_folder_and_keeps_the_outlines_names(
        tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    agent = _Agent([_write(1)], says=["wrote experiment.py\nDEPS: scipy"])
    eng = _engine(tmp_path, agent=agent)

    async def no_project(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(eng, "_refresh_code_project", no_project)
    design = {"hypothesis": "h", "protocol": {
        "grid": {"n": [1, 2]},
        "oracles": [{"name": "c1", "expected": 0.731259, "tolerance": 0.001, "case": {"n": 1}}]}}
    outline = {"functions": [{"name": "cooling_rate", "params": ["k", "t"]}]}
    out = asyncio.run(eng._implement_by_agent({"design": design}, outline, []))
    task = agent.prompts[0]
    assert "0.731259" not in task, "a check's expected value is not in the agent's task"
    assert str(eng.quest_root) not in task and eng.quest_root.name not in task
    assert "cooling_rate" in task and "The functions FI tests by name" in task
    assert out["code"].startswith("value = 1") and "scipy" in out["deps"]


def test_a_check_after_the_checks_before_the_run_sends_a_failed_run_to_the_repair_step(tmp_path: Path,
                                                                                    monkeypatch) -> None:  # noqa: ANN001
    eng = _engine(tmp_path)

    async def steps(_state: Any) -> Any:
        eng._log.warning("[figures] figure 2 draws numbers typed into the code (0.5, 0.25)")
        raise RepairForAgent("implement_figures")

    monkeypatch.setattr(eng, "_node_execute_steps", steps)
    out = asyncio.run(eng._node_execute({}))
    result = out["exec_result"]
    assert result["returncode"] == 1 and result["directive"] is True
    assert "each figure is drawn from computed numbers" in result["stderr_tail"]
    assert "0.25" not in result["stderr_tail"] and "figure 2 draws numbers typed into the code" not in result["stderr_tail"]
    assert "figure … draws numbers typed into the code (…, …)" in result["stderr_tail"]


def test_after_the_freeze_the_agent_gets_one_round_then_the_failure_stands(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    agent = _Agent([_write(1), _write(2), _write(3)])
    eng = _engine(tmp_path, agent=agent)
    _first_write(eng, agent)
    monkeypatch.setattr(ab._frozen, "load", lambda _root: {"frozen": True})
    failed = {"returncode": 1, "stderr_tail": "ZeroDivisionError", "stdout_tail": ""}
    patch = asyncio.run(eng._reflect_by_agent({"exec_reflect_history": []}, 0, 1, "1", "", failed, {}))
    assert patch["exec_patch_pending"] is True and patch["exec_reflect_iter"] == 1 and patch["code"].startswith("value = 2")
    assert "ZeroDivisionError" in agent.prompts[1]
    again = asyncio.run(eng._reflect_by_agent({"exec_reflect_history": []}, 1, 1, "1", "", failed, {}))
    assert "round after the freeze is used" in again["exec_give_up_reason"]
    assert len(agent.prompts) == 2


def test_the_improve_loop_does_not_run_on_the_agents_code(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    assert eng._improve_skip({}) == "the coding agent wrote this code, and FI does not change it with a chat model"


def test_the_evidence_says_the_code_writer_ran_without_isolation(tmp_path: Path) -> None:
    eng = _engine(tmp_path, agent=_Agent([_write(1)]))
    _first_write(eng, eng._agent_runner)  # type: ignore[attr-defined]
    settings = ac.evidence_settings(eng.quest_root)
    assert settings["code_writer_isolation"] == ac.ISOLATION_UNVERIFIED
    record = evidence.assess(eng.quest_root, {}, settings=settings)
    assert record["code_writer"]["isolation"] == ac.ISOLATION_UNVERIFIED
    assert "plan.md" in record["code_writer"]["why"]
    assert any("the code writer ran without isolation" in g for g in record["all_gaps"]["publication_ready"])


def test_a_chat_mode_quest_has_no_code_writer_in_its_evidence(tmp_path: Path) -> None:
    eng = _engine(tmp_path, writer="chat")
    assert "code_writer" not in evidence.assess(eng.quest_root, {}, settings=ac.evidence_settings(eng.quest_root))


# --- a round is judged check by check against the best earlier round ----------------------------------------------------


def _verdicts_by_value(table: dict[int, dict[str, bool]]) -> Callable[..., Any]:
    """A stand-in for FI's list of the code's faults: the verdicts for the ``value = <n>`` the code holds now."""
    async def check(root: Path, *_a: Any, **_k: Any) -> list[code_checks.Verdict]:
        text = (Path(root) / "code" / "experiment.py").read_text(encoding="utf-8")
        value = int(text.split("value = ")[1].split()[0])
        return [code_checks.Verdict(name, ok, f"{name} {'passed' if ok else 'found a fault'}")
                for name, ok in table[value].items()]
    return check


def _gate_always_asks(eng: Engine) -> Callable[..., Any]:
    async def gate(*_a: Any) -> tuple[Any, Any, bool]:
        await eng._chat("repair the script so the check passes", node="implement_oracle")
        raise AssertionError("the chat model was asked to repair the agent's code")
    return gate


def _run_rounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, table: dict[int, dict[str, bool]],
                values: list[int]) -> tuple[Engine, _Agent, Any]:
    agent = _Agent([_write(v) for v in values])
    eng = _engine(tmp_path, agent=agent)
    monkeypatch.setattr("core.code_checks.check_code", _verdicts_by_value(table))
    monkeypatch.setattr(eng, "_gate_pass", _gate_always_asks(eng))
    _first_write(eng, agent)
    out = asyncio.run(eng._agent_build({}, "python", {}, eng.quest_root / "code" / "experiment.py"))
    return eng, agent, out


# The e2 replay's shape: round 1 has the equations right and every trial failing; round 2 fixes the trials and breaks
# an equation.
FIRST = {"equation_tests": True, "trial_checks": False, "known_answers": False}
BROKE_AN_EQUATION = {"equation_tests": False, "trial_checks": True, "known_answers": False}


def test_a_round_that_breaks_a_check_that_passed_is_not_progress_and_the_break_is_named(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, agent, (stop, code, _r) = _run_rounds(tmp_path, monkeypatch, {1: FIRST, 2: BROKE_AN_EQUATION, 3: BROKE_AN_EQUATION},
                                              [1, 2, 3])
    rounds = ac.AgentCoder(eng.quest_root, agent).load()["rounds"]
    first, second, third = rounds
    assert "equation_tests" in first["passed"] and first["newly_passed"] == [] and first["newly_failed"] == []
    assert second["compared_with"] == 1
    assert second["newly_passed"] == ["trial_checks"] and second["newly_failed"] == ["equation_tests"]
    assert "trial_checks" in second["passed"] and "equation_tests" not in second["passed"]
    assert third["newly_passed"] == [] and third["newly_failed"] == []
    broke = "the test of each equation against the plan's worked example"
    log = (eng.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert any("[WARNING]" in line and "broke what passed before" in line and broke in line
               for line in log.splitlines()), "the run log names the check that broke"
    told = agent.prompts[2]
    assert told.index("Your last round broke what passed before (" + broke + ")") < told.index("known_answers")
    assert "broke what passed before" not in agent.prompts[1], "round 1 broke nothing"
    said = stop["exec_result"]["cannot_repair"]
    assert "made no check of FI's pass that failed before" in said
    assert "(passed in an earlier round and fails now: " + broke + ")" in said
    assert code is None and len(agent.prompts) == 3


def test_a_round_that_puts_back_what_it_broke_is_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    put_back = {"equation_tests": True, "trial_checks": True, "known_answers": False}
    every = {"equation_tests": True, "trial_checks": True, "known_answers": True}
    eng, agent, (stop, _code, _r) = _run_rounds(tmp_path, monkeypatch, {1: FIRST, 2: BROKE_AN_EQUATION, 3: put_back, 4: every},
                                               [1, 2, 3, 4])
    rounds = ac.AgentCoder(eng.quest_root, agent).load()["rounds"]
    assert rounds[2]["newly_passed"] == ["equation_tests"] and rounds[2]["newly_failed"] == []
    assert rounds[3]["newly_passed"] == ["known_answers"]
    assert f"its {ab.FIX_ROUNDS} round(s) of fixes before the run are used" in stop["exec_result"]["cannot_repair"]


def test_a_break_followed_by_another_round_that_is_not_progress_stops(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    broke_the_trials = {"equation_tests": True, "trial_checks": False, "known_answers": True}
    eng, agent, (stop, _code, _r) = _run_rounds(tmp_path, monkeypatch, {1: FIRST, 2: BROKE_AN_EQUATION, 3: broke_the_trials},
                                               [1, 2, 3, 3])
    said = stop["exec_result"]["cannot_repair"]
    assert "its round to put back what it broke was not progress either" in said
    assert len(agent.prompts) == 3


def test_the_best_earlier_round_has_the_fewest_failing_checks_and_the_gate_is_passed_by_getting_past_it() -> None:
    earlier = [(1, {"a": True, "b": False, "c": False}), (2, {"a": False, "b": True, "c": True}),
               (3, {"implement_oracle": False, "a": False, "b": False, "c": False})]
    got = ab.compare_rounds(earlier, {"implement_seed": False, "a": True, "b": True, "c": False})
    assert got["best"] == 2 and got["newly_passed"] == ["a"] and got["newly_failed"] == ["c"]
    assert got["still_broken"] == ["c"]
    gate = ab.compare_rounds([(1, {"implement_oracle": False, "a": None})], {"implement_seed": False, "a": None})
    assert gate["newly_passed"] == ["implement_oracle"] and gate["newly_failed"] == []
    assert ab.compare_rounds([], {"a": False}) == {"best": None, "newly_passed": [], "newly_failed": [], "still_broken": []}


# --- what the agent can see -------------------------------------------------------------------------------------------------


PLANTED = "aligns the noiseless-rate recovery check | noiseless_rate_accuracy: 3.747e-15 (not met, not compared)"


def _git(code: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(code), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          capture_output=True, text=True, check=True).stdout


def test_the_agents_folder_holds_no_history_of_fis_commits_and_fi_keeps_its_own(tmp_path: Path,
                                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def looks_then_writes(value: int) -> Callable[[Path], None]:
        def step(folder: Path) -> None:
            text = " ".join(p.read_text(encoding="utf-8", errors="replace") for p in folder.rglob("*") if p.is_file())
            seen.append({"folder": folder, "git": [p for p in folder.rglob("*") if p.name == ".git"],
                         "named": [w for w in ("noiseless_rate_accuracy", "3.747e-15")
                                   if w in text]})
            _write(value)(folder)
        return step

    agent = _Agent([looks_then_writes(1), looks_then_writes(2)])
    eng = _engine(tmp_path, agent=agent)
    code = eng.quest_root / "code"
    (code / "notes.txt").write_text("FI's earlier version\n", encoding="utf-8")
    _git(code, "init", "-q")
    _git(code, "add", "-A")
    _git(code, "commit", "-q", "-m", PLANTED)
    design = {"hypothesis": "h", "protocol": {"grid": {"n": [1, 2]}}}
    asyncio.run(eng._implement_by_agent({"design": design}, {}, []))
    assert "code written by the coding agent" in _git(code, "log", "--format=%s"), "FI's own commit after the write"
    _no_code_checks(monkeypatch)
    monkeypatch.setattr(eng, "_gate_pass", _gate_wanting(eng, 2))
    assert asyncio.run(eng._agent_build({}, "python", {}, code / "experiment.py")) == (None, None, False)
    assert len(seen) == 2
    for round_ in seen:
        assert round_["git"] == [] and round_["named"] == [], round_
        assert eng.quest_root.resolve() not in round_["folder"].resolve().parents and round_["folder"] != code
    assert PLANTED in _git(code, "log", "--format=%B"), "FI's history of code/ is kept"
    assert (code / "experiment.py").read_text(encoding="utf-8").startswith("value = 2")


# A tiny quest whose plan holds values the agent must never see: the worked examples' outputs (5.0 and 12.0, which FI
# computes from the formulas), a known-answer check's expected value (8.0) and two tolerances.
HIDDEN = ("5.0", "12.0", "8.0", "0.000731", "0.000617")
SPRING = {
    "grid": {"c": [0.5, 1.0], "extra": [1, 2]},
    "model": {"summary": "a mass on a damped spring", "equations": [
        {"id": "E1", "formula": "w = sqrt(k / m)", "role": "generates", "source": "derivation", "derivation": "Newton's law",
         "example": {"inputs": {"k": 50.0, "m": 2.0}, "expected_formula": "sqrt(k / m)", "tolerance": 0.000617}},
        {"id": "E2", "formula": "Q = m * w / c", "role": "generates", "source": "derivation", "derivation": "energy ratio",
         "example": {"inputs": {"m": 2.0, "w": 3.0, "c": 0.5}, "expected_formula": "m * w / c"}},
    ]},
    "oracles": [{"name": "quality at one damping", "kind": "closed_form", "check": "Q = m * w / c at c = 0.75",
                 "expected": 8.0, "tolerance": 0.000731, "case": {"c": 0.75, "extra": 1}, "measure": "q"}],
}
WRONG = {
    "spring/__init__.py": "from . import model\n",
    "spring/model.py": "import math\n\n\n# E1\ndef frequency(k, m):\n    return math.sqrt(k * m)\n\n\n"
                       "# E2\ndef quality(m, w, c):\n    return m * w / c\n",
    "simulate.py": "from spring import model\n\n\ndef run_cell(cell):\n"
                   "    return {\"q\": model.quality(2.0, 3.0, cell[\"c\"] * 2) * cell[\"extra\"]}\n",
}


def test_no_expected_value_or_tolerance_of_the_plan_reaches_the_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.execution import SharedInterpreterExecutor

    def writes_the_wrong_code(folder: Path) -> None:
        for rel, text in WRONG.items():
            (folder / rel).parent.mkdir(parents=True, exist_ok=True)
            (folder / rel).write_text(text, encoding="utf-8")
        _write(1)(folder)

    agent = _Agent([writes_the_wrong_code, writes_the_wrong_code, writes_the_wrong_code])
    eng = _engine(tmp_path, agent=agent)
    eng.executor = SharedInterpreterExecutor(python_version="3.11")

    async def no_project(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(eng, "_refresh_code_project", no_project)
    state = {"design": {"hypothesis": "h", "protocol": SPRING}}

    async def gate(*_a: Any) -> tuple[Any, Any, bool]:
        # What the real known-answer check logs before it asks for a repair, its expected value and tolerance included.
        eng._log.warning("[oracle] the check 'quality at one damping' measured 4.0, expected 8.0 within 0.000731")
        eng._log.warning("[equations] E1 (frequency) gave 10.0 where the example gives 5.0 within 0.000617")
        await eng._chat("repair the script; expected 8.0 within 0.000731", node="implement_oracle")
        raise AssertionError("the chat model was asked to repair the agent's code")

    monkeypatch.setattr(eng, "_gate_pass", gate)
    asyncio.run(eng._implement_by_agent(state, {}, []))
    stop, _code, _r = asyncio.run(eng._agent_build(state, sys.executable, None, eng.quest_root / "code" / "experiment.py"))
    assert stop is not None and len(agent.prompts) >= 2, "the checks found the faults and the agent was told"
    feedback = "\n".join(agent.prompts[1:])
    assert "equation E1" in feedback and "quality at one damping" in feedback, "the real verdicts reached the agent"
    for k, prompt in enumerate(agent.prompts):
        for value in HIDDEN:
            assert not re.search(rf"(?<![\d.]){re.escape(value)}(?!\d)", prompt), (k, value, prompt[-1500:])
        assert not re.search(r"7\.31e-0?4|6\.17e-0?4", prompt), (k, prompt[-1500:])


def test_a_round_that_wrote_no_script_is_sent_back_not_a_crash(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """A real first round ran out of its 20 minutes before writing any script, and FI's checks then read the missing
    file and stopped the quest on an error."""
    agent = _Agent([lambda code: code.mkdir(parents=True, exist_ok=True), _write(2)])
    eng = _engine(tmp_path, agent=agent)
    _no_code_checks(monkeypatch)
    monkeypatch.setattr(eng, "_gate_pass", _gate_wanting(eng, 2))
    coder = asyncio.run(eng._code_agent())
    asyncio.run(coder.write("write the study"))
    assert not (eng.quest_root / "code" / "experiment.py").exists()
    out = asyncio.run(eng._agent_build({}, "python", {}, eng.quest_root / "code" / "experiment.py"))
    assert out == (None, None, False), "the next round wrote the script and the checks passed"
    assert "There is no `experiment.py`" in agent.prompts[1]


def test_a_study_with_its_simulation_apart_sends_back_one_script_until_simulate_py_defines_the_function(
        tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """A real round wrote one script for a study that keeps its simulation apart from the analysis; FI accepted it and
    ran it as it was, so no check that needs FI to call the simulation itself could run."""

    def two_scripts(code: Path) -> None:
        _write(2)(code)
        (code / "simulate.py").write_text("def run_cell(cell):\n    return {'y': 1.0}\n", encoding="utf-8")

    agent = _Agent([_write(2), two_scripts])
    eng = _engine(tmp_path, agent=agent)
    monkeypatch.setattr(eng, "_split_on", lambda _state: True)
    _no_code_checks(monkeypatch)
    monkeypatch.setattr(eng, "_gate_pass", _gate_wanting(eng, 2))
    coder = asyncio.run(eng._code_agent())
    asyncio.run(coder.write("write the study"))
    out = asyncio.run(eng._agent_build({}, "python", {}, eng.quest_root / "code" / "experiment.py"))
    assert out == (None, None, False), "the next round moved the simulation into simulate.py and the checks passed"
    assert "`simulate.py` is missing" in agent.prompts[1] and "run_cell(cell)" in agent.prompts[1]
    assert ab.check_words("two_scripts") != "two_scripts"


def _write_grid(values: str) -> Callable[[Path], None]:
    def step(code: Path) -> None:
        _write(2)(code)
        text = (code / "experiment.py").read_text(encoding="utf-8")
        (code / "experiment.py").write_text(f"C_VALUES = {values}\n" + text, encoding="utf-8")
    return step


def test_the_agents_code_is_held_to_the_plans_protocol_and_sent_back_when_it_differs(tmp_path: Path,
                                                                                     monkeypatch) -> None:  # noqa: ANN001
    """The agent's code was never compared with the plan's protocol, so every agent quest showed the protocol check as
    not run and none could show that its script follows the plan."""
    agent = _Agent([_write_grid("[1.0, 3.0]"), _write_grid("[1.0, 2.0]")])
    eng = _engine(tmp_path, agent=agent)
    monkeypatch.setattr(eng, "_protocol_of", lambda _state: {"grid": {"c": [1.0, 2.0]}})
    _no_code_checks(monkeypatch)
    monkeypatch.setattr(eng, "_gate_pass", _gate_wanting(eng, 2))
    coder = asyncio.run(eng._code_agent())
    asyncio.run(coder.write("write the study"))
    out = asyncio.run(eng._agent_build({}, "python", {}, eng.quest_root / "code" / "experiment.py"))
    assert out == (None, None, False), "the next round used the plan's settings and the checks passed"
    assert len(agent.prompts) == 2 and "differs from the plan's protocol" in agent.prompts[1]
    record = json.loads((eng.quest_root / "needs" / "PROTOCOL_CHECK.json").read_text(encoding="utf-8"))
    assert record["status"] == "ok" and record["differences"] == []
    assert ab.check_words("protocol") != "protocol"


def test_with_the_protocol_check_set_to_warn_a_difference_is_recorded_and_the_checks_go_on(tmp_path: Path,
                                                                                           monkeypatch) -> None:  # noqa: ANN001
    agent = _Agent([_write_grid("[1.0, 3.0]")])
    eng = _engine(tmp_path, agent=agent)
    eng.config.engine.protocol_check = "warn"
    monkeypatch.setattr(eng, "_protocol_of", lambda _state: {"grid": {"c": [1.0, 2.0]}})
    _no_code_checks(monkeypatch)
    monkeypatch.setattr(eng, "_gate_pass", _gate_wanting(eng, 2))
    coder = asyncio.run(eng._code_agent())
    asyncio.run(coder.write("write the study"))
    out = asyncio.run(eng._agent_build({}, "python", {}, eng.quest_root / "code" / "experiment.py"))
    assert out == (None, None, False) and len(agent.prompts) == 1
    record = json.loads((eng.quest_root / "needs" / "PROTOCOL_CHECK.json").read_text(encoding="utf-8"))
    assert record["status"] == "warned" and record["differences"]


def test_a_round_fi_did_not_keep_for_reaching_outside_its_folder_is_told_why(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    """A live round called a script in the home folder; FI kept nothing of it and told the agent only that no script was
    there, so the next round wrote the same code and the quest stopped."""

    def outside(code: Path) -> None:
        _write(2)(code)
        text = (code / "experiment.py").read_text(encoding="utf-8")
        (code / "experiment.py").write_text("import os\nTOOL = os.path.expanduser('~')\n" + text, encoding="utf-8")

    agent = _Agent([outside, _write(2)])
    eng = _engine(tmp_path, agent=agent)
    _no_code_checks(monkeypatch)
    monkeypatch.setattr(eng, "_gate_pass", _gate_wanting(eng, 2))
    coder = asyncio.run(eng._code_agent())
    asyncio.run(coder.write("write the study"))
    assert coder.last_cheats() and "expanduser" in coder.last_cheats()[0]
    out = asyncio.run(eng._agent_build({}, "python", {}, eng.quest_root / "code" / "experiment.py"))
    assert out == (None, None, False)
    assert "FI did not keep your last round, because its code reaches outside its folder" in agent.prompts[1]
    assert "expanduser" in agent.prompts[1]


def test_what_a_readers_extension_added_beyond_the_protocol_is_not_sent_back_to_the_agent(tmp_path: Path,
                                                                                          monkeypatch) -> None:  # noqa: ANN001
    """A reader asked for one more setting; the agent added it, and the checks before the run told the agent the code
    differed from the plan, so it was asked to take out what the person asked for."""
    agent = _Agent([_write_grid("[1.0, 2.0, 3.0]")])
    eng = _engine(tmp_path, agent=agent)
    monkeypatch.setattr(eng, "_protocol_of", lambda _state: {"grid": {"c": [1.0, 2.0]}})
    monkeypatch.setattr(eng, "_skill_files_to_code", lambda _state: [])

    async def no_refresh(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        return None

    monkeypatch.setattr(eng, "_refresh_code_project", no_refresh)
    _no_code_checks(monkeypatch)
    monkeypatch.setattr(eng, "_gate_pass", _gate_wanting(eng, 2))
    asyncio.run(eng._implement_by_agent({}, {}, ["the result at c = 3"]))
    record = json.loads((eng.quest_root / "needs" / "PROTOCOL_CHECK.json").read_text(encoding="utf-8"))
    assert record["status"] == "extended_by_person" and record["differences"]
    out = asyncio.run(eng._agent_build({}, "python", {}, eng.quest_root / "code" / "experiment.py"))
    assert out == (None, None, False) and len(agent.prompts) == 1, "nothing was sent back for the person's setting"
    record = json.loads((eng.quest_root / "needs" / "PROTOCOL_CHECK.json").read_text(encoding="utf-8"))
    assert record["status"] == "extended_by_person"


def test_code_written_again_without_an_extension_is_held_to_the_whole_protocol_again(tmp_path: Path,
                                                                                    monkeypatch) -> None:  # noqa: ANN001
    """An earlier extension's record kept excusing its setting after the code was written again without one (a rewrite
    after the evidence gate): the chat writer's code is held to the frozen protocol then, and so is the agent's."""
    agent = _Agent([_write_grid("[1.0, 2.0, 3.0]"), _write_grid("[1.0, 2.0, 3.0]"), _write_grid("[1.0, 2.0]")])
    eng = _engine(tmp_path, agent=agent)
    monkeypatch.setattr(eng, "_protocol_of", lambda _state: {"grid": {"c": [1.0, 2.0]}})
    monkeypatch.setattr(eng, "_skill_files_to_code", lambda _state: [])

    async def no_refresh(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        return None

    monkeypatch.setattr(eng, "_refresh_code_project", no_refresh)
    _no_code_checks(monkeypatch)
    monkeypatch.setattr(eng, "_gate_pass", _gate_wanting(eng, 2))
    asyncio.run(eng._implement_by_agent({}, {}, ["the result at c = 3"]))
    asyncio.run(eng._implement_by_agent({}, {}, []))
    assert eng._goes_beyond_protocol_by_request() == []
    asyncio.run(eng._agent_build({}, "python", {}, eng.quest_root / "code" / "experiment.py"))
    assert len(agent.prompts) == 3 and "differs from the plan's protocol" in agent.prompts[2]
