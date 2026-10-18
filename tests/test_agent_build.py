"""Agent mode (``engine.code_writer: agent``, core/agent_build.py): the coding agent writes the code, FI's checks judge it.

A stand-in agent (no model, no command-line tool) writes files in ``code/``; FI's checks before the run are stood in for by
a function that asks for a repair the way a real check does (``Engine._chat`` with a code-writing node), so the report mode
itself is what is tested. No model is called.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

import pytest

from core import agent_build as ab
from core import agent_coder as ac
from core import evidence
from core.agent_build import RepairForAgent
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine


def _cfg(tmp_path: Path, writer: str) -> Config:
    return Config(
        topic="a neutral topic about a cooling curve", title="agent-build", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off", oracle_check="off",
                            code_writer=writer),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
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
    line = "[oracle] the check 'c1' at pitch80 (E1) measured 0.61, expected 6.2e-1 within 0.01 after 3 runs"
    shown = ab.without_numbers(line)
    assert "0.61" not in shown and "6.2e-1" not in shown and "0.01" not in shown and " 3 " not in shown
    assert "pitch80" in shown and "E1" in shown and "c1" in shown


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

