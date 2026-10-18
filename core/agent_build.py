"""Agent mode (``engine.code_writer: agent``): one coding agent writes the whole program, and FI's own checks judge it.

In chat mode (the default) a chat model writes the code and every check of the code repairs what it finds by asking that
model again. In agent mode the code is the agent's (:mod:`core.agent_coder`, the Claude command-line tool on a copy of
``code/``), and FI's checks only judge it:

* :meth:`AgentBuildMixin._implement_by_agent` replaces the code-writing chat of the implement step: the agent writes the
  code from the design as a code writer may see it (no expected value of a check, no worked example's answer, never the
  quest's folder), keeping the function names of the outline the equation tests call;
* :meth:`AgentBuildMixin._agent_build` replaces the one call of the checks before the run (``Engine._gate_pass``): the
  checks run with every repair turned into a report (:class:`RepairForAgent`), and what they found goes back to the same
  agent session, at most :data:`FIX_ROUNDS` times, :data:`ROUND_S` seconds a round and :data:`TOTAL_S` seconds in all.
  The loop ends early when a round makes no check newly pass. The agent's own account counts for nothing: the code is
  accepted only when FI's checks pass it, and a stop says which check did not;
* after the freeze the run's own failure gets :data:`AFTER_FREEZE_ROUNDS` agent round
  (:meth:`AgentBuildMixin._reflect_by_agent`), then the stop the chat path has; the improve loop does not run.

Report mode is one switch read in one place: ``Engine._chat`` refuses a request that would write or repair code
(:func:`writes_code`) by raising :class:`RepairForAgent`, so no repair site needs its own branch and chat mode is
unchanged. What goes to the agent is what the checks found, in plain words, and never a value a check compares with:
the run log's lines are passed with their numbers taken out (:func:`without_numbers`), and the verdicts of
:func:`core.code_checks.check_code` carry no expected value by construction.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from . import agent_coder as _agent_coder
from . import frozen_protocol as _frozen

#: Fix rounds of the coding agent before the run (after its first write).
FIX_ROUNDS = 3
#: The wall-clock cap of one round of the agent.
ROUND_S = 20 * 60
#: The time the coding agent may take before the run, its first write included.
TOTAL_S = 45 * 60
#: Rounds of the agent after the protocol is frozen (a failure of the study's own run).
AFTER_FREEZE_ROUNDS = 1

#: The model requests that write or repair the experiment's code. In agent mode none is made.
CODE_WRITING_NODES = frozenset({
    "implement", "implement_figures", "implement_protocol", "implement_oracle", "implement_seed", "execute_reflect",
    "implement.finding_repair",
})

#: What a refused request was for, in a scientist's words (the check that wanted the code changed).
_SITE = {
    "implement": "a test of one of the model's functions against the plan's worked example",
    "implement_figures": "the check that each figure is drawn from computed numbers",
    "implement_protocol": "the check that the script follows the plan's protocol (settings, runs, seeds)",
    "implement_oracle": "a known-answer check of the plan",
    "implement_seed": "the check that each run uses the seed FI gives it",
    "execute_reflect": "a check of the code before the run (a result typed into the code, or a static fault)",
    "implement.finding_repair": "a fault another model found in the code and FI's test showed",
}


def writes_code(node: str | None) -> bool:
    """A request (by its node name) that writes or repairs the experiment's code: refused in agent mode."""
    node = str(node or "")
    return node in CODE_WRITING_NODES or node.endswith(".repair")


class RepairForAgent(BaseException):
    """A check asked for the code to be repaired while the code is the coding agent's. A ``BaseException`` so that the
    ``except Exception`` a check keeps for a failed model call does not take it for one: it reaches the agent loop."""

    def __init__(self, node: str) -> None:
        super().__init__(node)
        self.node = node

    @property
    def what(self) -> str:
        return _SITE.get(self.node, "a check of the code")


_NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?(?![A-Za-z_\d])")


def without_numbers(text: str) -> str:
    """``text`` with every stand-alone number replaced by ``…`` (a name like ``E1`` or ``pitch80`` is kept): a line of the
    run log can name a check's expected value, which the agent must never see."""
    return _NUMBER.sub("…", text)


class _Lines(logging.Handler):
    """The warnings the engine logs while the checks run (what each check found)."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(record.getMessage())
        except Exception:  # noqa: BLE001 -- a line that cannot be formatted is left out
            pass


def _record(quest_root: Path) -> dict[str, Any]:
    try:
        data = json.loads((quest_root / _agent_coder.RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_record(quest_root: Path, data: dict[str, Any]) -> None:
    path = quest_root / _agent_coder.RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")


def seconds_used(quest_root: Path) -> float:
    """The time the coding agent's rounds have taken in this quest, from its record."""
    return sum(float(r.get("duration_s") or 0.0) for r in _record(quest_root).get("rounds") or [] if isinstance(r, dict))


def note_judgement(quest_root: Path, *, claimed: str, found: list[str], passed: bool, after_freeze: bool = False) -> None:
    """FI's judgement of the code the last round left, beside what the agent said of it (``.fi/agent_coder.json``):
    ``claimed`` is the agent's last message, ``found`` what FI's checks found."""
    data = _record(quest_root)
    rounds = [r for r in data.get("rounds") or [] if isinstance(r, dict)]
    if not rounds:
        return
    rounds[-1]["claimed"] = claimed[-600:]
    rounds[-1]["found"] = [f[:600] for f in found][:20]
    rounds[-1]["fi_passed"] = passed
    if after_freeze:
        rounds[-1]["after_freeze"] = True
    _save_record(quest_root, data)


def after_freeze_rounds(quest_root: Path) -> int:
    return sum(1 for r in _record(quest_root).get("rounds") or [] if isinstance(r, dict) and r.get("after_freeze"))


class AgentBuildMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    def _agent_mode(self) -> bool:
        return getattr(self.config.engine, "code_writer", "chat") == "agent"  # type: ignore[attr-defined]

    async def _code_agent(self) -> _agent_coder.AgentCoder:
        """The quest's coding agent (one per engine). The runner is the one a test gave (``self._agent_runner``), else the
        Claude command-line tool on this computer (in a container only when Docker is running)."""
        coder = getattr(self, "_agent_coder_obj", None)
        if coder is not None:
            return coder
        runner = getattr(self, "_agent_runner", None)
        if runner is None:
            runner = await _agent_coder.make_runner(self.quest_root, model=self._model_for_node("implement") or "",  # type: ignore[attr-defined]
                                                    log=self._log)  # type: ignore[attr-defined]
        coder = _agent_coder.AgentCoder(self.quest_root, runner, max_fixes=FIX_ROUNDS + AFTER_FREEZE_ROUNDS,  # type: ignore[attr-defined]
                                        timeout_s=ROUND_S, log=self._log)  # type: ignore[attr-defined]
        self._agent_coder_obj = coder
        return coder

    # -- the first write -----------------------------------------------------------------------------------------------

    def _agent_task(self, state: Any, outline: dict[str, Any], extra: str = "") -> str:
        """The agent's task: the design as a code writer may see it, and the names of the functions FI's equation tests
        call (the outline's)."""
        from . import engine as _e

        task = self._prompts["implement_agent"].substitute(  # type: ignore[attr-defined]
            topic_block=self._topic_block(),  # type: ignore[attr-defined]
            design_block=self._code_design_block(state),  # type: ignore[attr-defined]
            clarify_block=_e._format_clarify(state),
            timeout_s=self._planning_limit(state),  # type: ignore[attr-defined]
            skills_block=(self._skills_block(state) + self._dropped_skills_note(state))  # type: ignore[attr-defined]
            or "(no skills selected for this quest)",
            inputs_block=self._inputs_block(),  # type: ignore[attr-defined]
            job_block=self._job_block(state),  # type: ignore[attr-defined]
        )
        functions = [f for f in (outline or {}).get("functions") or [] if isinstance(f, dict) and f.get("name")]
        if functions:
            task += ("\n\n# The functions FI tests by name\n\nFI calls these functions of the model itself, with these "
                     "parameter names, to test each equation; keep the names and parameters exactly:\n\n"
                     + json.dumps(functions, indent=1) + "\n")
        return task + extra

    async def _implement_by_agent(self, state: Any, outline: dict[str, Any], extend: list[str]) -> dict[str, Any]:
        """The implement step in agent mode: the agent writes the code in ``code/``; nothing here repairs it (the checks
        before the run judge it, :meth:`_agent_build`)."""
        from . import engine as _e

        extra = ""
        if extend:
            extra = ("\n\n# Extend the code that is there\n\nA reader of the result asked for: "
                     + "; ".join(p[:200] for p in extend)
                     + ". Add what computes it to the code in this folder and report it in RESULT_JSON; change "
                       "nothing else.\n")
        coder = await self._code_agent()
        result = await coder.write(self._agent_task(state, outline, extra))
        code_path = self.quest_root / "code" / "experiment.py"  # type: ignore[attr-defined]
        code = code_path.read_text(encoding="utf-8") if code_path.is_file() else ""
        if not result.ok and not code:
            self._log.warning("[implement] the coding agent did not write the code (%s); FI's checks will say what is "  # type: ignore[attr-defined]
                              "missing", (result.error or "no message").strip()[:200])
        deps = sorted({*(_agent_deps(result.message)),
                       *_e._coerce_dep_list((state.get("design") or {}).get("dependencies"))})
        await self._refresh_code_project(state, deps, "code written by the coding agent", category="added")  # type: ignore[attr-defined]
        out: dict[str, Any] = {**_e._FRESH_SCRIPT, "code": code, "deps": deps, "refine_extend": [], "extend_missed": []}
        if not extend:
            out["extend_check"], out["extend_unreported"] = {}, []
        return out

    # -- the checks before the run ---------------------------------------------------------------------------------------

    async def _agent_judge(self, state: Any, py: Any, exec_env: Any, seed_path: Path) -> tuple[
            tuple[Any, str | None, bool] | None, list[str], set[str]]:
        """FI's checks of the code the agent left, every repair turned into a report. ``(the gate's result when the code
        passed, or a stop that is not the code's to fix; what the checks found; the names of the failing checks)``."""
        from . import code_checks as _code_checks

        lines = _Lines()
        self._log.addHandler(lines)  # type: ignore[attr-defined]
        found: list[str] = []
        failing: set[str] = set()
        try:
            stop, oracle_code, repaired = await self._gate_pass(state, py, exec_env, seed_path)  # type: ignore[attr-defined]
        except RepairForAgent as asked:
            failing.add(asked.node)
            found.append(f"{asked.what} did not pass.")
            found.extend(without_numbers(line) for line in lines.lines[-6:])
            stop = None
        else:
            if stop is None:
                return (None, oracle_code, repaired), [], set()
            result = stop.get("exec_result") or {}
            if stop.get("exec_give_up_reason"):
                return (stop, None, False), [], set()   # a stop that is not the code's (a plan's fault, FI's limit)
            failing.add("run")
            found.append("Running the code failed:\n" + str(result.get("stderr_tail") or "")[-2000:])
        finally:
            self._log.removeHandler(lines)  # type: ignore[attr-defined]
        try:
            verdicts = await _code_checks.check_code(
                self.quest_root, self._protocol_block(state), executor=self.executor, python=py,  # type: ignore[attr-defined]
                env=exec_env if isinstance(exec_env, dict) else None, log=self._log,  # type: ignore[attr-defined]
                design=state.get("design") if isinstance(state.get("design"), dict) else None)
        except Exception as exc:  # noqa: BLE001 -- the gate's own finding is reason enough
            self._log.info("[implement] FI's list of the code's faults could not be made (%r)", exc)  # type: ignore[attr-defined]
            verdicts = []
        for v in verdicts:
            if v.passed is False:
                failing.add(v.check)
                found.append(v.says)
        return None, found, failing

    async def _agent_build(self, state: Any, py: Any, exec_env: Any, seed_path: Path) -> tuple[Any, str | None, bool]:
        """The checks before the run in agent mode, in place of ``Engine._gate_pass`` and with its result: the agent fixes
        what FI's checks find, round by round, until they pass, the rounds or the time run out, or a round makes no check
        newly pass; then the honest stop names what still fails."""
        coder = await self._code_agent()
        self._repairs_go_to = "agent"
        failing_before: set[str] | None = None
        while True:
            passed, found, failing = await self._agent_judge(state, py, exec_env, seed_path)
            note_judgement(self.quest_root, claimed=coder.last_message, found=found, passed=passed is not None  # type: ignore[attr-defined]
                           and passed[0] is None)
            if passed is not None:
                if passed[0] is None:
                    self._log.info("[implement] FI's checks pass the coding agent's code")  # type: ignore[attr-defined]
                return passed
            why = ""
            if failing_before is not None and not (failing_before - failing):
                why = "its last round made no check of FI's pass that failed before"
            elif coder.fixes_left() <= AFTER_FREEZE_ROUNDS:
                why = f"its {FIX_ROUNDS} round(s) of fixes before the run are used"
            elif seconds_used(self.quest_root) >= TOTAL_S:  # type: ignore[attr-defined]
                why = f"its {TOTAL_S // 60} minutes for the code are used"
            if why:
                text = ("FI's checks still find faults in the coding agent's code, and " + why + ": "
                        + "; ".join(f.splitlines()[0] for f in found if f.strip())[:1500])
                return self._stop_before_run(text, "cannot_repair", tried=[  # type: ignore[attr-defined]
                    "the coding agent wrote the code and fixed what FI's checks found, round by round",
                    *[f.splitlines()[0][:300] for f in found[:3] if f.strip()]]), None, False
            failing_before = failing
            self._log.info("[implement] FI's checks found %d problem(s) in the coding agent's code; sending them back to "  # type: ignore[attr-defined]
                           "it", len(found))
            await coder.continue_("\n\n".join(found))

    # -- after the freeze ------------------------------------------------------------------------------------------------

    async def _reflect_by_agent(self, state: Any, iters: int, bump: int, returncode: str, stdout: str,
                                exec_result: dict[str, Any], spent: dict[str, Any]) -> dict[str, Any]:
        """A failed run in agent mode: one round of the agent with what the run said, in place of the chat repair. After
        the freeze the agent has :data:`AFTER_FREEZE_ROUNDS` round(s); then the failure stands, as in chat mode."""
        frozen = _frozen.load(self.quest_root) is not None  # type: ignore[attr-defined]
        if frozen and after_freeze_rounds(self.quest_root) >= AFTER_FREEZE_ROUNDS:  # type: ignore[attr-defined]
            why = "the coding agent's round after the freeze is used; the run's failure stands"
            self._log.warning("[execute_reflect] %s", why)  # type: ignore[attr-defined]
            return {**spent, "exec_give_up_reason": why}
        coder = await self._code_agent()
        feedback = (f"FI ran the code and the run did not give a usable result (exit code {returncode}).\n\n"
                    f"What it printed (end):\n{stdout[-2000:]}\n\nErrors (end):\n"
                    f"{str(exec_result.get('stderr_tail') or '')[-2000:]}")
        result = await coder.continue_(feedback)
        if result.exhausted:
            why = "the coding agent's rounds are used; the run's failure stands"
            self._log.warning("[execute_reflect] %s", why)  # type: ignore[attr-defined]
            return {**spent, "exec_give_up_reason": why}
        note_judgement(self.quest_root, claimed=result.message, found=[feedback.splitlines()[0]], passed=False,  # type: ignore[attr-defined]
                       after_freeze=frozen)
        history = list(state.get("exec_reflect_history") or [])
        history.append({"iter": iters + 1, "returncode": exec_result.get("returncode"),
                        "stderr_tail": str(exec_result.get("stderr_tail") or "")[-400:],
                        "patch_summary": "the coding agent fixed the code"})
        code_path = self.quest_root / "code" / "experiment.py"  # type: ignore[attr-defined]
        patch: dict[str, Any] = {**spent, "exec_reflect_iter": iters + bump, "exec_reflect_history": history,
                                 "exec_patch_pending": True}
        if code_path.is_file():
            patch["code"] = code_path.read_text(encoding="utf-8")
        return patch

    def _agent_failure_as_run(self, asked: RepairForAgent, lines: list[str]) -> dict[str, Any]:
        """A check outside the checks before the run asked for a repair: a failed run, for the agent's round in the repair
        step."""
        text = "\n".join([f"{asked.what} did not pass.", *(without_numbers(x) for x in lines[-6:])])
        return self._code_review_failed_run(text)  # type: ignore[attr-defined]


def _agent_deps(message: str) -> list[str]:
    """The packages the agent's last message names on a ``DEPS:`` line."""
    out: list[str] = []
    for line in (message or "").splitlines():
        if line.strip().upper().startswith("DEPS:"):
            out.extend(p.strip() for p in line.split(":", 1)[1].split(",") if p.strip())
    return out

