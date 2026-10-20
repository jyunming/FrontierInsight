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
  Each round is judged check by check against the best earlier round (:func:`compare_rounds`): a check that passes
  now and failed there is newly passing, one that passed there and fails now is broken. A round is progress only when
  something newly passes and nothing is broken; a round that broke a check is said so, naming it, to the agent and in
  the run log, and gets one round to put it back. The loop ends early when a round makes nothing newly pass, or when a
  round that broke a check is followed by another round that is not progress. FI does not undo a round: its checks
  judge the code the last round left. The agent's own account counts for nothing: the code is accepted only when FI's
  checks pass it, and a stop says which check did not;
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


#: The checks of :func:`core.code_checks.check_code` and the run of the code, in a scientist's words.
_CHECK_WORDS = {
    "run": "running the code",
    "two_scripts": "the check that the simulation is in simulate.py, as the function FI calls for each setting",
    "protocol": "the check that the code runs the plan's settings, runs and results as the plan gives them",
    "typed_results": "the check that the results are computed, not typed into the code",
    "grid_keys": "the check that the code reads only settings the study has",
    "typed_constants": "the check for numbers typed into the model's functions that the plan never gives",
    "independent_check": "the check that a second implementation does not compare a value with itself",
    "small_run": "FI's small run of the simulation",
    "trial_checks": "the check of the trials that failed in FI's small run",
    "equation_tests": "the test of each equation against the plan's worked example",
    "known_answers": "the plan's known-answer checks",
    "unread_settings": "the check that the code reads every setting the study compares",
    "figure_data": "the check that each figure is drawn from computed numbers",
}


def check_words(check: str) -> str:
    """One of FI's checks of the code (a :func:`core.code_checks.check_code` name, a refused repair's node, ``run``),
    in plain words."""
    return _CHECK_WORDS.get(check) or _SITE.get(check) or check


def _from_gate(check: str) -> bool:
    """A check the gate reports by stopping at it (a refused repair's node, or ``run``): it is never seen to pass, so
    a later round that gets past it counts as having fixed it."""
    return check in ("run", "two_scripts") or writes_code(check)


def compare_rounds(earlier: list[tuple[int, dict[str, bool | None]]], now: dict[str, bool | None]) -> dict[str, Any]:
    """This round's checks (``{check: True passed | False failed | None not judged}``) against the best earlier round
    (``(round, checks)``; the one with the fewest failing checks, the later on a tie). ``{"best": its round or None,
    "newly_passed", "newly_failed", "still_broken"}``: a check failing there that passes now (or that the gate got
    past), one passing there that fails now, and one that passed in any earlier round and fails now."""
    out: dict[str, Any] = {"best": None, "newly_passed": [], "newly_failed": [], "still_broken": []}
    if not earlier:
        return out
    best_n, best = min(reversed(earlier), key=lambda r: sum(1 for v in r[1].values() if v is False))
    out["best"] = best_n
    out["newly_passed"] = sorted(c for c, v in best.items() if v is False and (
        now.get(c) is True or (c not in now and _from_gate(c))))
    out["newly_failed"] = sorted(c for c, v in best.items() if v is True and now.get(c) is False)
    out["still_broken"] = sorted(c for c in now if now[c] is False and any(r[1].get(c) is True for r in earlier))
    return out


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
    """``text`` with every stand-alone number replaced by ``…`` (a name like ``E1`` or ``size80`` is kept): a line of the
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


def note_judgement(quest_root: Path, *, claimed: str, found: list[str], passed: bool, after_freeze: bool = False,
                   checks: dict[str, bool | None] | None = None, compared: dict[str, Any] | None = None) -> None:
    """FI's judgement of the code the last round left, beside what the agent said of it (``.fi/agent_coder.json``):
    ``claimed`` is the agent's last message, ``found`` what FI's checks found; ``checks`` each check's result
    (recorded as ``checks`` and the list ``passed``), ``compared`` the comparison with the best earlier round
    (:func:`compare_rounds`: ``compared_with``, ``newly_passed``, ``newly_failed``)."""
    data = _record(quest_root)
    rounds = [r for r in data.get("rounds") or [] if isinstance(r, dict)]
    if not rounds:
        return
    rounds[-1]["claimed"] = claimed[-600:]
    rounds[-1]["found"] = [f[:600] for f in found][:20]
    rounds[-1]["fi_passed"] = passed
    if checks is not None:
        rounds[-1]["checks"] = dict(checks)
        rounds[-1]["passed"] = sorted(c for c, v in checks.items() if v is True)
    if compared is not None:
        rounds[-1]["compared_with"] = compared.get("best")
        rounds[-1]["newly_passed"] = list(compared.get("newly_passed") or [])
        rounds[-1]["newly_failed"] = list(compared.get("newly_failed") or [])
    if after_freeze:
        rounds[-1]["after_freeze"] = True
    _save_record(quest_root, data)


def judged_rounds(quest_root: Path) -> list[tuple[int, dict[str, bool | None]]]:
    """``(round, checks)`` of every round of the record FI's checks judged before the run, in order."""
    return [(int(r.get("n") or 0), r["checks"]) for r in _record(quest_root).get("rounds") or []
            if isinstance(r, dict) and isinstance(r.get("checks"), dict)]


def _record_round(quest_root: Path, n: int) -> dict[str, Any]:
    """Round ``n`` of the record (``{}`` when there is none)."""
    return next((r for r in _record(quest_root).get("rounds") or [] if isinstance(r, dict) and r.get("n") == n), {})


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
            skills_block=(self._skills_block(state, in_code=True) + self._dropped_skills_note(state))  # type: ignore[attr-defined]
            or "(no skills selected for this quest)",
            inputs_block=self._inputs_block(),  # type: ignore[attr-defined]
            # The two-script contract the chat writer is given (simulate.py's run_cell / run_trial, which FI calls for
            # every setting): without it an agent writes one script, which none of FI's checks of a simulation can run.
            job_block=(self._split_block(state) or self._job_block(state)),  # type: ignore[attr-defined]
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
        self._skill_files_to_code(state)  # type: ignore[attr-defined]
        result = await coder.write(self._agent_task(state, outline, extra))
        if not extend and self._goes_beyond_protocol_by_request():  # type: ignore[attr-defined]
            # Code written again without a person's extension (a redesign, a rewrite after the evidence gate) is held to
            # the frozen protocol as a whole, as the chat writer's is: an earlier extension no longer excuses anything.
            (self.quest_root / "needs" / "PROTOCOL_CHECK.json").unlink(missing_ok=True)  # type: ignore[attr-defined]
        if extend:
            # What the extension added beyond the frozen protocol is recorded as the person's request, as the chat
            # writer's is (Engine._enforce_protocol with ``extended``), so the checks before the run do not fight it.
            protocol = self._protocol_of(state)  # type: ignore[attr-defined]
            if protocol is not None:
                from . import protocol_check as _protocol_check

                added = [m.message() for m in _protocol_check.check(protocol, self._scripts_on_disk())]  # type: ignore[attr-defined]
                self._protocol_record({"status": "extended_by_person", "attempts": [], "differences": added,  # type: ignore[attr-defined]
                                       "protocol": protocol, "written_by": "the coding agent"})
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

    def _agent_protocol(self, state: Any, found: list[str], checks: dict[str, bool | None]) -> bool:
        """Compare the agent's scripts with the plan's protocol and record it in ``needs/PROTOCOL_CHECK.json`` (the record
        the evidence reads). ``True`` when the code goes back to the agent: it differs and ``engine.protocol_check`` is
        ``block``; with ``warn`` the difference is recorded and the checks go on, with ``off`` nothing is compared."""
        from . import protocol_check as _protocol_check

        protocol = self._protocol_of(state)  # type: ignore[attr-defined]
        if protocol is None:
            return False
        # What a reader's extension added beyond the frozen protocol is the person's request (Engine._enforce_protocol
        # with ``extended``): sent back as a difference, the agent was told to take out what the person asked for.
        by_person = self._goes_beyond_protocol_by_request()  # type: ignore[attr-defined]
        differences = [m.message() for m in _protocol_check.check(protocol, self._scripts_on_disk())  # type: ignore[attr-defined]
                       if m.message() not in set(by_person)]
        checks["protocol"] = not differences
        if by_person and not differences:
            return False  # the record of the person's extension stays as it is
        block = self.config.engine.protocol_check == "block"  # type: ignore[attr-defined]
        self._protocol_record({  # type: ignore[attr-defined]
            "status": "ok" if not differences else ("stopped" if block else "warned"),
            "attempts": [{"attempt": 0, "differences": differences}], "differences": differences,
            "protocol": protocol, "written_by": "the coding agent",
            **({"added_by_person": by_person} if by_person else {})})
        if not differences:
            return False
        found.append("The code differs from the plan's protocol: " + "; ".join(differences[:8])
                     + ". Make the code run the settings, the runs and the results the plan gives, as the plan names "
                       "them; change nothing else.")
        return block

    async def _agent_judge(self, state: Any, py: Any, exec_env: Any, seed_path: Path) -> tuple[
            tuple[Any, str | None, bool] | None, list[str], dict[str, bool | None]]:
        """FI's checks of the code the agent left, every repair turned into a report. ``(the gate's result when the code
        passed, or a stop that is not the code's to fix; what the checks found; each check's result: True passed, False
        failed, None not judged)``. The gate stops at the first check that fails, so only that one of its checks is seen."""
        from . import code_checks as _code_checks

        from . import split_run as _split_run

        found: list[str] = []
        checks: dict[str, bool | None] = {}
        # A round that left no script (it ran out of time before writing one) is a finding for the next round: the
        # checks below read the script, and one real round that wrote nothing in its 20 minutes stopped the quest there.
        if not (self.quest_root / "code" / _split_run.ANALYSIS_NAME).is_file():  # type: ignore[attr-defined]
            checks["run"] = False
            found.append(f"There is no `{_split_run.ANALYSIS_NAME}` in the folder: the code was not written. Write "
                         f"`{_split_run.ANALYSIS_NAME}` (and `{_split_run.SIMULATE_NAME}` where the plan needs it) "
                         "first, then run it on a small case; keep each of your own runs short.")
            return None, found, checks
        # A quest whose simulation FI runs itself (execution.split_analysis): the code is not the study's until
        # simulate.py defines the function FI calls for each setting. One script passes none of the checks that need it
        # (FI running a check's case itself, FI's record of every trial), so it is sent back, not run as it is.
        if self._split_on(state):  # type: ignore[attr-defined]
            from . import trial_runner as _trials

            simulate = self.quest_root / "code" / _split_run.SIMULATE_NAME  # type: ignore[attr-defined]
            entries = _trials.entries(simulate) if simulate.is_file() else set()
            if not ({"run_cell", "run_trial"} & set(entries)):
                checks["two_scripts"] = False
                found.append(
                    f"`{_split_run.SIMULATE_NAME}` "
                    + ("is missing" if not simulate.is_file() else "defines neither `run_cell(cell)` nor "
                       "`run_trial(cell, trial_id, seed)`")
                    + ": this study keeps the simulation apart from the analysis, and FI calls the simulation's "
                    f"function itself for every setting. Move the computation of one setting into `{_split_run.SIMULATE_NAME}` "
                    "as `run_cell(cell)` (no randomness) or `run_trial(cell, trial_id, seed)` (with randomness), with no "
                    f"loop over settings and no `__main__` block, and make `{_split_run.ANALYSIS_NAME}` read FI's record "
                    "of the trials (the path in `FI_TRIALS`), as the section on two scripts above says.")
                return None, found, checks
        # The plan's protocol held against the code, as the chat writer's code is (Engine._enforce_protocol): the agent's
        # code was never compared with it, so no agent quest could show that its script follows the plan.
        sent = self._agent_protocol(state, found, checks)
        if sent:
            return None, found, checks
        lines = _Lines()
        self._log.addHandler(lines)  # type: ignore[attr-defined]
        try:
            stop, oracle_code, repaired = await self._gate_pass(state, py, exec_env, seed_path)  # type: ignore[attr-defined]
        except RepairForAgent as asked:
            checks[asked.node] = False
            found.append(f"{asked.what} did not pass.")
            found.extend(without_numbers(line) for line in lines.lines[-6:])
            stop = None
        else:
            if stop is None:
                return (None, oracle_code, repaired), [], {}
            result = stop.get("exec_result") or {}
            if stop.get("exec_give_up_reason"):
                return (stop, None, False), [], {}   # a stop that is not the code's (a plan's fault, FI's limit)
            checks["run"] = False
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
            if checks.get(v.check) is not False:
                checks[v.check] = v.passed
            if v.passed is False:
                found.append(v.says)
        return None, found, checks

    async def _agent_build(self, state: Any, py: Any, exec_env: Any, seed_path: Path) -> tuple[Any, str | None, bool]:
        """The checks before the run in agent mode, in place of ``Engine._gate_pass`` and with its result: the agent fixes
        what FI's checks find, round by round, until they pass, the rounds or the time run out, or the rounds stop making
        progress (:func:`compare_rounds`); then the honest stop names what still fails. A round is never undone: FI's
        checks judge the code the last round left."""
        coder = await self._code_agent()
        self._repairs_go_to = "agent"
        while True:
            passed, found, checks = await self._agent_judge(state, py, exec_env, seed_path)
            n = coder.rounds_made()
            earlier = [r for r in judged_rounds(self.quest_root) if r[0] < n]  # type: ignore[attr-defined]
            compared = compare_rounds(earlier, checks) if passed is None else None
            note_judgement(self.quest_root, claimed=coder.last_message, found=found, passed=passed is not None  # type: ignore[attr-defined]
                           and passed[0] is None, checks=checks if passed is None else None, compared=compared)
            if passed is not None:
                if passed[0] is None:
                    self._log.info("[implement] FI's checks pass the coding agent's code")  # type: ignore[attr-defined]
                return passed
            broke = [check_words(c) for c in compared["newly_failed"]]
            if broke:
                self._log.warning("[implement] the coding agent's last round broke what passed before, so it is not "  # type: ignore[attr-defined]
                                  "progress: %s", "; ".join(broke))
            # The round before this one broke a check and had its one round to put it back.
            after_break = bool(earlier) and bool(_record_round(self.quest_root, earlier[-1][0]).get("newly_failed"))  # type: ignore[attr-defined]
            progress = bool(compared["newly_passed"]) and not broke
            why = ""
            if earlier and not compared["newly_passed"]:
                why = "its last round made no check of FI's pass that failed before"
            elif earlier and after_break and not progress:
                why = "its round to put back what it broke was not progress either"
            elif coder.fixes_left() <= AFTER_FREEZE_ROUNDS:
                why = f"its {FIX_ROUNDS} round(s) of fixes before the run are used"
            elif seconds_used(self.quest_root) >= TOTAL_S:  # type: ignore[attr-defined]
                why = f"its {TOTAL_S // 60} minutes for the code are used"
            if why:
                still = [check_words(c) for c in compared["still_broken"]]
                if still:
                    why += " (passed in an earlier round and fails now: " + "; ".join(still) + ")"
                text = ("FI's checks still find faults in the coding agent's code, and " + why + ": "
                        + "; ".join(f.splitlines()[0] for f in found if f.strip())[:1500])
                return self._stop_before_run(text, "cannot_repair", tried=[  # type: ignore[attr-defined]
                    "the coding agent wrote the code and fixed what FI's checks found, round by round",
                    *[f.splitlines()[0][:300] for f in found[:3] if f.strip()]]), None, False
            if broke:
                found.insert(0, "Your last round broke what passed before (" + "; ".join(broke) + "). Put it back "
                             "while you keep what you fixed: a round that breaks a check that passed is not progress.")
            # A round FI did not keep (its code reached outside its folder) says why, first: the checks only see that no
            # code is there, and a live agent told only that wrote the same code again.
            outside = coder.last_cheats()
            if outside:
                found.insert(0, "FI did not keep your last round, because its code reaches outside its folder: "
                             + "; ".join(outside[:6]) + ". FI keeps no round whose code reads or writes outside the "
                             "folder you are given (no path into the home folder, no absolute path, no `..`): write it "
                             "again without that.")
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

