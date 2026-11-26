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
  round that broke a check is followed by another round that is not progress, except that a round stopped at its time
  limit (the agent's task states it) is not judged so: it gets the next round, told that it was stopped, within the rounds
  and :data:`TOTAL_S` (checked before a round starts). A round that left no script fails only :data:`CODE_WRITTEN`, so
  the round that writes the script has made progress even when its first run fails. FI does not undo a round: its checks
  judge the code the last round left. The agent's own account counts for nothing: the code is accepted only when FI's
  checks pass it, and a stop says which check did not. A round that changes how the code is laid out (one script or
  two, the function simulate.py gives) goes back to the run step, which sets up again for it (:data:`LAYOUT_CHANGED`);
  the round after the freeze is not taken from the :data:`FIX_ROUNDS` (:func:`fixes_before_the_run`);
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
from . import agent_scope as _agent_scope
from .finding_repair import finding_block as _finding_block
from . import frozen_protocol as _frozen
from .split_run import ANALYSIS_NAME as _ANALYSIS_NAME

#: Fix rounds of the coding agent before the run (after its first write).
FIX_ROUNDS = 3
#: The wall-clock cap of one round of the agent, stated in its task (:func:`round_limit_words`). Stored agent rounds that
#: ended by themselves took at most 1,117 s (the 22 that first delivered the code: median 316 s; the fix rounds after
#: that: at most 217 s). A share of the commission's time would have cut some of them on short commissions, and a longer
#: cap was not shown to help, so it is one constant.
#: A round it stops is not judged as the agent's finished attempt (:func:`round_was_capped`).
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


#: The check that the round left the analysis script in the folder. Its own check, not ``run``: a round that writes the
#: code for the first time has made progress even when its first run fails (a real study was stopped at round 2 of 5 as
#: "no progress": round 1 left no script, round 2 wrote all of it and its first run failed, and both were ``run``).
CODE_WRITTEN = "code_written"

#: The checks of :func:`core.code_checks.check_code` and the run of the code, in a scientist's words.
_CHECK_WORDS = {
    CODE_WRITTEN: "the check that the code is written (experiment.py is in the folder)",
    "run": "running the code",
    "before_run": "a check of the code before the run (the code was not run)",
    "code_layout": "the check that the code folder is laid out as the quest needs (equation labels, package, tests)",
    "fis_stub": "the check that each function of derive.py a result needs is written, not FI's stub",
    "unread_compared_setting": "the check that the simulation reads every setting the study compares",
    "pilot_run": "FI's trial run of the study before it starts",
    "small_run_crash": "FI's small run of the simulation (it failed)",
    "small_run_plan": "FI's small run of the simulation compared with the plan",
    "analysis_gate": "the run of the analysis on the recorded results",
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
    "figure_dry_run": "the trial of each figure function on stand-in data",
}


def check_words(check: str) -> str:
    """One of FI's checks of the code (a :func:`core.code_checks.check_code` name, a refused repair's node, ``run``, a
    check before the run ``before_run:<check>``), in plain words."""
    if check.startswith(BEFORE_RUN + ":"):
        return _CHECK_WORDS.get(check.split(":")[1]) or _CHECK_WORDS[BEFORE_RUN]
    return _CHECK_WORDS.get(check) or _SITE.get(check) or check


def _from_gate(check: str) -> bool:
    """A check the gate reports by stopping at it (a refused repair's node, ``run``, a check before the run that sent the
    code back): it is never seen to pass, so a later round that gets past it counts as having fixed it."""
    return check in ("run", "two_scripts") or check.startswith(BEFORE_RUN) or writes_code(check)


#: The checks before the run that send the code back (``exec_result["check"]`` of ``Engine._code_review_failed_run``):
#: ``before_run:<check>``, each its own check; ``before_run`` alone for one that does not name itself.
BEFORE_RUN = "before_run"

#: What the agent is given of one finding at most; a longer one loses its middle, never its opening sentence or its end.
FINDING_CHARS = 8000


def before_run_check(check: Any, finding: str = "") -> str:
    """The name a check before the run is judged under: ``before_run:<check>:<what it found>``, the last part a short
    fingerprint of the finding with its numbers and spacing set aside. One check can find one fault after another (the
    first of several functions still FI's stub): a round that fixes the one it was sent and meets the next has made
    progress, and the same finding again has not. ``before_run`` alone for a check that does not name itself."""
    import hashlib

    if not check:
        return BEFORE_RUN
    said = " ".join(without_numbers(str(finding or "")).split())
    return f"{BEFORE_RUN}:{check}:{hashlib.sha256(said.encode('utf-8')).hexdigest()[:10]}" if said else f"{BEFORE_RUN}:{check}"


#: What a stop's record (needs/STUCK.json ``findings``) keeps of all the findings together; the earliest are kept whole.
RECORD_CHARS = 40000


def findings_for_record(found: list[str]) -> list[str]:
    """The findings a stop's record keeps: each at most :data:`FINDING_CHARS` (its middle cut, :func:`shorten_middle`),
    the earliest whole while they fit :data:`RECORD_CHARS` together, then one line saying how many were left out."""
    out: list[str] = []
    total = 0
    items = [shorten_middle(f, FINDING_CHARS) for f in found if f.strip()]
    for i, f in enumerate(items):
        if total + len(f) > RECORD_CHARS:
            out.append(f"[{len(items) - i} more finding(s) left out: the record keeps {RECORD_CHARS} characters]")
            break
        out.append(f)
        total += len(f)
    return out


def shorten_middle(text: str, limit: int) -> str:
    """``text`` as it is when it fits ``limit``, else its opening and its end with what was left out said in the middle:
    a finding's opening sentence names the check and its end the detail (a traceback's last frame)."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head - 60
    return text[:head] + f"\n[... {len(text) - head - tail} characters left out ...]\n" + text[-tail:]


def one_line(finding: str, limit: int = 300) -> str:
    """A finding on one line, its words kept: what a stop and its list of what was tried say. A finding's first line can be
    a heading alone ("FI ran ... and compared it with the plan:"), so the whole finding is joined, never only that line."""
    text = " ".join(str(finding or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def compare_rounds(earlier: list[tuple[int, dict[str, bool | None]]], now: dict[str, bool | None]) -> dict[str, Any]:
    """This round's checks (``{check: True passed | False failed | None not judged}``) against the best earlier round
    (``(round, checks)``; among the rounds that left code when there are any, the one with the fewest failing checks,
    the later on a tie: a round that left no script failed only :data:`CODE_WRITTEN`, so writing the code is progress
    once, against a round with no code, and never again). ``{"best": its round or None,
    "newly_passed", "newly_failed", "still_broken"}``: a check failing there that passes now (or that the gate got
    past), one passing there that fails now, and one that passed in any earlier round and fails now."""
    out: dict[str, Any] = {"best": None, "newly_passed": [], "newly_failed": [], "still_broken": []}
    if not earlier:
        return out
    with_code = [r for r in earlier if r[1].get(CODE_WRITTEN) is not False] or earlier
    best_n, best = min(reversed(with_code), key=lambda r: sum(1 for v in r[1].values() if v is False))
    out["best"] = best_n
    out["newly_passed"] = sorted(c for c, v in best.items() if v is False and (
        now.get(c) is True or (c not in now and _from_gate(c))))
    out["newly_failed"] = sorted(c for c, v in best.items() if v is True and now.get(c) is False)
    out["still_broken"] = sorted(c for c in now if now[c] is False and any(r[1].get(c) is True for r in earlier))
    return out


def verdict_files(details: Any) -> set[str]:
    """The files (relative to ``code/``) a verdict's details name: every ``script`` of the records the static checks and
    the typed-results check keep (``{"functions": [{"script": ..., "line": ...}]}``). Empty when the verdict names none
    (a check of a run, which is about the agent's code as a whole)."""
    out: set[str] = set()
    if isinstance(details, dict):
        if isinstance(details.get("script"), str) and details["script"]:
            out.add(details["script"])
        for value in details.values():
            if isinstance(value, (dict, list)):
                out |= verdict_files(value)
    elif isinstance(details, list):
        for item in details:
            out |= verdict_files(item)
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
                   checks: dict[str, bool | None] | None = None, compared: dict[str, Any] | None = None,
                   fi_own: list[str] | None = None) -> None:
    """FI's judgement of the code the last round left, beside what the agent said of it (``.fi/agent_coder.json``):
    ``claimed`` is the agent's last message, ``found`` what FI's checks found; ``checks`` each check's result
    (recorded as ``checks`` and the list ``passed``), ``compared`` the comparison with the best earlier round
    (:func:`compare_rounds`: ``compared_with``, ``newly_passed``, ``newly_failed``); ``fi_own`` the findings about a
    file the agent may not change (FI's own finding, kept as ``fi_own_findings``: not sent to the agent, not counted
    against its progress)."""
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
    if fi_own:
        rounds[-1]["fi_own_findings"] = [f[:600] for f in fi_own][:20]
    if after_freeze:
        rounds[-1]["after_freeze"] = True
    _save_record(quest_root, data)


def _saved_checks(r: dict[str, Any]) -> dict[str, bool | None] | None:
    """A judged round's checks as this code names them. A round saved before :data:`CODE_WRITTEN` existed named a missing
    script ``run``: ``{"run": False}`` with the missing script as its finding is ``{CODE_WRITTEN: False}``, any other saved
    round had code; ``{"run": False}`` with no finding saved cannot tell, so it is ``None`` (not compared with)."""
    checks = dict(r["checks"])
    if CODE_WRITTEN in checks:
        return checks
    if checks == {"run": False}:
        found = r.get("found") if isinstance(r.get("found"), list) else []
        if not found:
            return None
        if str(found[0]).startswith(f"There is no `{_ANALYSIS_NAME}`"):
            return {CODE_WRITTEN: False}
    return {**checks, CODE_WRITTEN: True}


def judged_rounds(quest_root: Path) -> list[tuple[int, dict[str, bool | None]]]:
    """``(round, checks)`` of every round of the record FI's checks judged before the run, in order (a round saved by an
    older FI whose checks cannot be read as this code names them is left out, :func:`_saved_checks`)."""
    out = []
    for r in _record(quest_root).get("rounds") or []:
        if isinstance(r, dict) and isinstance(r.get("checks"), dict):
            checks = _saved_checks(r)
            if checks is not None:
                out.append((int(r.get("n") or 0), checks))
    return out


def _record_round(quest_root: Path, n: int) -> dict[str, Any]:
    """Round ``n`` of the record (``{}`` when there is none)."""
    return next((r for r in _record(quest_root).get("rounds") or [] if isinstance(r, dict) and r.get("n") == n), {})


def round_was_capped(quest_root: Path, n: int) -> bool:
    """Round ``n`` was stopped at :data:`ROUND_S` (``timed_out`` in its record): the agent was stopped, not done, so the
    round is not judged as making no progress; the rounds and :data:`TOTAL_S` still bound the loop."""
    return bool(_record_round(quest_root, n).get("timed_out"))


def round_limit_words(seconds: float) -> str:
    """A round's time limit as the agent is told it (``20 minutes``)."""
    minutes = float(seconds) / 60
    return f"{minutes:g} minutes" if minutes >= 1 else f"{float(seconds):g} seconds"


def stopped_note(seconds: float) -> str:
    """What the round after one stopped at its limit is told first: told only what the checks found ("the code was not
    written"), a live agent started its work over."""
    return (f"FI stopped your last round at its limit of {round_limit_words(seconds)}, before it ended, and judged what it "
            "left in the folder. Continue from what is there instead of starting again: write the files the checks below "
            "need first, and keep each of your own runs to a few minutes, so that they are in the folder when this round "
            "ends.")


def after_freeze_rounds(quest_root: Path) -> int:
    return sum(1 for r in _record(quest_root).get("rounds") or [] if isinstance(r, dict) and r.get("after_freeze"))


def fixes_before_the_run(quest_root: Path) -> int:
    """The coding agent's fix rounds for the checks before the run: every fix round of the record but the round(s) given
    to a failure of the study's own run after the freeze (:data:`AFTER_FREEZE_ROUNDS`), which are not taken from these."""
    return sum(1 for r in _record(quest_root).get("rounds") or [] if isinstance(r, dict) and r.get("kind") == "fix"
               and not r.get("after_freeze"))


class _LayoutChanged:
    """The mark :meth:`AgentBuildMixin._agent_build` returns in place of a stop when a round changed how the code is
    laid out: the run step sets up again for the code that is there (``Engine._node_execute_steps``)."""

    def __repr__(self) -> str:
        return "LAYOUT_CHANGED"


LAYOUT_CHANGED: Any = _LayoutChanged()


def layout_words(layout: tuple[bool, frozenset[str]]) -> str:
    """A layout of the code (``Engine._script_layout``) in plain words, for run.log."""
    split, entries = layout
    if not split:
        return "one script"
    return ("simulate.py and experiment.py, simulate.py giving " + " and ".join(f"`{e}`" for e in sorted(entries))
            if entries else "simulate.py and experiment.py, simulate.py running its own loop")


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

    def _agent_task(self, state: Any, outline: dict[str, Any], extra: str = "", *, round_limit: bool = True) -> str:
        """The agent's task: the design as a code writer may see it, and the names of the functions FI's equation tests
        call (the outline's). ``round_limit=False``: without the line that states a round's time limit, the text whose
        hash says whether a saved write is this same task (:meth:`AgentCoder.write`): a quest saved before that line
        existed keeps its rounds, its session and its budget on resume."""
        from . import engine as _e

        task = _with_round_limit(self._prompts["implement_agent"], round_limit).substitute(  # type: ignore[attr-defined]
            topic_block=self._topic_block(),  # type: ignore[attr-defined]
            design_block=self._code_design_block(state),  # type: ignore[attr-defined]
            clarify_block=_e._format_clarify(state),
            timeout_s=self._planning_limit(state),  # type: ignore[attr-defined]
            # The limit the runner enforces on each of the agent's rounds: told nothing of it, a live agent spent two whole
            # rounds on its own long trial runs and never wrote the script.
            round_limit=round_limit_words(getattr(getattr(self, "_agent_coder_obj", None), "timeout_s", ROUND_S)),
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
        result = await coder.write(self._agent_task(state, outline, extra),
                                   same_task=self._agent_task(state, outline, extra, round_limit=False))
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
        scripts = self._scripts_on_disk()  # type: ignore[attr-defined]
        # The same reading as the chat writer's repair (Engine._enforce_protocol): a difference that rests only on a ``for``
        # loop's literal values never sends the code back; it is a note.
        differences = [m.message() for m in _protocol_check.check_repairable(protocol, scripts)
                       if m.message() not in set(by_person)]
        notes = _protocol_check.loop_hints(protocol, scripts)
        checks["protocol"] = not differences
        if by_person and not differences:
            return False  # the record of the person's extension stays as it is
        block = self.config.engine.protocol_check == "block"  # type: ignore[attr-defined]
        self._protocol_record({  # type: ignore[attr-defined]
            "status": "ok" if not differences else ("stopped" if block else "warned"),
            "attempts": [{"attempt": 0, "differences": differences}], "differences": differences,
            "protocol": protocol, "written_by": "the coding agent",
            **({"added_by_person": by_person} if by_person else {}), **({"notes": notes} if notes else {})})
        if not differences:
            return False
        found.append("The code differs from the plan's protocol: " + "; ".join(differences[:8])
                     + ". Make the code run the settings, the runs and the results the plan gives, as the plan names "
                       "them; change nothing else.")
        return block

    def _shown_findings(self, state: Any, found: list[str], checks: dict[str, bool | None]) -> None:
        """Each fault another model found in the code that FI's test showed, from FI's record of them
        (.fi/code_review.json): one check per fault (``finding <id>``: passed once the record no longer flags it), so a
        round that fixes one and has a new one shown is judged as the progress it is, and the fault still flagged is sent
        to the agent as the chat repair is given it (the finding and how its test failed with the plan's checked values
        taken out, the test whole). The gate's one line for them ("did not pass") named none of it."""
        from . import code_findings as _issues
        from . import code_review as _review
        from . import review_pairs as _pairs

        record = _review.load(self.quest_root)  # type: ignore[attr-defined]
        values = _pairs.plan_values(self._protocol_block(state))  # type: ignore[attr-defined]

        def scrub(text: str) -> str:
            return _pairs.scrub(text, values)

        for issue in _issues.ranked(record, only_open=False):
            if issue.get("demonstrated") is not True or issue.get("plan_conflict"):
                continue
            still = _issues.flagged_now(issue)
            checks[f"finding {issue.get('id')}"] = not still
            if not still:
                continue
            test = str(issue.get("test") or "")
            try:
                source = (self.quest_root / test).read_text(encoding="utf-8") if test else ""  # type: ignore[attr-defined]
            except OSError:
                source = ""
            text = ("A fault another model found in the code, which FI's test shows on your code:\n"
                    + _finding_block(issue, scrub))
            if source:
                # As the chat repair is given it (core/finding_repair.py): the test whole, so that it still runs.
                text += f"\nFI's test (it imports your files from the code folder):\n```python\n{source.rstrip()}\n```"
            if issue.get("failure"):
                text += "\nHow it failed: " + scrub(str(issue["failure"]))[:800]
            found.append(text)

    async def _agent_judge(self, state: Any, py: Any, exec_env: Any, seed_path: Path) -> tuple[
            tuple[Any, str | None, bool] | None, list[str], dict[str, bool | None]]:
        """FI's checks of the code the agent left, every repair turned into a report. ``(the gate's result when the code
        passed, or a stop that is not the code's to fix; what the checks found; each check's result: True passed, False
        failed, None not judged)``. The gate stops at the first check that fails, so only that one of its checks is seen."""
        from . import code_checks as _code_checks

        from . import split_run as _split_run

        found: list[str] = []
        checks: dict[str, bool | None] = {}
        self._fi_own_findings = []  # type: ignore[attr-defined]
        # A round that left no script (it ran out of time before writing one) is a finding for the next round: the
        # checks below read the script, and one real round that wrote nothing in its 20 minutes stopped the quest there.
        # It is its own check (CODE_WRITTEN), not a failed run: the round that then writes the script has made progress.
        if not (self.quest_root / "code" / _split_run.ANALYSIS_NAME).is_file():  # type: ignore[attr-defined]
            checks[CODE_WRITTEN] = False
            found.append(f"There is no `{_split_run.ANALYSIS_NAME}` in the folder: the code was not written. Write "
                         f"`{_split_run.ANALYSIS_NAME}` (and `{_split_run.SIMULATE_NAME}` where the plan needs it) "
                         "first, then run it on a small case; keep each of your own runs short.")
            return None, found, checks
        checks[CODE_WRITTEN] = True
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
            text = str(result.get("stderr_tail") or "").strip()
            if result.get("directive"):
                # A check before the run sent the code back with what it found (FI's small run, the layout of the code
                # folder, a stub of FI's): the code did not run, and the finding is the check's own text, whole. Each
                # check is its own (``before_run:<check>``), so a round that fixes one and meets the next is progress.
                checks[before_run_check(result.get("check"), text)] = False
                found.append(shorten_middle(text, FINDING_CHARS)
                             or "A check of the code before the run sent it back without saying why.")
            else:
                checks["run"] = False
                found.append(f"Running the code failed (exit code {result.get('returncode', '?')}): "
                             + (shorten_middle(text, FINDING_CHARS) or "it printed no error message."))
        finally:
            self._log.removeHandler(lines)  # type: ignore[attr-defined]
        self._shown_findings(state, found, checks)
        try:
            verdicts = await _code_checks.check_code(
                self.quest_root, self._protocol_block(state), executor=self.executor, python=py,  # type: ignore[attr-defined]
                env=exec_env if isinstance(exec_env, dict) else None, log=self._log,  # type: ignore[attr-defined]
                design=state.get("design") if isinstance(state.get("design"), dict) else None)
        except Exception as exc:  # noqa: BLE001 -- the gate's own finding is reason enough
            self._log.info("[implement] FI's list of the code's faults could not be made (%r)", exc)  # type: ignore[attr-defined]
            verdicts = []
        coder = await self._code_agent()
        fi_own: list[str] = []
        for v in verdicts:
            # A fault in a file the agent may not change (FI's own copies, the tests FI writes) cannot be cleared by it: it
            # is FI's finding, kept and disclosed, and neither sent to the agent nor counted as a check it fails.
            files = verdict_files(v.details) if v.passed is False else set()
            if files and not any(coder.may_edit(f) for f in files):
                fi_own.append(f"{check_words(v.check)}: {v.says} (in {', '.join(sorted(files))}, which the coding agent may "
                              "not change)")
                continue
            if checks.get(v.check) is not False:
                checks[v.check] = v.passed
            if v.passed is False:
                found.append(v.says)
        self._fi_own_findings = fi_own  # type: ignore[attr-defined]
        for line in fi_own:
            self._log.warning("[implement] FI's own finding, not the coding agent's to fix: %s", line)  # type: ignore[attr-defined]
        return None, found, checks

    async def _agent_build(self, state: Any, py: Any, exec_env: Any, seed_path: Path, *,
                           layout: tuple[bool, frozenset[str]] | None = None) -> tuple[Any, str | None, bool]:
        """The checks before the run in agent mode, in place of ``Engine._gate_pass`` and with its result: the agent fixes
        what FI's checks find, round by round, until they pass, the rounds or the time run out, or the rounds stop making
        progress (:func:`compare_rounds`); then the honest stop names what still fails. A round is never undone: FI's
        checks judge the code the last round left.

        ``layout`` is how the code was laid out when the run step set up (``Engine._script_layout``: one script or two, the
        function simulate.py gives). When a round changes it, :data:`LAYOUT_CHANGED` is returned before the checks judge
        the new code: the runner, the script the checks run and whether FI runs the trials were all set up for the old
        layout, and the checks made on them judged code FI would not run that way."""
        coder = await self._code_agent()
        self._repairs_go_to = "agent"
        while True:
            if layout is not None and self._script_layout(state) != layout:  # type: ignore[attr-defined]
                return LAYOUT_CHANGED, None, False
            passed, found, checks = await self._agent_judge(state, py, exec_env, seed_path)
            n = coder.rounds_made()
            earlier = [r for r in judged_rounds(self.quest_root) if r[0] < n]  # type: ignore[attr-defined]
            compared = compare_rounds(earlier, checks) if passed is None else None
            note_judgement(self.quest_root, claimed=coder.last_message, found=found, passed=passed is not None  # type: ignore[attr-defined]
                           and passed[0] is None, checks=checks if passed is None else None, compared=compared,
                           fi_own=list(getattr(self, "_fi_own_findings", None) or []))
            if passed is not None:
                if passed[0] is None:
                    self._log.info("[implement] FI's checks pass the coding agent's code")  # type: ignore[attr-defined]
                return passed
            broke = [check_words(c) for c in compared["newly_failed"]]
            if broke:
                self._log.warning("[implement] the coding agent's last round broke what passed before, so it is not "  # type: ignore[attr-defined]
                                  "progress: %s", "; ".join(broke))
            # The last earlier round that ended by itself broke a check, and this is its one round to put it back (a round
            # stopped at its limit in between is not that round).
            ended = [r for r in earlier if not round_was_capped(self.quest_root, r[0])]  # type: ignore[attr-defined]
            after_break = bool(ended) and bool(_record_round(self.quest_root, ended[-1][0]).get("newly_failed"))  # type: ignore[attr-defined]
            progress = bool(compared["newly_passed"]) and not broke
            # A round stopped at its time limit did not end: what it left is judged and sent back, but it is not the
            # agent's finished attempt, so it gets the next round (two real rounds stopped at the limit before the script
            # was written ended a study as "no progress" with three of its five rounds unused).
            capped = round_was_capped(self.quest_root, n)  # type: ignore[attr-defined]
            why = ""
            if earlier and not compared["newly_passed"] and not capped:
                why = "its last round made no check of FI's pass that failed before"
            elif earlier and after_break and not progress and not capped:
                why = "its round to put back what it broke was not progress either"
            elif coder.fixes_left() <= 0:
                why = f"all {coder.total_rounds} of its rounds are used (round {n} of {coder.total_rounds})"
            elif fixes_before_the_run(self.quest_root) >= FIX_ROUNDS:  # type: ignore[attr-defined]
                why = (f"its {FIX_ROUNDS} round(s) of fixes before the run are used (round {n} of {coder.total_rounds}; "
                       f"the last {AFTER_FREEZE_ROUNDS} is kept for a failure of the study's own run)")
            elif seconds_used(self.quest_root) >= TOTAL_S:  # type: ignore[attr-defined]
                why = f"its {TOTAL_S // 60} minutes for the code are used"
            if why and capped:
                why += f" (its last round was stopped at its limit of {round_limit_words(coder.timeout_s)})"
            if why:
                still = [check_words(c) for c in compared["still_broken"]]
                if still:
                    why += " (passed in an earlier round and fails now: " + "; ".join(still) + ")"
                text = ("FI's checks still find faults in the coding agent's code, and " + why + ": "
                        + "; ".join(one_line(f) for f in found if f.strip()))
                if getattr(self, "_fi_own_findings", None):
                    text += (" FI's own findings, which the agent may not change and which did not count against it: "
                             + "; ".join(self._fi_own_findings)[:800])  # type: ignore[attr-defined]
                stop = self._stop_before_run(text, "cannot_repair", tried=[  # type: ignore[attr-defined]
                    "the coding agent wrote the code and fixed what FI's checks found, round by round",
                    *[one_line(f) for f in found[:3] if f.strip()]])
                # every finding whole, for needs/STUCK.json (the stop's line above is the short form)
                stop["exec_result"]["cannot_repair_findings"] = findings_for_record(found)
                return stop, None, False
            if broke:
                found.insert(0, "Your last round broke what passed before (" + "; ".join(broke) + "). Put it back "
                             "while you keep what you fixed: a round that breaks a check that passed is not progress.")
            if capped:
                found.insert(0, stopped_note(coder.timeout_s))
            # A round FI did not keep (its code reached outside its folder) says why, first: the checks only see that no
            # code is there, and a live agent told only that wrote the same code again.
            outside = coder.last_cheats()
            if outside:
                found.insert(0, "FI did not keep your last round, because its code reaches outside its folder: "
                             + "; ".join(outside[:6]) + ". FI keeps no round whose code reads or writes outside the "
                             "folder you are given (no path into the home folder, no absolute path, no `..`): write it "
                             "again without that.")
            else:
                held = coder.last_not_kept()
                if held:
                    found.insert(0, "FI did not keep your last round: " + "; ".join(held[:6]) + ". Leave a file FI "
                                 "cannot read as it is in code/ (or remove it if the code does not need it), and make "
                                 "the change in Python or a plain text or data file instead.")
            self._log.info("[implement] FI's checks found %d problem(s) in the coding agent's code; sending them back to "  # type: ignore[attr-defined]
                           "it", len(found))
            # A round that has only the run's failure to fix is limited to the code the traceback names; a round with
            # other findings (the checks before the run) is not, since FI cannot name their place.
            failing = [c for c, ok in checks.items() if ok is False]
            only_run = len(failing) == 1 and (failing[0] == "run" or failing[0].startswith(BEFORE_RUN))
            scope = _agent_scope.scope_from_traceback("\n\n".join(found), self.quest_root) if only_run else None  # type: ignore[attr-defined]
            await _agent_scope.scoped_continue(coder, "\n\n".join(found), scope, self._log)  # type: ignore[attr-defined]

    # -- after the freeze ------------------------------------------------------------------------------------------------

    async def _reflect_by_agent(self, state: Any, iters: int, bump: int, returncode: str, stdout: str,
                                exec_result: dict[str, Any], spent: dict[str, Any], *, own_failure: str = "", tag: str = "",
                                attempt: int | None = None) -> dict[str, Any]:
        """A failed run in agent mode: one round of the agent with what the run said, in place of the chat repair. After
        the freeze the agent has :data:`AFTER_FREEZE_ROUNDS` round(s); then the failure stands, as in chat mode. ``tag``:
        what the repair is for when it has a budget of its own (the history entry says so); ``attempt``: its number in
        the one sequence of repairs (default ``iters + 1``)."""
        frozen = _frozen.load(self.quest_root) is not None  # type: ignore[attr-defined]
        if frozen and after_freeze_rounds(self.quest_root) >= AFTER_FREEZE_ROUNDS:  # type: ignore[attr-defined]
            kept = self._result_check_gave_up(state)  # type: ignore[attr-defined]
            if kept is not None:  # a run the results check sent back keeps its results
                return {**spent, **kept}
            why = "the coding agent's round after the freeze is used; the run's failure stands"
            self._log.warning("[execute_reflect] %s", why)  # type: ignore[attr-defined]
            return {**spent, "exec_give_up_reason": why}
        coder = await self._code_agent()
        feedback = (f"FI ran the code and the run did not give a usable result (exit code {returncode}).\n\n"
                    f"What it printed (end):\n{stdout[-2000:]}\n\nErrors (end):\n"
                    f"{str(exec_result.get('stderr_tail') or '')[-2000:]}")
        feedback = own_failure + feedback  # the failed function's own failure from FI's record (worded by its template)
        scope = _agent_scope.scope_from_traceback(feedback, self.quest_root)  # type: ignore[attr-defined]
        result = await _agent_scope.scoped_continue(coder, feedback, scope, self._log,  # type: ignore[attr-defined]
                                                    marks={"after_freeze": True} if frozen else None)
        # A round stopped at its limit did not end: while a round is left it gets the next one, told to continue from
        # what it left (as before the run, AgentBuildMixin._agent_build).
        def time_left() -> bool:
            return seconds_used(self.quest_root) < TOTAL_S  # type: ignore[attr-defined]

        while result.timed_out and not result.cheats and not result.not_kept and coder.fixes_left() > 0 and time_left():
            self._log.info("[execute_reflect] the coding agent's round was stopped at its limit; it continues in the "  # type: ignore[attr-defined]
                           "next round")
            result = await _agent_scope.scoped_continue(coder, stopped_note(coder.timeout_s) + "\n\n" + feedback,  # type: ignore[attr-defined]
                                                        scope, self._log, marks={"after_freeze": True} if frozen else None)
        if result.timed_out and not result.cheats and not result.not_kept and not time_left():
            # the same bound as before the run (AgentBuildMixin._agent_build): no round starts once the time is used
            kept = self._result_check_gave_up(state)  # type: ignore[attr-defined]
            if kept is not None:  # a run the results check sent back keeps its results
                return {**spent, **kept}
            why = (f"its {TOTAL_S // 60} minutes for the code are used (its last round was stopped at its limit of "
                   f"{round_limit_words(coder.timeout_s)}); the run's failure stands")
            self._log.warning("[execute_reflect] %s", why)  # type: ignore[attr-defined]
            return {**spent, "exec_give_up_reason": why}
        if result.exhausted:
            kept = self._result_check_gave_up(state)  # type: ignore[attr-defined]
            if kept is not None:  # a run the results check sent back keeps its results
                return {**spent, **kept}
            why = "the coding agent's rounds are used; the run's failure stands"
            self._log.warning("[execute_reflect] %s", why)  # type: ignore[attr-defined]
            return {**spent, "exec_give_up_reason": why}
        if not result.ok:  # the round did not finish, or FI undid it (invalid Python, a file it could not put back)
            why = (f"the coding agent's round did not give usable code ({(result.error or 'no message').strip()[:150]}); "
                   "the run's failure stands")
            self._log.warning("[execute_reflect] %s", why)  # type: ignore[attr-defined]
            return {**spent, "exec_give_up_reason": why}
        note_judgement(self.quest_root, claimed=result.message, found=[feedback.splitlines()[0]], passed=False,  # type: ignore[attr-defined]
                       after_freeze=frozen)
        history = list(state.get("exec_reflect_history") or [])
        history.append({**({"for": tag} if tag else {}), "iter": attempt if attempt is not None else iters + 1,
                        "returncode": exec_result.get("returncode"),
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


def _with_round_limit(template: Any, keep: bool) -> Any:
    """The agent's prompt template as it is (``keep``), or without its line that states a round's time limit: the text a
    quest saved before that line existed was hashed from."""
    from string import Template

    if keep:
        return template
    return Template("".join(line for line in template.template.splitlines(keepends=True) if "$round_limit" not in line))


def _agent_deps(message: str) -> list[str]:
    """The packages the agent's last message names on a ``DEPS:`` line."""
    out: list[str] = []
    for line in (message or "").splitlines():
        if line.strip().upper().startswith("DEPS:"):
            out.extend(p.strip() for p in line.split(":", 1)[1].split(",") if p.strip())
    return out

