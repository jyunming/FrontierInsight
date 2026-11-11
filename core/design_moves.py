"""Before a search, FI moves each design variable once and says which of them change the objective.

A search whose design changes nothing it returns spends its whole budget finding the baseline again, and the fact was
found only after it (the design-free check of the best design, :mod:`core.optimum_check`). The small run before the code
is read already calls the simulation; this pass adds a few calls through the search's own entry
(:func:`core.optimise._evaluate`, the function :func:`core.trial_runner.study_entry` names, the cell
:func:`core.optimise_search.cell_of` builds):

* the plan's baseline design, once;
* each design variable moved alone, in a part of the study that does not hold it (``sub_studies[].hold``; a part whose
  ``vary`` names it first, else the first part that does not hold it): a number to its declared low and high, a choice to
  up to two of its other values, an array as a whole to its low and its high. Each moved design goes through
  :meth:`core.optimise_search.Space.canonical` (its type, kept inside its range). A move that comes back as the starting
  design is "not moved". The part's starting design (the baseline with what the part holds) is evaluated too, so a move is
  always compared with a design of the same part.

None of these designs depends on another's answer, so they are evaluated at the same time, as the search's own batches
are (:func:`core.optimise.evaluate_many`, ``execution.settings_at_once``).

Movement is judged on what :func:`core.optimise_search.judge` reads from an answer: the objective, and each constraint on
its own, never on every number the simulation returns. The search has no rule that refuses a design inside the ranges
before it is evaluated (``Space.inside`` holds for every canonical design, and ``canonical`` does not know a rule that
ties two variables together), so a design is refused only by the simulation itself: an evaluation without an answer (it
raised, or its process was stopped) is "not tested at that end", with its reason, and counts for nothing. An answer
without an objective the search can read (missing, not a finite number) is an outcome of the design: a variable moves the
objective when two of its designs give different values of it, or one gives a value and another an answer without one
(the search sees that difference: it never chooses a design without one). A design that breaks a limit has an objective
and is judged like any other, the broken limit said.

No change at the two ends does not show that the simulation ignores a variable (it may act past a threshold between
them): it is one line in run.log and plan.md and a fact for the code's reader, never a stop and never a part dropped.
The pass saves no search time; it moves the fact before the search. It has its own time, the small run's
(:func:`core.small_run.budget`); a variable it did not reach is "not measured". Nothing here knows a field of science.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import optimise_search as _search

RECORD = Path(".fi") / "design_moves.json"
#: At most this many other values of a choice are tried.
CHOICES = 2

#: How the designs are evaluated: ``(cells, seconds) -> answers`` in the order of ``cells``, each as
#: :func:`core.optimise._evaluate` gives it (``{"values": ...}``, ``{"failed": why}``, or ``{"stop": "time"}`` for a design
#: the time did not reach; ``elapsed_s`` beside).
Evaluate = Callable[[list[dict[str, Any]], float], Awaitable[list[dict[str, Any]]]]


# --- which designs ---------------------------------------------------------------------------------------------------


def parts(block: dict[str, Any]) -> list[dict[str, Any]]:
    """The block's ``sub_studies`` that read (:func:`core.optimise_search.study_grid`), else ``[]``."""
    grid = block.get("grid") if isinstance(block.get("grid"), dict) else {}
    return list(getattr(_search.study_grid({"grid": grid, "sub_studies": block.get("sub_studies")}), "sub_studies", None) or [])


def part_for(block: dict[str, Any], name: str) -> tuple[dict[str, Any] | None, bool]:
    """``(part, held_everywhere)``: the part a variable is moved in (``None`` for a study without parts, which is moved with
    the plan's own conditions), and whether every part holds it (it is then not moved)."""
    found = parts(block)
    if not found:
        return None, False
    free = [p for p in found if name not in (p.get("hold") or {})]
    if not free:
        return None, True
    return next((p for p in free if name in (p.get("vary") or [])), free[0]), False


def part_block(block: dict[str, Any], part: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(block, starting design)`` of a part: the conditions it holds replace the plan's ``fixed`` ones, the design
    variables it holds replace the baseline's values; a setting of the grid the search does not hand the simulation (not
    in ``fixed``) stays out of the cell, as in every evaluation of the search. ``part=None``: the plan's block and
    baseline."""
    space = _search.Space(block["design_variables"])
    base = dict(block["baseline"]["values"])
    if part is None:
        return block, space.canonical(base)
    fixed = dict(block.get("fixed") if isinstance(block.get("fixed"), dict) else {})
    for key, value in (part.get("hold") or {}).items():
        if key in space.names:
            base[key] = value
        elif key in fixed:
            fixed[key] = value
    return {**block, "fixed": fixed}, space.canonical(base)


def _with(space: _search.Space, design: dict[str, Any], name: str, value: Any) -> dict[str, Any]:
    return space.canonical({**design, name: value})


def _array_at(space: _search.Space, design: dict[str, Any], name: str, end: str) -> dict[str, Any]:
    """``design`` with every element of the array variable ``name`` at its own ``low`` or ``high``."""
    flat = space.flat(design)
    for k, v in enumerate(space.scalars):
        if v.get("of") == name:
            flat[k] = v[end]
    return space.canonical(space.fold(flat))


def moves_of(space: _search.Space, variable: dict[str, Any], start: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """``[(label, moved design)]`` of one variable from ``start``: a number at its low and its high, a choice at up to
    :data:`CHOICES` other values (in the plan's order), an array whole at its low and its high. Each design is canonical."""
    name = str(variable["name"])
    kind = _search._kind(variable)
    if kind == "choice":
        others = [c for c in variable.get("values") or [] if str(c) != str(start.get(name))][:CHOICES]
        return [(f"at {_search._fmt(c)}", _with(space, start, name, c)) for c in others]
    if _search._shape(variable) is not None:
        return [(f"at its {end}", _array_at(space, start, name, end)) for end in ("low", "high")]
    return [(f"at its {end}", _with(space, start, name, variable[end])) for end in ("low", "high")]


# --- what it gave ----------------------------------------------------------------------------------------------------


@dataclass
class Point:
    """One evaluated design of a variable: its starting design or one end."""
    label: str  # "at its starting design", "at its low", "at its high", "at <value>"
    status: str = "not_measured"  # not_moved / tested / no_objective / not_tested / not_measured
    reason: str = ""  # why it gave no objective, why it was not tested, or the limit it breaks
    objective: float | None = None
    constraints: dict[str, float] = field(default_factory=dict)
    feasible: bool | None = None

    def take(self, answer: dict[str, Any] | None, judged: dict[str, Any] | None) -> None:
        """What :func:`core.optimise_search.judge` read from the evaluation's ``answer`` (``None``: the time ran out)."""
        if answer is None or judged is None:
            self.status = "not_measured"
        elif judged["status"] == "ok":
            self.status, self.objective, self.constraints = "tested", judged["objective"], dict(judged["constraints"])
            self.feasible = bool(judged["feasible"])
            self.reason = "" if self.feasible else "this design breaks a limit of the plan"
        elif "values" not in answer:
            # The simulation gave no answer for this design (it raised, or its process was stopped from outside): a
            # design it refuses, the search's own rule for a design it can never choose, says nothing of the objective.
            self.status, self.reason = "not_tested", str(answer.get("failed") or judged.get("reason") or "no answer")
        else:
            # It answered, and the answer has no objective the search can read (missing, not finite): an outcome of
            # the design, which the search sees.
            self.status, self.reason = "no_objective", str(judged.get("reason") or "the evaluation failed")


@dataclass
class Variable:
    name: str
    part: str = ""  # the part it was moved in ("" for the plan's own conditions)
    held: bool = False  # every part holds it
    start: Point = field(default_factory=lambda: Point("at its starting design"))
    ends: list[Point] = field(default_factory=list)

    def _evaluated(self) -> list[Point]:
        return [self.start, *(e for e in self.ends if e.status != "not_moved")]

    @property
    def moved_constraints(self) -> list[str]:
        """The limits whose quantity differs between two of its designs that gave one, each judged on its own."""
        tested = [p for p in self._evaluated() if p.status == "tested"]
        names = sorted({c for p in tested for c in p.constraints})
        return [c for c in names if len(tested) >= 2 and any(not _same(p.constraints.get(c), tested[0].constraints.get(c))
                                                             for p in tested)]

    @property
    def verdict(self) -> str:
        """``moves``: two of its designs gave different objectives, or one gave an objective and another none (the search
        sees that difference: it never chooses a design without one). ``no_change``: at least two gave an objective, all
        the same, and none failed for the design. ``not_moved``: no end differs from the starting design, or every part
        holds it. ``not_measured``: the time ran out first. ``not_tested``: too few designs could be judged."""
        if self.held or not self.ends or all(e.status == "not_moved" for e in self.ends):
            return "not_moved"
        points = self._evaluated()
        tested = [p for p in points if p.status == "tested"]
        if tested and (any(not _same(p.objective, tested[0].objective) for p in tested)
                       or any(p.status == "no_objective" for p in points)):
            return "moves"
        if len(tested) >= 2:
            return "no_change"
        return "not_measured" if any(p.status == "not_measured" for p in points) else "not_tested"

    def sentence(self, objective: str) -> str:
        where = f" (in the part `{self.part}`)" if self.part else ""
        verdict = self.verdict
        if self.held:
            return f"`{self.name}` is held at one value in every part of the study, so it was not moved."
        if verdict == "not_moved":
            return f"`{self.name}` was not moved{where}: each end it was moved to is its starting value."
        if verdict == "not_measured" and all(p.status in ("not_measured", "not_moved") for p in self._evaluated()):
            return f"`{self.name}` was not measured{where}: the time FI gives this check ran out first."
        said = []
        for p in self._evaluated():
            if p.status == "tested":
                said.append(f"{_search._fmt(p.objective)} {p.label}")
            elif p.status == "no_objective":
                said.append(f"no objective {p.label} ({p.reason})")
            elif p.status == "not_tested":
                said.append(f"not tested {p.label} ({p.reason})")
            else:
                said.append(f"{p.label} not measured (the time ran out)")
        shown = "; ".join(said)
        moved = self.moved_constraints
        limits = (f"; the limit quantit{'y' if len(moved) == 1 else 'ies'} {', '.join(f'`{c}`' for c in moved)} moved"
                  if moved else "")
        if verdict == "moves":
            return f"`{self.name}` moves the objective `{objective}`{where}: {shown}{limits}."
        if verdict == "no_change":
            return (f"`{self.name}`: no change of the objective `{objective}` seen{where}: {shown}{limits}. No change at a few points does "
                    "not show the simulation ignores it (it may act past a threshold); the search will tell.")
        return f"`{self.name}` could not be judged{where}: {shown}."

    def record(self) -> dict[str, Any]:
        return {"name": self.name, "part": self.part, "held": self.held, "verdict": self.verdict,
                "moved_constraints": self.moved_constraints,
                "points": [{"label": p.label, "status": p.status, "reason": p.reason, "objective": p.objective,
                            "constraints": p.constraints, "feasible": p.feasible} for p in [self.start, *self.ends]]}


@dataclass
class Moves:
    objective: str
    variables: list[Variable]
    baseline_reason: str = ""  # why the plan's baseline gave no objective ("" when it did)
    evaluations: int = 0
    seconds: float = 0.0
    elapsed: list[float] = field(default_factory=list)  # each evaluation's own seconds
    at_once: int = 1

    def of(self, verdict: str) -> list[Variable]:
        return [v for v in self.variables if v.verdict == verdict]

    def says(self) -> str:
        """One plain line for run.log."""
        head = (f"FI moved each design variable once before the search ({self.evaluations} evaluation(s), "
                f"{self.seconds:.0f} s)")
        if self.baseline_reason:
            head += f"; the plan's baseline design gave no objective ({self.baseline_reason})"
        counts = []
        for verdict, words in (("moves", "move the objective"), ("no_change", "showed no change of it"),
                               ("not_moved", "were not moved"), ("not_tested", "could not be judged"),
                               ("not_measured", "were not measured")):
            names = [v.name for v in self.of(verdict)]
            if names:
                counts.append(f"{len(names)} {words} ({', '.join(f'`{n}`' for n in names[:6])}"
                              f"{', ...' if len(names) > 6 else ''})")
        return head + (": " + "; ".join(counts) if counts else "")

    def notice(self) -> str:
        """The line for plan.md and run.log when a variable showed no change of the objective, else ``""``."""
        flat = self.of("no_change")
        if not flat:
            return ""
        names = ", ".join(f"`{v.name}`" for v in flat)
        return (f"FI's small run moved each design variable before the search: {names} changed nothing of the objective "
                f"`{self.objective}` at {'its' if len(flat) == 1 else 'their'} ends. That does not show the simulation "
                f"ignores {'it' if len(flat) == 1 else 'them'}; every variable stays in the search.")

    def for_reader(self) -> str:
        """What the code's reader is shown: one sentence per variable."""
        if not self.variables:
            return ""
        lines = [f"FI moved each design variable alone through the search's own entry before this reading "
                 f"(the objective `{self.objective}` and each limit judged on their own):"]
        if self.baseline_reason:
            lines.append(f"- the plan's baseline design gave no objective: {self.baseline_reason}")
        lines += [f"- {v.sentence(self.objective)}" for v in self.variables]
        return "\n".join(lines)

    def record(self) -> dict[str, Any]:
        return {"says": self.says(), "notice": self.notice(), "objective": self.objective,
                "baseline_reason": self.baseline_reason, "evaluations": self.evaluations, "at_once": self.at_once,
                "seconds": round(self.seconds, 2), "elapsed_s": [round(x, 3) for x in self.elapsed],
                "variables": [v.record() for v in self.variables], "for_reader": self.for_reader()}


def _same(a: Any, b: Any) -> bool:
    return _search._same_result(a, b)


def designs(block: dict[str, Any]) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[Variable],
                                           list[list[tuple[Point, int]]]]:
    """Every design the pass evaluates, known before any answer: ``(cells, variables, where)``. ``cells``: ``(part's
    block, cell)`` without repeats, the plan's baseline first; ``where[i]``: for variable ``i``, each of its points with
    the index of its cell."""
    space = _search.Space(block["design_variables"])
    cells: list[tuple[dict[str, Any], dict[str, Any]]] = []
    index: dict[str, int] = {}

    def at(sub: dict[str, Any], design: dict[str, Any]) -> int:
        cell = _search.cell_of(sub, design)
        key = json.dumps(cell, sort_keys=True, default=str)
        if key not in index:
            index[key] = len(cells)
            cells.append((sub, cell))
        return index[key]

    at(*part_block(block, None))
    variables: list[Variable] = []
    where: list[list[tuple[Point, int]]] = []
    for variable in block["design_variables"]:
        name = str(variable["name"])
        part, held = part_for(block, name)
        out = Variable(name, (str(part.get("name") or "") or f"sub-study {parts(block).index(part) + 1}") if part else "",
                       held)
        variables.append(out)
        points: list[tuple[Point, int]] = []
        where.append(points)
        if held:
            continue
        sub, start = part_block(block, part)
        moved = moves_of(space, variable, start)
        out.ends = [Point(label, "not_moved" if space.key(design) == space.key(start) else "not_measured")
                    for label, design in moved]
        if all(e.status == "not_moved" for e in out.ends):
            continue
        points.append((out.start, at(sub, start)))
        points += [(e, at(sub, design)) for e, (_label, design) in zip(out.ends, moved) if e.status != "not_moved"]
    return cells, variables, where


async def measure(block: dict[str, Any], evaluate: Evaluate, *, budget_s: float,
                  clock: Callable[[], float] = time.monotonic) -> Moves:
    """The pass on ``block`` (see the module docstring): every design asked for at once through ``evaluate`` (none
    depends on another's answer), within ``budget_s``; each then judged by :func:`core.optimise_search.judge`."""
    began = clock()
    cells, variables, where = designs(block)
    moves = Moves(str(block["objective"]["quantity"]), variables)
    answers = list(await evaluate([cell for _sub, cell in cells], float(budget_s)))
    answers += [{"stop": "time"}] * (len(cells) - len(answers))
    judged: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
    for (sub, _cell), answer in zip(cells, answers):
        answer = answer if isinstance(answer, dict) else {"failed": "no answer for this design"}
        if answer.get("stop"):
            judged.append((None, None))  # cut off by the time: not an answer
            continue
        moves.evaluations += 1
        if isinstance(answer.get("elapsed_s"), (int, float)):
            moves.elapsed.append(float(answer["elapsed_s"]))
        judged.append((answer, _search.judge(sub, answer.get("values") if "values" in answer else None,
                                             str(answer.get("failed") or ""))))
    base = judged[0][1]
    if base is not None and base["status"] != "ok":
        moves.baseline_reason = str(base.get("reason") or "no reason given")
    for points in where:
        for point, k in points:
            point.take(*judged[k])
    moves.seconds = clock() - began
    return moves


# --- in a quest ------------------------------------------------------------------------------------------------------


def key_of(small_key: str, block: dict[str, Any]) -> str:
    """What the pass depends on: the small run's key (the simulation, its package, its settings) and the search block."""
    text = json.dumps({k: block.get(k) for k in ("design_variables", "baseline", "fixed", "sub_studies", "grid", "objective",
                                                  "constraints", "numerical_settings", "in_each_run")},
                      sort_keys=True, default=str)
    return hashlib.sha256(f"{small_key}|{text}".encode()).hexdigest()[:20]


async def run(executor: Any, python: Any, quest_root: Path, module: str, block: dict[str, Any], *, timeout_s: float,
              env: dict[str, str] | None, thresholds: dict[str, Any] | None, at_once: int = 1, log: Any = None,
              notes: list[str] | None = None) -> Moves | None:
    """The pass in a throwaway copy of the quest (:func:`core.outside_run.observation_copy`), every evaluation exactly as
    the search makes it (:func:`core.optimise.evaluate_many`: one run each, the search's seeds, up to ``at_once`` at the
    same time). ``None`` when the simulation has no study entry or the copy cannot be made."""
    from . import optimise as _optimise
    from . import outside_run as _outside
    from . import trial_runner as _trials

    quest_root = Path(quest_root)
    entry = _trials.study_entry(_trials.entries(quest_root / module))
    if not entry:
        return None
    inside = _search.in_each_run(block) or None
    try:
        with _outside.observation_copy(quest_root, notes=notes, log=log) as work:
            async def evaluate(cells: list[dict[str, Any]], budget_s: float) -> list[dict[str, Any]]:
                got = await _optimise.evaluate_many(
                    executor, python, work, module, entry, cells, at_once=max(1, int(at_once)), runs=1, base_seed=0,
                    deadline=time.monotonic() + budget_s, env=env, thresholds=thresholds, log=log, in_each_run=inside)
                return [answer for answer, _stderr, _load in got]

            moves = await measure(block, evaluate, budget_s=float(timeout_s))
            moves.at_once = max(1, int(at_once))
            return moves
    except OSError:
        return None


def read_record(quest_root: Path | str) -> dict[str, Any]:
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _plan_line(engine: Any, line: str) -> None:
    """``line`` once in plan.md (the section of checks already made), as a version by the engine."""
    from . import engine as _e
    from . import plan as _plan

    path = _plan.plan_path(engine.quest_root)
    try:
        text = path.read_text(encoding="utf-8")
        if line in text:
            return
        edited = _plan.add_to_section(text, _e._PART_HEADING, ["", f"- {line}"])
        _plan.note_edit(engine.quest_root, text)
        path.write_text(edited, encoding="utf-8")
        _plan.record_version(engine.quest_root, edited, by="engine",
                             note="FI moved each design variable before the search; nothing was changed")
    except OSError as e:
        engine._log.warning("[execute] FI could not add that line to plan.md: %r", e)


async def disclose(engine: Any, state: Any, py: Any, exec_env: Any, small_record: dict[str, Any] | None) -> None:
    """After FI's small run of a search, when it did not crash: the pass (kept in :data:`RECORD` for the same code and
    block), its line in run.log, a line in plan.md for a variable that showed no change, and its sentences added to what
    the code's reader is shown. Never a stop: whatever goes wrong here is a warning and the quest goes on."""
    try:
        if not small_record or small_record.get("crashed"):
            return
        from . import optimise as _optimise
        from . import small_run as _small
        from . import split_run as _split_run
        from . import trial_runner as _trials

        protocol = engine._protocol_block(state) or {}
        block = _optimise.block_of(protocol)
        if block is None or not block.get("design_variables") or not isinstance(block.get("baseline"), dict):
            return
        simulate = engine.quest_root / "code" / _split_run.SIMULATE_NAME
        key = key_of(str(small_record.get("key") or ""), block)
        record = read_record(engine.quest_root)
        if record.get("key") != key:
            thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None
            got = await run(engine.executor, py, engine.quest_root, simulate.relative_to(engine.quest_root).as_posix(), block,
                            timeout_s=_small.budget(engine.config.execution.timeout_s), env=dict(exec_env or {}),
                            thresholds=thresholds, log=engine._log,
                            at_once=_trials.settings_at_once(engine.config.execution.settings_at_once))
            if got is None:
                return
            record = {"key": key, **got.record()}
            path = engine.quest_root / RECORD
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
            engine._log.info("[execute] %s", record["says"])
            if record.get("notice"):
                engine._log.warning("[execute] %s", record["notice"])
                _plan_line(engine, str(record["notice"]))
        reader = str(record.get("for_reader") or "")
        note = str(getattr(engine, "_small_run_note", "") or "")
        if reader and reader not in note:
            engine._small_run_note = (note + "\n" + reader).strip()
    except Exception as exc:  # noqa: BLE001 -- a disclosure that could not be made never stops the quest
        log = getattr(engine, "_log", None)
        if log is not None:
            log.warning("[execute] FI could not move the design variables before the search: %r", exc)
