"""Warnings from a numerical run that a paper must not be written over.

A script can exit 0 and print a legal-looking ``RESULT_JSON`` while its numerics were telling it something was wrong. One
stored run passed ``abs_tol`` and ``rel_tol`` to ``scipy.integrate.solve_ivp`` (which takes ``atol`` and ``rtol``): SciPy
says the arguments "have no effect for a chosen solver", the run said it used a tolerance of 1e-6, and it used the
default. None of the existing checks reads a warning: ``result_assertions`` see only the results, the audits only the paper.

This module reads what the script wrote to stderr (and the results it printed) for the warnings that mean the numbers may
not be what the script claims, and names each one. It needs no model and no engine state; the engine decides what to do
(:mod:`core.engine`: ask for a repair, then stop the quest).

What counts, and only this, so that a warning about a font or a deprecation never stops a quest:

* ``overflow``, ``invalid value`` or ``divide by zero`` encountered in an array operation (a ``RuntimeWarning`` from NumPy);
* a solver or optimiser that did not converge: ``IntegrationWarning``, ``OptimizeWarning``, ``ConvergenceWarning``,
  ``LinAlgWarning``, a maximum number of iterations or subdivisions reached, an iteration "not making good progress";
* an argument that had no effect: "have no effect for a chosen solver", an unexpected or unrecognised keyword or option,
  "will be ignored", "is ignored";
* ``ComplexWarning`` (an imaginary part silently discarded);
* a ``NaN`` or an infinity in the results the script printed, when it is a result the plan declares
  (``design.protocol.metrics``). Any other non-finite number (a diagnostic the plan does not report) is disclosed
  (:func:`disclosed_non_finite`), not repaired: a NaN in a number nobody reports must not stop every result. Which
  printed number is a declared result is told by its name (a key on its path equal to a result's id), so a NaN in a
  number named otherwise (``rate_se`` beside the declared ``rate``) is disclosed too; the study contract's result roles
  replace that link. FI fills no non-finite number into the paper (:mod:`core.paper_values`). A plan that declares no
  result keeps every non-finite number.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

_WARNING_LINE = re.compile(r"(?P<cls>[A-Za-z]*Warning)\s*:\s*(?P<text>.*)$")
_IGNORED_CLASSES = ("DeprecationWarning", "FutureWarning", "PendingDeprecationWarning", "ResourceWarning", "SyntaxWarning")
_BENIGN_TEXT = re.compile(r"tight_layout|constrained_layout|font|glyph|matplotlib|cmap|colorbar|legend|no artists|Tight", re.IGNORECASE)
_RUNTIME = re.compile(r"(overflow encountered in|invalid value encountered in|divide by zero encountered in)\s*([\w.]*)", re.IGNORECASE)
_SOLVER_CLASSES = ("IntegrationWarning", "OptimizeWarning", "ConvergenceWarning", "LinAlgWarning")
_SOLVER_TEXT = re.compile(
    r"maximum number of (iterations|subdivisions)|iteration is not making good progress|did not converge|failed to converge"
    r"|singular matrix|ill-conditioned",
    re.IGNORECASE,
)
_IGNORED_ARG = re.compile(
    r"have no effect|has no effect|unexpected keyword|unrecognized (argument|option|parameter|keyword)|unrecognised "
    r"|will be ignored|is ignored|are ignored|not used by",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumericWarning:
    kind: str  # "runtime", "solver", "ignored_argument", "complex" or "non_finite"
    text: str

    def describe(self) -> str:
        return {
            "runtime": "a numerical warning from NumPy",
            "solver": "a solver or optimiser did not converge",
            "ignored_argument": "an argument had no effect",
            "complex": "an imaginary part was discarded",
            "non_finite": "a result is not finite",
        }.get(self.kind, self.kind) + f": {self.text}"


def _non_finite_leaves(result: Any, prefix: str = "", keys: tuple[str, ...] = ()) -> list[tuple[str, tuple[str, ...], float]]:
    """Every non-finite number in ``result``: its path as written in a warning, the dict keys on that path (verbatim, so a
    key holding a dot or a bracket is one key), and the value."""
    found: list[tuple[str, tuple[str, ...], float]] = []
    if isinstance(result, dict):
        for key, value in result.items():
            found += _non_finite_leaves(value, f"{prefix}.{key}" if prefix else str(key), keys + (str(key),))
    elif isinstance(result, list):
        for index, value in enumerate(result[:200]):
            found += _non_finite_leaves(value, f"{prefix}[{index}]", keys)
    elif isinstance(result, float) and not math.isfinite(result):
        found.append((prefix, keys, result))
    return found


def _non_finite(result: Any, prefix: str = "") -> list[str]:
    return [f"{path} is {value}" for path, _keys, value in _non_finite_leaves(result, prefix)]


def declared_results(protocol: Any) -> set[str]:
    """The ids of the results the plan declares (``design.protocol.metrics``)."""
    metrics = protocol.get("metrics") if isinstance(protocol, dict) else None
    return {str(m["id"]) for m in metrics or [] if isinstance(m, dict) and m.get("id")}


def _names_result(keys: tuple[str, ...], results: set[str]) -> bool:
    return any(key in results for key in keys)


def disclosed_non_finite(result_json: Any, results: set[str]) -> list[str]:
    """The non-finite numbers of ``result_json`` that are none of the plan's declared ``results``: disclosed, never
    repaired for (none when the plan declares no result)."""
    if not results:
        return []
    return [f"{path} is {value}" for path, keys, value in _non_finite_leaves(result_json) if not _names_result(keys, results)]


def scan(stderr: str, result_json: Any = None, *, limit: int = 8, results: set[str] | None = None) -> list[NumericWarning]:
    """The numeric warnings in ``stderr`` and non-finite numbers in ``result_json``, each once, at most ``limit``. With
    ``results`` (the plan's declared results, not empty) a non-finite number counts only when it is one of them;
    without, every one counts."""
    seen: set[tuple[str, str]] = set()
    out: list[NumericWarning] = []

    def add(kind: str, text: str) -> None:
        key = (kind, re.sub(r"\s+", " ", text.strip())[:160])
        if key not in seen and len(out) < limit:
            seen.add(key)
            out.append(NumericWarning(kind, key[1]))

    for line in (stderr or "").splitlines():
        match = _WARNING_LINE.search(line)
        if match is None:
            continue
        cls, text = match.group("cls"), match.group("text")
        if cls in _IGNORED_CLASSES or _BENIGN_TEXT.search(text):
            continue
        run = _RUNTIME.search(text)
        if cls == "RuntimeWarning" and run is not None:
            add("runtime", f"{run.group(1).strip()} {run.group(2)}".strip())
        elif cls in _SOLVER_CLASSES or (cls in ("RuntimeWarning", "UserWarning") and _SOLVER_TEXT.search(text)):
            add("solver", f"{cls}: {text}")
        elif cls == "ComplexWarning":
            add("complex", text)
        elif cls in ("UserWarning", "RuntimeWarning") and _IGNORED_ARG.search(text):
            add("ignored_argument", text)
    for path, keys, value in _non_finite_leaves(result_json):
        if not results or _names_result(keys, results):
            add("non_finite", f"{path} is {value}")
    return out


#: What a repair history entry (``exec_reflect_history``) says it was for when it was made for the run's numeric
#: warnings: those repairs have a budget of their own and do not move the repair counter (core/engine.py, execute_reflect).
REPAIR_FOR = "numeric_warnings"


def repairs_for_warnings(history: Any) -> int:
    """How many repairs in an ``exec_reflect_history`` were made for the run's numeric warnings."""
    return sum(1 for h in history or [] if isinstance(h, dict) and h.get("for") == REPAIR_FOR)


def repairs_made(state: Any) -> int:
    """Every repair of the run's scripts made so far: those the repair counter (``exec_reflect_iter``) counts and those
    made for the numeric warnings, which have a budget of their own."""
    state = state or {}
    return int(state.get("exec_reflect_iter", 0) or 0) + repairs_for_warnings(state.get("exec_reflect_history"))


def directive(found: list[NumericWarning]) -> str:
    """What stands where a traceback would in the repair request."""
    return (
        "This script ran to the end and printed a RESULT_JSON, but its numerics reported problems, and a paper must not be "
        "written over numbers whose computation warned:\n"
        + "\n".join(f"- {w.describe()}" for w in found)
        + "\n\nFind the cause before changing anything, and fix the computation: a solver that did not converge needs a "
        "different method, step or tolerance (and the tolerance arguments must be the ones the function takes: for SciPy's "
        "solve_ivp they are `atol` and `rtol`, not `abs_tol` and `rel_tol`); an argument that had no effect must be the "
        "right argument or be removed, and if the script claims a setting it must actually use it; an overflow, invalid "
        "value or division by zero means a quantity is out of range or a case is unguarded, and it must be handled where it "
        "arises. Do NOT silence the warning (`warnings.filterwarnings`, `np.seterr(all='ignore')`) to make it disappear. "
        "Only when the operation is intended and harmless may the script guard it with `np.errstate(...)` around that one "
        "operation, and it must say why in a comment and in `patch_summary`. A NaN or an infinity in a result is a finding, "
        "not a value: emit null for it with a flag saying why.\n\n"
        "Keep everything else unchanged: the same functions, outputs and figures, the handling of FI_PILOT, FI_ORACLE and "
        "FI_REPLICATE_SEED, and the same final RESULT_JSON line. Return only the functions you change in `functions` (each one whole; FI keeps every other line), one sentence in "
        "`patch_summary`, and leave `give_up_reason` empty."
    )
