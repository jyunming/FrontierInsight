"""What stays of the numeric oracle: reads of the results themselves, and the arithmetic a formula needs.

The paper's numbers are no longer compared with the results here. A result is a key FI fills in from its own records
(``core/paper_values.py``), and a number the writer types that is not a number of the plan or the topic is found by the
rule in ``core/typed_numbers.py``.

* :func:`flatten_numbers`: every numeric leaf of a results structure (used by the plausibility gate, the statistics check
  and the engine).
* :func:`trivial_reference_findings`: a reference quantity that is exactly 0 at every setting of a sweep. It is a check
  on the results, not on the paper: it is what a root finder returning the trivial root of its bracket looks like.
* :func:`latex_to_arithmetic`: LaTeX spelled as the calculator reads it (used by ``core/oracle_forms.py``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterator

# Values at or below this magnitude are skipped entirely. Small integers are
# overwhelmingly counts, indices, section numbers and figure references, and
# they collide with each other constantly.
MIN_MAGNITUDE = 1e-9

# How many settings a reference quantity must span before "it is zero at every
# one of them" means anything. Two points is a coincidence; a sweep is not.
TRIVIAL_MIN_SETTINGS = 3

# Leaf names that denote a computed reference value — the number the run is
# comparing itself against, rather than a number it measured. These are the
# ones a trivial root destroys.
_REFERENCE_NAME = re.compile(
    r"(?:^|_)(?:"
    r"deterministic|theoretical|theory|analytic|analytical|closed_form|"
    r"exact|predicted|prediction|reference|asymptotic|ode|mean_field|limit"
    r")(?:_|$)",
    re.IGNORECASE,
)

# ...except when the name says it is a setting rather than a result. A
# tolerance, a horizon or a step count is *supposed* to be identical at every
# point of a sweep, and several of them are legitimately 0. Known weak point:
# this is a name list, so a setting named outside it is not recognised.
_SETTING_NAME = re.compile(
    r"(?:^|_)(?:"
    r"tol|rtol|atol|tolerance|seed|seeds|dpi|horizon|step|steps|dt|nfev|"
    r"resamples|replicates|iterations|iters|maxiter|bins|timeout|version|"
    r"residual|precision"
    r")(?:_|$)",
    re.IGNORECASE,
)


def _leaf_name(path: str) -> str:
    """The final field name of a flattened path, without any list index."""
    return path.rsplit(".", 1)[-1].split("[")[0]


def trivial_reference_findings(result_json: Any) -> list["Finding"]:
    """Reference quantities that are exactly 0 at every setting of the sweep.

    See the module docstring. Returns one finding per offending leaf name.
    Silent unless some *other* quantity in the same results varies: a result
    set in which nothing varies at all is a different failure (the degenerate
    run guard owns it) and flagging it here would only duplicate that.
    """
    leaves = list(flatten_numbers(result_json, keep_zero=True))
    if not leaves:
        return []

    grouped: dict[str, list[tuple[str, float]]] = {}
    for path, value in leaves:
        grouped.setdefault(_leaf_name(path), []).append((path, value))

    something_varies = any(
        len({v for _, v in items}) > 1 for items in grouped.values()
    )
    if not something_varies:
        return []

    out: list[Finding] = []
    for name, items in sorted(grouped.items()):
        if len(items) < TRIVIAL_MIN_SETTINGS:
            continue
        if {v for _, v in items} != {0.0}:
            continue
        if not _REFERENCE_NAME.search(name) or _SETTING_NAME.search(name):
            continue
        paths = [p for p, _ in items]
        shown = ", ".join(paths[:3]) + (", …" if len(paths) > 3 else "")
        out.append(
            Finding(result_path=name, context=shown)
        )
    return out


@dataclass(frozen=True)
class Finding:
    """A reference quantity of the results that is exactly 0 at every setting."""

    result_path: str   # the leaf name
    context: str       # the first paths it was found at
    kind: str = "trivial_reference"

    def describe(self) -> str:
        return (
            f"`{self.result_path}` is exactly 0 at every one of its "
            f"settings ({self.context}). A reference value that does not "
            f"vary across the sweep meant to vary it is what a root "
            f"finder returning the trivial root on its bracket endpoint "
            f"looks like — check the bracket before the paper describes "
            f"this as a limit"
        )


_LATEX_NOISE = re.compile(r"\\(?:left|right|displaystyle|[,!;: ])|\$")
_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_ARITH_TOKEN = re.compile(
    r"\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+|[A-Za-z_][A-Za-z_0-9]*|\*\*|[-+*/()]|\S"
)
_RELATION_ANY = re.compile(r"=|[\u2248\u2243\u223c~]|\\(?:approx|simeq|sim)\b")


_GREEK = (
    "alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|vartheta|iota|kappa|lambda|mu|nu|xi|omicron|rho|varrho|"
    "sigma|tau|upsilon|phi|varphi|chi|psi|omega|Gamma|Delta|Theta|Lambda|Xi|Sigma|Upsilon|Phi|Psi|Omega"
)
_GREEK_RE = re.compile(r"\\(" + _GREEK + r")(?![A-Za-z])")


def latex_to_arithmetic(s: str, *, names: bool = False) -> str:
    """``s`` (one formula, no relation sign) with its LaTeX spelled as the calculator reads it: ``\\frac{a}{b}`` ->
    ``((a)/(b))``, ``\\cdot`` / ``\\times`` -> ``*``, ``\\sqrt{x}`` -> ``sqrt(x)``, ``\\pi`` -> ``pi``, ``^`` -> ``**``,
    braces -> parentheses. ``names``: a formula of named settings, where ``\\\\lambda`` (a backslash written twice, as it
    is in a YAML string) and ``\\lambda`` both become the plain name ``lambda`` (a Greek letter is a name, never a
    guess of a value), and a name is kept as it is."""
    if names:
        s = re.sub(r"\\{2,}(?=[A-Za-z])", r"\\", s)
        s = _GREEK_RE.sub(lambda m: f" {m.group(1)} ", s)
    s = _LATEX_NOISE.sub(" ", s)
    s = re.sub(r"\\(?:cdot|times)(?![A-Za-z])|[\u00d7\u00b7]", "*", s)
    s = re.sub(r"\\div(?![A-Za-z])|\u00f7", "/", s)
    s = re.sub(r"\\pi(?![A-Za-z])|\u03c0", " pi ", s)
    s = re.sub(r"\\ln(?![A-Za-z])", " ln ", s)
    s = re.sub(r"\\(exp|sinh|cosh|tanh|sin|cos|tan)(?![A-Za-z])", r" \1 ", s)
    s = re.sub(r"\^\s*\{([^{}]*)\}", r"**(\1)", s).replace("^", "**")
    for _ in range(10):
        t = _SQRT.sub(r"sqrt(\1)", _FRAC.sub(r"((\1)/(\2))", s))
        if t == s:
            break
        s = t
    s = re.sub(r"\u221a\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", s).replace("\u221a", "sqrt")
    s = s.replace("{", "(").replace("}", ")")
    if not names:  # in a formula an argument list is `max(3,400)`, not a thousands separator
        s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return s


def flatten_numbers(
    obj: Any, path: str = "", *, keep_zero: bool = False,
) -> Iterator[tuple[str, float]]:
    """Yield every numeric leaf of a JSON-ish structure as (path, value).

    Booleans are excluded: ``True`` is an ``int`` in Python and a flag is
    not a measurement. Values within ``MIN_MAGNITUDE`` of zero are left out
    unless ``keep_zero``: in a paper a zero matches every other zero, but in a
    range check a quantity computed as exactly 0 can be the bug.
    """
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj) and (keep_zero or abs(obj) > MIN_MAGNITUDE):
            yield path or "<root>", float(obj)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten_numbers(
                v, f"{path}.{k}" if path else str(k), keep_zero=keep_zero,
            )
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from flatten_numbers(v, f"{path}[{i}]", keep_zero=keep_zero)
