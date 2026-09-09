"""Range and unit assertions on a run's results, declared by its design.

Why this is not just a wider degeneracy check
=============================================
``_is_degenerate_result`` fires only when *every* numeric leaf is within
1e-12 of zero. That catches a simulation that produced nothing. It cannot
catch the far more common failure: a run that produced plausible-looking
numbers from wrong physics — a unit error, a factor of two, a sign flip.
Those exit 0, parse fine, and sail through every downstream gate.

The missing ingredient is not a cleverer heuristic. It is knowledge of
what the outputs are *allowed* to be, and only the design stage has that:
it chose the working point, so it knows k1 lives in roughly [0.25, 0.5],
that a CD is positive, that a normalised contrast cannot exceed 1.

So the design declares the bounds and this module enforces them, in code.
An assertion the design did not make is not checked — silence here means
"nothing was claimed", never "everything is fine".

Where the bounds come from later
================================
When a quest calls a trusted simulation skill rather than generating its
physics from scratch, the skill carries its own valid domain and can
supply these assertions without the design having to restate them. This
module takes assertions from a list; it does not care who wrote them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

# Assertion paths are matched against the flattened result_json keys used
# by ``numeric_oracle.flatten_numbers`` — "metrics.contrast", "cd_nm",
# "sweep[0].nils". A trailing ".*" or a bare name matches any leaf whose
# final segment equals it, so a design can say "cd_nm" without knowing how
# deeply the script nested it.
_LEAF = re.compile(r"[^.\[\]]+$")


@dataclass(frozen=True)
class Assertion:
    """One declared bound on a result value."""

    path: str
    min: float | None = None
    max: float | None = None
    unit: str = ""
    reason: str = ""

    def describe(self) -> str:
        if self.min is not None and self.max is not None:
            rng = f"[{self.min:g}, {self.max:g}]"
        elif self.min is not None:
            rng = f">= {self.min:g}"
        elif self.max is not None:
            rng = f"<= {self.max:g}"
        else:
            rng = "finite"
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.path} must be {rng}{unit}"


@dataclass(frozen=True)
class Violation:
    path: str
    value: float
    assertion: Assertion

    def describe(self) -> str:
        why = f" ({self.assertion.reason})" if self.assertion.reason else ""
        return (
            f"{self.path} = {self.value:g} violates "
            f"{self.assertion.describe()}{why}"
        )


def parse_assertions(design: Any) -> list[Assertion]:
    """Read ``result_assertions`` off a design spec.

    Tolerant by construction: this is LLM output, and a malformed entry
    must be dropped rather than raise. A design that declares nothing
    yields nothing, which is the correct no-op.
    """
    if not isinstance(design, dict):
        return []
    raw = design.get("result_assertions")
    if not isinstance(raw, list):
        return []

    out: list[Assertion] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue

        def _num(key: str) -> float | None:
            v = item.get(key)
            if isinstance(v, bool) or v is None:
                return None
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return f if math.isfinite(f) else None

        lo, hi = _num("min"), _num("max")
        if lo is None and hi is None:
            continue  # an assertion that asserts nothing
        if lo is not None and hi is not None and lo > hi:
            continue  # inverted bounds are a typo, not a constraint
        out.append(
            Assertion(
                path=path,
                min=lo,
                max=hi,
                unit=str(item.get("unit") or "").strip(),
                reason=str(item.get("reason") or "").strip(),
            )
        )
    return out


def _matches(assertion_path: str, leaf_path: str) -> bool:
    """Does an assertion path address this flattened result path?

    Exact match wins. Otherwise the assertion may name just the final
    segment ("cd_nm" matching "sweep[2].cd_nm"), so a design does not have
    to predict the script's nesting.
    """
    if assertion_path == leaf_path:
        return True
    a = assertion_path.rstrip(".*")
    if a == leaf_path:
        return True
    m = _LEAF.search(leaf_path)
    return bool(m and m.group(0) == a)


def violations(result_json: Any, assertions: list[Assertion]) -> list[Violation]:
    """Every declared bound the result actually breaks."""
    if not assertions:
        return []
    from core.numeric_oracle import flatten_numbers

    leaves = list(flatten_numbers(result_json))
    if not leaves:
        return []

    out: list[Violation] = []
    for a in assertions:
        for path, value in leaves:
            if not _matches(a.path, path):
                continue
            if a.min is not None and value < a.min:
                out.append(Violation(path, value, a))
            elif a.max is not None and value > a.max:
                out.append(Violation(path, value, a))
    return out


def check_design(result_json: Any, design: Any) -> list[Violation]:
    """Convenience: parse a design's assertions and apply them."""
    return violations(result_json, parse_assertions(design))
