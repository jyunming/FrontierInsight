"""Deterministic check that the paper's numbers match the experiment's.

Why this exists
===============
Every other correctness gate in FI terminates in a language model reading
text. ``claim_check`` asks one LLM call whether a number "appears" in a
JSON blob rendered as text; there is no arithmetic comparison anywhere,
and the check degrades silently to a no-op when its JSON fails to parse.
``analyze`` is the single translation point from ``result_json`` into
prose, and nothing downstream re-derives it.

So a mis-transcription — the run produced 2.14, the paper says 2.41 —
passes every gate. This module is the one check that does not route back
through a model: it is arithmetic.

What it does NOT do
===================
It does **not** demand that every number in the paper trace to
``result_json``. Papers legitimately quote parameters from the design,
figures from the literature, dates, counts and section numbers; flagging
those would produce constant false positives, and a false positive here
costs a whole revision iteration because the finding is a blocking
must-flag.

Instead it looks for the *signature of a transcription error*: a paper
number that sits **near** a real result without matching it, where the
gap cannot be explained by rounding. A number far from every result is
assumed to come from somewhere else and is ignored. A number that rounds
correctly is correct.

Two signals, strongest first:

``transposed``
    Same digits, different order (2.41 vs 2.14; 0.045 vs 0.054). Almost
    never a coincidence, and the classic way a number gets copied wrong.

``near_miss``
    Within ``NEAR_REL`` of a result value but outside what rounding to
    the paper's own precision would allow.

Both are reported with the JSON path they contradict, so the writer node
gets told which number to fix rather than "something is wrong".
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

# A paper number must land within this relative distance of a result value
# before it is even considered a candidate mis-transcription. Beyond it, the
# number is assumed to be unrelated (a parameter, a literature figure, a
# date) and is ignored. Deliberately tight: the goal is to catch copies that
# went wrong, not to audit provenance.
NEAR_REL = 0.25

# Values at or below this magnitude are skipped entirely. Small integers are
# overwhelmingly counts, indices, section numbers and figure references, and
# they collide with each other constantly.
MIN_MAGNITUDE = 1e-9

# Numbers written with fewer than this many significant digits carry too
# little information to distinguish a transcription error from a rounded
# quote. "about 2" vs 2.14 is not evidence of anything.
MIN_SIG_DIGITS = 2

# Contexts that are never measurements, matched against the text immediately
# before a number. Kept narrow and literal; a regex that tries to be clever
# here is how false positives get in.
_CONTEXT_SKIP = re.compile(
    r"(?:"
    r"figure|fig\.|table|tab\.|section|sect\.|chapter|eq\.|equation|"
    r"reference|ref\.|page|p\.|pp\.|doi|arxiv|isbn|"
    r"\[|version|v"
    r")\s*$",
    re.IGNORECASE,
)

# Number tokens: optional sign, digits, optional decimal, optional exponent.
# Thousands separators are handled by stripping commas between digit groups.
_NUMBER = re.compile(
    r"(?<![\w.])"
    r"(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"(?:[eE]([-+]?\d+))?"
    r"(?![\w])"
)

# Markdown constructs whose numbers are structural rather than claimed.
_STRIP_BLOCKS = [
    re.compile(r"^```.*?^```", re.S | re.M),      # fenced code
    re.compile(r"^\s*\|.*\|\s*$", re.M),           # table rows (kept out: see note)
    re.compile(r"!\[[^\]]*\]\([^)]*\)"),           # image embeds
    re.compile(r"^#+ .*$", re.M),                  # headings
    re.compile(r"^\s*\d+\.\s+[A-Z][^\n]*\(\d{4}\)", re.M),  # reference entries
]


@dataclass(frozen=True)
class Finding:
    """One number in the paper that contradicts a computed result."""

    kind: str          # "transposed" | "near_miss"
    paper_value: float
    result_value: float
    result_path: str
    context: str       # surrounding prose, trimmed
    rel_error: float

    def describe(self) -> str:
        pct = self.rel_error * 100
        lead = (
            "digits transposed"
            if self.kind == "transposed"
            else f"off by {pct:.1f}%"
        )
        return (
            f"paper says {_fmt(self.paper_value)} but "
            f"`{self.result_path}` = {_fmt(self.result_value)} ({lead}) "
            f"— “{self.context}”"
        )


@dataclass
class OracleReport:
    findings: list[Finding] = field(default_factory=list)
    paper_numbers: int = 0
    result_numbers: int = 0
    skipped: bool = False
    skip_reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason or None,
            "paper_numbers_checked": self.paper_numbers,
            "result_numbers": self.result_numbers,
            "findings": [
                {
                    "kind": f.kind,
                    "paper_value": f.paper_value,
                    "result_value": f.result_value,
                    "result_path": f.result_path,
                    "rel_error": round(f.rel_error, 6),
                    "context": f.context,
                    "message": f.describe(),
                }
                for f in self.findings
            ],
        }


def _fmt(v: float) -> str:
    """Render a float the way a paper would, without trailing noise."""
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:g}"


def flatten_numbers(obj: Any, path: str = "") -> Iterator[tuple[str, float]]:
    """Yield every numeric leaf of a JSON-ish structure as (path, value).

    Booleans are excluded: ``True`` is an ``int`` in Python and a flag is
    not a measurement.
    """
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj) and abs(obj) > MIN_MAGNITUDE:
            yield path or "<root>", float(obj)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten_numbers(v, f"{path}.{k}" if path else str(k))
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from flatten_numbers(v, f"{path}[{i}]")


def _significant_digits(token: str) -> int:
    """Count significant digits in the number as the paper wrote it."""
    t = token.lstrip("-+").replace(",", "")
    if "." in t:
        whole, frac = t.split(".", 1)
        whole = whole.lstrip("0")
        return len(whole) + len(frac) if whole else len(frac.lstrip("0")) or len(frac)
    return len(t.strip("0")) or 1


def _digit_bag(v: float) -> str:
    """Sorted significant digits, for transposition detection."""
    s = f"{abs(v):.10g}".replace(".", "").replace("-", "").lstrip("0").rstrip("0")
    return "".join(sorted(s)) if s else ""


def _rounds_to(paper: float, actual: float, token: str) -> bool:
    """True when `actual`, rounded to the paper's own precision, is `paper`."""
    t = token.replace(",", "")
    decimals = len(t.split(".", 1)[1]) if "." in t else 0
    try:
        return round(actual, decimals) == round(paper, decimals)
    except (ValueError, OverflowError):
        return False


def extract_paper_numbers(text: str) -> list[tuple[float, str, str]]:
    """Return (value, raw token, surrounding context) for prose numbers.

    Code blocks, headings, image embeds and reference entries are removed
    first: their numbers are structural, not claimed. Markdown tables are
    *kept* — a results table is exactly where a mis-transcribed number
    does the most damage.
    """
    cleaned = text
    for pat in _STRIP_BLOCKS[:1] + _STRIP_BLOCKS[2:]:  # keep table rows
        cleaned = pat.sub(" ", cleaned)

    out: list[tuple[float, str, str]] = []
    for m in _NUMBER.finditer(cleaned):
        token, exp = m.group(1), m.group(2)
        before = cleaned[max(0, m.start() - 24): m.start()]
        if _CONTEXT_SKIP.search(before):
            continue
        if _significant_digits(token) < MIN_SIG_DIGITS:
            continue
        try:
            value = float(token.replace(",", "") + (f"e{exp}" if exp else ""))
        except ValueError:
            continue
        if not math.isfinite(value) or abs(value) <= MIN_MAGNITUDE:
            continue
        ctx = " ".join(
            cleaned[max(0, m.start() - 60): m.end() + 30].split()
        )
        out.append((value, token, ctx))
    return out


def check(paper_text: str, result_json: Any) -> OracleReport:
    """Compare a paper's prose numbers against the computed results."""
    report = OracleReport()

    results = list(flatten_numbers(result_json))
    report.result_numbers = len(results)
    if not results:
        report.skipped = True
        report.skip_reason = "result_json holds no numeric values"
        return report
    if not paper_text.strip():
        report.skipped = True
        report.skip_reason = "paper text is empty"
        return report

    numbers = extract_paper_numbers(paper_text)
    report.paper_numbers = len(numbers)

    seen: set[tuple[float, str]] = set()
    for value, token, ctx in numbers:
        best: tuple[float, str, float] | None = None  # (result, path, rel)
        for path, actual in results:
            if actual == value or _rounds_to(value, actual, token):
                best = None
                break  # an exact or correctly-rounded match clears it
            denom = max(abs(actual), abs(value))
            rel = abs(actual - value) / denom if denom else 0.0
            if rel <= NEAR_REL and (best is None or rel < best[2]):
                best = (actual, path, rel)
        if best is None:
            continue

        actual, path, rel = best
        key = (value, path)
        if key in seen:
            continue
        seen.add(key)

        kind = (
            "transposed"
            if _digit_bag(value) and _digit_bag(value) == _digit_bag(actual)
            else "near_miss"
        )
        report.findings.append(
            Finding(
                kind=kind,
                paper_value=value,
                result_value=actual,
                result_path=path,
                context=ctx,
                rel_error=rel,
            )
        )

    # Strongest signal first, so a truncated report still leads with the
    # finding most likely to be a real transcription error.
    report.findings.sort(key=lambda f: (f.kind != "transposed", -f.rel_error))
    return report
