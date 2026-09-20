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

Numbers that are not results are not compared
=============================================
A paper prints three kinds of number that sit near a result without being one,
and each is left out rather than compared:

* **A confidence level** -- the ``95`` in ``95% CI`` or ``Wilson 95%
  intervals``. It lands within ``NEAR_REL`` of any result near 90-100 (a count
  of 93, a population of 100) and was the single most frequent false finding
  across the stored quests. Only a number followed by ``%`` and then interval
  wording counts as a level; ``93% of runs`` is a result and is still checked.
* **A setting the run was given** -- ``R0 = 1.5`` printed next to an outcome
  that happens to be 1.33. ``declared_numbers`` collects the numbers the topic,
  the design's ``variables`` and its ``method`` state, and ``check`` does not
  report a ``near_miss`` on a number equal to one of them (at the paper's own
  precision). Deliberately NOT the design's ``hypothesis`` /
  ``expected_outcome``: those hold what the author *predicted* a result would
  be, and a paper that prints the prediction where the measured value belongs
  is the very mistake this check exists to find. A digit transposition is still
  reported for a declared number -- it is the signal least likely to be a
  coincidence. Known blind spot: a result that really is equal to a declared
  setting, and is misquoted as it, goes unreported.
* **A sign that was dropped** -- ``check`` reads U+2212 (``−``) as a minus, as
  it already read ``-``. Left alone, ``−0.114`` was compared as ``0.114`` and
  reported against an unrelated positive result.

A third signal does not read the paper at all
=============================================
``trivial_reference`` compares the results against *themselves*. A
closed-form reference value — the deterministic limit, the theoretical
prediction, the analytic solution — is computed by model-written code,
and the way that code fails is remarkably consistent: a root finder is
bracketed so the useless root sits on the bracket endpoint
(``brentq(f, 0, 1)`` where ``f(1) == 0``, or a bracket of
``[1e-10, 1.0]``). The solver returns the trivial root and the reference
comes out as exactly 0 at *every* point of the sweep — after which the
paper draws that flat line of zeros in a figure captioned "convergence to
the deterministic limit".

Nothing else catches it. The range gate only checks quantities the design
declared a bound for, and a design that names ``final_size`` does not
cover a leaf called ``deterministic_final_size``; a zero is inside
``[0, 1]`` anyway. So the signal here is the one the sweep itself
provides: **a reference quantity that does not vary at all across the
settings that were supposed to make it vary, and whose constant value is
zero.**

Only zero, deliberately. A correctly computed deterministic final size
does *not* depend on the population size, so a reference that is
legitimately identical across a one-dimensional sweep over N is exactly
what a convergence study plots. Flagging "identical" in general would
therefore fire on correct work; flagging "identically zero" fires on the
failure. Values the run uses as settings rather than results (tolerances,
horizons, seeds, step counts) are excluded by name, since a tolerance of
0.0 repeated at every point is configuration, not a computation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

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
            Finding(
                kind="trivial_reference",
                paper_value=0.0,
                result_value=0.0,
                result_path=name,
                context=shown,
                rel_error=0.0,
            )
        )
    return out

# Contexts that are never measurements, matched against the text immediately
# before a number. Kept narrow and literal; a regex that tries to be clever
# here is how false positives get in.
_CONTEXT_SKIP = re.compile(
    r"(?:"
    r"figure|fig\.|table|tab\.|section|sect\.|chapter|eq\.|equation|"
    r"reference|ref\.|page|p\.|pp\.|doi|arxiv|isbn|"
    r"\[|version|v"
    # A separator may sit between the keyword and the number. Anchoring on
    # ``\s*$`` alone meant "DOI 10.5281" was skipped but "DOI: 10.5281" was
    # not -- and every reference list writes the colon. A real quest flagged
    # three DOI registrant prefixes (10.5281 / 10.1088 / 10.1016) as
    # contradicted measurements because of this one character.
    r")\s*[:.\-–—=]?\s*$",
    re.IGNORECASE,
)

# LaTeX scientific notation, normalised to e-notation BEFORE numbers are
# extracted. Papers write ``$9.65 \times 10^{-6}$``; reading that as the bare
# mantissa ``9.65`` invents a claim the paper never made and then compares it
# against unrelated results. Observed doing exactly that: a paper stating
# ``1.21 \times 10^{-2}`` was reported as "paper says 1.21".
_LATEX_SCI = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|×|\\!)\s*10\s*\^\s*\{?\s*(-?\+?\d+)\s*\}?"
)

# Number tokens: optional sign, digits, optional decimal, optional exponent.
# Thousands separators are handled by stripping commas between digit groups.
#
# The trailing guard also refuses a token that is followed by ``.<digit>``.
# ``97.5th`` cannot be read whole (``5th`` is a word), so the pattern used to
# back off to ``97`` -- the integer part of a number nobody wrote -- and a
# percentile rank was reported against a count of 98. The same guard stops
# ``1.2.3`` being read as ``1.2``.
_NUMBER = re.compile(
    r"(?<![\w.])"
    r"(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"(?:[eE]([-+]?\d+))?"
    r"(?!\w|\.\d)"
)

# U+2212 MINUS SIGN, spelled as a code point so it cannot be mistaken for the
# hyphen next to it. Papers typeset a negative number with it.
_MINUS_SIGN = chr(0x2212)

# The text right after a number that marks it as a confidence LEVEL: ``95% CI``,
# ``95% Wilson interval``, ``95% bootstrap confidence interval``, LaTeX
# ``$95\%$ CI``. A level is a convention the author chose, never a result, yet
# ``95`` sits within 25% of any result between 72 and 126.
#
# ``%`` alone is not enough (``93% of runs`` is a measurement) and neither is
# any interval word after it: ``of`` may not stand between them, or
# ``94% of the intervals covered the truth`` -- an empirical coverage, which IS
# a result -- would be treated as a level. Listed words only; no ``coverage``,
# ``level`` or ``band``, which no stored paper needed and which read as results
# more often than as levels.
_LEVEL_AFTER = re.compile(
    r"^\s*\\?%\$?\s*[-–]?\s*"
    r"(?:(?!of\b)[A-Za-z]+[-\s]+){0,2}"
    r"(?:CIs?|confidence|credible|intervals?)\b",
    re.IGNORECASE,
)

# Markdown constructs whose numbers are structural rather than claimed.
_STRIP_BLOCKS = [
    re.compile(r"^```.*?^```", re.S | re.M),      # fenced code
    re.compile(r"^\s*\|.*\|\s*$", re.M),           # table rows (kept out: see note)
    # The References and Further reading sections, before the heading rule
    # below removes their headings: an entry can open with a lower-case name
    # ("van der Berg") or a year, which the entry rule misses.
    re.compile(
        r"^#{1,6}[ \t]*(?:references|further[ \t]+reading)[ \t]*$.*?(?=^#{1,6}[ \t]|\Z)",
        re.S | re.M | re.I,
    ),
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
        if self.kind == "trivial_reference":
            return (
                f"`{self.result_path}` is exactly 0 at every one of its "
                f"settings ({self.context}). A reference value that does not "
                f"vary across the sweep meant to vary it is what a root "
                f"finder returning the trivial root on its bracket endpoint "
                f"looks like — check the bracket before the paper describes "
                f"this as a limit"
            )
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
    # Fold `1.21 \times 10^{-2}` into `1.21e-2` so the exponent survives into
    # the extracted value instead of being dropped on the floor.
    cleaned = _LATEX_SCI.sub(lambda m: f"{m.group(1)}e{m.group(2).replace('+', '')}", cleaned)

    out: list[tuple[float, str, str]] = []
    for m in _NUMBER.finditer(cleaned):
        token, exp = m.group(1), m.group(2)
        before = cleaned[max(0, m.start() - 24): m.start()]
        if _CONTEXT_SKIP.search(before):
            continue
        if _LEVEL_AFTER.match(cleaned[m.end(): m.end() + 40]):
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


# The design fields that say how the experiment was SET UP: the parameters, the
# sweep values, the controls. Not ``hypothesis`` / ``expected_outcome`` (what
# the author predicted a result would be) and not ``result_assertions`` (bounds
# on results): a paper that prints one of those where the measured value
# belongs is the mistake this check is for.
_SETUP_FIELDS = ("variables", "method")


def _text_and_numbers(obj: Any, texts: list[str], values: list[float]) -> None:
    """Collect every string and every numeric leaf under ``obj``."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, str):
        texts.append(obj)
    elif isinstance(obj, (int, float)):
        if math.isfinite(obj):
            values.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _text_and_numbers(v, texts, values)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _text_and_numbers(v, texts, values)


def declared_numbers(design: Any = None, topic: Any = None) -> list[float]:
    """The numbers the run was *given*: written in the topic, or in the design's
    ``variables`` and ``method``.

    ``R0 in {0.9, 1.5, 3.0}`` and ``recovery rate gamma = 1`` are settings, and
    a paper that prints them is quoting its own setup, not a result. A
    percentage in the setup is also declared as the fraction a paper may print
    for it (``10% of N`` -> 0.10).
    """
    texts: list[str] = []
    values: list[float] = []
    if isinstance(topic, str):
        texts.append(topic)
    if isinstance(design, dict):
        for name in _SETUP_FIELDS:
            _text_and_numbers(design.get(name), texts, values)

    for text in texts:
        folded = _LATEX_SCI.sub(
            lambda m: f"{m.group(1)}e{m.group(2).replace('+', '')}",
            text.replace(_MINUS_SIGN, "-"),
        )
        for m in _NUMBER.finditer(folded):
            try:
                value = float(m.group(1).replace(",", "") + (f"e{m.group(2)}" if m.group(2) else ""))
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            values.append(value)
            if re.match(r"\s*\\?%", folded[m.end(): m.end() + 4]):
                values.append(value / 100.0)
    return sorted({v for v in values if abs(v) > MIN_MAGNITUDE})


def _is_declared(value: float, token: str, declared: list[float]) -> bool:
    """Is the paper's number one of the settings, at the precision it wrote?"""
    return any(d == value or _rounds_to(value, d, token) for d in declared)


# Report order: the self-contradiction first, then the classic copy error,
# then the weaker distance signal.
_KIND_RANK = {"trivial_reference": 0, "transposed": 1, "near_miss": 2}


def check(
    paper_text: str, result_json: Any, *, declared: Iterable[float] | None = None,
) -> OracleReport:
    """Compare a paper's prose numbers against the computed results.

    ``declared`` are the settings the run was given (see ``declared_numbers``).
    A paper number equal to one of them is a quoted setting, so it is not
    reported as a ``near_miss`` -- but a ``transposed`` one still is.
    """
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

    # U+2212 is the minus a paper typeset with (and a language model writes).
    # ``_NUMBER`` reads only ``-``, so ``−0.114`` was compared as 0.114 --
    # a positive number, near some positive result. Confined to this check:
    # ``number_provenance`` shares the extractor and is left as it was.
    numbers = extract_paper_numbers(paper_text.replace(_MINUS_SIGN, "-"))
    report.paper_numbers = len(numbers)
    settings = list(declared) if declared else []

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

        kind = (
            "transposed"
            if _digit_bag(value) and _digit_bag(value) == _digit_bag(actual)
            else "near_miss"
        )
        if kind == "near_miss" and settings and _is_declared(value, token, settings):
            continue
        seen.add(key)
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

    # A reference quantity that is zero everywhere contradicts the results
    # themselves, so it is found without reading the paper at all.
    report.findings.extend(trivial_reference_findings(result_json))

    # Strongest signal first, so a truncated report still leads with the
    # finding most likely to be a real error.
    report.findings.sort(key=lambda f: (_KIND_RANK.get(f.kind, 9), -f.rel_error))
    return report
