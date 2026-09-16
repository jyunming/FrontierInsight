"""Deterministic check that a number in the paper CAME FROM somewhere.

Why this exists
===============
Three checks already stand between a run and its paper and none of them asks
this question.

* The **numeric oracle** (``core/numeric_oracle.py``) looks for the signature
  of a copy that went wrong: a paper number sitting *near* a real result
  without matching it. It deliberately ignores a number that is far from every
  result — "assumed to come from somewhere else".
* The **statistics claim check** (``core/stat_claims.py``) asks whether a
  number is the *quantity* the paper says it is, by recomputing that quantity.
* ``claim_check`` asks a model whether a sentence is supported by a **source**.

So prose is read against the literature, and a number is read against a
*nearby* result — but nothing reads the paper's numbers against **our own
computed results**. A number that appears nowhere in the run falls straight
through: it is too far away for the oracle, it names no statistic for the
claim check, and it cites no source for the grounding check.

That gap shipped a graded paper. One run's Table 1 printed eight cells —
``0.7093 (0.6934–0.7252)``, ``0.6070 (0.5982–0.6158)``, ``0.2974``, ``0.2648``,
``0.0102`` and their intervals — that match **no** value the run computed, no
per-seed replicate, and none of the three-seed aggregates. The grader failed
the run's "numbers are correct" gate over exactly those cells. The numeric
oracle passed the paper, because each wrong number was far enough from every
real one to be assumed foreign.

What makes the check possible
=============================
The naive form of this check does not work, and measuring it is what makes the
difference. A paper does **not** print the contents of ``RESULT_JSON``: it
prints the **mean over the replicate seeds** and that mean's confidence
interval, and neither is a value any single seed recorded. Checked against
``RESULT_JSON`` alone, all eleven of that paper's *correct* headline numbers
are "untraceable" too — the check would flag correct prose, which is worse
than no check at all.

Once the seed aggregates are part of what a number may trace to, the two
classes separate cleanly:

* 11 of 11 numbers the grader accepted → traceable,
* 11 of 13 numbers the grader failed the run over → traceable **nowhere**.

The two exceptions are worth naming, because they bound what this check can
claim: both are values the run really did compute, printed against the *wrong
cell*. That is a mislabel, not a number from nowhere, and no provenance check
can see it — the statistics claim check is the one that recomputes quantities.

So the traceable set is deliberately generous, and everything below widens it
rather than narrowing it. A number is flagged only when *nothing* in the run
can account for it.

What a number may trace to
--------------------------
``RESULT_JSON`` and every per-seed replicate; the seed aggregates (mean, 95%
confidence bounds, SD, standard error, min/max, and the seed count); the
length of any array the results hold (so "300 runs" traces to the 300 entries
that produced it); the numeric parts of a result *path*, which are the stratum
labels a sweep is keyed by (``by_R0.1.5`` accounts for "R0 = 1.5"); every
number a recorded figure holds, including its axis limits — an axis limit
printed as a count is a real defect, but it is ``stat_claims``'s to report,
and this check must not flag it a second time; and every number in the run
**configuration**, including the ones written into the topic text the user
set, because "300 realizations per setting" is a number the user chose rather
than one the experiment computed.

A value also counts as traced when it matches after a **scale change** (a
results value of ``0.15`` printed as ``15%``) or as a **complement**
(``1 − p``), each compared at the precision the shift implies.

The guards are the check
========================
A check that flags correct prose costs a repair round, and FI has measured
runs losing tens of thousands of tokens to exactly that. So the failure mode
being defended against here is the false positive, and most of this module is
the defence:

``precision``
    The load-bearing guard. A number written with too few significant digits
    carries no information about where it came from: "3 seeds", "300 runs",
    "R0 = 1.5", "95% CI", "0.05" and "4 pages" are all rhetorical or
    conventional, and none of them should be traced anywhere. Requiring three
    significant digits — or two written to four decimals, which is how a small
    probability like ``0.0068`` is printed — removes that entire class without
    a single hand-written rule about round numbers. A **known blind spot**
    follows from it and is accepted deliberately: a two-significant-digit
    textbook value quoted as the run's own result (``0.94``) is not checked.

``years`` / ``versions``
    An integer in 1900–2099 is a year, and a dotted ``1.11.4`` is a version;
    neither is a measurement. Reference markers, figure/table/section numbers,
    DOIs and equation numbers are removed upstream by the two tokenizers this
    module reuses rather than re-implements.

``quoted and cited material``
    Reference lists and citation brackets are stripped before a number is
    read, so a figure belonging to a cited source is never asked to trace to
    our results.

``a run with no results``
    If the experiment recorded nothing numeric — it crashed, or its last
    ``RESULT_JSON`` is empty — the check goes **quiet** rather than flagging
    the entire paper. A run that produced no results is a different problem,
    and one this check has nothing useful to say about.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

from core.numeric_oracle import extract_paper_numbers, flatten_numbers
from core.stat_claims import normalise

# A number must carry this many significant digits before its value says
# anything about where it came from. Below it, numbers collide constantly:
# "3 seeds", "300 runs", "R0 = 1.5" and "0.05" would each have to be traced to
# something, and every paper writes them.
MIN_SIG_DIGITS = 3

# ...unless it is written to enough decimal places to be a measurement anyway.
# A small probability is printed "0.0068": two significant digits, but four
# decimals, and the run either computed it or did not.
LOW_PRECISION_SIG_DIGITS = 2
LOW_PRECISION_MIN_DECIMALS = 4

# Integers in this range are years, not measurements.
YEAR_LO, YEAR_HI = 1900, 2099

# How far a hedged number ("approximately 583", "~0.34") may sit from a
# computed value and still be considered that value.
HEDGED_REL_TOL = 0.10

# A table of wrong cells should not bury every other finding in the review.
MAX_FINDINGS = 8

_CONTEXT_CHARS = 170


@dataclass(frozen=True)
class ProvenanceFinding:
    """One number in the paper that nothing in the run accounts for."""

    kind: str
    value: float
    token: str
    context: str
    message: str

    def describe(self) -> str:
        ctx = f" — “{self.context}”" if self.context else ""
        return f"{self.message}{ctx}"


@dataclass
class ProvenanceReport:
    findings: list[ProvenanceFinding] = field(default_factory=list)
    paper_numbers: int = 0
    traceable_values: int = 0
    untraceable: int = 0
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
            "traceable_values": self.traceable_values,
            "untraceable_numbers": self.untraceable,
            "findings": [
                {
                    "kind": f.kind,
                    "value": f.value,
                    "token": f.token,
                    "context": f.context,
                    "message": f.message,
                }
                for f in self.findings
            ],
        }


# --- reading the paper --------------------------------------------------------

# A dotted release like "1.11.4" or "0.15.9". The number tokenizer would read
# its head ("1.11") as a three-significant-digit measurement.
_VERSION = re.compile(r"(?<![\w.])\d+\.\d+(?:\.\d+)+")

# A range written between two numbers: "95% CI 0.528--0.585", "0.919–0.944".
# The dash belongs to the RANGE, but a number tokenizer reads it as the sign of
# the upper bound and yields -0.585 — a value no run will ever have computed.
# Measured over the graded papers this single artifact produced the large
# majority of all flags (every one in six runs), so it is rewritten to a word
# before any number is read. The lookbehind keeps a genuine negative: "Cohen's
# d = -178.3" follows a space, not a digit.
_RANGE_DASH = re.compile(r"(?<=\d)\s*(?:-{1,3}|[‐-―])\s*(?=\d)")

# An arithmetic step the paper shows in full ("0.3396 x 0.5617 + ... = 0.1907").
_OPERATOR = re.compile(r"[=+*/×÷]|\\times|\\cdot|\\frac")

# A random seed is an identifier, not a measurement: it is chosen in the code,
# never computed, and it appears in no result. One graded paper documents its
# stream as `SeedSequence(20260916)`, and that eight-digit integer was the only
# number this check flagged in a paper whose numbers the grader passed.
_SEED_CONTEXT = re.compile(
    r"\bseed|SeedSequence|\brng\b|random[_ ]state|\bPCG\b|Mersenne|Generator\(",
    re.I,
)

# Words that turn an exact number into an approximate one.
_HEDGE = re.compile(
    r"~|\babout\b|\bapproximately\b|\broughly\b|\baround\b|\bnearly\b|"
    r"\bapprox\.?|\border of\b|≈",
    re.I,
)


def _significant_digits(token: str) -> int:
    """Significant digits in the number as the paper wrote it."""
    t = token.lstrip("-+").replace(",", "")
    if "." in t:
        whole, frac = t.split(".", 1)
        whole = whole.lstrip("0")
        return len(whole) + len(frac) if whole else len(frac.lstrip("0")) or len(frac)
    return len(t.strip("0")) or 1


def _decimals(token: str) -> int:
    t = token.replace(",", "")
    return len(t.split(".", 1)[1]) if "." in t else 0


def _carries_enough_precision(token: str) -> bool:
    """Is this number written precisely enough to say where it came from?

    Three significant digits, or two written to four decimals. Everything
    below is a round number, a convention or a stratum label, and asking it to
    trace to a result is how a check starts flagging correct prose.
    """
    sig = _significant_digits(token)
    if sig >= MIN_SIG_DIGITS:
        return True
    return sig >= LOW_PRECISION_SIG_DIGITS and _decimals(token) >= LOW_PRECISION_MIN_DECIMALS


def _is_year(value: float, token: str) -> bool:
    return (
        _decimals(token) == 0
        and float(value).is_integer()
        and YEAR_LO <= value <= YEAR_HI
    )


def paper_numbers(paper_text: str) -> list[tuple[float, str, str]]:
    """``(value, token, context)`` for every number the paper *claims*.

    Both existing tokenizers are reused rather than re-implemented:
    ``stat_claims.normalise`` unwraps the LaTeX a paper writes its numbers in
    and removes the reference list and citation brackets, and
    ``numeric_oracle.extract_paper_numbers`` folds ``1.21 \\times 10^{-2}``
    into a single value and drops figure/table/section/DOI references. Tables
    survive both, deliberately: a results table is where an untraceable number
    does the most damage, and it is where the observed one lived.
    """
    text = _RANGE_DASH.sub(" to ", _VERSION.sub(" ", normalise(paper_text)))
    return extract_paper_numbers(text)


def _derivable(value: float, token: str, context: str, traceable: Traceable) -> bool:
    """Is this number an obvious derivation of the numbers standing beside it?

    Two cases, and the second is deliberately the stricter one:

    * The paper **writes the arithmetic out** — ``0.3396 x 0.5617 + (1 -
      0.3396) x 0.0076 = 0.1907 + 0.0050 = 0.1957``. Those intermediate values
      exist nowhere in the results and should not; the derivation is on the
      page, so any numbers beside it may be combined.
    * There is **no operator**, as in a table row whose last column is a
      difference: ``| 1.5 | 100 | 0.607 | 0.583 | 0.0240 |``. Here the operands
      must themselves be values the run computed, so a derivation is only
      claimed between two real results — otherwise any three numbers standing
      near each other would explain away a fourth, and the check would go
      blind.
    """
    dec = _decimals(token)
    shown = bool(_OPERATOR.search(context))
    operands = [
        (v, tok) for v, tok in _numbers_in_text(context)
        if v != value and (shown or traceable.find(v, _decimals(tok)) is not None)
    ]
    for i, (a, _a_token) in enumerate(operands):
        for b, _b_token in operands[i:]:
            for cand in (a + b, a - b, b - a, a * b):
                try:
                    if round(cand, dec) == round(value, dec):
                        return True
                except (ValueError, OverflowError):
                    continue
            for num, den in ((a, b), (b, a)):
                if not den:
                    continue
                try:
                    if round(num / den, dec) == round(value, dec):
                        return True
                except (ZeroDivisionError, OverflowError, ValueError):
                    continue
    return False


# --- what a number may trace to -----------------------------------------------

# A numeric run inside a result path: `by_R0.1.5.outbreak_probability` is keyed
# by the stratum R0 = 1.5, and a paper naming that stratum is naming a number
# the run was configured with. List indices are excluded (they would make every
# small integer traceable and quietly blind the check).
_PATH_NUMBER = re.compile(r"(?<![\w])\d+(?:\.\d+)*(?![\w])")

# A number written inside a configuration string — the topic text states the
# grid ("R0 in {0.9, 1.5, 3.0}", "300 stochastic runs per setting").
_TEXT_NUMBER = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)(?![\w])")


class Traceable:
    """The set of values this run can account for, indexed for rounded lookup.

    A paper number matches when it equals a known value *rounded to the
    paper's own precision*, so ``0.8342`` printed as ``0.83`` is the same
    number. Rounding tables are built per decimal-count and cached, which
    keeps the check linear in the number of results rather than quadratic —
    one run stores 14 565 of them.
    """

    def __init__(self, values: dict[str, float]) -> None:
        self.values = values
        self._by_dec: dict[int, dict[float, str]] = {}
        self._trunc_by_dec: dict[int, dict[float, str]] = {}

    def __len__(self) -> int:
        return len(self.values)

    def _table(self, dec: int) -> dict[float, str]:
        table = self._by_dec.get(dec)
        if table is None:
            table = {}
            for path, value in self.values.items():
                try:
                    table.setdefault(round(value, dec), path)
                except (ValueError, OverflowError):
                    continue
            self._by_dec[dec] = table
        return table

    def find(self, value: float, dec: int) -> str | None:
        try:
            key = round(value, dec)
        except (ValueError, OverflowError):
            return None
        return self._table(dec).get(key)

    def _trunc_table(self, dec: int) -> dict[float, str]:
        table = self._trunc_by_dec.get(dec)
        if table is None:
            table = {}
            scale = 10.0 ** dec
            for path, value in self.values.items():
                try:
                    table.setdefault(math.floor(value * scale) / scale, path)
                except (ValueError, OverflowError):
                    continue
            self._trunc_by_dec[dec] = table
        return table

    def find_truncated(self, value: float, dec: int) -> str | None:
        """A paper that TRUNCATES rather than rounds still quotes the run's own
        number. ``0.05/36 = 0.0013888…`` gets written ``0.001388``, and against
        rounding alone that reads as a number from nowhere — the one flag this
        check raised on a threshold the run really did compute."""
        scale = 10.0 ** dec
        try:
            key = math.floor(value * scale) / scale
        except (ValueError, OverflowError):
            return None
        return self._trunc_table(dec).get(key)

    def find_near(self, value: float, tol: float) -> str | None:
        """Nearest value within a relative tolerance, for a hedged number."""
        for path, actual in self.values.items():
            denom = max(abs(value), abs(actual))
            if denom and abs(value - actual) / denom <= tol:
                return path
        return None


def _list_lengths(obj: Any, path: str, out: dict[str, float]) -> None:
    """The length of every array in the results. "300 runs" is a real fact
    about the experiment even when no stored value equals 300.

    An EMPTY array records nothing: counting its zero length as a result would
    make a crashed run — whose last ``RESULT_JSON`` holds no numbers at all —
    look like it computed something, and the check would then flag every
    number in the paper instead of going quiet.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            _list_lengths(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(obj, (list, tuple)):
        if obj:
            out.setdefault(f"len({path or '<root>'})", float(len(obj)))
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list, tuple)):
                _list_lengths(v, f"{path}[{i}]", out)


def _strings(obj: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(obj, str):
        yield path or "<root>", obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}[{i}]")


def _numbers_in_text(text: str) -> Iterator[tuple[float, str]]:
    """``(value, token)`` for each number written in a run of text. The token
    is kept because the precision a number was written at decides what it may
    be compared against."""
    for m in _TEXT_NUMBER.finditer(text):
        token = m.group(1)
        try:
            yield float(token.replace(",", "")), token
        except ValueError:
            continue


def _stat_values(
    stats: dict[str, dict[str, Any]] | None, label: str, out: dict[str, float],
) -> None:
    """Mean / CI / SD / min / max / n, plus the standard error a paper prints
    beside a mean (``SE 0.0115``), which is the SD over the root of the seeds."""
    for path, stat in (stats or {}).items():
        if not isinstance(stat, dict):
            continue
        for key, value in stat.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            out.setdefault(f"{label}.{path}.{key}", float(value))
        std, n = stat.get("std"), stat.get("n")
        if (
            isinstance(std, (int, float)) and not isinstance(std, bool)
            and isinstance(n, (int, float)) and not isinstance(n, bool) and n > 0
        ):
            out.setdefault(f"{label}.{path}.stderr", float(std) / (float(n) ** 0.5))


def build_traceable(
    *,
    result_json: Any = None,
    replicates: Any = None,
    intervals: dict[str, dict[str, Any]] | None = None,
    aggregate: dict[str, dict[str, Any]] | None = None,
    figure_records: Any = None,
    comparison_stats: Any = None,
    config: Any = None,
    design: Any = None,
) -> tuple[Traceable, int]:
    """Everything this run can account for, and how much of it came from the
    experiment itself (the second number gates the whole check: a run that
    computed nothing has nothing to check a paper against)."""
    values: dict[str, float] = {}

    for path, value in flatten_numbers(result_json or {}):
        values.setdefault(path, value)
    for path, value in flatten_numbers(replicates or []):
        values.setdefault(f"replicates{path}", value)
    _stat_values(intervals, "mean_over_seeds", values)
    _stat_values(aggregate, "over_seeds", values)
    _list_lengths(result_json or {}, "", values)
    _list_lengths(replicates or [], "replicates", values)
    from_results = len(values)

    # Stratum labels, read off the result paths themselves.
    for path in list(values):
        for m in _PATH_NUMBER.finditer(re.sub(r"\[\d+\]", "", path)):
            try:
                values.setdefault(f"stratum {m.group(0)}", float(m.group(0)))
            except ValueError:
                continue

    # A figure's own numbers, axis limits included. An axis limit printed as a
    # count IS a defect, but stat_claims recomputes and names it; flagging it
    # here too would report one mistake twice.
    for path, value in flatten_numbers(figure_records or {}):
        values.setdefault(f"figure.{path}", value)

    # The pairwise comparison aggregate: each stratum's interval and every
    # Cohen's d the replicates support. These are computed FOR the paper and
    # are absent from RESULT_JSON, so an effect size the paper quotes has
    # nowhere else to trace to. Whether such a claim is overstated is a
    # different question, and stat_claims already answers it.
    for path, value in flatten_numbers(comparison_stats or {}):
        values.setdefault(f"comparisons.{path}", value)

    # The run configuration, including numbers the user wrote into the topic.
    if config is not None:
        for path, value in flatten_numbers(config):
            values.setdefault(f"config.{path}", value)
        for path, text in _strings(config):
            for value, _token in _numbers_in_text(text):
                values.setdefault(f"config.{path}={value:g}", value)

    # The design: the parameters and reference values the quest declared before
    # it ran. A paper legitimately quotes these — one graded paper prints
    # "as design-time reference values, tau_det = 0.5828", which is a constant
    # the design named and the experiment never re-derived. Measured over the
    # graded papers, including the design clears exactly that number and costs
    # none of the values a failing run was actually caught on. The blind spot
    # it buys is deliberate and worth naming: a number the design DECLARED but
    # the experiment never computed is treated as accounted for.
    if design is not None:
        for path, value in flatten_numbers(design):
            values.setdefault(f"design.{path}", value)
        for path, text in _strings(design):
            for value, _token in _numbers_in_text(text):
                values.setdefault(f"design.{path}={value:g}", value)

    return Traceable(values), from_results


# --- the check ----------------------------------------------------------------

def _variants(value: float, token: str) -> Iterator[tuple[float, int, str]]:
    """``(candidate, decimals, how)`` — the same number under the scale and
    complement shifts a paper legitimately applies. Dividing by 100 moves the
    decimal point two places, so the precision it must be compared at moves
    with it."""
    dec = _decimals(token)
    yield value, dec, ""
    yield value / 100.0, dec + 2, " (as a fraction rather than a percentage)"
    yield value * 100.0, max(dec - 2, 0), " (as a percentage rather than a fraction)"
    if 0.0 < value < 1.0:
        yield 1.0 - value, dec, " (as its complement)"
    if 0.0 < value < 100.0:
        yield 1.0 - value / 100.0, dec + 2, " (as the complement of a percentage)"


def check(
    paper_text: str,
    *,
    result_json: Any = None,
    replicates: Any = None,
    intervals: dict[str, dict[str, Any]] | None = None,
    aggregate: dict[str, dict[str, Any]] | None = None,
    figure_records: Any = None,
    comparison_stats: Any = None,
    config: Any = None,
    design: Any = None,
    n_seeds: int = 0,
) -> ProvenanceReport:
    """Flag every number in the paper that nothing in this run accounts for.

    The aggregates are passed in rather than recomputed so this module stays
    free of ``core.engine`` and can be tested without one.
    """
    report = ProvenanceReport()
    if not paper_text.strip():
        report.skipped = True
        report.skip_reason = "paper text is empty"
        return report

    traceable, from_results = build_traceable(
        result_json=result_json, replicates=replicates, intervals=intervals,
        aggregate=aggregate, figure_records=figure_records,
        comparison_stats=comparison_stats, config=config, design=design,
    )
    report.traceable_values = len(traceable)
    # The experiment recorded nothing: it crashed, or its last RESULT_JSON is
    # empty. A paper written over no results is a real problem and not this
    # check's — with nothing to compare against, every number would be
    # "untraceable" and the whole paper would be flagged.
    if not from_results:
        report.skipped = True
        report.skip_reason = "the run recorded no numeric results"
        return report

    seen: set[float] = set()
    for value, token, context in paper_numbers(paper_text):
        if not _carries_enough_precision(token) or _is_year(value, token):
            continue
        # A seed is chosen, never computed, so it appears in no result.
        if _decimals(token) == 0 and _SEED_CONTEXT.search(context):
            continue
        report.paper_numbers += 1
        hedged = bool(_HEDGE.search(context))
        if any(
            traceable.find(cand, dec) is not None
            or traceable.find_truncated(cand, dec) is not None
            for cand, dec, _ in _variants(value, token)
        ):
            continue
        if hedged and traceable.find_near(value, HEDGED_REL_TOL) is not None:
            continue
        if _derivable(value, token, context, traceable):
            continue
        report.untraceable += 1
        if value in seen:
            continue
        seen.add(value)
        if len(report.findings) >= MAX_FINDINGS:
            continue
        seeds = f"the {n_seeds}-seed aggregates" if n_seeds >= 2 else "the seed aggregates"
        report.findings.append(ProvenanceFinding(
            kind="untraceable_number",
            value=value,
            token=token,
            context=" ".join(context.split())[:_CONTEXT_CHARS],
            message=(
                f"the paper prints {token}, and nothing in this run accounts "
                f"for it: it matches no value in the experiment's results or "
                f"per-seed replicates, none of {seeds} (mean, confidence "
                f"bounds, SD, standard error, min/max), no recorded figure, "
                f"and no number in the run configuration — checked exactly, "
                f"rounded to its own precision, and under a scale or "
                f"complement change. Report the value the run computed, or "
                f"say where this number comes from."
            ),
        ))

    # A paper can print more untraceable numbers than a review should carry —
    # one observed table held eighteen. Reporting only the first eight would
    # have the writer fix those, and the next round flag the rest, spending
    # the iteration budget a round at a time. So the last finding says how
    # many more there are and where the full list is.
    hidden = len(seen) - len(report.findings)
    if hidden > 0 and report.findings:
        last = report.findings[-1]
        report.findings[-1] = ProvenanceFinding(
            kind=last.kind, value=last.value, token=last.token,
            context=last.context,
            message=(
                f"{last.message} A further {hidden} number(s) in this paper "
                f"trace to nothing either — the complete list is in "
                f"paper/provenance_audit.json, and they need the same fix."
            ),
        )
    return report
