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

``trailing zeros``
    An integer written ``2330`` from a run that holds ``2326.66`` is that value
    to three significant digits, the same reading ``_significant_digits``
    gives it, so it is compared at the tens (``-1`` decimals), not at the
    units, where ``2327`` is not ``2330``. A token with ``k`` trailing zeros
    and no decimal point is compared at ``-k`` decimals; ``2316.0`` still
    does not account for ``2330`` (it rounds to ``2320``). Any integer with at
    least three significant digits and trailing zeros (``1550``, ``20300``) is
    compared this way, at +-5 or +-50; measured, it added no coincidental match.
    Rounding only: a rounded integer is never matched by truncation, and is not
    scaled by 100 upwards. A number with fewer
    than three significant digits (``2000``) is not checked at all, as before.

``unit prefixes``
    A change of unit by 1000 (J to kJ, ms to s) is deliberately NOT accepted as
    a scale change. It was tried and measured: checking the stored papers
    against the *wrong* quest's results (every acceptance a coincidence), 71 of
    133 numbers were accepted without it and 76 with it. A check is not
    loosened for that; a paper that changes unit should print the value the run
    computed.

``years`` / ``versions``
    An integer in 1900–2099 is a year, and a dotted ``1.11.4`` is a version;
    neither is a measurement. Reference markers, figure/table/section numbers,
    DOIs and equation numbers are removed upstream by the two tokenizers this
    module reuses rather than re-implements.

``how a number is read``
    A number is checked as a reader reads it, and the shared tokenizer reads a
    few things differently, so the text is rewritten before it is tokenized
    (``paper_numbers``). Measured over the stored papers, sixteen of the
    numbers this check flagged were the tokenizer's reading, not the paper's:

    * a **minus sign** typeset as U+2212 was read as no sign at all, so a
      pullback of "U+2212 38.61 nm" was checked as ``38.61`` and reported
      against a run that holds ``-38.61``. It is now one number with its sign,
      compared with its sign;
    * a **range dash after a percent sign** (``68.1%-85.5%``) was read as the
      minus sign of the upper bound, the reverse mistake;
    * ``±18.5`` names two numbers, ``18.5`` and ``-18.5``, and either one
      traces;
    * ``n=107,104,94`` is a list, not ``107104`` and ``94``: a thousands
      separator is followed by exactly three digits, so a comma run in which
      a later group is not three digits is read as separate numbers;
    * ``SPIE 11609`` is an identifier, an integer of four or more digits that
      follows a capitalised acronym, and ``Abbe's 1873`` a year in the
      possessive attribution of a work.

    Nothing here is a derived-value allowance. A number the run never
    produced, next to a sign, in a list or after an acronym, is still flagged.

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
import unicodedata
from array import array
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from core.numeric_oracle import MIN_MAGNITUDE, _MINUS_SIGN, _TEX_THOUSANDS, extract_paper_numbers, flatten_numbers, roundings
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

# A column of more than this many numbers under one result name (an array of per-trial values, or one field of a long
# group of records) is RAW DATA: a paper reports what was computed from it, never one element picked at random. Such a
# column accounts for a paper number only through the statistics derived from it (see ``_column_stats``); its
# individual elements do not. Shorter columns (a sweep, a table row, a per-seed list) keep every element, because a
# paper does print those one by one. Without this, a result file with hundreds of thousands of trial values held
# nearly every number a paper could print: papers checked against ANOTHER quest's results had more than half their
# numbers "accounted for" by pure coincidence.
RAW_COLUMN_MIN = 200

# ...except that a long column keeps its elements when they cannot be a crowd of lucky matches: it takes few distinct
# values (a 0/1 outcome, a coded level).
_MAX_DISTINCT_KEPT = 50

# A string or boolean field of a record that takes at most this many values names a GROUP (arm, method, dataset): the
# records are pooled within a group, never across groups. A field with more values (an id, a timestamp) names nothing.
_MAX_GROUPS = 20


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
#
# A percent sign ends a number too: "sat 68.108593%-85.522718% below" is a range,
# and its upper bound was read as -85.522718 -- the run holds 85.522718. The
# sign of "-0.175906 for CAR" in the same sentence follows a space and stays.
_RANGE_DASH = re.compile(r"(?<=[\d%])\s*(?:-{1,3}|[‐-―])\s*(?=\d)")

# A comma run written with no space after the commas: "n_major=107,104,94". The
# tokenizer reads the well-formed head of it, "107,104", as one number with a
# thousands separator (107104) and then "94". But a thousands separator is
# followed by exactly three digits, and only three, so a run in which a later
# group is not three digits is a list. Such a run is spaced so that each member
# is read on its own. A well-formed run ("1,000" and "1,000,000") is left alone,
# and so are "1,000, 2,000" and "3, 4, 5", which the tokenizer already reads as
# two and three numbers. A well-formed run that is really a list ("202,206,189")
# cannot be told from one number by its digits, and is not touched here: it is
# read as a list only when the check finds the number nowhere and every member in
# the run (_a_list_the_run_holds).
_COMMA_RUN = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{1,3})+(?!\w|\.\d|,\d)")
_THOUSANDS_RUN = re.compile(r"\d{1,3}(?:,\d{3})+")

# An identifier is not a measurement: "the SPIE 11609 review" cites a proceedings
# volume, and the tokenizer read 11609 as a five-digit result. It is an integer
# of four or more digits that follows a capitalised acronym (two to eight capital
# letters and a space). "13.5 nm" and "NA 0.55" are decimals and "N = 5000" has a
# single capital and an operator; none is touched.
_ACRONYM_ID = re.compile(r"\b([A-Z]{2,8})([ \xa0])(\d{4,})\b(?![.,]?\d)")

# A year that credits a work to its author in the possessive: "Ernst Abbe's 1873
# description". The year guard below stops at 1900, and a count in the same range
# ("1600 qubits", "1024 samples") is a result, so the year is recognised by what
# stands before it, not by its size. Years from 1500 on; a citation in
# parentheses ("Abbe (1873)") is not read this way, because a group label
# followed by a parenthesised count ("Control (1200)") has the same shape.
_ATTRIBUTED_YEAR = re.compile(r"\b([A-Z][a-z]+['\u2019]s)(\s+)(1[5-8]\d\d)\b(?![.,]?\d)")

# "±18.5", "\pm 18.5", "+/-18.5": the number stands for +18.5 and -18.5.
_PLUS_MINUS = r"(?:\xb1|\\pm|\+\s*/\s*-|\+-)"

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


def _written_precision(token: str) -> int:
    """The decimal place the paper rounded to: the decimals of ``0.583`` (3), and
    minus the trailing zeros of an integer written to fewer digits than it has
    (``2330`` is rounded to the tens: -1). ``round(v, -1)`` is how it compares."""
    t = token.lstrip("-+").replace(",", "")
    if "." in t:
        return _decimals(token)
    return -(len(t) - len(t.rstrip("0")))


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
    # A year has no thousands separator: "1,920" is a quantity of 1920 (a standard deviation in the paper this was found
    # in), and reading it as the year 1920 exempted it from every check.
    return (
        "," not in token
        and _decimals(token) == 0
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

    The text is rewritten first so that the tokenizer reads what a reader does
    (see the module docstring, "how a number is read"). The order matters: the
    minus sign is folded before the range dash is looked for, so a U+2212
    between two numbers is a range or a subtraction exactly as ``-`` is.
    """
    text = normalise(_TEX_THOUSANDS.sub(",", paper_text)).replace(_MINUS_SIGN, "-")
    text = _RANGE_DASH.sub(" to ", _VERSION.sub(" ", text))
    text = _COMMA_RUN.sub(_spaced_if_a_list, text)
    # An underscore is a word character, so the tokenizer no longer starts a
    # number on the digits it now follows.
    text = _ACRONYM_ID.sub(r"\1_\3", text)
    text = _ATTRIBUTED_YEAR.sub(r"\1_\3", text)
    return extract_paper_numbers(text)


def _spaced_if_a_list(m: re.Match[str]) -> str:
    """A comma run whose groups are not all thousands groups is a list."""
    run = m.group(0)
    return run if _THOUSANDS_RUN.fullmatch(run) else ", ".join(run.split(","))


def _a_list_the_run_holds(token: str, traceable: "Traceable") -> bool:
    """A comma run of three-digit groups ("100,250,500") read as one number with thousands separators, when that number
    is nowhere in the run but every group is: it is a list (a real paper's grid of population sizes, which the check
    called one untraceable number, and the rewrite then cut from Methods). A group with a leading zero ("1,050") is a
    thousands group, never a list member, so such a run is left as one number."""
    groups = token.split(",")
    # One sign, on the first member only ("-100,250,500"); "+-100" is not a number and fails the digit check.
    digits = [g[1:] if i == 0 and g[:1] in "+-" else g for i, g in enumerate(groups)]
    if len(groups) < 2 or any(len(d) > 1 and d.startswith("0") for d in digits) or not all(d.isdigit() for d in digits):
        return False
    return all(traceable.find(float(g), 0) is not None for g in groups)


def _either_sign(token: str, context: str) -> bool:
    """Is this number written after a plus-minus sign, so that it stands for
    both ``x`` and ``-x``? ``\\(\\pm 18.5\\) nm`` names two edges, and the run
    holds one of them. The token's own sign is set aside: ``+/-18.5`` is read by
    the tokenizer as ``-18.5``, and it is the plus-minus that says what it is."""
    return re.search(
        _PLUS_MINUS + r"\s*" + re.escape(token.lstrip("-")) + r"(?!\d|\.\d)",
        context,
    ) is not None


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
                    if round(value, dec) in roundings(cand, dec):
                        return True
                except (ValueError, OverflowError):
                    continue
            for num, den in ((a, b), (b, a)):
                if not den:
                    continue
                try:
                    if round(value, dec) in roundings(num / den, dec):
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


# Words that name nothing: a connective, a statistic every result has, the structure the run's own record is stored in,
# and the words every paper and every result uses (a rate, a score, an error, a run). The list is applied to the 4-letter
# STEMS, so every inflection ("means", "values") is stopped with its root.
_STOP_WORDS = (
    "the and for with from that this these those are was were per over all each both also than then when which where "
    "while into onto between across under using used use show shows shown table figure paper study "
    "value values mean means median std sem min max low lower high upper ledger trial trials result results data "
    "time times rate rates score scores error errors test tests case cases step steps run runs number total sample "
    "samples size level levels point points"
).split()
# The two-letter words of ordinary English. Written in capitals (a heading, a sentence in capitals, emphasis) or as a
# part of a result's path (`in`, `no`, `id`), they are still connectives, not abbreviations: only a two-letter word
# that is NOT one of these (SD, CI, SE) can name a result.
_TWO_LETTER_WORDS = frozenset(
    "ad ah am an as at aw ax be by do eh er ex go ha he hi id if in is it me my no of oh ok on or ow ox pa re so to uh um "
    "up us we ye yo".split()
)
_COMMON_STOP = frozenset(w[:4] for w in _STOP_WORDS) | _TWO_LETTER_WORDS
_PATH_STOP = _COMMON_STOP | frozenset(w[:4] for w in (
    "replicates config design oracle figure comparisons optimisation stratum len records cells settings setting seeds "
    "seed over key keys estimate"
).split())
_WORD = re.compile(r"[a-z]{3,}")
# An abbreviation is the one two-letter word that names something (SD, CI, SE): in a sentence it is written in capitals,
# in a result's name it is a lowercase snake_case part (`Ea_estimate_sd_J_mol`). Any other two-letter word is a
# connective ("of", "in", "is") and names nothing.
_ABBREVIATION_IN_TEXT = re.compile(r"(?<![A-Za-z])[A-Z]{2}(?![A-Za-z])")
_ABBREVIATION_IN_NAME = re.compile(r"(?<![A-Za-z])(?:[a-z]{2}|[A-Z]{2})(?![A-Za-z])")
_GROUP_TAG = re.compile(r"\{[^}]*\}")
_INDEX_OR_STAT = re.compile(r"\[\d*\]|\.@\w+$")

# A value the run was told or declared (configuration, design) is not a computed value that could match by luck.
_UNGATED_PREFIXES = ("config.", "design.")


def _leaf_name(path: str) -> str:
    """The last segment of a result path, which names the quantity (``by_noise.0.1.k_coverage_probability[1]`` ->
    ``k_coverage_probability``); the segments before it are the strata it was computed in."""
    path = _GROUP_TAG.sub("", _INDEX_OR_STAT.sub("", _INDEX_OR_STAT.sub("", path)))
    return path.rsplit(".", 1)[-1] if "." in path else path


def _name_stems(text: str, stop: frozenset[str], *, of_a_name: bool = False) -> frozenset[str]:
    """The first four letters of each word in ``text`` that names something, so ``coverage`` and ``covered`` meet.
    The stop list is applied to the stem, so ``means`` is stopped with ``mean``. An abbreviation of two letters is
    kept (see ``_ABBREVIATION_IN_TEXT``); ``of_a_name`` says ``text`` is a result's name, not a sentence."""
    stems = {w[:4] for w in _WORD.findall(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text).lower())}
    stems.update(
        ([w.lower() for w in _ABBREVIATION_IN_NAME.findall(text)] if of_a_name else [w.lower() for w in _ABBREVIATION_IN_TEXT.findall(text)])
    )
    return frozenset(stems - stop)


class Traceable:
    """The set of values this run can account for, indexed for rounded lookup.

    A paper number matches when it equals a known value *rounded to the
    paper's own precision*, so ``0.8342`` printed as ``0.83`` is the same
    number. Rounding tables are built per decimal-count and cached, which
    keeps the check linear in the number of results rather than quadratic —
    one run stores 14 565 of them.
    """

    def __init__(self, values: dict[str, float], derived: dict[str, float] | None = None) -> None:
        # ``values`` are the numbers the run stored or declared: exact-match targets, and the operands of a derivation.
        # ``derived`` are statistics FI computed from raw columns (``mean``, ``sd``...): they account for a paper number
        # that names them, but only through the same luck gate as everything else, and they are never an operand.
        self.values = values
        self.derived = derived or {}
        self._all = {**values, **self.derived} if self.derived else values
        self._by_dec: dict[int, dict[float, str]] = {}
        self._trunc_by_dec: dict[int, dict[float, str]] = {}
        self._lists_by_dec: dict[int, dict[float, list[str]]] = {}
        self._trunc_lists_by_dec: dict[int, dict[float, list[str]]] = {}
        self._buckets_by_dec: dict[int, dict[tuple[bool, int], int]] = {}
        self._words_of: dict[str, frozenset[str]] = {}

    def all_paths(self) -> Iterator[str]:
        """Every path this set holds, stored and derived."""
        return iter(self._all)

    def __len__(self) -> int:
        return len(self._all)

    def _table(self, dec: int) -> dict[float, str]:
        table = self._by_dec.get(dec)
        if table is None:
            table = {}
            for path, value in self.values.items():
                try:
                    for key in roundings(value, dec):
                        table.setdefault(key, path)
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
        if dec < 0:
            # Nobody truncates 2326.66 to 2320: a number rounded to the tens is rounded, and truncating would widen
            # the window of 2320 to [2315, 2330).
            return None
        scale = 10.0 ** dec
        try:
            key = math.floor(value * scale) / scale
        except (ValueError, OverflowError):
            return None
        return self._trunc_table(dec).get(key)

    def _lists(self, dec: int) -> dict[float, list[str]]:
        table = self._lists_by_dec.get(dec)
        if table is None:
            table = {}
            for path, value in self._all.items():
                try:
                    for key in roundings(value, dec):
                        table.setdefault(key, []).append(path)
                except (ValueError, OverflowError):
                    continue
            self._lists_by_dec[dec] = table
        return table

    def _trunc_lists(self, dec: int) -> dict[float, list[str]]:
        table = self._trunc_lists_by_dec.get(dec)
        if table is None:
            table = {}
            scale = 10.0 ** dec
            for path, value in self._all.items():
                try:
                    table.setdefault(math.floor(value * scale) / scale, []).append(path)
                except (ValueError, OverflowError):
                    continue
            self._trunc_lists_by_dec[dec] = table
        return table

    def paths_at(self, value: float, dec: int) -> list[str]:
        """Every path holding ``value`` rounded (or truncated) to ``dec`` decimals; empty when none does."""
        scale = 10.0 ** dec
        try:
            hit = list(self._lists(dec).get(round(value, dec), ()))
            if dec >= 0:  # nobody truncates a number rounded to the tens (see find_truncated)
                hit += self._trunc_lists(dec).get(math.floor(value * scale) / scale, ())
        except (ValueError, OverflowError):
            return []
        return hit

    def chance(self, value: float, dec: int) -> float:
        """How likely a number written like this is to equal SOME value here by luck alone: the share of the numbers
        of its size and sign, at its precision, that this run's values already take (rounded, and truncated where a
        truncated match is accepted). A run holding 400 distinct values between 0.1 and 1 holds about 40% of the 900
        numbers ``0.xyz`` can be, so a 3-digit number in that range matches one of them on a coin flip; a run of 5
        such values holds 0.6% of them.

        A number that rounds to zero carries no information about where it came from (there is one such number, and
        every run holds something near it), so its chance is 1: it is accepted only when the sentence names it."""
        try:
            key = round(value, dec)
        except (ValueError, OverflowError):
            return 1.0
        if key == 0:
            return 1.0
        buckets = self._buckets_by_dec.get(dec)
        if buckets is None:
            keys = set(self._lists(dec))
            if dec >= 0:
                keys |= set(self._trunc_lists(dec))
            buckets = {}
            for k in keys:
                if k:
                    b = (k > 0, math.floor(math.log10(abs(k))))
                    buckets[b] = buckets.get(b, 0) + 1
            self._buckets_by_dec[dec] = buckets
        e = math.floor(math.log10(abs(key)))
        slots = 9.0 * 10.0 ** (max(e + 1 + dec, 1) - 1)
        return min(1.0, buckets.get((key > 0, e), 0) / slots)

    def is_declared(self, path: str) -> bool:
        """Is this path something the run was told (its configuration or its design)?"""
        return path.startswith(_UNGATED_PREFIXES)

    def words_of(self, path: str) -> frozenset[str]:
        """The words a result path names its quantity with (``by_noise.0.1.coverage_95`` -> ``noise``, ``coverage``)."""
        got = self._words_of.get(path)
        if got is None:
            tags = " ".join(_GROUP_TAG.findall(path))
            got = _name_stems(_leaf_name(path) + " " + tags, _PATH_STOP, of_a_name=True)
            if not got:
                # ``trials.error.@mean`` or ``ols.Ea_sd.estimate``: the leaf names nothing, so the quantity is named by
                # what it is of
                bare = _GROUP_TAG.sub("", _INDEX_OR_STAT.sub("", path))
                for segment in reversed(bare.split(".")[:-1]):
                    got = _name_stems(segment, _PATH_STOP, of_a_name=True)
                    if got:
                        break
            self._words_of[path] = got
        return got

    def find_near(self, value: float, tol: float) -> str | None:
        """Nearest value within a relative tolerance, for a hedged number."""
        for path, actual in self._all.items():
            denom = max(abs(value), abs(actual))
            if denom and abs(value - actual) / denom <= tol:
                return path
        return None


# --- raw columns: only what is derived from them is a result -------------------

def _plain_numbers(seq: Any) -> "array | None":
    """``seq`` as an array of doubles when every element is a plain finite number, else ``None``. Fast on millions of
    elements: the conversion itself rejects a string or a ``None``; a boolean first element is not a number."""
    first = seq[0]
    if isinstance(first, bool) or not isinstance(first, (int, float)):
        return None
    try:
        arr = array("d", seq)
    except (TypeError, OverflowError):
        return None
    total = sum(arr)
    if total - total != 0.0:  # a NaN or an infinity somewhere: drop them
        arr = array("d", (v for v in arr if v - v == 0.0))
    return arr


def _quantile(ordered: list[float], pct: float) -> float:
    """The linear-interpolation percentile (numpy's default) of an already sorted list."""
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _column_stats(path: str, vals: Any, derived: dict[str, float]) -> None:
    """What a paper reports from a column of raw values, none of which it reports itself: mean, median, sample SD, min
    and max (and the count). Only these: a percentile or an interval is a result when the run stored it, and is not
    made up here. Computed deterministically from the values themselves."""
    n = len(vals)
    if n < 2:
        return
    ordered = sorted(vals)
    mean = math.fsum(vals) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in vals) / (n - 1))
    stats = {
        "mean": mean, "median": _quantile(ordered, 50.0), "sd": sd, "min": ordered[0], "max": ordered[-1],
        "count": float(n),
    }
    for name, value in stats.items():
        if math.isfinite(value) and abs(value) > MIN_MAGNITUDE:
            derived.setdefault(f"{path}.@{name}", value)


def _kept_elements(vals: Any) -> list[int]:
    """The positions of a long column whose elements a paper may still quote one by one, else ``[]``: only a column with
    few distinct values (a coded level, a 0/1 outcome), one position per value. A long column in order is NOT kept: a
    cumulative sum, sorted errors or arrival times are results of a computation over thousands of values, and keeping
    each as a target would make nearly any number match."""
    first: dict[float, int] = {}
    for i, v in enumerate(vals):
        if v not in first:
            first[v] = i
            if len(first) > _MAX_DISTINCT_KEPT:
                return []
    return sorted(first.values())


def _emit_column(
    label: str, vals: Any, element_path: Callable[[int], str],
    kept: list[tuple[str, float]], derived: dict[str, float],
) -> None:
    """A long column: its derived statistics, and its elements only when ``_kept_elements`` says a paper may quote them."""
    _column_stats(label, vals, derived)
    for i in _kept_elements(vals):
        if abs(vals[i]) > MIN_MAGNITUDE:
            kept.append((element_path(i), vals[i]))


def _walk(obj: Any, path: str, kept: list[tuple[str, float]], derived: dict[str, float]) -> None:
    """Collect the numbers of ``obj`` as ``kept`` (stored values) and ``derived`` (statistics of raw columns)."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(obj) and abs(obj) > MIN_MAGNITUDE:
            kept.append((path or "<root>", float(obj)))
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, f"{path}.{k}" if path else str(k), kept, derived)
        return
    if isinstance(obj, (list, tuple)):
        if len(obj) > RAW_COLUMN_MIN:
            arr = _plain_numbers(obj)
            if arr is not None:
                _emit_column(path or "<root>", arr, lambda i: f"{path}[{i}]", kept, derived)
                return
            if isinstance(obj[0], dict):
                _walk_records(obj, path, kept, derived)
                return
        for i, v in enumerate(obj):
            _walk(v, f"{path}[{i}]", kept, derived)


def _walk_records(
    records: list, path: str, kept: list[tuple[str, float]], derived: dict[str, float],
) -> None:
    """A long list of records. The records are split into GROUPS by their string / boolean fields that take few values
    (control vs treatment, method A vs B), and each field of a group is one column of that group alone: pooling across
    groups would make a subgroup mean no one's statistic. A group of at most ``RAW_COLUMN_MIN`` records keeps every
    element. Per-record path strings are only built for columns that keep their elements."""
    seen: dict[str, set] = {}
    for rec in records:
        if isinstance(rec, dict):
            for k, v in rec.items():
                if isinstance(v, (str, bool)) or v is None:
                    vs = seen.setdefault(k, set())
                    if len(vs) <= _MAX_GROUPS:
                        vs.add(v)
    group_keys = [k for k, vs in seen.items() if len(vs) <= _MAX_GROUPS]
    groups: dict[tuple, list[int]] = {}
    for i, rec in enumerate(records):
        if isinstance(rec, dict):
            # Only a string, boolean or null names a group; a list or dict in that field is treated as missing.
            key = tuple((k, v if isinstance(v, (str, bool)) or v is None else None)
                        for k in group_keys if k in rec for v in (rec[k],))
        else:
            key = ()
        groups.setdefault(key, []).append(i)
    for key, idxs in groups.items():
        tag = "{" + ",".join(f"{k}={v}" for k, v in key) + "}" if key else ""
        cols: dict[str, tuple[array, array]] = {}
        names: dict[tuple[str, str], str] = {}

        def collect(obj: Any, col: str, i: int) -> None:
            if isinstance(obj, bool):
                return
            if isinstance(obj, (int, float)):
                if math.isfinite(obj):
                    ix, vs = cols.get(col) or cols.setdefault(col, (array("q"), array("d")))
                    ix.append(i)
                    vs.append(obj)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    sub = names.get((col, k))
                    if sub is None:
                        sub = names[(col, k)] = f"{col}.{k}"
                    collect(v, sub, i)
            elif isinstance(obj, (list, tuple)):
                if len(obj) > RAW_COLUMN_MIN and (arr := _plain_numbers(obj)) is not None:
                    _emit_column(f"{path}[{i}]{col}", arr, lambda j: f"{path}[{i}]{col}[{j}]", kept, derived)
                    return
                sub = names.get((col, "[]"))
                if sub is None:
                    sub = names[(col, "[]")] = f"{col}[]"
                for v in obj:
                    collect(v, sub, i)

        for i in idxs:
            collect(records[i], "", i)
        for col, (ix, vs) in cols.items():
            label = f"{path}[]{tag}{col}"
            if len(vs) <= RAW_COLUMN_MIN:
                # a small group keeps every element, and its mean / SD / extremes are a group's results too
                _column_stats(label, vs, derived)
                for i, v in zip(ix, vs):
                    if abs(v) > MIN_MAGNITUDE:
                        kept.append((f"{path}[{i}]{col}", v))
            else:
                _emit_column(label, vs, lambda j, ix=ix, col=col: f"{path}[{ix[j]}]{col}", kept, derived)


def _trace_values(obj: Any, prefix: str = "") -> tuple[list[tuple[str, float]], dict[str, float]]:
    """Like ``flatten_numbers``, except that raw columns (see ``RAW_COLUMN_MIN``) give their derived statistics and not
    their elements. Returns ``(stored values, derived statistics)``."""
    kept: list[tuple[str, float]] = []
    derived: dict[str, float] = {}
    _walk(obj, prefix, kept, derived)
    return kept, derived


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
        if obj and isinstance(obj[0], (dict, list, tuple)):
            # Of a long list of records only the first is read: the others have the same shape, and formatting a path
            # for each of millions would cost far more than the lengths are worth.
            for i, v in enumerate(obj if len(obj) <= RAW_COLUMN_MIN else obj[:1]):
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
    be compared against. A U+2212 is read as the minus it is, on this side as on
    the paper's: a design that declares a value with one and a paper that
    prints it must both read a negative."""
    for m in _TEXT_NUMBER.finditer(text.replace(_MINUS_SIGN, "-")):
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
    oracle_checks: Any = None,
    optimisation: Any = None,
    run_facts: Any = None,
) -> tuple[Traceable, int]:
    """Everything this run can account for, and how much of it came from the
    experiment itself (the second number gates the whole check: a run that
    computed nothing has nothing to check a paper against)."""
    values: dict[str, float] = {}
    derived: dict[str, float] = {}

    def take(obj: Any, prefix: str = "") -> None:
        kept, stats = _trace_values(obj, prefix)
        for path, value in kept:
            values.setdefault(path, value)
        for path, value in stats.items():
            derived.setdefault(path, value)

    take(result_json or {})
    take(replicates or [], "replicates")
    _stat_values(intervals, "mean_over_seeds", values)
    _stat_values(aggregate, "over_seeds", values)
    _list_lengths(result_json or {}, "", values)
    _list_lengths(replicates or [], "replicates", values)
    from_results = len(values) + len(derived)

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
    take(figure_records or {}, "figure")

    # The pairwise comparison aggregate: each stratum's interval and every
    # Cohen's d the replicates support. These are computed FOR the paper and
    # are absent from RESULT_JSON, so an effect size the paper quotes has
    # nowhere else to trace to. Whether such a claim is overstated is a
    # different question, and stat_claims already answers it.
    take(comparison_stats or {}, "comparisons")

    # The engine's oracle checks before the main run: what the script measured on each check case, and the expected
    # value and limit it was judged against. These are numbers this run computed, just not in RESULT_JSON; a paper
    # that reports "the energy drift was 1.25e-5, within 2e-5" quotes them.
    take(oracle_checks or [], "oracle")

    # A search for the best design: FI's own record of the search and of its check at finer numerical settings
    # (results/best_design.json, needs/OPTIMUM_CHECK.json: core/best_design_report.py). The paper's "Best design found"
    # section prints them at every level of the numerical settings; they are numbers this run computed, not the
    # script's, and FI wrote them.
    take(optimisation or {}, "optimisation")

    # What FI recorded about the run itself (how long it took): a paper or an analysis that says "the run took
    # 111.3 s" quotes FI's record, not a result.
    take(run_facts or {}, "run")

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

    return Traceable(values, derived), from_results


# --- the check ----------------------------------------------------------------

def _variants(value: float, token: str) -> Iterator[tuple[float, int, str]]:
    """``(candidate, decimals, how)`` — the same number under the scale and
    complement shifts a paper legitimately applies. Dividing by 100 moves the
    decimal point two places, so the precision it must be compared at moves
    with it. The precision is the
    one the paper wrote, so ``2330`` is compared at the tens (``-1``)."""
    dec = _written_precision(token)
    yield value, dec, ""
    # A rounded integer is divided at the precision it had before trailing zeros counted (2 decimals), not at
    # dec + 2: 233000 / 100 compared at the tens would match an unrelated 2326.66.
    yield value / 100.0, max(dec, 0) + 2, " (as a fraction rather than a percentage)"
    if dec >= 0:  # never for a rounded integer: 2330 x 100 would match an unrelated 233,450
        yield value * 100.0, max(dec - 2, 0), " (as a percentage rather than a fraction)"
    if 0.0 < value < 1.0:
        yield 1.0 - value, dec, " (as its complement)"
    if 0.0 < value < 100.0:
        yield 1.0 - value / 100.0, dec + 2, " (as the complement of a percentage)"


# A match that shares no word with the sentence it stands in is accepted only when it is unlikely to be luck.
MAX_CHANCE_UNNAMED = 0.2
SHIFT_CHANCE_DIVISOR = 4


# The words that name a number's quantity are read from the sentence it stands in; a sentence that is too
# long to say anything (a run-on, a paragraph with no full stop) falls back to this many characters either side.
_NAMING_FALLBACK = 120
_SENTENCE_MAX = 500


_BLANK_LINE = re.compile(r"\n[ \t]*\n")


def _sentence_around(text: str, start: int, end: int) -> str:
    """The sentence of ``text`` holding ``[start, end)``. A table row is one line; elsewhere a sentence ends at a full
    stop followed by a capital or at a blank line, so a paragraph soft-wrapped over several lines is read whole."""
    lo, hi = _sentence_span(text, start, end)
    return text[lo:hi]


def _sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Where :func:`_sentence_around` is in ``text``."""
    line_lo = text.rfind("\n", 0, start) + 1
    line_hi = text.find("\n", end)
    line_hi = len(text) if line_hi < 0 else line_hi
    if text[line_lo:line_hi].lstrip().startswith("|"):
        lo, hi = line_lo, line_hi
    else:
        lo = 0
        for m in _BLANK_LINE.finditer(text, 0, start):
            lo = m.end()
        m_end = _BLANK_LINE.search(text, end)
        hi = m_end.start() if m_end else len(text)
        stop = hi
        for m in _SENTENCE_END.finditer(text, lo, hi):
            if m.end() <= start:
                lo = m.end()
            else:
                stop = m.start()
                break
        hi = stop
    if hi - lo <= _SENTENCE_MAX:
        return lo, hi
    return max(0, start - _NAMING_FALLBACK), min(len(text), end + _NAMING_FALLBACK)


def _naming_words(text: str, token: str, context: str) -> frozenset[str]:
    """The words that name the quantity ``token`` stands for: those of the sentence it stands in, wherever the paper
    writes it (at most four places)."""
    words: set[str] = set()
    found = 0
    for m in re.finditer(r"(?<![\d.])" + re.escape(token) + r"(?!\d)", text):
        words |= _name_stems(_sentence_around(text, m.start(), m.end()), _COMMON_STOP)
        found += 1
        if found >= 4:
            break
    if not found:  # the tokenizer rewrote the text around it; the short context it kept is what is left
        words |= _name_stems(context, _COMMON_STOP)
    return frozenset(words)


# A value with its interval beside it: ``0.918 (367/400; 0.886 to 0.941)``, ``0.813 for OLS (0.771-0.848)``,
# ``0.918 [0.886, 0.941]``, ``0.918 (95% CI 0.886-0.941)``. At most three words between the value and the bracket, and
# the bracket must close right after the upper bound, so the pair is the interval.
_NUM_TOKEN = r"(?<![\w.])(-?[0-9]+(?:\.[0-9]+)?)"
_INTERVAL_BESIDE = re.compile(
    _NUM_TOKEN + r"\s*%?(?:\s+[A-Za-z]+){0,3}\s*[(\[][^()\[\]]{0,60}?" + _NUM_TOKEN
    + r"\s*%?\s*(?:to|\u2013|\u2014|-|,|;)\s*" + _NUM_TOKEN + r"\s*%?\s*[)\]]",
    re.I,
)


def _intervals_beside(text: str) -> dict[str, list[tuple[str, str, str]]]:
    """Each interval endpoint the paper prints beside a value, by its token: ``{"0.886": [("0.918", "0.886",
    "0.941")]}``. Only a pair that holds the value between its ends is taken for its interval."""
    out: dict[str, list[tuple[str, str, str]]] = {}
    for m in _INTERVAL_BESIDE.finditer(text):
        value, lower, upper = m.group(1), m.group(2), m.group(3)
        try:
            if not float(lower) <= float(value) <= float(upper):
                continue
        except ValueError:
            continue
        for end in (lower, upper):
            out.setdefault(end, []).append((value, lower, upper))
    return out


#: The most numbers one result may hold for its value and interval to be read as one: an estimate, its interval's two
#: ends and a few counts. A larger group (a list, a whole table) holds an unrelated pair by luck.
RESULT_GROUP_MAX = 8


def _parent(path: str) -> str:
    """``path`` without its last part (``a.b.estimate`` -> ``a.b``, ``a.vals[3]`` -> ``a.vals``); ``""`` at the top."""
    cut = max(path.rfind("."), path.rfind("["))
    return path[:cut] if cut > 0 else ""


def _under(path: str, parent: str) -> bool:
    return path.startswith(parent + ".") or path.startswith(parent + "[")


def _interval_of_a_result(token: str, beside: dict[str, list[tuple[str, str, str]]], traceable: Traceable) -> bool:
    """Is ``token`` an end of the interval printed beside a value, where one result holds the value and both ends:
    ``….coverage.estimate`` 0.9175 and ``….coverage.wilson_95_ci[0]``/``[1]`` 0.886/0.941? A run holding thousands of
    numbers matches a 3-digit endpoint by luck, so the endpoint alone is not enough; the value and both ends under one
    result -- a group of at most ``RESULT_GROUP_MAX`` numbers -- are."""
    for value, lower, upper in beside.get(token, ()):
        ends = [traceable.paths_at(float(t), _written_precision(t)) for t in (lower, upper)]
        if not all(ends):
            continue
        for path in traceable.paths_at(float(value), _written_precision(value)):
            parent = _parent(path)
            if not parent or not all(any(_under(p, parent) for p in paths) for paths in ends):
                continue
            if sum(1 for p in traceable.all_paths() if _under(p, parent)) <= RESULT_GROUP_MAX:
                return True
    return False


def _is_accounted_for(
    traceable: Traceable, cand: float, dec: int, named: Callable[[], frozenset[str]], *, shifted: bool = False,
    max_chance: float = MAX_CHANCE_UNNAMED,
) -> bool:
    """Does a result hold ``cand`` (rounded or truncated at ``dec`` decimals) as the quantity the sentence names?

    A value equal to the number is not enough when the run holds thousands of values: by luck alone a 3-digit number
    equals one of them. So a match counts when the words around the number name what the matching result is
    (``coverage`` in the sentence, ``coverage`` in the result's name), or when the run holds so few values of that
    size that a match is not luck (``Traceable.chance``).
    """
    paths = traceable.paths_at(cand, dec)
    if not paths:
        return False
    # A value the run was told or declared (configuration, design) is no luck.
    if any(traceable.is_declared(path) for path in paths):
        return True
    # A scale or complement change is one more way to match by luck, so the unnamed match must be that much rarer.
    if traceable.chance(cand, dec) <= max_chance / (SHIFT_CHANCE_DIVISOR if shifted else 1):
        return True
    words = named()
    return any(traceable.words_of(path) & words for path in paths)


# --- a number of FI's per-setting table, named by the setting and the quantity -------------------------------------

#: The statistics :func:`core.trial_table.traceable` gives each value of each setting.
_TABLE_STATS = ("n", "mean", "sd", "ci_lower", "ci_upper")
#: A paragraph longer than this says too much to name one number; the text this far either side is read instead.
_PARAGRAPH_MAX = 2500
_PARAGRAPH_FALLBACK = 400
_CONTEXT_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")
_SMALL_WORD = re.compile(r"(?<![a-z])[a-z]{2}(?![a-z])")


def _stems_of_text(text: str) -> frozenset[str]:
    """:func:`_name_stems` of ``text``, with a two-letter abbreviation written in lowercase too (``pp`` in "gap in
    pp"), so a value's name part such as ``_pp`` or ``_bh`` can be found in prose and table headers."""
    small = {w for w in _SMALL_WORD.findall(text.lower()) if w not in _TWO_LETTER_WORDS}
    return frozenset(_name_stems(text, _COMMON_STOP) | small)


def _numbers_of(text: str) -> list[float]:
    out: list[float] = []
    for m in _CONTEXT_NUMBER.finditer(text):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            continue
    return out


def _table_cells(line: str) -> list[str]:
    cells = line.strip().split("|")
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return cells


def _caption_above(lines: list[str], first: int) -> str:
    """The caption of a table whose header is line ``first``: the paragraph right above it (a ``Table 1.`` line)."""
    i = first - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    end = i + 1
    while i >= 0 and lines[i].strip() and not lines[i].lstrip().startswith("|"):
        i -= 1
    caption = " ".join(lines[i + 1:end])
    return caption if re.match(r"\s*(?:\*\*|_)?\s*Table\b", caption, re.I) else ""


@dataclass(frozen=True)
class _Place:
    """Where a number stands. In prose: ``label`` is ``None``, ``names`` its paragraph, ``sentence`` its sentence and
    ``at`` where the number starts in the sentence. In a Markdown table: ``label`` is its column's header (with the
    row's first cell when that is not a number), ``names`` the label with the table's caption, ``sentence`` the row's
    line, ``header`` and ``cells`` the table's header and the row's cells, ``col`` the number's column and ``caption``
    the table's caption."""

    label: str | None
    names: str
    sentence: str = ""
    at: int = -1
    header: tuple[str, ...] = ()
    cells: tuple[str, ...] = ()
    col: int = -1
    caption: str = ""

    @property
    def column(self) -> str:
        return self.header[self.col] if 0 <= self.col < len(self.header) else ""


#: At most this many places of one number are read.
_MAX_PLACES = 12


def _setting_contexts(text: str, token: str) -> list[_Place]:
    """Every place ``token`` stands in ``text`` (:class:`_Place`), at most :data:`_MAX_PLACES`."""
    out: list[_Place] = []
    lines = text.split("\n")
    starts: list[int] = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    for m in re.finditer(r"(?<![\d.])" + re.escape(token) + r"(?!\d)", text):
        row = bisect_right(starts, m.start()) - 1
        line = lines[row]
        if line.lstrip().startswith("|"):
            first = row
            while first > 0 and lines[first - 1].lstrip().startswith("|"):
                first -= 1
            if first == row or first + 1 == row:
                continue  # the number is in the header or the rule under it, not a cell of the table
            header = _table_cells(lines[first])
            cells = _table_cells(line)
            offset = m.start() - starts[row] - (line.find("|") + 1)
            col, run = 0, 0
            for i, cell in enumerate(cells):
                if run <= offset < run + len(cell) + 1:
                    col = i
                    break
                run += len(cell) + 1
            caption = _caption_above(lines, first)
            column = header[col] if col < len(header) else ""
            lead = cells[0] if cells and col != 0 and not _numbers_of(cells[0]) else ""
            label = f"{column} {lead}".strip()
            out.append(_Place(label, f"{label} {caption}", line, -1, tuple(header), tuple(cells), col, caption))
        else:
            lo = 0
            for b in _BLANK_LINE.finditer(text, 0, m.start()):
                lo = b.end()
            nxt = _BLANK_LINE.search(text, m.end())
            hi = nxt.start() if nxt else len(text)
            if hi - lo > _PARAGRAPH_MAX:
                lo, hi = max(lo, m.start() - _PARAGRAPH_FALLBACK), min(hi, m.end() + _PARAGRAPH_FALLBACK)
            s_lo, s_hi = _sentence_span(text, m.start(), m.end())
            out.append(_Place(None, text[lo:hi], text[s_lo:s_hi], m.start() - s_lo))
        if len(out) >= _MAX_PLACES:
            break
    return out


def _trial_path(path: str, labels: list[str]) -> tuple[str, str, str] | None:
    """``(setting label, value name, statistic)`` of a path of FI's per-setting table
    (``trials.m=20,pi1=0.2.fdr_bh.mean``)."""
    for label in labels:
        head = f"trials.{label}."
        if path.startswith(head):
            rest = path[len(head):]
            value, _, stat = rest.rpartition(".")
            if value and stat in _TABLE_STATS:
                return label, value, stat
    return None


# --- the settings a place names: "axis = value", a column headed by the axis, a text value by its name ---------------


class _Unknown:
    """A setting value the paper writes that is none of the study's: it equals nothing."""

    def __repr__(self) -> str:
        return "<not a value of the study>"


_NOT_A_VALUE = _Unknown()


def _canon(name: str) -> str:
    """A setting's name as compared: ``π₁``, ``\\pi_1``, ``pi_{1}``, ``π1`` and ``pi1`` are one name; so are ``n_points``
    and ``n points``."""
    out: list[str] = []
    for ch in unicodedata.normalize("NFKC", name):
        if "Ͱ" <= ch <= "Ͽ":
            parts = unicodedata.name(ch, "").split()
            if "LETTER" in parts and parts.index("LETTER") + 1 < len(parts):
                ch = parts[-1]
            elif "SYMBOL" in parts and parts.index("SYMBOL") > 0:
                ch = parts[parts.index("SYMBOL") - 1]
        out.append(ch)
    return re.sub(r"[^a-z0-9]", "", "".join(out).lower()).replace("lamda", "lambda")


def _same(a: Any, b: Any) -> bool:
    """Whether two setting values are the same value."""
    if isinstance(a, _Unknown) or isinstance(b, _Unknown):
        return False
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a.strip().lower() == b.strip().lower()
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _rounds_to(value: float, written: float, dec: int) -> bool:
    """Whether ``value`` printed at ``dec`` decimals (rounded, or truncated) is ``written``."""
    try:
        key = round(written, dec)
        if key in roundings(value, dec):
            return True
        return dec >= 0 and math.floor(value * 10.0 ** dec) / 10.0 ** dec == key
    except (ValueError, OverflowError):
        return False


@dataclass(frozen=True)
class _Axes:
    """The axes FI's per-setting numbers vary over: the name a paper writes for each (``by_name``: the axis's own full
    name, in any case, with Greek letters spelt out and punctuation dropped: ``π1`` for ``pi1``; never a part of it, so
    the ``n`` of "n = 400 runs" does not name ``n_points``), its values, which take text values (a method, a solver),
    and a pattern for each text value."""

    by_name: dict[str, str]
    values: dict[str, tuple[Any, ...]]
    text: frozenset[str]
    patterns: tuple[tuple[str, Any, "re.Pattern[str]"], ...]


def _axes_of(settings: list[dict[str, Any]]) -> _Axes:
    values: dict[str, list[Any]] = {}
    for s in settings:
        for k, v in s.items():
            seen = values.setdefault(str(k), [])
            if not any(_same(v, w) for w in seen) and not (isinstance(v, str) and v in seen):
                seen.append(v)
    text = frozenset(a for a, vs in values.items() if any(isinstance(v, str) for v in vs))
    by_name = {_canon(a): a for a in values}
    patterns: list[tuple[str, Any, re.Pattern[str]]] = []
    for a in text:
        for v in values[a]:
            words = [re.escape(w) for w in re.split(r"[\s_\-]+", str(v).strip()) if w]
            if isinstance(v, str) and words:
                patterns.append((a, v, re.compile(r"(?<![A-Za-z0-9])" + r"[\s_\-]+".join(words) + r"(?![A-Za-z0-9])",
                                                  re.I)))
    patterns.sort(key=lambda p: -len(str(p[1])))
    return _Axes(by_name, {a: tuple(vs) for a, vs in values.items()}, text, tuple(patterns))


def _text_value(axes: _Axes, axis: str, text: str) -> Any:
    for a, value, pattern in axes.patterns:
        if a == axis and pattern.fullmatch(text.strip()):
            return value
    return _NOT_A_VALUE


def _resolve(axes: _Axes, axis: str, token: str, percent: bool) -> Any:
    """The value of ``axis`` a paper's ``token`` names: the value it equals, else the one value it rounds to (``310.5``
    for 310.49); with a percent sign, as a fraction too. :data:`_NOT_A_VALUE` when it names none or several."""
    plain = token.replace(",", "")
    try:
        written = float(plain)
    except ValueError:
        return _NOT_A_VALUE
    dec = _written_precision(plain)
    numeric = [v for v in axes.values.get(axis, ()) if not isinstance(v, str)]
    readings = [(written, dec)] + ([(written / 100.0, max(dec, 0) + 2)] if percent else [])
    for x, _ in readings:
        for v in numeric:
            if _same(v, x):
                return v
    for x, d in readings:
        hits = [v for v in numeric if _rounds_to(float(v), x, d)]
        if len(hits) == 1:
            return hits[0]
    return _NOT_A_VALUE


_NAME_BEFORE_EQUALS = re.compile(r"(?<![\w\\{}])(\\?[^\W\d_][\w\\{}]*)\s*=\s*")
_SETTING_NUMBER = re.compile(r"(-?(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?)(\s*%)?(?!\w)")
_LIST_SEP = re.compile(r"\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or)\s+")
_TEXT_WORD = re.compile(r"[A-Za-z][\w\-]*")
_HEADER_NOTE = re.compile(r"\([^)]*\)|[*`$]")


def _bindings(text: str, axes: _Axes) -> list[tuple[str, Any, int, int]]:
    """``(axis, value, start, end)`` of each setting ``text`` names, in order: ``axis = value`` (``m = 20, 200 and 2000``
    names three: a value after the first counts only when it is one of the axis's values, so ``m = 200, 0.177`` names
    one), and a text value by its name (``RK4``). A number that merely equals a setting (the 20 of "20 repeats", the
    0.05 of "α = 0.05") names nothing."""
    out: list[tuple[str, Any, int, int]] = []
    for m in _NAME_BEFORE_EQUALS.finditer(text):
        axis = axes.by_name.get(_canon(m.group(1)))
        if axis is None:
            continue
        if axis in axes.text:
            word = _TEXT_WORD.match(text, m.end())
            if word:
                out.append((axis, _text_value(axes, axis, word.group(0)), m.start(1), word.end()))
            continue
        pos, first = m.end(), True
        while True:
            v = _SETTING_NUMBER.match(text, pos)
            if v is None:
                break
            value = _resolve(axes, axis, v.group(1), bool(v.group(2)))
            if not first and isinstance(value, _Unknown):
                break
            out.append((axis, value, m.start(1) if first else v.start(), v.end()))
            first = False
            sep = _LIST_SEP.match(text, v.end())
            if sep is None:
                break
            pos = sep.end()
    taken = [(a, b) for *_, a, b in out]
    for axis, value, pattern in axes.patterns:
        for t in pattern.finditer(text):
            if not any(t.start() < b and a < t.end() for a, b in taken):
                out.append((axis, value, t.start(), t.end()))
                taken.append((t.start(), t.end()))
    return sorted(out, key=lambda b: b[2])


@dataclass
class _Pool:
    """Numbers FI computed per setting: ``traceable`` for the lookup, ``of`` each path's ``(setting label, quantity,
    statistic)``, ``settings`` each label's settings, ``by_quantity`` the numbers of each ``(quantity, statistic)``."""

    traceable: Traceable
    of: dict[str, tuple[str, str, str]]
    settings: dict[str, dict[str, Any]]
    by_quantity: dict[tuple[str, str], list[tuple[str, float]]]

    def at(self, setting: dict[str, Any], quantity: str, stat: str) -> list[float]:
        """The numbers of ``quantity``'s ``stat`` at each label whose settings ``setting`` all gives."""
        return [number for label, number in self.by_quantity.get((quantity, stat), ())
                if all(a in setting and _same(setting[a], v) for a, v in (self.settings.get(label) or {}).items())]


def _pool(traceable: Traceable, of: dict[str, tuple[str, str, str]], settings: dict[str, dict[str, Any]]) -> _Pool:
    by_quantity: dict[tuple[str, str], list[tuple[str, float]]] = {}
    numbers = {**traceable.values, **traceable.derived}
    for path, (label, quantity, stat) in of.items():
        if path in numbers:
            by_quantity.setdefault((quantity, stat), []).append((label, numbers[path]))
    return _Pool(traceable, of, settings, by_quantity)


def _label_setting(label: str) -> dict[str, Any]:
    """``{"m": 20.0, "pi1": 0.2}`` of ``m=20,pi1=0.2`` (a reference's setting, core/reference_values.py :func:`pool`)."""
    out: dict[str, Any] = {}
    for part in label.split(","):
        k, eq, v = part.partition("=")
        if eq and k.strip():
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v.strip()
    return out


#: ", and" between two numbers joins two clauses ("0.653 at m = 20 against an exact 0.642, and 1.0 at m = 200"); after
#: two or more it ends a list ("0.177, 0.198, and 0.199").
_CLAUSE_AND = re.compile(r"\s*,\s*(?:and|or)\s+")
#: Settings named together: ``m = 200, π1 = 0``, ``(m = 20, π1 = 0.05)``, ``m = 200 and 2000``, ``Euler and RK4``.
_JOINED = re.compile(r"\s*(?:,|,?\s*(?:and|or))?\s*")
_PAREN = re.compile(r"\([^()]*\)")
_GAP_WORD = re.compile(r"[^\W\d_]+")
_GAP_SKIP = frozenset({"the", "a", "an", "of"})
#: Words that set one number against another ("0.198 at m = 20 versus 0.177 at m = 200"): a setting across one is
#: further from a number than any setting on its own side, so it names the number only when its own side names no
#: value of that axis ("0.0415 against 0.04 at m = 200": both at m = 200).
_CONTRAST = frozenset({"versus", "vs", "compared", "than", "while", "whereas", "but", "against"})
_TEX_MATH = re.compile(r"\\[()]")
_CHANGE_TO = re.compile(r"\b(?:rising|falling|rose|fell)\s+to\b", re.I)
#: A setting is named next to a number when at most this many words (another number counts as one; "the", "a", "an",
#: "of" and a bracket count as none) stand between them: "0.177 at m = 20", "at m = 200, π1 = 0.2 was 0.177", "0.0415
#: against 0.04 at m = 200".
_NEXT_TO = 4


@dataclass
class _Cluster:
    """Settings named together, and the settings each of its rows stands for (``m = 20, 200, 2000``: three rows)."""

    start: int
    end: int
    values: dict[str, list[Any]]
    in_brackets: bool = False

    def rows(self) -> list[dict[str, Any]] | None:
        longest = max(len(v) for v in self.values.values())
        if any(len(v) not in (1, longest) for v in self.values.values()):
            return None
        return [{a: vs[i] if len(vs) == longest else vs[0] for a, vs in self.values.items()} for i in range(longest)]


def _gap_words(text: str) -> tuple[int, int] | None:
    """``(the contrasts, the words)`` standing between a setting and a number (a contrast: :data:`_CONTRAST`, "rising
    to"), or ``None`` when a clause ends between them (a ``;``, an "and" / "or", a comma other than one comma alone:
    "at s = 2, 0.4200"; a bracket opened or closed between them: "0.199 at m = 20 (m = 200: 0.198)", but not TeX's
    backslash-bracket, which opens and closes mathematics)."""
    inner = _PAREN.sub(" ", _TEX_MATH.sub(" ", text))
    numbers = len(_CONTEXT_NUMBER.findall(inner))
    inner = _CONTEXT_NUMBER.sub(" ", inner)
    if ";" in inner or ("," in inner and inner.strip() != ",") or "(" in inner or ")" in inner:
        return None
    contrasts = len(_CHANGE_TO.findall(inner))
    inner = _CHANGE_TO.sub(" ", inner)
    words = [w.lower() for w in _GAP_WORD.findall(inner)]
    if any(w in ("and", "or") for w in words):
        return None
    contrasts += sum(1 for w in words if w in _CONTRAST)
    return contrasts, numbers + sum(1 for w in words if w not in _GAP_SKIP and w not in _CONTRAST)


def _merge(rows: list[dict[str, Any]], more: list[dict[str, Any]], words: tuple[int, int],
           near: dict[str, tuple[int, int]]) -> list[dict[str, Any]]:
    """``rows`` with ``more`` (``words`` from the number, :func:`_gap_words`) beside them: one row with many, many with one, or as many
    with as many in order. An axis named differently twice keeps the nearer setting (``near``: each axis's distance so
    far); two at the same distance name a value of none (:data:`_NOT_A_VALUE`), so the number fits no setting."""
    if len(more) == 1:
        pairs = [(r, more[0]) for r in rows]
    elif len(rows) == 1:
        pairs = [(rows[0], m) for m in more]
    elif len(rows) == len(more):
        pairs = list(zip(rows, more))
    else:
        return rows
    out = []
    for r, m in pairs:
        row = dict(r)
        for a, v in m.items():
            if a not in row or _same(row[a], v) or words < near[a]:
                row[a] = v
            elif words == near[a]:
                row[a] = _NOT_A_VALUE
        out.append(row)
    for a in {a for m in more for a in m}:
        near[a] = min(near.get(a, words), words)
    return out


def _named_beside(sentence: str, axes: _Axes, at: int) -> list[dict[str, Any]] | None:
    """The settings the sentence names for the number at ``at`` (one row, or one per setting it stands for), ``None``
    when the number is not one the sentence lists.

    A setting is named next to its number (:data:`_NEXT_TO`): on either side, with no clause ending between
    (:func:`_gap_words`); a setting in brackets is the number's it brackets: "40.6 points (m = 2000, π1 = 0.2)", "FDR
    (m = 20, π1 = 0.2) was 0.199". A list of numbers takes a list of settings in order ("0.177, 0.198 and 0.199 for m =
    20, 200, 2000"; two numbers joined by ", and" are two clauses, not a list), and a setting named before the
    sentence's first number, and nowhere else in it, holds for the whole sentence ("At π1 = 0.05 BH power was 0.393,
    0.267 and 0.234"). One axis named twice keeps the nearer setting (:func:`_merge`), never neither."""
    found = _bindings(sentence, axes)
    clusters: list[_Cluster] = []
    for axis, value, a, b in found:
        if clusters and _JOINED.fullmatch(sentence[clusters[-1].end:a]):
            clusters[-1].values.setdefault(axis, []).append(value)
            clusters[-1].end = b
        else:
            clusters.append(_Cluster(a, b, {axis: [value]}))
    for c in clusters:
        c.in_brackets = sentence[:c.start].rstrip().endswith("(") and sentence[c.end:].lstrip().startswith(")")
    spans = [(a, b) for *_, a, b in found]
    numbers = [m for m in _CONTEXT_NUMBER.finditer(sentence) if not any(a <= m.start() < b for a, b in spans)]
    groups: list[list[re.Match[str]]] = []
    for m in numbers:
        sep = sentence[groups[-1][-1].end():m.start()] if groups else ""
        if groups and _LIST_SEP.fullmatch(sep) and not (len(groups[-1]) == 1 and _CLAUSE_AND.fullmatch(sep)):
            groups[-1].append(m)
        else:
            groups.append([m])
    group = next((g for g in groups if any(m.start() == at for m in g)), None)
    if group is None:
        return None
    first_number = numbers[0].start() if numbers else len(sentence)

    def outer(c: _Cluster) -> tuple[int, int]:
        """``c``'s span with its brackets."""
        if not c.in_brackets:
            return c.start, c.end
        return sentence.rindex("(", 0, c.start), sentence.index(")", c.end) + 1

    def bracketed(c: _Cluster) -> bool:
        """Whether ``c`` is in brackets right after a number (that number's)."""
        lo, _ = outer(c)
        before = [m for m in numbers if m.end() <= lo]
        words = gap(before[-1].end(), lo) if before else None
        return words is not None and words[0] == 0 and words[1] <= _NEXT_TO

    def gap(lo: int, hi: int) -> tuple[int, int] | None:
        text = "".join(ch if not any(c.start <= i < c.end for c in clusters) else " "
                       for i, ch in enumerate(sentence[lo:hi], lo))
        return _gap_words(text)

    attached: list[tuple[_Cluster, tuple[int, int]]] = []
    for side in (1, -1):
        near = [c for c in clusters if (c.start >= group[-1].end() if side > 0 else c.end <= group[0].start())]
        for c in (near if side > 0 else reversed(near)):
            if side < 0 and c.in_brackets and bracketed(c):
                continue  # a setting in brackets right after another number is that number's
            lo, hi = outer(c)
            words = gap(group[-1].end(), lo) if side > 0 else gap(hi, group[0].start())
            if words is None or words[1] > _NEXT_TO:
                break
            attached.append((c, words))
    distance: dict[str, tuple[int, int]] = {}
    rows: list[dict[str, Any]] = [{}]
    for c, words in sorted(attached, key=lambda cw: cw[0].start):
        more = c.rows()
        if more:
            rows = _merge(rows, more, words, distance)
    elsewhere = [c for c in clusters if c.start >= first_number]
    whole: dict[str, Any] = {}
    for c in clusters:
        more = c.rows()
        if c.end <= first_number and more and len(more) == 1 and all(c is not near for near, _ in attached) \
                and not any(name in o.values for o in elsewhere for name in c.values):
            for name, v in more[0].items():  # two such settings of one axis name a value of none
                whole[name] = v if name not in whole or _same(whole[name], v) else _NOT_A_VALUE
    rows = [{**whole, **r} for r in rows]
    if len(rows) == len(group) > 1:
        return [rows[[m.start() for m in group].index(at)]]
    if len(rows) > 1 and len(group) > 1:
        shared = {a: v for a, v in rows[0].items() if all(a in r and _same(r[a], v) for r in rows)}
        return [shared]
    return rows


def _prose_fits(place: _Place, pool: _Pool, axes: _Axes, label: str, quantity: str, stat: str, *, every: bool) -> bool:
    """Whether the settings ``place``'s sentence names beside its number (:func:`_named_beside`) are ``label``'s (and,
    with ``every``, name each of them). A number standing for several settings ("1.0 at m = 200 and 2000") must be the
    number of each of them."""
    own = pool.settings.get(label) or {}
    rows = _named_beside(place.sentence, axes, place.at)
    if rows is None:
        return not every
    mine = [r for r in rows if all(_same(v, own[a]) for a, v in r.items() if a in own)]
    if not mine:
        return False
    if len(rows) > 1:
        token = next((m.group(0) for m in _CONTEXT_NUMBER.finditer(place.sentence) if m.start() == place.at), "")
        if not token:
            return False
        x, dec = float(token), _written_precision(token)
        if not all(any(_rounds_to(v, x, dec) for v in pool.at({**own, **r}, quantity, stat)) for r in rows):
            return False
    return not every or any(all(a in r for a in own) for r in mine)


def _table_fits(place: _Place, pool: _Pool, axes: _Axes, label: str, *, every: bool) -> bool:
    """Whether the settings ``place``'s table row names (:func:`_table_named`) are ``label``'s (and, with ``every``, name
    each of them)."""
    own = pool.settings.get(label) or {}
    named = _table_named(place, axes)
    if every and any(a not in named for a in own):
        return False
    return all(_same(v, own[a]) for a, vs in named.items() if a in own for v in vs)


def _table_named(place: _Place, axes: _Axes) -> dict[str, list[Any]]:
    """The settings a table row names for its number: a column headed by an axis, an ``axis = value`` in another cell
    of the row or in the number's column header, or in the caption an axis named nowhere in the row, with one value."""
    named: dict[str, list[Any]] = {}
    for i, head in enumerate(place.header):
        axis = axes.by_name.get(_canon(_HEADER_NOTE.sub("", head)))
        if axis is None or i == place.col or i >= len(place.cells):
            continue
        cell = place.cells[i].strip()
        if axis in axes.text:
            value = _text_value(axes, axis, cell)
        else:
            v = _SETTING_NUMBER.match(cell)
            value = _resolve(axes, axis, v.group(1), bool(v.group(2))) if v else _NOT_A_VALUE
        named.setdefault(axis, []).append(value)
    for text in (*(c for i, c in enumerate(place.cells) if i != place.col), place.column):
        for axis, value, *_ in _bindings(text, axes):
            named.setdefault(axis, []).append(value)
    said: dict[str, list[Any]] = {}
    for axis, value, *_ in _bindings(place.caption, axes):
        said.setdefault(axis, []).append(value)
    for axis, values in said.items():
        if axis not in named and len(values) == 1:
            named[axis] = values
    return named


def _fits(place: _Place, pool: _Pool, axes: _Axes, label: str, quantity: str, stat: str, *, every: bool) -> bool:
    if place.label is None:
        return _prose_fits(place, pool, axes, label, quantity, stat, every=every)
    return _table_fits(place, pool, axes, label, every=every)


def _names_a_setting(place: _Place, axes: _Axes) -> bool:
    """Whether ``place`` names any setting for its number (a sentence that names none, "the 0.0415 seen in the table",
    says nothing for or against one)."""
    if place.label is None:
        return any(_named_beside(place.sentence, axes, place.at) or ())
    return bool(_table_named(place, axes))


def _names_quantity(place: _Place, quantity: str, setting: dict[str, Any]) -> bool:
    """Whether ``place`` names ``quantity``: in a table, the column's header (or the row's first cell) names nothing
    but it and the caption completes its name; in prose, the paragraph names it."""
    wanted = _name_stems(quantity, _PATH_STOP, of_a_name=True)
    if not wanted:
        return False
    if place.label is not None:
        settings_words = frozenset().union(*(_stems_of_text(str(k)) for k in setting),
                                           *(_stems_of_text(v) for v in setting.values() if isinstance(v, str)))
        said = _stems_of_text(place.label) - settings_words
        if not said or not said <= wanted:
            return False  # the column is headed by something else (or by nothing that names a value)
    return wanted <= _stems_of_text(place.names)


#: Words that state a value as a reference, not as what was measured.
_REFERENCE_WORD = re.compile(r"\b(?:referen\w*|exact\w*|theor\w*|expected|expectation|nominal\w*|bound\w*|predict\w*"
                             r"|analytic\w*|closed[- ]form|formula\w*)\b", re.I)
_MEASURED_WORD = re.compile(r"\b(?:measur\w*|observ\w*|simulat\w*|reali[sz]\w*|estimat\w*|empiric\w*)\b", re.I)
#: A reference word governs a number at most this many words after it, besides the quantity's own name and its
#: settings ("the exact values 0.379, 0.163 and 0.060", "the exact Bonferroni FWER at m = 200 is 0.04878").
_REFERENCE_REACH = 3
_NUMBER_LIST = re.compile(r"(?:(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\w)%?\s*(?:,\s*(?:and\s+|or\s+)?|\s+and\s+|"
                          r"\s+or\s+)?)+")


def _stated_as_reference(place: _Place, measured: frozenset[str], quantity: frozenset[str], axes: _Axes) -> bool:
    """Whether ``place`` states its number as a reference value of the quantity whose name's words are ``quantity``:
    in prose, a reference word stands before it in the same clause, at most :data:`_REFERENCE_REACH` other words before
    it and no word of a measurement between ("in agreement with the exact values 0.379, 0.163 and 0.060"; not "it was
    0.3794, as the exact formula gives", nor "matching the exact formula, power was 0.3794"); in a table, its column's
    header names a reference, or the caption does and the column's header names none of the measured quantities
    (``measured``)."""
    if place.label is None:
        before = place.sentence[:max(place.at, 0)]
        last = None
        for last in _REFERENCE_WORD.finditer(before):
            pass
        if last is None:
            return False
        between = before[last.end():]
        for *_, a, b in sorted(_bindings(between, axes), key=lambda x: -x[2]):
            between = between[:a] + "\x00" + between[b:]
        between = _NUMBER_LIST.sub(" ", re.sub(r"\x00(?:\s*(?:,|,?\s*and|,?\s*or)?\s*\x00)*", " ", between))
        if _MEASURED_WORD.search(between) or re.search(r"[,;:]", between):
            return False
        words = [w for w in re.findall(r"[^\W\d_]+", between)
                 if w.lower() not in ("and", "or") and not (_name_stems(w, _COMMON_STOP) <= quantity)]
        return len(words) <= _REFERENCE_REACH
    if _REFERENCE_WORD.search(place.column):
        return True
    return bool(_REFERENCE_WORD.search(place.caption)) and not (_stems_of_text(place.column) & measured)


def _place_holds(
    place: _Place, cand: float, dec: int, how: str, table: _Pool | None, references: _Pool | None, axes: _Axes,
    measured: frozenset[str],
) -> bool:
    """Whether ``place`` accounts for ``cand`` as one of FI's per-setting numbers.

    A number of FI's per-setting table: never where the settings the place names are another setting's. Where they are
    not, as before (the sentence names the quantity, or the run holds so few values like it that a match is not luck),
    or, unshifted, where the place names its setting in full and its quantity (a table row through its header).

    A reference value FI computed: only where the place states it as a reference (:func:`_stated_as_reference`), names
    its quantity and names no other setting; never as its complement. A reference presented as the measured result is
    not accounted for by it."""
    if table is not None:
        words: frozenset[str] | None = None
        for path in table.traceable.paths_at(cand, dec):
            hit = table.of.get(path)
            if hit is not None and not _fits(place, table, axes, *hit, every=False):
                continue
            limit = MAX_CHANCE_UNNAMED / SHIFT_CHANCE_DIVISOR / (SHIFT_CHANCE_DIVISOR if how else 1)
            if table.traceable.chance(cand, dec) <= limit:
                return True
            if words is None:
                words = _name_stems(place.sentence, _COMMON_STOP)
            if table.traceable.words_of(path) & words:
                return True
            if hit is not None and not how and _fits(place, table, axes, *hit, every=True) \
                    and _names_quantity(place, hit[1], table.settings.get(hit[0]) or {}):
                return True
    if references is not None and "complement" not in how:
        for path in references.traceable.paths_at(cand, dec):
            hit = references.of.get(path)
            if hit is None:
                continue
            wanted = _name_stems(hit[1], _PATH_STOP, of_a_name=True)
            if wanted <= _stems_of_text(place.names) and _stated_as_reference(place, measured, wanted, axes) \
                    and _fits(place, references, axes, *hit, every=False):
                return True
    return False


@dataclass
class _PerSetting:
    """FI's own per-setting numbers for :func:`check`: its table of the trials and its reference values, with the axes
    their settings vary over and the words of every measured quantity."""

    table: _Pool | None
    references: _Pool | None
    axes: _Axes
    measured: frozenset[str]

    @classmethod
    def of(cls, table: Traceable | None, settings: dict[str, dict[str, Any]],
           references: Traceable) -> "_PerSetting":
        labels = sorted(settings, key=len, reverse=True)  # a longer label first: `m=20,pi1=0.2` is not `m=20`
        table_pool = None
        if table is not None:
            of = {p: hit for p in table.all_paths() if (hit := _trial_path(p, labels)) is not None}
            table_pool = _pool(table, of, settings)
        reference_pool = None
        ref_settings: dict[str, dict[str, Any]] = {}
        if len(references):
            of = {}
            for path in references.all_paths():
                metric, _, label = path[len("reference."):].partition(".")
                ref_settings[label] = _label_setting(label)
                of[path] = (label, metric, "reference")
            reference_pool = _pool(references, of, ref_settings)
        measured = frozenset().union(*(_name_stems(q, _PATH_STOP, of_a_name=True)
                                       for _, q, _ in (table_pool.of.values() if table_pool else ())))
        return cls(table_pool, reference_pool, _axes_of([*settings.values(), *ref_settings.values()]), measured)

    def accounts_for(self, places: list[_Place], token: str, readings: tuple[float, ...]) -> bool:
        """Whether a place the paper prints ``token`` accounts for it (:func:`_place_holds`) and every place that names a
        setting for it does: a number printed right in one place and beside another setting in the next is wrong in the
        second, while "the 0.0415 seen in the table", naming no setting, is the same number again."""
        candidates = [(cand, dec, how) for reading in readings for cand, dec, how in _variants(reading, token)]
        held = False
        for place in places:
            if any(_place_holds(place, cand, dec, how, self.table, self.references, self.axes, self.measured)
                   for cand, dec, how in candidates):
                held = True
            elif _names_a_setting(place, self.axes):
                return False
        return held


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
    oracle_checks: Any = None,
    n_seeds: int = 0,
    optimisation: Any = None,
    run_facts: Any = None,
    trial_table: Any = None,
    trial_settings: dict[str, dict[str, Any]] | None = None,
    references: dict[str, float] | None = None,
) -> ProvenanceReport:
    """Flag every number in the paper that nothing in this run accounts for.

    The aggregates are passed in rather than recomputed so this module stays
    free of ``core.engine`` and can be tested without one.

    ``trial_settings``: each setting label of ``trial_table`` with the settings that tell it apart (its axes that take
    more than one value, :func:`core.trial_table.settings`). A table number is never accounted for where the paper
    prints it beside another setting ("m = 200" for the number of m = 20, a permuted list), and it is also accounted for
    where the paper names its setting in full and its quantity. ``references``: the reference values FI computed itself
    from the frozen plan's formulas (:mod:`core.reference_values`), a pool of their own, accounting for a number only
    where the paper states it as a reference ("the exact value 0.379"), never for one presented as what was measured.
    Both are read place by place: every place the paper prints the number must account for it (:func:`_place_holds`).
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
        oracle_checks=oracle_checks, optimisation=optimisation, run_facts=run_facts,
    )
    # FI's own table of what every trial gave, per setting (core/trial_table.py): the mean, SD and interval of each value
    # over the trials, which the paper may quote and RESULT_JSON need not hold. A pool of its own, read only when the
    # results account for nothing: added to the results, its hundreds of numbers would make every match there likelier
    # to be luck, and correct numbers the results do account for would be flagged.
    table_kept, table_stats = _trace_values(trial_table or {}, "trials")
    table = Traceable(dict(table_kept), table_stats) if table_kept or table_stats else None
    settings = {str(k): dict(v) for k, v in (trial_settings or {}).items() if isinstance(v, dict)}
    reference_pool = Traceable({f"reference.{k}": float(v) for k, v in (references or {}).items()
                                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)})
    per_setting = _PerSetting.of(table, settings, reference_pool) if table is not None or len(reference_pool) else None
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
    flat_text = normalise(_TEX_THOUSANDS.sub(",", paper_text)).replace(_MINUS_SIGN, "-")
    beside = _intervals_beside(flat_text)
    for value, token, context in paper_numbers(paper_text):
        if not _carries_enough_precision(token) or _is_year(value, token):
            continue
        # A seed is chosen, never computed, so it appears in no result.
        if _decimals(token) == 0 and _SEED_CONTEXT.search(context):
            continue
        report.paper_numbers += 1
        hedged = bool(_HEDGE.search(context))
        # A number after a plus-minus sign is two numbers, and either traces.
        readings = (value, -value) if _either_sign(token, context) else (value,)
        if any(
            _is_accounted_for(
                traceable, cand, dec, lambda: _naming_words(flat_text, token, context), shifted=bool(how),
            )
            for reading in readings
            for cand, dec, how in _variants(reading, token)
        ):
            continue
        if hedged and traceable.find_near(value, HEDGED_REL_TOL) is not None:
            continue
        if _derivable(value, token, context, traceable):
            continue
        if _a_list_the_run_holds(token, traceable):
            continue
        if _interval_of_a_result(token, beside, traceable):
            continue
        if table is not None and _interval_of_a_result(token, beside, table):
            continue
        # FI's per-setting numbers and reference values, read place by place (:func:`_place_holds`). A number gets a
        # second chance to match by luck here, so an unnamed match must be that much rarer.
        if per_setting is not None:
            places = _setting_contexts(flat_text, token)
            if places:
                if per_setting.accounts_for(places, token, readings):
                    continue
            elif table is not None and any(  # the tokenizer rewrote the text around it: only its short context is left
                _is_accounted_for(
                    table, cand, dec, lambda: _name_stems(context, _COMMON_STOP), shifted=bool(how),
                    max_chance=MAX_CHANCE_UNNAMED / SHIFT_CHANCE_DIVISOR,
                )
                for reading in readings
                for cand, dec, how in _variants(reading, token)
            ):
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


# --- what a refine's extension added: is it in the paper? --------------------

_INDEX = re.compile(r"\[\d+\]")
_NAME_PART = re.compile(r"[A-Za-z]+")
_NAME_STOP = frozenset({"the", "and", "for", "per", "all", "val", "value", "values", "list", "data", "result",
                        "results", "each", "over", "with", "add", "also", "include", "report", "show", "compute",
                        "please", "missing", "number"})
# The short words result names are written with, and the words a paper writes them as. A short word is matched only
# as itself or one of these ("max" is "maximum", not "maximal"; "abs" is "absolute", not "above").
_SHORT_WORDS = {
    "max": ("max", "maximum", "maxima"), "min": ("min", "minimum", "minima"), "abs": ("abs", "absolute"),
    "rel": ("rel", "relative"), "err": ("err", "error", "errors"), "std": ("std", "standard"),
    "avg": ("avg", "average", "mean"), "var": ("var", "variance"), "num": ("num", "number"), "tot": ("tot", "total"),
    "cum": ("cum", "cumulative"), "sum": ("sum", "sums"), "inf": ("inf", "infinity"), "sec": ("sec", "second", "seconds"),
}
# A sentence that says a thing was NOT done names it without reporting it.
_NOT_DONE = re.compile(
    r"(?:\bnot|\bnever|n't)\s+(?:been\s+|yet\s+)?(?:computed|calculated|reported|measured|available|included|shown|done"
    r"|obtained|estimated|recorded|run)\b|\bcould\s*n[o']t\b|\bcannot\b|\bunavailable\b|\bomitted\b"
    # ... and one that restates the request, or talks about it without giving it, names it without reporting it.
    r"|\basked\b|\brequested\b|\bqualitativ",
    re.I,
)
# A number that points somewhere is not a reported value: a figure, table, section or equation number, a citation.
_POINTER = re.compile(
    r"\b(?:Fig(?:ure)?s?\.?|Tables?|Sections?|Sec\.|Eqs?\.?|Equations?|Appendix)\s*[A-Z]?\d+(?:\.\d+)*|\[\d+(?:\s*[,–-]\s*\d+)*\]",
    re.I,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def result_paths(result_json: Any) -> dict[str, int]:
    """How many numbers the results hold under each name, list indices dropped (``errors.euler[3]`` counts for
    ``errors.euler``): what the results held before a refine extended the script."""
    counts: dict[str, int] = {}
    for p, _v in flatten_numbers(result_json or {}, keep_zero=True):
        name = _INDEX.sub("", p)
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def added_results(result_json: Any, before: Any) -> tuple[dict[str, list[float]], set[str], list[float]]:
    """What an extension added to the results: ``(groups, grown, old)``.

    ``groups`` maps each new result to its values: a name the results did not have, grouped by the first part of its
    path they did not have (``max_abs_errors.euler`` -> ``max_abs_errors``), or a list the results had that now holds
    more numbers (``errors`` with n=64 appended; only the added numbers). ``grown`` names the second kind. ``old`` is
    every number under a name the results had. ``before`` is :func:`result_paths` of the earlier results (a plain
    list of names is read as names whose count is not known, so nothing counts as appended)."""
    counts: dict[str, float] = (
        {str(k): float(v) for k, v in before.items()} if isinstance(before, dict)
        else {str(k): math.inf for k in before or []}
    )
    known = {".".join(p.split(".")[:i]) for p in counts for i in range(1, p.count(".") + 2)}
    groups: dict[str, list[float]] = {}
    grown: set[str] = set()
    old: list[float] = []
    seen: dict[str, int] = {}
    for path, value in flatten_numbers(result_json or {}, keep_zero=True):
        name = _INDEX.sub("", path)
        if name in counts:
            seen[name] = seen.get(name, 0) + 1
            if seen[name] <= counts[name]:
                old.append(value)
            else:
                groups.setdefault(name, []).append(value)
                grown.add(name)
            continue
        parts = name.split(".")
        group = next((".".join(parts[:i]) for i in range(1, len(parts) + 1) if ".".join(parts[:i]) not in known), name)
        groups.setdefault(group, []).append(value)
    return groups, grown, old


def _name_words(name: str) -> list[str]:
    """``max_abs_errors`` -> ``["max", "abs", "error"]``: the words of a result's name, split on ``_``, ``.`` and
    camelCase, a plural ``s`` dropped, short and generic words left out."""
    words = []
    for part in _NAME_PART.findall(re.sub(r"([a-z])([A-Z])", r"\1 \2", name)):
        w = part.lower()
        w = w[:-1] if len(w) > 3 and w.endswith("s") else w
        if len(w) >= 3 and w not in _NAME_STOP:
            words.append(w)
    return words


def _says_word(word: str, found: set[str]) -> bool:
    if len(word) <= 3:
        return any(f in _SHORT_WORDS.get(word, (word, word + "s")) for f in found)
    return any(f.startswith(word) for f in found)


def _names_it(words: list[str], unit: str) -> bool:
    """Does this sentence or table report the result: every word of its name, a number (not a figure, table or
    section number, not a citation), no "not", and not a heading?"""
    if not words or unit.lstrip().startswith("#") or _NOT_DONE.search(unit):
        return False
    if not re.search(r"\d", _POINTER.sub(" ", unit)):
        return False
    found = {m.lower() for m in _NAME_PART.findall(unit)}
    return all(_says_word(w, found) for w in words)


def _units(paper_text: str) -> list[str]:
    """The paper as sentences, with each table kept whole (its header names what its rows print)."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", paper_text):
        para = "\n".join(line for line in para.splitlines() if not line.lstrip().startswith("#"))
        if not para.strip():
            continue
        if para.lstrip().startswith("|"):
            out.append(para)
        else:
            out.extend(s for s in _SENTENCE_END.split(" ".join(para.split())) if s.strip())
    return out


def _within(value: float, values: list[float], tol: float) -> bool:
    for v in values:
        denom = max(abs(value), abs(v))
        if denom == 0.0 or abs(value - v) / denom <= tol:
            return True
    return False


def _prints_new_value(paper_text: str, new: list[float], old: list[float]) -> bool:
    """Does the paper print one of ``new`` (at its own precision) that is not also one of ``old``? A value the results
    already held under another name says nothing about whether the new quantity is reported."""
    for value, token, _ctx in paper_numbers(paper_text):
        tol = 0.51 * 10.0 ** -(max(_significant_digits(token), 1) - 1)
        if _within(value, new, tol) and not _within(value, old, tol):
            return True
    # A value below the tokenizer's smallest magnitude, written as it is stored. Not a whole number: "3" is also a
    # section, a figure or a citation. The reference list and citation brackets are removed first.
    text = normalise(paper_text)
    for v in new:
        if v == 0.0 or float(v).is_integer() or _within(v, old, 1e-9):
            continue
        for form in {f"{v:g}", f"{v:.3g}", f"{v:.4g}"}:
            if re.search(r"(?<![\w.])" + re.escape(form) + r"(?![\w]|\.\d)", text):
                return True
    return False


def unreported_results(paper_text: str, result_json: Any, before: Any, asked: list[str]) -> list[str]:
    """The results a refine's extension added, for the numbers a person asked for, that the paper does not report.

    What the extension added is :func:`added_results`. The new results whose names share a word with the request are
    the ones asked for, and each of them must be in the paper; when none does, one new result in the paper is enough.
    A result is in the paper when the paper prints one of its values (rounded as the paper rounds it) that no old
    result also has, or, for a new name, when one sentence or table says the words that set its name apart from the
    nearest old name together with a number and without a "not" ("the maximum absolute error ... equals the error at
    t=1" for ``max_abs_errors`` beside ``errors``). The name matters when the values are old ones again: a maximum
    error that equals the error at the end point is reported only by saying so. Numbers appended to a list are
    reported only by printing one of them.

    No model call; ``[]`` when the extension added nothing new (then there is nothing to look for)."""
    groups, grown, old = added_results(result_json, before)
    if not groups:
        return []
    # A number the request itself names ("add n=64") is the setting asked about: printing it ("we added n = 64") is
    # not printing the result, so it is no evidence (the list of settings that grew by 64 is reported by its partner).
    named = [v for a in asked for v, _t in _numbers_in_text(a)]
    groups = {g: [v for v in vals if not _within(v, named, 1e-12)] for g, vals in groups.items()}
    groups = {g: vals for g, vals in groups.items() if vals or g not in grown}
    if not groups:
        return []
    asked_words = [w for a in asked for w in _name_words(a)]
    related = [g for g in groups
               if any(w.startswith(a) or a.startswith(w) for w in _name_words(g) for a in asked_words)]
    units = _units(paper_text)
    old_names = [set(_name_words(p)) for p in (before or [])]

    def distinct(words: list[str]) -> list[str]:
        """The words that set a new result apart from the old one nearest to it: ``euler_max_abs_errors`` beside
        ``euler_errors`` is ``max``, ``abs``. ``euler`` and ``error`` say which case and what kind, and the paper says
        those already; only saying the new words reports the new result. Nearest: the most words in common, then the
        shortest name."""
        near = max(old_names, key=lambda n: (len(n & set(words)), -len(n)), default=set())
        return [w for w in words if w not in near] or words

    def reported(group: str) -> bool:
        if _prints_new_value(paper_text, groups[group], old):
            return True
        if group in grown:
            return False
        words = distinct(_name_words(group))
        return any(_names_it(words, u) for u in units)

    if related:
        return [g for g in related if not reported(g)]
    return [] if any(reported(g) for g in groups) else sorted(groups)


