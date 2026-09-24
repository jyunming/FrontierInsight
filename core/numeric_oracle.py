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
number that is a **slip** of a real result -- its digits in another order, or
its last digit one off the correct rounding. A number far from every result
is assumed to come from somewhere else and is ignored. A number that rounds
correctly is correct.

Two signals, strongest first:

``transposed``
    Same digits, different order (2.41 vs 2.14; 0.045 vs 0.054). Almost
    never a coincidence, and the classic way a number gets copied wrong.

``near_miss``
    A last-digit slip: a result whose correct rounding to the
    paper's own precision is one unit of the paper's last printed digit from
    the number the paper prints (2.12 for 2.1259, which rounds to 2.13; 0.184
    for 0.18346, which rounds to 0.183), for a number printed with at least
    ``MIN_SLIP_DECIMALS`` decimals. That is the truncation, or the mistyped
    last digit, that stored quests really contain.

Both are reported with the JSON path they contradict, so the writer node
gets told which number to fix rather than "something is wrong".

Why ``near_miss`` is that narrow
================================
It used to be any number within ``NEAR_REL`` (25%) of a result. A result
dictionary holds hundreds of numbers, so almost every number a paper prints is
within a quarter of one: replayed over the 136 stored quests with results, the
check made 197 of these findings in 59 quests (median 2, most 37), and most were
not copying errors: confidence-interval bounds, hand-computed sums, settings,
years, citation numbers, file counts, siblings in a list of similar values, and
the theoretical values 0.333 and 0.667 read against the nearest result. A
``near_miss`` is therefore a last-digit slip only. What remains is 20 findings in
9 of those quests, and six rules keep the rest out:

* a number inside a citation bracket (``[6, 15]``), which is a reference number;
* a result whose name is an identifier or a count of files (``id``, ``index``,
  ``seed``, ``year``, ``n_files``, ...);
* a result the paper also prints correctly rounded, to two decimals or more,
  somewhere else: the paper states that result right, so a number one digit off
  beside it is another quantity (a sibling, a bound, a sum worked by hand). This
  one takes the place of leaving out the elements of an array, which also left out
  a slip stored quests really contain, in a nine-point sweep;
* a repeating-decimal constant printed with three decimals or more (0.333, 0.667,
  0.167, 0.143: 1/3, 2/3, 1/6, 1/7 ...), which is a theoretical value more often
  than a measurement;
* a result that is a shorter rounding of the number the paper prints (0.667
  against a stored 0.67), read against that result only: a stored 0.2 must not
  clear every number in [0.15, 0.25);
* a setting the run was given (below), as before.

A digit transposition is read against every result, and only the first rule (a
citation number is not a measurement) applies to it: the same digits in another
order is rarely a coincidence, wherever the result sits.

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

# A ``near_miss`` is only a last-digit slip, and only for a number printed with at
# least this many decimals: at one decimal or none the "last digit" is a whole unit
# of a count or a ratio ("26 times" against 25), which says nothing about rounding.
MIN_SLIP_DECIMALS = 2

# A result named like one of these is an identifier, an index or a count of files,
# not a quantity the paper reports, so a paper number near it is a coincidence.
_IDENTIFIER_LEAF = re.compile(
    r"(?:^|_)(?:id|ids|index|idx|seed|seeds|year|years|n_files|n_obs|n_observations|file|files)(?:_|$)",
    re.IGNORECASE,
)
_ARRAY_INDEX = re.compile(r"\[\d+\]")
# The denominators of the theoretical values a paper prints beside its results
# (1/3 = 0.333, 2/3 = 0.667, 1/6 = 0.167, 1/7 = 0.143 ...): repeating decimals.
_CONSTANT_DENOMINATORS = (3, 6, 7, 9, 11, 12)
# ``[6, 15]`` and ``[3-5]``: reference numbers, not measurements.
_CITATION_GROUP = re.compile(r"\[\s*\d+(?:\s*[,;–—-]\s*\d+)*\s*\]")

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
# A closing brace takes the space before it, but a bare caret exponent leaves
# the space after it alone: "1.25×10^-5 here" once became "1.25e-5here", which
# the tokenizer then did not read at all.
_LATEX_SCI = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|×|·|\\!)\s*10\s*\^\s*\{?\s*(-?\+?\d+)(?:\s*\})?"
)

# The same power of ten in Unicode superscripts: a real paper's "1.25×10⁻⁵" was
# read as 1.25, and the number provenance check flagged it as a number the run
# never computed (the run held 1.25e-05). Rewritten to the caret form first.
_SUPERSCRIPT_SCI = re.compile(r"(\d)\s*([×·])\s*10([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")

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
    printed: str = ""  # the number as the paper wrote it ("2.12"), for a near_miss

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
        if self.kind == "transposed":
            lead = "digits transposed"
        elif self.printed:
            decimals = _decimals_of_token(self.printed)
            lead = f"rounds to {round(self.result_value, decimals):.{decimals}f}, not {self.printed}"
        else:
            lead = f"off by {self.rel_error * 100:.1f}%"
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
    cleaned = _SUPERSCRIPT_SCI.sub(
        lambda m: f"{m.group(1)}{m.group(2)}10^{m.group(3).translate(_SUPERSCRIPT_DIGITS)}", cleaned,
    )
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


def _decimals_of_token(token: str) -> int:
    t = token.replace(",", "")
    return len(t.split(".", 1)[1]) if "." in t else 0


def _short_decimals(actual: float) -> int | None:
    """How many decimals a stored result is written with, when it is a short number
    (a rounding someone stored: 0.67), and ``None`` for a long one (0.6666666666)
    or one in exponent notation."""
    text = repr(float(actual))
    if "e" in text.lower() or "." not in text:
        return None
    decimals = len(text.split(".", 1)[1])
    return decimals if decimals <= 6 else None


def _is_rounding_of(paper: float, actual: float, token: str) -> bool:
    """True when ``actual`` is a shorter rounding of the number the paper prints:
    0.67 stored and 0.667 printed. ``_rounds_to`` reads the other way round."""
    shorter = _short_decimals(actual)
    if shorter is None or shorter >= _decimals_of_token(token):
        return False
    return round(paper, shorter) == round(actual, shorter)


def _is_eligible_near_result(path: str) -> bool:
    """May a paper number be read as a slip of the result at ``path``? Not an
    identifier, an index, a seed, a year or a count of files. An element of an
    array is eligible: a slip stored quests really contain sits in a nine-point
    sweep (``coherence_sweep.dipole_nils[8]``)."""
    return not _IDENTIFIER_LEAF.search(_ARRAY_INDEX.sub("", path.rsplit(".", 1)[-1]))


def _stated_correctly(actual: float, numbers: list[tuple[float, str, str]]) -> bool:
    """Does the paper print ``actual`` correctly rounded, to ``MIN_SLIP_DECIMALS``
    decimals or more, somewhere? Then it states that result right, and a number
    beside it that is one digit off is another quantity: a sibling, a bound, a
    sum worked by hand."""
    return any(
        _decimals_of_token(token) >= MIN_SLIP_DECIMALS and _rounds_to(value, actual, token)
        for value, token, _ctx in numbers
    )


def _is_repeating_constant(value: float, token: str) -> bool:
    """Is the number, printed with three decimals or more, a repeating-decimal
    constant (0.333, 0.667, 0.167, 0.143)? That is a theoretical value more often
    than a measurement, and it is compared with the result nearest to it."""
    decimals = _decimals_of_token(token)
    if decimals < 3:
        return False
    for q in _CONSTANT_DENOMINATORS:
        p = round(value * q)
        if p and math.gcd(abs(p), q) == 1 and round(p / q, decimals) == round(value, decimals):
            return True
    return False


def _is_last_digit_slip(value: float, token: str, actual: float) -> bool:
    """Is the paper's number one unit of its own last printed digit away from
    ``actual`` correctly rounded to that precision (2.12 for 2.1259)?"""
    decimals = _decimals_of_token(token)
    return decimals >= MIN_SLIP_DECIMALS and abs(round(actual, decimals) - value) <= 1.01 * 10 ** -decimals


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
    # a positive number, near some positive result. Folded here, before the
    # shared extractor is called, and not inside it: ``number_provenance`` folds
    # it the same way in its own reading of the paper, so neither check's
    # findings depend on what the other does with the extractor.
    # A citation bracket holds reference numbers, not measurements: taken out for
    # this check only, since the extractor is shared with ``number_provenance``.
    numbers = extract_paper_numbers(_CITATION_GROUP.sub(" ", paper_text.replace(_MINUS_SIGN, "-")))
    report.paper_numbers = len(numbers)
    settings = list(declared) if declared else []

    seen: set[tuple[float, str]] = set()
    for value, token, ctx in numbers:
        nearest: tuple[float, str, float] | None = None  # (result, path, rel), any result
        eligible: tuple[float, str, float] | None = None  # the nearest one a near_miss may be read against
        cleared = False
        for path, actual in results:
            if actual == value or _rounds_to(value, actual, token):
                cleared = True
                break  # an exact or correctly-rounded match clears it
            denom = max(abs(actual), abs(value))
            rel = abs(actual - value) / denom if denom else 0.0
            if rel <= NEAR_REL:
                if nearest is None or rel < nearest[2]:
                    nearest = (actual, path, rel)
                if _is_eligible_near_result(path) and (eligible is None or rel < eligible[2]):
                    eligible = (actual, path, rel)
        if cleared or nearest is None:
            continue

        actual, path, rel = nearest
        if _digit_bag(value) and _digit_bag(value) == _digit_bag(actual):
            kind = "transposed"
        else:
            # Only a last-digit slip, against a result the paper does not also state right,
            # for a number that is not a theoretical constant or a setting the run was given.
            kind = "near_miss"
            if eligible is None:
                continue
            actual, path, rel = eligible
            if not _is_last_digit_slip(value, token, actual) or _is_repeating_constant(value, token):
                continue
            if _is_rounding_of(value, actual, token) or _stated_correctly(actual, numbers):
                continue
            if settings and _is_declared(value, token, settings):
                continue
        key = (value, path)
        if key in seen:
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
                printed=token if kind == "near_miss" else "",
            )
        )

    # A reference quantity that is zero everywhere contradicts the results
    # themselves, so it is found without reading the paper at all.
    report.findings.extend(trivial_reference_findings(result_json))

    # Strongest signal first, so a truncated report still leads with the
    # finding most likely to be a real error.
    report.findings.sort(key=lambda f: (_KIND_RANK.get(f.kind, 9), -f.rel_error))
    return report
