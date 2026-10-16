"""Deterministic check that a number is the QUANTITY the paper says it is.

Why this exists
===============
The claim check asks a model whether a cited sentence is supported by evidence: meaning, but with no arithmetic anywhere.
Between that and the rule that the writer types no result (``core/typed_numbers.py``) sits a gap that shipped in delivered
papers:

* a paper labelled its three-seed **t-intervals** "exact binomial 95% CI";
* a paper called the **major-outbreak probability** (~34%) the probability the disease would "simply vanish": the
  extinction probability is its complement (~66%). The number matched a result and the reversed meaning was invisible.

Both numbers are correct but mislabelled. So this module reads what a number is **said to be** and tests that claim
against the run. (The checks that read the paper's numbers against the run's, for an effect size, a threshold printed as
a p-value, an axis limit printed as a count and a seed the paper states, are gone: a result is a key FI fills in from its
records, and a number the writer types that is not a number of the plan is found by ``core/typed_numbers.py``.)

The design rule: say nothing unless the contradiction can be NAMED
=================================================================
False positives are the failure mode that matters here. A spurious flag costs a whole rewrite. So every check below is a
**recomputation**, not a pattern match: each one fires only when it can point at the run's own numbers and say what the
paper called them instead. A claim this module cannot recompute is a claim it stays silent about.

Two checks, each independently gated
------------------------------------
``interval_method_mismatch``
    An interval labelled with a named method (exact binomial, Clopper-Pearson, Wilson, bootstrap, ...) whose endpoints
    reproduce FI's own t-interval over the replicate seeds. The contradiction is not "this interval is wrong" but "this
    interval IS the k-seed t-interval, under another name", so no second interval estimator has to be implemented to
    prove it.

``probability_complement``
    A sentence giving the probability that the disease/outbreak dies out, whose number matches a result named for the
    *opposite* event (a major outbreak) and matches no result named for extinction.

Both are recorded through ``StatReport`` with the contradiction spelled out, so the paper's writer is told which number to
relabel rather than "a statistic is wrong somewhere".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterator

from core.disclosure import strip_for_checks
from core.numeric_oracle import flatten_numbers

# A replicated run needs at least this many seeds before its spread means
# anything; below it there is no t-interval and no effect size to compare.
MIN_SEEDS = 2

# How far a hedged number ("~34%", "about 0.34") may sit from a computed value
# and still be considered the same quantity. An unhedged number must match to
# the precision the paper itself wrote.
HEDGED_REL_TOL = 0.10

# Longest window allowed between the parts of the reversed-probability
# pattern, so the match cannot wander into a neighbouring clause.
_SPAN = 100

_CONTEXT_CHARS = 170


@dataclass(frozen=True)
class StatFinding:
    """One statistic the paper describes as something the run did not compute."""

    kind: str
    message: str
    context: str = ""

    def describe(self) -> str:
        ctx = f" — “{self.context}”" if self.context else ""
        return f"{self.message}{ctx}"


@dataclass
class StatReport:
    findings: list[StatFinding] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
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
            "checks_run": list(self.checks_run),
            "findings": [
                {"kind": f.kind, "message": f.message, "context": f.context}
                for f in self.findings
            ],
        }


# --- text normalisation -------------------------------------------------------

# Blocks whose numbers are structural rather than claimed. Markdown TABLES are
# deliberately kept: a results table is exactly where a mislabelled interval
# does its damage (the "exact binomial" case lived in a table header).
_STRIP_BLOCKS = (
    re.compile(r"^```.*?^```", re.S | re.M),
    re.compile(
        r"^#{1,6}[ \t]*(?:references|further[ \t]+reading)[ \t]*$.*?(?=^#{1,6}[ \t]|\Z)",
        re.S | re.M | re.I,
    ),
    re.compile(r"!\[[^\]]*\]\([^)]*\)"),
    # Inline citation markers: "[6, 7]", "[12]", "[3-5]". A bracket holding
    # only digits and separators is a reference, never a measurement — the
    # numeric oracle learned the same lesson from DOI prefixes. A real paper's
    # "...based on current state probabilities [6, 7]." had its 7 read as a
    # count of runs, the only false positive this check produced over 35
    # papers. Intervals keep their decimal points, so "[0.251, 0.309]" is
    # not matched here.
    re.compile(r"\[\s*[0-9]+(?:\s*[,;\-–]\s*[0-9]+)*\s*\]"),
)

_LATEX_CMD = re.compile(
    r"\\(?:mathrm|mathbf|mathit|text|textbf|textit|emph|operatorname)\s*\{([^{}]*)\}"
)


def normalise(text: str) -> str:
    """Strip the LaTeX a paper writes its statistics in, so ``Cohen's $d > 16$``
    and ``$p < 0.001388$`` read as plain text. Both observed claims were inside
    ``$…$``; leaving the delimiters in place made neither of them parse.

    The engine's paragraph on how the result was reached (core/disclosure.py) is taken out first: its counts are
    about the quest, not results."""
    t = strip_for_checks(text)
    for pat in _STRIP_BLOCKS:
        t = pat.sub(" ", t)
    t = re.sub(r"\\(?:sim|approx|thicksim|simeq)\b", "~", t)
    t = _LATEX_CMD.sub(r"\1", t)
    t = re.sub(r"\\(?:gt|geq|ge)\b", ">", t)
    t = re.sub(r"\\(?:lt|leq|le)\b", "<", t)
    t = t.replace("\\%", "%").replace("\\,", " ").replace("\\!", "").replace("\\;", " ")
    t = t.replace("$", " ")
    return t


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[*])")


def sentences(text: str) -> list[str]:
    """Sentences of the normalised text. The lookahead keeps ``e.g.`` and a
    decimal point from splitting a sentence in half."""
    out: list[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if block:
            out.extend(s.strip() for s in _SENTENCE_SPLIT.split(block) if s.strip())
    return out


def _decimals(token: str) -> int:
    return len(token.split(".", 1)[1]) if "." in token else 0


_NUM = re.compile(r"(?<![\w.])(-?[0-9]+(?:\.[0-9]+)?)(?![\w])")


def _numbers(text: str) -> Iterator[tuple[float, str, int, bool]]:
    """``(value, token, start, is_percent)`` for each number in ``text``."""
    for m in _NUM.finditer(text):
        token = m.group(1)
        try:
            value = float(token)
        except ValueError:
            continue
        pct = text[m.end(): m.end() + 2].lstrip().startswith("%")
        yield value, token, m.start(), pct


_HEDGE = re.compile(r"~|\babout\b|\bapproximately\b|\broughly\b|\baround\b|\bnearly\b", re.I)


def _close(paper: float, actual: float, token: str, *, hedged: bool) -> bool:
    """Is ``actual`` the number the paper wrote? A hedged number ("~34%") gets a
    relative tolerance; an exact one must survive rounding to its own precision."""
    if hedged:
        denom = max(abs(paper), abs(actual))
        return denom > 0 and abs(paper - actual) / denom <= HEDGED_REL_TOL
    dec = _decimals(token)
    try:
        return round(actual, dec) == round(paper, dec)
    except (ValueError, OverflowError):
        return False


def _ctx(text: str, at: int = 0) -> str:
    lo = max(0, at - _CONTEXT_CHARS // 2)
    return " ".join(text[lo: lo + _CONTEXT_CHARS].split())


# --- 1. an interval labelled as a method that did not produce it ---------------

_METHOD_LABEL = re.compile(
    r"exact binomial|binomial exact|clopper[-\s]?pearson|wilson|bootstrap|"
    r"jeffreys|agresti[-\s]?coull|\bbca\b|profile likelihood|likelihood ratio",
    re.I,
)
_INTERVAL = re.compile(
    r"([0-9]*\.?[0-9]+)\s*(?:[-\u2013\u2014]|to)\s*([0-9]*\.?[0-9]+)"
    r"|\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)"
    r"|\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]"
)


def _intervals_in(text: str) -> Iterator[tuple[float, str, float, str]]:
    for m in _INTERVAL.finditer(text):
        for lo_i, hi_i in ((1, 2), (3, 4), (5, 6)):
            lo_t, hi_t = m.group(lo_i), m.group(hi_i)
            if lo_t and hi_t:
                try:
                    lo, hi = float(lo_t), float(hi_t)
                except ValueError:
                    break
                if lo < hi:
                    yield lo, lo_t, hi, hi_t
                break


def _scopes(text: str, sents: list[str]) -> list[tuple[str, str]]:
    """``(label_text, body_text)`` pairs to look for a method label in.

    A markdown table is one scope: the observed paper put "exact binomial" in
    the header row and the caption, with the intervals in the cells below, so
    a sentence-only search would never see them together. Prose sentences are
    each their own scope — a document-wide search would let one correctly
    labelled binomial interval condemn every unlabelled t-interval elsewhere.
    """
    out: list[tuple[str, str]] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            start = i
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                i += 1
            body = "\n".join(lines[start:i])
            caption = " ".join(lines[i: i + 3])
            out.append((lines[start] + " " + caption, body))
            continue
        i += 1
    out.extend((s, s) for s in sents)
    return out


def _check_interval_method(
    text: str, sents: list[str], intervals: dict[str, dict[str, float]], n_seeds: int,
) -> list[StatFinding]:
    """An interval named for an estimator that did not produce it.

    Proving the label wrong needs no second estimator: it is enough that the
    printed endpoints ARE the t-interval over the seeds, which is a different
    quantity from any binomial interval on the trials within a seed.
    """
    if not intervals or n_seeds < MIN_SEEDS:
        return []
    out: list[StatFinding] = []
    seen: set[tuple[float, float]] = set()
    for label, body in _scopes(text, sents):
        m = _METHOD_LABEL.search(label)
        if not m:
            continue
        matched: list[tuple[str, str, str]] = []
        for lo, lo_t, hi, hi_t in _intervals_in(body):
            if (lo, hi) in seen:
                continue
            for path, stat in intervals.items():
                # Only an interval that IS the seed-level t interval can be one under another name. A pooled Wilson or
                # bootstrap interval (``ci_method``) is the estimator its label says; a paper that prints it under that
                # label was condemned as mislabelled by a real quest until this was told apart.
                if stat.get("ci_method", "t_between_seeds") != "t_between_seeds":
                    continue
                cl, cu = stat.get("ci_lower"), stat.get("ci_upper")
                if cl is None or cu is None:
                    continue
                if _close(lo, cl, lo_t, hedged=False) and _close(hi, cu, hi_t, hedged=False):
                    seen.add((lo, hi))
                    matched.append((lo_t, hi_t, path))
                    break
        if not matched:
            continue
        # One finding per labelled scope, not per cell: a table of six
        # intervals under one wrong header is one mislabel to fix, and six
        # near-identical must-fix bullets would bury the other findings.
        shown = "; ".join(
            f"{lo_t}–{hi_t} is the t-interval for `{path}`"
            for lo_t, hi_t, path in matched[:2]
        )
        more = f", and {len(matched) - 2} more" if len(matched) > 2 else ""
        out.append(StatFinding(
            kind="interval_method_mismatch",
            message=(
                f"{len(matched)} interval(s) labelled “{m.group(0)}” are really "
                f"the Student-t interval over this run's {n_seeds} replicate "
                f"seeds: {shown}{more}. A t-interval across seeds and an "
                f"interval over the trials within a run are different "
                f"quantities — label these as the seed-to-seed interval, or "
                f"compute the named one."
            ),
            context=_ctx(label),
        ))
    return out


# --- 2. a probability described as its own complement -------------------------

_OPPOSITE_EVENT = re.compile(r"outbreak|p_?major|\bmajor\b|epidemic|invasion", re.I)
_EXTINCTION_NAME = re.compile(r"extinct|p_?minor|\bminor\b|die|fade|fizzl", re.I)
_REVERSED = re.compile(
    r"(probabilit\w*|chance|likelihood|risk)"
    rf".{{0,{_SPAN}}}?\b(?:that|of)\b"
    r".{0,60}?\b(?:disease|epidemic|infection|outbreak|pathogen|virus|it|chain)\b"
    r".{0,40}?\b(vanish\w*|dies? out|died out|go(?:es)? extinct|went extinct|"
    r"extinction|extinguish\w*|fizzl\w*|disappear\w*|never takes? off)",
    re.I,
)


def _check_probability_complement(
    sents: list[str], values: dict[str, float],
) -> list[StatFinding]:
    """A probability of extinction that is really the probability of an outbreak.

    Deliberately narrow. The extinction word must be the EVENT the probability
    is about ("the probability that the disease will vanish"), never the verb
    of the probability itself — "the outbreak probability vanished as N
    increased" is a correct sentence about a probability going to zero and
    must not match. The number must also match a result named for the
    opposite event and match none named for extinction, so the complement can
    be stated explicitly.
    """
    if not values:
        return []
    opposite = {
        p: v for p, v in values.items()
        if _OPPOSITE_EVENT.search(p.rsplit(".", 1)[-1]) and 0.0 < v < 1.0
    }
    if not opposite:
        return []
    out: list[StatFinding] = []
    for s in sents:
        m = _REVERSED.search(s)
        if not m:
            continue
        hedged = bool(_HEDGE.search(s))
        for value, token, _at, pct in _numbers(m.group(0)):
            prob = value / 100.0 if pct else value
            if not 0.0 < prob < 1.0:
                continue
            # A run that computed an extinction probability equal to this
            # number means the paper quoted the right quantity.
            if any(
                _EXTINCTION_NAME.search(p.rsplit(".", 1)[-1])
                and _close(prob, v, token, hedged=hedged)
                for p, v in values.items()
            ):
                continue
            # Nearest first: a hedged number ("~34%") can sit within tolerance
            # of several outbreak probabilities, and naming the closest makes
            # the finding something the writer can check against the results.
            hit = next(
                (
                    p for p, v in sorted(
                        opposite.items(), key=lambda kv: abs(kv[1] - prob),
                    )
                    if _close(prob, v, token, hedged=hedged)
                ),
                None,
            )
            if hit is None:
                continue
            out.append(StatFinding(
                kind="probability_complement",
                message=(
                    f"the paper gives {token}{'%' if pct else ''} as the "
                    f"probability that the disease dies out, but that number is "
                    f"`{hit}` = {opposite[hit]:.6g}, which this run computed as "
                    f"the probability of a MAJOR OUTBREAK — the opposite event. "
                    f"The probability of dying out is its complement, "
                    f"{1.0 - opposite[hit]:.6g}."
                ),
                context=_ctx(s),
            ))
            break
    return out


# --- entry point ---------------------------------------------------------------

def check(
    paper_text: str,
    *,
    result_json: Any = None,
    intervals: dict[str, dict[str, float]] | None = None,
    n_seeds: int = 0,
) -> StatReport:
    """Test what the paper says its numbers ARE against what the run computed.

    ``intervals`` are the engine's own replicate aggregates (``_replicate_result_intervals``), passed in rather than
    recomputed here so this module stays free of ``core.engine`` and can be tested without one.
    """
    report = StatReport()
    if not paper_text.strip():
        report.skipped = True
        report.skip_reason = "paper text is empty"
        return report

    intervals = intervals or {}

    values: dict[str, float] = {
        path: value for path, value in flatten_numbers(result_json or {})
    }
    # The mean over the seeds is what a paper reports, so it, not seed 0's
    # value, is what a claim about a quantity should be matched against.
    for path, stat in intervals.items():
        if isinstance(stat.get("mean"), (int, float)):
            values[path] = float(stat["mean"])

    text = normalise(paper_text)
    sents = sentences(text)

    report.checks_run = ["interval_method_mismatch", "probability_complement"]
    findings: list[StatFinding] = []
    findings += _check_interval_method(text, sents, intervals, n_seeds)
    findings += _check_probability_complement(sents, values)

    seen: set[str] = set()
    for f in findings:
        if f.message in seen:
            continue
        seen.add(f.message)
        report.findings.append(f)
    return report
