"""Deterministic check that a number is the QUANTITY the paper says it is.

Why this exists
===============
Two checks already stand between a run and its paper, and neither asks this
question. The **numeric oracle** (``core/numeric_oracle.py``) asks whether a
number in the prose matches a number the run computed — arithmetic, but blind
to what the number *means*. The **claim check** asks a model whether a cited
sentence is supported by evidence — meaning, but with no arithmetic anywhere.

Between them sits a gap that shipped in five delivered papers:

* a paper printed the Bonferroni *threshold* ``0.001388`` as if it were a
  computed **p-value**, and claimed "Cohen's d > 16 for all pairwise
  comparisons" when 17 of the run's 36 pairwise effect sizes were below it;
* another claimed "Cohen's d > 39" with 6 of 12 below;
* a paper labelled its three-seed **t-intervals** "exact binomial 95% CI";
* a paper printed a figure's **y-axis limits** (231/240/280, which matplotlib
  pads 5% above the tallest bar) as the **modal bin counts** (220/229/267);
* a paper called the **major-outbreak probability** (~34%) the probability the
  disease would "simply vanish" — the extinction probability is its
  complement (~66%). The numeric oracle matched the number and could not see
  the reversed meaning.

Every one of those numbers is either correct-but-mislabelled or contradicted
by a quantity the run *did* compute. So this module reads what a number is
**said to be** and tests that claim against the run.

The design rule: say nothing unless the contradiction can be NAMED
=================================================================
False positives are the failure mode that matters here. FI has already paid
for two checks that cried wolf — the bounds gate calling a correct ``0``
"trivial", and the claim check's budget — and a spurious flag costs a whole
rewrite. So every check below is a **recomputation**, not a pattern match:
each one fires only when it can point at the run's own numbers and say what
the paper called them instead. A claim this module cannot recompute is a
claim it stays silent about.

Five checks, each independently gated
-------------------------------------
``effect_size_overstated``
    A universally-quantified effect-size claim ("Cohen's d > 16 for *all*
    pairwise comparisons") against every pairwise Cohen's d the run's
    replicates actually support. Needs the quantifier: a single "d = 16.2" is
    one number and the numeric oracle's business.

``threshold_as_pvalue``
    A number presented as a p-value (``p < x``) that equals this run's
    Bonferroni-corrected threshold ``0.05/n``, when the run computed no
    p-value at all. Matched to one unit in the paper's own last decimal,
    because ``0.05/36 = 0.0013888…`` is written ``0.001388`` by truncation
    and would *miss* a rounding comparison.

``interval_method_mismatch``
    An interval labelled with a named method (exact binomial, Clopper-Pearson,
    Wilson, bootstrap, …) whose endpoints reproduce FI's own t-interval over
    the replicate seeds. The contradiction is not "this interval is wrong" —
    it is "this interval IS the k-seed t-interval, under another name", so no
    second interval estimator has to be implemented to prove it.

``axis_limit_as_count``
    A number presented as a count of runs that equals a recorded figure's
    y-axis upper limit and appears nowhere in the results. An axis limit is a
    property of the drawing, not a measurement.

``probability_complement``
    A sentence giving the probability that the disease/outbreak dies out,
    whose number matches a result named for the *opposite* event (a major
    outbreak) and matches no result named for extinction.

All five are recorded through ``StatReport`` with the contradiction spelled
out, so the paper's writer is told which number to relabel rather than "a
statistic is wrong somewhere".
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

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

# A number has to carry at least this many significant digits before equalling
# ``0.05/n`` means anything. The conventional cutoffs (0.05, 0.01, 0.001) carry
# one, and a paper writing "p < 0.01" is quoting a convention, not this run's
# correction — matching those would flag ordinary, correct sentences.
MIN_THRESHOLD_SIG_DIGITS = 3


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
    ``$…$``; leaving the delimiters in place made neither of them parse."""
    t = text
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


def _significant_digits(token: str) -> int:
    """Significant digits in the number as the paper wrote it."""
    t = token.lstrip("-+").replace(",", "")
    if "." in t:
        whole, frac = t.split(".", 1)
        whole = whole.lstrip("0")
        return len(whole) + len(frac) if whole else len(frac.lstrip("0")) or len(frac)
    return len(t.strip("0")) or 1


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


# --- 1. effect size claimed for ALL comparisons -------------------------------

_COHEN_CLAIM = re.compile(
    r"\bd\b\s*(>=|<=|>|<|\u2265|\u2264)\s*([0-9]+(?:\.[0-9]+)?)"
)
_EFFECT_WORD = re.compile(r"cohen|effect size", re.I)
_UNIVERSAL = re.compile(r"\ball\b|\bevery\b|\beach\b", re.I)
_PAIRWISE = re.compile(r"pairwise|comparison|contrast|\bpairs?\b", re.I)


def _check_effect_sizes(
    sents: list[str], comparison_stats: dict[str, Any],
) -> list[StatFinding]:
    """"Cohen's d > 16 for all pairwise comparisons" against every pairwise d
    the replicates support.

    The universal quantifier is required. Without it the sentence quotes one
    effect size, which is a single number and already the numeric oracle's
    job; with it the sentence makes a statement about the whole comparison
    set, and that set is something this run computed in full.
    """
    effects = [
        e for e in (comparison_stats.get("effect_sizes") or [])
        if isinstance(e, dict) and isinstance(e.get("cohens_d"), (int, float))
    ]
    if not effects:
        return []
    out: list[StatFinding] = []
    for s in sents:
        if not (_EFFECT_WORD.search(s) and _UNIVERSAL.search(s) and _PAIRWISE.search(s)):
            continue
        m = _COHEN_CLAIM.search(s)
        if not m:
            continue
        op, token = m.group(1), m.group(2)
        thr, dec = float(token), _decimals(token)
        lower_bound = op in (">", ">=", "\u2265")
        # Rounded to the paper's own precision, so a computed 15.94 does not
        # contradict "d > 16": only a shortfall the paper's own rounding
        # cannot explain counts.
        if lower_bound:
            bad = [e for e in effects if round(abs(e["cohens_d"]), dec) < thr]
        else:
            bad = [e for e in effects if round(abs(e["cohens_d"]), dec) > thr]
        if not bad:
            continue
        bad.sort(key=lambda e: abs(e["cohens_d"]), reverse=not lower_bound)
        worst = ", ".join(
            f"|d|={abs(e['cohens_d']):.2f} ({e.get('factor')}/{e.get('metric')}: "
            f"{e.get('a')} vs {e.get('b')})"
            for e in bad[:2]
        )
        side = "below" if lower_bound else "above"
        out.append(StatFinding(
            kind="effect_size_overstated",
            message=(
                f"the paper claims Cohen's d {op} {token} for all pairwise "
                f"comparisons, but {len(bad)} of {len(effects)} pairwise effect "
                f"sizes this run computed are {side} it — smallest: {worst}. "
                f"Report the range of d, or restrict the claim to the "
                f"comparisons that meet it."
            ),
            context=_ctx(s),
        ))
    return out


# --- 2. a correction threshold printed as a p-value ----------------------------

_PVALUE = re.compile(
    r"\bp\b\s*(?:-?\s*value\s*)?(<=|<|=|\u2264)\s*([0-9]*\.?[0-9]+)", re.I
)
# When the number is introduced AS a threshold the paper is right to print it;
# only a number asserted as the observed statistic is the defect.
_NAMED_AS_THRESHOLD = re.compile(
    r"(?:threshold|alpha|significance level|criterion|cut-?off|corrected level)"
    r"[^.]{0,24}$",
    re.I,
)
_PVALUE_PATH = re.compile(r"p_?values?$|^p_?values?|pvalue", re.I)


def _check_pvalue_threshold(
    sents: list[str], comparison_stats: dict[str, Any], values: dict[str, float],
) -> list[StatFinding]:
    """A p-value that is really this run's Bonferroni threshold.

    ``0.05/36 = 0.0013888…``; the observed paper wrote ``0.001388``, a
    truncation. ``round(0.00138888, 6) = 0.001389`` so a rounding comparison
    misses it — the match is therefore one unit in the paper's last decimal.
    Gated on the run having computed no p-value of its own: if it had, an
    equal number could be a coincidence rather than a mislabel.
    """
    comps = comparison_stats.get("comparisons") or {}
    n = comps.get("n")
    alpha = comps.get("bonferroni_alpha")
    if not isinstance(n, int) or n < 2 or not isinstance(alpha, (int, float)):
        return []
    if any(_PVALUE_PATH.search(p.rsplit(".", 1)[-1]) for p in values):
        return []
    out: list[StatFinding] = []
    for s in sents:
        for m in _PVALUE.finditer(s):
            token = m.group(2)
            try:
                value = float(token)
            except ValueError:
                continue
            if _NAMED_AS_THRESHOLD.search(s[: m.start()]):
                continue
            if _significant_digits(token) < MIN_THRESHOLD_SIG_DIGITS:
                continue
            # The number must BE the threshold at the precision the paper
            # wrote it, under rounding OR truncation — 0.05/36 = 0.0013888…
            # was written 0.001388, which a rounding comparison alone misses.
            # Compared at integer scale so no float tolerance is involved: a
            # tolerance that scales with the paper's precision let "p < 0.01"
            # match a threshold of 0.00139.
            scale = 10.0 ** _decimals(token)
            scaled, want = float(alpha) * scale, round(value * scale)
            if round(scaled) != want and math.floor(scaled) != want:
                continue
            out.append(StatFinding(
                kind="threshold_as_pvalue",
                message=(
                    f"the paper reports p {m.group(1)} {token} as a p-value, but "
                    f"{token} is this run's Bonferroni-corrected significance "
                    f"threshold (0.05/{n} = {float(alpha):.8g} for its {n} "
                    f"pairwise comparisons) and no p-value was computed anywhere "
                    f"in the results. A significance threshold is not evidence "
                    f"of significance — either report the p-values the test "
                    f"produced, or say this is the corrected threshold."
                ),
                context=_ctx(s),
            ))
    return out


# --- 3. an interval labelled as a method that did not produce it ---------------

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


# --- 4. a figure's axis limit printed as a count ------------------------------

_COUNT_NOUN = r"runs?|realizations?|realisations?|observations?|samples?|simulations?|trials?"
_COUNT_SENTENCE = re.compile(rf"\b(?:{_COUNT_NOUN})\b", re.I)
_COUNT_DENOM = re.compile(rf"^\s*(?:of\s+)?(?:{_COUNT_NOUN})\b", re.I)

# How close a number must sit to the noun it counts before the paper can be
# said to present it AS that count. The real mislabels sat 10-24 characters
# from "runs"; a citation index in a long methods sentence that happens to
# mention runs sat 161 away.
COUNT_PROXIMITY = 40


def _axis_limits(figure_records: dict[str, Any]) -> list[tuple[float, str, str]]:
    """``(upper_limit, figure, panel)`` for every recorded panel."""
    out: list[tuple[float, str, str]] = []
    for name, rec in (figure_records or {}).items():
        if not isinstance(rec, dict):
            continue
        for ax in rec.get("axes") or []:
            if not isinstance(ax, dict):
                continue
            ylim = ax.get("ylim")
            if isinstance(ylim, list) and len(ylim) == 2:
                try:
                    out.append((float(ylim[1]), str(name), str(ax.get("title") or "")))
                except (TypeError, ValueError):
                    continue
    return out


def _check_axis_limit_counts(
    sents: list[str], figure_records: dict[str, Any], values: dict[str, float],
    series_values: set[float],
) -> list[StatFinding]:
    """A count of runs that is really a y-axis limit.

    matplotlib leaves headroom above the tallest bar (5% by default), so an
    axis limit sits just above a real bin count and reads like one: the
    observed paper printed 231/240/280 where the bins held 220/229/267, and
    87 where the bin held 83. The number is only reported when it appears
    nowhere in the results and matches no plotted series value — an axis
    limit is a property of the drawing, never a measurement.
    """
    limits = _axis_limits(figure_records)
    if not limits:
        return []
    out: list[StatFinding] = []
    for s in sents:
        at_nouns = [m.start() for m in _COUNT_SENTENCE.finditer(s)]
        if not at_nouns:
            continue
        for value, token, at, pct in _numbers(s):
            if pct or "." in token or value <= 1:
                continue
            # The number must sit NEXT TO the thing it counts. A number
            # elsewhere in a long methods sentence is not "a count of runs"
            # merely because the sentence mentions runs somewhere.
            if min(abs(p - at) for p in at_nouns) > COUNT_PROXIMITY:
                continue
            # "300 runs" is the denominator the counts are out of, not a count.
            if _COUNT_DENOM.match(s[at + len(token):]):
                continue
            if any(_close(value, v, token, hedged=False) for v in values.values()):
                continue
            if any(_close(value, v, token, hedged=False) for v in series_values):
                continue
            hit = next(
                (lim for lim in limits if _close(value, lim[0], token, hedged=False)), None,
            )
            if hit is None:
                continue
            limit, fig, panel = hit
            out.append(StatFinding(
                kind="axis_limit_as_count",
                message=(
                    f"the paper reports {token} as a count of runs, but {token} "
                    f"is the y-axis upper limit ({limit:.6g}) of "
                    f"{'panel “' + panel + '” of ' if panel else ''}`{fig}`, and "
                    f"no result this run computed equals it. An axis limit is "
                    f"padding above the tallest bar, not the bar's height — read "
                    f"the count off the data, not the axis."
                ),
                context=_ctx(s),
            ))
    return out


# --- 5. a probability described as its own complement -------------------------

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

def _series_extremes(figure_records: dict[str, Any]) -> set[float]:
    """Every min/max a recorded figure's series reaches, so a count that really
    is plotted data is never mistaken for an axis limit."""
    out: set[float] = set()
    for rec in (figure_records or {}).values():
        if not isinstance(rec, dict):
            continue
        for ax in rec.get("axes") or []:
            if not isinstance(ax, dict):
                continue
            for series in ax.get("series") or []:
                if not isinstance(series, dict):
                    continue
                for key in ("min", "max"):
                    v = series.get(key)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        out.add(float(v))
    return out


def check(
    paper_text: str,
    *,
    result_json: Any = None,
    intervals: dict[str, dict[str, float]] | None = None,
    comparison_stats: dict[str, Any] | None = None,
    figure_records: dict[str, Any] | None = None,
    n_seeds: int = 0,
) -> StatReport:
    """Test what the paper says its numbers ARE against what the run computed.

    ``intervals`` and ``comparison_stats`` are the engine's own replicate
    aggregates (``_replicate_result_intervals`` and ``_result_comparison_stats``)
    — passed in rather than recomputed here so this module stays free of
    ``core.engine`` and can be tested without one.
    """
    report = StatReport()
    if not paper_text.strip():
        report.skipped = True
        report.skip_reason = "paper text is empty"
        return report

    comparison_stats = comparison_stats or {}
    intervals = intervals or {}
    figure_records = figure_records or {}

    values: dict[str, float] = {
        path: value for path, value in flatten_numbers(result_json or {})
    }
    # The mean over the seeds is what a paper reports, so it — not seed 0's
    # value — is what a claim about a quantity should be matched against.
    for path, stat in intervals.items():
        if isinstance(stat.get("mean"), (int, float)):
            values[path] = float(stat["mean"])

    text = normalise(paper_text)
    sents = sentences(text)

    report.checks_run = [
        "effect_size_overstated", "threshold_as_pvalue",
        "interval_method_mismatch", "axis_limit_as_count",
        "probability_complement",
    ]
    findings: list[StatFinding] = []
    findings += _check_effect_sizes(sents, comparison_stats)
    findings += _check_pvalue_threshold(sents, comparison_stats, values)
    findings += _check_interval_method(text, sents, intervals, n_seeds)
    findings += _check_axis_limit_counts(
        sents, figure_records, values, _series_extremes(figure_records),
    )
    findings += _check_probability_complement(sents, values)

    seen: set[str] = set()
    for f in findings:
        if f.message in seen:
            continue
        seen.add(f.message)
        report.findings.append(f)
    return report
