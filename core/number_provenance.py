"""What a refine's extension added to the results, and whether the paper reports it.

A refine can ask the existing script to compute more (``add n=64``). :func:`result_paths` and :func:`added_results` say what
the results held before and what the extension added; :func:`unreported_results` says which of the added results the paper
does not report. A result is reported when the paper holds a key for it (``{{v:...}}``, filled by FI from its records,
core/paper_values.py), when it prints one of its values (rounded as the paper rounds it), or when one sentence or table says
its name together with a number. This answers "was the person's request answered"; it is not a check that a paper number
came from the run (a number the writer types that is not a number of the plan is found by core/typed_numbers.py, and a
result is a key FI fills in).
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterator

from core.numeric_oracle import flatten_numbers
from core.stat_claims import normalise

_MINUS_SIGN = "−"
_TEXT_NUMBER = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)(?![\w])")
# A number as a paper prints it: optional sign, thousands separators, decimals, an exponent.
_PRINTED = re.compile(r"(?<![\w.])(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)(?:[eE]([-+]?\d+))?(?!\w|\.\d)")
# ``1.21 	imes 10^{-2}`` and ``1.21 x 10⁻²`` are one number, 1.21e-2.
_LATEX_SCI = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|×|·|\\!)\s*10\s*\^\s*\{?\s*(-?\+?\d+)(?:\s*\})?")
_SUPERSCRIPT_SCI = re.compile(r"(\d)\s*([×·])\s*10([⁻⁺]?[⁰¹²³⁴-⁹]+)")
_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


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


def _significant_digits(token: str) -> int:
    """Significant digits in the number as the paper wrote it."""
    t = token.lstrip("-+").replace(",", "")
    if "." in t:
        whole, frac = t.split(".", 1)
        whole = whole.lstrip("0")
        return len(whole) + len(frac) if whole else len(frac.lstrip("0")) or len(frac)
    return len(t.strip("0")) or 1


def _printed_numbers(paper_text: str) -> list[tuple[float, str]]:
    """``(value, token)`` for each number the paper prints with at least two significant digits: the reference list,
    citation brackets, and figure, table, section and equation numbers are left out."""
    text = normalise(paper_text).replace(_MINUS_SIGN, "-")
    text = _SUPERSCRIPT_SCI.sub(lambda m: f"{m.group(1)}{m.group(2)}10^{m.group(3).translate(_SUPERSCRIPT_DIGITS)}", text)
    text = _LATEX_SCI.sub(lambda m: f"{m.group(1)}e{m.group(2).replace('+', '')}", text)
    text = _POINTER.sub(" ", text)
    out: list[tuple[float, str]] = []
    for m in _PRINTED.finditer(text):
        token, exp = m.group(1), m.group(2)
        if _significant_digits(token) < 2:
            continue
        try:
            value = float(token.replace(",", "") + (f"e{exp}" if exp else ""))
        except ValueError:
            continue
        if math.isfinite(value) and value != 0.0:
            out.append((value, token))
    return out


def _numbers_in_text(text: str) -> Iterator[float]:
    """Each number written in a run of text (a U+2212 is read as the minus it is)."""
    for m in _TEXT_NUMBER.finditer(text.replace(_MINUS_SIGN, "-")):
        try:
            yield float(m.group(1).replace(",", ""))
        except ValueError:
            continue


def _within(value: float, values: list[float], tol: float) -> bool:
    for v in values:
        denom = max(abs(value), abs(v))
        if denom == 0.0 or abs(value - v) / denom <= tol:
            return True
    return False


def _prints_new_value(paper_text: str, new: list[float], old: list[float]) -> bool:
    """Does the paper print one of ``new`` (at its own precision) that is not also one of ``old``? A value the results
    already held under another name says nothing about whether the new quantity is reported."""
    for value, token in _printed_numbers(paper_text):
        tol = 0.51 * 10.0 ** -(max(_significant_digits(token), 1) - 1)
        if _within(value, new, tol) and not _within(value, old, tol):
            return True
    # A value below the smallest magnitude read, written as it is stored. Not a whole number: "3" is also a section, a
    # figure or a citation. The reference list and citation brackets are removed first.
    text = normalise(paper_text)
    for v in new:
        if v == 0.0 or float(v).is_integer() or _within(v, old, 1e-9):
            continue
        for form in {f"{v:g}", f"{v:.3g}", f"{v:.4g}"}:
            if re.search(r"(?<![\w.])" + re.escape(form) + r"(?![\w]|\.\d)", text):
                return True
    return False


def _keyed_names(paper_text: str) -> list[list[str]]:
    """The words of the quantity each key of the paper names (``{{v:Ea_rel_bias:sd @ ...}}`` -> ``["rel", "bias"]``)."""
    from . import paper_values as _values

    out: list[list[str]] = []
    for key in _values.keys_in(paper_text):
        try:
            metric = _values.parse(key).metric
        except Exception:  # noqa: BLE001 -- a key FI cannot read names nothing
            continue
        words = _name_words(str(metric))
        if words:
            out.append(words)
    return out


def unreported_results(paper_text: str, result_json: Any, before: Any, asked: list[str]) -> list[str]:
    """The results a refine's extension added, for the numbers a person asked for, that the paper does not report.

    What the extension added is :func:`added_results`. The new results whose names share a word with the request are
    the ones asked for, and each of them must be in the paper; when none does, one new result in the paper is enough.
    A result is in the paper when a key of the paper names it (FI fills the key from its records), when the paper
    prints one of its values (rounded as the paper rounds it) that no old result also has, or, for a new name, when one
    sentence or table says the words that set its name apart from the nearest old name together with a number and without
    a "not" ("the maximum absolute error ... equals the error at t=1" for ``max_abs_errors`` beside ``errors``). The name
    matters when the values are old ones again: a maximum error that equals the error at the end point is reported only by
    saying so. Numbers appended to a list are reported only by a key or by printing one of them.

    No model call; ``[]`` when the extension added nothing new (then there is nothing to look for)."""
    groups, grown, old = added_results(result_json, before)
    if not groups:
        return []
    # A number the request itself names ("add n=64") is the setting asked about, not a result.
    named = [v for a in asked for v in _numbers_in_text(a)]
    groups = {g: [v for v in vals if not _within(v, named, 1e-12)] for g, vals in groups.items()}
    groups = {g: vals for g, vals in groups.items() if vals or g not in grown}
    if not groups:
        return []
    asked_words = [w for a in asked for w in _name_words(a)]
    related = [g for g in groups
               if any(w.startswith(a) or a.startswith(w) for w in _name_words(g) for a in asked_words)]
    units = _units(paper_text)
    keyed = _keyed_names(paper_text)
    old_names = [set(_name_words(p)) for p in (before or [])]

    def distinct(words: list[str]) -> list[str]:
        """The words that set a new result apart from the old one nearest to it: ``euler_max_abs_errors`` beside
        ``euler_errors`` is ``max``, ``abs``. ``euler`` and ``error`` say which case and what kind, and the paper says
        those already; only saying the new words reports the new result. Nearest: the most words in common, then the
        shortest name."""
        near = max(old_names, key=lambda n: (len(n & set(words)), -len(n)), default=set())
        return [w for w in words if w not in near] or words

    def reported(group: str) -> bool:
        words = _name_words(group)
        if words and any(set(words) <= set(k) for k in keyed):
            return True
        if _prints_new_value(paper_text, groups[group], old):
            return True
        if group in grown:
            return False
        return any(_names_it(distinct(words), u) for u in units)

    if related:
        return [g for g in related if not reported(g)]
    return [] if any(reported(g) for g in groups) else sorted(groups)
