"""Rule R1: a number the writer types is a number of the plan, a number of the topic, or a label.

FI fills every result number a paper states from its own records (:mod:`core.paper_values`), so the writer types none. This
module is the other half: it reads the writer's own text (never the values FI filled in, and never the sources) and asks of
each number only one question, is it

* a number of the frozen plan (a numeric field of ``needs/FROZEN_PROTOCOL.json``, or a number in one of its formulas) or of the
  topic, written exactly as the plan writes it (a percent may be written as the plan's fraction times 100), or
* a structural label (``Figure 2``, ``Section 3``, ``Table 1``, ``[4]``, a heading's or list's number), or
* a whole number from 0 to 10 inside math (an exponent, an index, the 2 in ``sqrt(2(n-1))``)?

Anything else is a number the writer typed that FI did not fill, and is reported. There is no tolerance, no "near a record
value" and no pool of candidates anywhere here: a typed result is wrong because the writer typed it, whether or not it
happens to be right. A number in a sentence that cites a source is left to the claim check, which reads the source.

The numbers of the plan are read from the plan as it is frozen, never from prose (``plan.md``) and never from a run fact the
paper gets as a placeholder (the runs per setting and the seeds): those are FI's, filled by ``{{trials_per_setting}}`` and
``{{seed}}``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

REMOVED = "[number removed]"
#: The most sentences one report names; the rest are counted in one more line.
MAX_SENTENCES = 30
#: Plan fields whose numbers are run facts: the paper gets them through placeholders, never typed.
_RUN_FACT_FIELDS = re.compile(r"runs_per_setting|min_runs_per_setting|seed")
_FORMULA_PATH = re.compile(r"(formula|expected_formula)$")
_PLAN_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?:[eE]-?\d+)?")

# --- what is FI's or not the writer's: blanked out, length kept, before any number is read --------------------------------

_FI_V = re.compile(r"<!-- fi:v \{\{v:.*?\}\} -->.*?<!-- /fi:v -->", re.S)
_FI_PAIR = re.compile(r"<!-- fi:([\w-]+)(?::[0-9a-f]+)? -->.*?<!-- /fi:\1 -->", re.S)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_PLACEHOLDER = re.compile(r"\{\{[^{}\n]{1,320}\}\}")
_FENCE = re.compile(r"^[ \t]*(?P<f>```|~~~)[^\n]*\n.*?^[ \t]*(?P=f)[^\n]*$|^[ \t]*(?:```|~~~).*\Z", re.M | re.S)
_TAIL = re.compile(r"^#{1,6}[ \t]+(?:\d+\.?[ \t]*)?(references|further reading|bibliography|works cited)\b.*\Z",
                   re.I | re.M | re.S)
_IMG_TARGET = re.compile(r"\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_CODE = re.compile(r"`[^`\n]+`")
_HEADING_NUMBER = re.compile(r"^(#{1,6}[ \t]+)(?:\d+(?:\.\d+)*\.?[ \t]|[IVXLC]+\.[ \t])", re.M)
_LIST_NUMBER = re.compile(r"^([ \t]*(?:[-*+][ \t]+)?)\d+[.)][ \t]", re.M)
MATH = re.compile(r"\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)|(?<![\\$])\$(?=\S)[^$\n]+?(?<=\S)\$(?!\d)", re.S)
#: A label word, then its number: ``Figure 2``, ``Section 3.1``, ``Step 4``; for a whole number also a list of them after
#: the label (``Figures 2 and 3``, ``Tables 1, 2``). A decimal after a list is a value, not a label.
_LABEL_WORDS = (r"(?:Figures?|Figs?\.|Tables?|Eqs?\.|Equations?|Sections?|§§?|Steps?|Stages?|Panels?|Appendix|Appendices|"
                r"Supplement\w*)[ \t\u00a0]*\(?")
_LABEL = re.compile(_LABEL_WORDS + r"$")
_LABEL_LIST = re.compile(_LABEL_WORDS + r"(?:\d+[a-z]?[ \t]*(?:,|and|&|or|to|\u2013|-)[ \t]*)+$")
_CITE = re.compile(r"\[\s*W?\d+(?:\s*[–-]\s*\d+)?(?:\s*[,;]\s*W?\d+(?:\s*[–-]\s*\d+)?)*\s*\]")


_TAGS = re.compile(r"<!-- (?:/fi:[\w-]+|fi:v \{\{v:.*?\}\}|fi:fact-[\w-]+) -->")


def _plain(sentence: str) -> str:
    """A sentence as a reader sees it: FI's value markers taken out, the values kept."""
    return _TAGS.sub("", sentence)


def _blank(match: re.Match[str]) -> str:
    return re.sub(r"[^\n]", " ", match.group(0))


def writers_text(markdown: str) -> str:
    """``markdown`` with everything that is not the writer's own words blanked out (same length, so offsets hold): FI's value
    markers and blocks, other HTML comments, double-brace placeholders, code, URLs, image targets, a heading's or list's
    number, and the sources from ``References`` / ``Further reading`` on."""
    t = markdown or ""
    for pattern in (_FI_V, _FI_PAIR, _COMMENT, _FENCE, _TAIL, _PLACEHOLDER, _IMG_TARGET, _URL, _CODE):
        t = pattern.sub(_blank, t)
    t = _HEADING_NUMBER.sub(lambda m: m.group(1) + re.sub(r"[^\n]", " ", m.group(0)[len(m.group(1)):]), t)
    return _LIST_NUMBER.sub(lambda m: m.group(1) + " " * (len(m.group(0)) - len(m.group(1))), t)


# --- numbers ---------------------------------------------------------------------------------------------------------


def _normalise(t: str) -> str:
    """LaTeX spellings of a number, with length kept: ``2{,}000``, ``50\\,000`` and a subscript digit (notation)."""
    t = t.replace("{,}", ",  ")
    t = re.sub(r"(\d),  (\d{3})", lambda m: m.group(1) + "," + m.group(2) + "  ", t)
    t = re.sub(r"(\d)\\[,;:!]", lambda m: m.group(1) + "  ", t)
    t = re.sub(r"_\{[^}]*\}", lambda m: " " * len(m.group(0)), t)
    t = re.sub(r"_[0-9A-Za-z]", "  ", t)
    return re.sub(r"[₀-₉]+", lambda m: " " * len(m.group(0)), t)


_SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"
_NUM = re.compile(
    r"(?P<sign>[−\-+±]\s?)?"
    r"(?P<int>\d{1,3}(?:,\d{3})+|\d{1,3}(?:[   ]{1,2}\d{3})+(?!\d)|\d+)"
    r"(?P<frac>\.\d+)?"
    r"(?P<exp>(?:[eE][−\-+]?\d+)|(?:\s?(?:×|\\times|x)\s?10(?:\^\{?[−\-+]?\d+\}?|[" + _SUP + r"]+)))?"
    r"(?P<pct>\s?%|\s?percent(?:age points?)?|\s?pp\b)?"
)


def _exponent(text: str) -> int:
    if not text:
        return 0
    m = re.search(r"[eE]([−\-+]?\d+)", text) or re.search(r"\^\{?([−\-+]?\d+)\}?", text)
    if m:
        return int(m.group(1).replace("−", "-"))
    sup = re.search("[" + _SUP + "]+", text)
    return int(sup.group(0).translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-"))) if sup else 0


@dataclass(frozen=True)
class Typed:
    """One number as the writer typed it."""

    token: str
    start: int
    end: int
    value: Decimal
    pct: bool
    integer: bool  # written without decimals or an exponent
    in_math: bool
    structural: bool
    sentence: str
    cited: bool  # the sentence carries a citation label


def _sentence(text: str, start: int, end: int) -> tuple[int, int]:
    a = max(text.rfind(". ", 0, start), text.rfind("\n", 0, start))
    ends = [x for x in (text.find(". ", end), text.find("\n", end)) if x != -1]
    return a + 1, (min(ends) + 1 if ends else len(text))


def sentence_around(markdown: str, start: int, end: int) -> str:
    """The sentence of ``markdown`` that holds ``start:end``, on one line, with FI's value markers taken out."""
    a, b = _sentence(markdown, start, end)
    return _plain(re.sub(r"\s+", " ", markdown[a:b]).strip())


def numbers(markdown: str) -> list[Typed]:
    """Every number in the writer's own words, with where it is and whether it is a label or inside math."""
    text = writers_text(markdown)
    t = _normalise(text)
    maths = [m.span() for m in MATH.finditer(markdown or "")]
    cites = [m.span() for m in _CITE.finditer(text)]
    out: list[Typed] = []
    for m in _NUM.finditer(t):
        s, e = m.span()
        before = t[max(0, s - 1):s]
        if before and (before.isalpha() or before in "_\\^" or before in _SUP):
            continue  # glued to a word or a symbol: m2, z196, fig1, x^2
        digits = re.sub(r"[ ,  ]", "", m.group("int")) + (m.group("frac") or "")
        sign = (m.group("sign") or "").strip()
        negative = sign in ("−", "-")
        if negative and re.search(r"\d\s?$", t[max(0, s - 3):s]):
            negative = False  # a dash between two numbers is a range, not a sign
        exp = _exponent(m.group("exp") or "")
        try:
            value = Decimal(digits).scaleb(exp)
        except InvalidOperation:
            continue
        if negative:
            value = -value
        a, b = _sentence(text, s, e)
        sentence = re.sub(r"\s+", " ", (markdown or "")[a:b]).strip()
        near = text[max(0, s - 24):s]
        label = bool(_LABEL.search(near)) or (digits.isdigit() and bool(_LABEL_LIST.search(near)))
        out.append(Typed(
            token=(markdown or "")[s:e].strip(), start=s, end=e, value=value, pct=bool(m.group("pct")),
            integer=not m.group("frac") and exp == 0 and not m.group("pct"),
            in_math=any(x <= s < y for x, y in maths),
            structural=label or any(x <= s < y for x, y in cites),
            sentence=_plain(sentence), cited=bool(_CITE.search(text[a:b]))))
    return out


# --- the plan's numbers ----------------------------------------------------------------------------------------------


def _decimal(x: Any) -> Decimal | None:
    if isinstance(x, bool) or x is None:
        return None
    try:
        return Decimal(str(x)) if isinstance(x, (int, float)) else Decimal(x)
    except (InvalidOperation, ValueError, TypeError):
        return None


def plan_numbers(protocol: dict[str, Any] | None, topic: str = "") -> frozenset[Decimal]:
    """The numbers the writer may type as the plan writes them: every numeric field of the frozen protocol (not the run
    facts), every number in one of its formulas, and every number of the topic. The plan's ``constants`` are among its
    fields, which is how a constant the method uses (1.96, the gas constant) is allowed."""
    found: set[Decimal] = set()

    def take(x: Any) -> None:
        d = _decimal(x)
        if d is not None and d.is_finite():
            found.add(d)

    def walk(o: Any, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}/{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            if not _RUN_FACT_FIELDS.search(path):
                take(o)
        elif isinstance(o, str) and _FORMULA_PATH.search(path):
            for m in _PLAN_NUMBER.finditer(o):
                take(m.group(0))

    walk(protocol if isinstance(protocol, dict) else {}, "protocol")
    for m in _PLAN_NUMBER.finditer(topic or ""):
        take(m.group(0))
    return frozenset(found)


def allowed(n: Typed, plan: frozenset[Decimal]) -> bool:
    """Rule R1 for one typed number."""
    if n.structural:
        return True
    if n.value in plan or (n.pct and n.value / 100 in plan):
        return True
    return n.in_math and n.integer and abs(n.value) <= 10


@dataclass(frozen=True)
class Report:
    """The typed numbers R1 does not allow, by sentence, and the ones it leaves to the claim check."""

    sentences: tuple[tuple[str, tuple[str, ...]], ...]  # (sentence, its numbers as written)
    left_to_claim_check: int = 0

    @property
    def count(self) -> int:
        return sum(len(tokens) for _s, tokens in self.sentences)


def typed_outside_the_plan(markdown: str, plan: frozenset[Decimal]) -> list[Typed]:
    """The numbers of ``markdown`` that R1 does not allow and the claim check does not own (a cited sentence's)."""
    return [n for n in numbers(markdown) if not allowed(n, plan) and not n.cited]


def report(markdown: str, plan: frozenset[Decimal]) -> Report:
    bad = typed_outside_the_plan(markdown, plan)
    by: dict[str, list[str]] = {}
    for n in bad:
        by.setdefault(n.sentence, []).append(n.token)
    cited = sum(1 for n in numbers(markdown) if n.cited and not n.structural and not allowed(n, plan))
    return Report(tuple((s, tuple(dict.fromkeys(t))) for s, t in by.items()), cited)


def hits(markdown: str, plan: frozenset[Decimal], name: str = "paper_record_mismatch") -> list[str]:
    """One must-fix hit per sentence that holds a number R1 does not allow, shaped ``name: message — “sentence”`` (the shape
    the passage editor locates, :mod:`core.paper_patch`). ``markdown`` is the paper with FI's values as the writer's keys
    (:func:`core.paper_facts.to_placeholders`), so the sentence quoted is the writer's own text."""
    rep = report(markdown, plan)
    out: list[str] = []
    for sentence, tokens in rep.sentences[:MAX_SENTENCES]:
        many = len(tokens) > 1
        shown = " and ".join(tokens) if len(tokens) <= 3 else ", ".join(tokens[:3]) + f" and {len(tokens) - 3} more"
        out.append(
            f"{name}: you typed {shown}, which {'are not numbers' if many else 'is not a number'} of the plan or the topic. "
            "If it is a result, place its key from the \"Results you can place\" list (FI fills it in); if no key gives it, "
            "leave it out or say it in words. A number of the method is written exactly as the plan writes it. "
            f"— “{sentence[:400]}”")
    rest = sum(len(tokens) for _s, tokens in rep.sentences[MAX_SENTENCES:])
    if rest:
        out.append(f"{name}: a further {rest} typed number(s) in other sentences are not numbers of the plan or the topic.")
    return out


def remove(markdown: str, plan: frozenset[Decimal]) -> tuple[str, list[tuple[str, str]]]:
    """``markdown`` with each number R1 does not allow replaced by :data:`REMOVED`, and ``(number, sentence)`` for each one
    taken out (the sentence as it read before)."""
    bad = typed_outside_the_plan(markdown, plan)
    taken = [(n.token, n.sentence) for n in bad]
    text = markdown or ""
    for n in sorted(bad, key=lambda x: -x.start):
        text = text[:n.start] + REMOVED + text[n.end:]
    return text, taken
