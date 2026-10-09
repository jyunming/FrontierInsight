"""Every citation in the final paper is backed by the retrieved text of the source it names, or it is fixed.

The claim check (``Engine._node_claim_check``) asks one model which claims the paper makes and grounds each in ONE
source; a bracket naming several sources was judged through one of them, a citing sentence the model did not list was
not judged at all, and a sentence it found unsupported only started another rewrite that the iteration budget could
end. This module is the check that runs last, on the text that will be delivered, and it cannot be exhausted:

1. **Every (sentence, source) pair** the text cites is listed from the text itself (:func:`find_sites`), including each
   number of ``[2, 5]`` and each ``[W3]``.
2. **Rules no model can loosen** (:func:`pre_problem`, :func:`post_problem`) only ever add "unsupported": a retracted
   source or one with no retrieved text, a number the sentence states that is in neither the source nor the run's own
   results, a source whose retrieved text is only a snippet, a contents list or a title (it supports "X studied Y", not
   a number, a method or a finding), and a quote that is not in the source's own text.
3. **A reader judges the rest** (``agents/citation_check.md``): is there a passage of the retrieved text that says what
   the sentence takes from the source? The reader is the step a person routes with ``provider.node_providers.citation_check``
   and, while that is not set, the plan's checks' reader (``core/review_pairs.py``).
4. **What is unsupported is fixed, by FI, never by a person** (``agents/citation_repair.md``): cite another retrieved
   source that does say it (its quote is looked up in that source), reword the sentence to what the source does say,
   drop the citation (the sentence rests on the paper's own result), or drop the sentence. A fix is read once more by
   the reader before it stands; anything the reply leaves unresolved or the reader does not confirm is settled by a rule
   that needs no model (:func:`fallback`). A citation is never kept unsupported and the paper is never stopped for this.

The orchestrator :func:`check_paper` is pure over its callbacks (the two model calls, the saving of the record, the
quote test), so it is tested with scripted replies. Every model call is counted in the record BEFORE it is made, so a
resume never asks again for one it already spent, and each kind of call has a cap.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import paper_patch as _pp
from .knowledge import _content_stems
from .passages import rank_by_relevance

#: Where the engine keeps the record of this check (under ``.fi/``), the paper's ledger of it, and its receipt.
FILE = "citation_check.json"
LEDGER_JSON = "paper/citations.json"
LEDGER_MD = "paper/CITATIONS.md"

#: Model calls, per distinct final paper: the reader's judgement (the second is for the pairs the first left unanswered),
#: the repair, and the reader's second look at the fixes it made. And the most in a quest, over every paper it checks.
MAX_JUDGE_CALLS = 2
MAX_REPAIR_CALLS = 1
MAX_VERIFY_CALLS = 1
MAX_CALLS_IN_QUEST = 24
#: Pairs the reader judges in one call. A paper with more is read in several calls, so a normal paper is covered whole
#: within the cap: the cap for a paper is one call per batch plus ``MAX_JUDGE_CALLS - 1`` for the pairs a reply left out.
BATCH_PAIRS = 40

#: A sentence that names this many words its source's title, authors and stored text do not, when that text is only a
#: snippet, a contents list or a title, says more than the source shows (the same bar as the title-only rule).
UNEXPLAINED_STEMS = 4
#: A retrieved text with fewer than this many prose sentences (eight words or more) is a snippet or a contents list.
_PROSE_SENTENCES = 3
_PROSE_WORDS = 8

_PART = r"(?:W\d+|\d+(?:\s*[–-]\s*\d+)?)"
CITE_RE = re.compile(rf"\[({_PART}(?:\s*[,;]\s*{_PART})*)\]")
_NOT_PROSE_RE = re.compile(r"`[^`\n]*`|\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)
_REFERENCES_RE = re.compile(r"^#{1,6}\s*references\s*$", re.IGNORECASE | re.MULTILINE)
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*%?")
_SENTENCES_RE = re.compile(r"(?<=[.!?])\s+")
_ACTIONS = ("cite_other", "reword", "drop_citation", "drop_sentence")
_TAIL_RE = re.compile(r"[.!?]+[\s*_)\]\"']*$")


# --- the pairs -----------------------------------------------------------------------------------------------------


def labels_in(inside: str) -> list[str]:
    """The source labels one bracket names, ranges spelled out (``2–4`` is 2, 3, 4)."""
    out: list[str] = []
    for part in re.split(r"[,;]", inside):
        part = part.strip()
        if not part:
            continue
        if part.startswith("W"):
            out.append(part)
            continue
        ends = [int(x) for x in re.split(r"[–-]", part) if x.strip()]
        out += [str(n) for n in range(ends[0], min(ends[-1], ends[0] + 50) + 1)]
    return out


def references_start(paper: str) -> int:
    """Where the paper's ``## References`` section begins (the end of the text that cites), or its length."""
    m = _REFERENCES_RE.search(paper or "")
    return m.start() if m else len(paper or "")


@dataclass
class Site:
    """One sentence of the paper that cites, as an exact span of its text."""

    start: int
    end: int
    text: str
    labels: list[str]
    kind: str  # "prose" | "list" | "table" | "figure": only prose is ever cut out whole
    key: str = ""  # names the sentence in the record: its words, and which of the same words it is


def find_sites(paper: str) -> list[Site]:
    """Every sentence of the paper's text that cites a source, with the labels it names. Brackets in inline code or
    math, a bracket followed by ``(`` (a link) and a bracket holding 0 or a year-sized number are not citations."""
    end = references_start(paper)
    region = paper[:end]
    spans = [(m.start(), m.end()) for m in _NOT_PROSE_RE.finditer(region)]
    sites: list[Site] = []
    seen: dict[str, int] = {}
    for start, stop, _block in _pp.sentence_units(region):
        seg = region[start:stop]
        if seg.lstrip().startswith("#"):
            continue
        labels: list[str] = []
        for m in CITE_RE.finditer(seg):
            at = start + m.start()
            if region[start + m.end():start + m.end() + 1] == "(" or any(a <= at < b for a, b in spans):
                continue
            found = labels_in(m.group(1))
            numbers = [int(x) for x in found if x.isdigit()]
            if any(n == 0 or n >= 1000 for n in numbers):
                continue
            labels += [x for x in found if x not in labels]
        if not labels:
            continue
        line_start = region.rfind("\n", 0, start) + 1
        line_end = region.find("\n", start)
        if seg.lstrip().startswith("|"):
            kind = "table"
        elif seg.lstrip().startswith("!["):
            kind = "figure"
        elif _pp._LIST_MARK_RE.match(region[line_start:line_end if line_end >= 0 else len(region)]):
            kind = "list"
        else:
            kind = "prose"
        digest = hashlib.sha1(" ".join(seg.split()).encode("utf-8")).hexdigest()[:10]
        seen[digest] = seen.get(digest, 0) + 1
        sites.append(Site(start, stop, seg, labels, kind, digest if seen[digest] == 1 else f"{digest}.{seen[digest]}"))
    return sites


def pair_key(site: Site, label: str) -> str:
    return f"{site.key}:{label}"


# --- what a source's retrieved text can show -------------------------------------------------------------------------


@dataclass
class Source:
    """A citable source as the check sees it: the words of its retrieved text, never a model's reading of a figure."""

    label: str
    title: str
    text: str
    full: bool = False  # the retrieved text is the source's full text
    authors: list[str] = field(default_factory=list)
    retracted: bool = False
    usable: bool = True  # may a fix cite it (a web page the Further reading no longer lists may not)

    def kind(self) -> str:
        """``full text`` | ``abstract`` | ``snippet or contents list`` | ``title only`` | ``nothing``: what the text is."""
        body = self.text.strip()
        if not body:
            return "nothing"
        if self.full:
            return "full text"
        # What the search returned about the source, not a page fetched after it (which may be a login page).
        head = body.partition("---FULL TEXT")[0].strip() or body
        if len(head.replace(self.title, "", 1).strip()) < 100:
            return "title only"
        return "abstract" if _prose_sentences(head, self.title) >= _PROSE_SENTENCES else "snippet or contents list"


def _prose_sentences(text: str, title: str = "") -> int:
    body = text.strip()
    if title and body.startswith(title):
        body = body[len(title):]
    return sum(1 for s in _SENTENCES_RE.split(body) if len(s.split()) >= _PROSE_WORDS)


def _numbers(sentence: str) -> list[str]:
    """The numbers a sentence states in its prose (not the citation brackets, not math or code): a decimal, a
    percentage or an integer of two digits or more, and not a year. A lone digit is a count or a step, not a finding."""
    text = _NOT_PROSE_RE.sub(" ", CITE_RE.sub(" ", sentence))
    out: list[str] = []
    for m in _NUMBER_RE.finditer(text):
        tok = m.group(0).rstrip("%").replace(",", "")
        digits = tok.replace(".", "")
        if len(digits) < 2 and "." not in tok and not m.group(0).endswith("%"):
            continue
        if tok.isdigit() and len(tok) == 4 and 1500 <= int(tok) <= 2100:
            continue
        if tok not in out:
            out.append(tok)
    return out


class OwnNumbers:
    """The numbers this study's own results hold (as written), so a number the paper states that comes from the run is
    told from one a source has to show. A number is the run's when it equals one to the decimals it is written to."""

    def __init__(self, tokens: Any = ()) -> None:
        self.tokens = {str(t).rstrip("%").replace(",", "") for t in tokens}
        self.floats: list[float] = []
        for t in self.tokens:
            try:
                self.floats.append(float(t))
            except ValueError:
                continue
        self.floats.sort()

    def __contains__(self, token: object) -> bool:
        tok = str(token).rstrip("%").replace(",", "")
        if tok in self.tokens:
            return True
        try:
            value = float(tok)
        except ValueError:
            return False
        tol = 0.51 * 10 ** -(len(tok.split(".")[1]) if "." in tok else 0)
        i = bisect.bisect_left(self.floats, value - tol)
        return any(abs(f - value) <= tol or abs(f * 100 - value) <= tol for f in self.floats[i:i + 3]) or any(
            abs(f * 100 - value) <= tol for f in self.floats)

    def __len__(self) -> int:
        return len(self.tokens)


def _has_number(text: str, token: str) -> bool:
    flat = text.replace(",", "")
    return token in flat or (token.startswith("0.") and token[1:] in flat)


def pre_problem(site: Site, label: str, src: Source | None, own_numbers: Any) -> str:
    """Why the sentence cannot be backed by this source whatever a reader says, or ``""``. Each rule here only ever adds
    "unsupported"; none can be loosened by a reply."""
    if src is None:
        return f"[{label}] is not one of the paper's sources"
    if src.retracted:
        return f"[{label}] has been retracted"
    kind = src.kind()
    if kind == "nothing":
        return f"no text of [{label}] was retrieved, so nothing can be quoted from it"
    sentence = site.text
    missing = [n for n in _numbers(sentence) if n not in own_numbers and not _has_number(src.text, n)]
    if missing:
        return (f"the sentence states {', '.join(missing[:4])}, which is neither in the retrieved text of [{label}] "
                f"({kind}) nor among this study's own results")
    if kind in ("title only", "snippet or contents list"):
        known = _content_stems(" ".join([src.title, *src.authors, src.text]))
        said = _content_stems(_NOT_PROSE_RE.sub(" ", CITE_RE.sub(" ", sentence)))
        extra = sorted(said - known)
        if len(extra) >= UNEXPLAINED_STEMS:
            return (f"the retrieved text of [{label}] is only a {kind}; the sentence says more than it shows "
                    f"({', '.join(extra[:6])})")
    return ""


def post_problem(quote: str, src: Source, quote_ok: Callable[[str, str], bool]) -> str:
    """Why the passage a reader named does not back the sentence: it is not in the source's own text."""
    if not " ".join((quote or "").split()):
        return "no passage of the retrieved text was named"
    if not quote_ok(quote, src.text):
        return f"the passage named is not in the retrieved text of [{src.label}]"
    return ""


# --- rewriting a sentence's brackets --------------------------------------------------------------------------------


def _bracket_text(labels: list[str]) -> str:
    numbers = sorted({int(x) for x in labels if x.isdigit()})
    out: list[str] = []
    i = 0
    while i < len(numbers):
        j = i
        while j + 1 < len(numbers) and numbers[j + 1] == numbers[j] + 1:
            j += 1
        out += [f"{numbers[i]}–{numbers[j]}"] if j - i >= 2 else [str(n) for n in numbers[i:j + 1]]
        i = j + 1
    return ", ".join(out + sorted({x for x in labels if not x.isdigit()}, key=lambda w: int(w[1:])))


def rewrite_brackets(sentence: str, *, remove: set[str], add: list[str] | None = None) -> str:
    """``sentence`` with the ``remove`` labels taken out of its brackets (a bracket left empty goes, with the space before
    it) and the ``add`` labels put in its last bracket, or before its closing punctuation when none is left."""
    spans = [(m.start(), m.end()) for m in _NOT_PROSE_RE.finditer(sentence)]
    pieces: list[str] = []
    last = 0
    kept: list[tuple[int, int, list[str]]] = []
    for m in CITE_RE.finditer(sentence):
        if sentence[m.end():m.end() + 1] == "(" or any(a <= m.start() < b for a, b in spans):
            continue
        labels = [x for x in labels_in(m.group(1)) if x not in remove]
        kept.append((m.start(), m.end(), labels))
    add = [a for a in (add or []) if a]
    placed = False
    for idx, (a, b, labels) in enumerate(kept):
        if add and idx == len(kept) - 1 and not placed:
            labels = labels + [x for x in add if x not in labels]
            placed = True
        start = a
        if not labels:
            while start > last and sentence[start - 1] in " \t":
                start -= 1
        pieces += [sentence[last:start], f"[{_bracket_text(labels)}]" if labels else ""]
        last = b
    pieces.append(sentence[last:])
    out = "".join(pieces)
    if add and not placed:
        tail = re.search(r"[.!?]+[\s*_)\]]*$", out)
        cut = tail.start() if tail else len(out)
        out = out[:cut].rstrip() + f" [{_bracket_text(add)}]" + out[cut:]
    return out


# --- the model's replies --------------------------------------------------------------------------------------------


def _json_object(reply: str) -> dict[str, Any]:
    data = _pp._load_json(reply or "")
    return data if isinstance(data, dict) else ({"verdicts": data} if isinstance(data, list) else {})


def parse_verdicts(reply: str, asked: set[str]) -> dict[str, dict[str, Any]]:
    """``{pair id: {"supported": bool, "quote": str, "why": str}}`` for the ids asked about; a reply entry for another id
    or without a yes/no is left out (that pair is asked again, or left unchecked)."""
    out: dict[str, dict[str, Any]] = {}
    for item in _json_object(reply).get("verdicts") or []:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("id") or "").strip()
        flag = item.get("supported")
        if isinstance(flag, str):
            flag = {"true": True, "yes": True, "false": False, "no": False}.get(flag.strip().lower())
        if pid in asked and isinstance(flag, bool) and pid not in out:
            out[pid] = {"supported": flag, "quote": " ".join(str(item.get("quote") or "").split()),
                        "why": " ".join(str(item.get("why") or item.get("note") or "").split())[:300]}
    return out


def parse_fixes(reply: str, asked: set[str]) -> dict[str, dict[str, Any]]:
    """``{sentence id: {"action", "source", "quote", "text"}}`` for the sentences asked about, with an action FI knows."""
    out: dict[str, dict[str, Any]] = {}
    for item in _json_object(reply).get("fixes") or []:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or "").strip()
        action = str(item.get("action") or "").strip().lower()
        if sid in asked and action in _ACTIONS and sid not in out:
            src = str(item.get("source") or "").strip().strip("[]")
            out[sid] = {"action": action, "source": src, "quote": " ".join(str(item.get("quote") or "").split()),
                        "text": str(item.get("text") or "").strip()}
    return out


# --- the check -----------------------------------------------------------------------------------------------------


@dataclass
class Row:
    """What became of one citation: the passage that backs it, or what was done instead."""

    sentence: str
    label: str
    title: str
    kind: str
    status: str  # supported | unchecked | replaced | reworded | citation_dropped | sentence_dropped
    passage: str = ""
    problem: str = ""
    action: str = ""
    new_sentence: str = ""
    new_label: str = ""
    note: str = ""  # what was left in the paper, said plainly (a number no source shows, kept with its citation gone)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v != ""}


@dataclass
class Result:
    paper: str
    rows: list[Row]
    counts: dict[str, int]
    changed: bool
    judge_failed: bool = False

    @property
    def clean(self) -> bool:
        return not self.judge_failed and not self.counts.get("unchecked")


def count_rows(rows: list[Row]) -> dict[str, int]:
    c = {"citations": len(rows), "supported": 0, "found_unsupported": 0, "replaced": 0, "reworded": 0,
         "citation_dropped": 0, "sentence_dropped": 0, "clause_dropped": 0, "left_in_place": 0, "unchecked": 0}
    for r in rows:
        if r.status == "supported":
            c["supported"] += 1
        elif r.status == "unchecked":
            c["unchecked"] += 1
        else:
            c["found_unsupported"] += 1
            c[r.status] = c.get(r.status, 0) + 1
            if r.note:
                c["left_in_place"] += 1
    return c


def _bounded(record: dict[str, Any], kind: str, cap: int) -> bool:
    """Whether a call of ``kind`` may still be made: this paper's cap and the quest's cap."""
    return int(record.get(f"{kind}_calls") or 0) < cap and int(record.get("calls_in_quest") or 0) < MAX_CALLS_IN_QUEST


def _spend(record: dict[str, Any], kind: str, save: Callable[[dict[str, Any]], None]) -> None:
    """Count a call in the record and save it BEFORE the call is made: a stop in the middle never asks again for it."""
    record[f"{kind}_calls"] = int(record.get(f"{kind}_calls") or 0) + 1
    record["calls_in_quest"] = int(record.get("calls_in_quest") or 0) + 1
    save(record)


def _best_passage(src: Source, query: str, chars: int = 700) -> tuple[float, str]:
    """The passage of the source's retrieved text most like ``query`` (the lexical ranking the claim check's excerpts
    use) and its score."""
    text = src.text[:200_000]
    chunks = [text[i:i + 1200] for i in range(0, len(text), 1000)] or [text]
    scores = rank_by_relevance(chunks, query)
    i = max(range(len(chunks)), key=lambda k: scores[k])
    return scores[i], chunks[i][:chars]


def _plain(sentence: str) -> str:
    return _NOT_PROSE_RE.sub(" ", CITE_RE.sub(" ", sentence))


def _candidates(sentence: str, sources: dict[str, Source], exclude: set[str], *, top: int = 3) -> list[tuple[str, str]]:
    """The other sources that most look like they say the sentence, as ``(label, best passage)``."""
    best: list[tuple[float, str, str]] = []
    for label, src in sources.items():
        if label in exclude or not src.usable or src.retracted or not src.text.strip():
            continue
        score, passage = _best_passage(src, _plain(sentence))
        if score > 0:
            best.append((score, label, passage))
    best.sort(key=lambda t: -t[0])
    return [(label, passage) for _s, label, passage in best[:top]]


def drop_clause(sentence: str, foreign: list[str], own_numbers: Any) -> str | None:
    """``sentence`` without the one comma- or semicolon-separated clause that holds every number no source shows, when that
    clause is not the first, holds none of the study's own numbers, and something else is left; ``None`` otherwise (the
    sentence is then not cut: a clause cut anywhere else could leave a sentence that does not read)."""
    parts = re.split(r"(?<=[,;])\s+", sentence)
    if len(parts) < 2:
        return None
    hit = [i for i, p in enumerate(parts) if any(n in _numbers(p) for n in foreign)]
    if len(hit) != 1 or hit[0] == 0 or any(n in own_numbers for n in _numbers(parts[hit[0]])):
        return None
    tail = _TAIL_RE.search(sentence)
    out = " ".join(parts[:hit[0]] + parts[hit[0] + 1:]).rstrip(" ,;")
    if not out.strip() or any(n in _numbers(out) for n in foreign):
        return None
    return out if _TAIL_RE.search(out) else out + (tail.group(0) if tail else ".")


def settle(site: Site, labels: set[str], own_numbers: Any, claim_unsupported: bool) -> tuple[str, str, str]:
    """``(action, new text, note)``: the rule that settles a sentence nobody could fix (needs no model). The study's own
    result is never cut out. A plain prose sentence with no result of the study goes when it states a number no source
    shows or the claim check found nothing backing it either. A sentence that carries one of the study's own results keeps
    it: the clause holding the numbers no source shows goes when that can be cut cleanly; otherwise only the citation goes
    and the note says what was left in the paper."""
    numbers = _numbers(site.text)
    own = [n for n in numbers if n in own_numbers]
    foreign = [n for n in numbers if n not in own_numbers]
    plain = rewrite_brackets(site.text, remove=labels)
    if not own and site.kind == "prose" and (foreign or claim_unsupported):
        return "drop_sentence", "", ""
    if own and foreign:
        clause = drop_clause(site.text, foreign, own_numbers)
        if clause:
            return "drop_clause", rewrite_brackets(clause, remove=labels), ""
    left = ""
    if foreign:
        left = f"left in the paper with its citation taken out: {', '.join(foreign[:4])}, which no source shows"
    elif claim_unsupported:
        left = "left in the paper with its citation taken out: the claim check found nothing backing it either"
    return "drop_citation", plain, left


def _apply(paper: str, edits: list[tuple[int, int, str]]) -> str:
    """The paper with each ``(start, end, replacement)`` made, from the end back so the spans stay true. A sentence cut
    out takes one of the spaces beside it."""
    out = paper
    for start, stop, text in sorted(edits, key=lambda e: -e[0]):
        if not text:
            if stop < len(out) and out[stop] in " \t":
                stop += 1
            elif start > 0 and out[start - 1] in " \t":
                start -= 1
        out = out[:start] + text + out[stop:]
        if not text:  # a paragraph that was one sentence leaves no empty paragraph behind
            lo, hi = max(0, start - 2), min(len(out), start + 2)
            out = out[:lo] + re.sub(r"\n{3,}", "\n\n", out[lo:hi]) + out[hi:]
    return out


def _shape(text: str) -> tuple[list[str], list[str]]:
    return _pp._shape(text)


async def check_paper(
    paper: str,
    sources: dict[str, Source],
    *,
    own_numbers: Any,
    quote_ok: Callable[[str, str], bool],
    judge: Callable[[str], Awaitable[str]],
    repair: Callable[[str], Awaitable[str]],
    record: dict[str, Any],
    save: Callable[[dict[str, Any]], None],
    judge_prompt: Callable[[list[tuple[str, Site, str]], dict[str, Source]], str],
    repair_prompt: Callable[[list[dict[str, Any]]], str],
    claim_unsupported: Callable[[str], bool] = lambda sentence: False,
) -> Result:
    """Check every citation of ``paper`` and fix what is unsupported. See the module text for the steps.

    ``record`` is the check's record on disk (loaded by the caller for this paper, saved through ``save``): it holds the
    calls spent and the replies already had, so a resumed check goes on from them. ``judge`` and ``repair`` each make
    ONE model call and return its text."""
    sites = find_sites(paper)
    if not sites:
        return Result(paper, [], count_rows([]), False)
    by_key = {s.key: s for s in sites}
    pairs: list[tuple[str, Site, str]] = []
    for site in sites:
        for label in site.labels:
            pairs.append((pair_key(site, label), site, label))

    # 1. the rules that need no model, then the reader's judgement of the pairs left.
    problems: dict[str, str] = {}
    for pid, site, label in pairs:
        why = pre_problem(site, label, sources.get(label), own_numbers)
        if why:
            problems[pid] = why
    judged: dict[str, dict[str, Any]] = dict(record.setdefault("judged", {}))
    to_ask = [(pid, s, lb) for pid, s, lb in pairs if pid not in problems and pid not in judged]
    judge_failed = False
    cap = max(int(record.get("judge_cap") or 0), -(-len(to_ask) // BATCH_PAIRS) + MAX_JUDGE_CALLS - 1)
    record["judge_cap"] = cap
    while to_ask and _bounded(record, "judge", cap):
        _spend(record, "judge", save)
        chunk, rest = to_ask[:BATCH_PAIRS], to_ask[BATCH_PAIRS:]
        try:
            reply = await judge(judge_prompt(chunk, sources))
        except Exception as e:  # noqa: BLE001 -- a check that could not be had leaves the paper as it is, and says so
            record["judge_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            save(record)
            judge_failed = True
            break
        got = parse_verdicts(reply, {pid for pid, _s, _l in chunk})
        judged.update(got)
        record["judged"] = judged
        save(record)
        # the pairs a reply left out go to the back, so one that answers nothing cannot keep the others waiting
        to_ask = rest + [c for c in chunk if c[0] not in judged]
    for pid, site, label in pairs:
        if pid in problems or pid not in judged:
            continue
        v = judged[pid]
        src = sources[label]
        if not v["supported"]:
            problems[pid] = v.get("why") or f"the reader found no passage of the retrieved text of [{label}] that says it"
        else:
            bad = post_problem(v["quote"], src, quote_ok)
            if bad:
                problems[pid] = bad
    unchecked = {pid for pid, _s, _l in pairs if pid not in problems and pid not in judged}

    # 2. fix what is unsupported: one repair call for all of it, then the reader looks at the fixes.
    bad_sites: dict[str, list[str]] = {}
    for pid, site, label in pairs:
        if pid in problems:
            bad_sites.setdefault(site.key, []).append(label)
    plans: dict[str, dict[str, Any]] = {}  # sentence key -> {action, text, add, source, quote}
    if bad_sites and not judge_failed:
        targets = []
        for sk, labels in bad_sites.items():
            site = by_key[sk]
            targets.append({
                "id": sk, "sentence": site.text, "kind": site.kind, "labels": labels,
                "why": {lb: problems[pair_key(site, lb)] for lb in labels},
                "kept": [lb for lb in site.labels if lb not in labels],
                "cited": [(lb, _best_passage(sources[lb], _plain(site.text))[1]) for lb in labels
                          if lb in sources and sources[lb].text.strip()],
                "others": _candidates(site.text, sources, set(site.labels)),
            })
        fixes = dict(record.get("fixes") or {})
        if not all(t["id"] in fixes for t in targets) and _bounded(record, "repair", MAX_REPAIR_CALLS):
            _spend(record, "repair", save)
            try:
                fixes.update(parse_fixes(await repair(repair_prompt(targets)), {t["id"] for t in targets}))
            except Exception as e:  # noqa: BLE001 -- the rules below settle every sentence the reply did not
                record["repair_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            record["fixes"] = fixes
            save(record)
        for t in targets:
            site = by_key[t["id"]]
            plan = _validate_fix(site, t, fixes.get(t["id"]), sources, own_numbers, quote_ok)
            if plan:
                plans[site.key] = plan
    # What the repair proposed, read once more by the reader before it stands.
    proposed = {sk: p for sk, p in plans.items() if p["action"] in ("cite_other", "reword")}
    if proposed:
        confirm: list[tuple[str, Site, str]] = []
        for sk, p in proposed.items():
            new_site = Site(0, 0, p["text"], labels_in_text(p["text"]), by_key[sk].kind)
            p["site"] = new_site
            for lb in new_site.labels:
                confirm.append((f"{sk}>{lb}", new_site, lb))
        verified = dict(record.get("verified") or {})
        ask = [c for c in confirm if c[0] not in verified]
        if ask and _bounded(record, "verify", MAX_VERIFY_CALLS):
            _spend(record, "verify", save)
            try:
                verified.update(parse_verdicts(await judge(judge_prompt(ask, sources)), {c[0] for c in ask}))
            except Exception as e:  # noqa: BLE001 -- an unconfirmed fix is not kept
                record["verify_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            record["verified"] = verified
            save(record)
        for sk, p in list(proposed.items()):
            ok = True
            for lb in p["site"].labels:
                v = verified.get(f"{sk}>{lb}")
                if not v or not v["supported"] or post_problem(v["quote"], sources[lb], quote_ok):
                    ok = False
            if not ok:
                plans.pop(sk, None)

    # 3. the paper as it is delivered, and the ledger of what became of each citation. Each edit is checked on its own: one
    # that would change a heading or a figure link is settled by the rule that needs no model instead; the others stand.
    rows: list[Row] = []
    edits: list[tuple[int, int, str]] = []
    base_shape = _shape(paper[:references_start(paper)])

    def keeps_shape(site: Site, text: str) -> bool:
        done = _apply(paper, [(site.start, site.end, text)])
        return _shape(done[:references_start(done)]) == base_shape

    for sk, labels in bad_sites.items():
        site = by_key[sk]
        plan = plans.get(sk)
        fell_back = plan is None
        note = ""
        if plan is None:
            plan = {}
        else:
            new_text = "" if plan["action"] == "drop_sentence" else (
                plan["text"] if plan["action"] in ("cite_other", "reword") else rewrite_brackets(site.text, remove=set(labels)))
            if not keeps_shape(site, new_text):
                plan, fell_back = {}, True
        if fell_back:
            action, new_text, note = settle(site, set(labels), own_numbers, claim_unsupported(site.text))
            if not keeps_shape(site, new_text):
                action, new_text = "drop_citation", rewrite_brackets(site.text, remove=set(labels))
                note = note or "left in the paper with its citation taken out: removing more would change a heading or a figure"
            plan = {"action": action, "text": ""}
        elif plan["action"] == "drop_sentence":
            new_text = ""
        elif plan["action"] in ("cite_other", "reword"):
            new_text = plan["text"]
        else:
            new_text = rewrite_brackets(site.text, remove=set(labels))
        edits.append((site.start, site.end, new_text))
        plan.update(final=new_text, fell_back=fell_back, note=note)
        plans[sk] = plan
    new_paper = _apply(paper, edits) if edits else paper
    status_of = {"cite_other": "replaced", "reword": "reworded", "drop_citation": "citation_dropped",
                 "drop_sentence": "sentence_dropped", "drop_clause": "clause_dropped"}
    for pid, site, label in pairs:
        src = sources.get(label)
        title = src.title if src else ""
        kind = src.kind() if src else "nothing"
        if pid in problems:
            plan = plans.get(site.key, {"action": "drop_citation", "final": ""})
            rows.append(Row(site.text, label, title, kind, status_of[plan["action"]], problem=problems[pid],
                            action=plan["action"], new_sentence=plan.get("final", ""), note=plan.get("note", ""),
                            new_label=str(plan.get("source") or "") if plan["action"] == "cite_other" else ""))
        elif pid in unchecked:
            rows.append(Row(site.text, label, title, kind, "unchecked",
                            problem=record.get("judge_error") or "the reader gave no verdict"))
        else:
            rows.append(Row(site.text, label, title, kind, "supported", passage=judged[pid]["quote"]))
    for sk, plan in plans.items():
        if plan["action"] in ("cite_other", "reword") and not plan.get("fell_back"):
            for lb in plan["site"].labels:
                v = (record.get("verified") or {}).get(f"{sk}>{lb}", {})
                if lb == plan.get("source") or plan["action"] == "reword":
                    src = sources[lb]
                    rows.append(Row(plan["final"], lb, src.title, src.kind(), "supported", passage=v.get("quote", ""),
                                    action=f"added by {plan['action']}"))
    return Result(new_paper, rows, count_rows([r for r in rows if not r.action.startswith("added")]),
                  new_paper != paper, judge_failed)


def labels_in_text(sentence: str) -> list[str]:
    """The labels a sentence cites (outside code and math)."""
    spans = [(m.start(), m.end()) for m in _NOT_PROSE_RE.finditer(sentence)]
    out: list[str] = []
    for m in CITE_RE.finditer(sentence):
        if sentence[m.end():m.end() + 1] == "(" or any(a <= m.start() < b for a, b in spans):
            continue
        out += [x for x in labels_in(m.group(1)) if x not in out]
    return out


def _validate_fix(
    site: Site, target: dict[str, Any], fix: dict[str, Any] | None, sources: dict[str, Source],
    own_numbers: Any, quote_ok: Callable[[str, str], bool],
) -> dict[str, Any] | None:
    """The fix as a plan the check can apply, or ``None`` when the reply gave none or it breaks a rule (then the
    sentence is settled by :func:`fallback`). Every source a new sentence cites passes the rules that need no model."""
    if not fix:
        return None
    action = fix["action"]
    if action in ("drop_citation", "drop_sentence"):
        if action == "drop_sentence" and site.kind != "prose":
            return {"action": "drop_citation", "text": ""}
        return {"action": action, "text": ""}
    bad = set(target["labels"])
    if action == "cite_other":
        label = fix["source"]
        src = sources.get(label)
        if src is None or label in bad or not src.usable:
            return None
        if post_problem(fix["quote"], src, quote_ok):
            return None
        new = rewrite_brackets(site.text, remove=bad, add=[label])
        probe = Site(0, 0, new, labels_in_text(new), site.kind)
        if pre_problem(probe, label, src, own_numbers):
            return None
        if _numbers(new) != _numbers(site.text):
            return None
        return {"action": action, "text": new, "source": label}
    # reword: a new sentence that cites only sources it may (the ones already supported, and the one it names)
    new = " ".join(fix["text"].split())
    if not new or "\n" in fix["text"].strip() or new == " ".join(site.text.split()):
        return None
    # The study's own results in the sentence are kept exactly, and none is added: a new wording may not change a result.
    kept_own = {n for n in _numbers(site.text) if n in own_numbers}
    new_numbers = _numbers(new)
    if not kept_own <= set(new_numbers) or any(n in own_numbers and n not in kept_own for n in new_numbers):
        return None
    allowed = set(target["kept"]) | ({fix["source"]} if fix["source"] else set())
    cited = labels_in_text(new)
    if any(lb not in allowed for lb in cited):
        return None
    for lb in cited:
        src = sources.get(lb)
        if src is None:
            return None
        probe = Site(0, 0, new, cited, site.kind)
        if pre_problem(probe, lb, src, own_numbers):
            return None
    if fix["source"] and fix["source"] in cited:
        src = sources[fix["source"]]
        if post_problem(fix["quote"], src, quote_ok):
            return None
    return {"action": action, "text": new, "source": fix["source"]}


# --- the record and the ledger --------------------------------------------------------------------------------------


def paper_hash(paper: str) -> str:
    return hashlib.sha256(paper.encode("utf-8", errors="replace")).hexdigest()


def new_record(paper: str, previous: dict[str, Any] | None) -> dict[str, Any]:
    """The record for a paper: its own counts, on top of the quest's total (the calls a different paper already spent)."""
    return {"paper_sha": paper_hash(paper), "judge_calls": 0, "repair_calls": 0, "verify_calls": 0,
            "calls_in_quest": int((previous or {}).get("calls_in_quest") or 0)}


def ledger(result: Result, *, reader: str, same_model: bool | None) -> dict[str, Any]:
    return {"counts": result.counts, "reader": reader, "reader_is_writer": same_model,
            "judge_failed": result.judge_failed, "citations": [r.as_dict() for r in result.rows]}


def render_md(data: dict[str, Any]) -> str:
    c = data.get("counts") or {}
    lines = ["# Citations", "",
             f"{c.get('citations', 0)} citation(s) in the final text: {c.get('supported', 0)} backed by a passage of the "
             f"retrieved text as written, {c.get('found_unsupported', 0)} found unsupported and fixed "
             f"({c.get('replaced', 0)} cited to another source, {c.get('reworded', 0)} reworded, "
             f"{c.get('citation_dropped', 0)} citation dropped, {c.get('sentence_dropped', 0)} sentence dropped)"
             + (f", {c.get('unchecked')} could not be checked" if c.get("unchecked") else "") + ".", ""]
    if data.get("reader_is_writer"):
        lines += [f"Read by {data.get('reader')}, the model that wrote the paper: a self-review.", ""]
    elif data.get("reader"):
        lines += [f"Read by {data.get('reader')}.", ""]
    for r in data.get("citations") or []:
        if str(r.get("action") or "").startswith("added"):
            lines.append(f"- [{r['label']}] now cites: {r['sentence']} -- passage: \"{r.get('passage', '')}\"")
            continue
        head = f"- [{r['label']}] {r.get('title', '')} ({r.get('kind', '')}): {r['sentence']}"
        if r["status"] == "supported":
            lines.append(f"{head}\n  - supported -- passage: \"{r.get('passage', '')}\"")
        elif r["status"] == "unchecked":
            lines.append(f"{head}\n  - not checked: {r.get('problem', '')}")
        else:
            lines.append(f"{head}\n  - {r['status'].replace('_', ' ')}: {r.get('problem', '')}"
                         + (f" -- now: {r['new_sentence']}" if r.get("new_sentence") else "")
                         + (f"\n  - note: {r['note']}" if r.get("note") else ""))
    return "\n".join(lines) + "\n"


def dump(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n"
