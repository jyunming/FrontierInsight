"""The supplement FI's own parts go to, out of the paper's body.

Every paper sends two of FI's own parts here, with or without a page limit and without a model call
(``Engine._supplement_always``): its tables of every setting (:data:`TABLES`; the paper keeps one sentence that points
here, :func:`table_pointer`) when the block holds a table, and its Further reading (:data:`FURTHER_READING`; the
References keep one line that says how many pages it lists and where, :func:`further_pointer`) when the text cites none
of its pages. These are :data:`ALWAYS`. What moved is written, unchanged, to ``paper/supplement.md`` (:data:`FILE`),
which ships with the paper.

An older FI also moved its Limitations block here for a page limit (:data:`LIMITATIONS`), with the paper's own
paragraphs of that section; FI writes no such block now, so that part never moves, and a record that holds those
paragraphs gives them back to the draft they came from (:func:`put_back_writer_limitations`).

Sentences the page trim takes out of the paper's background and discussion (core/paper_trim.py, after FI's parts)
move here too, under :data:`SENTENCES`, each with the section it came from; their own record (:data:`TRIM_RECORD`)
keeps the text around the place each stood, so a later draft gets it back word for word where that text is unchanged
(:func:`put_back_sentences`), and one that cannot be placed again stays here: a sentence the trim took is always in
the paper or in the supplement.

The decision is a record (:data:`RECORD`), not an edit of the text: FI writes its blocks into the paper again after many
steps (core/paper_facts.py::place_blocks), and the block it writes for a moved part is the pointer
(core/paper_facts.py::block_text). The writer's paragraphs are not FI's blocks, so they are taken out of the text once
and kept in the record."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

RECORD = Path(".fi") / "supplement.json"
FILE = Path("paper") / "supplement.md"
SHOWN = "paper/supplement.md"
TRIM_RECORD = Path(".fi") / "supplement_trim.json"
TABLES, LIMITATIONS, FURTHER_READING = "results_table", "limitations", "further_reading"
SENTENCES = "moved_sentences"
ORDER = (TABLES, LIMITATIONS, FURTHER_READING)
#: The parts every paper sends here, with or without a page limit; the others move only for a page limit.
ALWAYS = (TABLES, FURTHER_READING)
_TITLES = {TABLES: "Results for every setting", LIMITATIONS: "Limitations", FURTHER_READING: "Further reading",
           SENTENCES: "Sentences moved for length"}
#: How much of the text on each side of a moved sentence its record keeps, to find its place again (never past the
#: paragraph it stood in).
_ANCHOR = 60
BLANK = chr(10) * 2
#: What the supplement's first lines call each part.
_WHY = {TABLES: "tables of every setting", FURTHER_READING: "Further reading", LIMITATIONS: "the Limitations in full",
        SENTENCES: "some sentences of its text"}
#: What the run log calls each part.
NAMES = {TABLES: "tables of every setting", LIMITATIONS: "Limitations", FURTHER_READING: "Further reading"}


def read(quest_root: Path | str) -> dict[str, Any]:
    """The record: ``{"moved": [...], "further_reading": [labels], "limit": n}`` (``limit`` ``None`` without a page
    limit); ``{}`` when nothing moved or it cannot be read."""
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or not isinstance(data.get("moved"), list):
        return {}
    return data


def moved(quest_root: Path | str) -> list[str]:
    """The parts moved to the supplement, in :data:`ORDER`."""
    got = {str(m) for m in read(quest_root).get("moved") or []}
    return [p for p in ORDER if p in got]


def further_labels(quest_root: Path | str) -> list[str]:
    """The labels (``"W3"``...) of the web pages the Further reading listed when it moved; ``[]`` when it did not."""
    rec = read(quest_root)
    if FURTHER_READING not in (rec.get("moved") or []):
        return []
    return [str(x) for x in rec.get("further_reading") or []]


def further_text(quest_root: Path | str) -> str:
    """The entry lines of the Further reading as they were when it moved (``""`` when it did not): the supplement lists
    them, and a draft that fits without the move gets them back, unchanged."""
    rec = read(quest_root)
    return str(rec.get("further_reading_text") or "") if FURTHER_READING in (rec.get("moved") or []) else ""


def writer_text(quest_root: Path | str) -> str:
    """The paper's own paragraphs of its Limitations section that moved with an older FI's Limitations (``""`` when none
    did): the supplement shows them until they go back."""
    rec = read(quest_root)
    return str(rec.get("limitations_writer") or "") if LIMITATIONS in (rec.get("moved") or []) else ""


def writer_section(quest_root: Path | str) -> tuple[str, str] | None:
    """``(before, after)``: the paper's Limitations section as it was before the writer's paragraphs moved and as the move
    left it (:func:`section_body`), so the draft they came from gets them back exactly; ``None`` when none moved."""
    rec = read(quest_root)
    if LIMITATIONS not in (rec.get("moved") or []) or not rec.get("limitations_writer"):
        return None
    return str(rec.get("limitations_before") or ""), str(rec.get("limitations_after") or "")


def with_writer_paragraphs(text: str, quest_root: Path | str) -> str:
    """``text`` with the paper's own Limitations paragraphs that an older FI moved to the supplement with its block put
    back where they stood, while the section is still as that move left it (:func:`put_back_writer_limitations`): they
    are the writer's text, so taking FI's old block out of a resumed paper never takes them with it. ``text`` unchanged
    when no paragraphs moved or the section changed."""
    section = writer_section(quest_root)
    if section is None:
        return text
    back = put_back_writer_limitations(text, *section)
    return text if back is None else back


def writer_text_shown(quest_root: Path | str, paper: str) -> str:
    """The writer's moved Limitations paragraphs the supplement still shows (:func:`writer_text`): ``""`` once ``paper``
    holds them again, so they are never in both."""
    own = writer_text(quest_root).strip()
    return "" if not own or own in paper else own


_MARKED = re.compile(r"<!-- fi:fact-([\w-]+) -->.*?<!-- /fi:fact-\1 -->", re.S)
_HEADING = re.compile(r"^#{1,6}[ \t]+(.*)$")
_TOKEN = re.compile(r"\x00(\d+)\x00")


def _masked(text: str) -> tuple[str, list[str]]:
    """``text`` with each of FI's marked blocks made one token, so a blank line inside one never splits it."""
    blocks: list[str] = []

    def keep(m: re.Match[str]) -> str:
        blocks.append(m.group(0))
        return f"\x00{len(blocks) - 1}\x00"

    return _MARKED.sub(keep, text), blocks


def _unmasked(text: str, blocks: list[str]) -> str:
    return _TOKEN.sub(lambda m: blocks[int(m.group(1))], text)


def _limitations_section(lines: list[str], blocks: list[str]) -> tuple[int, int] | None:
    """(the heading line, the end) of the section that holds FI's Limitations block, when that section is the paper's
    Limitations (its title, numbering taken off, begins "Limitation"); ``None`` otherwise: a Discussion that holds the
    block, a bold run-in Limitations paragraph, no block."""
    at = next((i for i, ln in enumerate(lines) for m in _TOKEN.finditer(ln)
               if blocks[int(m.group(1))].startswith("<!-- fi:fact-limitations -->")), None)
    if at is None:
        return None
    head = next((i for i in range(at, -1, -1) if _HEADING.match(lines[i])), None)
    if head is None:
        return None
    title = re.sub(r"[*_`]+", "", _HEADING.match(lines[head]).group(1))  # type: ignore[union-attr]
    title = re.sub(r"^\s*[\dIVXivx]+(?:\.\d+)*[.)]?\s+", "", title).strip().lower()
    if not title.startswith("limitation"):
        return None
    end = next((i for i in range(head + 1, len(lines)) if _HEADING.match(lines[i]) or lines[i].strip() == "---"),
               len(lines))
    return head, end


def _name_of(block: str) -> str:
    m = _MARKED.match(block)
    return m.group(1) if m else ""


def _body(lines: list[str], blocks: list[str], head: int, end: int) -> str:
    """The section's text below its heading, FI's blocks as their names only (FI writes them again from its records),
    without FI's line pointing to the Further reading (:func:`_pointers`): it follows the last section of the paper,
    whichever that is, and is no part of it."""
    text = "\n".join(ln for ln in lines[head + 1:end] if not FURTHER_POINTER.match(ln)).strip()
    return _TOKEN.sub(lambda m: "\x00" + _name_of(blocks[int(m.group(1))]) + "\x00", text)


def _pointers(lines: list[str], head: int, end: int) -> list[str]:
    """FI's line pointing to the supplement's Further reading, where it stands after the section's own text."""
    return [ln for ln in lines[head + 1:end] if FURTHER_POINTER.match(ln)]


def section_body(text: str) -> str | None:
    """The paper's Limitations section below its heading, FI's blocks as their names only; ``None`` with no such
    section (see :func:`_limitations_section`)."""
    masked, blocks = _masked(text)
    lines = masked.split("\n")
    span = _limitations_section(lines, blocks)
    return None if span is None else _body(lines, blocks, *span)


def put_back_writer_limitations(text: str, before: str, after: str) -> str | None:
    """``text`` with its Limitations section as it was before the writer's paragraphs moved (``before``, from
    :func:`section_body`), when the section is still exactly as the move left it (``after``): the same draft gets back
    exactly what was taken from it, FI's blocks as FI writes them now. ``None`` for any other draft (a draft that changed
    its Limitations, or has none): it is measured as it was written."""
    masked, blocks = _masked(text)
    lines = masked.split("\n")
    span = _limitations_section(lines, blocks)
    if span is None:
        return None
    head, end = span
    if _body(lines, blocks, head, end) != after:
        return None
    now = {_name_of(blocks[int(m.group(1))]): blocks[int(m.group(1))]
           for ln in lines[head + 1:end] for m in _TOKEN.finditer(ln)}
    names = re.findall(r"\x00([\w-]+)\x00", before)
    if any(n not in now for n in names):
        return None
    body = re.sub(r"\x00([\w-]+)\x00", lambda m: now[m.group(1)], before)
    pointers = _pointers(lines, head, end)
    return _unmasked("\n".join([*lines[:head + 1], "", body, *(["", *pointers] if pointers else []),
                                *([""] if end < len(lines) else []), *lines[end:]]), blocks)


def save(quest_root: Path | str, parts: list[str], *, limit: int | None, further: list[str] | None = None,
         further_lines: str = "", writer: str = "", section: tuple[str, str] | None = None) -> None:
    """Keep ``parts`` as moved (none: the record and the file go); ``further``/``further_lines``: the labels and the
    entry lines of the Further reading that moved; ``writer``: the paper's own Limitations paragraphs that moved, and
    ``section`` the Limitations section before and after (:func:`writer_section`)."""
    root = Path(quest_root)
    parts = [p for p in ORDER if p in set(parts)]
    if not parts:
        (root / RECORD).unlink(missing_ok=True)
        (root / FILE).unlink(missing_ok=True)
        return
    data: dict[str, Any] = {"moved": parts, "limit": None if limit is None else int(limit)}
    if FURTHER_READING in parts:
        data["further_reading"] = list(further or [])
        data["further_reading_text"] = further_lines
    if LIMITATIONS in parts and writer.strip():
        data["limitations_writer"] = writer.strip()
        data["limitations_before"], data["limitations_after"] = section or ("", "")
    (root / RECORD).parent.mkdir(parents=True, exist_ok=True)
    (root / RECORD).write_text(json.dumps(data) + "\n", encoding="utf-8")


def sentences(quest_root: Path | str) -> list[dict[str, Any]]:
    """The places the page trim took text out of, as kept (:func:`places_of`); ``[]`` when there are none."""
    try:
        data = json.loads((Path(quest_root) / TRIM_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = data.get("places") if isinstance(data, dict) else None
    return [r for r in rows or [] if isinstance(r, dict) and isinstance(r.get("text"), str) and r["text"].strip()]


def anything(quest_root: Path | str) -> bool:
    """Whether the supplement holds anything: a part FI moved, or sentences the page trim moved."""
    return bool(read(quest_root)) or bool(sentences(quest_root))


def save_sentences(quest_root: Path | str, places: list[dict[str, Any]]) -> None:
    """Keep ``places`` as the moved sentences (none: the record goes)."""
    path = Path(quest_root) / TRIM_RECORD
    if not places:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"places": places}) + "\n", encoding="utf-8")


def places_of(trimmed: str, gaps: Any, *, step: int = 0) -> list[dict[str, Any]]:
    """The record of a trim (core/paper_trim.py ``Trimmed.gaps`` on the trimmed paper ``trimmed``): for each place, the
    exact text taken out, the text just left and right of where it stood, its sentences with their sections, and
    ``step``, the round of taking out it belongs to (a later round may take text next to it, so the rounds are put
    back latest first)."""
    def left(at: int) -> str:  # within the paragraph, so an edit of another paragraph leaves the place findable
        start = max(0, at - _ANCHOR)
        cut = trimmed.rfind(BLANK, start, at)
        return trimmed[cut + len(BLANK) if cut >= 0 else start:at]

    def right(at: int) -> str:
        stop = min(len(trimmed), at + _ANCHOR)
        cut = trimmed.find(BLANK, at, stop)
        return trimmed[at:cut if cut >= 0 else stop]

    return [{"text": removed, "left": left(at), "right": right(at), "step": step,
             "sentences": [{"text": x.text, "section": x.section} for x in taken]}
            for at, removed, taken in gaps]


def next_step(places: list[dict[str, Any]]) -> int:
    """The round number for the next taking out, after the rounds of ``places``."""
    return 1 + max((int(p.get("step") or 0) for p in places), default=0)


def put_back_sentences(text: str, places: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """``text`` with each moved place put back where the text around it is found once, unchanged; and the places that
    stay moved (the text around one was changed, or it is no longer found once). A place whose text the draft already
    holds is in the paper: it is neither put back nor kept. The rounds go back latest first: the text around a place
    is as it was when it was taken out once every later round is back."""
    kept: list[dict[str, Any]] = []
    for step in sorted({int(p.get("step") or 0) for p in places}, reverse=True):
        text, left = _put_back_round(text, [p for p in places if int(p.get("step") or 0) == step])
        kept += left
    return text, kept


def _put_back_round(text: str, places: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """:func:`put_back_sentences` for the places of one round."""
    found: list[tuple[int, dict[str, Any]]] = []
    kept: list[dict[str, Any]] = []
    for place in places:
        if place["text"].strip() in text:
            continue
        key = str(place.get("left") or "") + str(place.get("right") or "")
        if key and text.count(key) == 1:
            found.append((text.index(key) + len(str(place.get("left") or "")), place))
        else:
            kept.append(place)
    if len({at for at, _p in found}) != len(found):  # two places at one spot: their order is not known, both stay moved
        kept += [p for _a, p in found]
        found = []
    for at, place in sorted(found, key=lambda x: x[0], reverse=True):
        text = text[:at] + place["text"] + text[at:]
    return text, kept


def sentences_text(quest_root: Path | str) -> str:
    """The supplement's part for the moved sentences: each as it stood, under the section it came from."""
    rows = [x for r in sentences(quest_root) for x in r.get("sentences") or [] if isinstance(x, dict) and x.get("text")]
    if not rows:
        return ""
    out = ["FI took these sentences out of the paper's text to keep it within its page limit; each stands here as it "
           "stood there, under the section it came from."]
    for section in dict.fromkeys(str(x.get("section") or "") for x in rows):
        out.append((f"**{section}**\n\n" if section else "")
                   + "\n".join(f"- {' '.join(str(x['text']).split())}" for x in rows if str(x.get("section") or "") == section))
    return "\n\n".join(out)


def _listed(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def text(limit: int | None, parts: dict[str, str]) -> str:
    """The supplement's Markdown: one section per moved part with text, in :data:`ORDER`, then the moved sentences
    (:data:`SENTENCES`); ``""`` with none. Its first lines say why each part is here: the tables of every setting and
    the Further reading are always kept here, the others moved for the page limit (``limit``: its pages, ``None``
    without one)."""
    shown = [p for p in (*ORDER, SENTENCES) if (parts.get(p) or "").strip()]
    if not shown:
        return ""
    sections = [f"## {_TITLES[p]}\n\n{parts[p].strip()}\n" for p in shown]
    always = [_WHY[p] for p in shown if p in ALWAYS]
    for_limit = [_WHY[p] for p in shown if p not in ALWAYS]
    said = ["These parts of the paper stand here as FI wrote them; nothing in them was changed."]
    if always:
        said.append(f"FI always keeps the paper's {_listed(always)} here, out of its main text.")
    if for_limit:
        within = f" within its page limit of {limit} pages" if limit else " within its page limit"
        listed = _listed(for_limit)
        said.append(f"{listed[0].upper()}{listed[1:]} moved here to keep the paper{within}.")
    return "# Supplement\n\n" + " ".join(said) + "\n\n" + "\n".join(sections)


def write(quest_root: Path | str, limit: int | None, parts: dict[str, str]) -> bool:
    """Write :data:`FILE` from ``parts`` (or remove it when none has text); a file that already says exactly this is
    left as it is. Returns whether a file is there."""
    path = Path(quest_root) / FILE
    body = text(limit, parts)
    if not body:
        path.unlink(missing_ok=True)
        return False
    try:
        if path.read_text(encoding="utf-8") == body:
            return True
    except (OSError, UnicodeDecodeError):
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return True


def further_pointer(n: int) -> str:
    """The line the References keep when the Further reading moved here: how many web pages it lists and where
    (``""`` for none)."""
    if n <= 0:
        return ""
    return (f"Further reading: {n} web page{'s' if n != 1 else ''} the text does not cite "
            f"{'are' if n != 1 else 'is'} listed in the supplement (`{SHOWN}`).")


#: The line :func:`further_pointer` writes, whatever its count.
FURTHER_POINTER = re.compile(r"^Further reading: \d+ web pages? the text does not cite (?:are|is) listed in the "
                             r"supplement \(`" + re.escape(SHOWN) + r"`\)\.[ \t]*$", re.M)


def table_pointer() -> str:
    """What the paper keeps where its tables of every setting were."""
    return (f"FI's tables of every setting the study computed, each value with its interval and reference, are in the "
            f"supplement (`{SHOWN}`; every number also in `paper/results_by_setting.csv`).")
