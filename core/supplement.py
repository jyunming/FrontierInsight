"""The supplement a paper over its page limit sends FI's own parts to.

A paper rendered over the topic's page limit loses FI's own text first, in a fixed order and without a model call, one
part at a time until it fits (``Engine._fit_with_supplement``): its tables of every setting (:data:`TABLES`), then its
Limitations (:data:`LIMITATIONS`: the section keeps one line that says how many things FI recorded, names each part of
the topic the study did not do, and points here; the paper's own paragraphs of that section go with it, see
:func:`take_writer_limitations`), then its Further reading (:data:`FURTHER_READING`). The References stay. What moved
is written, unchanged, to ``paper/supplement.md`` (:data:`FILE`), which ships with the paper.

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
_TITLES = {TABLES: "Results for every setting", LIMITATIONS: "Limitations", FURTHER_READING: "Further reading",
           SENTENCES: "Sentences moved for length"}
#: How much of the text on each side of a moved sentence its record keeps, to find its place again (never past the
#: paragraph it stood in).
_ANCHOR = 60
BLANK = chr(10) * 2
#: What the run log calls each part.
NAMES = {TABLES: "tables of every setting", LIMITATIONS: "Limitations", FURTHER_READING: "Further reading"}


def read(quest_root: Path | str) -> dict[str, Any]:
    """The record: ``{"moved": [...], "further_reading": [labels], "limit": n}``; ``{}`` when nothing moved or it cannot
    be read."""
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
    """The paper's own paragraphs of its Limitations section that moved with FI's Limitations (``""`` when none did):
    the supplement shows them before FI's list."""
    rec = read(quest_root)
    return str(rec.get("limitations_writer") or "") if LIMITATIONS in (rec.get("moved") or []) else ""


def writer_section(quest_root: Path | str) -> tuple[str, str] | None:
    """``(before, after)``: the paper's Limitations section as it was before the writer's paragraphs moved and as the move
    left it (:func:`section_body`), so the draft they came from gets them back exactly; ``None`` when none moved."""
    rec = read(quest_root)
    if LIMITATIONS not in (rec.get("moved") or []) or not rec.get("limitations_writer"):
        return None
    return str(rec.get("limitations_before") or ""), str(rec.get("limitations_after") or "")


_MARKED = re.compile(r"<!-- fi:fact-([\w-]+) -->.*?<!-- /fi:fact-\1 -->", re.S)
_HEADING = re.compile(r"^#{1,6}[ \t]+(.*)$")
_CITES = re.compile(r"\[W?\d")
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


def _paragraphs(lines: list[str]) -> list[list[str]]:
    out: list[list[str]] = [[]]
    for ln in lines:
        if ln.strip():
            out[-1].append(ln)
        elif out[-1]:
            out.append([])
    return [p for p in out if p]


def _name_of(block: str) -> str:
    m = _MARKED.match(block)
    return m.group(1) if m else ""


def _body(lines: list[str], blocks: list[str], head: int, end: int) -> str:
    """The section's text below its heading, FI's blocks as their names only (FI writes them again from its records)."""
    text = "\n".join(lines[head + 1:end]).strip()
    return _TOKEN.sub(lambda m: "\x00" + _name_of(blocks[int(m.group(1))]) + "\x00", text)


def section_body(text: str) -> str | None:
    """The paper's Limitations section below its heading, FI's blocks as their names only; ``None`` with no such
    section (see :func:`_limitations_section`)."""
    masked, blocks = _masked(text)
    lines = masked.split("\n")
    span = _limitations_section(lines, blocks)
    return None if span is None else _body(lines, blocks, *span)


def _movable(par: list[str]) -> bool:
    """A paragraph of the writer's own prose: no FI block or mark, no figure, table, code or placeholder, and no
    citation (moving one would leave a source in the References that the paper no longer cites)."""
    body = "\n".join(par)
    first = par[0].lstrip()
    return not (_TOKEN.search(body) or "<!--" in body or "![" in body or "{{" in body or _CITES.search(body)
                or first.startswith(("|", "```", "~~~", "<", "$$", "\\[")))


def take_writer_limitations(text: str) -> tuple[str, str]:
    """``text`` without the writer's own paragraphs of its Limitations section, and those paragraphs as they were
    (``""``, and ``text`` unchanged, when there are none or the section is not the paper's Limitations). The heading,
    FI's block, figures, tables and any paragraph that cites a source stay."""
    masked, blocks = _masked(text)
    lines = masked.split("\n")
    span = _limitations_section(lines, blocks)
    if span is None:
        return text, ""
    head, end = span
    pars = _paragraphs(lines[head + 1:end])
    moved = [p for p in pars if _movable(p)]
    if not moved:
        return text, ""
    kept = "\n\n".join("\n".join(p) for p in pars if not _movable(p))
    new = [*lines[:head + 1], "", *([kept, ""] if kept else []), *lines[end:]]
    return _unmasked("\n".join(new), blocks), "\n\n".join("\n".join(p) for p in moved)


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
    return _unmasked("\n".join([*lines[:head + 1], "", body, *([""] if end < len(lines) else []), *lines[end:]]),
                     blocks)


def save(quest_root: Path | str, parts: list[str], *, limit: int, further: list[str] | None = None,
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
    data: dict[str, Any] = {"moved": parts, "limit": int(limit)}
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


def sentence_count(quest_root: Path | str) -> int:
    """How many sentences the page trim moved to the supplement."""
    return sum(len(r.get("sentences") or [None]) for r in sentences(quest_root))


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


def moved_sentences_line(n: int) -> str:
    """The line the paper's Limitations carry when the trim moved ``n`` sentences (``""`` for none)."""
    if n <= 0:
        return ""
    return (f"FI moved {n} sentence{'s' if n != 1 else ''} of the paper's background and discussion to the supplement "
            f"(`{SHOWN}`) to keep the paper within its page limit; {'each stands' if n != 1 else 'it stands'} there as "
            "written.")


def text(limit: int | None, parts: dict[str, str]) -> str:
    """The supplement's Markdown: one section per moved part with text, in :data:`ORDER`, then the moved sentences
    (:data:`SENTENCES`); ``""`` with none."""
    sections = [f"## {_TITLES[p]}\n\n{parts[p].strip()}\n" for p in (*ORDER, SENTENCES) if (parts.get(p) or "").strip()]
    if not sections:
        return ""
    within = f" within its page limit of {limit} pages" if limit else " within its page limit"
    return (f"# Supplement\n\nFI moved these parts of the paper here to keep the paper{within}; nothing in them was "
            "changed.\n\n" + "\n".join(sections))


def write(quest_root: Path | str, limit: int | None, parts: dict[str, str]) -> bool:
    """Write :data:`FILE` from ``parts`` (or remove it when none has text). Returns whether a file is there."""
    path = Path(quest_root) / FILE
    body = text(limit, parts)
    if not body:
        path.unlink(missing_ok=True)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return True


def table_pointer() -> str:
    """What the paper keeps where its tables of every setting were."""
    return (f"FI's tables of every setting the study computed, each value with its interval and reference, are in the "
            f"supplement (`{SHOWN}`; every number also in `paper/results_by_setting.csv`).")


def limitations_pointer(n_items: int, topic_parts: list[str], *, discussion: bool = False, known_answers: int = 0,
                        none_judged: bool = False, moved_sentences: int = 0) -> str:
    """The one line the Limitations keep: how many things FI recorded, each part of the topic the study did not do by
    name and how many checks of the code against a known answer did not confirm the results (``known_answers``: a reader
    is told in the paper itself what it leaves out and what is not validated), and where they all are; ``discussion``: the
    paper's own paragraphs of that section are there too; ``none_judged``: no such check could be judged at all;
    ``moved_sentences``: how many sentences of the paper's text the page trim moved there too."""
    if n_items <= 0:
        return moved_sentences_line(moved_sentences)
    line = f"FI recorded {n_items} limitation{'s' if n_items != 1 else ''} of this study"
    if topic_parts:
        shown = "; ".join(topic_parts)  # every one by name: a reader is told in the paper what the study leaves out
        k = len(topic_parts)
        line += f", among them {k} part{'s' if k != 1 else ''} of the topic it did not do ({shown})"
    if known_answers > 0:
        checks = (f"{known_answers} check{'s' if known_answers != 1 else ''} of the code against a known answer that did "
                  "not confirm the results")
        line += f" and {checks}" if topic_parts else f", among them {checks}"
    if none_judged:
        line += "; no check of the code against a known answer could be judged"
    where = f"the supplement (`{SHOWN}`)" + (", with the paper's own discussion of them" if discussion else "")
    also = (f" The supplement also holds {moved_sentences} sentence{'s' if moved_sentences != 1 else ''} of the paper's "
            "text, moved there for length.") if moved_sentences > 0 else ""
    return line + f"; each is stated in full in {where}.{also}"
