"""The study cards: a few different studies a topic could mean, what a person picks, and what the plan left out.

Pure functions and one small record file; no model calls and no engine import. The ``scope`` node in
``core/engine.py`` asks the model for the cards, and this module checks the reply against FI's own record of the
literature it read, builds the question, reads the answer, writes ``.fi/scope.json`` and compares a plan with the
chosen card. The comparison is by names (a reading), so it is disclosed and never stops a quest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

STUDY_TYPES = ("measure", "find_best_design")
RECORD = ".fi/scope.json"
ANSWER = ".fi/scope_answer.json"

_MAX_CARDS = 4
_MIN_CARDS = 3
_TYPE_ALIASES = {"compare": "measure"}


def _key(text: Any) -> str:
    """Lower-cased text with runs of spaces joined; underscores count as spaces."""
    return " ".join(str(text).lower().replace("_", " ").split())


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(x).strip() for x in value if str(x).strip()]


def _dicts(value: Any) -> list[dict]:
    """The mappings of a list; anything that is not a list holds none."""
    return [x for x in value if isinstance(x, dict)] if isinstance(value, (list, tuple)) else []


def normalize(reply: Any, source_ids: set[str]) -> tuple[list[dict], list[str]]:
    """The model's reply as checked cards, plus one sentence for each literature line FI removed."""
    if isinstance(reply, str):
        try:
            reply = json.loads(reply)
        except ValueError:
            return [], []
    raw = reply.get("cards") if isinstance(reply, dict) else reply
    if not isinstance(raw, list):
        return [], []
    cards: list[dict] = []
    removed: list[str] = []
    seen: set[tuple[str, frozenset[str]]] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        kind = str(item.get("study_type") or "").strip().lower()
        kind = _TYPE_ALIASES.get(kind, kind)
        if not question or kind not in STUDY_TYPES:
            continue
        results = _strings(item.get("main_results"))
        fingerprint = (kind, frozenset(" ".join(x.lower().split()) for x in results))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        literature: list[dict[str, str]] = []
        for line in _dicts(item.get("literature")):
            if not isinstance(line, dict):
                continue
            says, source = str(line.get("says") or "").strip(), str(line.get("source") or "").strip()
            if not says:
                continue
            if source in source_ids:
                literature.append({"says": says, "source": source})
            else:
                removed.append(f"A literature line on the study \"{question}\" was removed because its source "
                               f"({source or 'none given'}) is not one FI read: \"{says}\"")
        card: dict[str, Any] = {
            "question": question, "study_type": kind, "main_results": results,
            "compared_settings": _strings(item.get("compared_settings")), "literature": literature,
            "run_time": str(item.get("run_time") or "").strip(),
        }
        if item.get("widest") is True:
            card["widest"] = True
        cards.append(card)
        if len(cards) == _MAX_CARDS:
            break
    return cards, removed


def distinct_enough(cards: list[dict]) -> bool:
    return len(cards) >= _MIN_CARDS


def has_widest(cards: list[dict]) -> bool:
    return any(c.get("widest") is True or c.get("study_type") == "find_best_design" for c in cards)


def _block(number: int, card: dict) -> str:
    kind = "find the best design" if card.get("study_type") == "find_best_design" else "measure and compare"
    lines = [f"{number}. {card.get('question', '')}", f"   Kind: {kind}"]
    if card.get("main_results"):
        lines.append("   Reports: " + "; ".join(card["main_results"]))
    if card.get("compared_settings"):
        lines.append("   Varies: " + "; ".join(card["compared_settings"]))
    for line in card.get("literature") or []:
        lines.append(f"   Literature: {line.get('says', '')} [{line.get('source', '')}]")
    if card.get("run_time"):
        lines.append(f"   Run time: {card['run_time']}")
    return "\n".join(lines)


def question(cards: list[dict]) -> dict:
    """The question in the shape the clarify answer channel carries."""
    head = "Pick the study to run (a number), or write the study you want in a sentence:"
    text = head + "\n\n" + "\n\n".join(_block(i, c) for i, c in enumerate(cards, 1))
    return {"study": {"question": text, "default": "", "cards": cards}}


def _from_number(number: int, cards: list[dict]) -> dict | None:
    return cards[number - 1] if 1 <= number <= len(cards) else None


def read_answer(answer: Any, cards: list[dict]) -> dict | None:
    """The card the answer names, a card written by the person, or None when nobody chose."""
    if isinstance(answer, dict) and "study" in answer:
        answer = answer["study"]
    if isinstance(answer, bool) or answer is None:
        return None
    if isinstance(answer, int):
        return _from_number(answer, cards)
    if isinstance(answer, dict):
        text = str(answer.get("question") or "").strip()
        if not text:
            return None
        return {"question": text, "study_type": answer.get("study_type") or None,
                "main_results": _strings(answer.get("main_results")),
                "compared_settings": _strings(answer.get("compared_settings")),
                "literature": _dicts(answer.get("literature")),
                "run_time": str(answer.get("run_time") or "").strip()}
    if not isinstance(answer, str):
        return None
    text = answer.strip()
    if not text:
        return None
    if text.isdigit():
        return _from_number(int(text), cards)
    if text[0] in "eE" and text[1:].strip().isdigit():
        return _from_number(int(text[1:].strip()), cards)  # "e 2" whose edit never came: card 2 as offered
    if len(text) == 1:
        return None  # a single letter names no study
    return {"question": text, "study_type": None, "main_results": [], "compared_settings": [],
            "literature": [], "run_time": "", "written_by_person": True}


def read_record(quest_root: Path) -> dict | None:
    try:
        data = json.loads((Path(quest_root) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_record(quest_root: Path, cards: list[dict], chosen: dict | None, removed: list[str], *,
                 widest: bool) -> None:
    """Write ``.fi/scope.json``; an earlier choice in the file moves to ``history``."""
    old = read_record(quest_root)
    history = list(old.get("history") or []) if old else []
    if old and old.get("chosen"):
        history.append(old["chosen"])
    path = Path(quest_root) / RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "cards": cards, "chosen": chosen, "chosen_by": "person" if chosen else "nobody",
        "removed": removed, "widest_offered": widest, "history": history,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def plan_note(card: dict) -> str:
    """The text the plan prompt gets when a person chose a study."""
    lines = ["The person chose this study to run. Write the plan for it.", f"Study: {card.get('question', '')}"]
    if card.get("study_type"):
        lines.append(f"Kind: {card['study_type']}")
    if card.get("main_results"):
        lines.append("Main results: " + "; ".join(card["main_results"]))
    if card.get("compared_settings"):
        lines.append("Compared settings: " + "; ".join(card["compared_settings"]))
    lines.append("Cover each main result and each compared setting in the plan, or say in the plan why one is "
                 "left out.")
    return "\n".join(lines)


def _plan_names(design: dict) -> tuple[set[str], set[str]]:
    parts = [design] + ([design["protocol"]] if isinstance(design.get("protocol"), dict) else [])
    metrics: set[str] = set()
    settings: set[str] = set()
    for part in parts:
        for m in part.get("metrics") or []:
            if isinstance(m, dict):
                metrics.update(_key(m[k]) for k in ("id", "label") if m.get(k))
        if isinstance(part.get("grid"), dict):
            settings.update(_key(k) for k in part["grid"])
        block = part.get("optimisation")
        for v in (block.get("design_variables") or []) if isinstance(block, dict) else []:
            if isinstance(v, dict) and v.get("name"):
                settings.add(_key(v["name"]))
    for part in parts:
        # the reader's names the protocol gives a setting or a result ("Cup material $m$ (-)"), with and without the
        # symbol and unit, count as its names too
        labels = part.get("labels")
        for code_name, shown in (labels.items() if isinstance(labels, dict) else ()):
            names = {_key(shown), _key(re.split(r"[$(\[]", str(shown), maxsplit=1)[0])} - {""}
            for found in (metrics, settings):
                if _key(code_name) in found:
                    found.update(names)
    return metrics, settings


def gaps(card: dict, design: dict) -> list[str]:
    """One sentence for each result or setting the card names that the plan has no metric, axis or variable for."""
    metrics, settings = _plan_names(design if isinstance(design, dict) else {})
    out = [f"The chosen study reports \"{r}\", and the plan has no result by that name."
           for r in card.get("main_results") or [] if _key(r) not in metrics]
    out += [f"The chosen study compares \"{s}\", and the plan does not vary a setting by that name."
            for s in card.get("compared_settings") or [] if _key(s) not in settings]
    return out


def nobody_line() -> str:
    return "Nobody picked the study to run; FI planned from the topic alone. The studies it could also have been:"


def plan_section(record: dict | None) -> str:
    """plan.md's section on the study to run: the card a person picked, or that nobody picked one and the cards that
    were offered. ``""`` when no cards were offered."""
    if not record or not record.get("cards"):
        return ""
    chosen = record.get("chosen")
    if isinstance(chosen, dict):
        body = "A person picked this study to run: " + str(chosen.get("question", "")) + "\n\n" + "\n".join(
            line.strip() for line in _block(1, chosen).split("\n")[1:])
    else:
        body = nobody_line() + "\n\n" + "\n".join(f"- {c.get('question', '')}" for c in record["cards"])
    return "## The study to run\n\n" + body + "\n"
