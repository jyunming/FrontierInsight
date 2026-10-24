"""Step cards: the non-equation steps of the plan's method, each described so one function can be written and tested for it.

``protocol.model.steps`` lists ``[{id: "S1", name, does, inputs: {name: unit or meaning}, output, uses: ["E1"], source,
example: {inputs, expected_formula, tolerance}, relations: [...]}]``. ``example`` and ``relations`` are optional; ``example``
follows the same rules as an equation's worked example and is judged by the same code
(:func:`core.equation_tests.example_rows`), so FI computes ``expected_formula`` itself and a model never supplies an
expected number. A plan with no ``steps`` is an old plan: no cards, no problems.

Pure: no model calls, no engine state.
"""

from __future__ import annotations

import re
from typing import Any

from . import equation_tests as _eq
from . import oracle_check as _oracle

#: Plain-language text for the plan directive (spliced into the plan request by the engine).
DIRECTIVE_TEXT = (
    "Every step of the method that is not an equation gets a step card in `protocol.model.steps`: "
    '`{"id": "S1", "name": "<short name>", "does": "<one sentence: what the step does>", "inputs": {"<name>": '
    '"<unit or meaning>"}, "output": "<what it returns, with unit>", "uses": ["E1"], "source": "<where the method is '
    'from>", "example": {"inputs": {"<name>": <number>}, "expected_formula": "<formula in those inputs only>", '
    '"tolerance": <number>}, "relations": ["<a property the output must have, e.g. it doubles when x doubles>"]}`. '
    "Give `example` for every step that computes a stated formula or a value with a known answer, and `relations` for "
    "every step with a property its output must have; leave both out only for a step that has neither, or write "
    '`"example": {"untestable": "<why>"}`. Take the inputs of the example at a case whose output is known by another route '
    "than the step's own form (a limit, a special value, a value a source prints), so that a slip in the step cannot "
    "pass. FI computes the expected value from your formula itself; never write a number you did not compute.")

_ID = re.compile(r"S[0-9]+")
_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _text(value: Any, limit: int = 400) -> str:
    return " ".join(str(value if value is not None else "").replace("`", "'").split())[:limit]


def _steps_of(protocol: Any) -> Any:
    model = protocol.get("model") if isinstance(protocol, dict) else None
    return model.get("steps") if isinstance(model, dict) else None


def parse(protocol: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """``(cards, problems)``. Each card: ``id``, ``name``, ``does``, ``inputs`` ({name: text}), ``output``, ``uses``,
    ``source``, ``relations``, ``example`` (as written, or ``None``), ``example_state`` (``none`` / ``ok`` / ``untestable``
    / ``unusable``), ``expected`` (the number FI computed, when ``ok``), ``untested_by_example`` (no example and no
    relations). Problems are plain sentences naming the card. An absent ``steps`` list is no cards and no problem."""
    raw = _steps_of(protocol)
    if raw is None or raw == "" or raw == []:
        return [], []
    if not isinstance(raw, list):
        return [], ["`model.steps` must be a list of step cards"]
    model = protocol.get("model")
    view = _oracle.model_view(model) or {}
    equation_ids = {_text(e.get("id")).lower() for e in view.get("equations", []) if isinstance(e, dict)}
    cards: list[dict[str, Any]] = []
    problems: list[str] = []
    seen: set[str] = set()
    pseudo: list[dict[str, Any]] = []
    for n, entry in enumerate(raw, 1):
        if not isinstance(entry, dict):
            problems.append(f"step card {n} must be a mapping, not {type(entry).__name__}")
            continue
        sid = _text(entry.get("id"), 20)
        label = sid or f"number {n}"
        if not _ID.fullmatch(sid):
            problems.append(f"step card {label} needs an id like S1, S2")
        elif sid.lower() in seen:
            problems.append(f"step id {sid} is used twice")
            continue
        else:
            seen.add(sid.lower())
        inputs_raw = entry.get("inputs")
        inputs = ({_text(k, 60): _text(v, 120) for k, v in inputs_raw.items()} if isinstance(inputs_raw, dict) else {})
        card: dict[str, Any] = {
            "id": sid, "name": _text(entry.get("name"), 100), "does": _text(entry.get("does")), "inputs": inputs,
            "output": _text(entry.get("output"), 200), "source": _text(entry.get("source"), 300),
            "uses": [], "relations": [], "example": None, "example_state": "none", "expected": None,
            "random": entry.get("random")}
        for field in ("name", "does", "output"):
            if not card[field]:
                problems.append(f"step {label} has no `{field}`")
        if not isinstance(inputs_raw, dict) or not inputs:
            problems.append(f"step {label} needs `inputs`, a mapping of each input's name to its unit or meaning")
        else:
            for k in inputs:
                if not _NAME.fullmatch(k):
                    problems.append(f"step {label}: the input name `{k}` is not a valid parameter name")
        uses = entry.get("uses")
        if uses not in (None, "") and not isinstance(uses, list):
            problems.append(f"step {label}: `uses` must be a list of equation ids")
        else:
            for u in uses or []:
                u = _text(u, 20)
                if u.lower() in equation_ids:
                    card["uses"].append(u)
                else:
                    problems.append(f"step {label} uses {u or '(blank)'}, which is not an equation of the model")
        rel = entry.get("relations")
        if rel not in (None, "") and not (isinstance(rel, list) and all(isinstance(r, str) for r in rel)):
            problems.append(f"step {label}: `relations` must be a list of plain sentences")
        else:
            card["relations"] = [_text(r) for r in rel or [] if _text(r)]
        example = entry.get("example")
        if example is not None and not (isinstance(example, str) and not example.strip()):
            card["example"] = example
            pseudo.append({"id": sid, "formula": card["does"] or "step", "role": "analyses", "example": example,
                           "random": entry.get("random")})
        cards.append(card)
    if pseudo:
        # The same judge as an equation's worked example, so FI computes `expected_formula` the same way.
        rows = {r["id"]: r for r in _eq.example_rows({"model": {"equations": pseudo}})}
        for card in cards:
            row = rows.get(card["id"])
            if row is None:
                continue
            card["example_state"] = row["state"] if row["state"] in ("ok", "untestable", "unusable") else (
                "untestable" if row["state"] == "unprotected" else "unusable")
            if row["state"] == "ok":
                card["expected"] = row["expected"]
                names = set(card["inputs"])
                given = set(row["inputs"])
                if names and given != names and not row["random"]:
                    card["example_state"] = "unusable"
                    problems.append(f"step {card['id']}: the example's inputs ({', '.join(sorted(given))}) must be the "
                                    f"card's inputs ({', '.join(sorted(names))})")
            elif row["state"] == "unusable":
                problems.append(f"step {card['id']}: its `example` cannot be used: {row['why']}")
    for card in cards:
        card["untested_by_example"] = card["example_state"] in ("none", "unusable") and not card["relations"]
    return cards, problems
