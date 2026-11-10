"""Citations by the label of FI's prior-work block.

The writer cites a source with ``{{cite:3}}`` (``{{cite:W2}}`` for a web page, ``{{cite:2, 5}}`` for several), where the
label is the one FI gives each entry of the prior-work block in the writer's input. FI renders the placeholders to the
paper's numbered citations and numbers them in the order of first use (``core.engine._renumber_citations``). Nothing is
guessed from the characters around a bracket: ``Smith[2]`` and ``cell_values[0]`` are the writer's own text.

A label no source has is never rendered as a number: the citation is removed and the label is listed in the quality
report (``.fi/citations_removed.json``, read by ``core.quality_report``).

A paper written before this change has no placeholder; its bracketed citations are still read by the old rules
(``_renumber_citations``, legacy branch), which are kept only so an old record can be rendered again.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

#: ``{{cite:3}}``, ``{{ cite : W2 }}``, ``{{cite:2,5}}``, ``{{cite:2-4}}``.
CITE = re.compile(r"\{\{\s*cite\s*:\s*([^{}\n]{1,80}?)\s*\}\}", re.IGNORECASE)
_VALID = re.compile(r"W?\d+\Z")
REMOVED_RECORD = ".fi/citations_removed.json"


def split_labels(inside: str, expand: "callable") -> tuple[list[str], list[str]]:
    """``(valid labels, text that is no label)`` of one placeholder's content. ``expand`` spells ranges out."""
    labels, bad = [], []
    for part in re.split(r"[,;]", inside):
        part = part.strip()
        if not part:
            continue
        if re.fullmatch(r"W?\d+\s*[–-]\s*W?\d+", part, re.I):
            labels += expand(part)
        elif _VALID.match(part.upper()):
            labels.append(part.upper())
        else:
            bad.append(part)
    return labels, bad


def has_placeholder(body: str, spans: list[tuple[int, int]]) -> bool:
    """Whether ``body`` cites with a placeholder outside inline code and math."""
    return any(not any(s <= m.start() < e for s, e in spans) for m in CITE.finditer(body))


def write_removed(root: Path | str, removed: list[str]) -> None:
    """Record the citations taken out of the new draft (none: the earlier draft's record goes)."""
    path = Path(root) / REMOVED_RECORD
    if not removed:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"removed": removed}, indent=1), encoding="utf-8")
