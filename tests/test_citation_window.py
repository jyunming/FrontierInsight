"""The window of a cited source the citation judge is shown opens with a commission's verified passage for that source
(core/engine_write.py, ``Engine._node_citation_check``); without a commission it is what it always was."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core.engine import _CITATION_SOURCE_CHARS
from tests.test_citation_check_node import _engine

PASSAGE = "| 0.30 | 0.25 | 0.21 | 0.17 |"
FILLER = "Tanks drain with a constant that depends on the outlet and the fill level, as the note explains. " * 90
TEXT = FILLER + "\n\n" + PASSAGE + ".\n\n" + FILLER
SENTENCE = "Tanks drain with a constant that depends on the outlet and the fill level [1]."


def _commission_file(folder: Path) -> Path:
    path = folder / "commission.json"
    path.write_text(json.dumps({
        "question": "How fast does a tank drain?", "scope": "one outlet", "limits": {"hours": 1, "model_calls": 10},
        "known_values": [{"name": "drain constant", "value": 0.25, "tolerance": 0.01, "case": "one outlet", "passage": PASSAGE,
                          "source": {"title": "A made-up tank note", "doi": "10.0000/tank.1", "file": "tank_note.md"}}],
        "success": [], "failure": [], "doubts": []}), encoding="utf-8")
    return path


def _window(tmp_path: Path, *, commissioned: bool) -> str:
    eng = _engine(tmp_path)
    if commissioned:
        eng.config.commission = _commission_file(tmp_path)  # type: ignore[assignment]
    paper = tmp_path / "paper" / "paper.md"
    paper.parent.mkdir(parents=True)
    paper.write_text(f"# A tank\n\n## Introduction\n\n{SENTENCE}\n\n## References\n\n1. Anon. A made-up tank note. 2020.\n",
                     encoding="utf-8")
    prompts: list[str] = []

    async def chat(prompt: str, *, node: str = "") -> str:
        prompts.append(prompt)
        return json.dumps({"verdicts": [], "claims": []})

    eng._chat = chat  # type: ignore[assignment,method-assign]
    literature = [{"content": TEXT, "metadata": {
        "title": "A made-up tank note", "doi": "10.0000/tank.1", "authors": ["Anon"], "year": 2020,
        "source": "user_supplied", "path": str(tmp_path / "inputs" / "papers" / "tank_note.md"), "full_text": True}}]
    state = {"topic": "t", "paper_md": str(paper), "literature": literature,
             "review": {"verdict": "accept", "must_flag_hits": []}, "iteration": 2}
    asyncio.run(eng._node_citation_check(state))  # type: ignore[arg-type]
    judged = [p for p in prompts if "**Citation Reader**" in p]
    assert judged, prompts
    return judged[0]


def test_a_commissions_passage_opens_the_window_of_its_source(tmp_path: Path) -> None:
    prompt = _window(tmp_path, commissioned=True)
    text = prompt.split("Text:\n", 1)[1]
    assert text.startswith(PASSAGE)
    assert PASSAGE not in TEXT[:_CITATION_SOURCE_CHARS]  # the passage sat beyond what the window used to start with


def test_without_a_commission_the_window_is_the_same_as_before(tmp_path: Path) -> None:
    from core.engine import _format_lit_excerpt

    prompt = _window(tmp_path, commissioned=False)
    old = _format_lit_excerpt(TEXT, "A made-up tank note", query=SENTENCE, budget=_CITATION_SOURCE_CHARS).strip()
    assert ("Text:\n" + old) in prompt
