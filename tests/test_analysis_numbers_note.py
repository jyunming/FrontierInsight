"""The writer is told which numbers of the analysis nothing in the run accounts for, before the paper is written
(replay of suite r9k first_order_kinetics: the analysis gave the OLS Ea SD as 1,920 and the OLS ln A SD as 0.745 where
the run computed 1,849.0 and 0.7057; the paper printed the 1,920)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from tests.test_disclosure import _engine

FIXTURE = Path(__file__).parent / "fixtures" / "provenance_r9k"


def _state(analysis: Any) -> dict[str, Any]:
    return {"topic": "t", "analysis": analysis, "iteration": 0,
            "exec_result": {"returncode": 0, "duration_s": 111.29700000002049},
            "result_json": json.loads((FIXTURE / "result_json.json").read_text(encoding="utf-8"))}


def _analysis() -> Any:
    return json.loads((FIXTURE / "analysis.json").read_text(encoding="utf-8"))


def test_r9k_s_wrong_numbers_are_named_before_writing(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    note = eng._analysis_numbers_note(_state(_analysis()))  # type: ignore[arg-type]
    assert "1,920" in note and "0.745" in note
    assert "Do not print them" in note
    assert "111.3" not in note  # the run time is FI's own record
    for right in ("1849", "709", "0.276"):  # values the run computed are not listed
        assert f" {right} (" not in note


def test_nothing_to_say_when_every_number_traces(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    assert eng._analysis_numbers_note(_state({"summary": "The OLS SD fell to 709 J/mol."})) == ""  # type: ignore[arg-type]
    assert eng._analysis_numbers_note(_state({})) == ""  # type: ignore[arg-type]


def test_the_writer_gets_the_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    async def fake_chat(self: Any, messages: Any, **kw: Any) -> str:
        seen.append(messages[-1]["content"] if isinstance(messages, list) else str(messages))
        return "# T\n\n## Results\nThe OLS SD fell to 709 J/mol.\n\n## Limitations\nNone.\n"

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    eng = _engine(tmp_path)
    asyncio.run(eng._node_write({**_state(_analysis()), "title": "t"}))  # type: ignore[arg-type]
    assert any("These match nothing it computed: 1,920" in p for p in seen)
