"""A part of the code that was read before and has not changed is not read again (core/code_review.py: ``request_key``,
``part_dependencies``; the reading in core/engine.py). The reader requests of a quest are bounded, so a re-read of what
nobody changed used up the requests the version that runs needed. Fake models, neutral topics (a mass on a damped
spring)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import code_review as cr
from tests.test_code_review import WITH_PLAN, _engine
from tests.test_code_review_parts import EXP, SIM, _PartReader, _write

pytestmark = pytest.mark.usefixtures("this_python")

#: simulate.py defines run_cell; experiment.py calls it. They are two parts (each fits one request alone).
CALLER = EXP.replace("save('decay.png')", "save('decay.png')\nvalue = run_cell(1)")


async def _first_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, experiment: str = CALLER):
    engine = _engine(tmp_path)
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": experiment})
    monkeypatch.setattr(cr, "_MAX_CODE_CHARS", 600)
    assert len(cr.source_parts(engine.quest_root / "code")) == 2
    reader = _PartReader()
    engine._client = reader
    await engine._code_review_gate(WITH_PLAN)
    assert len(reader.prompts) == 2
    return engine, reader


@pytest.mark.asyncio
async def test_a_version_changed_after_the_one_reading_is_not_read_again(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    engine, reader = await _first_reading(tmp_path, monkeypatch)
    code = engine.quest_root / "code"
    before = cr.load(engine.quest_root)["reader_calls"]
    # a part changes after the reading (a new line at the end of experiment.py): the code is read once, so nothing is asked
    (code / "experiment.py").write_text(CALLER + "other = 3\n", encoding="utf-8")
    reader.prompts.clear()
    await engine._code_review_gate(WITH_PLAN)
    assert reader.prompts == [], "the one reading is spent: the changed part is not asked"
    record = cr.load(engine.quest_root)
    assert record["reader_calls"] == before and record["read_version"] != cr.version(code)


@pytest.mark.asyncio
async def test_a_reply_is_kept_under_the_text_sent_and_the_model_and_the_same_text_is_not_asked_again(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache holds the raw reply under the exact text sent: the same text to the same model is not asked again, and what is
    kept is read again with today's rules."""
    from core import reuse_key

    engine, reader = await _first_reading(tmp_path, monkeypatch)
    kept = cr.load_reading_cache(engine.quest_root)
    assert kept and all(isinstance(v, dict) and isinstance(v.get("reply"), str) for v in kept.values())
    assert all(len(k) == 64 for k in kept) and reuse_key.answer_key("a", "m") != reuse_key.answer_key("a", "n")
    assert reuse_key.answer_key("a", "m") != reuse_key.answer_key("b", "m")
    reader.prompts.clear()
    await engine._code_review_gate(WITH_PLAN)
    assert reader.prompts == [], "the same text to the same model is not asked again"


@pytest.mark.asyncio
async def test_a_part_to_ask_reserves_one_request_not_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": CALLER})
    monkeypatch.setattr(cr, "_MAX_CODE_CHARS", 600)
    monkeypatch.setattr(cr, "MAX_READER_CALLS", 2)  # two parts: one request each fits; two each (4) did not
    reader = _PartReader()
    engine._client = reader
    await engine._code_review_gate(WITH_PLAN)
    assert len(reader.prompts) == 2 and cr.load(engine.quest_root)["reader_calls"] == 2


@pytest.mark.asyncio
async def test_a_reading_that_cannot_fit_at_all_still_stops_and_names_every_part(tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": CALLER})
    monkeypatch.setattr(cr, "_MAX_CODE_CHARS", 600)
    monkeypatch.setattr(cr, "MAX_READER_CALLS", 1)
    reader = _PartReader()
    engine._client = reader
    await engine._code_review_gate(WITH_PLAN)
    record = cr.load(engine.quest_root)
    assert reader.prompts == [] and "needs 2 more" in record["error"] and len(record["parts_not_read"]) == 2


class _FailsOnExperiment(_PartReader):
    async def chat(self, messages, **kw):  # noqa: ANN001
        if "### experiment.py" in messages[-1]["content"] and messages[-1]["content"].lstrip().startswith("# Code Against"):
            raise RuntimeError("the provider failed")
        return await super().chat(messages, **kw)


@pytest.mark.asyncio
async def test_a_failed_request_with_no_request_left_stops_the_reading_partway_and_names_the_parts_not_read(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": CALLER})
    monkeypatch.setattr(cr, "_MAX_CODE_CHARS", 600)
    monkeypatch.setattr(cr, "MAX_READER_CALLS", 2)
    engine._client = _FailsOnExperiment()
    await engine._code_review_gate(WITH_PLAN)
    record = cr.load(engine.quest_root)
    assert "not read in full" in record["error"] and not record.get("verdicts")
    assert len(record["parts_not_read"]) == 1 and "experiment.py" in record["parts_not_read"][0]
    assert len(cr.load_reading_cache(engine.quest_root)) == 1, "the part that was read is kept"


def test_the_quality_report_names_the_parts_of_the_code_that_were_not_read(tmp_path: Path) -> None:
    from core import quality_report as qr

    root = tmp_path / "q"
    (root / ".fi").mkdir(parents=True)
    (root / ".fi" / "code_review.json").write_text(json.dumps({
        "error": "the code was not read in full", "final_read": False,
        "parts_not_read": ["part 2 of 2 (experiment.py lines 1-24)"]}), encoding="utf-8")
    text = qr.render(qr.build(root))
    assert "not read by a second model." in text and "part 2 of 2 (experiment.py lines 1-24)" in text
