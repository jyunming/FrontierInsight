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
async def test_a_part_nobody_changed_is_not_read_again_and_a_changed_one_is(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    engine, reader = await _first_reading(tmp_path, monkeypatch)
    code = engine.quest_root / "code"
    before = cr.load(engine.quest_root)["reader_calls"]
    # only experiment.py changes (a new line at its end, so no line of simulate.py moves)
    (code / "experiment.py").write_text(CALLER + "other = 3\n", encoding="utf-8")
    reader.prompts.clear()
    await engine._code_review_gate(WITH_PLAN)
    assert len(reader.prompts) == 1, "only the part that changed is asked"
    assert "### experiment.py" in reader.prompts[0]
    record = cr.load(engine.quest_root)
    assert record["reader_calls"] == before + 1
    assert record["parts_reused"] == 1 and [r.get("reused") for r in record["requests"]] == [True, None]
    # the reused part still counts: every requirement is judged from both parts' answers
    assert record["problems"] == [] and record["verdicts"]


@pytest.mark.asyncio
async def test_a_part_is_reused_when_only_a_function_it_calls_in_another_part_changed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, reader = await _first_reading(tmp_path, monkeypatch)
    code = engine.quest_root / "code"
    # run_cell (in simulate.py) changes; experiment.py, which calls it, is the same text: the reader is shown only its part
    (code / "simulate.py").write_text(SIM.replace("1.0", "2.0"), encoding="utf-8")
    reader.prompts.clear()
    await engine._code_review_gate(WITH_PLAN)
    assert len(reader.prompts) == 1 and "### simulate.py" in reader.prompts[0]
    assert cr.load(engine.quest_root)["parts_reused"] == 1


@pytest.mark.asyncio
async def test_a_changed_settings_file_is_read_again_only_by_the_request_that_shows_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": CALLER, "study.json": '{"a": 1}\n'})
    monkeypatch.setattr(cr, "_MAX_CODE_CHARS", 600)
    reader = _PartReader()
    engine._client = reader
    await engine._code_review_gate(WITH_PLAN)
    n = len(reader.prompts)
    (engine.quest_root / "code" / "study.json").write_text('{"a": 2}\n', encoding="utf-8")
    reader.prompts.clear()
    await engine._code_review_gate(WITH_PLAN)
    assert n >= 2 and len(reader.prompts) == 1, "only the request that shows the settings file is a new question"


@pytest.mark.asyncio
async def test_a_part_is_kept_when_only_something_it_does_not_use_changed(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    engine, reader = await _first_reading(tmp_path, monkeypatch)
    code = engine.quest_root / "code"
    (code / "simulate.py").write_text(SIM + "def unrelated_helper(z):\n    return z\n", encoding="utf-8")
    reader.prompts.clear()
    await engine._code_review_gate(WITH_PLAN)
    assert len(reader.prompts) == 1 and "### simulate.py" in reader.prompts[0]


@pytest.mark.asyncio
async def test_a_changed_plan_is_a_new_question_so_nothing_is_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, reader = await _first_reading(tmp_path, monkeypatch)
    code = engine.quest_root / "code"
    (code / "experiment.py").write_text(CALLER + "other = 3\n", encoding="utf-8")
    plan = {**WITH_PLAN, "design": {**WITH_PLAN["design"], "protocol": {
        **WITH_PLAN["design"]["protocol"], "grid": {"c": [0.2, 0.5, 1.0, 2.0]}}}}
    reader.prompts.clear()
    await engine._code_review_gate(plan)
    assert len(reader.prompts) == 2


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
async def test_the_cap_still_stops_a_reading_it_cannot_cover(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, reader = await _first_reading(tmp_path, monkeypatch)
    (engine.quest_root / "code" / "experiment.py").write_text(CALLER + "other = 3\n", encoding="utf-8")
    # 2 requests are used; the one changed part needs 1 more: 2 cannot cover it
    monkeypatch.setattr(cr, "MAX_READER_CALLS", 2)
    reader.prompts.clear()
    await engine._code_review_gate(WITH_PLAN)
    record = cr.load(engine.quest_root)
    assert reader.prompts == [] and "not read in full" in record["error"] and "needs 1 more" in record["error"]
    assert record["parts_not_read"] and "experiment.py" in record["parts_not_read"][0]


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


@pytest.mark.asyncio
async def test_a_part_is_asked_again_when_the_findings_of_the_version_before_it_is_shown_changed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The earlier findings are part of the text the reader is sent, so a new one is a new question (the old key left them out)."""
    engine, reader = await _first_reading(tmp_path, monkeypatch)
    (engine.quest_root / "code" / "experiment.py").write_text(CALLER + "other = 3"+chr(92)+"n", encoding="utf-8")
    monkeypatch.setattr(cr, "earlier_block", lambda previous: "- L1 an earlier finding about simulate.py line 2")
    reader.prompts.clear()
    await engine._code_review_gate(WITH_PLAN)
    assert len(reader.prompts) == 2 and all("an earlier finding" in p for p in reader.prompts)
