"""The code reading's cache of a part's answer (core/engine.py ``_code_review_gate``, ``core/code_review.py``
``load_reading_cache``): a follow-up ask after a reply that said nothing does not overwrite the cache, and a part answered
only in part is not kept, so its next reading asks again. Fake models, a neutral topic (a mass on a damped spring)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import code_review as cr
from tests.test_code_review import WITH_PLAN, _engine
from tests.test_code_review_parts import EXP, SIM, _PartReader, _write

pytestmark = pytest.mark.usefixtures("this_python")


class _SaysNothing(_PartReader):
    """Every reply names no requirement, no problem and nothing it checked (a reply that says nothing)."""

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if not prompt.lstrip().startswith("# Code Against the Plan"):
            return ""
        self.prompts.append(prompt)
        return json.dumps({"requirements": [], "findings": [], "checked": [], "summary": "s"})


class _LeavesOneOut(_PartReader):
    """Answers every requirement but the first it is given, in every reply."""

    async def chat(self, messages, **kw):  # noqa: ANN001
        reply = await super().chat(messages, **kw)
        if not reply:
            return reply
        data = json.loads(reply)
        data["requirements"] = data["requirements"][1:]
        return json.dumps(data)


@pytest.mark.asyncio
async def test_a_follow_up_ask_of_a_part_given_no_requirement_does_not_break_the_reading(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A request given no requirement to judge (a plan part that defines none of them) is read for problems of logic;
    # a reply that then says nothing is asked once more. Old code: the follow-up reused the name of the reading's cache,
    # so keeping the part's answer raised a TypeError and the reading was recorded as a failed call.
    engine = _engine(tmp_path)
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": EXP})
    monkeypatch.setattr(cr, "asked_ids", lambda reqs, plans: [[] for _ in plans])
    reader = _SaysNothing()
    engine._client = reader
    await engine._code_review_gate(WITH_PLAN)
    record = cr.load(engine.quest_root)
    assert len(reader.prompts) == 2, "the reply that said nothing is asked once more"
    assert "the call failed" not in str(record.get("error") or ""), record.get("error")
    assert len(cr.load_reading_cache(engine.quest_root)) == 1, "the part's answer is kept for the next reading"


@pytest.mark.asyncio
async def test_a_part_answered_only_in_part_is_asked_again_at_the_next_reading(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    code = engine.quest_root / "code"
    _write(code, {"simulate.py": SIM, "experiment.py": EXP})
    monkeypatch.setattr(cr, "_MAX_CODE_CHARS", 600)
    assert len(cr.source_parts(code)) == 2
    reader = _LeavesOneOut()
    engine._client = reader
    await engine._code_review_gate(WITH_PLAN)
    assert reader.prompts, "the code was read"
    # only experiment.py changes; simulate.py is the same text, but its earlier answer left a requirement out
    (code / "experiment.py").write_text(EXP + "other = 3\n", encoding="utf-8")
    reader.prompts.clear()
    await engine._code_review_gate(WITH_PLAN)
    assert any("### simulate.py" in p for p in reader.prompts), "the part whose answer was not whole is asked again"
