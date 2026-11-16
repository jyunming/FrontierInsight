"""Replay of a real quest whose results came from a code version nobody read. In tests/fixtures/replay/final_code_unread/
the three readings of a quest were spent (the third was the reserved one, on the version that first ran), then the run
broke the protocol's contract and ``execute_reflect`` repaired the analysis twice; each repaired version ran its study
with ``final_read: false``. No model is called: a stub reader answers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import code_review as cr
from tests.test_code_review import STATE, _engine, _Reader

FIXTURE = Path(__file__).parent / "fixtures" / "replay" / "final_code_unread"


def _spent_quest(tmp_path: Path):
    """A quest whose record is the real one's shape: three readings spent, the last on a version that has since changed."""
    engine = _engine(tmp_path)
    real = json.loads((FIXTURE / "code_review_euv.json").read_text(encoding="utf-8"))
    assert real["reviews_total"] == 3 and real["final_read"] is False and "spent" in real["skipped"]
    record = {k: real[k] for k in cr.CARRIED if k in real}
    record["read_version"] = "0123456789abcdef"  # the version the third reading read: not the code now
    cr.save(engine.quest_root, record)
    return engine


def test_the_evidence_is_what_the_run_log_says() -> None:
    log = (FIXTURE / "run_log_excerpt_euv.txt").read_text(encoding="utf-8")
    assert log.count("execute_reflect] iter=") == 3
    assert log.count("the code was not read by another model (the 3 readings") == 2
    duv = json.loads((FIXTURE / "code_review_duv.json").read_text(encoding="utf-8"))
    assert duv["reviews_total"] == 2 and duv["final_read"] is False  # its study never ran: the run size stopped it first


@pytest.mark.asyncio
async def test_before_the_fix_a_repair_of_the_run_ran_unread(tmp_path: Path) -> None:
    engine = _spent_quest(tmp_path)
    engine._client = _Reader()
    (engine.quest_root / "code" / "experiment.py").write_text("print('RESULT_JSON: {\"x\": 1}')\n", encoding="utf-8")
    await engine._code_review_gate(STATE, final=True)  # a state with no repair of the run: the old, capped behaviour
    assert engine._client.prompts == []
    record = cr.load(engine.quest_root)
    assert record["final_read"] is False and "spent" in record["skipped"]


def test_the_quality_report_says_why_the_repairs_ended() -> None:
    from core import quality_report as qr

    real = json.loads((FIXTURE / "code_review_euv.json").read_text(encoding="utf-8"))
    real["repairs_stopped"] = "the last repair did not reduce the open high-severity findings"
    q = type("Q", (), {"code_review": real, "function_steps": {}})()
    code = qr._code_metrics(q)  # type: ignore[arg-type]
    assert code["repairs_stopped"].startswith("the last repair did not reduce") and code["repair_rounds"] == 1
