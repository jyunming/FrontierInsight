"""The run record in the engine: the write step puts FI's own record of the run at the end of the limitations whatever
the writer wrote there, before the open code findings, and the claim check's evidence holds the same text (replay of
suite r10b false_discovery's records: the graded paper gave no seeds)."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from core import disclosure, numeric_oracle
from tests.test_disclosure import _engine

FIXTURES = Path(__file__).parent / "fixtures" / "run_record"


def _quest(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURES / "fdr_r10b", tmp_path, dirs_exist_ok=True)
    return tmp_path


def test_the_paper_has_no_run_record_and_the_writer_gets_the_facts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The run record is not written into the paper (graded papers carried false facts in it): a block the writer forged is
    taken out, the writer is given the record's facts instead, and the quality report holds the record."""
    from core import quality_report

    root = _quest(tmp_path)
    forged = f"<!-- fi:run-record:{'0' * 12} -->\nThe run used seed 42 and took 3 s.\n{disclosure.RUN_RECORD_END}"
    draft = ("# FDR\n\n## Methods\nWe simulate z-tests.\n\n## Results\nThe realised FDR was 0.049.\n\n"
             "## Limitations\nThe grid is coarse.\n\n" + forged + "\n\n## References\n[1] A.\n")
    prompts: list[str] = []

    async def fake_chat(self: Any, messages: Any, **kw: Any) -> str:  # a writer that forges the block
        prompts.append(json.dumps(messages))
        return draft

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    eng = _engine(root)
    state = {"topic": "t", "title": "t", "iteration": 0, "exec_result": {"returncode": 0, "duration_s": 73.6}}
    out = asyncio.run(eng._node_write(state))  # type: ignore[arg-type]
    paper = Path(out["paper_md"]).read_text(encoding="utf-8")
    assert "seed 42" not in paper and "fi:run-record" not in paper and "What FI recorded about this run" not in paper
    assert "The grid is coarse." in paper
    assert any("FI'S RECORD OF THIS RUN" in p and "base seed 0" in p for p in prompts), "the writer is given the facts"
    held = quality_report.build(root, running=False)["paper_records"]["run_record"]
    assert "base seed 0" in held
    # The numbers of the paper are read as before; FI's record has nothing in the paper to leave out.
    numbers = [n[1] for n in numeric_oracle.extract_paper_numbers(paper)]
    assert "0.049" in numbers


def test_the_claim_check_sees_the_run_record_without_the_last_execution_s_time(tmp_path: Path) -> None:
    from core.engine import _claim_method_block

    root = _quest(tmp_path)
    block = _claim_method_block(root, {"topic": "t", "exec_result": {"duration_s": 73.6}})  # type: ignore[arg-type]
    assert "WHAT FI RECORDED ABOUT THIS RUN" in block
    assert "base seed 0" in block
    # This run's trials are in a ledger and FI recorded no time for them: the time of the last execution (an analysis run
    # again on stored trials, as often as not) is not the experiment's, so it is not given.
    assert "73.6 s" not in block and "ran for" not in block
