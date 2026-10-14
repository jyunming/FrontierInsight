"""The run record in the engine: the write step puts FI's own record of the run at the end of the limitations whatever
the writer wrote there, before the open code findings, and the claim check's evidence holds the same text (replay of
suite r10b false_discovery's records: the graded paper gave no seeds)."""

from __future__ import annotations

import asyncio
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


def test_the_paper_gets_fi_s_run_record_whatever_the_writer_put_there(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path)
    forged = f"<!-- fi:run-record:{'0' * 12} -->\nThe run used seed 42 and took 3 s.\n{disclosure.RUN_RECORD_END}"
    draft = ("# FDR\n\n## Methods\nWe simulate z-tests.\n\n## Results\nThe realised FDR was 0.049.\n\n"
             "## Limitations\nThe grid is coarse.\n\n" + forged + "\n\n## References\n[1] A.\n")

    async def fake_chat(self: Any, messages: Any, **kw: Any) -> str:  # a writer that forges the block
        return draft

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    eng = _engine(root)
    state = {"topic": "t", "title": "t", "iteration": 0, "exec_result": {"returncode": 0, "duration_s": 73.6}}
    out = asyncio.run(eng._node_write(state))  # type: ignore[arg-type]
    paper = Path(out["paper_md"]).read_text(encoding="utf-8")
    expected = disclosure.run_record(root, topic=str(eng.config.topic or ""), duration_s=73.6)
    assert "base seed 0" in expected and "73.6 s" in expected
    assert "seed 42" not in paper and paper.count("fi:run-record:") == 1
    assert expected in paper
    assert paper.index("## Limitations") < paper.index("The grid is coarse.") < paper.index("<!-- fi:run-record:")
    # The number checks do not read FI's block; the paper's own numbers are still read.
    numbers = [n[1] for n in numeric_oracle.extract_paper_numbers(paper)]
    assert "73.6" not in numbers and "0.049" in numbers


def test_the_claim_check_sees_the_same_run_record(tmp_path: Path) -> None:
    from core.engine import _claim_method_block

    root = _quest(tmp_path)
    block = _claim_method_block(root, {"topic": "t", "exec_result": {"duration_s": 73.6}})  # type: ignore[arg-type]
    assert "WHAT FI RECORDED ABOUT THIS RUN" in block
    assert "base seed 0" in block and "73.6 s" in block
