"""The curated view of a running quest (core/engine.py:STAGE_PROGRESS, PROGRESS_LOG_NAME): a fixed, short set of
plain-English stage lines for the CLI console, VS Code and the web page, while `run.log` keeps every internal
detail. What a first-time user watching a quest sees should be a handful of readable lines, not hundreds of them."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from core.engine import PROGRESS_LOG_NAME, STAGE_PROGRESS, Engine, _ProgressOnly, _close_quest_logger, _quest_logger
from tests.test_frozen_protocol import _cfg, _fake


def _record(name: str, msg: str, *, progress: bool) -> logging.LogRecord:
    r = logging.LogRecord(name, logging.INFO, __file__, 1, msg, (), None)
    if progress:
        r.progress = True
    return r


def test_the_filter_lets_through_only_records_marked_progress() -> None:
    f = _ProgressOnly()
    assert f.filter(_record("x", "a stage line", progress=True)) is True
    assert f.filter(_record("x", "hundreds of internal detail", progress=False)) is False


def test_every_visible_graph_node_that_should_say_something_has_a_plain_sentence() -> None:
    """The phrase map is what a reader sees; each entry must read as a sentence on its own (capitalized, ends in a
    period), and covers the stages a person actually watches for: literature, design, implement, execute, analyze,
    write, review."""
    for node in ("literature", "design", "implement", "execute", "analyze", "cross_check", "write", "review"):
        assert node in STAGE_PROGRESS, node
    for node, phrase in STAGE_PROGRESS.items():
        assert phrase[0].isupper() and phrase.endswith("."), (node, phrase)
        assert len(phrase) < 60, "a stage line is one short sentence, not a paragraph"


def test_quest_logger_writes_the_curated_file_beside_run_log(tmp_path: Path) -> None:
    fi_dir = tmp_path / ".fi"
    qid = "test-progress-file"
    logger = _quest_logger(qid, fi_dir)
    try:
        logger.info("internal detail nobody but a developer needs")
        logger.info("[design] Designing the experiment.", extra={"progress": True})
        run_log = (fi_dir / "run.log").read_text(encoding="utf-8")
        progress_log = (fi_dir / PROGRESS_LOG_NAME).read_text(encoding="utf-8")
        assert "internal detail" in run_log and "Designing the experiment" in run_log
        assert "internal detail" not in progress_log and "Designing the experiment" in progress_log
    finally:
        _close_quest_logger(qid)


def test_a_repeated_stage_line_is_not_shown_twice(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    engine = Engine(cfg)
    engine._progress_stage("design")
    engine._progress_stage("design")   # a node re-entered after a pause: the identical line is not repeated
    engine._progress_stage("execute")
    _close_quest_logger(engine.quest_id)
    lines = [l for l in (engine.fi_dir / PROGRESS_LOG_NAME).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2 and "[design]" in lines[0] and "[execute]" in lines[1]


@pytest.mark.asyncio
async def test_a_real_quest_leaves_a_short_readable_progress_log_beside_a_long_run_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(prompts, second_design={}))
    engine = Engine(_cfg(tmp_path))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None

    run_log = (engine.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    progress_log = (engine.quest_root / ".fi" / PROGRESS_LOG_NAME).read_text(encoding="utf-8")
    progress_lines = [l for l in progress_log.splitlines() if l.strip()]

    run_lines = [l for l in run_log.splitlines() if l.strip()]
    assert len(progress_lines) < 30, "a first-time user should see a handful of lines, not the internal log"
    assert len(progress_lines) < len(run_lines) / 2, "the curated file must be a small fraction of the full log"
    assert progress_log.count(run_log.splitlines()[0]) == 0, "the two files hold different content, not one nested in the other"
    for expect in ("[literature]", "[design]", "[implement", "[execute]", "[analyze]", "[write]", "[review]", "[evidence]"):
        assert any(expect in line for line in progress_lines), f"{expect} missing from the curated view"
    # Nothing internal leaked into the curated file: none of the noisy phrases a real run produces belong here.
    for leak in ("quarantined", "verdict=sufficient route=write decided=", "LLM head:", "response not parseable"):
        assert leak not in progress_log, leak
    # Everything the curated file says is also, verbatim, in the full log — it is a subset, not a rewrite. Both files
    # prefix each line with "<date> <time> "; run.log adds a level tag (`[INFO] `) progress.log does not.
    import re
    def _message(line: str) -> str:
        return re.sub(r"^\S+ \S+ (\[[A-Z]+\] )?", "", line)
    run_messages = {_message(l) for l in run_lines}
    for line in progress_lines:
        assert _message(line) in run_messages, line


def test_a_console_handler_only_ever_shows_progress_lines(tmp_path: Path) -> None:
    """The same guarantee for the CLI console (and, through it, VS Code's stderr-reading bridge): the StreamHandler
    `_quest_logger` attaches carries the identical filter as the progress file."""
    qid = "test-console-filter"
    logger = _quest_logger(qid, tmp_path / ".fi")
    try:
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) == 1
        assert any(isinstance(f, _ProgressOnly) for f in stream_handlers[0].filters)
    finally:
        _close_quest_logger(qid)


# --- the web page's live log -----------------------------------------------------------------------------------------


def _client(output_root: Path):
    from web.server import make_app

    return TestClient(make_app(output_root))


def test_the_web_log_endpoint_prefers_the_curated_file_and_falls_back_for_an_old_quest(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    new_quest = root / "q-new" / ".fi"
    new_quest.mkdir(parents=True)
    (new_quest / PROGRESS_LOG_NAME).write_text("2026-01-01 [design] Designing the experiment.\n", encoding="utf-8")
    (new_quest / "run.log").write_text("2026-01-01 [INFO] a hundred lines of internal detail\n", encoding="utf-8")
    old_quest = root / "q-old" / ".fi"
    old_quest.mkdir(parents=True)
    (old_quest / "run.log").write_text("2025-01-01 [INFO] a quest run before this feature existed\n", encoding="utf-8")

    client = _client(root)
    got_new = client.get("/api/quests/q-new/log").json()["lines"]
    assert any("Designing the experiment" in l for l in got_new) and not any("internal detail" in l for l in got_new)
    got_old = client.get("/api/quests/q-old/log").json()["lines"]
    assert any("run before this feature" in l for l in got_old)
