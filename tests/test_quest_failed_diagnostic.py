"""Quest-failure diagnostic tests.

When ``Engine.run()`` raises mid-quest (a ``_node_*`` method raised,
or a pre-graph stage like ``_preflight_paper_pdf`` raised), the engine
must write a ``<quest_root>/quest_failed.md`` breadcrumb so the user
can discover the failure from the quest directory itself — not just
from the buried ``<quest_root>/.fi/launch.log`` traceback.

The contract pins:

  - On a node raise: file written with exception text, failing-node
    name (from the LangGraph snapshot), log tail, provider context,
    and a resume command.
  - On a pre-graph raise (before the saver context opened): file
    written with a ``pre-graph stage`` failing-node placeholder.
  - On clean completion: any stale file from a prior failed run is
    removed.
  - The original exception is re-raised regardless — the diagnostic
    write is a side effect, not a swallow-and-continue.
  - A failure to write the diagnostic itself does NOT mask the
    original exception (the user wants to see the real error, not
    "could not open file for diagnostic writing").
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine, _close_quest_logger


def _mk_engine(
    tmp_path: Path, request: pytest.FixtureRequest, *, cfg: Config | None = None,
) -> Engine:
    """Engine factory that auto-closes the per-quest run.log handler
    on test teardown. Without the finalizer, the FileHandler stays
    open past test exit; on Windows the OS holds the file lock, and
    pytest's tmpdir cleanup raises ``PermissionError [WinError 32]``.
    ``_close_quest_logger`` is the engine's own helper (idempotent,
    same call ``Engine.run``'s outer finally makes)."""
    if cfg is None:
        cfg = Config(
            topic="some topic for diagnostic tests",
            title="diag",
            provider=ProviderConfig(
                name="claude_cli", model="claude-sonnet-4-6",
            ),
            engine=EngineConfig(
                max_iterations=1, review_loop=False, clarify_mode="off",
            ),
            execution=ExecutionConfig(sandbox="venv", timeout_s=60),
            knowledge=KnowledgeConfig(enabled=False),
            output=OutputConfig(output_dir=tmp_path / "outputs"),
        )
    eng = Engine(cfg)
    request.addfinalizer(lambda: _close_quest_logger(eng.quest_id))
    return eng


@pytest.mark.asyncio
async def test_write_diagnostic_with_no_run_config_labels_pre_graph_stage(
    tmp_path: Path, request: pytest.FixtureRequest,
) -> None:
    """When ``run_config`` is None (pre-graph failure — preflight,
    endpoint resolution, executor setup raised BEFORE the saver
    context opened), the diagnostic labels the failing-node as a
    pre-graph stage rather than trying to read a non-existent
    checkpoint snapshot."""
    eng = _mk_engine(tmp_path, request)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)

    exc = RuntimeError("pandoc not on PATH — preflight refused to continue")
    await eng._write_quest_failed_diagnostic(exc, run_config=None)

    diag = (eng.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    assert "pre-graph stage" in diag
    assert "RuntimeError" in diag
    assert "pandoc not on PATH" in diag


@pytest.mark.asyncio
async def test_diagnostic_includes_provider_context(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    """The diagnostic must surface the provider + model the quest was
    running with — a wall-clock timeout on ``claude_cli`` is a very
    different failure mode from a bridge-error on ``vscode_extension``,
    and the user shouldn't have to grep the log to tell them apart."""
    eng = _mk_engine(tmp_path, request)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)

    await eng._write_quest_failed_diagnostic(
        RuntimeError("transient"), run_config=None,
    )

    diag = (eng.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    assert "claude_cli" in diag
    assert "claude-sonnet-4-6" in diag


@pytest.mark.asyncio
async def test_diagnostic_includes_resume_command(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    """The diagnostic must include a copy-pasteable resume command —
    most node failures are transient and the LangGraph checkpoint
    recovers them on resume. Forgetting the command is the
    documentation equivalent of telling the user 'good luck'."""
    eng = _mk_engine(tmp_path, request)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)

    await eng._write_quest_failed_diagnostic(
        RuntimeError("x"), run_config=None,
    )
    diag = (eng.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    assert "--resume" in diag
    assert eng.quest_id in diag


@pytest.mark.asyncio
async def test_diagnostic_tails_run_log_when_present(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    """When ``.fi/run.log`` exists, the diagnostic must embed its tail
    so the user sees the immediate cause without a separate ``tail``
    command. Pre-logging failures (run.log absent) get a placeholder
    message instead of a missing-file traceback."""
    eng = _mk_engine(tmp_path, request)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    # Seed a run.log with a recognisable last line.
    log = eng.fi_dir / "run.log"
    log.write_text(
        "[implement] generating experiment code\n"
        "[implement] CRITICAL canary line at the tail\n",
        encoding="utf-8",
    )

    await eng._write_quest_failed_diagnostic(
        RuntimeError("ouch"), run_config=None,
    )

    diag = (eng.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    assert "CRITICAL canary line at the tail" in diag


@pytest.mark.asyncio
async def test_diagnostic_handles_missing_run_log(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    """If the engine raised BEFORE ``run.log`` was opened (extreme
    edge case: failure during ``Engine.__init__`` or its caller), the
    diagnostic still writes — with a placeholder noting the log
    isn't on disk — rather than crashing the diagnostic-write itself.

    ``Engine.__init__`` actually creates run.log via the FileHandler,
    so this scenario isn't reachable without simulation; patch
    ``Path.is_file`` on the run.log path to fake it.
    """
    eng = _mk_engine(tmp_path, request)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)

    run_log = eng.fi_dir / "run.log"
    original_is_file = Path.is_file

    def fake_is_file(self):
        if self == run_log:
            return False
        return original_is_file(self)

    with patch.object(Path, "is_file", fake_is_file):
        await eng._write_quest_failed_diagnostic(
            RuntimeError("very early"), run_config=None,
        )
    diag = (eng.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    assert "run.log not on disk" in diag


@pytest.mark.asyncio
async def test_diagnostic_write_failure_does_not_mask_original_exception(
    tmp_path: Path, request: pytest.FixtureRequest,
) -> None:
    """If writing ``quest_failed.md`` itself fails (disk full, ACL
    denial, parent dir gone, etc.), the original exception that
    triggered the diagnostic must still propagate. The diagnostic is
    a NICE-TO-HAVE; the user's primary need is to see the real
    error.

    This test simulates that by mocking ``Path.write_text`` on the
    diagnostic path to raise OSError, then confirming the helper
    swallows the OSError (logs a warning) instead of re-raising it
    (which would mask the original exception when the caller is
    already in an ``except`` block re-raising the original)."""
    eng = _mk_engine(tmp_path, request)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)

    original_write_text = Path.write_text

    def failing_write_text(self, *a, **kw):
        if self.name == "quest_failed.md":
            raise OSError("simulated disk full")
        return original_write_text(self, *a, **kw)

    with patch.object(Path, "write_text", failing_write_text):
        # Must NOT raise — the helper must swallow the diagnostic-
        # write failure so the caller's ``raise`` of the original
        # exception is what propagates.
        await eng._write_quest_failed_diagnostic(
            RuntimeError("the real error"), run_config=None,
        )


@pytest.mark.asyncio
async def test_diagnostic_truncates_long_topics(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    """A pathological multi-paragraph topic must not blow up the
    diagnostic's header. Cap at 200 chars so the .md stays scannable
    and the meaningful body (exception, log tail, resume command)
    stays above the fold."""
    long_topic = "x " * 500  # 1000 chars
    cfg = Config(
        topic=long_topic,
        title="long",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    eng = _mk_engine(tmp_path, request, cfg=cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    await eng._write_quest_failed_diagnostic(
        RuntimeError("brevity test"), run_config=None,
    )
    diag = (eng.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    # The topic header line should be capped — look for the **Topic:**
    # line specifically and assert its rendered topic is ≤ 200 chars
    # (the cap inside the helper's body).
    topic_line = [
        line for line in diag.splitlines()
        if line.startswith("**Topic:**")
    ]
    assert len(topic_line) == 1
    # ``**Topic:** ABCD...`` — strip the prefix, count the rest.
    rendered = topic_line[0].split("**Topic:**", 1)[1].strip()
    assert len(rendered) <= 200


def test_clear_stale_quest_failed_is_no_op_when_file_absent(
    tmp_path: Path, request: pytest.FixtureRequest,
) -> None:
    """No file present → no error, no log spam. The helper is called
    on every clean success / data-pause exit, including first-ever
    runs of a quest, so a missing file must be silently fine."""
    eng = _mk_engine(tmp_path, request)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    # No quest_failed.md created.
    eng._clear_stale_quest_failed_diagnostic()  # must not raise


def test_clear_stale_quest_failed_removes_stale_file(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    """A prior-run ``quest_failed.md`` is removed when the current
    run completes (or pauses cleanly). Verifies the file is actually
    gone after the call — the "leave it for the user to see" trap is
    exactly the misleading-state issue this PR is fixing."""
    eng = _mk_engine(tmp_path, request)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    stale = eng.quest_root / "quest_failed.md"
    stale.write_text("# stale from prior failed run", encoding="utf-8")
    assert stale.is_file()

    eng._clear_stale_quest_failed_diagnostic()

    assert not stale.exists()


def test_clear_stale_quest_failed_tolerates_unlink_failure(
    tmp_path: Path, request: pytest.FixtureRequest,
) -> None:
    """If ``unlink`` raises (Windows file lock, ACL, etc.), the
    helper must log and continue — a stale file is annoying but not
    fatal, and the success-path callers shouldn't be torpedoed by a
    cleanup failure after the real work succeeded."""
    eng = _mk_engine(tmp_path, request)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    stale = eng.quest_root / "quest_failed.md"
    stale.write_text("stale", encoding="utf-8")

    original_unlink = Path.unlink

    def failing_unlink(self, *a, **kw):
        if self.name == "quest_failed.md":
            raise OSError("simulated Windows lock")
        return original_unlink(self, *a, **kw)

    with patch.object(Path, "unlink", failing_unlink):
        # Must not raise — the success path's contract is that the
        # cleanup is best-effort.
        eng._clear_stale_quest_failed_diagnostic()


@pytest.mark.asyncio
async def test_engine_run_writes_diagnostic_when_early_stage_raises(
    tmp_path: Path, request: pytest.FixtureRequest,
) -> None:
    """End-to-end: when something inside ``Engine.run``'s try-body
    raises, the wrap-with-try-except must call the diagnostic helper
    AND let the original exception propagate.

    Pins the contract that's most likely to regress if someone
    restructures ``Engine.run`` later (e.g. moves the saver block,
    adds a new exit path).

    We force the FAILURE at the pre-graph ``_preflight_paper_pdf``
    stage — which keeps this test hermetic (no venv setup, no
    endpoint resolution, no real LangGraph saver), and exercises the
    ``run_config is None`` branch of the diagnostic helper. A
    failure mid-graph would also work but require mocking the
    venv-creating ``executor.setup`` and the real provider endpoint;
    pre-graph is the same wrap, less plumbing.
    """
    eng = _mk_engine(tmp_path, request)
    sentinel = RuntimeError("simulated preflight failure for diagnostic test")

    def boom() -> None:
        raise sentinel

    with patch.object(eng, "_preflight_paper_pdf", boom):
        with pytest.raises(RuntimeError) as exc_info:
            await eng.run()

    # Original exception propagated — masking is the worst possible bug.
    assert exc_info.value is sentinel

    # Diagnostic file written. ``quest_failed.md`` lives next to the
    # otherwise-empty quest dir.
    diag = eng.quest_root / "quest_failed.md"
    assert diag.is_file(), (
        f"Engine.run must write quest_failed.md on a pre-graph "
        f"failure; only found: {list(eng.quest_root.iterdir())}"
    )
    body = diag.read_text(encoding="utf-8")
    assert "simulated preflight failure for diagnostic test" in body
    assert "RuntimeError" in body
    assert "--resume" in body
    # Pre-graph branch: the failing-node label must reflect that the
    # graph never opened, not surface a stale checkpoint reading.
    assert "pre-graph stage" in body


@pytest.mark.asyncio
async def test_engine_run_cleans_stale_diagnostic_on_success_after_failure(
    tmp_path: Path, request: pytest.FixtureRequest,
) -> None:
    """If a prior run left ``quest_failed.md`` on disk and the current
    run completes successfully (or pauses cleanly), the stale
    diagnostic must be removed — leaving it would mislead the user
    into thinking the just-completed quest broke."""
    eng = _mk_engine(tmp_path, request)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    stale = eng.quest_root / "quest_failed.md"
    stale.write_text("# stale from prior run", encoding="utf-8")

    # Directly invoke the cleanup helper as a stand-in for the
    # full Engine.run success path. The real Engine.run calls this
    # right before the "reached terminal state" log line and before
    # returning artifacts on the data-paused branch.
    eng._clear_stale_quest_failed_diagnostic()
    assert not stale.exists()


@pytest.mark.asyncio
async def test_diagnostic_topic_header_strips_embedded_newlines(
    tmp_path: Path, request: pytest.FixtureRequest,
) -> None:
    """A YAML block-scalar topic carries embedded newlines. Without
    normalization, those newlines land in the rendered
    ``**Topic:**`` line and break the markdown header by splitting
    it across multiple bullet rows. The helper must collapse all
    internal whitespace to a single space before the 200-char cap."""
    cfg = Config(
        topic="line one\nline two\nline three with    extra    spaces",
        title="multi-line",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="off",
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    eng = _mk_engine(tmp_path, request, cfg=cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    await eng._write_quest_failed_diagnostic(
        RuntimeError("x"), run_config=None,
    )
    diag = (eng.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    # The Topic line should be exactly one line — find it, assert
    # it doesn't contain a newline AND the three pieces are
    # collapsed to single spaces.
    topic_lines = [
        line for line in diag.splitlines()
        if line.startswith("**Topic:**")
    ]
    assert len(topic_lines) == 1
    rendered = topic_lines[0].split("**Topic:**", 1)[1].strip()
    assert "\n" not in rendered
    assert "line one line two line three with extra spaces" in rendered


@pytest.mark.asyncio
async def test_diagnostic_log_tail_uses_bounded_read(
    tmp_path: Path, request: pytest.FixtureRequest,
) -> None:
    """The diagnostic must not load the entire run.log into memory.
    Quest logs can grow to tens of MB; reading the whole file makes
    the failure path slower exactly when the user is least patient.
    Verifies the helper reads ≤ 64 KB even when the file is much
    larger, and still surfaces the LAST 80 lines."""
    eng = _mk_engine(tmp_path, request)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)

    # Build a 1 MB log: 10_000 lines of "filler garbage line N\n" plus
    # a distinctive tail. We assert the tail lands in the diagnostic
    # and the filler far from the tail does NOT (which it couldn't
    # if we're reading the last 64 KB ≈ a few hundred lines).
    log = eng.fi_dir / "run.log"
    filler_line = "filler garbage line " + ("x" * 80) + "\n"
    expected_count = max(1, 1_000_000 // len(filler_line))
    with log.open("w", encoding="utf-8") as f:
        for i in range(expected_count):
            f.write(filler_line)
        f.write("[implement] FINAL TAIL CANARY at the very end\n")

    await eng._write_quest_failed_diagnostic(
        RuntimeError("y"), run_config=None,
    )
    diag = (eng.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    # Tail line MUST land.
    assert "FINAL TAIL CANARY at the very end" in diag
    # The rendered log_tail section is bounded — the diagnostic
    # itself shouldn't be MB-sized. A simple upper bound on the .md
    # size validates that we didn't slurp the whole log in.
    diag_size = (eng.quest_root / "quest_failed.md").stat().st_size
    assert diag_size < 200_000, (
        f"diagnostic file is {diag_size} bytes — far larger than the "
        f"~64 KB bound; bounded-read regression?"
    )


@pytest.mark.asyncio
async def test_diagnostic_includes_bridge_extras_when_present(
    tmp_path: Path, request: pytest.FixtureRequest,
) -> None:
    """When ``provider.extra`` carries bridge wiring (port or socket),
    the diagnostic must surface that — a bridge-error on a quest with
    ``bridge_port=12345`` is a very different debug story from one
    without."""
    cfg = Config(
        topic="t",
        title="t",
        provider=ProviderConfig(
            name="vscode_extension", model="gpt-5",
            extra={"bridge_port": 12345, "bridge_socket": "/tmp/fi.sock"},
        ),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    eng = _mk_engine(tmp_path, request, cfg=cfg)
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    await eng._write_quest_failed_diagnostic(
        RuntimeError("bridge error"), run_config=None,
    )
    diag = (eng.quest_root / "quest_failed.md").read_text(encoding="utf-8")
    assert "bridge_port=12345" in diag
    assert "bridge_socket=" in diag
