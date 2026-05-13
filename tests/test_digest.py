"""Direct tests for `core.digest`.

Unit-tested at fine granularity: quest-id timestamp parsing, terminal-
node detection against synthetic state.sqlite files, snapshot
collection with date-window filtering, prior-digest discovery, the
WeekDiff computation against a prior digest's markdown, and an
end-to-end with a mocked LLM that asserts the prompt + the on-disk
output. No real LLM calls; no real Axon.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.config import ProviderConfig
from core.digest import (
    QuestSnapshot,
    WeekDiff,
    _collect_quest_snapshots,
    _compute_diff,
    _detect_terminal_node,
    _digest_id,
    _find_prior_digest,
    _parse_quest_id_timestamp,
    _quest_ids_in_digest,
    _quest_ids_marked_in_progress,
    _quest_ids_marked_still_in_progress,
    _quest_title_from_id,
    generate_digest,
)


# ---------- quest_id parsing helpers ----------------------------------------


def test_parse_quest_id_timestamp_well_formed() -> None:
    qid = "1778452404-euv-mor-photon-shot-noise-ler-e6bfe5"
    ts = _parse_quest_id_timestamp(qid)
    assert ts is not None
    assert ts.tzinfo is timezone.utc
    assert ts == datetime.fromtimestamp(1778452404, tz=timezone.utc)


def test_parse_quest_id_timestamp_malformed_returns_none() -> None:
    assert _parse_quest_id_timestamp("not-a-quest-id") is None
    assert _parse_quest_id_timestamp("123-too-short-aabbcc") is None
    assert _parse_quest_id_timestamp("1778452404-no-nonce") is None


def test_quest_title_from_id_dehyphenates_and_trims_nonce() -> None:
    qid = "1778452404-euv-mor-photon-shot-noise-ler-e6bfe5"
    assert _quest_title_from_id(qid) == "euv mor photon shot noise ler"
    # Malformed input returns the input verbatim.
    assert _quest_title_from_id("bogus") == "bogus"


# ---------- terminal-node detection -----------------------------------------


def _make_state_sqlite(path: Path, *, with_review: bool) -> None:
    """Create a minimal LangGraph-shaped checkpoint sqlite. The
    digest module only reads from the `writes` table and only checks
    whether a `review` channel row exists — so we can fake just that."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE writes ("
        "thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT, "
        "task_id TEXT, idx INTEGER, channel TEXT, type TEXT, value BLOB)"
    )
    if with_review:
        con.execute(
            "INSERT INTO writes (channel) VALUES ('review')"
        )
    else:
        con.execute(
            "INSERT INTO writes (channel) VALUES ('ideate')"
        )
    con.commit()
    con.close()


def test_detect_terminal_node_completed(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    _make_state_sqlite(db, with_review=True)
    assert _detect_terminal_node(db) == "review"


def test_detect_terminal_node_in_progress(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite"
    _make_state_sqlite(db, with_review=False)
    assert _detect_terminal_node(db) == "in_progress"


def test_detect_terminal_node_missing_file(tmp_path: Path) -> None:
    assert _detect_terminal_node(tmp_path / "missing.sqlite") == "unknown"


def test_detect_terminal_node_garbage_file(tmp_path: Path) -> None:
    p = tmp_path / "garbage.sqlite"
    p.write_bytes(b"not a sqlite db at all")
    assert _detect_terminal_node(p) == "unknown"


def test_detect_terminal_node_schema_drift_returns_unknown(tmp_path: Path) -> None:
    p = tmp_path / "schema-drift.sqlite"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE checkpoints (foo TEXT)")  # no writes table
    con.commit()
    con.close()
    assert _detect_terminal_node(p) == "unknown"


# ---------- snapshot collection ---------------------------------------------


def _make_quest_dir(
    outputs: Path, *, quest_id: str, with_paper: bool = True,
    paper_body: str = "# A toy paper\n\nIt computes things.",
    with_review: bool | None = None,
    summary_json: dict | None = None,
) -> Path:
    qdir = outputs / quest_id
    qdir.mkdir(parents=True, exist_ok=True)
    if with_paper:
        (qdir / "paper").mkdir(exist_ok=True)
        (qdir / "paper" / "paper.md").write_text(paper_body, encoding="utf-8")
    if summary_json is not None:
        import json
        (qdir / "frontier_insight_summary.json").write_text(
            json.dumps(summary_json), encoding="utf-8",
        )
    if with_review is not None:
        _make_state_sqlite(qdir / ".fi" / "state.sqlite", with_review=with_review)
    return qdir


def test_collect_snapshots_filters_by_window(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    # quest_id epochs spread across a wide range.
    inside = "1700000700-recent-quest-aabbcc"      # 2023-11-14
    outside = "1600000000-old-quest-001100"        # 2020-09-13
    _make_quest_dir(outputs, quest_id=inside, with_review=True)
    _make_quest_dir(outputs, quest_id=outside, with_review=True)

    since = datetime.fromtimestamp(1699000000, tz=timezone.utc)
    until = datetime.fromtimestamp(1701000000, tz=timezone.utc)

    # Window is by last-modified, not creation. Touch `inside`'s files
    # to land them in window; leave `outside` with default mtime
    # (which will be `now` since pytest just created them, putting it
    # outside the [1699, 1701] window).
    # Easier: bend `inside`'s mtime into the window and `outside`'s out.
    import os
    inside_target = (since + (until - since) / 2).timestamp()
    outside_target = (since - timedelta(days=10)).timestamp()
    for p in (outputs / inside).rglob("*"):
        os.utime(p, (inside_target, inside_target))
    os.utime(outputs / inside, (inside_target, inside_target))
    for p in (outputs / outside).rglob("*"):
        os.utime(p, (outside_target, outside_target))
    os.utime(outputs / outside, (outside_target, outside_target))

    snaps = _collect_quest_snapshots(outputs, since, until)
    ids = [s.quest_id for s in snaps]
    assert inside in ids
    assert outside not in ids


def test_collect_snapshots_skips_underscore_and_dot_dirs(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "_digests").mkdir()
    (outputs / "_drafts").mkdir()
    (outputs / ".cache").mkdir()
    _make_quest_dir(outputs, quest_id="1700000700-real-quest-aabbcc", with_review=True)

    since = datetime.fromtimestamp(1600000000, tz=timezone.utc)
    until = datetime.fromtimestamp(1900000000, tz=timezone.utc)
    snaps = _collect_quest_snapshots(outputs, since, until)
    assert len(snaps) == 1
    assert snaps[0].quest_id == "1700000700-real-quest-aabbcc"


def test_collect_snapshots_extracts_terminal_node_and_abstract(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    qid = "1700000700-finished-quest-aabbcc"
    _make_quest_dir(
        outputs, quest_id=qid, with_review=True,
        paper_body="# Title here\n\nThe abstract paragraph that should appear in the digest.",
        summary_json={"title": "Finished quest", "provider": "openai", "topic": "X"},
    )
    snaps = _collect_quest_snapshots(
        outputs,
        datetime.fromtimestamp(1600000000, tz=timezone.utc),
        datetime.fromtimestamp(1900000000, tz=timezone.utc),
    )
    assert len(snaps) == 1
    s = snaps[0]
    assert s.title == "Finished quest"
    assert s.provider == "openai"
    assert s.terminal_node == "review"
    assert "The abstract paragraph" in s.paper_abstract
    assert "Title here" not in s.paper_abstract  # title was stripped


# ---------- digest_id naming ------------------------------------------------


def test_digest_id_uses_iso_week_for_7_day_window() -> None:
    until = datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
    since = until - timedelta(days=7)
    did = _digest_id(since, until)
    iso = until.isocalendar()
    assert did == f"{iso.year}-W{iso.week:02d}"


def test_digest_id_uses_dated_form_for_custom_window() -> None:
    until = datetime(2026, 5, 13, tzinfo=timezone.utc)
    since = until - timedelta(days=14)
    did = _digest_id(since, until)
    assert did == "2026-04-29-to-2026-05-13"


# ---------- prior digest discovery ------------------------------------------


def test_find_prior_digest_prefers_iso_week(tmp_path: Path) -> None:
    digests = tmp_path / "_digests"
    digests.mkdir()
    (digests / "2026-W18.md").write_text("# Week 18", encoding="utf-8")
    (digests / "2026-W19.md").write_text("# Week 19", encoding="utf-8")
    (digests / "2026-04-29-to-2026-05-13.md").write_text("# Custom", encoding="utf-8")

    prev = _find_prior_digest(digests, "2026-W20")
    assert prev is not None
    assert prev.name == "2026-W19.md"


def test_find_prior_digest_excludes_self(tmp_path: Path) -> None:
    digests = tmp_path / "_digests"
    digests.mkdir()
    (digests / "2026-W19.md").write_text("# W19", encoding="utf-8")
    (digests / "2026-W20.md").write_text("# W20", encoding="utf-8")
    # Searching with "this_digest_id=2026-W20" should not return the W20 file.
    prev = _find_prior_digest(digests, "2026-W20")
    assert prev is not None and prev.name == "2026-W19.md"


def test_find_prior_digest_falls_back_to_dated_when_no_iso(tmp_path: Path) -> None:
    digests = tmp_path / "_digests"
    digests.mkdir()
    (digests / "2026-04-29-to-2026-05-13.md").write_text("# old", encoding="utf-8")
    (digests / "2026-05-01-to-2026-05-15.md").write_text("# newer", encoding="utf-8")
    prev = _find_prior_digest(digests, "2026-05-15-to-2026-05-30")
    assert prev is not None
    assert prev.name == "2026-05-01-to-2026-05-15.md"


def test_find_prior_digest_empty_dir(tmp_path: Path) -> None:
    assert _find_prior_digest(tmp_path / "missing", "2026-W20") is None


# ---------- WeekDiff logic ---------------------------------------------------


def _snap(qid: str, terminal: str = "review") -> QuestSnapshot:
    return QuestSnapshot(
        quest_id=qid, title=qid, topic="", created_at=datetime.now(timezone.utc),
        last_modified=datetime.now(timezone.utc), terminal_node=terminal,
        paper_abstract="", provider=None, quest_root=Path("/tmp"),
    )


def test_compute_diff_first_run_lists_everything_under_new(tmp_path: Path) -> None:
    snapshots = [_snap("1700000001-a-aabbcc", "review"),
                 _snap("1700000002-b-aabbcc", "in_progress")]
    diff = _compute_diff(snapshots, prev_md=None, prev_path=None)
    assert diff.prev_digest_id is None
    assert [q.quest_id for q in diff.newly_completed] == ["1700000001-a-aabbcc"]
    assert [q.quest_id for q in diff.new_in_progress] == ["1700000002-b-aabbcc"]
    assert diff.promoted == []
    assert diff.still_in_progress == []


def test_compute_diff_promotes_in_progress_to_completed(tmp_path: Path) -> None:
    # Prior digest lists quest A under "In progress"; this week A is
    # complete → A should appear in `promoted`, not `newly_completed`.
    prev_md = (
        "# Digest 2026-W18\n\n"
        "## Completed this week\n_None._\n\n"
        "## In progress\n"
        "- [1700000001-a-aabbcc] **Quest A** progress\n\n"
        "## What changed since last digest\n(first run)\n"
    )
    prev_path = tmp_path / "2026-W18.md"
    prev_path.write_text(prev_md, encoding="utf-8")
    snaps = [_snap("1700000001-a-aabbcc", "review")]
    diff = _compute_diff(snaps, prev_md=prev_md, prev_path=prev_path)
    assert [q.quest_id for q in diff.promoted] == ["1700000001-a-aabbcc"]
    assert diff.newly_completed == []


def test_compute_diff_stalled_when_still_in_progress_carried_over(tmp_path: Path) -> None:
    """A quest is `stalled` when it's still_in_progress in this digest
    AND was marked still_in_progress in the prior digest's
    "What changed" carry-over section — i.e. it's been ≥3 consecutive
    digests in flight."""
    prev_md = (
        "# Digest 2026-W18\n\n"
        "## In progress\n"
        "- [1700000001-stuck-aabbcc] **Stuck quest**\n\n"
        "## What changed since last digest\n"
        "**⚠️ Still in progress from last digest:**\n"
        "- [1700000001-stuck-aabbcc] Stuck quest\n"
    )
    snaps = [_snap("1700000001-stuck-aabbcc", "in_progress")]
    diff = _compute_diff(snaps, prev_md=prev_md, prev_path=tmp_path / "2026-W18.md")
    assert "1700000001-stuck-aabbcc" in diff.stalled
    assert any(q.quest_id == "1700000001-stuck-aabbcc" for q in diff.still_in_progress)


def test_compute_diff_dropped_quests_from_prior_not_in_this(tmp_path: Path) -> None:
    prev_md = (
        "## In progress\n"
        "- [1700000001-vanished-aabbcc] **Disappeared quest**\n"
    )
    snaps: list[QuestSnapshot] = []   # nothing in this week
    diff = _compute_diff(snaps, prev_md=prev_md, prev_path=tmp_path / "2026-W18.md")
    assert diff.dropped == ["1700000001-vanished-aabbcc"]


def test_compute_diff_dropped_excludes_prior_completions_and_citations(
    tmp_path: Path,
) -> None:
    """Regression: ``dropped`` is narrowed to quests that were
    in-progress in the prior digest and went silent. Prior-week
    completions, theme citations, and suggested-next-quest citations
    should NOT appear as dropped — those quests aren't expected to
    reappear in subsequent digests."""
    prev_md = (
        "## Completed this week\n"
        "- [1700000001-completed-aabbcc] **Finished quest**\n\n"
        "## In progress\n"
        "- [1700000002-stuck-aabbcc] **Stuck quest**\n\n"
        "## Themes\n"
        "- A theme that cites [1700000003-themed-aabbcc]\n\n"
        "## Suggested next quests\n"
        "- Idea referencing [1700000004-suggested-aabbcc]\n"
    )
    snaps: list[QuestSnapshot] = []
    diff = _compute_diff(snaps, prev_md=prev_md, prev_path=tmp_path / "2026-W18.md")
    assert diff.dropped == ["1700000002-stuck-aabbcc"]
    assert "1700000001-completed-aabbcc" not in diff.dropped
    assert "1700000003-themed-aabbcc" not in diff.dropped
    assert "1700000004-suggested-aabbcc" not in diff.dropped


def test_render_diff_section_includes_newly_completed_subsection() -> None:
    """A quest that completed this window but was NOT in the prior
    digest at all (e.g. started + finished in the same week) must
    appear under the '🎉 Newly completed' bullet."""
    from core.digest import _render_diff_section

    diff = WeekDiff(
        prev_digest_id="2026-W18",
        newly_completed=[_snap("1700000001-fresh-aabbcc", "review")],
    )
    rendered = _render_diff_section(diff)
    assert "🎉 Newly completed" in rendered
    assert "1700000001-fresh-aabbcc" in rendered


def test_render_diff_section_dropped_label_reflects_narrowed_semantics() -> None:
    """The label under the dropped bullet must reflect the narrowed
    'in-progress quests that went silent' semantics so users aren't
    misled into thinking prior completions were abandoned."""
    from core.digest import _render_diff_section

    diff = WeekDiff(
        prev_digest_id="2026-W18",
        dropped=["1700000002-stuck-aabbcc"],
    )
    rendered = _render_diff_section(diff)
    assert "went silent" in rendered.lower()
    assert "1700000002-stuck-aabbcc" in rendered


def test_build_prompt_velocity_uses_full_snapshots_not_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: when the manifest is truncated for prompt budget,
    the velocity numbers must still come from the FULL snapshot count
    — otherwise the 'deterministic numbers' the prompt asks the LLM
    to quote disagree with the manifest's own truncation note."""
    import core.digest as dm
    monkeypatch.setattr(dm, "_MAX_PROMPT_QUESTS", 3)

    snaps = [_snap(f"170000000{i}-q-aabbcc", "review") for i in range(7)]
    diff = WeekDiff(prev_digest_id=None)
    since = datetime(2026, 5, 1, tzinfo=timezone.utc)
    until = datetime(2026, 5, 8, tzinfo=timezone.utc)
    prompt = dm._build_digest_prompt(
        snapshots=snaps, diff=diff, prev_digest_md=None,
        since=since, until=until,
    )
    # Velocity should report 7 (full set), not 3 (truncated slice).
    assert "Quests touched this window: 7" in prompt
    # The manifest itself is truncated, with a note that 4 more exist.
    assert "4 additional quests" in prompt


def test_quest_ids_in_digest_finds_ids_regardless_of_layout() -> None:
    md = (
        "Random text. [1700000001-foo-aabbcc] in a bullet.\n"
        "Inline ref like 1700000002-bar-001122 with no brackets.\n"
        "Bogus id-like-but-not 12345-foo-x and 1234567890-foo-z (too few hex)."
    )
    ids = _quest_ids_in_digest(md)
    assert "1700000001-foo-aabbcc" in ids
    assert "1700000002-bar-001122" in ids
    # The hex-suffix check rejects "1234567890-foo-z" (single char).
    assert len(ids) == 2


def test_quest_ids_marked_in_progress_finds_section() -> None:
    md = (
        "## Completed this week\n- [1700000001-a-aabbcc] X\n\n"
        "## In progress\n"
        "- [1700000002-b-aabbcc] Y\n"
        "- [1700000003-c-aabbcc] Z\n\n"
        "## Themes\n- something cited [1700000004-d-aabbcc]\n"
    )
    ip = _quest_ids_marked_in_progress(md)
    assert ip == {"1700000002-b-aabbcc", "1700000003-c-aabbcc"}


def test_quest_ids_marked_still_in_progress_pulls_from_change_section() -> None:
    md = (
        "## What changed since last digest\n"
        "**⚠️ Still in progress from last digest:**\n"
        "- [1700000001-stuck-aabbcc] **Stuck quest**\n"
        "## Themes\n- something else [1700000002-other-aabbcc]\n"
    )
    s = _quest_ids_marked_still_in_progress(md)
    assert s == {"1700000001-stuck-aabbcc"}


# ---------- end-to-end with mocked LLM --------------------------------------


@pytest.mark.asyncio
async def test_generate_digest_end_to_end_writes_file_and_returns_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    qid_done = "1700000700-finished-aabbcc"
    qid_in_progress = "1700001700-running-001122"
    _make_quest_dir(
        outputs, quest_id=qid_done, with_review=True,
        summary_json={"title": "Finished one", "provider": "openai"},
    )
    _make_quest_dir(
        outputs, quest_id=qid_in_progress, with_review=False,
        summary_json={"title": "Still running", "provider": "openai"},
    )

    # Mock LLM that records the prompt + returns canned markdown.
    captured: dict[str, str] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        captured["prompt"] = messages[-1]["content"]
        return "# Weekly Digest\n\n## Completed this week\n- [done]\n"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    # The window check is on file last-modified, not on quest_id epoch.
    # pytest just created these files with mtime ~ wall-clock now, so
    # we have to make the window encompass that. Use the actual wall-
    # clock time as `now` and a very wide retrospective window.
    now = datetime.now(timezone.utc) + timedelta(hours=1)
    art = await generate_digest(
        outputs, days=365 * 100,   # 100-year window so mtimes definitely fall in
        provider=ProviderConfig(name="openai"),
        knowledge=None,
        now=now,
    )

    assert art.digest_path.is_file()
    body = art.digest_path.read_text(encoding="utf-8")
    assert body.startswith("# Weekly Digest")

    assert art.quest_count == 2
    assert art.completed_count == 1
    assert art.in_progress_count == 1
    assert art.ingested_to_axon is False  # knowledge=None

    # Prompt was rendered with the manifest containing both quest IDs
    # and the velocity numbers.
    prompt = captured["prompt"]
    assert qid_done in prompt
    assert qid_in_progress in prompt
    assert "Quests touched this window: 2" in prompt
    assert "Completed this window: 1" in prompt


@pytest.mark.asyncio
async def test_generate_digest_empty_window_writes_marker_no_llm_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()

    # Patch LLMClient.chat to detect (incorrect) invocation.
    called = {"n": 0}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        called["n"] += 1
        return "should not be invoked"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    art = await generate_digest(
        outputs, days=7,
        provider=ProviderConfig(name="openai"),
        knowledge=None,
    )
    assert art.quest_count == 0
    assert art.digest_path.is_file()
    body = art.digest_path.read_text(encoding="utf-8")
    assert "No quests were touched in this window" in body
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_generate_digest_uses_prior_digest_to_compute_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a prior digest exists with quest A marked in-progress, and
    this digest sees A as completed, the diff section in the rendered
    prompt should report A under ✅ Promoted, not 🆕 Newly started."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    digests = outputs / "_digests"
    digests.mkdir()
    (digests / "2026-W18.md").write_text(
        "## In progress\n- [1700000700-finished-aabbcc] Promoted\n",
        encoding="utf-8",
    )
    _make_quest_dir(
        outputs, quest_id="1700000700-finished-aabbcc", with_review=True,
        summary_json={"title": "Promoted quest", "provider": "openai"},
    )

    captured: dict[str, str] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        captured["prompt"] = messages[-1]["content"]
        return "# Digest body\n"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    art = await generate_digest(
        outputs, days=365 * 100,
        provider=ProviderConfig(name="openai"),
        knowledge=None,
        now=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    assert art.diff.prev_digest_id == "2026-W18"
    assert [q.quest_id for q in art.diff.promoted] == ["1700000700-finished-aabbcc"]
    assert art.diff.newly_completed == []
    assert "✅ Promoted to complete this week" in captured["prompt"]
    assert "1700000700-finished-aabbcc" in captured["prompt"]


@pytest.mark.asyncio
async def test_generate_digest_ingests_to_axon_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _make_quest_dir(
        outputs, quest_id="1700000700-x-aabbcc", with_review=True,
        summary_json={"title": "X", "provider": "openai"},
    )

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return "# Body\n"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    knowledge = MagicMock()
    knowledge.enabled = True
    knowledge.add_text = MagicMock(return_value=True)

    art = await generate_digest(
        outputs, days=365 * 100,
        provider=ProviderConfig(name="openai"),
        knowledge=knowledge,
        now=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    assert art.ingested_to_axon is True
    knowledge.add_text.assert_called_once()
    kwargs = knowledge.add_text.call_args.kwargs
    assert kwargs["kind"] == "fi_digest"
    assert kwargs["metadata"]["digest_id"] == art.digest_id
