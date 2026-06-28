"""Tests for the generic pause-drop-anytime user-input gate.

Behaviour pinned:

  1. ``pauses.supply`` default ``never`` keeps existing flow (no pause).
  2. ``"before_build"`` / ``"before_review"`` / ``"both"`` fire the
     interrupt at the right stage(s).
  3. The ``user_pauses_fired`` field prevents re-pausing at the same
     stage after resume.
  4. ``_pick_up_user_dropped_datasets`` walks ``inputs/data/`` and
     returns a clean sorted list of relative paths, filtering by
     known scientific-data suffixes and skipping the README.

Uses the in-process engine + a fake ``interrupt()`` so the LangGraph
side-effect is observable without running the full graph.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, PausesConfig, ProviderConfig,
)
from core.engine import Engine, _pick_up_user_dropped_datasets


def _mk_engine(tmp_path: Path, mode: str) -> Engine:
    cfg = Config(
        topic="pause-drop tests",
        title="pause-drop",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        pauses=PausesConfig(
            clarify="off", review="off",
            supply=mode,  # type: ignore[arg-type]
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    return eng


# --- _pick_up_user_dropped_datasets ----------------------------------------


def test_pick_up_datasets_walks_inputs_data_and_filters_extensions(
    tmp_path: Path,
) -> None:
    """Walk ``<quest_root>/inputs/data/`` and return one entry per
    file with a recognised tabular/scientific-data extension. Skip
    README, dotfiles, and any unrecognised suffix."""
    data = tmp_path / "inputs" / "data"
    data.mkdir(parents=True)
    (data / "samples.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (data / "labels.json").write_text("{}", encoding="utf-8")
    (data / "extra.parquet").write_bytes(b"PAR1")
    (data / "notes.docx").write_bytes(b"binary-blob")  # unknown suffix → skipped
    (data / "README.md").write_text("user-readme", encoding="utf-8")  # skipped
    (data / ".DS_Store").write_bytes(b"junk")  # dotfile → skipped
    found = _pick_up_user_dropped_datasets(tmp_path)
    assert set(found) == {
        "inputs/data/extra.parquet",
        "inputs/data/labels.json",
        "inputs/data/samples.csv",
    }


def test_pick_up_datasets_returns_empty_when_dir_missing(tmp_path: Path) -> None:
    """No inputs/data/ dir → empty list. Used by analyze on quests
    that never enabled the pause-drop gate."""
    assert _pick_up_user_dropped_datasets(tmp_path) == []


# --- pause routing ---------------------------------------------------------


def _drive_pause(
    eng: Engine, state: dict, stage: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Run ``_maybe_pause_for_user_input(state, stage)`` and stub
    ``interrupt()`` so we can observe whether it fired. Returns
    ``(fired, payload)``."""
    import core.engine as engine_mod
    real_interrupt = engine_mod.interrupt
    payload: dict[str, Any] | None = None
    fired = {"v": False}

    def fake_interrupt(value: Any) -> None:
        fired["v"] = True
        nonlocal payload
        payload = value
        # Simulate LangGraph: interrupt raises GraphInterrupt; the
        # caller's normal control flow halts. We model with a sentinel
        # exception that the test catches.
        raise _InterruptedSentinel(payload)

    engine_mod.interrupt = fake_interrupt
    try:
        eng._maybe_pause_for_user_input(state, stage)  # type: ignore[arg-type]
    except _InterruptedSentinel:
        pass
    finally:
        engine_mod.interrupt = real_interrupt
    return fired["v"], payload


class _InterruptedSentinel(Exception):
    """Stand-in for LangGraph's GraphInterrupt in tests."""


def test_pause_never_does_not_fire(tmp_path: Path) -> None:
    """Default mode is the no-op path — no pause regardless of stage."""
    eng = _mk_engine(tmp_path, "never")
    fired, _ = _drive_pause(eng, {}, "before_build")
    assert fired is False
    fired, _ = _drive_pause(eng, {}, "before_review")
    assert fired is False


def test_pause_before_build_only_fires_at_design(tmp_path: Path) -> None:
    """``"before_build"`` fires when stage matches and no-ops at
    the other gates so a config of before_build doesn't accidentally
    pause at before_review."""
    eng = _mk_engine(tmp_path, "before_build")
    fired_design, payload = _drive_pause(eng, {}, "before_build")
    assert fired_design is True
    assert payload is not None
    assert payload["user_input_required"] is True
    assert payload["stage"] == "before_build"
    fired_paper, _ = _drive_pause(eng, {}, "before_review")
    assert fired_paper is False


def test_pause_before_review_only_fires_at_paper(tmp_path: Path) -> None:
    eng = _mk_engine(tmp_path, "before_review")
    fired_design, _ = _drive_pause(eng, {}, "before_build")
    assert fired_design is False
    fired_paper, payload = _drive_pause(eng, {}, "before_review")
    assert fired_paper is True
    assert payload["stage"] == "before_review"


def test_pause_both_fires_at_both_stages(tmp_path: Path) -> None:
    eng = _mk_engine(tmp_path, "both")
    fired_design, _ = _drive_pause(eng, {}, "before_build")
    assert fired_design is True
    fired_paper, _ = _drive_pause(eng, {}, "before_review")
    assert fired_paper is True


def test_pause_does_not_refire_when_stage_already_in_user_pauses_fired(
    tmp_path: Path,
) -> None:
    """The single most important guarantee: after a resume, the engine
    re-invokes the node that paused. ``user_pauses_fired`` carries
    the stage from the prior pause so the gate falls through this
    time instead of pausing the user again."""
    eng = _mk_engine(tmp_path, "before_build")
    state = {"user_pauses_fired": ["before_build"]}
    fired, _ = _drive_pause(eng, state, "before_build")
    assert fired is False


def test_pause_does_not_refire_when_disk_marker_present(tmp_path: Path) -> None:
    """The disk marker at ``<quest_root>/.fi/paused_at_<stage>.flag``
    is the authoritative resume-safe signal. State-only tracking
    fails because ``interrupt()`` raises BEFORE the node returns a
    state patch, so the prior pause's ``user_pauses_fired`` update
    never lands in the checkpoint. The marker survives any resume."""
    eng = _mk_engine(tmp_path, "before_build")
    eng.fi_dir.mkdir(parents=True, exist_ok=True)
    (eng.fi_dir / "paused_at_before_build.flag").write_text(
        "before_build", encoding="utf-8",
    )
    fired, _ = _drive_pause(eng, {}, "before_build")
    assert fired is False


def test_pause_writes_disk_marker_before_interrupt(tmp_path: Path) -> None:
    """First pause must write the disk marker BEFORE firing
    interrupt() — otherwise an immediate resume re-enters the node
    without the marker present and pauses again, looping the user."""
    eng = _mk_engine(tmp_path, "before_build")
    _drive_pause(eng, {}, "before_build")
    assert (eng.fi_dir / "paused_at_before_build.flag").is_file()


def test_collect_artifacts_keeps_next_step_on_pause(tmp_path: Path) -> None:
    """Regression: ``_collect_artifacts`` is called on BOTH the pause-exit and
    the completion paths, so it must NOT delete NEXT_STEP.md — otherwise a
    paused quest's "Action needed" pointer is wiped the instant it pauses. The
    deletion lives only in the completion branch of ``run``."""
    eng = _mk_engine(tmp_path, "before_build")
    nxt = tmp_path / "NEXT_STEP.md"
    nxt.write_text("# Action needed — add any papers or data\n", encoding="utf-8")
    eng._collect_artifacts({"topic": "t"})  # type: ignore[arg-type]
    assert nxt.is_file(), "pause-exit must keep NEXT_STEP.md"


def test_pause_writes_next_step_with_drop_zone_hints(
    tmp_path: Path,
) -> None:
    """The unified NEXT_STEP.md the gate drops at the quest root makes the
    drop zones discoverable without consulting docs."""
    eng = _mk_engine(tmp_path, "before_build")
    _drive_pause(eng, {}, "before_build")
    nxt = tmp_path / "NEXT_STEP.md"
    assert nxt.exists()
    body = nxt.read_text(encoding="utf-8")
    assert "Action needed" in body
    assert "inputs/papers/" in body
    assert "inputs/data/" in body
    assert eng.quest_id in body  # resume command carries the quest id


# --- yaml emitter ---------------------------------------------------------


def test_interview_yaml_emits_supply_pause_when_non_default() -> None:
    """The interview emits the supply pause under the unified ``pauses:``
    section, translated from interview vocabulary (after_design →
    before_build), and only when the user chose something other than the
    ``never`` default so common-case YAMLs stay tidy."""
    from core.interview import InterviewAnswers, answers_to_yaml

    base_kwargs: dict[str, Any] = dict(
        topic="t", title="t",
        output_kinds=["paper_md"], paper_format="generic",
        no_simulation=False, study_depth="brief preprint",
        comparative_baseline="", success_metric="", budget="",
        clarify_mode="off", review_panel=[], knowledge_enabled=False,
    )
    default = InterviewAnswers(**base_kwargs)
    supply = InterviewAnswers(
        **{**base_kwargs, "pause_for_user_input": "after_design"},
    )

    yaml_default = answers_to_yaml(default, frontend="cli")
    yaml_pause = answers_to_yaml(supply, frontend="cli")

    # Default never → no supply line under pauses.
    assert "supply:" not in yaml_default
    # Explicit choice → supply line under pauses, translated to new vocab.
    assert "pauses:" in yaml_pause
    assert 'supply: "before_build"' in yaml_pause
