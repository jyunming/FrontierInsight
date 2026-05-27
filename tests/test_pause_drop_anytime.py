"""Tests for the generic pause-drop-anytime user-input gate.

Behaviour pinned:

  1. ``engine.pause_for_user_input`` default ``never`` keeps existing
     flow (no pause).
  2. ``"after_design"`` / ``"after_paper"`` / ``"both"`` fire the
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

import asyncio
from pathlib import Path
from typing import Any

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine, _pick_up_user_dropped_datasets


def _mk_engine(tmp_path: Path, mode: str) -> Engine:
    cfg = Config(
        topic="pause-drop tests",
        title="pause-drop",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False,
            clarify_mode="off",
            pause_for_user_input=mode,  # type: ignore[arg-type]
            human_feedback_gate="off",
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
    fired, _ = _drive_pause(eng, {}, "after_design")
    assert fired is False
    fired, _ = _drive_pause(eng, {}, "after_paper")
    assert fired is False


def test_pause_after_design_only_fires_at_design(tmp_path: Path) -> None:
    """``"after_design"`` fires when stage matches and no-ops at
    the other gates so a config of after_design doesn't accidentally
    pause at after_paper."""
    eng = _mk_engine(tmp_path, "after_design")
    fired_design, payload = _drive_pause(eng, {}, "after_design")
    assert fired_design is True
    assert payload is not None
    assert payload["user_input_required"] is True
    assert payload["stage"] == "after_design"
    fired_paper, _ = _drive_pause(eng, {}, "after_paper")
    assert fired_paper is False


def test_pause_after_paper_only_fires_at_paper(tmp_path: Path) -> None:
    eng = _mk_engine(tmp_path, "after_paper")
    fired_design, _ = _drive_pause(eng, {}, "after_design")
    assert fired_design is False
    fired_paper, payload = _drive_pause(eng, {}, "after_paper")
    assert fired_paper is True
    assert payload["stage"] == "after_paper"


def test_pause_both_fires_at_both_stages(tmp_path: Path) -> None:
    eng = _mk_engine(tmp_path, "both")
    fired_design, _ = _drive_pause(eng, {}, "after_design")
    assert fired_design is True
    fired_paper, _ = _drive_pause(eng, {}, "after_paper")
    assert fired_paper is True


def test_pause_does_not_refire_when_stage_already_in_user_pauses_fired(
    tmp_path: Path,
) -> None:
    """The single most important guarantee: after a resume, the engine
    re-invokes the node that paused. ``user_pauses_fired`` carries
    the stage from the prior pause so the gate falls through this
    time instead of pausing the user again."""
    eng = _mk_engine(tmp_path, "after_design")
    state = {"user_pauses_fired": ["after_design"]}
    fired, _ = _drive_pause(eng, state, "after_design")
    assert fired is False


def test_pause_writes_inputs_readme_with_drop_zone_hints(
    tmp_path: Path,
) -> None:
    """The README the gate drops alongside the pause makes the
    drop zones discoverable without consulting docs."""
    eng = _mk_engine(tmp_path, "after_design")
    _drive_pause(eng, {}, "after_design")
    readme = tmp_path / "inputs" / "README.md"
    assert readme.exists()
    body = readme.read_text(encoding="utf-8")
    assert "inputs/papers/" in body
    assert "inputs/data/" in body
    assert eng.quest_id in body  # resume command carries the quest id


# --- yaml emitter ---------------------------------------------------------


def test_interview_yaml_emits_pause_for_user_input_when_non_default() -> None:
    """The interview YAML emitter only writes the line when the user
    chose something other than ``never`` (default), so hand-edited
    YAMLs stay tidy for the common case."""
    from core.interview import InterviewAnswers, answers_to_yaml

    base_kwargs: dict[str, Any] = dict(
        topic="t", title="t",
        output_kinds=["paper_md"], paper_format="generic",
        no_simulation=False, study_depth="brief preprint",
        comparative_baseline="", success_metric="", budget="",
        clarify_mode="off", review_panel=[], knowledge_enabled=False,
    )
    default = InterviewAnswers(**base_kwargs)
    after_design = InterviewAnswers(
        **{**base_kwargs, "pause_for_user_input": "after_design"},
    )

    yaml_default = answers_to_yaml(default, frontend="cli")
    yaml_pause = answers_to_yaml(after_design, frontend="cli")

    # Default never → no line in YAML.
    assert "pause_for_user_input" not in yaml_default
    # Explicit choice → line is written.
    assert "pause_for_user_input: \"after_design\"" in yaml_pause
