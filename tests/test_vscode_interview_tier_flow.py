"""Content-level invariants for the VSCode @fi /new tier flow in
``vscode-frontier-insight/src/interview.ts``.

The flow itself runs inside the VSCode runtime (showInputBox /
showQuickPick are not available in plain node), so we can't drive
it end-to-end here. Instead we pin the surface area: the JS hooks
exist, the markdown review block lists the right fields, and the
extension's manifest entry for /new is still present.
"""
from __future__ import annotations

from pathlib import Path

import pytest

EXT_SRC = Path(__file__).resolve().parent.parent / "vscode-frontier-insight" / "src"
INTERVIEW_TS = EXT_SRC / "interview.ts"
INTERVIEW_CORE_TS = EXT_SRC / "interview-core.ts"


@pytest.fixture(scope="module")
def interview_ts() -> str:
    return INTERVIEW_TS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def core_ts() -> str:
    return INTERVIEW_CORE_TS.read_text(encoding="utf-8")


# ---- Tier-1 surface (4 modals) ----


@pytest.mark.parametrize(
    "needle",
    [
        "quest topic",          # topic modal title fragment
        "paper format / venue", # paper_format modal
        "which deliverables",   # output_kinds modal
        "study depth",          # study_depth modal
    ],
)
def test_interview_keeps_four_tier1_modal_titles(
    interview_ts: str, needle: str,
) -> None:
    """The first four QuickPick/InputBox titles correspond to the
    tier-1 questions VSCode users see. A future edit that removes
    one of these regresses the always-ask scope."""
    assert needle in interview_ts.lower(), (
        f"interview.ts lost the {needle!r} tier-1 modal title"
    )


def test_interview_no_longer_asks_title_modal(interview_ts: str) -> None:
    """Title is auto-slugged from topic in the new flow — no
    standalone "Title (short slug)" modal. If a future edit puts
    one back, the user count goes from 4 tier-1 modals to 5.
    (Confirmed via the prompt string the old modal used.)"""
    assert "quest title (short slug)" not in interview_ts.lower(), (
        "title modal should not exist in the tier-1 flow"
    )


# ---- Tier-2 derivation block + review markdown ----


@pytest.mark.parametrize(
    "needle",
    [
        "PROSE_FORMATS",            # set driving no_simulation derivation
        "probeAxonReachable",       # auto-detect for knowledge_enabled
        "reviewBlockMarkdown",      # streamed review table builder
        "editTier2Field",           # inline editor for the 6 defaults
        "editTier3Field",           # inline editor for the 4 advanced
        "Review before launch",     # heading streamed into chat
    ],
)
def test_tier_flow_function_or_marker_present(
    interview_ts: str, needle: str,
) -> None:
    assert needle in interview_ts, f"interview.ts lost the {needle!r} tier-flow hook"


@pytest.mark.parametrize(
    "label",
    [
        "Launch quest",
        "Edit a default",
        "Edit an advanced field",
        "Cancel",
    ],
)
def test_action_picker_has_four_options(interview_ts: str, label: str) -> None:
    """The action picker that runs after tier-1 collection must
    offer all four options: launch, edit default, edit advanced,
    cancel. Removing any of them breaks a documented user path."""
    assert label in interview_ts, (
        f"action picker missing {label!r}"
    )


# ---- InterviewAnswers shape (audience + knowledge_top_k) ----


def test_interview_answers_carries_audience_and_top_k(core_ts: str) -> None:
    """The new audience + knowledge_top_k slots must round-trip
    through the YAML emitter — pin their presence in the interface
    definition."""
    assert "audience" in core_ts, "InterviewAnswers must carry audience"
    assert "knowledge_top_k" in core_ts, (
        "InterviewAnswers must carry knowledge_top_k"
    )


def test_yaml_emitter_emits_audience_only_when_non_default(core_ts: str) -> None:
    """audience defaults to "external" and is omitted from the YAML
    in that case to keep the generated config small; only emit it
    when the user picked "internal"."""
    # The check: there should be a guard around the audience emit
    # comparing to "external".
    assert '"external"' in core_ts and "answers.audience" in core_ts


# ---- Defensive yamlEscape ----


def test_yaml_escape_tolerates_undefined(core_ts: str) -> None:
    """A stub answers object missing a field would otherwise crash
    yamlEscape with .replace-on-undefined. The defensive guard
    makes the function tolerate null/undefined by returning "".
    Pins the guard so a future edit doesn't strip it."""
    assert "null | undefined" in core_ts or "=== undefined" in core_ts, (
        "yamlEscape must tolerate undefined inputs"
    )
