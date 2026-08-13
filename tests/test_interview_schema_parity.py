"""Cross-frontend parity test for the Frontier Insight interview.

Loads three artifacts and asserts they agree on:
  * the set of question IDs the user is asked
  * the paper_format / venue list (must match PaperFormat literal too)
  * the provider list

Artifacts:
  1. ``core/interview_schema.json`` (the snapshot of Python QUESTIONS).
  2. ``vscode-frontier-insight/src/interview-core.ts``
     (PaperFormat type + InterviewAnswers interface).
  3. ``vscode-frontier-insight/src/interview.ts`` — checked for the
     four new question titles so the schema-vs-prompt-text drift is
     caught.

The check is intentionally tight: drift between Python and TS is a
real bug class (8 vs 12 questions per surface = different YAML
schema = engine confusion). The Python source of truth wins; the TS
mirror needs to keep pace."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO / "core" / "interview_schema.json"
INTERVIEW_CORE_TS = REPO / "vscode-frontier-insight" / "src" / "interview-core.ts"
INTERVIEW_TS = REPO / "vscode-frontier-insight" / "src" / "interview.ts"


def test_schema_json_exists() -> None:
    """The committed snapshot lives at core/interview_schema.json.
    Regenerate via ``python -c 'from core.interview import
    write_schema_json; write_schema_json()'``."""
    assert SCHEMA_PATH.is_file(), (
        f"{SCHEMA_PATH} missing. Generate with "
        f"`python -c 'from core.interview import write_schema_json; write_schema_json()'`."
    )


def test_paper_format_choices_match_ts_literal() -> None:
    """The TS file declares ``PaperFormat`` as a string union. Parse
    it and compare against the schema JSON's paper_formats list."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    schema_values = sorted(c["value"] for c in schema["paper_formats"])

    ts_text = INTERVIEW_CORE_TS.read_text(encoding="utf-8")
    # Find the export type PaperFormat = "a" | "b" | ... block.
    m = re.search(
        r"export type PaperFormat\s*=\s*([^;]+);",
        ts_text,
    )
    assert m, "PaperFormat type declaration not found in interview-core.ts"
    body = m.group(1)
    ts_values = sorted(re.findall(r'"([a-z_]+)"', body))

    assert schema_values == ts_values, (
        f"PaperFormat drift. Python schema has {schema_values}; "
        f"TS interview-core.ts has {ts_values}. "
        f"Update one to match the other."
    )


# Python InterviewAnswers fields that legitimately have NO TS counterpart.
# Keep this list SHORT and justified — anything not here is required to appear
# in the TS interface (see the auto-derived parity test below).
_TS_EXEMPT_ANSWER_FIELDS = frozenset({
    # CLI / serve only — VSCode pins the provider to ``vscode_extension``
    # silently and routes human-in-the-loop pauses through the bridge, so
    # neither field is part of the TS answer surface.
    "provider",
    "pause_for_user_input",
})


def test_ts_interview_answers_has_all_python_managed_fields() -> None:
    """EVERY field on the Python ``InterviewAnswers`` dataclass (minus an
    explicit, justified exempt set) must appear in the TS ``InterviewAnswers``
    interface, so a Python answer can't silently fail to round-trip through the
    VSCode half.

    The field list is AUTO-DERIVED from ``InterviewAnswers.__dataclass_fields__``
    rather than hardcoded — a hardcoded list is itself a drift source (add a
    Python field, forget to list it here, and the parity check has a blind
    spot). Adding a Python answer field now automatically requires a matching
    TS field or an explicit entry in ``_TS_EXEMPT_ANSWER_FIELDS``."""
    from core.interview import InterviewAnswers

    ts_text = INTERVIEW_CORE_TS.read_text(encoding="utf-8")
    m = re.search(
        r"export interface InterviewAnswers\s*\{([^}]+)\}",
        ts_text,
        re.DOTALL,
    )
    assert m, "InterviewAnswers interface not found in interview-core.ts"
    body = m.group(1)

    required = sorted(
        set(InterviewAnswers.__dataclass_fields__) - _TS_EXEMPT_ANSWER_FIELDS
    )
    # Guard against the exempt set drifting out of sync with the dataclass.
    stale_exempt = _TS_EXEMPT_ANSWER_FIELDS - set(InterviewAnswers.__dataclass_fields__)
    assert not stale_exempt, (
        f"_TS_EXEMPT_ANSWER_FIELDS lists fields no longer on InterviewAnswers: "
        f"{sorted(stale_exempt)} — remove them."
    )

    missing = [
        field for field in required
        # ``\??`` tolerates optional TS fields (e.g. ``survey_mode?: boolean``).
        if not re.search(rf"\b{field}\??\s*:", body)
    ]
    assert not missing, (
        f"TS InterviewAnswers is missing Python answer field(s): {missing}. "
        f"All Python interview answers must round-trip to YAML identically. "
        f"Add the field to vscode-frontier-insight/src/interview-core.ts, or "
        f"— if it is genuinely CLI/serve-only — add it to "
        f"_TS_EXEMPT_ANSWER_FIELDS with a justification."
    )


def test_ts_interview_prompts_for_new_questions() -> None:
    """The four new Phase-R questions (study_depth +
    comparative_baseline + success_metric + budget) must have
    prompts in interview.ts so VSCode actually asks them. If the
    interview.ts file was updated to add the answer fields but never
    actually prompt, the YAML emits placeholders forever."""
    ts_text = INTERVIEW_TS.read_text(encoding="utf-8")
    expected_prompts = (
        "study depth",            # showQuickPick title for study_depth
        "comparative baseline",   # showInputBox title for comparative_baseline
        "success metric",         # showInputBox title for success_metric
        "compute budget",         # showInputBox title for budget ("time / compute budget")
    )
    lowered = ts_text.lower()
    for prompt in expected_prompts:
        assert prompt in lowered, (
            f"interview.ts is missing a prompt containing {prompt!r}. "
            f"The Python schema includes a question that's not asked "
            f"in the VSCode interview — frontends drift here."
        )


def test_ts_ensemble_model_trios_match_python() -> None:
    """``ENSEMBLE_MODEL_TRIOS`` in interview-core.ts must match
    ``_ENSEMBLE_MODEL_TRIOS`` in core/interview.py (mirrored to the
    schema JSON snapshot). Drift would cause one frontend to produce
    a different node_ensemble.models list than the others for the
    same provider, breaking the three-interface parity guarantee."""
    from_schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))[
        "ensemble_model_trios"
    ]
    ts_text = INTERVIEW_CORE_TS.read_text(encoding="utf-8")
    # Match the ENSEMBLE_MODEL_TRIOS literal: capture object body.
    m = re.search(
        r"export const ENSEMBLE_MODEL_TRIOS[^=]*=\s*\{([^}]+)\};",
        ts_text, re.DOTALL,
    )
    assert m, "ENSEMBLE_MODEL_TRIOS literal not found in interview-core.ts"
    body = m.group(1)
    # Each row looks like `"openai": ["gpt-4o", "gpt-4o-mini", "o1-mini"],`
    row_re = re.compile(
        r'"([a-z_]+)"\s*:\s*\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]',
    )
    ts_trios = {m.group(1): [m.group(2), m.group(3), m.group(4)]
                for m in row_re.finditer(body)}
    assert ts_trios == from_schema, (
        f"ENSEMBLE_MODEL_TRIOS drift between TS and Python:\n"
        f"TS:     {sorted(ts_trios.items())}\n"
        f"Python: {sorted(from_schema.items())}"
    )


def test_ts_emits_clarify_overrides_in_yaml() -> None:
    """The TS YAML emitter must write the ``clarify_overrides`` block
    so the engine's clarify node short-circuits in auto mode (same
    as the Python emitter)."""
    ts_text = INTERVIEW_CORE_TS.read_text(encoding="utf-8")
    assert "clarify_overrides" in ts_text, (
        "interview-core.ts must emit a `clarify_overrides:` block — "
        "without it, the engine fires the clarify LLM and the "
        "interview's research-shaping answers get overwritten by "
        "the agent's auto-defaults."
    )
    # Every clarify-override key the Python emits should also appear
    # in the TS emitter, so the YAML shape is identical.
    for key in (
        "study_depth", "paper_venue", "output_kinds", "simulatability",
        "comparative_baseline", "success_metric", "budget",
    ):
        assert key in ts_text, (
            f"interview-core.ts clarify_overrides is missing key {key!r}. "
            f"Drift with core/interview.py:answers_to_yaml."
        )
