"""The numeric oracle must reach a HUMAN, and must not force a rewrite.

``test_numeric_oracle.py`` pins the arithmetic. These pin the wiring.

The findings used to be merged into ``must_flag_hits``, which the router treats
as non-bypassable (``_route_after_review`` forces ``revise`` even with
``review_loop=False``). A real quest showed the cost when a finding is wrong:
false positives -- DOI prefixes read as measurements, a mantissa read without
its exponent -- forced a re-design, a re-implement, twelve more experiment runs
and a rewrite that left the paper worse (11/11 claims grounded -> 9/11).

So findings are now ADVISORY: recorded in ``review["numeric_oracle_warnings"]``,
copied into the human-review snapshot every interface renders, and kept OUT of
``must_flag_hits``. A correct checker that nobody sees still changes nothing,
which is why the snapshot test below matters as much as the router one.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.engine import Engine


class _Recorder:
    """Minimal stand-in exposing only what the oracle helper touches."""

    def __init__(self, tmp: Path, result_json: dict) -> None:
        self.quest_root = tmp
        self._log = SimpleNamespace(
            warning=lambda *a, **k: self.msgs.append(("warn", a)),
            info=lambda *a, **k: self.msgs.append(("info", a)),
        )
        self.msgs: list = []
        self.state = {"result_json": result_json}

    def hits(self, paper: str) -> list[str]:
        return Engine._numeric_oracle_hits(self, paper, self.state)  # type: ignore[arg-type]


def test_transcription_error_becomes_a_finding(tmp_path: Path) -> None:
    """The end the whole feature exists for: computed 2.14, written 2.41.
    The helper still returns one line per finding; whether that line blocks is
    decided by where the review node puts it (see the snapshot tests)."""
    rec = _Recorder(tmp_path, {"nils_dipole": 2.14})
    hits = rec.hits("Dipole illumination raised NILS to 2.41 at best focus.")
    assert len(hits) == 1
    assert hits[0].startswith("unverified_number:")
    # The hit names the value and the path, so the rewrite knows what to fix.
    assert "2.41" in hits[0] and "nils_dipole" in hits[0]


def test_consistent_paper_produces_no_hits(tmp_path: Path) -> None:
    rec = _Recorder(tmp_path, {"nils_dipole": 2.14})
    assert rec.hits("Dipole illumination raised NILS to 2.14.") == []


def test_audit_is_written_even_when_clean(tmp_path: Path) -> None:
    """A clean run must leave evidence the check ran — silence would be
    indistinguishable from the check being skipped."""
    rec = _Recorder(tmp_path, {"nils": 2.14})
    rec.hits("NILS was 2.14.")
    audit = tmp_path / "paper" / "numeric_audit.json"
    assert audit.is_file()
    data = json.loads(audit.read_text(encoding="utf-8"))
    assert data["ok"] is True
    assert data["result_numbers"] == 1


def test_audit_records_the_finding_when_dirty(tmp_path: Path) -> None:
    rec = _Recorder(tmp_path, {"nils": 2.14})
    rec.hits("NILS was 2.41.")
    data = json.loads((tmp_path / "paper" / "numeric_audit.json").read_text("utf-8"))
    assert data["ok"] is False
    assert data["findings"][0]["kind"] == "transposed"
    assert data["findings"][0]["result_path"] == "nils"


def test_survey_quest_with_no_results_is_not_blocked(tmp_path: Path) -> None:
    """A literature synthesis runs no experiment. The oracle must stay out
    of its way rather than flag every number in it."""
    rec = _Recorder(tmp_path, {})
    hits = rec.hits("The literature reports NILS values between 1.8 and 2.41.")
    assert hits == []
    data = json.loads((tmp_path / "paper" / "numeric_audit.json").read_text("utf-8"))
    assert data["skipped"] is True


def test_checker_failure_is_never_quest_fatal(tmp_path: Path, monkeypatch) -> None:
    """A bug in the oracle must not block a paper. Blocking publication over
    a defect in the checker is strictly worse than not checking."""
    from core import numeric_oracle

    def boom(*_a, **_k):
        raise RuntimeError("oracle exploded")

    monkeypatch.setattr(numeric_oracle, "check", boom)
    rec = _Recorder(tmp_path, {"nils": 2.14})
    assert rec.hits("NILS was 2.41.") == []
    assert any(kind == "warn" for kind, _ in rec.msgs)


def test_router_treats_the_hit_as_blocking() -> None:
    """Pin the router's contract for must-flag hits in general: a non-empty
    list forces revise even when review_loop is off. Methodologist must-flags
    still rely on this. The numeric oracle deliberately no longer does."""
    engine = SimpleNamespace(
        config=SimpleNamespace(
            engine=SimpleNamespace(max_iterations=4, review_loop=False),
            pauses=SimpleNamespace(review="off"),
        ),
        _log=SimpleNamespace(info=lambda *a, **k: None),
    )
    state = {
        "review": {"verdict": "accept", "must_flag_hits": ["unverified_number: x"]},
        "iteration": 0,
    }
    assert Engine._route_after_review(engine, state) == "revise"  # type: ignore[arg-type]


def test_max_iterations_zero_still_bypasses_the_gate() -> None:
    """Known limitation, pinned so it is a decision rather than a surprise.

    ``_route_after_review`` gates must-flag on ``iteration < max_iterations``,
    so a zero budget disables the blocking behaviour entirely. Fixing that is
    roadmap item A4; this test documents the hole until then.
    """
    engine = SimpleNamespace(
        config=SimpleNamespace(
            engine=SimpleNamespace(max_iterations=0, review_loop=False),
            pauses=SimpleNamespace(review="off"),
        ),
        _log=SimpleNamespace(info=lambda *a, **k: None),
    )
    state = {
        "review": {"verdict": "accept", "must_flag_hits": ["unverified_number: x"]},
        "iteration": 0,
    }
    assert Engine._route_after_review(engine, state) != "revise"


# --- advisory, not blocking: the path a finding takes now ---------------------

def _review_engine(tmp_path: Path, *, gate: str = "off") -> Engine:
    from core.config import (
        Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
        ProviderConfig,
    )
    cfg = Config(
        topic="t", title="t",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            clarify_mode="off", review_loop=True, max_iterations=2,
            human_feedback_gate=gate,  # type: ignore[arg-type]
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    eng = Engine(cfg)
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    eng.fi_dir = tmp_path / ".fi"  # type: ignore[attr-defined]
    return eng


def test_review_records_the_finding_as_advisory_not_blocking(tmp_path: Path) -> None:
    """Drives the real review node, so the merge itself is under test -- not a
    hand-built state. The reviewer accepts; the paper misquotes a number."""
    import asyncio

    eng = _review_engine(tmp_path)

    async def fake_chat(prompt, *, node=None):  # noqa: ANN001
        return json.dumps({"verdict": "accept", "score": 4,
                           "suggestions": [], "must_flag_hits": []})

    eng._chat = fake_chat  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text("Dipole illumination raised NILS to 2.41 at best focus.",
                     encoding="utf-8")
    state = {"topic": "t", "iteration": 0, "review": {},
             "paper_md": str(paper), "result_json": {"nils_dipole": 2.14}}

    patch = asyncio.run(eng._node_review(state))  # type: ignore[arg-type]
    review = patch["review"]

    warnings = review.get("numeric_oracle_warnings") or []
    assert len(warnings) == 1 and warnings[0].startswith("unverified_number:")
    assert review["must_flag_hits"] == [], "a numeric finding must not block"
    assert "iteration" not in patch, "an advisory finding must not spend budget"


def test_clean_paper_leaves_no_warning_key(tmp_path: Path) -> None:
    import asyncio

    eng = _review_engine(tmp_path)

    async def fake_chat(prompt, *, node=None):  # noqa: ANN001
        return json.dumps({"verdict": "accept", "score": 4, "suggestions": []})

    eng._chat = fake_chat  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text("Dipole illumination raised NILS to 2.14.", encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {},
        "paper_md": str(paper), "result_json": {"nils_dipole": 2.14},
    }))
    assert "numeric_oracle_warnings" not in patch["review"]


def test_human_review_snapshot_carries_the_warnings(tmp_path: Path) -> None:
    """The snapshot is what the web UI and the VSCode chat panel render. A
    warning that never reaches it is invisible in two of three interfaces."""
    import asyncio

    eng = _review_engine(tmp_path, gate="after_review")
    eng._pause_for_human = lambda **_kw: {"action": "accept"}  # type: ignore[method-assign]
    state = {
        "iteration": 0,
        "review": {"verdict": "accept", "must_flag_hits": [],
                   "numeric_oracle_warnings": ["unverified_number: 2.41 vs nils=2.14"]},
    }
    asyncio.run(eng._node_human_feedback(state))  # type: ignore[arg-type]

    snap = json.loads((tmp_path / ".fi" / "human_review.json").read_text("utf-8"))
    assert snap["numeric_oracle_warnings"] == ["unverified_number: 2.41 vs nils=2.14"]
    assert snap["must_flag_hits"] == [], "kept apart so the UIs can label them"
