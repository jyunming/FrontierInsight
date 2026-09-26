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

# Every label the oracle can put on a finding. Each finding carries its own
# kind rather than one blanket prefix, so "did the oracle block?" is asked
# against all three.
_ORACLE_KINDS = ("transposed:", "near_miss:", "trivial_reference:")


class _Recorder:
    """Minimal stand-in exposing only what the oracle helper touches."""

    def __init__(self, tmp: Path, result_json: dict, **state) -> None:
        self.quest_root = tmp
        self._log = SimpleNamespace(
            warning=lambda *a, **k: self.msgs.append(("warn", a)),
            info=lambda *a, **k: self.msgs.append(("info", a)),
        )
        self.msgs: list = []
        # ``design`` and ``topic`` are what the quest gave the run.
        self.state = {"result_json": result_json, **state}

    def hits(self, paper: str) -> list[str]:
        return Engine._numeric_oracle_hits(self, paper, self.state)  # type: ignore[arg-type]


def test_transcription_error_becomes_a_finding(tmp_path: Path) -> None:
    """The end the whole feature exists for: computed 2.14, written 2.41.
    The helper still returns one line per finding; whether that line blocks is
    decided by where the review node puts it (see the snapshot tests)."""
    rec = _Recorder(tmp_path, {"nils_dipole": 2.14})
    hits = rec.hits("Dipole illumination raised NILS to 2.41 at best focus.")
    assert len(hits) == 1
    # Labelled by the signal that fired, not by one blanket prefix over all
    # of them: 2.14 and 2.41 hold the same digits, so this one is transposed.
    assert hits[0].startswith("transposed:")
    # The hit names the value and the path, so the rewrite knows what to fix.
    assert "2.41" in hits[0] and "nils_dipole" in hits[0]


def test_a_near_miss_is_labelled_a_near_miss(tmp_path: Path) -> None:
    """The second signal, under its own name — a last digit the paper's own
    rounding cannot explain, which is a different claim from digits swapped."""
    rec = _Recorder(tmp_path, {"metrics": {"contrast": {"quadrupole": 0.7912}}})
    hits = rec.hits("Contrast reached 0.792 under quadrupole illumination.")
    assert len(hits) == 1
    assert hits[0].startswith("near_miss:")


# The trivial-root shape: one deterministic reference solved at nine (R0, N)
# cells and 0.0 at every one of them, alongside a quantity that does vary.
_TRIVIAL_ROOT_RESULTS = {
    "by_R0_N": {
        f"{r0}_{n}": {
            "deterministic_final_size": 0.0,
            "outbreak_probability": p,
        }
        for (r0, n), p in zip(
            [(r, n) for r in ("0.9", "1.5", "3.0")
             for n in ("100", "1000", "5000")],
            [0.01, 0.05, 0.2, 0.31, 0.33, 0.45, 0.66, 0.67, 0.67],
        )
    }
}


def test_a_trivial_reference_is_not_called_an_unverified_number(
    tmp_path: Path,
) -> None:
    """The label used to contradict the finding it was pasted onto.

    ``deterministic_final_size`` WAS computed and exported correctly; what is
    suspect is the bracket the root finder was handed, which is exactly what
    the finding's own sentence says. Prefixing it "unverified_number" told
    the reader the opposite about the one number in the sentence — and the
    label is the half they read first.
    """
    rec = _Recorder(tmp_path, _TRIVIAL_ROOT_RESULTS)
    hits = rec.hits("The deterministic limit is approached as N grows.")
    assert len(hits) == 1
    assert hits[0].startswith("trivial_reference:")
    assert not hits[0].startswith("unverified_number")
    # The finding still points at the cause rather than the value.
    assert "check the bracket" in hits[0]


def test_a_setting_the_quest_gave_the_run_is_not_a_finding(tmp_path: Path) -> None:
    """``R_0 = 1.5`` is the paper quoting its own setup. The engine hands the
    oracle the design and the topic, so a number they state is not reported
    as a near-miss of the result whose last digit it happens to sit beside."""
    paper = "At R_0 = 1.50 the simulations ran 250 replicates."
    results = {"mean_final_size": 1.4949}  # rounds to 1.49, one digit from the printed 1.50

    # Nothing declares 1.5: it reads as a last-digit slip of 1.4949 and is reported.
    hits = _Recorder(tmp_path, results).hits(paper)
    assert len(hits) == 1 and hits[0].startswith("near_miss:") and "1.5" in hits[0]

    design = {"variables": {"independent": ["R0 (0.9, 1.5, 3.0)"]}}
    assert _Recorder(tmp_path, results, design=design).hits(paper) == []

    topic = "Compare models for R0 in {0.9, 1.5, 3.0}."
    assert _Recorder(tmp_path, results, topic=topic).hits(paper) == []

    # The audit still says the number was read and the paper was clean.
    data = json.loads((tmp_path / "paper" / "numeric_audit.json").read_text("utf-8"))
    assert data["ok"] is True and data["paper_numbers_checked"] >= 2


def test_a_declared_prediction_does_not_hide_a_wrong_result(tmp_path: Path) -> None:
    """The design predicted a 28 nm crossover; the run measured 32. Only the
    setup is declared, so a paper that prints the prediction is still caught."""
    design = {
        "variables": {"controls": ["dose 1.4"]},
        "expected_outcome": "the outbreak probability is 0.58",
    }
    rec = _Recorder(tmp_path, {"outbreak_probability": 0.5749}, design=design)
    hits = rec.hits("The outbreak probability is 0.58.")
    assert len(hits) == 1 and hits[0].startswith("near_miss:")


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
    assert len(warnings) == 1 and warnings[0].startswith("transposed:")
    # The ORACLE's own finding stays advisory, which is what this test is for:
    # it is a near-miss judgement over prose, and a wrong one once cost a full
    # re-design, so it never appears among the blocking hits.
    assert not any(
        str(h).startswith(_ORACLE_KINDS) for h in review["must_flag_hits"]
    ), "the oracle's own finding must not block"
    # 2.41 is also a number nothing in this run accounts for — the experiment
    # computed 2.14 — so the provenance check blocks on it, deliberately. That
    # hit is text-only: it routes back to `write` alone and never re-runs the
    # experiment, which is why it can be forced where the oracle's cannot.
    assert any(
        str(h).startswith("unsourced_number") for h in review["must_flag_hits"]
    ), "a number the run never produced is blocking"


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
    seen: dict = {}

    def pause(**_kw):  # noqa: ANN003, ANN202 -- what a UI reads while the gate waits
        seen["snap"] = json.loads((tmp_path / ".fi" / "human_review.json").read_text("utf-8"))
        return {"action": "accept"}

    eng._pause_for_human = pause  # type: ignore[method-assign]
    state = {
        "iteration": 0,
        "review": {"verdict": "accept", "must_flag_hits": [],
                   "numeric_oracle_warnings": ["transposed: 2.41 vs nils=2.14"]},
    }
    asyncio.run(eng._node_human_feedback(state))  # type: ignore[arg-type]

    snap = seen["snap"]
    assert snap["numeric_oracle_warnings"] == ["transposed: 2.41 vs nils=2.14"]
    assert snap["must_flag_hits"] == [], "kept apart so the UIs can label them"


def test_panel_review_runs_the_check_too(tmp_path: Path) -> None:
    """Panel mode used to skip the oracle entirely, so asking for more
    reviewers meant fewer checks on the numbers."""
    import asyncio

    eng = _review_engine(tmp_path)
    eng.config.engine.review_panel = ["methodologist", "statistician"]
    nodes: list[str] = []

    async def fake_chat(prompt, *, node=None):  # noqa: ANN001
        nodes.append(node or "")
        if node == "review_moderator":
            return json.dumps({"rationale": "both reviewers accept"})
        return json.dumps({"verdict": "accept", "score": 4,
                           "suggestions": [], "must_flag_hits": []})

    eng._chat = fake_chat  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text("Dipole illumination raised NILS to 2.41 at best focus.",
                     encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {},
        "paper_md": str(paper), "result_json": {"nils_dipole": 2.14},
    }))
    review = patch["review"]

    assert any(n.startswith("review_panel.") for n in nodes), "panel path not taken"
    warnings = review.get("numeric_oracle_warnings") or []
    assert len(warnings) == 1 and warnings[0].startswith("transposed:")
    # As on the single-reviewer path: the oracle advises, and never blocks.
    assert not any(
        str(h).startswith(_ORACLE_KINDS) for h in review["must_flag_hits"]
    ), "the oracle's own finding must not block"
    # And the provenance check runs on this path too, so asking for more
    # reviewers still does not mean fewer checks on the numbers.
    assert any(
        str(h).startswith("unsourced_number") for h in review["must_flag_hits"]
    ), "a number the run never produced is blocking"
