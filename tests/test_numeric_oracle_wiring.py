"""The check on the results that stays (a reference quantity that is 0 at every setting) must reach a HUMAN, and must not force
a rewrite.

The paper's numbers are no longer compared with the results by this module (core/typed_numbers.py and the keys of
core/paper_values.py do that); ``trivial_reference_findings`` is a check on the results themselves. Its findings are ADVISORY:
recorded in ``review["numeric_oracle_warnings"]``, copied into the human-review snapshot every interface renders, and kept OUT
of ``must_flag_hits``. A correct checker that nobody sees still changes nothing, which is why the snapshot test below matters
as much as the router one.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from core.engine import Engine

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


class _Recorder:
    """Minimal stand-in exposing only what the helper touches."""

    def __init__(self, tmp: Path, result_json: dict, **state) -> None:
        self.quest_root = tmp
        self._log = SimpleNamespace(
            warning=lambda *a, **k: self.msgs.append(("warn", a)),
            info=lambda *a, **k: self.msgs.append(("info", a)),
        )
        self.msgs: list = []
        self.state = {"result_json": result_json, **state}

    def hits(self) -> list[str]:
        return Engine._reference_check_hits(self, self.state)  # type: ignore[arg-type]


def test_a_trivial_reference_is_not_called_an_unverified_number(tmp_path: Path) -> None:
    """``deterministic_final_size`` WAS computed and exported correctly; what is suspect is the bracket the root finder was
    handed, which is exactly what the finding's own sentence says."""
    hits = _Recorder(tmp_path, _TRIVIAL_ROOT_RESULTS).hits()
    assert len(hits) == 1
    assert hits[0].startswith("trivial_reference:")
    assert not hits[0].startswith("unverified_number")
    assert "check the bracket" in hits[0]


def test_a_study_with_no_results_is_not_blocked(tmp_path: Path) -> None:
    """A literature synthesis runs no experiment: the check stays out of its way."""
    assert _Recorder(tmp_path, {}).hits() == []
    assert not (tmp_path / "paper" / "numeric_audit.json").exists(), "no audit file of the paper's numbers is written any more"


def test_checker_failure_is_never_quest_fatal(tmp_path: Path, monkeypatch) -> None:
    from core import numeric_oracle

    def boom(*_a, **_k):
        raise RuntimeError("oracle exploded")

    monkeypatch.setattr(numeric_oracle, "trivial_reference_findings", boom)
    rec = _Recorder(tmp_path, _TRIVIAL_ROOT_RESULTS)
    assert rec.hits() == []
    assert any(kind == "warn" for kind, _ in rec.msgs)


def test_router_treats_the_hit_as_blocking() -> None:
    """Pin the router's contract for must-flag hits in general: a non-empty list forces revise even when review_loop is off.
    The check on the results deliberately no longer does."""
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
    assert Engine._route_after_review(engine, state) == "rewrite"  # type: ignore[arg-type]


def test_max_iterations_zero_still_bypasses_the_gate() -> None:
    """Known limitation, pinned so it is a decision rather than a surprise: a zero budget disables the blocking behaviour."""
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
    """Drives the real review node, so the merge itself is under test. The reviewer accepts; the results hold a reference
    that is 0 at every setting."""
    eng = _review_engine(tmp_path)

    async def fake_chat(prompt, *, node=None):  # noqa: ANN001
        return json.dumps({"verdict": "accept", "score": 4, "suggestions": [], "must_flag_hits": []})

    eng._chat = fake_chat  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text("The deterministic limit is approached as N grows.", encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {}, "paper_md": str(paper), "result_json": _TRIVIAL_ROOT_RESULTS}))
    review = patch["review"]
    warnings = review.get("numeric_oracle_warnings") or []
    assert len(warnings) == 1 and warnings[0].startswith("trivial_reference:")
    assert not any(str(h).startswith("trivial_reference:") for h in review["must_flag_hits"]), "the finding must not block"


def test_clean_results_leave_no_warning_key(tmp_path: Path) -> None:
    eng = _review_engine(tmp_path)

    async def fake_chat(prompt, *, node=None):  # noqa: ANN001
        return json.dumps({"verdict": "accept", "score": 4, "suggestions": []})

    eng._chat = fake_chat  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text("Dipole illumination raised NILS.", encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {}, "paper_md": str(paper), "result_json": {"nils_dipole": 2.14}}))
    assert "numeric_oracle_warnings" not in patch["review"]


def test_human_review_snapshot_carries_the_warnings(tmp_path: Path) -> None:
    """The snapshot is what the web UI and the VSCode chat panel render. A warning that never reaches it is invisible in two
    of three interfaces."""
    eng = _review_engine(tmp_path, gate="after_review")
    seen: dict = {}

    def pause(**_kw):  # noqa: ANN003, ANN202 -- what a UI reads while the gate waits
        seen["snap"] = json.loads((tmp_path / ".fi" / "human_review.json").read_text("utf-8"))
        return {"action": "accept", "answer": "yes"}

    eng._pause_for_human = pause  # type: ignore[method-assign]
    state = {
        "iteration": 0,
        "review": {"verdict": "accept", "must_flag_hits": [],
                   "numeric_oracle_warnings": ["trivial_reference: `deterministic_final_size` is exactly 0"]},
    }
    asyncio.run(eng._node_human_feedback(state))  # type: ignore[arg-type]

    snap = seen["snap"]
    assert snap["numeric_oracle_warnings"] == ["trivial_reference: `deterministic_final_size` is exactly 0"]
    assert snap["must_flag_hits"] == [], "kept apart so the UIs can label them"


def test_panel_review_runs_the_check_too(tmp_path: Path) -> None:
    """Panel mode must not mean fewer checks."""
    eng = _review_engine(tmp_path)
    eng.config.engine.review_panel = ["methodologist", "statistician"]
    nodes: list[str] = []

    async def fake_chat(prompt, *, node=None):  # noqa: ANN001
        nodes.append(node or "")
        if node == "review_moderator":
            return json.dumps({"rationale": "both reviewers accept"})
        return json.dumps({"verdict": "accept", "score": 4, "suggestions": [], "must_flag_hits": []})

    eng._chat = fake_chat  # type: ignore[assignment]
    paper = tmp_path / "paper.md"
    paper.write_text("The deterministic limit is approached as N grows.", encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": "t", "iteration": 0, "review": {}, "paper_md": str(paper), "result_json": _TRIVIAL_ROOT_RESULTS}))
    review = patch["review"]
    assert any(n.startswith("review_panel.") for n in nodes), "panel path not taken"
    warnings = review.get("numeric_oracle_warnings") or []
    assert len(warnings) == 1 and warnings[0].startswith("trivial_reference:")
    assert not any(str(h).startswith("trivial_reference:") for h in review["must_flag_hits"])
