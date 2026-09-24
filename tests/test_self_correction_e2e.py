"""End-to-end self-correction smoke: broken script → reflect → fix → paper.

The execute-repair loop end-to-end. Sequence:
  1. Implement node returns broken code on first call.
  2. Execute runs it; exits with rc != 0 and no RESULT_JSON.
  3. execute_reflect generates a patched code.
  4. Execute runs the patched code; succeeds.
  5. Analyze → cross_check → write → review → paper.md exists on disk.

A separate test exercises the cross-check re-route: analyze returns
`next_step="re_experiment"` on iteration 0; the graph routes back to
design; iteration 1 returns `publish`; quest reaches paper.md.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine


_GOOD_EXPERIMENT = textwrap.dedent("""\
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs('figures', exist_ok=True)
    plt.figure(); plt.plot([0, 1, 2], [0, 1, 4])
    plt.savefig('figures/result.png', dpi=72)
    print('RESULT_JSON: {"score": 0.5}')
""")

# Intentionally broken: references an undefined name; will exit rc != 0
# with no RESULT_JSON line.
_BAD_EXPERIMENT = textwrap.dedent("""\
    import sys
    # `undefined_name` doesn't exist; this raises NameError immediately.
    print(undefined_name)
""")


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        topic="self-correction end-to-end smoke",
        title="self-corr",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=2, review_loop=False,
            clarify_mode="off", ideate_reflect=False,
            exec_reflect_max_iterations=3,
            enable_analyze_reroute=True,
            # Default gate is "after_review"; auto-accept finalises
            # the clean-verdict happy path without a callback.
            auto_accept_on_pass=True,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("attempts", [3, 1], ids=["attempts left", "patch uses the last attempt"])
async def test_full_dag_with_execute_repair_loop(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, attempts: int,
) -> None:
    """Implement returns BAD code first, then GOOD code on the second
    visit (after reflect routes back). Reflect generates the patch.
    The quest reaches paper.md despite the initial failure.

    With one repair attempt the patch is written with the last attempt, and
    it must still run: the router used to go straight to analyze, leaving the
    failed run's empty result and no figure next to code that never ran."""
    cfg.engine.exec_reflect_max_iterations = attempts
    state_flags = {"implement_calls": 0}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        head = prompt.lstrip().splitlines()[0]
        if "scoping a research quest" in prompt[:200]:
            return "{}"
        if "Ideation Reflection" in head:
            return json.dumps({
                "strongest_objection": "x",
                "swap_to": "",
                "refined_rationale": "ok",
            })
        if "Ideation" in head:
            return json.dumps({
                "ideas": [{"title": "t", "summary": "s", "feasibility": "high", "novelty": "low"}],
                "chosen": {"title": "t", "rationale": "r"},
            })
        if "Experiment Design" in head:
            return json.dumps({
                "hypothesis": "h",
                "variables": {"independent": ["x"], "dependent": ["y"], "controls": []},
                "method": "m",
                "expected_outcome": "monotonic",
                "figures_planned": ["result.png"],
                "dependencies": ["matplotlib"],
            })
        if "Implementation" in head:
            # The two-stage implement node calls the LLM twice per
            # design iteration: once for the outline (scaffold +
            # function signatures), once for the body. This mock
            # returns the SAME shape for both — the outline parser
            # then rejects it (no ``scaffold`` key), the body falls
            # through to the legacy single-shot path, and we end up
            # with the broken code we want for the repair-loop test.
            # The execute_reflect node patches it; implement_outline +
            # implement are NOT called again within the repair loop,
            # so this gate fires exactly twice per design iteration.
            state_flags["implement_calls"] += 1
            return json.dumps({"code": _BAD_EXPERIMENT, "deps": []})
        if "Execute-Reflect" in head:
            # The reflect node patches the broken code with a working
            # version. This is the test's heart.
            return json.dumps({
                "code": _GOOD_EXPERIMENT,
                "deps": ["matplotlib"],
                "patch_summary": "replaced undefined-name reference with a working plot+RESULT_JSON",
                "give_up_reason": "",
            })
        if "Analysis" in head:
            return json.dumps({
                "summary": "Curve plotted; trivially monotonic.",
                "key_findings": ["y grows with x"],
                "claims_supported": [], "claims_unsupported": [], "limitations": [],
                "next_step": "publish",
            })
        if "Cross-Paper Check" in head:
            return json.dumps({
                "supporting": [], "conflicting": [], "neutral": [],
                "summary": "no related literature surfaced (knowledge disabled)",
            })
        if "Writing" in head:
            return "# self-corr\n\nResults. ![r](figures/result.png)\n\n## References\n1. Smith 2020.\n"
        if "Review" in head:
            return json.dumps({
                "verdict": "accept", "score": 4, "strengths": [],
                "weaknesses": [], "suggestions": [], "blocking": "",
            })
        return "{}"

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    engine = Engine(cfg)
    artifacts = await engine.run()

    # Paper landed despite the first execute failing.
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    assert (artifacts.figures_dir / "result.png").exists()

    # Repair history captured exactly one patch iteration.
    raw = artifacts.raw_state
    history = raw.get("exec_reflect_history") or []
    assert len(history) == 1
    assert history[0]["iter"] == 1
    assert "undefined" in history[0].get("patch_summary", "").lower() or \
           "RESULT_JSON" in history[0].get("patch_summary", "")
    # Reflect was the LAST visitor; the script that ran successfully
    # was the patched one. The state's `code` reflects the fixed code.
    assert "undefined_name" not in (raw.get("code") or "")
    assert "RESULT_JSON" in (raw.get("code") or "")
    assert raw.get("result_json") == {"score": 0.5}, "the paper was not written from the patched run"
    # Implement gates fired exactly twice (outline + body for ONE
    # design iteration). If design had looped, the count would be 4+.
    assert state_flags["implement_calls"] == 2


@pytest.mark.asyncio
async def test_full_dag_with_analyze_re_experiment_reroute(
    cfg: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analyze returns `next_step="re_experiment"` on iteration 0, then
    `publish` on iteration 1. The graph re-runs design→implement→
    execute on iteration 1 before reaching write. Bounded by
    max_iterations (set to 2 in the fixture)."""

    state_flags = {"analyze_calls": 0, "design_calls": 0}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        head = prompt.lstrip().splitlines()[0]
        if "Ideation" in head and "Reflection" not in head:
            return json.dumps({
                "ideas": [{"title": "t", "summary": "s", "feasibility": "high", "novelty": "low"}],
                "chosen": {"title": "t", "rationale": "r"},
            })
        if "Experiment Design" in head:
            state_flags["design_calls"] += 1
            return json.dumps({
                "hypothesis": "h",
                "variables": {"independent": ["x"], "dependent": ["y"], "controls": []},
                "method": "m",
                "expected_outcome": "monotonic",
                "figures_planned": ["result.png"],
                "dependencies": ["matplotlib"],
            })
        if "Implementation" in head:
            return json.dumps({"code": _GOOD_EXPERIMENT, "deps": ["matplotlib"]})
        if "Execute-Reflect" in head:
            return "{}"  # script succeeded on the first run
        if "Analysis" in head:
            state_flags["analyze_calls"] += 1
            # First visit asks for a re-experiment; second visit publishes.
            if state_flags["analyze_calls"] == 1:
                next_step = "re_experiment"
            else:
                next_step = "publish"
            return json.dumps({
                "summary": "first pass was inconclusive" if next_step == "re_experiment"
                           else "second pass settled it",
                "key_findings": ["y grows with x"],
                "claims_supported": [], "claims_unsupported": [], "limitations": [],
                "next_step": next_step,
                "next_step_reason": "noise dominated" if next_step == "re_experiment"
                                    else "publishable",
            })
        if "Cross-Paper Check" in head:
            return json.dumps({"supporting": [], "conflicting": [], "neutral": [],
                               "summary": "ok"})
        if "Writing" in head:
            return "# rerun\n\nResults.\n\n## References\n1. X 2020.\n"
        if "Review" in head:
            return json.dumps({"verdict": "accept", "score": 4, "strengths": [],
                               "weaknesses": [], "suggestions": [], "blocking": ""})
        return "{}"

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    engine = Engine(cfg)
    artifacts = await engine.run()

    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    # Analyze fired twice (re_experiment → re-run → publish).
    assert state_flags["analyze_calls"] == 2
    # Design fired twice — the re-route DID send us back through design.
    assert state_flags["design_calls"] == 2
    # Final iteration counter was bumped by cross_check on the re-route.
    assert (artifacts.raw_state.get("iteration") or 0) >= 1


@pytest.mark.asyncio
async def test_a_redesigned_script_gets_its_own_repairs(
    cfg: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two live quests spent every repair on their first script, were sent back to design, and the new script's first
    crash then logged "iterations exhausted" with no repair at all: the counter belonged to the old script. Both papers
    were written with no results."""
    flags = {"design_calls": 0, "reflect_after_redesign": 0, "analyze_calls": 0}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        head = prompt.lstrip().splitlines()[0]
        if "Ideation" in head and "Reflection" not in head:
            return json.dumps({"ideas": [{"title": "t", "summary": "s", "feasibility": "high", "novelty": "low"}],
                               "chosen": {"title": "t", "rationale": "r"}})
        if "Experiment Design" in head:
            flags["design_calls"] += 1
            return json.dumps({"hypothesis": "h", "variables": {"independent": ["x"], "dependent": ["y"], "controls": []},
                               "method": "m", "expected_outcome": "monotonic", "figures_planned": ["result.png"],
                               "dependencies": ["matplotlib"]})
        if "Implementation" in head:
            return json.dumps({"code": _BAD_EXPERIMENT, "deps": []})
        if "Execute-Reflect" in head:
            if flags["design_calls"] < 2:  # the first script's repairs all fail
                return json.dumps({"code": _BAD_EXPERIMENT, "deps": [], "patch_summary": "tried", "give_up_reason": ""})
            flags["reflect_after_redesign"] += 1
            return json.dumps({"code": _GOOD_EXPERIMENT, "deps": ["matplotlib"], "patch_summary": "fixed",
                               "give_up_reason": ""})
        if "Analysis" in head:
            flags["analyze_calls"] += 1
            step = "re_experiment" if flags["analyze_calls"] == 1 else "publish"
            return json.dumps({"summary": "s", "key_findings": ["k"], "claims_supported": [], "claims_unsupported": [],
                               "limitations": [], "next_step": step, "next_step_reason": "r"})
        if "Cross-Paper Check" in head:
            return json.dumps({"supporting": [], "conflicting": [], "neutral": [], "summary": "ok"})
        if "Writing" in head:
            return "# rerun\n\nResults. ![r](figures/result.png)\n\n## References\n1. X 2020.\n"
        if "Review" in head:
            return json.dumps({"verdict": "accept", "score": 4, "strengths": [], "weaknesses": [], "suggestions": [],
                               "blocking": ""})
        return "{}"

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    artifacts = await Engine(cfg).run()
    assert flags["design_calls"] == 2, "the first script's failure was sent back to design"
    assert flags["reflect_after_redesign"] >= 1, "the redesigned script got a repair of its own"
    assert artifacts.raw_state.get("result_json") == {"score": 0.5}


@pytest.mark.asyncio
async def test_a_review_that_could_not_run_stops_and_runs_again_on_resume(
    cfg: Config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two live quests hit a usage limit at the review, and FI recorded "accept, score 3": what the person was shown,
    what an automatic accept takes, what a write-back to Axon trusts. Now the quest stops, and the resumed run asks the
    reviewer again."""
    flags = {"review_calls": 0}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        head = prompt.lstrip().splitlines()[0]
        if "Ideation" in head and "Reflection" not in head:
            return json.dumps({"ideas": [{"title": "t", "summary": "s", "feasibility": "high", "novelty": "low"}],
                               "chosen": {"title": "t", "rationale": "r"}})
        if "Experiment Design" in head:
            return json.dumps({"hypothesis": "h", "variables": {"independent": ["x"], "dependent": ["y"], "controls": []},
                               "method": "m", "expected_outcome": "monotonic", "figures_planned": ["result.png"],
                               "dependencies": ["matplotlib"]})
        if "Implementation" in head:
            return json.dumps({"code": _GOOD_EXPERIMENT, "deps": ["matplotlib"]})
        if "Analysis" in head:
            return json.dumps({"summary": "s", "key_findings": ["k"], "claims_supported": [], "claims_unsupported": [],
                               "limitations": [], "next_step": "publish"})
        if "Cross-Paper Check" in head:
            return json.dumps({"supporting": [], "conflicting": [], "neutral": [], "summary": "ok"})
        if "Writing" in head:
            return "# t\n\nResults. ![r](figures/result.png)\n\n## References\n1. X 2020.\n"
        if "Review" in head:
            flags["review_calls"] += 1
            if flags["review_calls"] == 1:
                raise RuntimeError("You've hit your usage limit. Try again at 8:35 PM.")
            return json.dumps({"verdict": "accept", "score": 4, "strengths": [], "weaknesses": [], "suggestions": [],
                               "blocking": ""})
        return "{}"

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    first = Engine(cfg)
    await first.run()
    text = (first.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert "its review could not run" in text and "usage limit" in text
    assert not (first.quest_root / ".fi" / "human_review.json").exists(), "no verdict was recorded"
    assert flags["review_calls"] == 1, "the first run stopped at the failed call"

    artifacts = await Engine(cfg, resume_quest_id=first.quest_id).run()
    assert flags["review_calls"] >= 2, "the resumed run asked the reviewer again"
    review = artifacts.raw_state.get("review") or {}
    assert review.get("score") == 4 and review.get("status") == "ok"
