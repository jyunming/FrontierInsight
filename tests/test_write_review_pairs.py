"""One model writes, a different model reviews (core/review_pairs.py): each reviewed stage (the plan's checks, the design,
the code) has its own step to route, the review finds problems (each with where, what, why, how to fix) and lists what it
checked, and silence is never approval. Fake models and neutral topics (a mass on a damped spring); no real model is called."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core import attempt_records as ar
from core import review_pairs as pairs
from core.config import Config, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from tests.test_step_providers import KIMI, MAIN_URL, STEP_URL, _Fake, _connected, fake  # noqa: F401


def _engine(tmp_path: Path, **provider: Any) -> Engine:
    base: dict[str, Any] = dict(name="openai", model="main-model", base_url=MAIN_URL, api_key_env="MAIN_TEST_KEY")
    base.update(provider)
    cfg = Config(topic="a mass on a damped spring", title="spring", knowledge=KnowledgeConfig(enabled=False),
                 output=OutputConfig(output_dir=tmp_path / "out"), provider=ProviderConfig.model_validate(base))
    return Engine(cfg)


# --- part 1: one step per reviewer -------------------------------------------------------------------------------------


def test_a_reviewer_keeps_the_step_it_had_until_a_person_sets_its_own(tmp_path: Path) -> None:
    unset = _engine(tmp_path)
    assert unset._review_node("design_review") == "design_self_critique"
    assert unset._review_node("code_review") == "oracle_review.code"
    # the plan's checks are read by `oracle_review`; the code follows it while `code_review` is not set
    follows = _engine(tmp_path, node_models={"oracle_review": "reader-model"})
    assert follows._review_node("code_review") == "oracle_review.code"
    assert follows._model_for_node(follows._review_node("code_review")) == "reader-model"
    assert follows._model_for_node(follows._review_node("design_review")) is None  # the design stays on the main model


@pytest.mark.parametrize("block", ["node_models", "node_providers"])
def test_each_reviewer_is_routed_on_its_own(tmp_path: Path, block: str) -> None:
    entry: Any = (lambda m: m) if block == "node_models" else (lambda m: {**KIMI, "model": m})
    engine = _engine(tmp_path, **{block: {"design_review": entry("design-reader"), "code_review": entry("code-reader"),
                                          "oracle_review": entry("checks-reader")}})
    assert engine._review_node("design_review") == "design_review"
    assert engine._review_node("code_review") == "code_review"
    assert engine._model_for_node("design_review") == "design-reader"
    assert engine._model_for_node("code_review") == "code-reader"
    assert engine._model_for_node("oracle_review") == "checks-reader"
    only_design = _engine(tmp_path, **{block: {"design_review": entry("design-reader")}})
    assert only_design._review_node("code_review") == "oracle_review.code"
    assert only_design._model_for_node("oracle_review") is None


@pytest.mark.asyncio
async def test_a_reviewers_calls_go_to_its_providers_and_are_recorded_under_its_step(tmp_path: Path, fake: _Fake) -> None:  # noqa: F811
    engine = await _connected(_engine(tmp_path, node_providers={"design_review": KIMI, "oracle_review": KIMI}).config)
    try:
        await engine._chat("p", node=engine._review_node("design_review"))
        await engine._chat("p", node=engine._review_node("code_review"))   # follows oracle_review (another provider)
        await engine._chat("p", node=engine._review_node("code_review") + "")
        await engine._chat("p", node="design")
    finally:
        await engine._close_step_clients()
        await engine._client.aclose()
    calls = ar.read(engine.fi_dir, ar.MODEL_CALLS)
    assert [(c["node"], c["served_model"]) for c in calls] == [
        ("design_review", "kimi-k3"), ("oracle_review.code", "kimi-k3"), ("oracle_review.code", "kimi-k3"),
        ("design", "main-model")]
    assert pairs.READER_STEP == {"checks": "oracle_review", "design": "design_review", "code": "code_review", "paper": "citation_check"}


def test_the_quality_report_names_the_new_steps(tmp_path: Path) -> None:
    from core import quality_report as qr

    assert qr._stage_of("design_review") == "design" and qr._stage_of("code_review") == "code"
    assert qr._stage_of("oracle_review.code") == "code" and qr._stage_of("oracle_review.recompute") == "checks"
    assert qr._reads("code", "oracle_review.code") and not qr._reads("checks", "oracle_review.code")
    assert qr._reads("checks", "oracle_review") and qr._reads("design", "design_review")


# --- part 2: the design is read by a different model -------------------------------------------------------------------

import json  # noqa: E402

DESIGN = {"hypothesis": "a heavier mass settles later", "variables": {"independent": ["mass"], "dependent": ["settling_time"],
                                                                       "controls": ["damping"]},
          "method": "simulate the mass at three settings", "expected_outcome": "later settling", "figures_planned": ["a.png"],
          "dependencies": ["numpy"]}
AMENDED = {**DESIGN, "method": "simulate the mass at three settings and repeat each with five random starts"}
OBJECTION = {"check": "precision", "where": "method", "objection": "one run per setting", "why": "no spread is estimated",
             "fix": "repeat each setting", "severity": "high"}


class _Seen:
    """A stand-in model client: answers the design review with ``reply`` and keeps the model each call asked for."""

    def __init__(self, reply: Any) -> None:
        self.reply, self.models, self.prompts = reply, [], []

    async def chat(self, messages, **kw):  # noqa: ANN001, ANN201
        self.models.append(kw.get("model"))
        self.prompts.append(messages[-1]["content"])
        reply = self.reply.pop(0) if isinstance(self.reply, list) else self.reply
        return reply if isinstance(reply, str) else json.dumps(reply)

    async def aclose(self) -> None:
        return None


def _critique_entry(engine: Engine) -> dict[str, Any]:
    return json.loads((engine.quest_root / "needs" / "DESIGN_CRITIQUE.json").read_text(encoding="utf-8"))[-1]


@pytest.mark.asyncio
async def test_a_different_model_reads_the_design_and_its_objections_are_adopted_as_before(tmp_path: Path) -> None:
    engine = _engine(tmp_path, node_models={"design_review": "reviewer-model"})
    client = _Seen({"objections_addressed": [OBJECTION], "amended_design": AMENDED,
                    "checked_no_problem": [{"check": "circular_evaluation", "note": "the evaluator is separate"}]})
    engine._client = client
    design, objections = await engine._audit_design({"topic": "a mass on a spring", "iteration": 0}, dict(DESIGN))
    assert design["method"] == AMENDED["method"] and objections == [OBJECTION]   # the existing adoption rule
    assert client.models == ["reviewer-model"]
    review = _critique_entry(engine)["review"]
    assert review["kind"] == "independent_review" and review["same_model"] is False
    assert review["reviewer"] == "reviewer-model" and review["writer"] == "main-model" and review["step"] == "design_review"
    # a design that is already settled is read by the same reviewer and nothing it proposes is taken
    settled, _ = await engine._audit_design({"topic": "t", "iteration": 0}, dict(DESIGN), adopt=False)
    assert settled["method"] == DESIGN["method"]


@pytest.mark.asyncio
async def test_the_main_model_reading_its_own_design_is_a_self_review_never_an_independent_one(tmp_path: Path) -> None:
    engine = _engine(tmp_path)   # no design_review: the main model audits, as before
    engine._client = _Seen({"objections_addressed": [OBJECTION], "amended_design": AMENDED})
    await engine._audit_design({"topic": "a mass on a spring", "iteration": 0}, dict(DESIGN))
    review = _critique_entry(engine)["review"]
    assert review["kind"] == "self_review" and review["same_model"] is True and review["step"] == "design_self_critique"
    # the same model named for the reader is still the same model
    named_same = _engine(tmp_path / "b", node_models={"design_review": "main-model"})
    named_same._client = _Seen({"objections_addressed": [OBJECTION], "amended_design": AMENDED})
    await named_same._audit_design({"topic": "t", "iteration": 0}, dict(DESIGN))
    assert _critique_entry(named_same)["review"]["kind"] == "self_review"
    # and a design written by a routed step and read by the main model is read by a different model
    routed = _engine(tmp_path / "c", node_models={"design": "writer-model"})
    routed._client = _Seen({"objections_addressed": [], "checked_no_problem": [{"check": "oracle", "note": "present"}]})
    await routed._audit_design({"topic": "t", "iteration": 0}, dict(DESIGN))
    assert _critique_entry(routed)["review"]["kind"] == "independent_review"


def test_the_quality_report_tells_an_independent_design_review_from_a_self_review(tmp_path: Path) -> None:
    from core import quality_report as qr

    def quest(name: str, kind: str, same: bool) -> Path:
        root = tmp_path / name
        (root / ".fi").mkdir(parents=True)
        (root / "needs").mkdir()
        rows = [{"node": "design", "provider": "p", "served_model": "model-a", "outcome": "ok"},
                {"node": "design_review", "provider": "p", "served_model": "model-a" if same else "model-b", "outcome": "ok",
                 "usage": {"prompt_tokens": 100, "completion_tokens": 40}}]
        (root / ".fi" / "model_calls.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        (root / "needs" / "DESIGN_CRITIQUE.json").write_text(json.dumps([{
            "status": "ok", "objections": [OBJECTION], "review": {"kind": kind, "same_model": same}}]), encoding="utf-8")
        return root

    other = qr.build(quest("other", "independent_review", False))
    design = next(s for s in other["stages"] if s["stage"] == "design")
    assert design["same_model"] is False and design["reviewed_by"] == "p/model-b"
    assert design["problems"]["items"][0]["found_by"].endswith("(design review)")
    own = qr.build(quest("own", "self_review", True))
    design = next(s for s in own["stages"] if s["stage"] == "design")
    assert design["same_model"] is True and design["problems"]["items"][0]["found_by"].endswith("(design self-critique)")
    assert "design_review" in " ".join(s["text"] for s in own["suggestions"])
