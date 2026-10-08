"""The quality report shows what each stage's reading cost, and the default pairing shown in docs/USAGE.md is a config FI
accepts and routes. Neutral topics; no model is called."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from core import quality_report as qr
from core.config import Config, KnowledgeConfig, OutputConfig
from core.engine import Engine


def _quest(root: Path) -> Path:
    (root / ".fi").mkdir(parents=True)
    (root / "needs").mkdir()
    rows = [
        {"node": "plan", "provider": "p", "served_model": "model-a", "outcome": "ok", "usage": {"prompt_tokens": 900, "completion_tokens": 400}},
        {"node": "oracle_review", "provider": "p", "served_model": "model-b", "outcome": "ok", "usage": {"prompt_tokens": 300, "completion_tokens": 100}},
        {"node": "design", "provider": "p", "served_model": "model-a", "outcome": "ok"},
        {"node": "design_review", "provider": "p", "served_model": "model-b", "outcome": "ok", "usage": {"prompt_tokens": 200, "completion_tokens": 50}},
        {"node": "design_review", "provider": "p", "served_model": "model-b", "outcome": "ReadTimeout"},
        {"node": "implement", "provider": "p", "served_model": "model-a", "outcome": "ok"},
        {"node": "oracle_review.code", "provider": "p", "served_model": "model-b", "outcome": "ok", "usage": {"prompt_tokens": 1000, "completion_tokens": 120}},
        {"node": "code_review", "provider": "p", "served_model": "model-b", "outcome": "ok", "usage": {"prompt_tokens": 500, "completion_tokens": 60}},
    ]
    (root / ".fi" / "model_calls.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (root / ".fi" / "cost.summary.json").write_text(json.dumps({"by_node": {
        "oracle_review": {"cost_usd": 0.01}, "oracle_review.code": {"cost_usd": 0.04}, "code_review": {"cost_usd": 0.02},
        "design_review": {"cost_usd": None}}}), encoding="utf-8")
    (root / "paper").mkdir()
    (root / "paper" / "paper.md").write_text("x", encoding="utf-8")
    return root


def test_the_report_shows_what_each_stages_reading_cost(tmp_path: Path) -> None:
    report = qr.build(_quest(tmp_path / "q"))
    cost = {s["stage"]: s["review_cost"] for s in report["stages"]}
    assert cost["checks"] == {"calls": 1, "failed_calls": 0, "prompt_tokens": 300, "completion_tokens": 100, "usd": 0.01}
    assert cost["design"]["calls"] == 2 and cost["design"]["failed_calls"] == 1 and cost["design"]["usd"] is None
    # the code's reading is its own: the call made under the checks' step for the code is not counted for the checks
    assert cost["code"] == {"calls": 2, "failed_calls": 0, "prompt_tokens": 1500, "completion_tokens": 180, "usd": 0.06}
    assert cost["plan"] is None and cost["run"] is None
    text = qr.render(report)
    assert "| Reading cost |" in text and "2 calls, 1,500 / 180 tokens, $0.06" in text and "1 call, 300 / 100 tokens, $0.01" in text


def test_the_default_pairing_in_the_usage_guide_is_a_config_that_routes_all_three_reviewers(tmp_path: Path) -> None:
    usage = (Path(__file__).resolve().parents[1] / "docs" / "USAGE.md").read_text(encoding="utf-8")
    block = re.search(r"### One model writes, another model reviews.*?```yaml\n(.*?)```", usage, re.S)
    assert block, "the pairing example is in docs/USAGE.md"
    provider = yaml.safe_load(block.group(1))["provider"]
    cfg = Config(topic="a mass on a damped spring", title="spring", knowledge=KnowledgeConfig(enabled=False),
                 output=OutputConfig(output_dir=tmp_path / "out"), provider=provider)
    engine = Engine(cfg)
    assert [engine._review_node(s) for s in ("design_review", "code_review")] == ["design_review", "code_review"]
    for step in ("oracle_review", "design_review", "code_review"):
        assert engine._model_for_node(step) == "reviewer-model", step
    assert engine._model_for_node("plan") is None and engine._model_for_node("implement") is None
