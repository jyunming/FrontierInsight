"""Intervals that use the trials, and a label saying what an interval is an interval of (core/stats.py, the aggregate)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from core import stats
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig,
)
from core.engine import Engine, _aggregate_result_json_replicates as aggregate, _elide_value_lists
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for

AGENTS = Path(__file__).resolve().parent.parent / "agents"


# --- the intervals ------------------------------------------------------------------------------------------------------


def test_the_wilson_interval_reproduces_the_audits_numbers() -> None:
    """R0 = 1.5, N = 5000, three batches of 300 runs: the engine reported a t interval over the three batch
    proportions; the 900 runs behind them give this one."""
    lo, hi = stats.wilson_interval(297, 900)
    assert (round(lo, 4), round(hi, 4)) == (0.3001, 0.3614)
    lo, hi = stats.wilson_interval(round(0.3433 * 300) + round(0.3767 * 300) + round(0.29 * 300), 900)
    assert (round(lo, 4), round(hi, 4)) == (0.3065, 0.3682)


def test_the_wilson_interval_stays_inside_zero_and_one_and_is_none_for_impossible_counts() -> None:
    lo, hi = stats.wilson_interval(0, 300)
    assert lo == 0.0 and 0.0 < hi < 0.02
    lo, hi = stats.wilson_interval(300, 300)
    assert hi == pytest.approx(1.0) and hi <= 1.0 and 0.98 < lo < 1.0
    assert stats.wilson_interval(0, 0) is None
    assert stats.wilson_interval(5, 3) is None
    assert stats.wilson_interval(-1, 10) is None


def test_the_bootstrap_interval_is_deterministic_and_holds_the_mean() -> None:
    values = [float(i % 17) for i in range(400)]
    first = stats.bootstrap_mean_interval(values)
    assert first == stats.bootstrap_mean_interval(values)
    assert first[0] < sum(values) / len(values) < first[1]
    assert stats.bootstrap_mean_interval([1.0]) is None and stats.bootstrap_mean_interval([]) is None
    assert stats.bootstrap_mean_interval([3.0, 3.0, 3.0]) == (3.0, 3.0)


def test_a_very_large_pool_is_thinned_and_still_gives_an_interval() -> None:
    values = [math.sin(i) for i in range(50000)]
    lo, hi = stats.bootstrap_mean_interval(values, cap=1000)
    assert lo < hi and abs((lo + hi) / 2 - sum(values) / len(values)) < 0.2


# --- the aggregate --------------------------------------------------------------------------------------------------------


def _seed(i: int, k: int, n: int = 300, **more: Any) -> dict[str, Any]:
    return {"_seed": i, "p": k / n, "p_count": k, "p_total": n, **more}


def test_counts_beside_a_probability_give_an_interval_from_the_pooled_trials() -> None:
    reps = [_seed(0, 100), _seed(1, 99), _seed(2, 98)]
    result = aggregate(reps)
    entry = result["p"]
    assert "p_count" not in result and "p_total" not in result, "the counts are folded into the probability's entry"
    assert entry["ci_method"] == "wilson_pooled_counts"
    assert (entry["n_trials"], entry["n_successes"]) == (900, 297)
    assert (round(entry["ci_lower"], 4), round(entry["ci_upper"], 4)) == (0.3001, 0.3614)
    assert entry["mean"] == pytest.approx(0.33)
    assert entry["batch_mean"] == pytest.approx((100 + 99 + 98) / 900) and entry["batch_std"] > 0
    assert entry["se"] == pytest.approx(math.sqrt(0.33 * 0.67 / 900))
    assert entry["n"] == 3, "n stays the number of seeds"


def test_a_metric_without_counts_keeps_the_seed_interval_and_says_so() -> None:
    entry = aggregate([{"_seed": 0, "rmse": 0.10}, {"_seed": 1, "rmse": 0.12}, {"_seed": 2, "rmse": 0.11}])["rmse"]
    assert entry["ci_method"] == "t_between_seeds"
    assert entry["ci_lower"] is not None and "n_trials" not in entry


def test_counts_pool_where_the_batches_have_different_sizes() -> None:
    entry = aggregate([_seed(0, 10, 100), _seed(1, 90, 300)])["p"]
    assert entry["n_trials"] == 400 and entry["mean"] == pytest.approx(100 / 400)
    assert entry["batch_mean"] == pytest.approx((0.1 + 0.3) / 2)


def test_counts_are_found_beside_a_probability_inside_nested_results() -> None:
    reps = [
        {"_seed": s, "by_R0": {"1.5": {"N_1000": {"p": k / 300, "p_count": k, "p_total": 300}}}}
        for s, k in enumerate((100, 99, 98))
    ]
    entry = aggregate(reps)["by_R0.1.5.N_1000.p"]
    assert entry["ci_method"] == "wilson_pooled_counts" and entry["n_trials"] == 900


@pytest.mark.parametrize("bad", [
    {"p_count": 5, "p_total": 3},      # more successes than trials
    {"p_count": 1.5, "p_total": 300},  # not a whole number
    {"p_count": 5},                    # no total
    {"p_count": True, "p_total": 300},
])
def test_counts_that_do_not_add_up_are_not_used(bad: dict[str, Any]) -> None:
    reps = [{"_seed": 0, "p": 0.3, **bad}, {"_seed": 1, "p": 0.31, **bad}]
    assert aggregate(reps)["p"]["ci_method"] == "t_between_seeds"


def test_counts_missing_from_one_seed_are_not_pooled() -> None:
    reps = [_seed(0, 100), {"_seed": 1, "p": 0.33}, _seed(2, 98)]
    assert aggregate(reps)["p"]["ci_method"] == "t_between_seeds"


def test_values_beside_a_mean_are_pooled_into_a_bootstrap_interval() -> None:
    reps = [
        {"_seed": s, "size": sum(v) / len(v), "size_values": v}
        for s, v in enumerate(([0.5, 0.6, 0.7] * 10, [0.55, 0.65, 0.6] * 10, [0.5, 0.7, 0.6] * 10))
    ]
    entry = aggregate(reps)["size"]
    assert entry["ci_method"] == "bootstrap_pooled_values" and entry["n_values"] == 90
    assert entry["ci_lower"] < entry["mean"] < entry["ci_upper"]
    assert aggregate(reps)["size"] == entry, "the same pooled values give the same interval"


def test_long_value_lists_are_left_out_of_what_analyze_reads() -> None:
    shown = _elide_value_lists([{"_seed": 0, "size": 0.6, "size_values": [0.1] * 500, "short_values": [1, 2, 3]}])
    assert shown[0]["size_values"] == "<500 values, pooled in aggregate_mean_std>"
    assert shown[0]["short_values"] == [1, 2, 3] and shown[0]["size"] == 0.6


# --- what the model is told ---------------------------------------------------------------------------------------------


def test_the_code_writing_prompts_ask_for_counts_beside_a_probability() -> None:
    for name in ("implement.md", "implement_body.md", "implement_outline.md"):
        text = (AGENTS / name).read_text(encoding="utf-8")
        assert "_count" in text and "_total" in text, name
    assert '"outbreak_probability_count": 99' in (AGENTS / "implement.md").read_text(encoding="utf-8")


def test_analyze_and_write_are_told_what_each_interval_is_an_interval_of() -> None:
    analyze = (AGENTS / "analyze.md").read_text(encoding="utf-8")
    for label in ("wilson_pooled_counts", "bootstrap_pooled_values", "t_between_seeds", "n_trials"):
        assert label in analyze, label
    assert "never call N the sample size" in analyze
    assert "not the number of trials" in (AGENTS / "write.md").read_text(encoding="utf-8")


# --- through the real graph and a real checkpoint -------------------------------------------------------------------------

_COUNTS_CODE = """\
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

seed = int(os.environ.get("FI_REPLICATE_SEED", "0"))
k = 100 - (seed // 1000000) % 3
os.makedirs('figures', exist_ok=True)
plt.figure(); plt.plot([0, 1, 2], [0, 1, 4]); plt.savefig('figures/result.png', dpi=72)
print('RESULT_JSON: ' + json.dumps({"p": k / 300, "p_count": k, "p_total": 300}))
"""


@pytest.mark.asyncio
async def test_analyze_is_shown_the_pooled_interval_and_its_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Implementation":
            return json.dumps({"code": _COUNTS_CODE, "deps": ["matplotlib"]})
        if _classify(prompt) == "Analysis":
            prompts.append(prompt)
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    cfg = Config(
        topic="smoke topic", title="estimand-smoke", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=3),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    )
    artifacts = await Engine(cfg).run()

    assert artifacts.paper_md is not None
    assert prompts, "analyze was not called"
    assert "wilson_pooled_counts" in prompts[0] and '"n_trials": 900' in prompts[0]
    assert '"n_successes": 297' in prompts[0]
