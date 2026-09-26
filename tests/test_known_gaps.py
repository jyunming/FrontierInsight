"""Gaps a real quest found and left open (2026-09-13), closed on 2026-09-26.

The ideate seed search and the no-simulation data collection searched with the topic as written: the search engines
behind them match keywords, and a long topic statement found little or nothing there (the literature step already
derived keyword queries; these two did not).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

TOPIC = (
    "I would like to understand how the basic reproduction number of an infectious disease changes the size of "
    "the outbreak it causes in a well-mixed population, using a stochastic model."
)


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic=TOPIC, title="known-gaps", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off", ideate_reflect=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    eng.knowledge.enabled = True  # a knowledge layer to search, without Axon
    eng.knowledge.asearch = AsyncMock(return_value=[])  # type: ignore[method-assign]
    return eng


def test_the_ideate_seed_search_uses_keywords_derived_from_the_topic(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._derive_literature_queries = AsyncMock(return_value=["SIR model final epidemic size", "b", "c"])  # type: ignore[method-assign]
    eng._chat = AsyncMock(return_value='{"ideas": [{"title": "t", "hypothesis": "h"}], "chosen": 0}')  # type: ignore[method-assign]
    asyncio.run(eng._node_ideate({"topic": TOPIC}))  # type: ignore[arg-type]
    assert eng.knowledge.asearch.await_args.args[0] == "SIR model final epidemic size"


def test_the_ideate_seed_search_keeps_the_topic_when_no_keywords_come_back(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._derive_literature_queries = AsyncMock(return_value=[])  # type: ignore[method-assign]
    eng._chat = AsyncMock(return_value='{"ideas": [{"title": "t", "hypothesis": "h"}], "chosen": 0}')  # type: ignore[method-assign]
    asyncio.run(eng._node_ideate({"topic": TOPIC}))  # type: ignore[arg-type]
    assert eng.knowledge.asearch.await_args.args[0] == TOPIC


def test_data_collection_searches_with_the_literature_steps_keywords(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._run_dataset_adapters = AsyncMock(return_value=0)  # type: ignore[method-assign]
    state = {"topic": TOPIC, "design": {"hypothesis": "a larger R0 gives a larger outbreak"},
             "literature_query": "SIR model final epidemic size", "literature": []}
    asyncio.run(eng._node_auto_collect_data(state))  # type: ignore[arg-type]
    assert eng._run_dataset_adapters.await_args.args[0] == "SIR model final epidemic size"
    # With no derived query (an older quest), the topic and hypothesis as before.
    eng._run_dataset_adapters.reset_mock()
    asyncio.run(eng._node_auto_collect_data({**state, "literature_query": ""}))  # type: ignore[arg-type]
    assert eng._run_dataset_adapters.await_args.args[0] == f"{TOPIC} a larger R0 gives a larger outbreak"


# --- the cap check: a cap counts for the quantity it acts on ------------------------------------------------------------

SOFTMAX_THEN_ACCURACY = """
import numpy as np
p = np.clip(logits_softmax, 1e-9, 1.0)
acc = float((pred == y).mean())
print("RESULT_JSON: " + json.dumps({"accuracy": acc}))
"""

RMSE_CAPPED = """
rmse = compute()
rmse = min(rmse, 10.0)
print("RESULT_JSON: " + json.dumps({"rmse": rmse}))
"""


def test_a_cap_on_another_quantity_does_not_make_a_real_value_on_its_bound_clamped() -> None:
    from core.plausibility import Assertion, violations

    found = violations({"accuracy": 1.0}, [Assertion(path="accuracy", min=0.0, max=1.0)], code=SOFTMAX_THEN_ACCURACY)
    assert not any(v.kind == "clamped" for v in found), found


def test_a_value_on_the_cap_of_its_own_quantity_is_clamped_even_inside_the_range() -> None:
    from core.plausibility import Assertion, violations

    (found,) = violations({"rmse": 10.0}, [Assertion(path="rmse", min=0.0, max=100.0)], code=RMSE_CAPPED)
    assert found.kind == "clamped" and "caps this quantity at" in found.describe()
    assert violations({"rmse": 3.2}, [Assertion(path="rmse", min=0.0, max=100.0)], code=RMSE_CAPPED) == []


def test_when_the_script_does_not_say_where_a_quantity_comes_from_every_cap_counts_as_before() -> None:
    from core.plausibility import Assertion, violations

    dynamic = "vals = compute()\nvals = [min(v, 1.0) for v in vals]\nout = {k: v for k, v in zip(names, vals)}\n"
    (found,) = violations({"accuracy": 1.0}, [Assertion(path="accuracy", min=0.0, max=1.0)], code=dynamic)
    assert found.kind == "clamped"


# --- nested comparisons: only groups named for the protocol's factor values -------------------------------------------


def _reps(block: dict) -> list[dict]:
    return [{"by_method": {m: {k: {"mean": v["mean"] + i * 0.01, "max": v["max"] + i * 0.01} for k, v in inner.items()}
                           for m, inner in block.items()}} for i in range(3)]


def test_nested_groups_that_are_not_levels_of_a_factor_are_not_compared() -> None:
    from core.engine import _factor_levels, _result_comparison_stats

    block = {"RK4": {"errors": {"mean": 1.0, "max": 2.0}, "timing": {"mean": 5.0, "max": 9.0}},
             "Euler": {"errors": {"mean": 3.0, "max": 4.0}, "timing": {"mean": 1.0, "max": 2.0}}}
    levels = _factor_levels({"grid": {"method": ["RK4", "Euler"]}})
    out = _result_comparison_stats(_reps(block), factor_levels=levels)
    factors = set(out.get("strata", {}))
    assert "by_method" in factors, "RK4 against Euler: levels of the factor"
    assert not any(f.startswith("by_method.") for f in factors), f"errors against timing is noise: {factors}"
    # Without a protocol grid, as before: the level with one key structure is compared.
    before = _result_comparison_stats(_reps(block))
    assert any(f.startswith("by_method.") for f in before.get("strata", {}))


def test_nested_groups_named_for_factor_values_are_still_compared() -> None:
    from core.engine import _factor_levels, _result_comparison_stats

    reps = [{"by_h": {"0.5": {"rk4": {"err": 1.0 + i * 0.01}, "euler": {"err": 3.0 + i * 0.01}},
                      "0.1": {"rk4": {"err": 0.1 + i * 0.01}, "euler": {"err": 0.9 + i * 0.01}}}} for i in range(3)]
    levels = _factor_levels({"grid": {"h": [0.5, 0.1], "method": ["RK4", "Euler"]}})
    out = _result_comparison_stats(reps, factor_levels=levels)
    assert any(f.startswith("by_h.") for f in out.get("strata", {})), out.get("strata", {}).keys()
