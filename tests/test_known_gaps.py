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
             "literature_query": "SIR model final epidemic size", "literature_query_derived": True, "literature": []}
    asyncio.run(eng._node_auto_collect_data(state))  # type: ignore[arg-type]
    assert eng._run_dataset_adapters.await_args.args[0] == "SIR model final epidemic size"
    # With no derived query (an older quest), the topic and hypothesis as before.
    eng._run_dataset_adapters.reset_mock()
    asyncio.run(eng._node_auto_collect_data({**state, "literature_query": ""}))  # type: ignore[arg-type]
    assert eng._run_dataset_adapters.await_args.args[0] == f"{TOPIC} a larger R0 gives a larger outbreak"


# --- the cap check: a cap that plainly acts on another quantity does not count ------------------------------------------

from core.plausibility import Assertion, violations  # noqa: E402

ACC = [Assertion(path="accuracy", min=0.0, max=1.0)]
RMSE = [Assertion(path="rmse", min=0.0, max=100.0)]


def _kinds(found: list) -> list[str]:
    return [v.kind for v in found]


def test_a_softmax_clipped_to_one_does_not_make_a_real_accuracy_of_one_clamped() -> None:
    code = (
        "import numpy as np\n"
        "p = np.clip(np.exp(z) / np.exp(z).sum(), 1e-9, 1.0)\n"
        "acc = float(np.mean(pred == y))\n"
        'print("RESULT_JSON: " + json.dumps({"accuracy": acc}))\n'
    )
    assert "clamped" not in _kinds(violations({"accuracy": 1.0}, ACC, code=code))


def test_function_and_module_names_do_not_tie_a_cap_to_a_quantity() -> None:
    # The reviewer's case: `int(np.sqrt(n))` capped at 1 and a ratio of 1.0 both "read" np, float, int.
    code = (
        "import numpy as np\n"
        "w = max(1, int(np.sqrt(n)))\n"
        'rows.append({"rel_rmse": float(np.mean(e) / base)})\n'
    )
    assert violations({"rel_rmse": 1.0}, [Assertion(path="rel_rmse", min=0.0, max=10.0)], code=code) == []


def test_a_cap_inside_the_reported_value_still_counts() -> None:
    code = 'print("RESULT_JSON: " + json.dumps({"rmse": min(rmse, 10.0)}))\n'
    (found,) = violations({"rmse": 10.0}, [Assertion(path="rmse", min=0.0, max=10.0)], code=code)
    assert found.kind == "clamped"
    (inside,) = violations({"rmse": 10.0}, RMSE, code=code)
    assert inside.kind == "clamped" and "inside its range" in inside.describe()


def test_a_cap_assigned_to_the_reported_variable_catches_a_value_inside_the_range() -> None:
    code = "rmse = compute()\nrmse = min(rmse, 10.0)\n" 'print("RESULT_JSON: " + json.dumps({"rmse": rmse}))\n'
    (found,) = violations({"rmse": 10.0}, RMSE, code=code)
    assert found.kind == "clamped"
    assert violations({"rmse": 3.2}, RMSE, code=code) == []


def test_a_cap_set_under_the_key_when_a_run_diverges_counts() -> None:
    code = 'results = {"rmse": rmse}\nif diverged:\n    results["rmse"] = 10.0\n'
    (found,) = violations({"rmse": 10.0}, RMSE, code=code)
    assert found.kind == "clamped"


def test_a_cap_outside_any_assignment_counts_for_every_quantity_as_before() -> None:
    code = "errs.append(min(e, 1.0))\nout = {k: v for k, v in zip(names, vals)}\n"
    (found,) = violations({"accuracy": 1.0}, ACC, code=code)
    assert found.kind == "clamped"


def test_a_generic_key_keeps_the_old_behaviour() -> None:
    code = "p = np.clip(q, 0.0, 1.0)\nsummary = {'mean': float(np.mean(acc))}\n"
    (found,) = violations({"mean": 1.0}, [Assertion(path="mean", min=0.0, max=1.0)], code=code)
    assert found.kind == "clamped", "'mean' does not say which quantity: every cap counts"


# --- nested comparisons: a level of declared metrics is not a level of a factor ----------------------------------------


def test_nested_groups_named_for_declared_metrics_are_not_compared() -> None:
    from core.engine import _result_comparison_stats

    reps = [{"by_method": {m: {"errors": {"mean": e + i * 0.01, "max": e + 1}, "timing": {"mean": t + i * 0.01, "max": t + 2}}
                           for m, e, t in (("RK4", 1.0, 5.0), ("Euler", 3.0, 1.0))}} for i in range(3)]
    out = _result_comparison_stats(reps, metric_ids={"errors", "timing"})
    factors = set(out.get("strata", {}))
    assert "by_method" in factors and not any(f.startswith("by_method.") for f in factors), factors
    # With no declared metrics, as before.
    assert any(f.startswith("by_method.") for f in _result_comparison_stats(reps).get("strata", {}))


def test_methods_at_each_step_size_are_still_compared() -> None:
    """The comparison the nesting exists for: a numeric grid (h), with the methods one level below it."""
    from core.engine import _result_comparison_stats

    reps = [{"by_h": {"0.5": {"RK4": {"err": 1.0 + i * 0.01}, "Euler": {"err": 3.0 + i * 0.01}},
                      "0.1": {"RK4": {"err": 0.1 + i * 0.01}, "Euler": {"err": 0.9 + i * 0.01}}}} for i in range(3)]
    out = _result_comparison_stats(reps, metric_ids={"err"})
    assert {"by_h.0.5", "by_h.0.1"} <= set(out.get("strata", {})), out.get("strata", {}).keys()


# --- data collection: keywords to search, the question to judge relevance ------------------------------------------


def test_data_collection_judges_relevance_against_the_question_not_the_keywords(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._run_dataset_adapters = AsyncMock(return_value=0)  # type: ignore[method-assign]
    eng._filter_relevant_docs = AsyncMock(return_value=[])  # type: ignore[method-assign]
    lit = [{"metadata": {"title": "A paper", "url": "https://x"}, "content": "c"}]
    state = {"topic": TOPIC, "design": {"hypothesis": "h"}, "literature_query": "SIR final size",
             "literature_query_derived": True, "literature": lit}
    asyncio.run(eng._node_auto_collect_data(state))  # type: ignore[arg-type]
    assert eng._filter_relevant_docs.await_args_list[0].args[0] == f"{TOPIC} h"
    assert eng._run_dataset_adapters.await_args.args[0] == "SIR final size"


def test_a_fallback_query_is_not_taken_for_keywords(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng._run_dataset_adapters = AsyncMock(return_value=0)  # type: ignore[method-assign]
    state = {"topic": TOPIC, "design": {"hypothesis": "h"}, "literature_query": "title " + TOPIC[:200],
             "literature_query_derived": False, "literature": []}
    asyncio.run(eng._node_auto_collect_data(state))  # type: ignore[arg-type]
    assert eng._run_dataset_adapters.await_args.args[0] == f"{TOPIC} h"
