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


# --- the cap check: every cap still counts on a bound; a cap written where the quantity is reported counts inside it -

from core.plausibility import Assertion, violations  # noqa: E402

ACC = [Assertion(path="accuracy", min=0.0, max=1.0)]
RMSE = [Assertion(path="rmse", min=0.0, max=100.0)]
RMSE_10 = [Assertion(path="rmse", min=0.0, max=10.0)]


def _kinds(found: list) -> list[str]:
    return [v.kind for v in found]


def test_a_cap_written_where_the_quantity_is_reported_catches_a_value_inside_the_range() -> None:
    code = 'print("RESULT_JSON: " + json.dumps({"rmse": min(rmse, 10.0)}))\n'
    (inside,) = violations({"rmse": 10.0}, RMSE, code=code)
    assert inside.kind == "clamped" and "inside its range" in inside.describe()
    assert violations({"rmse": 3.2}, RMSE, code=code) == []
    (on_bound,) = violations({"rmse": 10.0}, RMSE_10, code=code)
    assert on_bound.kind == "clamped"


def test_what_is_not_a_cap_where_the_quantity_is_reported_is_not_taken_for_one() -> None:
    # A guard against dividing by zero, an axis, an indicator: none of them caps the reported value.
    ratio = [Assertion(path="events_per_run", min=0.0, max=10.0)]
    assert violations({"events_per_run": 1.0}, ratio, code='out = {"events_per_run": total / max(n_runs, 1)}\n') == []
    peak = [Assertion(path="peak", min=0.0, max=10.0)]
    assert violations({"peak": 1.0}, peak, code='out = {"peak": float(np.max(x, axis=1).mean())}\n') == []
    peaks = [Assertion(path="n_peaks", min=0.0, max=50.0)]
    code = 'out = {"n_peaks": int(np.where(np.diff(np.sign(np.diff(I))) < 0, 1, 0).sum())}\n'
    assert violations({"n_peaks": 1.0}, peaks, code=code) == []


def test_every_cap_in_the_script_still_counts_for_a_value_on_its_bound() -> None:
    """The common shapes a narrower tie missed: cap each trial, then report an aggregate."""
    for code in (
        'import numpy as np\nerrs = np.minimum(errs, 10.0)\nprint(json.dumps({"rmse": float(errs.mean())}))\n',
        'errs = np.minimum(errs, 10.0)\nrmse = float(np.sqrt(np.mean(errs**2)))\nprint(json.dumps({"rmse": rmse}))\n',
        'rmses.append(min(e, 10.0))\nprint(json.dumps({"rmse": float(np.mean(rmses))}))\n',
    ):
        (found,) = violations({"rmse": 10.0}, RMSE_10, code=code)
        assert found.kind == "clamped", code


def test_a_generic_key_gets_no_cap_of_its_own() -> None:
    code = "summary = {'mean': min(float(np.mean(acc)), 0.5)}\n"
    assert violations({"mean": 0.5}, [Assertion(path="mean", min=0.0, max=1.0)], code=code) == []


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


def test_a_nested_level_with_one_declared_metric_among_its_groups_is_not_compared() -> None:
    from core.engine import _result_comparison_stats

    reps = [{"by_method": {m: {"errors": {"mean": e + i * 0.01}, "timing": {"mean": t + i * 0.01}}
                           for m, e, t in (("RK4", 1.0, 5.0), ("Euler", 3.0, 1.0))}} for i in range(3)]
    out = _result_comparison_stats(reps, metric_ids={"errors"})
    assert not any(f.startswith("by_method.") for f in out.get("strata", {}))


def test_a_value_put_under_its_key_when_a_run_diverges_counts_on_its_bound() -> None:
    for code in (
        'results = {"rmse": rmse}\nif diverged:\n    results["rmse"] = 10.0\n',
        'CAP = 10.0\nresults = {"rmse": rmse}\nif diverged:\n    results["rmse"] = CAP\n',
    ):
        (found,) = violations({"rmse": 10.0}, RMSE_10, code=code)
        assert found.kind == "clamped", code


def test_only_an_upper_cap_where_the_quantity_is_reported_counts_inside_the_range() -> None:
    upper = 'print(json.dumps({"rmse": np.clip(r, 0.5, 10.0)}))\n'
    (found,) = violations({"rmse": 10.0}, RMSE, code=upper)
    assert found.kind == "clamped"
    assert violations({"rmse": 0.5}, RMSE, code=upper) == [], "0.5 is its floor, not a cap"
    counts = [Assertion(path="n_peaks", min=0.0, max=50.0)]
    for code in (
        'out = {"n_peaks": int((x.max(1) > thr).sum())}\n',  # an axis, written as a position
        'out = {"n_peaks": int(np.max(x, 1).sum())}\n',
        'out = {"n_peaks": max(1, len(set(labels)) - 1)}\n',  # a floor
    ):
        assert violations({"n_peaks": 1.0}, counts, code=code) == [], code
    config = (
        'params = {"n_trials": 300}\nif os.environ.get("FI_PILOT"):\n    params["n_trials"] = 20\n'
        'else:\n    params["n_trials"] = 300\nresults = {"n_trials": params["n_trials"]}\n'
    )
    trials = [Assertion(path="n_trials", min=1.0, max=1e5)]
    assert violations({"n_trials": 300.0}, trials, code=config) == []


def test_a_clip_is_read_by_its_arguments_not_its_module_name() -> None:
    probs = [Assertion(path="p_hat", min=0.0, max=1.0)]
    for code in ('out = {"p_hat": float(cp.clip(p, 0.05, 0.95))}\nprint(json.dumps(out))\n', 'out = {"p_hat": float(jax.numpy.clip(p, 0.05, 0.95))}\nprint(json.dumps(out))\n'):
        assert violations({"p_hat": 0.05}, probs, code=code) == [], "0.05 is the floor"
        (found,) = violations({"p_hat": 0.95}, probs, code=code)
        assert found.kind == "clamped", code
    # Two positional arguments read as no cap: the method's cap (t.clamp(lo, HI)) and the function's floor
    # (torch.clamp(x, lo)) look alike.
    assert violations({"p_hat": 0.05}, probs, code='out = {"p_hat": float(torch.clamp(p, 0.05))}\nprint(json.dumps(out))\n') == []
    assert violations({"p_hat": 0.95}, probs, code='out = {"p_hat": float(t.clamp(0.05, 0.95))}\nprint(json.dumps(out))\n') == []


def test_a_cap_written_first_counts_too() -> None:
    for code in ('out = {"rmse": min(10.0, rmse)}\nprint(json.dumps(out))\n', 'out = {"rmse": float(np.minimum(10.0, r))}\nprint(json.dumps(out))\n'):
        (found,) = violations({"rmse": 10.0}, RMSE, code=code)
        assert found.kind == "clamped", code


def test_a_cap_in_a_denominator_is_a_guard_not_a_cap() -> None:
    ratio = [Assertion(path="rate", min=0.0, max=10.0)]
    assert violations({"rate": 1.0}, ratio, code='out = {"rate": total / min(max(n, 1), 1.0)}\nprint(json.dumps(out))\n') == []


def test_a_min_of_two_computed_values_is_no_cap_and_minimum_with_out_still_is() -> None:
    assert violations({"rmse": 4.0}, RMSE, code='out = {"rmse": min(a, b)}\nprint(json.dumps(out))\n') == []
    (found,) = violations({"rmse": 10.0}, RMSE, code='out = {"rmse": float(np.minimum(r, 10.0, out=r))}\nprint(json.dumps(out))\n')
    assert found.kind == "clamped"

