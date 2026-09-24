"""Metric specs (core/metric_spec.py) and the sampling distributions that go with each estimator (core/stats.py)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pytest

from core import metric_spec as ms
from core import plan, stats
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
from core.engine import Engine
from tests.test_engine_smoke import _FAKE_RESPONSES, _classify, _fake_response_for

# --- the sampling distributions -------------------------------------------------------------------------------------------


def test_two_proportions_from_independent_trials_get_a_difference_an_interval_and_a_p_value() -> None:
    got = stats.two_proportion_test(60, 300, 90, 300)
    assert got is not None and math.isclose(got["diff"], -0.1) and got["ci_lower"] < -0.1 < got["ci_upper"]
    assert 0.003 < got["p"] < 0.007 and got["method"] == "two_proportion_z_newcombe"
    same = stats.two_proportion_test(50, 200, 50, 200)
    assert same["diff"] == 0 and same["p"] == pytest.approx(1.0)
    edge = stats.two_proportion_test(0, 100, 0, 100)
    assert edge["p"] == 1.0 and edge["ci_lower"] <= 0 <= edge["ci_upper"]
    assert stats.two_proportion_test(5, 3, 1, 10) is None and stats.two_proportion_test(1, 0, 1, 10) is None


def test_holm_adjustment_is_monotone_capped_and_at_least_the_raw_value() -> None:
    raw = [0.01, 0.04, 0.03, 0.5]
    adjusted = stats.holm_adjust(raw)
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert all(a >= r for a, r in zip(adjusted, raw)) and stats.holm_adjust([0.4, 0.9]) == pytest.approx([0.8, 0.9])
    assert stats.holm_adjust([]) == [] and stats.holm_adjust([0.9])[0] == 0.9 and stats.holm_adjust([0.0001] * 3)[0] == pytest.approx(0.0003)


def test_a_paired_test_uses_the_pair_differences_and_is_exact_for_few_pairs() -> None:
    got = stats.paired_permutation_test([1, 1, 1, 1, 1, 1, 1, -1, 0, 0])
    assert got["p"] == pytest.approx(18 / 256) and got["n_pairs"] == 10 and got["diff"] == pytest.approx(0.6)
    assert got["ci_lower"] <= got["diff"] <= got["ci_upper"]
    assert stats.paired_permutation_test([0, 0, 0])["p"] == 1.0 and stats.paired_permutation_test([1.0]) is None
    many = stats.paired_permutation_test([1.0] * 80 + [-1.0] * 20)
    assert many["p"] < 0.001 and many["method"] == "paired_permutation_bootstrap"
    assert stats.paired_permutation_test([1.0, 2.0, 3.0] * 10) == stats.paired_permutation_test([1.0, 2.0, 3.0] * 10), "seeded: same data, same answer"


def test_two_means_get_a_bootstrap_interval_and_a_permutation_p_value() -> None:
    rng = random.Random(0)
    a = [rng.gauss(0.0, 1.0) for _ in range(120)]
    b = [rng.gauss(0.8, 1.0) for _ in range(120)]
    got = stats.two_sample_mean_test(a, b)
    assert got["diff"] < -0.4 and got["ci_upper"] < 0 and got["p"] < 0.01
    null = stats.two_sample_mean_test(a, [rng.gauss(0.0, 1.0) for _ in range(120)])
    assert null["p"] > 0.05 and null["ci_lower"] < 0 < null["ci_upper"]
    assert stats.two_sample_mean_test([1.0], [2.0, 3.0]) is None


def test_clusters_are_resampled_whole_so_correlated_observations_do_not_count_as_independent() -> None:
    rng = random.Random(1)
    clusters = [i // 20 for i in range(400)]  # 20 clusters of 20
    effect = {c: rng.random() < 0.5 for c in set(clusters)}
    values = [1.0 if effect[c] else 0.0 for c in clusters]  # everything inside a cluster agrees
    naive = stats.wilson_interval(sum(values), len(values))
    grouped = stats.cluster_bootstrap_interval(values, clusters)
    assert grouped["n_clusters"] == 20 and grouped["n_observations"] == 400
    assert (grouped["ci_upper"] - grouped["ci_lower"]) > 2 * (naive[1] - naive[0]), "clusters are wider than 400 independent trials"
    assert stats.cluster_bootstrap_interval([1.0, 0.0], [1, 1]) is None and stats.cluster_bootstrap_interval([1.0], [1, 2]) is None
    diff = stats.cluster_bootstrap_difference(values, clusters, [0.0] * 400, clusters)
    assert diff["diff"] == pytest.approx(sum(values) / 400) and diff["ci_lower"] <= diff["diff"] <= diff["ci_upper"]


# --- the spec's shape ---------------------------------------------------------------------------------------------------------


_ESTIMAND_UNIT = {"estimand": "P(outbreak | R0, N)", "unit": "trajectory"}


def test_a_metric_spec_is_checked_for_shape() -> None:
    good, why = ms.normalize([{"id": "p", "kind": "Proportion", "cluster": True, "paired": False, "family": "R0 contrasts", **_ESTIMAND_UNIT}])
    assert why is None and good == [{"id": "p", "kind": "proportion", "cluster": True, "paired": False, "family": "R0 contrasts", **_ESTIMAND_UNIT}]
    assert ms.normalize({"id": "p", "kind": "mean", **_ESTIMAND_UNIT})[0] == [{"id": "p", "kind": "mean", **_ESTIMAND_UNIT}]
    for bad, expect in [
        ("nope", "list of metric specs"), ([{"kind": "mean"}], "no `id`"), ([{"id": "p"}], "`kind` must be"),
        ([{"id": "p", "kind": "median"}], "`kind` must be"),
        ([{"id": "p", "kind": "mean", **_ESTIMAND_UNIT}, {"id": "p", "kind": "mean", **_ESTIMAND_UNIT}], "twice"),
        ([{"id": "p", "kind": "mean", **_ESTIMAND_UNIT, "paired": "yes"}], "`paired` must be"),
        ([{"id": "p", "kind": "mean", **_ESTIMAND_UNIT, "cluster": 3}], "`cluster` must be"),
        ([{"id": "p", "kind": "mean", "unit": "trajectory"}], "needs `estimand`"),
        ([{"id": "p", "kind": "mean", "estimand": "P(x)"}], "needs `unit`"),
        ([{"id": "p", "kind": "mean", "estimand": 5, "unit": "trajectory"}], "needs `estimand`"),
        ([{"id": "p", "kind": "mean", "estimand": "  ", "unit": "trajectory"}], "needs `estimand`"),
        ([{"id": "p", "kind": "mean", "estimand": "P(x)", "unit": 5}], "needs `unit`"),
        ([{"id": "p", "kind": "mean", "estimand": "P(x)", "unit": "  "}], "needs `unit`"),
    ]:
        assert expect in (ms.normalize(bad)[1] or ""), bad


def test_the_protocol_keeps_and_refuses_metrics_like_its_other_keys() -> None:
    design, error = plan.normalize_design({"hypothesis": "h", "protocol": {"metrics": [{"id": "p", "kind": "proportion", **_ESTIMAND_UNIT}]}})
    assert error is None and design["protocol"]["metrics"] == [{"id": "p", "kind": "proportion", **_ESTIMAND_UNIT}]
    assert plan.parse(plan.render("t", {}, design)).design["protocol"]["metrics"] == design["protocol"]["metrics"]
    refused, why = plan.normalize_design({"hypothesis": "h", "protocol": {"metrics": [{"id": "p", "kind": "guess"}]}})
    assert refused is None and "`kind` must be" in why
    no_estimand, why2 = plan.normalize_design({"hypothesis": "h", "protocol": {"metrics": [{"id": "p", "kind": "mean"}]}})
    assert no_estimand is None and "needs `estimand`" in why2


def test_the_plan_says_when_a_protocol_with_runs_declares_no_metric() -> None:
    from core import protocol_check as pc

    assert "declares no metric spec" in pc.metric_notes({"runs_per_setting": 300})[0]
    assert pc.metric_notes({"runs_per_setting": 300, "metrics": [{"id": "p", "kind": "mean", **_ESTIMAND_UNIT}]}) == []
    assert pc.metric_notes({"grid": {"a": [1]}}) == [] and pc.metric_notes(None) == []


# --- the statistics ------------------------------------------------------------------------------------------------------------


def _seeds(counts: dict[str, tuple[int, int]], n_seeds: int = 3, total: int = 300) -> list[dict[str, Any]]:
    return [
        {"_seed": s, "by_R0": {k: {"outbreak_probability": (a + s) / total, "outbreak_probability_count": a + s, "outbreak_probability_total": total}
                                for k, (a, _b) in counts.items()}}
        for s in range(n_seeds)
    ]


PROTO = {"metrics": [{"id": "outbreak_probability", "kind": "proportion", "family": "R0", **_ESTIMAND_UNIT}]}


def test_a_proportion_from_counts_gets_wilson_and_a_two_proportion_contrast_with_holm() -> None:
    reps = _seeds({"1.5": (60, 0), "3.0": (120, 0), "6.0": (121, 0)})
    out = ms.statistics(reps, PROTO)
    est = out["estimates"]["by_R0.1.5.outbreak_probability"]
    assert est["estimator"] == "wilson_pooled_counts" and est["n_trials"] == 900 and est["ci_lower"] < est["estimate"] < est["ci_upper"]
    assert len(out["contrasts"]) == 3 and out["unsupported"] == [] and out["undeclared"] == []
    by_pair = {(c["a"], c["b"]): c for c in out["contrasts"]}
    strong, weak = by_pair[("1.5", "3.0")], by_pair[("3.0", "6.0")]
    assert strong["method"] == "two_proportion_z_newcombe" and strong["significant_after_holm"] is True and strong["comparisons_in_family"] == 3
    assert weak["p"] > 0.5 and weak["significant_after_holm"] is False
    assert all(c["p_holm"] >= c["p"] for c in out["contrasts"])
    assert all(c["prespecified"] is False for c in out["contrasts"]), "no protocol.contrasts given: every pair, unmarked"


def test_a_prespecified_contrast_list_computes_only_those_pairs() -> None:
    reps = _seeds({"1.5": (60, 0), "3.0": (120, 0), "6.0": (121, 0)})
    proto = {**PROTO, "contrasts": [{"metric": "outbreak_probability", "a": "1.5", "b": "3.0"}]}
    out = ms.statistics(reps, proto)
    assert len(out["contrasts"]) == 1
    (c,) = out["contrasts"]
    assert {c["a"], c["b"]} == {"1.5", "3.0"} and c["prespecified"] is True
    assert c["comparisons_in_family"] == 1, "the un-asked-for pairs are not silently in the same family"
    assert out["contrasts_prespecified"] is True
    # The un-asked-for pairs are skipped outright -- not reported as unsupported (nothing is wrong with them).
    assert out["unsupported"] == []


def test_a_prespecified_list_for_a_different_metric_leaves_this_ones_pairwise_untouched() -> None:
    reps = _seeds({"1.5": (60, 0), "3.0": (120, 0), "6.0": (121, 0)})
    proto = {**PROTO, "metrics": [*PROTO["metrics"], {"id": "some_other_metric", "kind": "mean", **_ESTIMAND_UNIT}],
             "contrasts": [{"metric": "some_other_metric", "a": "1.5", "b": "3.0"}]}
    out = ms.statistics(reps, proto)
    assert len(out["contrasts"]) == 3 and out["contrasts_prespecified"] is False


def test_a_prespecified_pair_naming_no_real_setting_is_named_not_silently_zero() -> None:
    """A typo in `a`/`b` (e.g. "1.05" for "1.5") would otherwise silently filter out every real pair
    and produce zero contrasts for the metric, with nothing saying why."""
    reps = _seeds({"1.5": (60, 0), "3.0": (120, 0)})
    proto = {**PROTO, "contrasts": [{"metric": "outbreak_probability", "a": "1.05", "b": "3.0"}]}
    out = ms.statistics(reps, proto)
    assert out["contrasts"] == []
    assert any("no setting in the results matches both sides" in u["reason"] for u in out["unsupported"])


def test_a_prespecified_contrast_naming_an_unknown_metric_id_is_named() -> None:
    """A typo in `metric` (e.g. "outbrek_probability") would otherwise be silently ignored, and every
    real metric would fall back to all-pairwise indistinguishably from a protocol that simply chose
    not to prespecify anything."""
    reps = _seeds({"1.5": (60, 0), "3.0": (120, 0)})
    proto = {**PROTO, "contrasts": [{"metric": "outbrek_probability", "a": "1.5", "b": "3.0"}]}
    out = ms.statistics(reps, proto)
    assert any("matches none of the protocol's declared metric specs" in u["reason"] for u in out["unsupported"])


def test_the_p_values_of_a_family_are_corrected_together_and_not_with_other_families() -> None:
    reps = []
    for s in range(2):
        node: dict[str, Any] = {}
        for k, (count, count2) in {"a": (60, 80), "b": (90, 81)}.items():
            node[k] = {"x": count / 300, "x_count": count + s, "x_total": 300, "y": count2 / 300, "y_count": count2 + s, "y_total": 300}
        reps.append({"_seed": s, "by_g": node})
    proto = {"metrics": [{"id": "x", "kind": "proportion", "family": "F1", **_ESTIMAND_UNIT}, {"id": "y", "kind": "proportion", "family": "F2", **_ESTIMAND_UNIT}]}
    out = ms.statistics(reps, proto)
    assert {c["metric"]: c["comparisons_in_family"] for c in out["contrasts"]} == {"x": 1, "y": 1}
    single = {"metrics": [{"id": "x", "kind": "proportion", "family": "same", **_ESTIMAND_UNIT}, {"id": "y", "kind": "proportion", "family": "same", **_ESTIMAND_UNIT}]}
    together = ms.statistics(reps, single)
    assert {c["comparisons_in_family"] for c in together["contrasts"]} == {2}
    assert together["contrasts"][0]["p_holm"] >= out["contrasts"][0]["p_holm"]


def _values_reps(a: list[float], b: list[float], *, clusters: bool = False, seeds: int = 2) -> list[dict[str, Any]]:
    out = []
    for s in range(seeds):
        node = {k: {"m_values": list(v)} for k, v in (("a", a), ("b", b))}
        if clusters:
            for k in node:
                node[k]["m_clusters"] = [i // 5 for i in range(len(a))]
        out.append({"_seed": s, "by_g": node})
    return out


def test_a_paired_mean_is_compared_pair_by_pair_and_an_unpaired_one_is_not() -> None:
    rng = random.Random(4)
    base = [rng.gauss(0, 1) for _ in range(60)]
    shifted = [x + 0.3 + rng.gauss(0, 0.05) for x in base]  # the same random numbers plus a small consistent shift
    reps = _values_reps(shifted, base)
    paired = ms.statistics(reps, {"metrics": [{"id": "m", "kind": "mean", "paired": True, **_ESTIMAND_UNIT}]})
    unpaired = ms.statistics(reps, {"metrics": [{"id": "m", "kind": "mean", "paired": False, **_ESTIMAND_UNIT}]})
    (cp,), (cu,) = paired["contrasts"], unpaired["contrasts"]
    assert cp["method"] == "paired_permutation_bootstrap" and cu["method"] == "bootstrap_permutation"
    assert cp["p"] < 0.001 and cu["p"] > cp["p"], "pairing removes the shared noise: the same shift is far clearer"
    assert cp["diff"] == pytest.approx(0.3, abs=0.05)
    assert cp["paired_id_verified"] is False, "no `m_pair_id` was given, so this falls back to position order"


def test_a_paired_design_with_pair_ids_joins_by_id_even_when_the_order_differs() -> None:
    """The bug a shuffled `_values` order would cause without a pair_id join: trial i of one setting must be
    matched to the SAME trial of the other, not to whatever sits at position i."""
    base = [1.0, 2.0, 3.0, 4.0, 5.0]
    shifted = [x + 0.3 for x in base]
    # setting "a" keeps run order; setting "b" reports the same trials shuffled -- a real script might, e.g., if
    # it iterates a dict keyed by trial id rather than a list.
    reps = [{
        "_seed": s,
        "by_g": {
            "a": {"m_values": list(shifted), "m_pair_id": [0, 1, 2, 3, 4]},
            "b": {"m_values": [base[4], base[0], base[1], base[2], base[3]], "m_pair_id": [4, 0, 1, 2, 3]},
        },
    } for s in range(2)]
    out = ms.statistics(reps, {"metrics": [{"id": "m", "kind": "mean", "paired": True, **_ESTIMAND_UNIT}]})
    (c,) = out["contrasts"]
    assert c["paired_id_verified"] is True
    # `diff` (mean(a) - mean(b)) is invariant to how the two settings are matched up -- sum(a) - sum(b) does not
    # care about pairing order, so it is NOT a real check of the join. What DOES depend on correct pairing is the
    # per-pair variance: joined by id, every pair's difference is exactly 0.3 (zero variance -> a near-point
    # interval); joined by the shuffled list position instead, the differences would range from -3.7 to 1.3 (wide
    # variance) even though their sum is unchanged. A wide interval here would mean the join fell back to
    # position despite paired_id_verified saying otherwise.
    assert c["diff"] == pytest.approx(0.3, abs=1e-9)
    assert c["ci_upper"] - c["ci_lower"] < 0.01, (c["ci_lower"], c["ci_upper"])


def test_a_paired_design_whose_pair_ids_disagree_between_settings_is_refused() -> None:
    reps = [{
        "_seed": s,
        "by_g": {
            "a": {"m_values": [1.0, 2.0, 3.0], "m_pair_id": [0, 1, 2]},
            "b": {"m_values": [1.0, 2.0, 3.0], "m_pair_id": [0, 1, 99]},  # 99 names a trial "a" never reported
        },
    } for s in range(2)]
    out = ms.statistics(reps, {"metrics": [{"id": "m", "kind": "mean", "paired": True, **_ESTIMAND_UNIT}]})
    assert out["contrasts"] == []
    assert any("names different trials" in u["reason"] for u in out["unsupported"])


def test_a_pair_id_repeated_within_one_settings_own_seed_is_refused_not_silently_dropped() -> None:
    """`dict(zip(pair_id, values))` would silently keep only the LAST value for a repeated id,
    dropping a real observation with no error -- this must be refused instead."""
    reps = [{
        "_seed": s,
        "by_g": {
            "a": {"m_values": [10.0, 20.0, 30.0], "m_pair_id": [0, 0, 1]},  # id 0 used twice
            "b": {"m_values": [1.0, 2.0, 3.0], "m_pair_id": [0, 1, 2]},
        },
    } for s in range(2)]
    out = ms.statistics(reps, {"metrics": [{"id": "m", "kind": "mean", "paired": True, **_ESTIMAND_UNIT}]})
    assert out["contrasts"] == []
    assert any("malformed" in u["reason"] for u in out["unsupported"])


def test_a_pair_id_given_by_only_one_setting_is_refused_not_silently_positional() -> None:
    reps = [{
        "_seed": s,
        "by_g": {
            "a": {"m_values": [1.0, 2.0, 3.0], "m_pair_id": [0, 1, 2]},
            "b": {"m_values": [1.0, 2.0, 3.0]},  # no pair_id at all
        },
    } for s in range(2)]
    out = ms.statistics(reps, {"metrics": [{"id": "m", "kind": "mean", "paired": True, **_ESTIMAND_UNIT}]})
    assert out["contrasts"] == []
    assert any("only one setting" in u["reason"] for u in out["unsupported"])


def test_a_cluster_design_uses_the_clusters_and_says_what_it_lacks() -> None:
    rng = random.Random(5)
    a = [float(rng.random() < 0.5) for _ in range(100)]
    reps = _values_reps(a, [0.0] * 100, clusters=True)
    spec = {"metrics": [{"id": "m", "kind": "proportion", "cluster": True, **_ESTIMAND_UNIT}]}
    out = ms.statistics(reps, spec)
    assert out["estimates"]["by_g.a.m"]["estimator"] == "cluster_bootstrap" and out["estimates"]["by_g.a.m"]["n_clusters"] == 40
    assert out["contrasts"][0]["method"] == "cluster_bootstrap"
    no_clusters = ms.statistics(_values_reps(a, a), spec)
    assert no_clusters["estimates"] == {} and "cluster design needs" in no_clusters["unsupported"][0]["reason"]
    both = ms.statistics(reps, {"metrics": [{"id": "m", "kind": "proportion", "cluster": True, "paired": True, **_ESTIMAND_UNIT}]})
    assert any("paired design over clusters is not supported" in u["reason"] for u in both["unsupported"])
    unequal = _values_reps([1.0, 0.0, 1.0], [1.0, 0.0])
    uneq = ms.statistics(unequal, {"metrics": [{"id": "m", "kind": "mean", "paired": True, **_ESTIMAND_UNIT}]})
    assert "same number of trials" in uneq["unsupported"][0]["reason"]


def test_a_metric_that_is_not_in_the_results_and_one_that_no_spec_covers_are_said() -> None:
    reps = _seeds({"1.5": (60, 0), "3.0": (120, 0)})
    missing = ms.statistics(reps, {"metrics": [{"id": "final_size", "kind": "mean", **_ESTIMAND_UNIT}]})
    assert missing["unsupported"][0]["reason"] == "`final_size` does not appear in the results"
    assert ms.undeclared(reps, []) == ["outbreak_probability"] and ms.undeclared(reps, ms.declared(PROTO)) == []
    only_values = [{"_seed": 0, "cost_values": [1.0, 2.0]}]
    assert ms.undeclared(only_values, []) == ["cost"]
    assert ms.statistics(reps, None) == {} and ms.statistics([], PROTO) == {}


def test_what_keeps_the_statistics_from_being_adequate_is_named() -> None:
    reps = _seeds({"1.5": (60, 0), "3.0": (120, 0)})
    assert "guessed from the names" in ms.coverage_gaps({}, reps, None)[0]
    assert ms.coverage_gaps({}, [{"_seed": 0, "score": 1.0}], None) == [], "a result with nothing whose estimator matters has nothing to declare"
    with_contrasts = {**PROTO, "contrasts": [{"metric": "outbreak_probability", "a": "1.5", "b": "3.0"}]}
    covered = ms.statistics(reps, with_contrasts)
    assert ms.coverage_gaps(with_contrasts, reps, covered) == []
    # PROTO itself names no prespecified contrasts, so its one pairwise comparison is a gap of its own.
    unspecified = ms.coverage_gaps(PROTO, reps, ms.statistics(reps, PROTO))
    assert any("no prespecified contrasts" in g for g in unspecified)
    partial = {"metrics": [{"id": "other", "kind": "mean", **_ESTIMAND_UNIT}]}
    got = ms.coverage_gaps(partial, reps, ms.statistics(reps, partial))
    assert any("no metric spec covers outbreak_probability" in g for g in got) and any("could not be made" in g for g in got)


# --- through the real graph ---------------------------------------------------------------------------------------------------

_SCRIPT = """\
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
seed = int(os.environ.get("FI_REPLICATE_INDEX", "0"))
NUM_RUNS = 300
res = {"by_R0": {"1.5": {"outbreak_probability": (60 + seed) / 300, "outbreak_probability_count": 60 + seed, "outbreak_probability_total": 300},
                  "3.0": {"outbreak_probability": (120 + seed) / 300, "outbreak_probability_count": 120 + seed, "outbreak_probability_total": 300}}}
os.makedirs('figures', exist_ok=True)
plt.figure(); plt.plot([0, 1, 2], [0, 1, 4]); plt.savefig('figures/result.png', dpi=72)
print('RESULT_JSON: ' + json.dumps(res))
"""


def _cfg(tmp_path: Path) -> Config:
    return Config(
        topic="smoke topic for metric specs", title="metric-smoke", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=3, pilot_run=False,
                            oracle_check="off", run_manifest_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, split_analysis=False),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(review="off"),
    )


def _fake(prompts: list[str], protocol: dict[str, Any]):
    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        prompts.append(prompt)
        kind = _classify(prompt)
        if kind == "Experiment Design":
            body = json.loads(_FAKE_RESPONSES["design"])
            body["protocol"] = protocol
            return json.dumps(body)
        if kind == "Implementation":
            return json.dumps({"code": _SCRIPT, "deps": ["matplotlib"]})
        return _fake_response_for(prompt)

    return fake_chat


@pytest.mark.asyncio
async def test_the_analysis_is_given_the_engines_estimates_and_contrasts_and_the_evidence_names_what_is_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    declared_prompts: list[str] = []
    protocol = {"runs_per_setting": 300, "metrics": [{"id": "outbreak_probability", "kind": "proportion", "family": "R0 contrasts", **_ESTIMAND_UNIT}]}
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(declared_prompts, protocol))
    engine = Engine(_cfg(tmp_path / "declared"))
    artifacts = await engine.run()
    assert artifacts.paper_md is not None
    analysis = [p for p in declared_prompts if _classify(p) == "Analysis"]
    assert analysis and '"spec_statistics"' in analysis[0] and '"two_proportion_z_newcombe"' in analysis[0] and '"p_holm"' in analysis[0]
    evidence = json.loads((engine.quest_root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8"))
    ready = evidence["all_gaps"].get("statistically_adequate", [])
    # This fixture's protocol declares a metric spec but no protocol.contrasts, so the one new,
    # separate gap that names is expected (covered on its own in test_metric_spec.py); nothing else
    # (a guessed estimator, an unsupported contrast) should be there alongside it.
    assert not any("guessed" in g or "could not be made" in g for g in ready), ready
    assert any("no prespecified contrasts" in g for g in ready), ready

    guessed_prompts: list[str] = []
    monkeypatch.setattr("core.engine.LLMClient.chat", _fake(guessed_prompts, {"runs_per_setting": 300}))
    other = Engine(_cfg(tmp_path / "guessed"))
    await other.run()
    assert not any('"spec_statistics"' in p for p in guessed_prompts if _classify(p) == "Analysis")
    ready = json.loads((other.quest_root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8"))["all_gaps"]["statistically_adequate"]
    assert any("statistics are not shown to be adequate" in g and "guessed from the names" in g for g in ready)
    assert "declares no metric spec" in (other.quest_root / "plan.md").read_text(encoding="utf-8")


def test_the_prompts_ask_for_the_data_the_estimators_need_and_tell_the_analysis_to_quote_the_engine() -> None:
    agents = Path(__file__).resolve().parent.parent / "agents"
    for name in ("implement.md", "implement_body.md"):
        text = (agents / name).read_text(encoding="utf-8")
        assert "`<name>_clusters`" in text and "the same random numbers" in text, name
    analyze = (agents / "analyze.md").read_text(encoding="utf-8")
    assert "spec_statistics" in analyze and "Holm-adjusted" in analyze and "never work out significance yourself" in analyze


def test_one_metric_of_an_unknown_kind_is_left_out_not_the_list_it_is_in() -> None:
    """A live plan declared a failure count as `kind: count` beside valid metrics; the strict check threw the whole list
    away, and with it the spec that would have told the pooling a mean from a probability."""
    good = {"id": "prob_major", "kind": "proportion", "estimand": "P(major | R0, N)", "unit": "run"}
    bad = {"id": "failed_run_count", "kind": "count", "estimand": "failed runs", "unit": "run"}
    kept, notes = ms.repair([good, bad])
    assert [m["id"] for m in kept] == ["prob_major"] and len(notes) == 1 and "failed_run_count" in notes[0]
    fixed, notes = plan.repair_protocol({"runs_per_setting": 300, "metrics": [good, bad]})
    assert [m["id"] for m in fixed["metrics"]] == ["prob_major"] and fixed["runs_per_setting"] == 300
    assert len(notes) == 1 and "failed_run_count" in notes[0] and "left out of the plan" in notes[0]
    # A person's own edit is still held to the strict check.
    assert plan.normalize_protocol({"metrics": [good, bad]})[0] is None
