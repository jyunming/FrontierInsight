"""Tests for core/probes.py: the judges on synthetic data, parse strictness, saturation, mutators. No model, no executor."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from core import probes as P

GRID = {"dt": [0.4, 0.2, 0.1, 0.05], "n": [10, 100, 1000], "mode": ["a", "b"], "x": [1.0, 2.0, 4.0]}
PROTOCOL = {"grid": GRID, "metrics": [{"id": "err", "kind": "mean"}, {"id": "rate", "kind": "proportion"}]}


def rel(kind, **kw):
    base = dict(id="R1", kind=kind, metric="err", setting="dt", quote="q")
    base.update(kw)
    return P.Relation(**base)


def obs(setting, value, vals):
    return P.Observation({setting: value}, vals)


def noisy(rng, n, mean, sd):
    return [mean + rng.gauss(0, sd) for _ in range(n)]


def paired(rng, n, means, sd_common, sd_own=0.0):
    """Paired seeds: a shared per-seed shock plus a small own noise."""
    shock = [rng.gauss(0, sd_common) for _ in range(n)]
    return [[m + s + rng.gauss(0, sd_own) for s in shock] for m in means]


# ---------------------------------------------------------------------------------------------------------------- parse


def reply(*items):
    return json.dumps({"relations": list(items)})


def good(**kw):
    d = {"id": "R1", "kind": "sensitivity", "setting": "dt", "metric": "err", "strength": "core", "from_plan": "the step matters"}
    d.update(kw)
    return d


def test_parse_keeps_valid_and_reads_fenced_json():
    rels, dropped = P.parse("```json\n" + reply(good()) + "\n```", PROTOCOL)
    assert [r.id for r in rels] == ["R1"] and not dropped
    assert rels[0].strength == "core" and rels[0].quote == "the step matters"


def test_parse_drops_unknown_setting_metric_kind_and_missing_quote():
    rels, dropped = P.parse(reply(good(id="a", setting="nope"), good(id="b", metric="zzz"), good(id="c", kind="magic"),
                                  good(id="d", from_plan=""), good(id="e")), PROTOCOL)
    assert [r.id for r in rels] == ["e"]
    assert P.drop_counts(dropped) == {"setting not in the protocol's grid": 1, "metric not in the protocol": 1,
                                      "unknown kind": 1, "no quote from the plan": 1}


def test_parse_cuts_at_eight_keeping_core_first():
    items = [good(id=f"s{i}", strength="supporting") for i in range(7)] + [good(id=f"c{i}", strength="core") for i in range(3)]
    rels, dropped = P.parse(reply(*items), PROTOCOL)
    assert len(rels) == 8
    assert {"c0", "c1", "c2"} <= {r.id for r in rels}
    assert len(dropped) == 2 and all("limit of 8" in d["reason"] for d in dropped)


def test_parse_ignores_expected_values():
    rels, _ = P.parse(reply(good(expected=3.14, expected_value=2, value=7, result=1)), PROTOCOL)
    assert len(rels) == 1
    d = rels[0].to_dict()
    assert not any(k in d for k in ("expected", "expected_value", "value", "result"))


def test_parse_kind_specific_fields():
    rels, dropped = P.parse(reply(
        good(id="m1", kind="monotone", direction="rises"),
        good(id="m2", kind="monotone"),
        good(id="s1", kind="scaling", setting="x", factor=2, exponent=2),
        good(id="s2", kind="scaling", setting="x", factor=1, exponent=2),
        good(id="c1", kind="convergence", order=2, refine="decrease"),
        good(id="c2", kind="convergence", order=-1),
        good(id="e1", kind="equal_when", a={"x": 1.0}, b={"x": 2.0}),
        good(id="e2", kind="equal_when", a={"x": 1.0}, b={"x": 99.0}),
        good(id="e3", kind="equal_when", a={"mode": "a"}, b={"mode": "zzz"}),
        good(id="e4", kind="equal_when", a={"zz": 1}, b={"x": 2.0})), PROTOCOL)
    assert {r.id for r in rels} == {"m1", "s1", "c1", "e1"}
    assert rels[0].direction == "up"
    assert len(dropped) == 6


def test_parse_garbage_and_non_list():
    assert P.parse("not json at all", PROTOCOL) == ([], [])
    assert P.parse(json.dumps({"relations": "x"}), PROTOCOL) == ([], [])
    rels, dropped = P.parse(json.dumps([good(), 5]), PROTOCOL)
    assert len(rels) == 1 and dropped[0]["reason"] == "not an object"


def test_parse_duplicate_ids_are_renamed():
    rels, _ = P.parse(reply(good(), good()), PROTOCOL)
    assert len({r.id for r in rels}) == 2


# ------------------------------------------------------------------------------------------------------------ plan_runs


def test_plan_runs_uses_grid_values_and_shares_cells():
    rels = [rel("sensitivity", id="A"), rel("invariance", id="B", metric="rate")]
    plan = P.plan_runs(rels, GRID, seeds=8)
    assert len(plan.runs) == 2   # both relations use dt=0.05 and dt=0.4: shared
    assert all(r.seeds == 8 for r in plan.runs)
    assert sorted(r.cell["dt"] for r in plan.runs) == [0.05, 0.4]
    assert plan.uses["A"] == plan.uses["B"]


def test_plan_runs_seed_clamp_and_deterministic():
    assert P.plan_runs([rel("sensitivity")], GRID, seeds=99).runs[0].seeds == 20
    assert P.plan_runs([rel("sensitivity")], GRID, seeds=0).runs[0].seeds == 1
    assert P.plan_runs([rel("sensitivity")], GRID, seeds=8, deterministic=True).runs[0].seeds == 1


def test_plan_runs_monotone_and_convergence_levels():
    plan = P.plan_runs([rel("monotone", direction="down"), rel("convergence", id="C", order=1, refine="decrease")], GRID)
    assert len(plan.uses["R1"]) == 4 and len(plan.uses["C"]) == 4
    # an axis with two values is interpolated inside [min, max]
    plan = P.plan_runs([rel("monotone", direction="down")], {"dt": [0.1, 0.4]})
    vals = sorted(r.cell["dt"] for r in plan.runs)
    assert len(vals) == 3 and vals[0] == 0.1 and vals[-1] == 0.4 and 0.1 < vals[1] < 0.4
    # a one-valued axis cannot be tested
    plan = P.plan_runs([rel("sensitivity")], {"dt": [0.1]})
    assert plan.unrunnable and not plan.runs


def test_plan_runs_scaling_pair_from_grid_else_in_range():
    plan = P.plan_runs([rel("scaling", setting="x", factor=2.0, exponent=2.0)], GRID)
    assert sorted(r.cell["x"] for r in plan.runs) in ([1.0, 2.0], [2.0, 4.0])
    plan = P.plan_runs([rel("scaling", setting="x", factor=3.0, exponent=1.0)], GRID)
    assert sorted(r.cell["x"] for r in plan.runs) == [1.0, 3.0]
    plan = P.plan_runs([rel("scaling", setting="x", factor=100.0, exponent=1.0)], GRID)
    assert plan.unrunnable and not plan.runs


def test_plan_runs_equal_when_uses_its_cells():
    plan = P.plan_runs([rel("equal_when", setting=None, a={"x": 1.0}, b={"x": 4.0})], GRID)
    assert sorted(r.cell["x"] for r in plan.runs) == [1.0, 4.0]


# -------------------------------------------------------------------------------------------------------------- judges


def test_sensitivity_noisy_linear_held_and_constant_broken():
    rng = random.Random(1)
    a, b = paired(rng, 8, [1.0, 3.0], 0.2, 0.05)
    v = P.judge(rel("sensitivity"), [obs("dt", 0.05, a), obs("dt", 0.4, b)])
    assert v.verdict == "held" and v.numbers["mean_difference"] > 1.5
    same = [1.234] * 8
    v = P.judge(rel("sensitivity"), [obs("dt", 0.05, same), obs("dt", 0.4, same)])
    assert v.verdict == "broken" and "exactly the same" in v.says and "1.234" in v.says


def test_sensitivity_within_noise_is_unconfirmed_not_broken():
    rng = random.Random(2)
    a, b = paired(rng, 8, [1.0, 1.01], 0.0, 1.0)
    v = P.judge(rel("sensitivity"), [obs("dt", 0.05, a), obs("dt", 0.4, b)])
    assert v.verdict == "unconfirmed"


def test_sensitivity_deterministic_single_run():
    assert P.judge(rel("sensitivity"), [obs("dt", 1, [2.0]), obs("dt", 2, [2.5])]).verdict == "held"
    assert P.judge(rel("sensitivity"), [obs("dt", 1, [2.0]), obs("dt", 2, [2.0])]).verdict == "broken"


def test_invariance_stochastic_with_known_se():
    rng = random.Random(3)
    a, b = paired(rng, 20, [5.0, 5.0], 1.0, 0.1)
    assert P.judge(rel("invariance"), [obs("dt", 0.05, a), obs("dt", 0.4, b)]).verdict == "held"
    c, d = paired(rng, 20, [5.0, 6.0], 1.0, 0.1)
    v = P.judge(rel("invariance"), [obs("dt", 0.05, c), obs("dt", 0.4, d)])
    assert v.verdict == "broken" and v.numbers["mean_difference"] > 0.8


def test_known_se_threshold_is_four():
    n = 16
    d = [(+1 if i % 2 else -1) * 1.0 for i in range(n)]
    a = [10.0] * n
    se = P._se(d)
    for mult, expect in ((3.0, "held"), (5.0, "broken")):
        b = [10.0 + x + mult * se for x in d]
        assert P.judge(rel("invariance"), [obs("dt", 1, a), obs("dt", 2, b)]).verdict == expect


def test_equal_when_matches_cells():
    r = rel("equal_when", setting=None, a={"x": 1.0}, b={"x": 4.0})
    o = [P.Observation({"x": 1.0, "dt": 0.1}, [1.0, 2.0, 3.0]), P.Observation({"x": 4.0, "dt": 0.1}, [1.0, 2.0, 3.0])]
    assert P.judge(r, o).verdict == "held"
    o[1] = P.Observation({"x": 4.0, "dt": 0.1}, [2.0, 3.0, 4.0])
    assert P.judge(r, o).verdict == "broken"


def test_monotone_noisy_linear_and_wrong_direction():
    rng = random.Random(4)
    cols = paired(rng, 10, [1.0, 2.0, 3.0, 4.0], 0.3, 0.1)
    o = [obs("dt", x, c) for x, c in zip([0.05, 0.1, 0.2, 0.4], cols)]
    assert P.judge(rel("monotone", direction="up"), o).verdict == "held"
    down = P.judge(rel("monotone", direction="down"), o)
    assert down.verdict == "broken" and "other way" in down.says
    cols = paired(rng, 10, [1.0, 2.0, 1.2, 4.0], 0.1, 0.02)
    o = [obs("dt", x, c) for x, c in zip([0.05, 0.1, 0.2, 0.4], cols)]
    assert P.judge(rel("monotone", direction="up"), o).verdict == "broken"


def test_monotone_tolerates_a_dip_within_two_se():
    rng = random.Random(5)
    cols = paired(rng, 8, [1.0, 2.0, 1.99, 4.0], 0.2, 0.5)
    o = [obs("dt", x, c) for x, c in zip([0.05, 0.1, 0.2, 0.4], cols)]
    assert P.judge(rel("monotone", direction="up"), o).verdict == "held"


def test_monotone_flat_is_broken_and_two_levels_unconfirmed():
    o = [obs("dt", x, [3.0] * 5) for x in (0.1, 0.2, 0.4)]
    assert P.judge(rel("monotone", direction="up"), o).verdict == "broken"
    assert P.judge(rel("monotone", direction="up"), o[:2]).verdict == "unconfirmed"


def test_scaling_p2_deterministic_and_noisy():
    r = rel("scaling", setting="x", factor=2.0, exponent=2.0)
    ok = [P.Observation({"x": 1.0}, [3.0]), P.Observation({"x": 2.0}, [12.0])]
    v = P.judge(r, ok)
    assert v.verdict == "held" and abs(v.numbers["expected_ratio"] - 4.0) < 1e-12
    bad = [P.Observation({"x": 1.0}, [3.0]), P.Observation({"x": 2.0}, [6.0])]
    v = P.judge(r, bad)
    assert v.verdict == "broken" and "x4" in v.says
    rng = random.Random(6)
    a = noisy(rng, 12, 3.0, 0.05)
    b = [4.0 * x + rng.gauss(0, 0.05) for x in a]
    assert P.judge(r, [P.Observation({"x": 1.0}, a), P.Observation({"x": 2.0}, b)]).verdict == "held"


def conv_obs(rng, knobs, fn, n=1, sd=0.0):
    return [P.Observation({"dt": k}, [fn(k) + rng.gauss(0, sd) for _ in range(n)]) for k in knobs]


@pytest.mark.parametrize("order", [1, 2])
def test_convergence_order_held(order):
    rng = random.Random(7)
    ks = [0.4, 0.2, 0.1, 0.05]
    o = conv_obs(rng, ks, lambda k: 1.0 + 3.0 * k ** order)
    v = P.judge(rel("convergence", order=float(order), refine="decrease"), o)
    assert v.verdict == "held" and abs(v.numbers["observed_order"] - order) < 0.25


def test_convergence_too_slow_broken_and_half_order_ok():
    rng = random.Random(8)
    ks = [0.4, 0.2, 0.1, 0.05]
    slow = conv_obs(rng, ks, lambda k: 1.0 + 3.0 * k ** 0.5)
    assert P.judge(rel("convergence", order=2.0, refine="decrease"), slow).verdict == "broken"
    ok = conv_obs(rng, ks, lambda k: 1.0 + 3.0 * k ** 1.0)
    assert P.judge(rel("convergence", order=2.0, refine="decrease"), ok).verdict == "held"   # order 1 >= half of 2


def test_convergence_not_shrinking_and_increasing_knob():
    rng = random.Random(9)
    ks = [0.4, 0.2, 0.1, 0.05]
    grow = conv_obs(rng, ks, lambda k: 1.0 + 3.0 / k)   # error larger at finer levels
    assert P.judge(rel("convergence", order=1.0, refine="decrease"), grow).verdict == "broken"
    ns = [10, 20, 40, 80]
    o = [P.Observation({"dt": n}, [2.0 + 5.0 / n]) for n in ns]
    v = P.judge(rel("convergence", order=1.0, refine="increase"), o)
    assert v.verdict == "held" and abs(v.numbers["observed_order"] - 1.0) < 0.3


def test_convergence_spread_monte_carlo():
    rng = random.Random(10)
    ns = [25, 100, 400, 1600]
    o = [P.Observation({"dt": n}, [rng.gauss(0, 1.0 / n ** 0.5) for _ in range(20)]) for n in ns]
    v = P.judge(rel("convergence", order=0.5, refine="increase", of="spread"), o)
    assert v.verdict == "held" and 0.3 < v.numbers["observed_order"] < 0.7


def test_convergence_noise_dominated_is_not_held():
    rng = random.Random(11)
    o = [P.Observation({"dt": k}, [rng.gauss(1.0, 1.0) for _ in range(8)]) for k in (0.4, 0.2, 0.1, 0.05)]
    assert P.judge(rel("convergence", order=1.0, refine="decrease"), o).verdict in ("unconfirmed", "broken")


def test_missing_numbers_are_unconfirmed():
    o = [obs("dt", 0.05, [1.0, None]), obs("dt", 0.4, [2.0, 2.0])]
    assert P.judge(rel("sensitivity"), o).verdict == "unconfirmed"
    assert P.judge(rel("sensitivity"), [obs("dt", 0.05, [1.0, 2.0]), obs("dt", 0.4, [2.0])]).verdict == "unconfirmed"


def test_says_is_plain_and_has_numbers():
    v = P.judge(rel("sensitivity"), [obs("dt", 0.05, [1.5] * 4), obs("dt", 0.4, [1.5] * 4)])
    assert "`err`" in v.says and "1.5" in v.says and "expected" not in v.says.lower()


# ------------------------------------------------------------------------------------------------------------ saturation


def cell(**metrics):
    return {"metrics": {k: list(v) for k, v in metrics.items()}}


def test_saturation_nan_bound_constant():
    nan = P.saturation([cell(m=[1.0, float("nan")]), cell(m=[2.0, 2.0])])
    assert [v.relation_id for v in nan] == ["saturation:m"] and "not a number" in nan[0].says
    none = P.saturation([cell(m=[None, 1.0])])
    assert none and none[0].verdict == "broken"
    pinned = P.saturation([cell(p=[0.0] * 40, q=[1.0, 2.0]), cell(p=[0.0] * 40, q=[2.0, 3.0])])
    assert any(v.relation_id == "saturation:p" and v.verdict == "broken" and "exactly 0" in v.says for v in pinned)
    one = P.saturation([cell(p=[1.0] * 40, q=[1.0] * 40), cell(p=[1.0] * 40, q=[2.0] * 40)])
    assert any(v.relation_id == "saturation:p" and "exactly 1" in v.says for v in one)
    # one number per cell is a result, not a 0/1 draw: a value pinned at 0 is read from any number of cells
    assert P.saturation([cell(p=[0.0], q=[1.0]), cell(p=[0.0], q=[2.0]), cell(p=[0.0], q=[3.0])])[0].relation_id == "saturation:p"
    # constant while others move: unconfirmed (a fixed control looks the same)
    const = P.saturation([cell(dof=[0.7, 0.7], y=[i, i]) for i in range(1, 5)])
    assert [v.relation_id for v in const] == ["saturation:dof"] and "although other results change" in const[0].says
    assert const[0].verdict == "unconfirmed"


def test_saturation_few_draws_of_an_indicator_at_its_bound_prove_nothing():
    # a 95% coverage can come out all ones in a handful of draws; INDICATOR_MIN_VALUES draws are needed
    few = [cell(cov=[1.0] * 4, y=[float(i)] * 4) for i in range(4)]
    assert P.saturation(few) == []
    many = [cell(cov=[1.0] * (P.INDICATOR_MIN_VALUES // 3 + 1), y=[float(i)]) for i in range(4)]
    assert [v.relation_id for v in P.saturation(many)] == ["saturation:cov"]


def test_saturation_clean_data_and_short_sweeps_pass():
    assert P.saturation([cell(m=[1.0, 2.0], k=[5.0, 5.0]), cell(m=[2.0, 3.0], k=[5.0, 5.0])]) == []   # 2 cells: constant not read
    assert P.saturation([cell(m=[1.0, 2.0]), cell(m=[3.0, 4.0]), cell(m=[5.0, 6.0])]) == []


def test_saturation_exemptions_mirror_expected_constants():
    cells = [cell(viol=[0.5], y=[float(i)]) for i in range(4)]
    assert P.saturation(cells)
    protocol = {"oracles": [{"name": "o1", "kind": "invariant", "measure": "viol"}]}
    from core import degenerate_results as dr
    exempt = dr.expected_constants(protocol)
    assert exempt == P.exempt_names(protocol)
    assert "viol" in exempt
    assert P.saturation(cells, exempt) == []
    assert P.saturation([cell(viol=[float("nan")])], {"viol"})   # NaN is never exempt


def test_saturation_all_constant_run():
    got = P.saturation([cell(a=[2.0], b=[3.0]) for _ in range(4)])
    assert {v.relation_id for v in got} == {"saturation:a", "saturation:b"} and all(v.verdict == "broken" for v in got)


# ---------------------------------------------------------------------------------------------------------------- mutators

SAMPLE = '''
import math
import numpy as np

def run_trial(cell, trial_id, seed):
    rng = np.random.default_rng(seed)
    dt = float(cell["dt"])
    n = int(cell.get("n", 100))
    noise = rng.normal(0, 0.1)
    err = dt * 2.0 + noise
    rate = min(1.0, n / 1000.0 + noise)
    return {"err": err, "rate": rate, "total": err + rate}
'''


def run_src(src, cell, seed=0):
    ns: dict = {}
    exec(compile(src, "<m>", "exec"), ns)
    return ns["run_trial"](cell, 0, seed)


def test_mutators_make_compilable_changed_code():
    ms = P.mutants(SAMPLE, {"dt": 0.1, "n": 100}, ["err", "rate"], {"dt": 0.05})
    names = [n for n, _ in ms]
    assert names == ["ignore_setting:dt", "ignore_setting:n", "hard_code:err", "clip:err", "swap_sign:err", "hard_code:rate",
                     "clip:rate", "swap_sign:rate", "coarse_equals_fine:dt"]
    for name, src in ms:
        compile(src, name, "exec")
        assert src.strip() != SAMPLE.strip()
    assert names == [n for n, _ in P.mutants(SAMPLE, {"dt": 0.1, "n": 100}, ["err", "rate"], {"dt": 0.05})]


def test_mutators_behave_as_named():
    d = dict(P.mutants(SAMPLE, {"dt": 0.1, "n": 100}, ["err", "rate"], {"dt": 0.05}))
    a = run_src(d["ignore_setting:dt"], {"dt": 0.4}, 3)
    b = run_src(d["ignore_setting:dt"], {"dt": 0.05}, 3)
    assert a["err"] == b["err"]
    assert run_src(d["hard_code:err"], {"dt": 0.4})["err"] == 0.5
    assert run_src(d["clip:err"], {"dt": 0.4})["err"] == 1.0 and run_src(d["clip:err"], {"dt": 0.05})["err"] == 1.0
    base = run_src(SAMPLE, {"dt": 0.4}, 3)["err"]
    assert run_src(d["swap_sign:err"], {"dt": 0.4}, 3)["err"] != base
    assert run_src(d["coarse_equals_fine:dt"], {"dt": 0.4}, 3)["err"] == run_src(d["coarse_equals_fine:dt"], {"dt": 0.05}, 3)["err"]


def test_mutators_return_none_when_nothing_to_change():
    assert P.ignore_setting(SAMPLE, "absent", 1) is None
    assert P.hard_code_metric(SAMPLE, "absent") is None
    assert P.swap_sign("def f(:", "x") is None
    assert P.mutants("x = 1\n", {"dt": 1}, ["err"]) == []


def test_probes_catch_the_mutants_they_should():
    """End to end on synthetic cells: sensitivity to dt holds on the original and breaks on the mutants."""
    d = dict(P.mutants(SAMPLE, {"dt": 0.1}, ["err"]))

    def verdict(src):
        o = [P.Observation({"dt": dt}, [run_src(src, {"dt": dt}, s)["err"] for s in range(8)]) for dt in (0.05, 0.4)]
        return P.judge(rel("sensitivity"), o).verdict

    assert verdict(SAMPLE) == "held"
    assert verdict(d["ignore_setting:dt"]) == "broken"
    assert verdict(d["hard_code:err"]) == "broken"
    assert verdict(d["clip:err"]) == "broken"


KINETICS = sorted(Path("C:/fi_runs/suite/rounds/r6t/first_order_kinetics/out").glob("*/code/simulate.py"))


@pytest.mark.skipif(not KINETICS, reason="stored r6t kinetics simulate.py not on this machine")
def test_mutators_on_stored_simulate_py_import():
    src = KINETICS[0].read_text(encoding="utf-8")
    ms = P.mutants(src, {"noise_sigma_rel": 0.01, "n_time": 12}, ["fitted_ea_j_mol", "ln_a_hat"])
    assert any(n == "ignore_setting:noise_sigma_rel" for n, _ in ms)
    assert any(n.startswith("hard_code:") for n, _ in ms)
    for name, text in ms:
        compile(text, name, "exec")
