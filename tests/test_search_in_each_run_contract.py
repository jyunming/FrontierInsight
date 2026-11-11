"""A search block's own ``in_each_run``: the contract learns the setting, and the code is told that a quantity the plan
declares one number per run is returned as one number per call, not one per value of the setting."""
from core import contract as _contract
from core import in_run as _in_run
from core import skeleton as _skeleton


def _protocol(objective_depends_on):
    return {
        "form": "contract",
        "metrics": [
            {"id": "cost", "estimand": "total cost of the design", "depends_on": list(objective_depends_on)},
            {"id": "load", "estimand": "load at the design", "depends_on": ["size"]},
        ],
        "optimisation": {
            "objective": {"quantity": "cost", "direction": "minimise"},
            "design_variables": [{"name": "width", "kind": "continuous", "low": 1, "high": 2}],
            "constraints": [{"quantity": "strength", "limit": ">= 1"}],
            "in_each_run": {"mode": ["slow", "fast"]},
        },
    }


def _contract_of(protocol):
    return _contract.from_design({"protocol": protocol})


def test_contract_reads_the_search_blocks_in_each_run():
    c = _contract_of(_protocol(["size"]))
    assert c.settings.in_each_run == {"mode": ("slow", "fast")}


def test_search_blocks_own_wins_over_a_top_level_one_as_the_run_reads_it():
    p = _protocol(["mode"])
    p["in_each_run"] = {"other": [1, 2]}
    assert set(_contract_of(p).settings.in_each_run) == {"mode"}
    assert set(_in_run.of_study(p)) == {"mode"}


def test_top_level_in_each_run_still_read_when_the_block_has_none():
    p = _protocol(["size"])
    del p["optimisation"]["in_each_run"]
    p["in_each_run"] = {"other": [1, 2]}
    assert set(_contract_of(p).settings.in_each_run) == {"other"}


def test_objective_not_declared_as_a_metric_still_gets_the_rule():
    p = _protocol(["size"])
    p["metrics"] = [m for m in p["metrics"] if m["id"] != "cost"]
    assert _in_run.search_one_number_names(p) == ["cost", "strength"]
    p["metrics"].append({"id": "cost", "estimand": "x", "depends_on": []})
    assert _in_run.search_one_number_names(p) == ["cost", "strength"]


def test_one_number_rule_names_objective_and_limits():
    p = _protocol(["size"])
    assert _in_run.search_one_number_names(p) == ["cost", "strength"]
    rule = _in_run.one_number_rule(p)
    assert "`cost`" in rule and "`strength`" in rule and "`mode`" in rule and "one plain number" in rule
    assert "one plain number" in _skeleton.level_rule(p, "")


def test_known_positive_objective_that_depends_on_the_setting_keeps_per_value():
    p = _protocol(["size", "mode"])
    assert _in_run.search_one_number_names(p) == []
    assert _in_run.one_number_rule(p) == ""
    assert "one plain number" not in _skeleton.level_rule(p, "")
    assert "cost" not in _in_run.per_run_names(p)


def test_no_in_each_run_no_rule():
    p = _protocol(["size"])
    del p["optimisation"]["in_each_run"]
    assert _in_run.one_number_rule(p) == ""
