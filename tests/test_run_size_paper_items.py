"""What the paper must say when FI made the run smaller or went on with a difference in the run's record, and the precision
a run reached for a mean as for a probability.

* Every run FI made smaller is a required item of the paper's limitations, with the runs against the plan's, why, whether it
  is below the plan's minimum and the 95% interval those runs give: a cut for the topic's limit with no minimum stated (1000
  runs down to 195) was not named at all.
* A precision target on a mean whose spread the timed trials did not show is not known before the run: the plan's own runs
  are never read as its "minimum", and the paper is told the precision was not known.
* A missed precision target is a required item for a mean pooled from its values too.
* Under ``engine.run_manifest_check: warn`` a difference FI recorded and went on with is a required item.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core import paper_facts
from core import run_estimate as est
from core.engine import Engine, _aggregate_result_json_replicates

GRID = {"m": [20]}
MEAN_TARGET = {"precision": {"target_half_width": 0.01, "metric": "x"}, "metrics": [{"id": "x", "kind": "mean"}]}


def _probes() -> list[est.Probe]:
    # one setting: 0.5 s once, 0.1 s per run; 1000 runs = 100.5 s, over 30 s with the margin
    return [est.Probe(None, None, {"m": 20}, 0.5 + 1000 * 0.1, fixed_s=0.5, per_trial_s=0.1, timed_trials=50)]


def _fit(protocol: dict[str, Any], *, topic: bool, spreads: dict[str, float] | None = None) -> Any:
    return est.fit_runs(GRID, 1000, False, _probes(), protocol, limit_s=30.0, topic_limit=topic, spreads=spreads or {})


# ---- the sizing rule --------------------------------------------------------------------------------------------------


def test_with_no_minimum_stated_only_the_topics_limit_cuts_and_it_goes_no_lower_than_ten() -> None:
    assert _fit({}, topic=False) is None, "FI's own limit never makes up a number of runs"
    fit = _fit({}, topic=True)
    assert fit is not None and fit.runs == 195 and fit.plan_floor is None and not fit.below_plan and fit.precision_known
    # 10 runs per setting do not fit 0.5 s: nothing below MIN_RUNS_FOR_LIMIT
    tiny = est.fit_runs(GRID, 1000, False, _probes(), {}, limit_s=0.5, topic_limit=True)
    assert tiny is None and est.MIN_RUNS_FOR_LIMIT == 10


def test_a_mean_target_whose_spread_is_unknown_never_reads_the_plans_runs_as_its_minimum() -> None:
    assert _fit(MEAN_TARGET, topic=False) is None, "no cut for FI's own limit below a precision it cannot work out"
    fit = _fit(MEAN_TARGET, topic=True)
    assert fit is not None and fit.runs == 195
    assert fit.plan_floor is None and not fit.below_plan and not fit.precision_known  # before: plan_floor=1000, below_plan
    stated = _fit({**MEAN_TARGET, "min_runs_per_setting": 400}, topic=True)
    assert stated is not None and stated.plan_floor == 400 and stated.below_plan and not stated.precision_known
    known = _fit(MEAN_TARGET, topic=True, spreads={"x": 0.1})
    assert known is not None and known.precision_known and known.plan_floor == 385 and known.below_plan
    assert est.expected_precision(MEAN_TARGET, known.runs, {"x": 0.1})["half_width"] == pytest.approx(0.014, abs=1e-3)
    # A plan of means with no target: the precision of its first mean whose spread is known (before: nothing).
    means = {"metrics": [{"id": "y", "kind": "mean"}, {"id": "x", "kind": "mean"}]}
    assert est.expected_precision(means, 100, {"x": 1.0})["metric"] == "x"
    assert est.expected_precision(means, 100, {"x": 1.0})["half_width"] == pytest.approx(0.196, abs=1e-3)
    assert est.expected_precision(means, 100, {}) is None


def test_a_settings_cost_once_is_never_negative() -> None:
    probe = est.Probe(None, None, {"m": 20}, 49.7, fixed_s=-0.3, per_trial_s=0.05)
    assert probe.at(10, 1000).seconds == pytest.approx(0.5)  # was 0.2: less than the 10 runs alone


# ---- the paper's required item ------------------------------------------------------------------------------------------


def _quest(tmp_path: Path, reduced: dict[str, Any] | None = None, manifest: dict[str, Any] | None = None) -> Path:
    root = tmp_path / "quest"
    (root / ".fi").mkdir(parents=True)
    if reduced is not None:
        (root / ".fi" / "run_estimate.json").write_text(json.dumps({"fits": True, "reduced": reduced}), encoding="utf-8")
    if manifest is not None:
        (root / "needs").mkdir()
        (root / "needs" / "RUN_MANIFEST_CHECK.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _reduced(**over: Any) -> dict[str, Any]:
    return {"note": "n", "from": {"grid": GRID, "runs_per_setting": 1000}, "to": {"grid": GRID, "runs_per_setting": 195},
            "limit": "topic", "below_plan_minimum": False, "plan_minimum": None, "precision": None, "over_topic_limit": False,
            **over}


def test_every_cut_is_an_item_with_its_numbers_and_a_paper_that_leaves_it_out_is_sent_back(tmp_path: Path) -> None:
    root = _quest(tmp_path, _reduced(precision={"metric": "power", "runs": 195, "half_width": 0.0702, "how": "h",
                                                "target": None}))
    items = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.FEWER_RUNS]
    assert [i.name for i in items] == ["fewer runs than the plan asked for"]
    text = items[0].text
    assert "195 runs per setting instead of the 1000 the plan asked for" in text
    assert "fits the topic's time limit" in text and "the plan states no fewest number of runs" in text
    assert "95% interval of power is about ±0.0702" in text
    # The writer places it as {{not_done:<id>}} in the Limitations; FI puts the sentence there from its record.
    assert items[0].id == "fewer-runs" and "{{not_done:fewer-runs}}" in paper_facts.writer_section(root, topic="t")
    bare = "# P\n\n## Limitations\nThe grid is coarse.\n"
    hits = paper_facts.placement_hits(bare, root)
    assert len(hits) == 1 and hits[0].startswith("paper_record_mismatch:") and "not_done:fewer-runs" in hits[0]
    placed = bare + "{{not_done:fewer-runs}}\n"
    assert paper_facts.placement_hits(paper_facts.render(placed, root).text, root) == []
    assert "The experiment ran 195 runs per setting instead of the 1000" in paper_facts.render(placed, root).text


def test_the_item_says_when_the_precision_was_not_known_and_when_below_the_minimum(tmp_path: Path) -> None:
    text = paper_facts.fewer_runs_text(_reduced(below_plan_minimum=True, plan_minimum=400, precision_known=False))
    assert "below the plan's minimum of 400" in text and "not known before the run" in text
    # A cut for FI's own limit is named too.
    own = paper_facts.fewer_runs_text(_reduced(limit="execution.timeout_s", plan_minimum=100))
    assert "FI's time limit for one run" in own and "below the plan's minimum" not in own
    # No fewer runs, no item.
    assert paper_facts.fewer_runs_text(_reduced(to={"runs_per_setting": 1000})) == ""
    assert not paper_facts.required_items(_quest(tmp_path, _reduced(to={"runs_per_setting": 1000})))


@pytest.mark.asyncio
async def test_the_note_of_a_cut_with_an_unknown_spread_says_the_precision_was_not_known(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_run_checks import _sizing

    async def off(self: Engine, state: Any) -> None:
        return None

    monkeypatch.setattr(Engine, "_read_topic_before_the_prompt", off)  # the first scripted reply is the plan
    eng, _model, _runner, _ex = await _sizing(tmp_path, timeout_s=900, runs=1000)
    fit = _fit(MEAN_TARGET, topic=True)
    estimate = est.estimate(GRID, 1000, False, _probes())
    note = eng._lower_the_runs({"iteration": 0}, 1000, fit, estimate, 30.0, GRID, MEAN_TARGET, topic=True)
    assert note is not None and note["precision"] is None and note["precision_known"] is False
    assert note["plan_minimum"] is None and not note["below_plan_minimum"]
    assert "was not known before the run" in note["note"] and "minimum of 1000" not in note["note"]
    assert "not known before the run" in paper_facts.fewer_runs_text(note)


# ---- a missed precision target, for a mean too ----------------------------------------------------------------------------


def _annotate(protocol: dict[str, Any], replicates: list[dict[str, Any]], kinds: dict[str, str]) -> tuple[str, dict[str, Any]]:
    aggregate = _aggregate_result_json_replicates(replicates, kinds=kinds)
    fake = SimpleNamespace(_protocol_block=lambda state: protocol)
    return Engine._annotate_precision(fake, {}, aggregate), aggregate  # type: ignore[arg-type]


def test_a_mean_pooled_from_its_values_is_held_to_its_precision_target() -> None:
    # 2 seeds x 50 values spread about 1: a 95% interval of about ±0.2, against a target of ±0.01
    vals = [[float(i % 5) for i in range(50)], [float((i + 2) % 5) for i in range(50)]]
    replicates = [{"_seed": s, "x": sum(v) / len(v), "x_values": v} for s, v in enumerate(vals)]
    said, aggregate = _annotate(MEAN_TARGET, replicates, {"x": "mean"})
    assert aggregate["x"]["precision_reached"] is False and aggregate["x"]["n_values"] == 100
    assert "was not reached for 1 of the quantities it is on: x" in said
    # The same mean, the target on another metric (a proportion): a mean's scale is its own, it is not held to it.
    other = {"precision": {"target_half_width": 0.01, "metric": "p"},
             "metrics": [{"id": "x", "kind": "mean"}, {"id": "p", "kind": "proportion"}]}
    said, aggregate = _annotate(other, replicates, {"x": "mean"})
    assert said == "" and "precision_reached" not in aggregate["x"]


def test_a_probability_is_held_as_before() -> None:
    protocol = {"precision": {"target_half_width": 0.01}, "metrics": [{"id": "p", "kind": "proportion"}]}
    replicates = [{"_seed": s, "p": 0.5, "p_count": 50, "p_total": 100} for s in range(2)]
    said, aggregate = _annotate(protocol, replicates, {"p": "proportion"})
    assert aggregate["p"]["precision_reached"] is False and "of the quantities it is on: p" in said


# ---- a difference in the run's record, gone on with under warn ------------------------------------------------------------


def test_a_warned_difference_is_an_item_in_plain_words(tmp_path: Path) -> None:
    headline = "The analysis leaves out trials of one setting (for `final_size`: ...) (settings: R0=0.9)."
    root = _quest(tmp_path, manifest={"status": "warned", "problems": ["the metric `final_size` is measured ..."],
                                      "headline": headline})
    items = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.RUN_DIFFERENCE]
    assert [i.name for i in items] == ["difference from the run record"] and headline in items[0].text
    assert paper_facts.placement_hits("# P\n\nNothing.\n", root)
    placed = f"# P\n\n## Limitations\n{{{{not_done:{items[0].id}}}}}\n"
    assert paper_facts.placement_hits(paper_facts.render(placed, root).text, root) == []
    assert "does not match what the analysis reports" in paper_facts.render(placed, root).text
    # Without a plain sentence, the first finding stands for it; an ok record gives no item.
    first = _quest(tmp_path / "b", manifest={"status": "warned", "problems": ["one", "two"]})
    assert "one (and 1 more)" in paper_facts.required_items(first)[0].text
    assert not paper_facts.required_items(_quest(tmp_path / "c", manifest={"status": "ok", "problems": []}))
