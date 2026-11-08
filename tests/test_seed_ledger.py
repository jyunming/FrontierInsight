"""The seed FI gives each trial reaches the simulation when FI's own trial ledger shows it, and the writer is told what the
seed means for the intervals (core/run_facts.py::seed_reaches_by_ledger, core/paper_facts.py::seed_meaning_text).

Before: a simulation that hands the seed to a library FI cannot read was "unknown" even when the ledger showed 40 trials with
40 seeds and 40 different outcomes; the log said "may not reach (FI cannot tell)" and the writer, told only "do not state the
seed", typed that the intervals describe "one seed" and leave seed-to-seed variation out.
"""

from __future__ import annotations

import json
from pathlib import Path

from core import paper_facts, run_facts

LIBRARY = "import somelib\ndef run_trial(cell, trial_id, seed):\n    return somelib.simulate(cell, seed=seed)\n"


def _quest(tmp_path: Path, seeds: list[int], outcomes: list[str], drew: bool | None = None, src: str = LIBRARY,
           repeat: str | None = "same") -> Path:
    """``repeat``: the small run ran trial 0's seed again and got the study's outcome ("same"), another ("other"), or FI
    has no second run of any seed (None)."""
    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "raw").mkdir()
    (root / "code" / "simulate.py").write_text(src, encoding="utf-8")
    (root / "code" / "study.json").write_text(json.dumps({"base_seed": 11}), encoding="utf-8")
    rows = [{"event": "planned", "cell": "a", "trials": len(seeds)}]
    rows += [{"event": "trial", "cell": "a", "trial": i, "seed": s, "status": "ok", "values_sha256": h}
             for i, (s, h) in enumerate(zip(seeds, outcomes))]
    (root / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    if repeat is not None:
        for sub, vals in (("small_run/s0", {"x": 1.0}), ("trials", {"x": 1.0 if repeat == "same" else 2.0})):
            (root / ".fi" / sub).mkdir(parents=True, exist_ok=True)
            (root / ".fi" / sub / "cell0.out.jsonl").write_text(
                json.dumps({"trial": 0, "seed": seeds[0], "status": "ok", "values": vals}) + "\n", encoding="utf-8")
    if drew is not None:
        (root / ".fi" / "trials").mkdir(parents=True, exist_ok=True)
        (root / ".fi" / "trials" / "cell9.out.jsonl").write_text(
            "\n".join(json.dumps({"trial": i, "status": "ok", "drew": drew}) for i in range(len(seeds))), encoding="utf-8")
    return root


def test_different_seeds_with_different_outcomes_decide_it_not_the_code_trace(tmp_path: Path) -> None:
    root = _quest(tmp_path, [1, 2, 3, 4], ["a", "b", "c", "d"])
    got = run_facts.seed_verdict(root)
    assert got["verdict"] == "reaches" and got["fi_seed_reaches"] is True and got["decided_by"] == "ledger", got
    assert paper_facts._seed_text(root) == "a separate seed for each trial, derived from base seed 11"


def test_the_writer_is_told_the_intervals_include_seed_variation(tmp_path: Path) -> None:
    text = paper_facts.seed_meaning_text(_quest(tmp_path, [1, 2, 3, 4], ["a", "b", "c", "d"]))
    assert "own seed derived from base seed 11" in text and "include the variation between seeds" in text, text
    assert "single-seed" in text  # said as what not to write


def test_known_positive_one_seed_for_every_trial_is_still_said(tmp_path: Path) -> None:
    root = _quest(tmp_path, [5, 5, 5, 5], ["a", "a", "a", "a"])
    assert run_facts.seed_verdict(root)["verdict"] == "unknown"  # the ledger does not decide: the code stays unknown
    assert "same seed" in paper_facts.seed_meaning_text(root)


def test_known_positive_one_seed_that_reaches_is_said_to_be_one_seed(tmp_path: Path) -> None:
    src = "import numpy as np\ndef run_trial(cell, trial_id, seed):\n    return np.random.default_rng(seed).normal()\n"
    root = _quest(tmp_path, [5, 5, 5, 5], ["a", "b", "c", "d"], src=src)
    text = paper_facts.seed_meaning_text(root)
    assert "same seed" in text and "do not include seed-to-seed variation" in text, text


def test_known_positive_no_trial_drew_random_numbers(tmp_path: Path) -> None:
    root = _quest(tmp_path, [1, 2, 3], ["a", "a", "a"], drew=False)
    assert run_facts.seed_verdict(root)["verdict"] == "unknown"
    assert "No trial drew random numbers" in paper_facts.seed_meaning_text(root)


def test_the_writer_section_carries_the_seed_fact(tmp_path: Path) -> None:
    root = _quest(tmp_path, [1, 2, 3, 4], ["a", "b", "c", "d"])
    section = paper_facts.writer_section(root)
    assert "Seed: Each of the trials ran on its own seed" in section, section


def test_known_negative_distinct_outcomes_but_a_repeated_seed_differing_is_not_reaches(tmp_path: Path) -> None:
    """An unseeded generator: different seeds give different outcomes, and so does the same seed run twice."""
    root = _quest(tmp_path, [1, 2, 3, 4], ["a", "b", "c", "d"], repeat="other")
    assert run_facts.seed_repeat_agrees(root) is False
    assert run_facts.seed_verdict(root)["verdict"] == "unknown"


def test_no_seed_run_twice_keeps_unknown(tmp_path: Path) -> None:
    root = _quest(tmp_path, [1, 2, 3, 4], ["a", "b", "c", "d"], repeat=None)
    assert run_facts.seed_repeat_agrees(root) is None
    assert run_facts.seed_verdict(root)["verdict"] == "unknown"
