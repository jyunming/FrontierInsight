"""A name in the Limitations is the quantity's own, never the path to one cell of the analysis's output or a column pair."""
from __future__ import annotations

import json
from pathlib import Path

from core import disclosure, frozen_protocol, paper_facts, trial_checks

PROTOCOL = {
    "grid": {"n": [5, 10]}, "runs_per_setting": 20,
    "metrics": [{"id": "Ea_hat", "label": "Estimated activation energy", "kind": "mean",
                 "known_value": {"true_value": "Ea_true", "estimate": "Ea_hat", "standard_error": "Ea_se",
                                 "interval": ["Ea_lo", "Ea_hi"], "level": 0.95}}],
}


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "q"
    (root / ".fi").mkdir(parents=True)
    frozen_protocol.freeze(root, PROTOCOL, approved_by="test", source="test")
    return root


def test_per_cell_precision_lines_are_one_item_per_metric_not_per_path(tmp_path: Path) -> None:
    root = _root(tmp_path)
    missed = [{"name": f"by_n.n={n}.coverage_Ea", "half_width": 0.03 + n / 1000, "target": 0.03, "trials": 195} for n in (5, 8, 12)]
    missed.append({"name": "coverage_Ea", "half_width": 0.088, "target": 0.03, "trials": 39})
    (root / disclosure.PRECISION_RECORD).write_text(json.dumps(missed), encoding="utf-8")
    items = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.PRECISION]
    assert len(items) == 1
    assert "by_n" not in items[0].sentence and "n=5" not in items[0].sentence, items[0].sentence
    assert "`coverage_Ea`" in items[0].sentence and "4 of the settings" in items[0].sentence


def test_a_pair_of_interval_columns_is_said_by_the_plans_label(tmp_path: Path) -> None:
    root = _root(tmp_path)
    q = {"kind": "coverage", "label": "Ea_lo .. Ea_hi", "names": ["Ea_lo", "Ea_hi"], "level": 0.95, "judged": 25, "below": 1,
         "fault": False, "settings": [{"value": 0.82}]}
    trial_checks.write_record(root, {"calibration": {"checked": True, "quantities": [q]}})
    items = [i for i in paper_facts.required_items(root) if i.kind == paper_facts.CALIBRATION]
    assert items and "Ea_lo" not in items[0].sentence and "Estimated activation energy" in items[0].sentence
