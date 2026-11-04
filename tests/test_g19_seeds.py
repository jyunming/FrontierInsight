"""Seeds shared across settings are stated in the Limitations from FI's own trial ledger (core/paper_facts.py).
No model is called."""

from __future__ import annotations

import json
from pathlib import Path

from core import paper_facts


def _quest(tmp_path: Path, seeds: dict[str, list[int]], paired_flag: bool | None = None) -> Path:
    (tmp_path / "raw").mkdir()
    (tmp_path / ".fi").mkdir()
    lines = []
    for cell, row in seeds.items():
        lines.append({"event": "planned", "cell": cell, "trials": len(row)})
        for t, seed in enumerate(row):
            lines.append({"event": "trial", "cell": cell, "trial": t, "status": "ok", "seed": seed})
    (tmp_path / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in lines), encoding="utf-8")
    if paired_flag is not None:
        (tmp_path / "code").mkdir()
        (tmp_path / "code" / "study.json").write_text(json.dumps({"base_seed": 7, "paired": paired_flag}), encoding="utf-8")
    return tmp_path


def _items(root: Path):
    return [i for i in paper_facts.required_items(root) if i.kind == paper_facts.SHARED_SEEDS]


def test_settings_sharing_trial_seeds_are_an_item_of_the_limitations(tmp_path: Path) -> None:
    """Old code: a paired plan (trial 0 on one seed in every cell) was never named as an item, so the paper could pool
    the cells' intervals as if independent without saying the cells move together."""
    root = _quest(tmp_path, {"gain=1": [11, 12], "gain=2": [11, 12]})
    (item,) = _items(root)
    assert "same random seed" in item.sentence and "not independent" in item.sentence
    assert item.id.startswith("shared-seeds")


def test_settings_with_their_own_seeds_have_no_such_item(tmp_path: Path) -> None:
    """Known negative: different seeds per setting give no sentence."""
    assert _items(_quest(tmp_path, {"gain=1": [1, 2], "gain=2": [3, 4]})) == []


def test_one_setting_has_no_such_item(tmp_path: Path) -> None:
    assert _items(_quest(tmp_path, {"gain=1": [1, 2]})) == []
