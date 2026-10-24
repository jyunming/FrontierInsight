"""A precision target on a mean the run gave without its values: FI's own trial record (raw/trials.json) has each
setting's trials, so the target is held to each setting's own interval (never one pooled over settings whose true
values differ), said as k of n settings and the widest; it never sends the quest back to design."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core import evidence, quality_report as qr, trial_table
from core.config import (Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig)
from core.engine import Engine

PROTOCOL = {"precision": {"target_half_width": 0.05, "metric": "settle_time"},
            "metrics": [{"id": "settle_time", "kind": "mean"}]}


def _cells(spreads: list[float], n: int = 20) -> list[dict[str, Any]]:
    out = []
    for i, d in enumerate(spreads):
        values = [2.0 + (d if k % 2 else -d) for k in range(n)]
        out.append({"cell": {"gain": i}, "key": f"gain={i}", "metrics": {"settle_time": {"values": values}}})
    return out


def _record(root: Path, cells: list[dict[str, Any]]) -> Path:
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    return root


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"), engine=EngineConfig(),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    ))
    eng._protocol_block = lambda _state: PROTOCOL  # type: ignore[method-assign]
    return eng


def test_each_setting_is_judged_on_its_own_trials(tmp_path: Path) -> None:
    root = _record(tmp_path, _cells([0.01, 0.01, 0.5]) + [
        {"cell": {"gain": 9}, "key": "gain=9", "metrics": {"settle_time": {"values": [2.0]}}}])
    got = trial_table.per_setting_precision(root, "settle_time", 0.05)
    assert got is not None
    assert (got["settings"], got["met"], got["without_interval"], got["widest_setting"]) == (4, 2, 1, "gain=2")
    assert 0.2 < got["widest"] < 0.3
    words = evidence.precision_per_setting_words(got)
    assert "2 of 4 settings" in words and "`settle_time`" in words and "1 more setting" in words
    # every setting met: no gap; no record or no interval anywhere: nothing to say
    assert evidence.precision_per_setting_words(trial_table.per_setting_precision(
        _record(tmp_path / "ok", _cells([0.01, 0.02])), "settle_time", 0.05)) == ""
    assert trial_table.per_setting_precision(tmp_path / "none", "settle_time", 0.05) is None
    assert trial_table.per_setting_precision(root, "other", 0.05) is None


def test_a_setting_with_no_trials_is_not_a_target_met(tmp_path: Path) -> None:
    """Old code: 3 settings, one with no trials, read as 2 of 2 met."""
    cells = _cells([0.01, 0.01]) + [{"cell": {"gain": 7}, "key": "gain=7", "metrics": {}, "planned": 20, "ok": 0}]
    got = trial_table.per_setting_precision(_record(tmp_path, cells), "settle_time", 0.05)
    assert got is not None and (got["settings"], got["met"], got["without_value"]) == (3, 2, ["gain=7"])
    words = evidence.precision_per_setting_words(got)
    assert "2 of 3 settings" in words and "1 setting(s) have no value (gain=7)" in words


def test_settings_are_never_pooled(tmp_path: Path) -> None:
    """Each setting's interval comes from its 40 trials alone (about ±0.1), never from all 80 together."""
    cells = [{"cell": {"g": g}, "key": f"g={g}", "metrics": {"settle_time": {"values": [g + 0.3 * (-1) ** k for k in range(40)]}}}
             for g in (1.0, 2.0)]
    got = trial_table.per_setting_precision(_record(tmp_path, cells), "settle_time", 0.05)
    assert (got["settings"], got["met"]) == (2, 0) and 0.08 < got["widest"] < 0.11


def test_the_evidence_record_holds_the_target_to_each_setting_not_not_judged(tmp_path: Path) -> None:
    """Old code: the mean had no pooled interval, so the target was recorded as not judged although the trial record
    had every setting's trials."""
    eng = _engine(tmp_path)
    _record(eng.quest_root, _cells([0.01, 0.5]))
    state = {"result_json_replicates": [{"settle_time": 2.0}, {"settle_time": 2.1}]}
    record = eng._write_evidence(state, write=False)
    assert "precision_not_judged" not in record
    fact = record["precision_per_setting"]
    assert (fact["settings"], fact["met"]) == (2, 1)
    assert any("1 of 2 settings" in g for g in record["all_gaps"]["statistically_adequate"])
    # the analysis is told the same, in words
    assert "1 of 2 settings" in eng._annotate_precision(state, {})
    # never a route back to design
    assert eng._precision_shortfalls(state) == []
    # every setting met: the record keeps it, with no gap
    _record(eng.quest_root, _cells([0.01, 0.02]))
    record = eng._write_evidence(state, write=False)
    assert record["precision_per_setting"]["met"] == 2
    assert not any("precision target" in g for g in record["all_gaps"].get("statistically_adequate", []))
    assert eng._annotate_precision(state, {}) == ""


def test_known_positive_one_trial_per_setting_is_still_not_judged(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    _record(eng.quest_root, [{"cell": {"g": 1}, "key": "g=1", "metrics": {"settle_time": {"values": [2.0]}}}])
    record = eng._write_evidence({"result_json_replicates": [{"settle_time": 2.0}]}, write=False)
    assert record["precision_not_judged"] == {"target": 0.05, "metric": "settle_time"}
    assert "precision_per_setting" not in record


def test_the_quality_report_says_a_target_missed_in_some_settings(tmp_path: Path) -> None:
    fact = {"target": 0.05, "metric": "settle_time", "settings": 4, "met": 3, "widest": 0.08,
            "widest_setting": "gain=2", "without_interval": 0}
    q = qr._Quest.__new__(qr._Quest)
    q.evidence = {"precision_per_setting": fact}
    items = qr._precision_per_setting(q)
    assert len(items) == 1 and "3 of 4 settings" in items[0]["text"] and items[0]["status"] == "not fixed"
    q.evidence = {"precision_per_setting": {**fact, "met": 4}}
    assert qr._precision_per_setting(q) == []
    for bad in ({}, {"settings": "x"}, "not a record"):
        q.evidence = {"precision_per_setting": bad}
        assert qr._precision_per_setting(q) == []
