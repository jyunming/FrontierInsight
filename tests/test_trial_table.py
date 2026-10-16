"""FI's own table of what every trial gave, per setting (replay of suite r9k first_order_kinetics: the analysis and the
paper gave the OLS Ea SD at 8% noise and 5 points as 1,920; the 400 trials give 1,849.0)."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from core import stats
from core import trial_table

FIXTURE = Path(__file__).parent / "fixtures" / "trial_table_r9k"
FIVE = "concentration_noise_sd_fraction_of_c0=0.08,n_time_points=5"


def _quest(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    return tmp_path


def _row(table: list[dict[str, Any]], setting: str, metric: str) -> dict[str, Any]:
    return next(r for r in table if r["setting"] == setting and r["metric"] == metric)


def test_r9k_s_sd_comes_from_the_trials(tmp_path: Path) -> None:
    table = trial_table.rows(_quest(tmp_path))
    row = _row(table, FIVE, "ols_Ea_hat_J_mol")
    assert row["n"] == 400 and round(row["sd"], 1) == 1849.0
    assert row["method"] == "t interval of the mean" and row["ci_lower"] < row["mean"] < row["ci_upper"]


def test_index_and_constant_values_are_left_out(tmp_path: Path) -> None:
    metrics = {r["metric"] for r in trial_table.rows(_quest(tmp_path))}
    assert "trial_id" not in metrics  # 0..399 in every setting
    assert "noise_level" not in metrics  # the same in every trial of every setting


def test_a_value_every_trial_gives_as_0_or_1_gets_a_wilson_interval(tmp_path: Path) -> None:
    cells = [{"cell": {"m": m}, "key": f"m={m}", "metrics": {"any_false_discovery": {"values": vals}}}
             for m, vals in ((20, [0, 1, 1, 0, 1]), (200, [1, 1, 1, 0, 1]))]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    row = _row(trial_table.rows(tmp_path), "m=20", "any_false_discovery")
    assert row["method"] == "Wilson" and (row["ci_lower"], row["ci_upper"]) == stats.wilson_interval(3, 5)


def test_the_file_and_the_writer_s_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path)
    table = trial_table.write(root)
    text = (root / "paper" / "results_table.md").read_text(encoding="utf-8")
    assert f"| {FIVE} | ols_Ea_hat_J_mol | 400 |" in text and "| 1849 |" in text
    assert len(table) <= trial_table.WRITER_MAX_ROWS and "| 1849 |" in trial_table.write_note(table)
    monkeypatch.setattr(trial_table, "WRITER_MAX_ROWS", len(table) - 1)
    assert trial_table.write_note(table) == ""
    assert trial_table.write(tmp_path / "none") == [] and not (tmp_path / "none" / "paper").exists()


def test_the_writer_gets_the_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_disclosure import _engine

    seen: list[str] = []

    async def fake_chat(self: Any, messages: Any, **kw: Any) -> str:
        seen.append(messages[-1]["content"] if isinstance(messages, list) else str(messages))
        return "# T\n\n## Results\nThe OLS SD was 1,849.0 J/mol.\n\n## Limitations\nNone.\n"

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    root = _quest(tmp_path)
    eng = _engine(root)
    asyncio.run(eng._node_write({"topic": "t", "title": "t", "iteration": 0,  # type: ignore[arg-type]
                                 "exec_result": {"returncode": 0}, "result_json": {"x": 1.5}}))
    assert (root / "paper" / "results_table.md").exists()
    assert any("FI's table of what every trial gave" in p and "| 1849 |" in p for p in seen)


def test_malformed_records_are_skipped_not_fatal(tmp_path: Path) -> None:
    cells = [{"cell": {"m": 1}, "key": "m=1", "metrics": {"loss": 0.05, "x": {"values": 3.0},
                                                          "y": {"values": [1.0, float("nan"), 2.0, "a", None]}}},
             {"cell": {"m": 2}, "key": "m=2", "metrics": {"y": {"values": [4.0, 5.0]}}}]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}).replace("NaN", "NaN"), encoding="utf-8")
    table = trial_table.rows(tmp_path)
    assert {r["metric"] for r in table} == {"y"} and _row(table, "m=1", "y")["n"] == 2
    assert "nan" not in trial_table.markdown(table)


