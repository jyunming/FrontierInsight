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




def _left_out_quest(tmp_path: Path, record: str = "RESULT_CHECK.json") -> Path:
    cells = [{"cell": {"m": m}, "key": f"m={m}",
              "metrics": {"est": {"values": [1.0 + m, 2.0 + m, 3.5]}, "gone": {"values": [float(m), m * 2.0, 7.0 * m]}}}
             for m in (1, 2)]
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / "needs").mkdir()
    (tmp_path / "needs" / record).write_text(
        json.dumps({"removed_quantities": ["gone"], "note": "typed into the code"}), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("record", ["RESULT_CHECK.json", "TYPED_RESULTS_CHECK.json"])
def test_a_quantity_fi_left_out_is_in_no_table_shown_to_the_writer(tmp_path: Path, record: str) -> None:
    from core import results_by_setting
    root = _left_out_quest(tmp_path, record)
    assert {r["metric"] for r in trial_table.rows(root)} == {"est"}
    note = trial_table.write(root)
    assert {r["metric"] for r in note} == {"est"}
    assert "gone" not in (root / "paper" / "results_table.md").read_text(encoding="utf-8")
    plan = {"metrics": [{"id": "est", "kind": "mean"}, {"id": "gone", "kind": "mean"}]}
    built = results_by_setting.write(root, plan)
    assert built is not None and "gone" not in json.dumps(built.references, default=str)
    for rel in ("paper/results_by_setting.md", "paper/results_by_setting.csv", ".fi/results_table.md"):
        f = root / rel
        if f.is_file():
            assert "gone" not in f.read_text(encoding="utf-8"), rel
    # FI's own record is unchanged
    assert "gone" in (root / "raw" / "trials.json").read_text(encoding="utf-8")


def test_without_a_left_out_record_the_same_quantity_is_tabulated(tmp_path: Path) -> None:
    root = _left_out_quest(tmp_path)
    (root / "needs" / "RESULT_CHECK.json").unlink()
    assert {r["metric"] for r in trial_table.rows(root)} == {"est", "gone"}
