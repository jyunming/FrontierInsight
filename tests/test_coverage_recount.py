"""A coverage the analysis computes itself is recounted by FI from the trials (core/trial_checks.reported_coverage_mismatches,
``ResultChecksMixin._coverage_recount_findings``): a plan's `proportion` metric with a `known_value` is the coverage of
that interval, and an analysis that reports another number than FI's count of the same trials is sent back to be repaired
like a failed run, then left out. The trials are r6t's real record (tests/fixtures/replay/trial_checks/)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pytest

from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine, ExecutionResult

FIX = Path(__file__).parent / "fixtures" / "replay" / "trial_checks" / "r6t_kinetics.json.gz"
SPEC = {"id": "ea_coverage_95", "estimand": "coverage of the true activation energy", "kind": "proportion",
        "unit": "one trial", "known_value": {"true_value": "Ea_true_J_mol", "estimate": "weighted_ea",
                                             "standard_error": "weighted_ea_se", "level": 0.95}}


def _data() -> dict[str, Any]:
    with gzip.open(FIX, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Engine, dict[str, Any]]:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    eng = Engine(Config(
        topic="a neutral topic", title="recount", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, oracle_check="off"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60, split_analysis=True),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    eng._trial_mode = True
    data = _data()
    data["cells"] = [c for c in data["cells"] if c["metrics"]]
    (eng.quest_root / "raw").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "raw" / "trials.json").write_text(json.dumps({"schema": "fi.trials/v1", "thresholds":
                                                                    data["thresholds"], "cells": data["cells"]}),
                                                        encoding="utf-8")
    grid = {k: sorted({c["cell"][k] for c in data["cells"]}) for k in data["cells"][0]["cell"]}
    state = {"iteration": 0, "design": {"protocol": {"grid": grid, "thresholds": data["thresholds"], "metrics": [SPEC],
                                                     "runs_per_setting": 400}}}
    return eng, {"state": state, "cells": data["cells"]}


def _printed(cells: list[dict[str, Any]], z: float) -> ExecutionResult:
    def coverage(c: dict[str, Any]) -> float:
        e, s = c["metrics"]["weighted_ea"]["values"], c["metrics"]["weighted_ea_se"]["values"]
        return sum(abs(x - 50000.0) <= z * se for x, se in zip(e, s)) / len(e)

    printed = {c["key"]: {"ea_coverage_95": coverage(c)} for c in cells}
    return ExecutionResult(0, "RESULT_JSON: " + json.dumps(printed), "", 1.0, False)


@pytest.mark.asyncio
async def test_an_analysis_coverage_that_is_not_fis_count_is_sent_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eng, ctx = _engine(tmp_path, monkeypatch)
    state = ctx["state"]
    assert eng._protocol_block(state)["metrics"][0]["id"] == "ea_coverage_95"
    right, _ = await eng._results_match_run(state, _printed(ctx["cells"], NormalDist().inv_cdf(0.975)), None, True)
    assert right.returncode == 0
    wrong, script = await eng._results_match_run(state, _printed(ctx["cells"], NormalDist().inv_cdf(0.95)), None, True)
    assert wrong.returncode == 1 and script == "experiment.py"
    assert "FI counted, from the trials themselves" in wrong.stderr and "`ea_coverage_95`" in wrong.stderr
