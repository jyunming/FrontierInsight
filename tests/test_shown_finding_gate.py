"""The evidence gate stops a quest whose code holds a fault that changes a reported number, that a failing test showed, and
whose test still fails on the code that produced the results. A reader's word alone, a test that now passes, a test that
cannot be run, and a quest without a simulation never stop it. Real tiny test files and the real test runner; no model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from core import code_review
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

pytestmark = pytest.mark.usefixtures("this_python")

TYPED_IN = "def dose():\n    return 60.0 if True else 40.0\n"


#: What the code model answers when FI asks it once more for a test (``Engine._chat`` is replaced; no model is called).
REPLIES: list[str] = []
ASKED: list[str] = []


@pytest.fixture(autouse=True)
def _no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    REPLIES.clear()
    ASKED.clear()

    async def chat(self, prompt, *a, **kw):  # noqa: ANN001, ANN002, ANN003
        ASKED.append(str(prompt))
        return REPLIES.pop(0) if REPLIES else "no test"

    monkeypatch.setattr(Engine, "_chat", chat)


def _engine(tmp_path: Path) -> Engine:
    cfg = Config(topic="a dose to size", title="dose", provider=ProviderConfig(name="openai"),
                 engine=EngineConfig(max_iterations=1, review_loop=False),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
                 output=OutputConfig(output_dir=tmp_path / "outputs"))
    engine = Engine(cfg)
    (engine.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (engine.quest_root / "code" / "experiment.py").write_text(TYPED_IN, encoding="utf-8")
    return engine


CRASHES = "def dose():\n    return {}['missing']\n"


def test_a_fault_may_be_about_the_main_quantity_by_its_name_or_its_keys() -> None:
    result = {"total_yield": 3.2, "yield_ci": [3.0, 3.4], "runtime": 1.0}
    issue = {"id": "F-1", "quantity": "yield"}
    assert code_review.about_headline({}, issue, result, ["total_yield"])  # one name holds the other
    assert code_review.about_headline({}, {"id": "F-2", "quantity": "totalyield"}, result, ["total_yield"])
    assert not code_review.about_headline({}, {"id": "F-3", "quantity": "runtime"}, result, ["total_yield"])
    assert code_review.about_headline({}, {"id": "F-4", "headline": True}, result, [])
    # a short headline name (AUC) by its own name and by the keys it holds
    assert code_review.about_headline({}, {"id": "F-5", "quantity": "AUC"}, {"auc": 0.85, "n": 3}, ["auc"])
    assert code_review.about_headline({}, {"id": "F-6", "quantity": "auc"}, {"auc": 0.85}, ["AUC"])
    assert not code_review.about_headline({}, {"id": "F-7", "quantity": "n"}, {"auc": 0.85, "n": 3}, ["auc"])


# ---- a test that errors in its own code is asked for once more, as an assertion about values ------------------------------


# A fault shown in a declared result the run gave no number for under any name: a part not done, never a stop.
_PLAN = {"variables": {"dependent": ["yield"]},
         "protocol": {"metrics": [{"id": "yield", "estimand": "total"}, {"id": "per_batch_spread", "estimand": "spread"}]}}


def _derived(*ids: str) -> dict[str, Any]:
    """FI's record of the last run of its analysis (data/results/derived.json) listing ``ids`` as given no value."""
    return {"schema": "fi.derived/v1", "complete": not ids,
            "not_computed": [{"id": i, "status": "no_data", "why": "no trial gave a value for it"} for i in ids]}


def test_fi_s_record_of_the_run_decides_which_declared_results_have_no_value() -> None:
    derived = _derived("per_batch_spread", "not_a_plan_result")
    assert code_review.declared_names(_PLAN, _PLAN["protocol"], derived) == ["per_batch_spread"]
    assert code_review.declared_names(_PLAN, _PLAN["protocol"], {"not_computed": [{"id": "per_batch_spread"}]}) == []  # no schema


def test_an_earlier_runs_record_of_no_value_declares_nothing() -> None:
    """A record restored from an earlier run (the run now stopped before writing its own) says nothing about the code that
    ran: a shown fault keeps its stop, never becomes a part not done."""
    stale = {**_derived("per_batch_spread"), "from_earlier_run": True}
    assert code_review.declared_names(_PLAN, _PLAN["protocol"], stale) == []
    assert code_review.declared_names(_PLAN, _PLAN["protocol"], _derived("per_batch_spread")) == ["per_batch_spread"]
