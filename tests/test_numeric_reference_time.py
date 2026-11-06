"""The numerical computation of a known answer runs no longer than a run of the study may (a short study's check of one
number used to run for 300 s). Neutral; no model is called."""

from __future__ import annotations

from core import oracle_challenge


def test_the_reference_runs_no_longer_than_a_run_of_the_study() -> None:
    assert oracle_challenge.numeric_timeout(120) == 120
    assert oracle_challenge.numeric_timeout(3600) == oracle_challenge.NUMERIC_TIMEOUT_S
    assert oracle_challenge.numeric_timeout(2) == 2
    assert oracle_challenge.numeric_timeout(None) == oracle_challenge.NUMERIC_TIMEOUT_S


def test_the_engine_bounds_the_reference_by_the_runs_limit() -> None:
    import inspect

    from core.engine import Engine

    src = inspect.getsource(Engine._numeric_reference)
    assert "numeric_timeout(self.config.execution.timeout_s)" in src
