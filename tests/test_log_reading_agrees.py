"""A bare ``log(`` is ambiguous (base e or base 10) only when the two readings give different values: in a ratio of logs
of the same base the base cancels, and FI then works the formula out. Neutral numbers; no model is called."""

from __future__ import annotations

from core import oracle_triage

RATIO = "log(abs(3.7-1.2)/abs(1.9-0.6))/log(2)"


def test_a_ratio_of_logs_reads_the_same_in_either_base_and_is_not_ambiguous() -> None:
    assert not oracle_triage.ambiguous(RATIO)
    assert not oracle_triage.ambiguous("log(8)/log(2)")


def test_a_log_whose_base_changes_the_value_stays_ambiguous() -> None:
    assert oracle_triage.ambiguous("log(1000)")
    assert oracle_triage.ambiguous("log(8)/log(2) + log(3)")
    assert oracle_triage.ambiguous("log(x)/log(2)")  # not worked out: FI cannot show the readings agree


def test_a_trig_of_a_bare_number_stays_ambiguous_beside_a_log_ratio() -> None:
    assert oracle_triage.ambiguous("log(8)/log(2) * sin(30)")
