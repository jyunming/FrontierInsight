"""A count FI holds from its own records is not taken out of the paper as a typed number."""
from __future__ import annotations

import json
from decimal import Decimal

from core import paper_facts, typed_numbers as tn

PLAN = tn.plan_numbers({"constants": {"z": 1.96}}, "Estimate the settle time of a controller.")


def _quest(tmp_path, settings=12, trials=5, failed=0):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    rows = []
    for c in range(settings):
        rows.append({"event": "planned", "cell": f"c{c}", "trials": trials})
        for t in range(trials):
            rows.append({"event": "trial", "cell": f"c{c}", "trial": t, "status": "failed" if t < failed else "ok",
                         "seed": t})
    (raw / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path


def test_run_counts_come_from_the_ledger(tmp_path):
    counts = paper_facts.run_counts(_quest(tmp_path))
    assert counts["settings"] == frozenset({12}) and counts["trials_per_setting"] == frozenset({5})


def test_a_typed_count_of_settings_equal_to_fi_s_is_kept(tmp_path):
    """Old code replaced "Each of 12 settings" by "[number removed]" although 12 is FI's own count of settings."""
    counts = paper_facts.run_counts(_quest(tmp_path))
    text = "Each of 12 settings was run with 5 independent trials."
    new, taken = tn.remove(text, PLAN, counts)
    assert new == text and taken == []
    assert tn.hits(text, PLAN, counts=counts) == []
    old, old_taken = tn.remove(text, PLAN)
    assert tn.REMOVED in old and len(old_taken) == 2


def test_a_number_equal_to_a_count_but_counting_something_else_is_removed(tmp_path):
    counts = paper_facts.run_counts(_quest(tmp_path))
    new, taken = tn.remove("We asked 12 volunteers about the controller.", PLAN, counts)
    assert tn.REMOVED in new and [t[0] for t in taken] == ["12"]


def test_a_typed_result_not_in_fi_s_records_is_still_removed(tmp_path):
    """Known positive: the rule is not loosened for result numbers."""
    counts = paper_facts.run_counts(_quest(tmp_path))
    text = "Each of 12 settings reached a mean settle time of 3.42 and a hit rate of 87 settings."
    new, taken = tn.remove(text, PLAN, counts)
    assert "Each of 12 settings" in new
    assert sorted(t[0] for t in taken) == ["3.42", "87"]


def test_no_ledger_means_no_counts(tmp_path):
    assert paper_facts.run_counts(tmp_path) == {}
    assert paper_facts.run_counts(None) == {}
    assert tn.REMOVED in tn.remove("Each of 12 settings ran.", PLAN, {})[0]
    assert Decimal("12") not in PLAN
