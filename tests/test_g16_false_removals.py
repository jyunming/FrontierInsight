"""FI states its own counts through placeholders; a count the writer types as a bare number is not exempt."""
from __future__ import annotations

import json

from core import paper_facts, typed_numbers as tn

PLAN = tn.plan_numbers({"constants": {"z": 1.96}}, "Estimate the settle time of a controller.")


def _quest(tmp_path, settings=12, trials=5):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    rows = []
    for c in range(settings):
        rows.append({"event": "planned", "cell": f"c{c}", "trials": trials})
        for t in range(trials):
            rows.append({"event": "trial", "cell": f"c{c}", "trial": t, "status": "ok", "seed": t})
    (raw / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path


def test_a_typed_count_that_equals_the_ledger_is_still_flagged():
    """A bare "12 settings" is an unverified claim even when 12 is the ledger's count (no noun-based exemption)."""
    text = "All 12 settings support the proposed mechanism."
    new, taken = tn.remove(text, PLAN)
    assert tn.REMOVED in new and [t[0] for t in taken] == ["12"]
    assert tn.hits(text, PLAN)


def test_the_placeholders_fill_to_the_ledger_values(tmp_path):
    """Known positive: FI states the counts itself, as it states the seed."""
    facts = paper_facts.facts(_quest(tmp_path))
    assert facts["settings"].text == "12" and facts["trials_per_setting"].text == "5"
    assert "trials_failed" in facts


def test_the_writer_is_told_to_use_the_count_placeholders(tmp_path):
    section = paper_facts.writer_section(_quest(tmp_path))
    assert "{{settings}}" in section and "{{trials_per_setting}}" in section
