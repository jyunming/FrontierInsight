"""``{{interval_level}}``: the confidence level of every interval FI computes, filled by FI (core/paper_facts.py), so an
honest method sentence ("a 95% interval") types no 95 of its own and a typed 95% is traced like any number (core/typed_numbers.py).
No model is called; the quest is a made-up bridge sensor study."""

from __future__ import annotations

from pathlib import Path

from core import paper_facts as pf
from core import paper_values
from core import typed_numbers as tn

TOPIC = "Compare two sensor filters on a simulated bridge."
METHOD = "# T\n\n## Methods\nEach mean carries a {{interval_level}}% interval.\n"


def test_the_fact_is_filled_with_the_level_fi_computes(tmp_path: Path) -> None:
    assert "interval_level" in pf.FACT_NAMES
    out = pf.render(METHOD, tmp_path, topic=TOPIC)
    assert not out.unfilled
    assert "a <!-- fi:fact-interval_level -->95<!-- /fi:fact-interval_level -->% interval" in out.text
    assert pf.facts(tmp_path)["interval_level"].text == "95"
    # it follows the one level FI's intervals use
    assert pf.interval_level_percent() == str(int(float(paper_values.INTERVAL_LEVEL) * 100))


def test_the_filled_level_is_not_a_number_the_writer_typed(tmp_path: Path) -> None:
    filled = pf.render(METHOD, tmp_path, topic=TOPIC).text
    assert pf.typed_hits(filled, {}, TOPIC) == []
    assert pf.untraced_typed(filled, {}, TOPIC) == []
    # no unknown-name hit either: the name is one of FI's
    assert pf.placement_hits(filled, tmp_path, topic=TOPIC) == []


def test_inside_math_the_plain_value_is_filled(tmp_path: Path) -> None:
    out = pf.render("# T\n\n## Methods\nThe level is $1-\alpha = 0.{{interval_level}}$.\n", tmp_path, topic=TOPIC).text
    assert "0.95" in out and "fi:fact" not in out


def test_a_typed_95_percent_gets_a_hit_that_names_the_key() -> None:
    typed = "# T\n\n## Methods\nEach mean carries a 95% interval.\n"
    hits = pf.typed_hits(typed, {}, TOPIC)
    assert hits and "{{interval_level}}" in hits[0]
    assert pf.untraced_typed(typed, {}, TOPIC)
    # a known positive: the same sentence with the key has none (above), and a plan that states 0.95 still passes
    assert pf.typed_hits(typed, {"thresholds": {"nominal": 0.95}}, TOPIC) == []
    assert tn.hits(typed, tn.plan_numbers({}, ""))
