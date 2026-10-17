"""FI's tables of every setting go into the paper through ``{{results_table}}`` (core/paper_facts.py), filled from
``.fi/results_table.md`` (core/results_by_setting.py ``paper_block``), and the plan prompt asks for a metric's
reference value (core/reference_values.py ``PLAN_FIELD`` / ``PLAN_RULE``). No model is called."""

from __future__ import annotations

import json
from pathlib import Path

from core import paper_facts as pf
from core import reference_values as rv
from core import results_by_setting as rbs
from core import trial_table
from tests import reference_values_plans as P

PAPER = ("# A title\n\n## Methods\n\nWe ran it.\n\n## Results\n\nThe estimates agree.\n\n"
         "## Discussion\n\nIt worked.\n\n## Limitations\n\nFew.\n")


def _built(name: str, protocol: dict) -> rbs.Built:
    from core import grid_keys

    return rbs.build(P.load(name, "trial_summaries.json")["cells"], protocol,
                     grid_keys.code_files(P.FIXTURES / name / "code"), P.load(name, "judged.json"))


def _quest(tmp_path: Path) -> Path:
    cells = [{"cell": {"m": m, "x": x}, "key": f"m={m},x={x}", "planned": 3, "ok": 3, "failed": 0,
              "metrics": {"hit": {"values": [1.0, 0.0, 1.0] if m == 1 else [0.0, 0.0, 1.0]}}}
             for m in (1, 2) for x in (5, 6)]
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "simulate.py").write_text(
        "def run_trial(cell, i, seed):\n    return {'hit': float(cell['m'] == 1)}\n", encoding="utf-8")
    protocol = {"grid": {"m": [1, 2], "x": [5, 6]}, "thresholds": {"p": 0.5},
                "metrics": [{"id": "hit", "estimand": "P(hit)", "kind": "proportion", "unit": "a run",
                             "reference": {"formula": "p"}}]}
    assert rbs.write(tmp_path, protocol) is not None
    return tmp_path


def test_the_plan_prompt_asks_for_a_reference_value() -> None:
    from core.engine import _PLAN_DIRECTIVE

    assert rv.PLAN_FIELD in _PLAN_DIRECTIVE and rv.PLAN_RULE in _PLAN_DIRECTIVE
    assert "@@REFERENCE" not in _PLAN_DIRECTIVE


def test_the_paper_block_is_every_row_then_a_slice_then_a_sentence() -> None:
    built = _built("kin_s1", P.kin_protocol())
    full = rbs.paper_block(built, cap=200)
    assert full.count("| 2000 |") == 100 and "more rows than fit" not in full
    cut = rbs.paper_block(built)  # 100 rows > PAPER_MAX_ROWS: the slice
    assert "more rows than fit here" in cut and cut.count("| 2000 |") == 4 * 9
    assert "paper/results_by_setting.csv" in cut
    none = rbs.paper_block(built, cap=10)
    assert "Results for all 25 settings are in the supplementary table `paper/results_by_setting.csv`." in none
    assert "|" not in none
    assert rbs.paper_block(None) == ""


def test_write_saves_the_block_and_a_run_without_a_table_removes_it(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    block = root / rbs.PAPER_BLOCK
    assert block.is_file() and "**Table (hit).**" in block.read_text(encoding="utf-8")
    (root / "raw" / "trials.json").unlink()
    assert rbs.write(root, {"grid": {"m": [1, 2]}}) is None
    assert not block.exists()


def test_the_table_goes_where_the_writer_put_it_and_round_trips(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    md = PAPER.replace("The estimates agree.\n", "The estimates agree.\n\n{{results_table}}\n")
    out = pf.render(pf.place_blocks(md, root), root).text
    assert out.count("<!-- fi:fact-results_table -->\n") == 1 and out.count("**Table (hit).**") == 1
    assert out.index("fi:fact-results_table") < out.index("## Discussion")
    assert "{{results_table}}" not in out and not pf.unfilled_in(out)
    # The passage editor gets the placeholder back; a second render gives the same paper.
    assert "{{results_table}}" in pf.to_placeholders(out)
    assert pf.render(out, root).text == out
    # The checks of the writer's own text do not see it; a reader's copy keeps the table without the markers.
    assert "Table (hit)" not in pf.mask(out) and "Table (hit)" not in pf.without_blocks(out)
    assert "The estimates agree." in pf.without_blocks(out)
    plain = pf.without_marks(out)
    assert "**Table (hit).**" in plain and "fi:fact" not in plain
    assert not [h for h in pf.check(out, root) if "results_table" in h]


def test_fi_places_the_table_before_the_discussion_when_the_writer_did_not(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    placed = pf.place_blocks(PAPER, root)
    assert placed.count("{{results_table}}") == 1
    assert placed.index("The estimates agree.") < placed.index("{{results_table}}") < placed.index("## Discussion")
    assert pf.place_blocks(placed, root) == placed  # once
    assert pf.place_blocks(PAPER, tmp_path / "no_quest") == PAPER  # no table, nothing placed


def test_the_writer_is_told_about_the_table_only_when_there_is_one(tmp_path: Path) -> None:
    root = _quest(tmp_path)
    assert "{{results_table}}" in pf.writer_section(root)
    assert "{{results_table}}" not in pf.writer_section(tmp_path / "no_quest")


def test_a_placed_table_with_no_record_reads_not_recorded(tmp_path: Path) -> None:
    out = pf.render("## Results\n\n{{results_table}}\n", tmp_path)
    assert out.unfilled == ("results_table",) and pf.NOT_RECORDED in out.text


def test_the_trial_table_cap_and_the_paper_cap_are_named_once() -> None:
    assert rbs.PAPER_MAX_ROWS < trial_table.WRITER_MAX_ROWS
    assert pf._BLOCK_FILES["results_table"] == rbs.PAPER_BLOCK


def test_what_fi_cannot_follow_in_the_code_is_a_log_line_never_paper_text(tmp_path: Path) -> None:
    """The s3 papers said "Every grid axis is treated as read: c is used in a way FI cannot follow (line 66)." in their
    results: FI's reading of the code is a diagnostic (run.log), not text for the reader or the writer."""
    root = _quest(tmp_path)
    (root / "code" / "simulate.py").write_text(
        "import json\ndef run_trial(cell, i, seed):\n    return {'hit': float(len(json.dumps(cell)) > 0)}\n",
        encoding="utf-8")
    protocol = {"grid": {"m": [1, 2], "x": [5, 6]}, "thresholds": {"p": 0.5},
                "metrics": [{"id": "hit", "estimand": "P(hit)", "kind": "proportion", "unit": "a run",
                             "reference": {"formula": "p"}}]}
    built = rbs.write(root, protocol)
    assert built is not None and any("treated as read" in d for d in built.diagnostics)
    for text in (rbs.paper_block(built), rbs.writer_note([], built), "\n".join(built.notes),
                 (root / "paper" / "results_by_setting.md").read_text(encoding="utf-8")):
        assert "treated as read" not in text and "cannot follow" not in text
