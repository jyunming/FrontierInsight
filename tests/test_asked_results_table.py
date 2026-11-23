"""The one table of the results the topic asks for (core/asked_results.py): FI writes it from its own records and puts it in
the paper's results after every write round, so a rewrite of the whole paper can never drop a result the topic asked for.
Neutral made-up study: a decay rate measured per trial at three doses, and a settling limit the analysis works out."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

from core import goal_coverage as gc
from core import paper_facts as pf
from core import paper_values as pv
from tests.test_analysis_answers import PROTOCOL, REQ, _cells

FORBIDDEN = ["FI", "record", "records", "recorded", "unconfirmed", "not computed", "QUALITY.md", "check", "checked", "obtained",
             "number removed", "computed value", "frozen", "ledger"]

DRAFT = (
    "# A study of decay\n\n## Abstract\n\nWe measured decay.\n\n## Methods\n\nThree doses were run.\n\n"
    "## Results\n\nThe decay rate at dose 2 was {{v:decay_rate @ dose=2}}.\n\nIt did not change much.\n\n"
    "## Discussion\n\nThe decay is small.\n\n## References\n\n1. A source.\n"
)
REWRITTEN = (
    "# A study of decay\n\n## Abstract\n\nWe measured decay.\n\n## Methods\n\nThree doses were run.\n\n"
    "## Results\n\nThe decay was small at every dose.\n\n## Discussion\n\nThe decay is small.\n\n## References\n\n1. A source.\n"
)


def _quest(tmp_path: Path, answers: dict | None = None, req: dict | None = REQ) -> Path:
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": _cells()}), encoding="utf-8")
    (tmp_path / ".fi").mkdir(exist_ok=True)
    if req is not None:
        (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps(req), encoding="utf-8")
    if answers is not None:
        pv.write_answers(tmp_path, answers, PROTOCOL)
    return tmp_path


def _engine(root: Path):
    from core import engine as E

    me = SimpleNamespace(quest_root=root, fi_dir=root / ".fi", _log=logging.getLogger("t"),
                         config=SimpleNamespace(topic=""))
    me._protocol_block = lambda st: PROTOCOL
    me._paper_records = lambda st: pv.records_of(root, PROTOCOL)
    for name in ("_place_blocks", "_write_asked_results", "_write_quality_pointer", "_render_paper_facts", "_with_fi_blocks"):
        if hasattr(E.Engine, name):  # an older tree has no table: the tests then fail on what the paper lacks
            setattr(me, name, getattr(E.Engine, name).__get__(me))
    if not hasattr(me, "_write_asked_results"):
        me._write_asked_results = lambda st: None
    if not hasattr(me, "_write_quality_pointer"):
        me._write_quality_pointer = lambda: None
    return me


def _table(marked: str) -> str:
    return "".join(re.findall(r"<!-- fi:fact-asked_results -->(.*?)<!-- /fi:fact-asked_results -->", marked, re.S))


def _unplaced(root: Path, marked: str) -> list[dict]:
    rec = pv.records_of(root, PROTOCOL)
    return gc.unplaced_answers(REQ, PROTOCOL, rec.fillable(), pv.placed_metrics(marked), sources=rec.mean_sources())


def test_the_table_is_in_the_results_once_with_a_row_for_each_asked_quantity_FI_holds(tmp_path):
    root = _quest(tmp_path, {"settling_limit": 12.3456})
    marked = _engine(root)._with_fi_blocks(REWRITTEN, {})
    assert marked.count("<!-- fi:fact-asked_results -->") == 1
    text = pf.without_marks(marked)
    rows = [ln for ln in text.splitlines() if ln.startswith("| ") and "Quantity" not in ln]
    assert len(rows) == 2 and "Decay rate" in rows[0] and "Limit of the settling time (hours)" in rows[1]
    assert "12.35" in rows[1]  # the value FI's key fills, not a typed one
    assert "**Table 1.** The quantities the study set out to determine." in text
    # at the end of the Results, before the Discussion
    assert text.index("The decay was small at every dose.") < text.index("Table 1.") < text.index("## Discussion")


def test_the_table_is_filled_through_the_papers_value_keys_so_the_goal_check_counts_it_placed(tmp_path):
    root = _quest(tmp_path, {"settling_limit": 12.3456})
    marked = _engine(root)._with_fi_blocks(REWRITTEN, {})  # the writer placed neither result
    assert pv.placed_metrics(marked) == {"decay_rate", "settling_limit"}
    assert _unplaced(root, marked) == []
    assert _unplaced(root, REWRITTEN) != []  # without the table both are missing


def test_a_rewrite_that_drops_the_table_gets_it_back_and_a_second_round_does_not_repeat_it(tmp_path):
    root = _quest(tmp_path, {"settling_limit": 12.3456})
    eng = _engine(root)
    first = eng._with_fi_blocks(DRAFT, {})
    assert first.count("<!-- fi:fact-asked_results -->") == 1
    again = eng._with_fi_blocks(first, {})
    assert again.count("<!-- fi:fact-asked_results -->") == 1 and again == first
    rewritten = eng._with_fi_blocks(REWRITTEN, {})  # a whole rewrite wrote none of it
    assert rewritten.count("<!-- fi:fact-asked_results -->") == 1
    assert _table(rewritten) == _table(first)
    third = eng._with_fi_blocks(pf.to_placeholders(first), {})  # the writer is given the placeholder back and keeps it
    assert third.count("<!-- fi:fact-asked_results -->") == 1 and "{{asked_results}}" not in third


def test_a_quantity_FI_holds_no_value_for_is_no_row_and_no_value_at_all_is_no_table(tmp_path):
    root = _quest(tmp_path)  # the analysis recorded no settling limit
    eng = _engine(root)
    text = pf.without_marks(eng._with_fi_blocks(REWRITTEN, {}))
    assert "Decay rate" in text and "settling" not in text.lower()
    root2 = _quest(tmp_path / "other", req={"quantities": []})
    out = _engine(root2)._with_fi_blocks(REWRITTEN, {})
    assert "asked_results" not in out and not (root2 / '.fi' / 'asked_results.md').exists()
    stale = tmp_path / ".fi" / "asked_results.md"
    assert stale.is_file()
    eng._write_asked_results({})
    assert stale.is_file()  # still asked and held
    (tmp_path / ".fi" / "topic_requirements.json").write_text(json.dumps({"quantities": []}), encoding="utf-8")
    eng._write_asked_results({})
    assert not stale.exists()  # nothing asked any more: the table goes, and with it any copy a draft still carries
    assert "Table 1" not in pf.without_marks(eng._with_fi_blocks(pf.to_placeholders(
        "# T\n\n## Results\n\nx\n\n{{asked_results}}\n\n## Discussion\n\ny\n"), {}))


def test_the_table_says_nothing_about_how_FI_got_the_values(tmp_path):
    root = _quest(tmp_path, {"settling_limit": 12.3456})
    text = _table(_engine(root)._with_fi_blocks(REWRITTEN, {}))
    plain = pf.without_marks(text)
    for word in FORBIDDEN:
        flags = 0 if word == "FI" else re.I
        assert not re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", plain, flags), (word, plain)
    assert not re.search(r"\b[a-z]+(?:_[a-z0-9]+)+\b", plain)  # no identifier of the code
    assert "decay_rate" not in plain and "settling_limit" not in plain


def test_the_table_goes_at_the_end_of_the_section_with_most_of_the_results(tmp_path):
    root = _quest(tmp_path, {"settling_limit": 12.3456})
    draft = ("# T\n\n## Abstract\n\nDecay {{v:decay_rate @ dose=1}}.\n\n## Results\n\nA {{v:decay_rate @ dose=2}} and "
             "{{v:decay_rate @ dose=3}}.\n\n### Detail\n\nMore.\n\n## Discussion\n\nD.\n")
    text = pf.without_marks(_engine(root)._with_fi_blocks(draft, {}))
    assert text.index("More.") < text.index("**Table 1.**") < text.index("## Discussion")
    # a draft with no key anywhere: before the discussion; with none of those, at the end
    bare = pf.without_marks(_engine(root)._with_fi_blocks("# T\n\n## Intro\n\nx\n\n## Discussion\n\nD.\n", {}))
    assert bare.index("**Table 1.**") < bare.index("## Discussion")
    assert pf.without_marks(_engine(root)._with_fi_blocks("# T\n\nx\n", {})).rstrip().endswith("|")


def test_the_table_number_fits_the_numbered_tables_the_paper_has(tmp_path):
    root = _quest(tmp_path, {"settling_limit": 12.3456})
    draft = ("# T\n\n## Methods\n\n**Table 1.** Doses.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n## Results\n\n"
             "{{v:decay_rate @ dose=2}}\n\n**Table 3.** Later.\n\n## Discussion\n\nD.\n")
    text = pf.without_marks(_engine(root)._with_fi_blocks(draft, {}))
    assert "**Table 4.** The quantities the study set out to determine." in text  # 2 is before Table 3: take the next free
    clear = ("# T\n\n## Methods\n\n**Table 1.** Doses.\n\n## Results\n\n{{v:decay_rate @ dose=2}}\n\n## Discussion\n\nD.\n")
    assert "**Table 2.** The quantities" in pf.without_marks(_engine(root)._with_fi_blocks(clear, {}))


def test_the_writer_is_told_the_table_is_added_and_the_list_of_asked_results_stays(tmp_path):
    root = _quest(tmp_path, {"settling_limit": 12.3456})
    rec = pv.records_of(root, PROTOCOL)
    asked = gc.asked_answers(REQ, PROTOCOL, rec.fillable())
    block = pv.asked_lines(rec, asked)
    assert "FI adds a table of these quantities to the results section" in block
    assert "need not repeat each number" in block
    assert "`decay_rate`" in block and "`settling_limit`" in block  # the list itself is unchanged


def test_a_value_with_an_interval_shows_it_in_the_table(tmp_path):
    cells = [{"cell": {"dose": d}, "key": f"dose={d}", "planned": 5, "ok": 5, "failed": 0,
              "metrics": {"decay_rate": {"values": [0.1 * d + e for e in (0.01, -0.01, 0.02, -0.02, 0.0)],
                                         "trials": [0, 1, 2, 3, 4], "count": 5, "total": 0.5 * d, "non_finite": 0}}}
             for d in (1, 2, 3)]
    root = _quest(tmp_path)
    (root / "raw" / "trials.json").write_text(json.dumps({"cells": cells}), encoding="utf-8")
    marked = _engine(root)._with_fi_blocks(REWRITTEN, {})
    row = next(ln for ln in pf.without_marks(marked).splitlines() if ln.startswith("| Decay rate"))
    assert re.search(r"\(95% CI [\d.]+ to [\d.]+; 5 trials\)", row), row  # plain, no word about the plan or the estimator
    assert pv.placed_metrics(marked) == {"decay_rate"} and _unplaced(root, marked) == []


def _keep_break(root: Path, quantity: str = "decay_rate") -> None:
    from core import result_checks as rc

    (root / "needs").mkdir(exist_ok=True)
    (root / "needs" / rc.RECORD).write_text(json.dumps({rc.KEPT_BREAKS: [{"quantity": quantity, "note": "n"}]}), encoding="utf-8")


def test_a_value_beyond_a_bound_the_plan_expected_is_printed_as_it_is_and_the_paper_points_to_the_quality_report(tmp_path):
    """No mark follows a value in the reader's text, so the table is as plain as for any other quantity, and the check that
    puts a changed value back agrees with it; the paper ends its body with one sentence that says where the case is listed."""
    root = _quest(tmp_path, {"settling_limit": 12.3456})
    _keep_break(root)
    eng = _engine(root)
    marked = eng._with_fi_blocks(REWRITTEN, {})
    plain = pf.without_marks(marked)
    assert "unconfirmed" not in plain
    assert plain.count("Values that fell outside the range the study expected are listed in the accompanying quality report.") == 1
    assert "statements" not in plain.lower().split("## references")[0].split("## discussion")[1]
    assert plain.index("accompanying quality report") < plain.index("## References")
    assert pv.verify_marks(marked, pv.records_of(root, PROTOCOL)).repaired == []
    # a later round (the whole paper written again) has it once, never twice
    again = eng._with_fi_blocks(marked, {})
    assert pf.without_marks(again).count("accompanying quality report") == 1
    # and the last case gone: no sentence left
    (root / "needs" / "RESULT_CHECK.json").unlink()
    assert "accompanying quality report" not in pf.without_marks(eng._with_fi_blocks(again, {}))


def test_without_a_value_outside_its_range_the_paper_has_no_closing_sentence(tmp_path):
    root = _quest(tmp_path, {"settling_limit": 12.3456})
    assert "quality report" not in pf.without_marks(_engine(root)._with_fi_blocks(REWRITTEN, {}))




def test_the_table_is_never_put_in_the_abstract(tmp_path):
    """A draft with one key in its abstract and none in its Results: the table goes to the body, at the end of the results."""
    root = _quest(tmp_path, {"settling_limit": 12.3456})
    draft = ("# T\n\n## Abstract\n\nA {{v:decay_rate @ dose=1}}.\n\n## Methods\n\nM.\n\n## Results\n\nNo number here.\n\n"
             "## Discussion\n\nD.\n")
    text = pf.without_marks(_engine(root)._with_fi_blocks(draft, {}))
    assert text.index("## Results") < text.index("**Table 1.**") < text.index("## Discussion")
    assert "**Table 1.**" not in text.split("## Methods")[0]


def test_a_section_headed_as_the_results_is_preferred_to_one_with_more_keys(tmp_path):
    root = _quest(tmp_path, {"settling_limit": 12.3456})
    draft = ("# T\n\n## Methods\n\n{{v:decay_rate @ dose=1}} {{v:decay_rate @ dose=2}} {{v:decay_rate @ dose=3}}.\n\n"
             "## Results\n\n{{v:decay_rate @ dose=2}}.\n\n## Discussion\n\nD.\n")
    text = pf.without_marks(_engine(root)._with_fi_blocks(draft, {}))
    assert text.index("## Results") < text.index("**Table 1.**") < text.index("## Discussion")


ASKED_CASES = {
    "table cell": "# T\n\n## Methods\n\nM.\n\n## Results\n\n| a | b |\n|---|---|\n| x | {{asked_results}} |\n\nR.\n\n## Discussion\n\nD.\n\n## References\n\n1. A.\n",
    "reference list": "# T\n\n## Methods\n\nM.\n\n## Results\n\nR.\n\n## Discussion\n\nD.\n\n## References\n\n1. A. {{asked_results}}\n",
    "twice": "# T\n\n## Methods\n\nM.\n\n## Results\n\nR.\n\n{{asked_results}}\n\n## Discussion\n\nD.\n\n{{asked_results}}\n\n## References\n\n1. A.\n",
}


def test_a_placeholder_of_the_table_anywhere_in_the_draft_gives_one_table_at_the_end_of_the_results(tmp_path):
    root = _quest(tmp_path, {"settling_limit": 12.3456})
    for case, draft in ASKED_CASES.items():
        text = pf.without_marks(_engine(root)._with_fi_blocks(draft, {}))
        assert text.count("**Table 1.**") == 1 and "{{asked_results}}" not in text, case
        assert text.index("R.") < text.index("**Table 1.**") < text.index("## Discussion"), case
        assert "A. |" not in text.split("## References")[1], case


def test_the_table_goes_to_the_results_heading_even_when_methods_holds_the_key(tmp_path):
    """The reviewer's input: a key in Methods, none in Results. The table closes the Results, never the Methods."""
    root = _quest(tmp_path, {"settling_limit": 12.3456})
    draft = "# T\n\n## Methods\n\nM {{v:decay_rate @ dose=1}}.\n\n## Results\n\nNo number here.\n\n## Discussion\n\nD.\n"
    text = pf.without_marks(_engine(root)._with_fi_blocks(draft, {}))
    assert text.index("No number here.") < text.index("**Table 1.**") < text.index("## Discussion")
    for heading in ("Introduction", "Background", "Experimental set-up", "Discussion", "Conclusions"):
        other = f"# T\n\n## {heading}\n\nX {{{{v:decay_rate @ dose=1}}}}.\n\n## Findings\n\nF.\n\n## References\n\n1. A.\n"
        got = pf.without_marks(_engine(root)._with_fi_blocks(other, {}))
        assert got.index("F.") < got.index("**Table 1.**") < got.index("## References"), heading


def test_without_a_results_heading_the_section_with_most_keys_among_the_eligible_ones_is_chosen(tmp_path):
    root = _quest(tmp_path, {"settling_limit": 12.3456})
    draft = ("# T\n\n## Methods\n\n{{v:decay_rate @ dose=1}} {{v:decay_rate @ dose=2}}.\n\n## Analysis\n\nA {{v:decay_rate @ dose=3}}.\n\n"
             "## Discussion\n\nD.\n")
    text = pf.without_marks(_engine(root)._with_fi_blocks(draft, {}))
    assert text.index("## Analysis") < text.index("**Table 1.**") < text.index("## Discussion")
