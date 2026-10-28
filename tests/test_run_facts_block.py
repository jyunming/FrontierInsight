"""The run's facts reach the paper as one paragraph FI writes and places itself, ``{{run_facts}}`` (core/paper_facts.py):
composed from FI's records, put at the end of the Methods, never asked of the writer. No model is called; the quests are
made up here (a neutral study of a made-up bridge sensor)."""

from __future__ import annotations

import json
from pathlib import Path

from core import paper_facts as pf

TOPIC = "Compare two sensor filters on a simulated bridge. Keep the run under 60 seconds."


def _quest(tmp_path: Path, trials_run: dict | None, *, settings: int = 0, per: int = 0, failed: int = 0) -> Path:
    (tmp_path / ".fi").mkdir(parents=True, exist_ok=True)
    if trials_run is not None:
        (tmp_path / ".fi" / "run_facts.json").write_text(json.dumps({"trials_run": trials_run}), encoding="utf-8")
    if settings:
        (tmp_path / "raw").mkdir(exist_ok=True)
        rows = []
        for s in range(settings):
            rows.append({"event": "planned", "cell": f"gain={s}", "trials": per})
            rows += [{"event": "trial", "cell": f"gain={s}", "trial": t, "seed": 100 + t,
                      "status": "failed" if failed and s == 0 and t < failed else "ok", "duration_s": 0.1}
                     for t in range(per)]
        (tmp_path / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path


AT_ONCE = {"wall_s": 611.11, "trials": 680, "cells": 680, "trial_sum_s": 31.27, "at_once": 16, "settings_s": 9200.32}
PAPER = "# A title\n\n## Abstract\nShort.\n\n## Methods\nWe ran it.\n\n## Results\nIt agrees.\n\n## Limitations\nFew.\n"


# ---- the paragraph ---------------------------------------------------------------------------------------------------


def test_settings_run_at_the_same_time_read_as_whole_sentences(tmp_path: Path) -> None:
    root = _quest(tmp_path, AT_ONCE)
    text = pf.run_facts_text(root, topic=TOPIC)
    assert text == (
        "**Run.** FI ran the 680 settings 16 at a time; the run took 611.1 s (about 10.2 minutes). The trials' own "
        "computation adds up to 31.3 s. FI did not time the delivered code, which runs the settings one after another: "
        "by FI's measurements such a run takes at least 31.3 s and at most 9,200.3 s (each setting's time while 16 ran at "
        "once, added up). Whether it keeps to the topic's limit of 60 s is not shown.")
    assert "((" not in text and ") (" not in text, "no nested or stacked brackets"


def test_the_other_verdicts_of_the_delivered_code(tmp_path: Path) -> None:
    over = _quest(tmp_path / "a", {**AT_ONCE, "trial_sum_s": 75.0, "settings_s": 400.0})
    assert pf.run_facts_text(over, topic=TOPIC).endswith("It goes over the topic's limit of 60 s.")
    within = _quest(tmp_path / "b", {**AT_ONCE, "trial_sum_s": 5.0, "settings_s": 50.0})
    assert pf.run_facts_text(within, topic=TOPIC).endswith("It keeps to the topic's limit of 60 s.")


def test_settings_run_one_after_another_and_the_counts(tmp_path: Path) -> None:
    root = _quest(tmp_path, {"wall_s": 42.0, "trials": 12, "cells": 4, "trial_sum_s": 41.0, "at_once": 1},
                  settings=4, per=3, failed=2)
    text = pf.run_facts_text(root, topic=TOPIC)
    assert text.startswith("**Run.** The study ran 4 settings, 3 trials in each. 2 trials failed. ")
    assert "The run took 42 s." in text
    assert text.endswith("The run keeps to the topic's limit of 60 s.")
    clean = _quest(tmp_path / "c", {"wall_s": 42.0, "trials": 12, "cells": 4, "trial_sum_s": 41.0, "at_once": 1},
                   settings=4, per=3)
    assert "No trial failed." in pf.run_facts_text(clean, topic=TOPIC)


def test_no_record_gives_no_block_and_a_limit_alone_gives_one_sentence(tmp_path: Path) -> None:
    root = _quest(tmp_path, None)
    assert pf.run_facts_text(root, topic="Compare two filters.") == ""
    assert pf.block_text(root, "run_facts", topic="Compare two filters.") == ""
    assert pf.place_blocks(PAPER, root, topic="Compare two filters.") == PAPER, "nothing to place"
    only_limit = pf.run_facts_text(root, topic=TOPIC)
    assert only_limit == "**Run.** Whether the run keeps to the topic's limit of 60 s is not shown."


# ---- where FI puts it ------------------------------------------------------------------------------------------------


def _at(text: str, root: Path) -> str:
    return pf.place_blocks(text, root, topic=TOPIC)


def test_fi_puts_the_block_at_the_end_of_the_methods(tmp_path: Path) -> None:
    root = _quest(tmp_path, AT_ONCE)
    out = _at(PAPER, root)
    assert out.index("We ran it.") < out.index("{{run_facts}}") < out.index("## Results")
    assert "We ran it.\n\n{{run_facts}}\n\n## Results" in out


def test_without_a_methods_heading_it_goes_before_the_results_or_after_the_abstract(tmp_path: Path) -> None:
    root = _quest(tmp_path, AT_ONCE)
    no_methods = "# T\n\n## Abstract\nShort.\n\n## Results\nIt agrees.\n"
    out = _at(no_methods, root)
    assert out.index("Short.") < out.index("{{run_facts}}") < out.index("## Results")
    only_abstract = "# T\n\n## Abstract\nShort.\n\n## Introduction\nPlain text.\n"
    out = _at(only_abstract, root)
    assert out.index("Short.") < out.index("{{run_facts}}") < out.index("Plain text.")
    bare = "# T\n\nJust text.\n"
    assert _at(bare, root).index("{{run_facts}}") < _at(bare, root).index("Just text.")


def test_the_writers_own_place_is_kept_and_a_second_pass_adds_nothing(tmp_path: Path) -> None:
    root = _quest(tmp_path, AT_ONCE)
    own = PAPER.replace("It agrees.", "It agrees.\n\n{{run_facts}}")
    assert _at(own, root) == own, "the writer put it in the Results; FI leaves it there"
    placed = _at(PAPER, root)
    filled = pf.render(placed, root, topic=TOPIC).text
    assert filled.count("**Run.**") == 1 and "{{" not in filled
    again = pf.render(_at(filled, root), root, topic=TOPIC).text
    assert again == filled, "FI's own filled block comes back from a review and gets no second copy"
    assert again.count("**Run.**") == 1


def test_render_fills_the_block_between_markers_and_a_re_render_takes_the_records_as_they_are(tmp_path: Path) -> None:
    root = _quest(tmp_path, AT_ONCE)
    filled = pf.render(_at(PAPER, root), root, topic=TOPIC).text
    assert "<!-- fi:fact-run_facts -->\n**Run.** FI ran the 680 settings" in filled
    assert "FI ran the 680" not in pf.without_blocks(filled)
    assert "{{run_facts}}" in pf.to_placeholders(filled)
    _quest(tmp_path, {**AT_ONCE, "wall_s": 700.0})
    assert "the run took 700 s" in pf.render(filled, root, topic=TOPIC).text


# ---- the checks -------------------------------------------------------------------------------------------------------


def test_a_missing_seed_or_run_time_placeholder_is_no_fault_of_the_paper(tmp_path: Path) -> None:
    root = _quest(tmp_path, AT_ONCE)
    bare = pf.render(PAPER, root, topic=TOPIC).text
    assert pf.placement_hits(bare, root, topic=TOPIC) == []
    assert pf.check(bare, root, topic=TOPIC) == []
    # an unknown name between double braces is still sent back
    assert pf.placement_hits(pf.render("# T\n\n## Methods\nIt used {{bogus}}.\n", root, topic=TOPIC).text, root, topic=TOPIC)


def test_the_typed_number_check_does_not_read_fis_block_as_numbers_the_writer_typed(tmp_path: Path) -> None:
    root = _quest(tmp_path, AT_ONCE)
    filled = pf.render(_at(PAPER, root), root, topic=TOPIC).text
    assert "611.1 s" in filled and "680" in filled
    assert pf.typed_hits(filled, {}, TOPIC) == []
    assert pf.untraced_typed(filled, {}, TOPIC) == []
    # a known positive: the same number typed by the writer outside the block is read
    typed = filled.replace("We ran it.", "We ran it in 611.1 s.")
    assert any("611.1" in h for h in pf.typed_hits(typed, {}, TOPIC)) or pf.untraced_typed(typed, {}, TOPIC)


def test_the_writer_is_told_about_the_paragraph_and_no_longer_about_required_placeholders(tmp_path: Path) -> None:
    root = _quest(tmp_path, AT_ONCE)
    text = pf.writer_section(root, topic=TOPIC)
    assert "Required" not in text and "{{seed}}" not in text and "{{run_time}}" not in text
    assert "Example (in the Methods)" not in text
    assert "the run facts in the Methods" in text and "FI ran the 680 settings 16 at a time" in text
    assert "Do not state these facts yourself" in text
    assert pf.writer_section(_quest(tmp_path / "none", None), topic="Compare two filters.") == ""


def test_the_old_placeholders_of_a_resumed_draft_are_still_filled(tmp_path: Path) -> None:
    root = _quest(tmp_path, AT_ONCE)
    out = pf.render("# T\n\n## Methods\nIt took {{run_time}}.\n", root, topic=TOPIC)
    assert not out.unfilled and "611.1 s" in out.text


def test_one_trial_in_each_setting_reads_singular(tmp_path: Path) -> None:
    """A deterministic study runs one trial per setting: the count reads "1 trial", not "1 trials"."""
    root = _quest(tmp_path, None, settings=3, per=1)
    text = pf.run_facts_text(root, topic=TOPIC)
    assert "The study ran 3 settings, 1 trial in each." in text
    assert "1 trials" not in text
