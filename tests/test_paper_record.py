"""The paper's facts of the run come from FI's records through placeholders, and FI checks the paper against its records
(core/paper_facts.py, core/run_facts.py, core/stat_claims.py).

Replays of two real quests (``tests/fixtures/paper_record/``, see its README):

* ``kinetics_s1``: FI's block said "The experiment ran for 2.3 s"; the run took 223.2 s (2.3 s was the analysis run again on
  the stored trials), over the topic's 60 s limit.
* ``fdr_e1``: FI's block said "base seed 0" and "ran for 7.4 s ... within [the limit]"; ``run_trial`` never reads its seed
  (its generator is seeded from the literal 20261015 at simulate.py:106), and the run took 121.3 s against a limit of 60 s.

The bypass sentences are the review's (2026-10-15, findings P0-1, P0-2, P1-2, P1-3): each is either impossible (the
writer writes a placeholder and FI fills it) or flagged. The false alarms are its finding P1-1: none is flagged.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from core import paper_facts, quality_report, run_facts, trial_runner
from core.execution import SharedInterpreterExecutor

FIXTURES = Path(__file__).parent / "fixtures" / "paper_record"


def _quest(tmp_path: Path, name: str, *, record: bool = True) -> tuple[Path, str]:
    root = tmp_path / name
    shutil.copytree(FIXTURES / name, root)
    if not record:
        (root / ".fi" / "run_facts.json").unlink()
    return root, str(yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))["topic"])


# ---- A1: the run time is the experiment's, not the last execution's -----------------------------------------------------


def test_the_measured_time_is_the_trials_wall_time_not_the_analysis_rerun(tmp_path: Path) -> None:
    root, _ = _quest(tmp_path, "kinetics_s1")
    assert paper_facts.measured_run_time(root, 2.3)[0] == 223.2, "the last execution's 2.3 s is not the experiment's time"
    assert run_facts.run_seconds(root, 2.3) == 223.2


SIM = '''
import time

def run_trial(cell, trial_id, seed):
    time.sleep(0.4)
    return {"x": cell["a"] + seed % 3}
'''
ANALYSIS = '''
import json, os
data = json.load(open(os.environ["FI_TRIALS"], encoding="utf-8"))
print("RESULT_JSON: " + json.dumps({"n": len(data["cells"])}))
'''


def test_the_runner_keeps_the_trials_wall_time_and_an_analysis_rerun_leaves_it(tmp_path: Path) -> None:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(SIM, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")

    def runner() -> trial_runner.TrialsRunner:
        return trial_runner.TrialsRunner(
            SharedInterpreterExecutor(python_version="3.11"), quest_root=root,
            protocol=lambda: {"grid": {"a": [1, 2]}, "runs_per_setting": 3}, deterministic=False,
            simulate=root / "code" / "simulate.py", analysis=root / "code" / "experiment.py")

    cmd = [sys.executable, str(root / "code" / "experiment.py")]
    first = asyncio.run(runner().execute(cmd, cwd=root, timeout_s=120, env={}))
    record = run_facts.read(root)["trials_run"]
    assert first.returncode == 0 and record["trials"] == 6 and record["cells"] == 2
    assert record["wall_s"] >= 2.4, "six trials of 0.4 s each ran one after the other"
    assert record["trial_sum_s"] >= 2.4 and record["wall_s"] >= record["trial_sum_s"]
    again = asyncio.run(runner().execute(cmd, cwd=root, timeout_s=120, env={}))
    assert again.duration_s < record["wall_s"], "the analysis alone is much shorter"
    assert run_facts.read(root)["trials_run"] == record, "an analysis run again on stored trials leaves the record as it is"
    assert run_facts.run_seconds(root, again.duration_s) == record["wall_s"]


def test_an_explore_then_confirm_quest_says_which_run_the_time_is(tmp_path: Path) -> None:
    """The confirming run replaces the exploring run as the one whose trials are stored and reported; the time says so and
    keeps the exploring run's beside it."""
    root, topic = _quest(tmp_path, "kinetics_s1")
    (root / ".fi" / "run_facts.json").unlink()
    run_facts.record_trials(root, key="a", wall_s=40.0, trials=10, cells=2, trial_sum_s=5.0, stage="explore")
    run_facts.record_trials(root, key="b", wall_s=223.2, trials=10, cells=2, trial_sum_s=30.0, stage="confirm")
    said = paper_facts.facts(root, topic=topic)["run_time"].text
    assert said.startswith("223.2 s") and "confirming run, whose results are reported" in said and "40 s" in said
    assert run_facts.run_seconds(root) == 223.2


# ---- A2: the seed the code used ------------------------------------------------------------------------------------------


# ---- the placeholders: filled from the records ------------------------------------------------------------------------


def test_kinetics_placeholders_are_filled_with_the_true_values(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "kinetics_s1")
    draft = ("# Kinetics\n\n## Methods\nRandom numbers were drawn from {{seed}}; the experiment ran {{trials_per_setting}} "
             "trials in each of {{settings}} settings and took {{run_time}}, {{time_limit_met}} the topic's limit of "
             "{{time_limit}}.\n")
    out = paper_facts.render(draft, root, topic=topic)
    plain = paper_facts.without_marks(out.text)
    assert "{{" not in out.text and not out.unfilled
    assert ("took 223.2 s (about 3.7 minutes), over the topic's limit of 60 s" in plain), plain
    assert "derived from base seed 0" in plain
    assert "2.3 s" not in plain


def test_e1_placeholders_are_filled_with_the_code_s_seed_and_the_measured_time(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "fdr_e1")
    draft = "# FDR\n\n## Methods\nWe used {{seed}}. The run took {{run_time}}, {{time_limit_met}} the {{time_limit}} limit.\n"
    plain = paper_facts.without_marks(paper_facts.render(draft, root, topic=topic, fallback_s=7.4).text)
    assert "We used a seed fixed in the code (20261015, at code/simulate.py:106)." in plain
    assert "The run took 121.3 s, over the 60 s limit." in plain
    assert "base seed" not in plain and "7.4" not in plain and "17" not in plain


def test_rendering_is_idempotent_and_takes_the_records_as_they_are_now(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "kinetics_s1")
    once = paper_facts.render("# T\n\n## Methods\nIt took {{run_time}}.\n", root, topic=topic).text
    assert paper_facts.render(once, root, topic=topic).text == once
    assert paper_facts.to_placeholders(once) == "# T\n\n## Methods\nIt took {{run_time}}.\n"
    # A value edited by hand (or by a model) between the markers is put back from the record.
    edited = once.replace("223.2 s", "2.3 s", 1)
    assert "2.3 s" not in paper_facts.render(edited, root, topic=topic).text


def test_a_placeholder_fi_cannot_fill_reads_not_recorded_and_is_listed(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "kinetics_s1")
    (root / "raw" / "ledger.jsonl").unlink()
    out = paper_facts.render("# T\n\nIt had {{trials_failed}} failures, {{cpu_hours}} and {{not_done:topic-bogus}}. "
                             "Math: $x^{{2}}$ and {{a}}.\n", root, topic=topic)
    assert "{{trials_failed}}" not in out.text and "{{cpu_hours}}" not in out.text and "{{not_done" not in out.text
    assert set(out.unfilled) >= {"trials_failed", "cpu_hours", "not_done:topic-bogus"}
    assert paper_facts.NOT_RECORDED in out.text
    assert "$x^{{2}}$" in out.text, "double braces inside math (LaTeX) are left alone"
    assert "{{a}}" not in paper_facts.without_marks(out.text), "outside math and code, every {{name}} is FI's"
    assert set(paper_facts.unfilled_in(out.text)) >= {"trials_failed", "not_done:topic-bogus", "a"}


# Second review (P1-5): no {{...}} the writer typed reaches paper.md raw. A known name in extra braces is filled; any other
# name is "not recorded" between markers, so QUALITY.md lists it and the check asks for one of FI's names instead.
ODD_PLACEHOLDERS = {
    "nested 4 braces": ("Seed: {{{{seed}}}}.", "seed"),
    "triple": ("Seed: {{{seed}}}.", "seed"),
    "brace inside": ("Seed: {{ {seed} }}.", "seed"),
    "spaces": ("Seed: {{ seed }} and {{ run time }}.", "seed"),
    "dot": ("It took {{run.time}}.", "run_time"),
    "duration": ("The experiment took {{duration}}.", None),
    "time": ("It took {{time}}.", None),
    "seeds": ("Seeds: {{seeds}}.", None),
    "base_seed": ("Seed was {{base_seed}}.", None),
}


@pytest.mark.parametrize("label", sorted(ODD_PLACEHOLDERS))
def test_no_raw_double_braces_reach_the_paper(tmp_path: Path, label: str) -> None:
    root, topic = _quest(tmp_path, "kinetics_s1")
    sentence, known = ODD_PLACEHOLDERS[label]
    out = paper_facts.render(f"# K\n\n## Methods\n{sentence}\n", root, topic=topic).text
    plain = paper_facts.without_marks(out)
    assert "{{" not in plain and "}}" not in plain, plain
    if known:
        value = paper_facts.facts(root, topic=topic)[known].text
        assert value in plain and "{" + value not in plain and value + "}" not in plain, plain
    else:
        assert paper_facts.NOT_RECORDED in plain
        name = paper_facts._norm_name(sentence.split("{{")[1].split("}}")[0])
        assert name in paper_facts.unfilled_in(out), "an unknown name is listed in QUALITY.md"
        hits = paper_facts.placement_hits(out, root, topic=topic)
        assert any(f"{{{{{name}}}}}" in h and "{{run_time}}" in h for h in hits), hits
    # Re-rendering keeps it the same.
    assert paper_facts.render(out, root, topic=topic).text == out


def test_latex_double_braces_are_left_alone(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "kinetics_s1")
    src = "# K\n\n## Methods\n$\\frac{{a}}{b}$ and \\textbf{{x}} and $$y^{{n}}$$ and \\(z_{{k}}\\).\n"
    assert paper_facts.render(src, root, topic=topic).text == src


def test_an_invented_item_id_and_a_wrong_case_id(tmp_path: Path) -> None:
    root = _items_quest(tmp_path)
    out = paper_facts.render("# P\n\n## Limitations\n{{not_done:TOPIC-TEMPERATURE}} {{not_done:made-up-thing}}\n",
                             root).text
    plain = paper_facts.without_marks(out)
    item = next(i for i in paper_facts.required_items(root) if i.id == "topic-temperature")
    assert item.sentence in plain, "an id is matched whatever its case"
    assert paper_facts.unfilled_in(out) == ["not_done:made-up-thing"]
    hits = paper_facts.placement_hits(out, root)
    assert not any("Put {{not_done:topic-temperature}}" in h or "{{not_done:topic-temperature}} (" in h
                   for h in hits), hits
    assert any("{{not_done:made-up-thing}}" in h and "not one of" in h for h in hits), hits


def test_the_quality_report_lists_an_invented_name(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "kinetics_s1")
    paper = paper_facts.render("# P\n\n## Methods\nSeed was {{base_seed}}; it took {{duration}}.\n", root, topic=topic).text
    paper_facts.keep_check(root, paper_facts.check(paper, root, topic=topic), paper)
    md = quality_report.render(quality_report.build(root, running=False))
    assert "base_seed" in md and "duration" in md


# Second review (P1-6): a placeholder in math or code gets the plain value, never FI's markers.
@pytest.mark.parametrize("snippet", ["```\nseed = {{seed}}\n```", "Run with `--seed {{seed}}`.",
                                     "The run used $N = {{trials_per_setting}}$ trials.", "$$T = {{run_time}}$$",
                                     "\\(T = {{run_time}}\\)", "<!-- the run took {{run_time}} -->"])
def test_a_placeholder_in_math_code_or_a_comment_gets_the_plain_value(tmp_path: Path, snippet: str) -> None:
    root, topic = _quest(tmp_path, "kinetics_s1")
    out = paper_facts.render(f"# K\n\n## Methods\n{snippet}\n", root, topic=topic).text
    body = out.split("## Methods\n", 1)[1]
    assert "fi:fact" not in body and "{{" not in body, body
    name = snippet.split("{{")[1].split("}}")[0]
    assert paper_facts.facts(root, topic=topic)[name].text in body


# Second review (P2-2): a required placeholder counts only where a reader sees it.
@pytest.mark.parametrize("where", ["```\n{{seed}} {{run_time}} {{time_limit_met}}\n```",
                                   "<!-- {{seed}} {{run_time}} {{time_limit_met}} -->",
                                   "`{{seed}} {{run_time}} {{time_limit_met}}`"])
def test_a_required_fact_hidden_in_code_or_a_comment_is_not_placed(tmp_path: Path, where: str) -> None:
    root, topic = _quest(tmp_path, "kinetics_s1")
    out = paper_facts.render(f"# S\n\n## Methods\n{where}\n\n## Limitations\nNone.\n", root, topic=topic).text
    hits = paper_facts.placement_hits(out, root, topic=topic)
    assert any("does not state {{run_time}}" in h for h in hits), hits


def test_a_placeholder_between_two_dollar_amounts_is_prose(tmp_path: Path) -> None:
    """Pandoc's rule: a closing ``$`` right before a digit is no math, so a cost sentence is read as prose."""
    root, topic = _quest(tmp_path, "kinetics_s1")
    out = paper_facts.render("# S\n\n## Methods\nA trial cost $5 and {{seed}} was used, $20 in all; it took "
                             "{{run_time}}, {{time_limit_met}} the limit.\n", root, topic=topic).text
    assert "<!-- fi:fact-seed -->" in out
    assert paper_facts.placement_hits(out, root, topic=topic) == []


@pytest.mark.parametrize("heading", ["References", "Further reading", "Bibliography"])
def test_a_required_fact_in_the_source_list_is_not_placed(tmp_path: Path, heading: str) -> None:
    root, topic = _quest(tmp_path, "kinetics_s1")
    out = paper_facts.render(f"# S\n\n## Methods\nM.\n\n## {heading}\n{{{{seed}}}} {{{{run_time}}}} "
                             "{{time_limit_met}}\n", root, topic=topic).text
    hits = paper_facts.placement_hits(out, root, topic=topic)
    assert any("does not state {{seed}}" in h for h in hits), hits


@pytest.mark.parametrize("lim", ["<!-- {{not_done:topic-temperature}} -->",
                                 "```\n{{not_done:topic-temperature}}\n```"])
def test_an_item_hidden_in_the_limitations_is_put_in_by_fi(tmp_path: Path, lim: str) -> None:
    root, topic = _quest(tmp_path, "kinetics_s1")
    (root / ".fi" / "topic_covered.json").write_text(json.dumps({"left": [{"kind": "factor", "name": "temperature"}]}))
    draft = LIMITS_OK.replace("{{not_done:topic-temperature}} We could not.", lim)
    out = paper_facts.render(paper_facts.place_items(draft, root), root, topic=topic).text
    assert paper_facts.placement_hits(out, root, topic=topic) == []
    visible = out.replace(lim, "")
    assert "<!-- fi:fact-not_done-topic-temperature -->" in visible or "temperature" in visible


# ---- what the writer is given --------------------------------------------------------------------------------------------


def test_the_writer_gets_placeholders_with_their_values_and_one_example(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "fdr_e1")
    text = paper_facts.writer_section(root, topic=topic, fallback_s=7.4)
    for name in ("{{seed}}", "{{run_time}}", "{{time_limit}}", "{{time_limit_met}}", "{{trials_per_setting}}",
                 "{{settings}}", "{{trials_failed}}"):
        assert name in text, name
    assert "20261015, at code/simulate.py:106" in text and "121.3 s" in text and '"over"' in text
    assert "Example (in the Methods)" in text and "Do not type them" in text
    assert "Required: the paper must hold {{seed}}, {{run_time}}, {{time_limit_met}}" in text
    assert "reviewer" not in text.lower() and "another model" not in text.lower()


def _items_quest(tmp_path: Path) -> Path:
    root, _ = _quest(tmp_path, "kinetics_s1")
    (root / ".fi" / "topic_covered.json").write_text(json.dumps({"left": [
        {"kind": "factor", "name": "temperature", "main": True, "axis": "", "sentence": "The topic says to vary or compare "
         "temperature, but the plan's settings do not include it."},
        "the topic asks for a sensitivity analysis"]}))
    (root / ".fi" / "code_review.json").write_text(json.dumps({"issues": {
        "F-1": {"text": "x", "readings": [{"status": "open"}],
                "at_gate": {"outcome": "left_out", "quantity": "coverage_Ea", "keys": ["coverage_Ea"]}},
        "F-2": {"text": "y", "readings": [{"status": "open"}],
                "at_gate": {"outcome": "not_computed", "names": ["Ea_sd_se"]}}}}))
    (root / ".fi" / "precision_missed.json").write_text(json.dumps([
        {"name": "coverage_lnA", "half_width": 0.04, "target": 0.02, "trials": 400}]))
    (root / ".fi" / "second_impl.json").write_text(json.dumps({"functions": {
        "fit_k": {"status": "unresolved", "limitation": "Two independently written versions of `fit_k` gave different "
                  "results on t = 1; the number depends on which is right."},
        "weighted_temperature_law": {"status": "unconfirmed", "limitation": "could not be compared"}}}))
    (root / ".fi" / "equation_tests.json").write_text(json.dumps({"not_tested": [
        "equation E3 is not tested at all: the plan gives no example"]}))
    return root


def test_every_required_item_has_an_id_and_one_plain_sentence(tmp_path: Path) -> None:
    root = _items_quest(tmp_path)
    items: dict[str, Any] = {}
    for i in paper_facts.required_items(root):
        items.setdefault(i.kind, i)
    assert items["not_done"].id == "topic-temperature"
    assert items["not_done"].sentence == "The topic asks to vary temperature; this study did not vary it."
    assert items["left_out"].id == "left-out-coverage-ea" and items["not_computed"].id == "not-computed-ea-sd-se"
    assert items["precision"].id == "precision-coverage-lna"
    assert items["second_version"].id == "second-version-fit-k" and "fit_k" in items["second_version"].sentence
    assert items["untested"].sentence == "Equation E3 was not tested at all before the run."
    kinds = [i.kind for i in paper_facts.required_items(root)]
    assert kinds.count("not_done") == 2, "a plain sentence of an older record is an item too"
    assert "weighted_temperature_law" not in json.dumps([i.sentence for i in paper_facts.required_items(root)]), \
        "a second version that could not be compared is in the quality report only"
    section = paper_facts.writer_section(root, topic="t")
    for i in paper_facts.required_items(root):
        assert f"{{{{not_done:{i.id}}}}}" in section
    # A plain sentence of an older record: its id is cut after a whole word, and its sentence does not say "the topic"
    # twice.
    older = [i for i in paper_facts.required_items(root) if i.kind == "not_done" and i.id != "topic-temperature"][0]
    assert older.id == "topic-the-topic-asks-for-a-sensitivity", older.id
    assert older.sentence == "The topic asks for a sensitivity analysis; this study did not do it."


def test_an_open_code_finding_that_changes_a_result_is_an_item(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core import code_review

    root, _ = _quest(tmp_path, "kinetics_s1")
    found = {"missing": [], "different": [], "conflicts": [], "unshown": [], "never_run": [],
             "flagged": [{"fid": "abc", "text": "the variance of k uses n instead of n - 1", "severity": "high",
                          "kind": "logic"},
                         {"fid": "low", "text": "a variable name is unclear", "severity": "low", "kind": "logic"}]}
    monkeypatch.setattr(code_review, "open_findings", lambda q, c: found)
    items = [i for i in paper_facts.required_items(root) if i.kind == "code_finding"]
    assert [i.id for i in items] == ["finding-abc"]
    assert "the variance of k uses n instead of n - 1" in items[0].sentence


LIMITS_OK = ("# Kinetics\n\n## Methods\nWe used {{seed}}; the run took {{run_time}}, {{time_limit_met}} the limit.\n\n"
             "## Limitations\n{{not_done:topic-temperature}} We could not.\n")


def test_a_required_fact_must_be_placed_and_fi_places_the_items_itself(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "kinetics_s1")
    (root / ".fi" / "topic_covered.json").write_text(json.dumps({"left": [{"kind": "factor", "name": "temperature"}]}))

    def hits(draft: str) -> list[str]:
        return paper_facts.placement_hits(paper_facts.render(draft, root, topic=topic).text, root, topic=topic)

    def placed(draft: str) -> str:
        return paper_facts.render(paper_facts.place_items(draft, root), root, topic=topic).text

    mark = "fi:fact-not_done-topic-temperature -->"
    assert hits(LIMITS_OK) == []
    # An item the paper does not hold is put at the end of its Limitations by FI (the item's NAME in the text does not count).
    named = ("# Temperature dependence\n\n## Methods\n{{seed}} {{run_time}} {{time_limit_met}}\n\n"
             "## Limitations\nWe varied the temperature.\n")
    out = placed(named)
    assert out.index("## Limitations") < out.index(mark) and hits(out) == []
    # Wherever the writer put it, it stays there (no second copy, no hit): FI does not check the place.
    elsewhere = LIMITS_OK.replace("## Limitations\n{{not_done:topic-temperature}}", "## Results\n{{not_done:topic-temperature}}")
    assert placed(elsewhere).count(mark) == 2 and hits(elsewhere) == []  # the opening and the closing marker
    # Not in the title, though: the title does not hold it.
    title = "# {{not_done:topic-temperature}}\n\n## Methods\n{{seed}} {{run_time}} {{time_limit_met}}\n"
    assert "## Limitations" in placed(title)
    # A paper with no Limitations section gets one, before the sources.
    no_section = "# T\n\n## Methods\n{{seed}} {{run_time}} {{time_limit_met}}\n\n## References\n1. A.\n"
    out = placed(no_section)
    assert out.index("## Limitations") < out.index(mark) < out.index("## References")
    # A Limitations that is a bold paragraph inside the Discussion (s3 fdr) is not a section: FI adds its own.
    bold = "# T\n\n## Methods\n{{seed}}\n\n## Discussion\n**Limitations.** Few settings.\n\n## References\n1. A.\n"
    out = placed(bold)
    assert out.count("## Limitations") == 1 and out.index("## Discussion") < out.index("## Limitations") < out.index("## References")
    assert paper_facts.place_items(out, root) == out  # a second pass adds nothing
    no_time = LIMITS_OK.replace("{{run_time}}", "223 s")
    assert any("does not state {{run_time}}" in h for h in hits(no_time))


# ---- (b) a run time typed outside a placeholder ---------------------------------------------------------------------------


def test_without_the_record_the_ledger_gives_the_least_the_trials_took(tmp_path: Path) -> None:
    """A quest that predates the record: the last execution's time is not the experiment's, so only the ledger's per-trial
    times are known (a lower bound). The fixture's ledger is cut, so its durations are set to make the sum 75 s."""
    root, topic = _quest(tmp_path, "kinetics_s1", record=False)
    rows = [json.loads(line) for line in (root / "raw" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    trials = [r for r in rows if r["event"] == "trial"]
    for r in trials:
        r["duration_s"] = 75.0 / len(trials)
    (root / "raw" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    wall, least = paper_facts.measured_run_time(root, 2.3)
    assert wall is None and least == pytest.approx(75.0), "the 2.3 s of the last execution is not used"
    said = paper_facts.facts(root, topic=topic, fallback_s=2.3)["run_time"].text
    assert said.startswith("at least 75") and "2.3" not in said


# ---- the record of the check and the quality report -----------------------------------------------------------------------


def test_the_check_is_kept_for_the_quality_report(tmp_path: Path) -> None:
    root = _items_quest(tmp_path)
    bare = paper_facts.render("# P\n\n## Methods\nIt took {{bogus_thing}}.\n\n## Limitations\nThe grid is coarse.\n",
                              root).text
    paper_facts.keep_check(root, paper_facts.check(bare, root), bare)
    kept = paper_facts.kept_check(root)
    assert {i["id"] for i in kept["unmet_items"]} >= {"topic-temperature", "left-out-coverage-ea"}
    text = quality_report.render(quality_report.build(root, running=False))
    assert "## What FI recorded that the paper does not carry" in text
    assert "Where the paper still disagrees with these records" in text and "temperature" in text
    good = paper_facts.render("# P\n\n## Methods\n{{seed}} {{run_time}} {{time_limit_met}}\n\n## Limitations\n"
                              + " ".join(f"{{{{not_done:{i.id}}}}}" for i in paper_facts.required_items(root)) + "\n",
                              root).text
    paper_facts.keep_check(root, paper_facts.check(good, root), good)
    assert paper_facts.kept_check(root) == {}


def test_the_quality_report_lists_a_placeholder_that_reads_not_recorded(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "kinetics_s1")
    paper = paper_facts.render("# P\n\nIt had {{not_done:topic-bogus}}.\n", root, topic=topic).text
    paper_facts.keep_check(root, [], paper)
    md = quality_report.render(quality_report.build(root, running=False))
    assert "not recorded" in md and "topic-bogus" in md


def test_the_quality_report_holds_the_three_blocks_the_paper_no_longer_has(tmp_path: Path) -> None:
    root, _ = _quest(tmp_path, "fdr_e1")
    report = quality_report.build(root, running=False)
    records = report["paper_records"]
    assert "The seed FI gives each trial does not reach the simulation: run_trial takes `seed` but never uses it." \
        in records["run_record"], records["run_record"]
    assert records["run_record"].count("does not reach") == 1, "said once (the review's P2-7)"
    assert "ran for 121.3 s" in records["run_record"] and "was over it" in records["run_record"]
    assert "27 settings" in records["run_record"]
    md = quality_report.render(report)
    assert "### What FI recorded about the run" in md and "121.3 s" in md


def test_the_quality_report_describes_the_study_as_the_quest_resolved_it(tmp_path: Path) -> None:
    """A quest found at clarify to need no simulation (the YAML did not say so) gets the no-simulation wording."""
    root, _ = _quest(tmp_path, "fdr_e1")
    (root / ".fi" / "audit.jsonl").write_text(json.dumps({"kind": "route_decision", "node": "design", "chosen": "x",
                                                          "facts": {"no_simulation": True, "survey_mode": False}}) + "\n")
    seen: dict[str, Any] = {}

    def paragraph(r: Any, *, no_simulation: bool = False, survey: bool = False) -> str:
        seen.update(no_simulation=no_simulation, survey=survey)
        return ""

    import core.disclosure as d

    original = d.paragraph
    d.paragraph = paragraph  # type: ignore[assignment]
    try:
        quality_report.build(root, running=False)
    finally:
        d.paragraph = original  # type: ignore[assignment]
    assert seen == {"no_simulation": True, "survey": False}


# ---- the engine ----------------------------------------------------------------------------------------------------------


def _engine(root: Path, **engine: Any) -> Any:
    from tests.test_disclosure import _engine as make

    eng = make(root)
    for key, value in engine.items():
        setattr(eng.config.engine, key, value)
    eng.config.topic = str(yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))["topic"])
    return eng


WRITTEN = ("# FDR\n\n## Methods\nRandom numbers were drawn from {{seed}}; the run had {{settings}} settings with "
           "{{trials_per_setting}} trials per setting and took {{run_time}}, {{time_limit_met}} the topic's limit of "
           "{{time_limit}}.\n\n## Limitations\nThe grid is coarse.\n")


def test_the_paper_gets_none_of_the_three_blocks_and_holds_the_filled_values(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    root, topic = _quest(tmp_path, "fdr_e1")
    old = (root / "paper_old.md").read_text(encoding="utf-8")
    assert "fi:run-record:" in old and "fi:code-findings:" in old and "fi:attempts" in old
    seen: list[str] = []

    async def fake_chat(self: Any, messages: Any, **kw: Any) -> str:
        seen.append(json.dumps(messages))
        return old + "\n\n" + WRITTEN.split("\n\n", 1)[1]  # a writer that carried the old blocks over, and used placeholders

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    (root / "paper").mkdir(exist_ok=True)
    eng = _engine(root)
    state = {"topic": topic, "title": "t", "iteration": 0, "exec_result": {"returncode": 0, "duration_s": 7.4}}
    out = asyncio.run(eng._node_write(state))  # type: ignore[arg-type]
    paper = Path(out["paper_md"]).read_text(encoding="utf-8")
    for marker in ("fi:attempts", "fi:run-record", "fi:code-findings", "What FI recorded about this run",
                   "How this result was reached", "Open points in the code that produced", "{{"):
        assert marker not in paper, marker
    assert ("took 121.3 s, over the topic's limit of 60 s" in paper_facts.without_marks(paper))
    assert "a seed fixed in the code (20261015, at code/simulate.py:106)" in paper
    assert any("FI'S RECORD OF THIS RUN" in s and "{{seed}}" in s and "121.3 s" in s for s in seen)
    assert run_facts.read(root)["seed"]["verdict"] == "does_not_reach", "the write step keeps the seed verdict on record"


@pytest.mark.parametrize("panel", [False, True])
def test_the_review_adds_the_records_check_and_rewrites_the_text(tmp_path: Path, panel: bool) -> None:
    root, topic = _quest(tmp_path, "fdr_e1")
    eng = _engine(root, max_iterations=3)
    if panel:
        eng.config.engine.review_panel = ["methodologist", "statistician"]

    async def accepting(prompt: str, *, node: Any = None) -> str:
        if node == "review_moderator":
            return json.dumps({"rationale": "both accept"})
        return json.dumps({"verdict": "accept", "score": 5, "suggestions": [], "must_flag_hits": []})

    eng._chat = accepting  # type: ignore[assignment]
    # The sentence shape the writer is asked for, with the false values typed instead of the placeholders.
    paper = root / "paper" / "paper.md"
    paper.parent.mkdir(exist_ok=True)
    paper.write_text("# T\n\n## Methods\nRandom numbers were drawn from {{seed}}; the run had 27 settings with 4,096 trials "
                     "per setting and took 7.4 s, within the 60 s limit.\n", encoding="utf-8")
    patch = asyncio.run(eng._node_review({  # type: ignore[arg-type]
        "topic": topic, "iteration": 0, "review": {}, "paper_md": str(paper),
        "exec_result": {"returncode": 0, "duration_s": 7.4}}))
    hits = [h for h in patch["review"]["must_flag_hits"] if str(h).startswith("paper_record_mismatch")]
    assert hits and any("{{run_time}}" in h for h in hits), hits
    assert patch["iteration"] == 1
    route = eng._route_after_review({"review": patch["review"], "iteration": 1, "code": "x"})  # type: ignore[arg-type]
    assert route == "rewrite"
    assert "{{seed}}" not in paper.read_text(encoding="utf-8"), "the review re-rendered the paper before reading it"


def test_a_failed_records_check_never_goes_to_the_design_or_a_stop(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "fdr_e1")
    eng = _engine(root, max_iterations=3)
    paper = "# P\n\n## Methods\nThe experiment ran for 7.4 s.\n"
    hits = eng._paper_record_hits(paper, {"iteration": 0, "exec_result": {"duration_s": 7.4}})
    assert hits and all(h.startswith("paper_record_mismatch:") for h in hits)
    review = {"verdict": "accept", "must_flag_hits": hits}
    for iteration, record_rewrites in ((0, 0), (1, 5), (2, 5)):
        state = {"review": review, "iteration": iteration, "record_rewrites": record_rewrites, "code": "x"}
        assert eng._route_after_review(state) == "rewrite"  # type: ignore[arg-type]
    assert eng._route_after_review({"review": review, "iteration": 3}) in ("done", "human_feedback")  # type: ignore[arg-type]
    kept = paper_facts.kept_check(root)
    assert kept["hits"] and any("{{run_time}}" in h for h in kept["hits"])


def test_a_resume_at_the_review_takes_old_blocks_out_and_fills_placeholders(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "fdr_e1")
    eng = _engine(root)
    paper = root / "paper" / "paper.md"
    paper.parent.mkdir(exist_ok=True)
    old = (root / "paper_old.md").read_text(encoding="utf-8")
    paper.write_text(old + "\nIt took {{run_time}}.\n", encoding="utf-8")
    eng._refresh_paper_facts({"topic": topic, "paper_md": str(paper), "exec_result": {"duration_s": 7.4}})  # type: ignore[arg-type]
    now = paper.read_text(encoding="utf-8")
    assert "fi:run-record" not in now and "fi:attempts" not in now and "fi:code-findings" not in now
    assert "{{run_time}}" not in now and "121.3 s" in now


def test_the_passage_editor_gets_placeholders_and_fi_fills_them_again(tmp_path: Path) -> None:
    from core import paper_patch

    root, topic = _quest(tmp_path, "fdr_e1")
    rendered = paper_facts.render(WRITTEN, root, topic=topic).text
    given = paper_facts.to_placeholders(rendered)
    assert given == WRITTEN, "the editor sees exactly the writer's placeholders"
    end = len(given)
    # An edit near the placeholder keeps it; FI fills it again.
    edited = paper_patch.apply_edits(given, end, [paper_patch.Edit(find="The grid is coarse.",
                                                                   replace="The grid is coarse in m.")]).text
    again = paper_facts.render(edited, root, topic=topic).text
    assert "121.3 s" in again and "{{" not in again and "coarse in m." in again
    # An edit that types the value over the placeholder is caught by the placement check.
    typed = given.replace("{{run_time}}", "7.4 s")
    hits = paper_facts.check(paper_facts.render(typed, root, topic=topic).text, root, topic=topic)
    assert any("{{run_time}}" in h for h in hits)


def test_the_page_trim_never_offers_a_sentence_holding_fi_s_value(tmp_path: Path) -> None:
    from core import paper_trim

    root, topic = _quest(tmp_path, "fdr_e1")
    body = ("# T\n\n## Introduction\nMany earlier authors studied related corrections in other fields. "
            "Several reviews compare the two procedures in some detail. "
            "The run took {{run_time}} on one machine in the lab today. "
            "Other authors compare the power across many family sizes.\n")
    rendered = paper_facts.render(body, root, topic=topic).text
    offered = paper_trim.candidates(rendered, len(rendered))
    assert offered and not any("121.3" in s.text or "fi:fact" in s.text for s in offered)


def test_the_claim_check_gets_the_corrected_time_and_the_lower_bound_on_old_quests(tmp_path: Path) -> None:
    from core.engine import _claim_method_block

    root, topic = _quest(tmp_path, "kinetics_s1")
    block = _claim_method_block(root, {"topic": topic, "exec_result": {"duration_s": 2.3}})  # type: ignore[arg-type]
    assert "223.2 s" in block and "ran for 2.3 s" not in block
    old, _ = _quest(tmp_path / "old", "kinetics_s1", record=False)
    block = _claim_method_block(old, {"topic": topic, "exec_result": {"duration_s": 2.3}})  # type: ignore[arg-type]
    assert "at least" in block and "2.3 s" not in block


def test_a_rerun_from_the_writing_clears_the_record_check(tmp_path: Path) -> None:
    from core import rerun_from

    assert ".fi/paper_record_check.json" in rerun_from._OWN["writing"]
    assert ".fi/paper_record_check.json" in rerun_from._OWN["review"]


@pytest.mark.asyncio
async def test_the_speech_gets_fi_s_values_without_the_markers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import QuestArtifacts
    from generation.speech import SpeechGenerator
    from tests.test_slides_speech import _PLAIN_TALK, _make_config

    root, topic = _quest(tmp_path, "fdr_e1")
    paper = root / "paper" / "paper.md"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text(paper_facts.render("# T\n\n## Methods\nThe run took {{run_time}}.\n", root, topic=topic).text,
                     encoding="utf-8")
    seen: dict[str, str] = {}

    async def fake_chat(self: Any, messages: Any, **kw: Any) -> str:
        seen["prompt"] = messages[-1]["content"]
        return _PLAIN_TALK

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)
    art = QuestArtifacts(quest_id="q", quest_root=root, paper_md=paper, figures_dir=None)
    await SpeechGenerator(_make_config(tmp_path, kinds=["speech"])).generate(art, root)
    assert "The run took 121.3 s." in seen["prompt"] and "fi:fact" not in seen["prompt"]


# Second review (P1-6): the paper generator takes FI's markers out of every output; a paper written before placeholders in
# math got plain values may hold them there.
MARKED_EVERYWHERE = ("# Marker Probe\n\n## Methods\n\nThe run took <!-- fi:fact-run_time -->223.2 s<!-- /fi:fact-run_time -->"
                     " with $N = <!-- fi:fact-trials_per_setting -->400<!-- /fi:fact-trials_per_setting -->$ trials.\n\n"
                     "Run with `--seed <!-- fi:fact-seed -->7<!-- /fi:fact-seed -->`.\n")


def _paper_cfg(tmp_path: Path, kinds: list[str]) -> Any:
    from core.config import Config

    return Config.model_validate({"topic": "t", "title": "t", "output": {
        "kinds": kinds, "html_pdf_fallback": False, "output_dir": str(tmp_path / "outputs")}})


def test_the_delivered_paper_md_has_no_markers(tmp_path: Path) -> None:
    from core.engine import QuestArtifacts
    from generation.paper import PaperGenerator

    src = tmp_path / "q" / "paper" / "paper.md"
    src.parent.mkdir(parents=True)
    src.write_text(MARKED_EVERYWHERE, encoding="utf-8")
    out = tmp_path / "q"
    PaperGenerator(_paper_cfg(tmp_path, ["paper_md"])).generate(
        QuestArtifacts(quest_id="q", quest_root=out, paper_md=src), out)
    delivered = (out / "paper.md").read_text(encoding="utf-8")
    assert "fi:fact" not in delivered and "223.2 s" in delivered and "$N = 400$" in delivered
    assert "fi:fact" in src.read_text(encoding="utf-8"), "the engine's own copy keeps its markers for the next render"


@pytest.mark.slow
def test_a_real_xelatex_compile_prints_no_markers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from generation._cjk import find_xelatex
    from generation._pandoc import find_pandoc
    from generation._pdf_engine import find_pdf_engine
    from generation._pdf_measure import measure_pdf
    from generation.paper import PaperGenerator

    repo = Path(__file__).resolve().parent.parent
    xelatex = find_xelatex(find_pdf_engine(repo))
    if find_pandoc(repo) is None or xelatex is None:
        pytest.skip("needs pandoc and xelatex")
    monkeypatch.setattr(PaperGenerator, "_find_pdf_engine", lambda self: xelatex)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    paper_md = tmp_path / "paper.md"
    paper_md.write_text(MARKED_EVERYWHERE, encoding="utf-8")
    pdf, skip = PaperGenerator(_paper_cfg(tmp_path, ["paper_md", "paper_pdf"]))._compile_pdf(paper_md, out_dir)
    assert pdf is not None, skip
    source = out_dir / "paper_pdf_source.md"
    if source.is_file():
        assert "fi:fact" not in source.read_text(encoding="utf-8")
    text = " ".join(line.text for page in measure_pdf(pdf, max_pages=2).pages for line in page.lines)
    assert "400" in text and "223.2" in text, text
    assert "fact" not in text and "<!" not in text and "−−" not in text, text
