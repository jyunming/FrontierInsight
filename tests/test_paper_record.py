"""The paper states what FI recorded about the run, and FI checks the paper against its records (core/paper_facts.py,
core/run_facts.py).

Replays of two real quests (``tests/fixtures/paper_record/``, see its README):

* ``kinetics_s1``: FI's block said "The experiment ran for 2.3 s"; the run took 223.2 s (2.3 s was the analysis run again on
  the stored trials).
* ``fdr_e1``: FI's block said "base seed 0" and "ran for 7.4 s ... within [the limit]"; ``run_trial`` never reads its seed
  (the code seeds from the literal 20261015), and the run took 121.3 s against a limit of 60 s.
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

from core import disclosure, paper_facts, quality_report, run_facts, trial_runner
from core.execution import SharedInterpreterExecutor
from core.stat_claims import check, seed_records

FIXTURES = Path(__file__).parent / "fixtures" / "paper_record"


def _quest(tmp_path: Path, name: str, *, record: bool = True) -> tuple[Path, str]:
    root = tmp_path / name
    shutil.copytree(FIXTURES / name, root)
    if not record:
        (root / ".fi" / "run_facts.json").unlink()
    return root, str(yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))["topic"])


def _hit_text(hits: list[str]) -> str:
    return "\n".join(hits)


# ---- A1: the run time is the experiment's, not the last execution's -----------------------------------------------------


def test_the_measured_time_is_the_trials_wall_time_not_the_analysis_rerun(tmp_path: Path) -> None:
    root, _ = _quest(tmp_path, "kinetics_s1")
    assert paper_facts.measured_run_time(root, 2.3)[0] == 223.2, "the last execution's 2.3 s is not the experiment's time"
    assert run_facts.run_seconds(root, 2.3) == 223.2


def test_the_old_sentence_is_found_wrong_against_the_measured_time(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "kinetics_s1")
    old = "The experiment ran for 2.3 s; the topic sets a limit of 60 s, and the run was within it."
    hits = paper_facts.time_hits(old, root, topic, 2.3)
    assert hits and all(h.startswith("paper_record_mismatch:") for h in hits)
    assert "223.2 s" in hits[0] and "2.3 s" in hits[0]


def test_a_run_time_that_agrees_with_the_measurement_passes(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "kinetics_s1")
    for text in ("The whole experiment ran for 223 s on one core.", "The run took about 4 minutes (223 s).",
                 "The experiment took roughly 3.7 minutes of wall-clock time.",
                 "Each trial took 0.0007 s per setting.", "The analysis ran for 2.3 s on the stored trials."):
        assert paper_facts.time_hits(text, root, topic) == [], text


def test_a_run_over_the_limit_cannot_be_called_within_it(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "fdr_e1")
    assert disclosure.topic_limit(root, topic) == 60.0
    for text in ("The run stayed within the topic's runtime limit.", "The experiment ran for 7.4 s, within the 60 s limit.",
                 "The computation completed in under 60 seconds, as the topic requires."):
        assert paper_facts.time_hits(text, root, topic), text
    for text in ("The experiment took 121 s, over the topic's limit of 60 s.",
                 "The run did not meet the 60 s limit; it took 2 minutes.",
                 "The topic sets a limit of 60 s."):
        assert paper_facts.time_hits(text, root, topic) == [], text


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
    assert paper_facts.time_hits("The experiment ran for 2.3 s.", root, topic, 2.3)
    assert paper_facts.time_hits("The experiment ran for 4 minutes.", root, topic, 2.3) == []
    said = paper_facts.run_time_fact(root, topic, 2.3)
    assert "at least 75" in said and "Do not state a run time" in said and "2.3" not in said


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


# ---- A2: the seed the code used ------------------------------------------------------------------------------------------


def test_e1_run_trial_never_uses_the_seed_fi_gives_it(tmp_path: Path) -> None:
    root, _ = _quest(tmp_path, "fdr_e1")
    verdict = run_facts.seed_verdict(root)
    assert verdict["fi_seed_reaches"] is False and "never uses it" in verdict["why"]
    assert any(lit["value"] == 20261015 and lit["where"] == "code/simulate.py:106" for lit in verdict["code_seeds"])
    assert any(lit["value"] == 20261015 and lit["where"] == "code/simulate.py:176" for lit in verdict["code_seeds"])
    kinetics, _ = _quest(tmp_path, "kinetics_s1")
    assert run_facts.seed_verdict(kinetics)["fi_seed_reaches"] is True, "default_rng(seed) uses it"


def test_e1_paper_saying_base_seed_0_is_wrong_and_the_code_s_seed_passes(tmp_path: Path) -> None:
    root, _ = _quest(tmp_path, "fdr_e1")
    seeds = seed_records(root)
    assert seeds.fi_seed_reaches is False and seeds.base_seed == 0
    old = "Each trial's random seed is derived from base seed 0 and the trial number."
    found = [f for f in check(old, seeds=seeds).findings if f.kind == "seed_not_recorded"]
    assert len(found) == 1 and "20261015" in found[0].message and "never uses the seed FI gives each trial" in found[0].message
    for ok in ("The simulation seeds its generators with the number 20261015 written in the code.",
               "Every family is generated from seed 20261015, the trial number and the setting."):
        assert [f for f in check(ok, seeds=seeds).findings if f.kind == "seed_not_recorded"] == [], ok


def test_a_seed_that_reaches_the_code_keeps_the_existing_check(tmp_path: Path) -> None:
    root, _ = _quest(tmp_path, "kinetics_s1")
    seeds = seed_records(root)
    assert seeds.fi_seed_reaches is True
    assert [f for f in check("Each trial's seed is derived from base seed 0.", seeds=seeds).findings
            if f.kind == "seed_not_recorded"] == []
    assert [f for f in check("Each trial's seed is derived from base seed 5.", seeds=seeds).findings
            if f.kind == "seed_not_recorded"]


# ---- what the writer is given ---------------------------------------------------------------------------------------------


def test_the_writer_gets_the_measured_facts_and_the_seed_the_code_used(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "fdr_e1")
    text = paper_facts.writer_section(root, topic=topic, fallback_s=7.4)
    assert "does NOT reach the simulation" in text and "20261015" in text and "do not say the trials' seeds are derived" in text
    assert "ran for 121.3 s" in text and "OVER" in text and "do not say the limit was met" in text
    assert "27 settings with 4,096 trials per setting" in text
    assert "Methods" in text and "Limitations" not in text.split("In the Methods")[0]
    # What goes to the quality report only never reaches the writer.
    assert "reviewer" not in text.lower() and "another model" not in text.lower() and "second version" not in text.lower()


def _items_quest(tmp_path: Path) -> Path:
    root, _ = _quest(tmp_path, "kinetics_s1")
    (root / ".fi" / "topic_covered.json").write_text(json.dumps({"left": [
        {"kind": "factor", "name": "temperature", "main": True, "axis": "", "sentence": "The topic asks to vary the temperature."}]}))
    (root / ".fi" / "code_review.json").write_text(json.dumps({"issues": {
        "F-1": {"text": "x", "readings": [{"status": "open"}],
                "at_gate": {"outcome": "left_out", "quantity": "coverage_Ea", "keys": ["coverage_Ea"]}},
        "F-2": {"text": "y", "readings": [{"status": "open"}],
                "at_gate": {"outcome": "not_computed", "names": ["Ea_sd_se"]}}}}))
    (root / ".fi" / "precision_missed.json").write_text(json.dumps([
        {"name": "coverage_lnA", "half_width": 0.04, "target": 0.02, "trials": 400}]))
    return root


def test_every_item_not_done_left_out_not_computed_or_missed_is_given_by_name(tmp_path: Path) -> None:
    root = _items_quest(tmp_path)
    items = paper_facts.required_items(root)
    assert {(i.kind, i.name) for i in items} >= {("not_done", "temperature"), ("left_out", "coverage_Ea"),
                                                   ("not_computed", "Ea_sd_se"), ("precision", "coverage_lnA")}
    section = paper_facts.writer_section(root, topic="t")
    for name in ("temperature", "coverage_Ea", "Ea_sd_se", "coverage_lnA"):
        assert f'"{name}"' in section


def test_a_paper_that_leaves_an_item_out_is_sent_back_and_naming_it_clears_it(tmp_path: Path) -> None:
    root = _items_quest(tmp_path)
    bare = "# P\n\n## Limitations\nThe grid is coarse.\n"
    hits = paper_facts.check(bare, root)
    assert len(hits) == 4 and all(h.startswith("paper_record_mismatch:") for h in hits)
    assert "does not mention \"temperature\"" in _hit_text(hits)
    said = ("# P\n\n## Limitations\nThe temperature was not varied as the topic asked. `coverage_Ea` is left out because a "
            "test showed the code computes it wrongly. The code does not compute ea sd se. The target for coverage lnA "
            "was missed.\n")
    assert paper_facts.check(said, root) == [], "case, spacing and underscores are ignored"
    # FI's old block does not count as the paper saying it.
    old = disclosure.mark_run_record(bare, "**What FI recorded about this run.** temperature coverage_Ea Ea_sd_se coverage_lnA")
    assert len(paper_facts.check(old, root)) == 4


def test_the_check_is_kept_for_the_quality_report(tmp_path: Path) -> None:
    root = _items_quest(tmp_path)
    bare = "# P\n\n## Limitations\nThe grid is coarse.\n"
    paper_facts.keep_check(root, paper_facts.check(bare, root), bare)
    kept = paper_facts.kept_check(root)
    assert {i["name"] for i in kept["unmet_items"]} == {"temperature", "coverage_Ea", "Ea_sd_se", "coverage_lnA"}
    text = quality_report.render(quality_report.build(root, running=False))
    assert "## What FI recorded that the paper does not carry" in text
    assert "Where the paper still disagrees with these records" in text and '"temperature"' in text
    paper_facts.keep_check(root, [], "# P\n\ntemperature coverage_Ea Ea_sd_se coverage_lnA\n")
    assert paper_facts.kept_check(root) == {}


def test_the_quality_report_holds_the_three_blocks_the_paper_no_longer_has(tmp_path: Path) -> None:
    root, _ = _quest(tmp_path, "fdr_e1")
    report = quality_report.build(root, running=False)
    records = report["paper_records"]
    assert "The seed FI gives each trial does not reach the simulation" in records["run_record"]
    assert "ran for 121.3 s" in records["run_record"] and "was over it" in records["run_record"] or "over" in records["run_record"]
    assert "27 settings" in records["run_record"]
    md = quality_report.render(report)
    assert "### What FI recorded about the run" in md and "121.3 s" in md


# ---- the engine ----------------------------------------------------------------------------------------------------------


def _engine(root: Path, **engine: Any) -> Any:
    from tests.test_disclosure import _engine as make

    eng = make(root)
    for key, value in engine.items():
        setattr(eng.config.engine, key, value)
    return eng


def test_the_paper_gets_none_of_the_three_blocks_and_the_writer_gets_the_record(tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    root, topic = _quest(tmp_path, "fdr_e1")
    old = (root / "paper_old.md").read_text(encoding="utf-8")
    assert "fi:run-record:" in old and "fi:code-findings:" in old and "fi:attempts" in old
    seen: list[str] = []

    async def fake_chat(self: Any, messages: Any, **kw: Any) -> str:
        seen.append(json.dumps(messages))
        return old  # a writer that carried the old blocks over (a resumed or refined quest)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    (root / "paper").mkdir(exist_ok=True)
    eng = _engine(root)
    state = {"topic": topic, "title": "t", "iteration": 0, "exec_result": {"returncode": 0, "duration_s": 7.4}}
    out = asyncio.run(eng._node_write(state))  # type: ignore[arg-type]
    paper = Path(out["paper_md"]).read_text(encoding="utf-8")
    for marker in ("fi:attempts", "fi:run-record", "fi:code-findings", "What FI recorded about this run",
                   "How this result was reached", "Open points in the code that produced"):
        assert marker not in paper, marker
    assert any("FI'S RECORD OF THIS RUN" in s and "does NOT reach the simulation" in s and "ran for 121.3 s" in s for s in seen)


def test_a_failed_records_check_rewrites_the_text_and_never_goes_to_the_design_or_a_stop(tmp_path: Path) -> None:
    root, topic = _quest(tmp_path, "fdr_e1")
    eng = _engine(root, max_iterations=3)
    paper = "# P\n\n## Methods\nThe experiment ran for 7.4 s.\n"
    hits = eng._paper_record_hits(paper, {"iteration": 0, "exec_result": {"duration_s": 7.4}})
    assert hits and hits[0].startswith("paper_record_mismatch:")
    review = {"verdict": "accept", "must_flag_hits": hits}
    # Even past the first rewrites, a records mismatch is a text route: the design is not changed and the quest not stopped.
    for iteration, record_rewrites in ((0, 0), (1, 5), (2, 5)):
        state = {"review": review, "iteration": iteration, "record_rewrites": record_rewrites, "code": "x"}
        assert eng._route_after_review(state) == "rewrite"  # type: ignore[arg-type]
    # The rounds are used up: the paper goes out, and the point is on record.
    assert eng._route_after_review({"review": review, "iteration": 3}) in ("done", "human_feedback")  # type: ignore[arg-type]
    kept = paper_facts.kept_check(root)
    assert kept["hits"] and "7.4 s" in kept["hits"][0]


def test_the_claim_check_gets_the_corrected_time(tmp_path: Path) -> None:
    from core.engine import _claim_method_block

    root, topic = _quest(tmp_path, "kinetics_s1")
    block = _claim_method_block(root, {"topic": topic, "exec_result": {"duration_s": 2.3}})  # type: ignore[arg-type]
    assert "223.2 s" in block and "ran for 2.3 s" not in block
