"""The run record: FI writes the run's facts into the paper's limitations from its own records, no model.

The fixtures under ``tests/fixtures/run_record/`` are the small record files of three real runs (their ledgers are
cut to the planned lines and the first trials of the first settings)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from core import disclosure

FIXTURES = Path(__file__).parent / "fixtures" / "run_record"


def _run(tmp_path: Path, name: str) -> tuple[Path, str]:
    root = tmp_path / name
    shutil.copytree(FIXTURES / name, root)
    return root, (root / "topic.txt").read_text(encoding="utf-8")


def _write(root: Path, rel: str, data: object) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


def test_false_discovery_run_states_its_seeds_pairing_counts_and_untested_equation(tmp_path: Path) -> None:
    root, topic = _run(tmp_path, "fdr_r10b")
    text = disclosure.run_record(root, topic=topic)
    assert text.startswith("**What FI recorded about this run.**")
    assert "base seed 0" in text and "common random numbers" in text and "paired, not independent" in text
    assert "The run had 9 settings with 1,200 trials per setting" in text
    assert "0 failed" in text
    assert "Equation E6 was not tested against a worked example before the run" in text
    # No duration was given: no run time sentence. The topic parts the run did not do are only those FI recorded
    # (this run's record lists none), so none are invented.
    assert "ran for" not in text and "topic's requirements" not in text


def test_a_run_over_the_topics_limit_says_over_it_and_never_within(tmp_path: Path) -> None:
    root, topic = _run(tmp_path, "kinetics_r9k")
    # The topic's constraints record is empty, so the limit ("under 60 seconds of runtime") is read from the topic text.
    required = json.loads((root / ".fi" / "topic_requirements.json").read_text(encoding="utf-8"))
    assert required["constraints"] == []
    # 111.3 s: the execute stage's own line in that run's .fi/run.log ("rc=0 duration=111.3s").
    text = disclosure.run_record(root, topic=topic, duration_s=111.3)
    assert "The experiment ran for 111.3 s; the topic sets a limit of 60 s, and the run was over it." in text
    assert "within" not in text
    assert "30 settings with 400 trials per setting" in text


def test_a_reduced_run_gives_its_note_verbatim_and_the_topic_parts_not_done(tmp_path: Path) -> None:
    root, topic = _run(tmp_path, "fdr_r6t")
    note = json.loads((root / ".fi" / "run_estimate.json").read_text(encoding="utf-8"))["reduced"]["note"]
    text = disclosure.run_record(root, topic=topic)
    assert note in text
    assert "The topic states which values of m to use, but the values 200 are not among its settings." in text
    assert "The topic states which values of pi1 to use" in text
    # No ledger in this run: no counts, and the seed sentence rests on the study file alone.
    assert "The run had" not in text and "base seed 0" in text


def test_two_different_limits_in_the_topic_are_not_checked(tmp_path: Path) -> None:
    root = tmp_path / "q"
    topic = "Finish under 60 seconds of runtime in the first part and at most 5 minutes of computing in the second."
    text = disclosure.run_record(root, topic=topic, duration_s=73.6)
    assert "The experiment ran for 73.6 s; FI did not check it against a limit in the topic." in text
    assert "within" not in text and "over it" not in text


def test_no_limit_in_the_topic_is_not_checked(tmp_path: Path) -> None:
    text = disclosure.run_record(tmp_path / "q", topic="Study the effect of noise.", duration_s=5)
    assert "ran for 5 s; FI did not check it against a limit in the topic" in text


def test_a_run_under_its_limit_says_within_it(tmp_path: Path) -> None:
    root = tmp_path / "q"
    _write(root, ".fi/topic_requirements.json", {"constraints": ["CPU only", "at most 2 minutes of run time"]})
    text = disclosure.run_record(root, topic="a topic that says under 5 seconds of runtime", duration_s=73.6)
    # The constraints are read first, and only they when they state a limit.
    assert "the topic sets a limit of 120 s, and the run was within it." in text
    over = disclosure.run_record(root, duration_s=120.5)
    assert "was over it" in over and "within" not in over


def test_a_limit_without_a_word_about_the_run_is_not_a_run_limit(tmp_path: Path) -> None:
    text = disclosure.run_record(tmp_path / "q", topic="Samples are taken within 10 seconds of mixing.", duration_s=3)
    assert "did not check it against a limit" in text


def test_nothing_recorded_is_an_empty_paragraph_and_missing_records_leave_their_sentence_out(tmp_path: Path) -> None:
    assert disclosure.run_record(tmp_path / "nothing") == ""
    root = tmp_path / "q"
    _write(root, ".fi/equation_tests.json", {"not_tested": ["equation E1 is not tested before the run: no example"]})
    _write(root, ".fi/run_estimate.json", "{not json")
    text = disclosure.run_record(root)
    assert "base seed" not in text and "The run had" not in text and "ran for" not in text
    assert "Equation E1 was not tested against a worked example before the run" in text
    _write(root, "code/study.json", {"grid": {}, "paired": True})
    assert "base seed" not in disclosure.run_record(root)


def _ledger(root: Path, seeds: dict[str, list[int]], status: dict[tuple[str, int], str] | None = None) -> None:
    rows = [{"event": "planned", "cell": c, "trials": len(s)} for c, s in seeds.items()]
    for c, s in seeds.items():
        for t, seed in enumerate(s):
            rows.append({"event": "trial", "cell": c, "trial": t, "seed": seed,
                         "status": (status or {}).get((c, t), "ok")})
    _write(root, "raw/ledger.jsonl", "\n".join(json.dumps(r) for r in rows) + "\n")


def test_the_ledger_overrules_the_study_files_paired_flag(tmp_path: Path) -> None:
    root = tmp_path / "q"
    _write(root, "code/study.json", {"base_seed": 7, "paired": False})
    _ledger(root, {"a": [1, 2, 3], "b": [1, 2, 3]})
    text = disclosure.run_record(root)
    assert "the run's ledger shows that every setting used the same seeds" in text and "paired, not independent" in text
    assert "the setting and the trial number" not in text

    _write(root, "code/study.json", {"base_seed": 7, "paired": True})
    _ledger(root, {"a": [1, 2, 3], "b": [4, 5, 6]})
    text = disclosure.run_record(root)
    assert "the run's ledger shows that different settings used different seeds" in text
    assert "common random numbers" not in text

    _write(root, "code/study.json", {"base_seed": 7, "paired": False})
    assert "the setting and the trial number, so different settings use different seeds" in disclosure.run_record(root)


def test_counts_cover_ranges_and_failures(tmp_path: Path) -> None:
    root = tmp_path / "q"
    _ledger(root, {"a": [1, 2, 3, 4], "b": [5, 6]}, status={("a", 3): "failed"})
    text = disclosure.run_record(root)
    assert "The run had 2 settings with 2 to 4 trials per setting; 5 trials finished and 1 failed." in text


# ---- the paper -------------------------------------------------------------------------------------------------------

RECORD = ("**What FI recorded about this run.** Each trial's random seed is derived from base seed 4242 and the trial "
          "number. The experiment ran for 73.6 s; FI did not check it against a limit in the topic.")
FINDINGS = "**Open points in the code.** A second reading left one point open."
WITH_LIMITS = ("# Title\n\n## Methods\n\nWe did it.\n\n## Limitations\n\nThe writer's own limits.\n\n"
               "## References\n\n[1] A. Author.\n")
WITHOUT_LIMITS = "# Title\n\n## Methods\n\nWe did it.\n\n## Results\n\nNumbers.\n\n## References\n\n[1] A. Author.\n"


def test_the_block_goes_at_the_end_of_the_limitations_section_before_the_references() -> None:
    paper = disclosure.mark_run_record(WITH_LIMITS, RECORD)
    assert paper.index("The writer's own limits.") < paper.index("<!-- fi:run-record:") < paper.index("## References")
    assert paper.count("## Limitations") == 1
    assert disclosure.RUN_RECORD_END in paper


def test_a_paper_without_limitations_gets_a_section_before_the_references() -> None:
    paper = disclosure.mark_run_record(WITHOUT_LIMITS, RECORD)
    assert paper.index("## Results") < paper.index("## Limitations") < paper.index("<!-- fi:run-record:")
    assert paper.index(disclosure.RUN_RECORD_END) < paper.index("## References")


def test_run_record_comes_before_code_findings_and_marking_twice_changes_nothing() -> None:
    for source in (WITH_LIMITS, WITHOUT_LIMITS):
        once = disclosure.mark_findings(disclosure.mark_run_record(source, RECORD), FINDINGS)
        assert once.index("<!-- fi:run-record:") < once.index("<!-- fi:code-findings:")
        twice = disclosure.mark_findings(disclosure.mark_run_record(once, RECORD), FINDINGS)
        assert twice == once
        # The other call order lands the same way once the run record is placed.
        other = disclosure.mark_run_record(disclosure.mark_findings(source, FINDINGS), RECORD)
        assert other.index("<!-- fi:run-record:") < other.index("<!-- fi:code-findings:")
        assert other.count("## Limitations") == 1 and other.count("fi:run-record:") == 1
        assert disclosure.mark_run_record(other, RECORD) == other


def test_text_the_writer_put_between_the_markers_is_replaced_and_an_empty_record_leaves_none() -> None:
    forged = ("# T\n\n## Limitations\n\n<!-- fi:run-record:000000000000 -->\nThe run took 1 s.\n"
              "<!-- /fi:run-record -->\n\n## References\n")
    paper = disclosure.mark_run_record(forged, RECORD)
    assert "The run took 1 s." not in paper and paper.count("fi:run-record:") == 1
    gone = disclosure.mark_run_record(paper, "")
    assert "fi:run-record" not in gone
    assert "fi:run-record" not in disclosure.without_block(paper)


def test_the_checks_leave_out_the_unedited_block_and_keep_an_edited_one() -> None:
    paper = disclosure.mark_run_record(WITH_LIMITS, RECORD)
    stripped = disclosure.strip_for_checks(paper)
    assert "4242" not in stripped and "73.6" not in stripped and "fi:run-record" not in stripped
    assert "The writer's own limits." in stripped
    edited = paper.replace("73.6 s", "7.6 s")
    kept = disclosure.strip_for_checks(edited)
    assert "4242" in kept and "7.6 s" in kept
    both = disclosure.mark_findings(paper, FINDINGS)
    assert "4242" not in disclosure.strip_for_checks(both) and "second reading" not in disclosure.strip_for_checks(both)


def test_trimming_never_offers_the_block(tmp_path: Path) -> None:
    from core import paper_trim

    paper = disclosure.mark_run_record(WITH_LIMITS, RECORD)
    assert paper_trim._ENGINE_NOTE_RE.search(paper)
    cut = [(m.start(), m.end()) for m in paper_trim._ENGINE_NOTE_RE.finditer(paper)]
    assert any(paper[a:b].startswith("<!-- fi:run-record:") for a, b in cut)
