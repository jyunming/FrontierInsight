"""The self-benchmark's pieces, each on its own and fast: the replay client, the Crossref replay, the answer files and
the held-back tasks, the catalogue, the planters, the scorer's arithmetic and the report. The end-to-end campaign on
fake-model quests is tests/test_self_benchmark_e2e.py (slow)."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pytest

from core import retractions
from core.provider import LAST_CALL
from core.replay import CANNOT_FIX, CrossrefReplay, Recording, ReplayClient, read_events
from core.rerun_from import in_quest
from dev.evaluation.bench import answers, catalogue, plant, reference, report, score


class _Real:
    """A stand-in for the real client: answers with the node's name and says who answered."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def chat(self, messages, *, temperature=0.2, max_tokens=None, extra=None, model=None, node=""):  # noqa: ANN001
        self.asked.append(node)
        LAST_CALL.set({"provider": "openai", "model": model or "m", "reported": True})
        return f"real answer of {node}"


def _ask(client: ReplayClient, node: str) -> tuple[str, dict[str, Any]]:
    async def go() -> tuple[str, dict[str, Any]]:
        LAST_CALL.set(None)
        text = await client.chat([{"role": "user", "content": "p"}], node=node)
        return text, dict(LAST_CALL.get() or {})

    return asyncio.run(go())


# --- the replay client ----------------------------------------------------------------------------------------------

def test_a_recording_run_passes_every_call_on_and_keeps_it(tmp_path: Path) -> None:
    real = _Real()
    client = ReplayClient(mode="record", real=real, out_dir=tmp_path)
    assert _ask(client, "write")[0] == "real answer of write"
    _ask(client, "write")
    rec = Recording.load(tmp_path / "calls.jsonl")
    assert rec.get("write", 2)["response"] == "real answer of write" and rec.get("write", 2)["reported"] is True
    assert real.asked == ["write", "write"]


def test_a_full_replay_answers_from_the_recording_and_says_who_answered(tmp_path: Path) -> None:
    rec = Recording([{"node": "review_panel.statistician", "index": 1, "response": "ok",
                      "provider": "openai", "model": "m-other", "reported": True}])
    client = ReplayClient(mode="replay", recording=rec, out_dir=tmp_path)
    text, who = _ask(client, "review_panel.statistician")
    # The reviewer's model is the recorded one, so a replayed panel is still on two models.
    assert text == "ok" and who["model"] == "m-other" and who["reported"] is True


def test_an_unrecorded_call_gets_the_fixed_answer_and_a_divergence(tmp_path: Path) -> None:
    client = ReplayClient(mode="replay", recording=Recording([]), out_dir=tmp_path)
    text, who = _ask(client, "execute_reflect")
    assert text == CANNOT_FIX and json.loads(text)["cannot_fix"] is True
    assert who["reported"] is False and client.divergences == 1
    (event,) = read_events(tmp_path)
    assert event["event"] == "divergence" and event["node"] == "execute_reflect" and event["index"] == 1


def test_a_partial_replay_replays_until_the_planted_call_then_goes_real(tmp_path: Path) -> None:
    real = _Real()
    rec = Recording([{"node": "implement_outline", "index": 1, "response": "outline", "provider": "openai",
                      "model": "m", "reported": True},
                     {"node": "implement", "index": 1, "response": "clean code"}])
    client = ReplayClient(mode="partial", recording=rec, real=real, plants={("implement", 1): "planted code"},
                          out_dir=tmp_path)
    assert _ask(client, "implement_outline")[0] == "outline"
    assert _ask(client, "implement")[0] == "planted code"
    assert _ask(client, "analyze")[0] == "real answer of analyze"
    assert real.asked == ["analyze"]
    sources = [json.loads(line)["source"] for line in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sources == ["replayed", "planted", "real"]


def test_a_partial_replay_with_nothing_planted_is_a_real_run(tmp_path: Path) -> None:
    real = _Real()
    client = ReplayClient(mode="partial", recording=Recording([]), real=real, out_dir=tmp_path)
    assert _ask(client, "write")[0] == "real answer of write" and client.divergences == 0


def test_a_mode_that_reaches_a_model_needs_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ReplayClient(mode="partial", out_dir=tmp_path)
    with pytest.raises(ValueError):
        ReplayClient(mode="sometimes", real=_Real(), out_dir=tmp_path)


def test_a_recording_is_read_from_the_calls_a_quest_kept(tmp_path: Path) -> None:
    from core import attempt_records as ar

    fi = tmp_path / "q" / ".fi"
    (fi / "io").mkdir(parents=True)
    for n, (node, ts, text) in enumerate([("plan", 10.0, "p1"), ("write", 20.0, "w1"), ("write", 30.0, "w2")]):
        (fi / "io" / f"{1000 + n}-abc-{node}.json").write_text(json.dumps(
            {"ts": ts, "node": node, "model": "m", "usage": None, "messages": [], "response": text}), encoding="utf-8")
        ar.append_model_call(fi, "q", ar.model_call_row(node=node, attempt=1, served={"provider": "openai", "model": "m2",
                                                                                     "reported": True},
                                                        requested_model=None, reports_model=True, messages=[],
                                                        response=text))
    # A failed attempt kept for its cost has no answer, and is not a call to replay.
    (fi / "io" / "0999-abc-write.json").write_text(json.dumps({"ts": 5.0, "node": "write", "response": None}),
                                                   encoding="utf-8")
    rec = Recording.from_quest(tmp_path / "q")
    assert rec.get("write", 2)["response"] == "w2" and rec.get("write", 1)["model"] == "m2"
    after = Recording.from_quest(tmp_path / "q", after=15.0)
    assert after.get("plan", 1) is None and after.get("write", 1)["response"] == "w1"


# --- the Crossref replay ---------------------------------------------------------------------------------------------

def test_crossref_answers_are_replayed_and_an_unknown_doi_is_not_checked(tmp_path: Path) -> None:
    doi = "10.1016/s0140-6736(97)11096-0"
    replay = CrossrefReplay(mode="replay", out_dir=tmp_path, recorded={doi: {"status": "retracted", "why": "r",
                                                                              "notices": []}})
    before = retractions.check_dois
    with replay.installed():
        out = asyncio.run(retractions.check_dois([doi, "10.5555/unknown"]))
    assert out[doi]["status"] == "retracted" and out["10.5555/unknown"]["status"] == "not_checked"
    assert retractions.check_dois is before, "the lookup is put back after the run"
    assert [e["doi"] for e in read_events(tmp_path)] == ["10.5555/unknown"]


def test_crossref_answers_are_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def lookup(dois, **kw):  # noqa: ANN001, ARG001
        return {d: {"status": "not_retracted", "why": "", "notices": []} for d in dois}

    monkeypatch.setattr(retractions, "check_dois", lookup)
    with CrossrefReplay(mode="record", out_dir=tmp_path).installed():
        asyncio.run(retractions.check_dois(["10.1/a"]))
    assert json.loads((tmp_path / "crossref.json").read_text(encoding="utf-8"))["answers"]["10.1/a"]["status"] == "not_retracted"


# --- the answers -----------------------------------------------------------------------------------------------------

def test_every_answer_file_in_the_repository_is_usable() -> None:
    assert answers.check_all() == []
    tasks = {answers.load(p)["task"] for p in answers.files()}
    assert {"Q1", "Q4", "Q5", "Q7", "Q11"} <= tasks


def test_an_ask_sentence_that_does_not_name_the_metric_or_setting_is_refused() -> None:
    data = answers.load(answers.ANSWERS_DIR / "sir_final_size.answer.json")
    data["ask"] = "BENCHMARK. Report the final size."
    problems = answers.validate(data)
    assert any("does not name the metric 'final_fraction'" in p for p in problems)
    assert any("does not name the setting N = 1000" in p for p in problems)


def test_a_held_back_answer_lives_outside_the_repository_and_is_checked_against_its_hash(tmp_path: Path) -> None:
    adir, held = tmp_path / "answers", tmp_path / "held"
    adir.mkdir()
    src = answers.ANSWERS_DIR / "false_discovery.answer.json"
    (adir / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    dest = answers.hold_out(adir / src.name, answers_dir=adir, held_dir=held)
    assert not (adir / src.name).exists() and dest.is_file()
    assert answers.load_task("Q5", answers_dir=adir, held_dir=held)["task"] == "Q5"
    assert answers.check_all(adir) == []
    data = json.loads(dest.read_text(encoding="utf-8"))
    data["answers"][0]["expected"] = 0.05  # an answer changed after it was committed to
    dest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(answers.AnswerError, match="changed after its hash was recorded"):
        answers.load_task("Q5", answers_dir=adir, held_dir=held)


def test_an_answer_is_read_at_its_setting_within_its_tolerance() -> None:
    a = {"expected": 0.04, "tolerance": 0.008, "tolerance_kind": "absolute"}
    assert answers.within(0.047, a) and not answers.within(0.049, a)
    assert answers.within(105, {"expected": 100, "tolerance": 0.05, "tolerance_kind": "relative"})
    assert answers.cell_matches({"m": 200, "pi1": 0.2, "method": "BH"}, {"pi1": 0.2, "method": "bh"})
    assert not answers.cell_matches({"m": 200}, {"m": 2000})


def test_the_reference_computations_agree_with_the_answer_files() -> None:
    q1 = {a["cell"]["method"]: a for a in answers.load_task("Q1")["answers"]}
    for method, order in reference.q1_orders().items():
        assert answers.within(order, q1[method]), (method, order)
    assert reference.q5_bh_fdr(0.2) == pytest.approx(answers.load_task("Q5")["answers"][0]["expected"])
    assert reference.final_size_z(2.0) == pytest.approx(0.7968, abs=1e-4)


# --- the catalogue and the planters ----------------------------------------------------------------------------------

def test_the_catalogue_is_complete_and_names_only_known_gates() -> None:
    ids = set(catalogue.CATALOGUE)
    assert ids == {"N1", "N2", "N3", "N4", "N5", "S1", "S2", "S3", "S4", "L1", "L2", "L3", "L4", "R1", "R2", "R3", "O1"}
    assert {e.id for e in catalogue.CATALOGUE.values() if e.planter} == {"R1", "R2", "N3", "S1", "L1"}
    for e in catalogue.CATALOGUE.values():
        assert set(e.gates) <= set(catalogue.GATES), e.id
        assert e.how in ("edit", "answer", "input", "redesign"), e.id


def test_r1_changes_a_number_of_the_paper_and_nothing_else(tmp_path: Path) -> None:
    (tmp_path / "paper").mkdir()
    text = "# T\n\nIn 2020 we ran 300 runs.\n\nThe mean was 0.987 [1].\n\n## References\n1. A 12.345 B.\n"
    (tmp_path / "paper" / "paper.md").write_text(text, encoding="utf-8")
    rec = plant.plant_r1(tmp_path)
    assert (rec["original_value"], rec["planted_value"]) == ("0.987", "1.352")
    assert (tmp_path / "paper" / "paper.md").read_text(encoding="utf-8") == text.replace("0.987", "1.352")
    with pytest.raises(ValueError, match="nothing was planted"):
        plant.edit_file(tmp_path, "paper/paper.md", "4.567", "1")


def test_r2_adds_a_number_the_run_never_computed_to_the_writers_answer() -> None:
    rec = Recording([{"node": "write", "index": 1, "response": "# T\n\nIntro.\n\nResults."}])
    record, answers_ = plant.plant_r2(rec)
    assert record["call"] == "write#1" and plant.R2_SENTENCE in answers_[("write", 1)]
    with pytest.raises(ValueError):
        plant.plant_r2(Recording([]))


def test_a_code_planter_refuses_a_file_outside_the_code(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plant.plant_code("N3", tmp_path, file="paper/paper.md", find="a", replace="b")
    with pytest.raises(ValueError):
        plant.plant_code("R1", tmp_path, file="code/simulate.py", find="a", replace="b")


def test_l1_plants_a_search_result_with_its_doi() -> None:
    record, (hit,) = plant.plant_l1()
    assert hit["metadata"]["doi"] == plant.WAKEFIELD["doi"] and record["error"] == "L1"


# --- the scorer ------------------------------------------------------------------------------------------------------

def test_the_wilson_interval_matches_the_designs_table() -> None:
    lo, hi = score.wilson(0, 15)
    assert lo == 0 and hi == pytest.approx(0.204, abs=0.001)
    lo, hi = score.wilson(2, 100)
    assert (round(lo, 3), round(hi, 3)) == (0.006, 0.070)
    assert score.wilson(0, 0) is None


@pytest.mark.parametrize("level,gap,gate", [
    ("internally_reconciled", "the provenance check reported 3 finding(s)", "numbers"),
    ("protocol_runtime_matched", "the run was not shown to have done what the protocol fixed (run manifest check: stopped)",
     "run_record"),
    ("independently_validated", "the script did not pass an independent oracle (oracle check: failed)", "oracle"),
    ("statistically_adequate", "the target precision was not reached for p", "statistics"),
    ("publication_ready", "the claim check did not pass: 1 unsupported", "claim_check"),
    ("publication_ready", "the review left 3 must-fix finding(s)", "review"),
])
def test_an_evidence_gap_is_sorted_into_its_gate(level: str, gap: str, gate: str) -> None:
    assert score.gate_of(level, gap) == gate


def test_no_persons_accept_is_not_a_gate() -> None:
    from core import acceptance

    assert score.gate_of("publication_ready", acceptance.NO_PERSON_GAP) is None


def _trial_record(root: Path, values: list[float]) -> None:
    from core import trial_runner as tr

    (root / "raw").mkdir(parents=True)
    (root / ".fi" / "trials").mkdir(parents=True)
    (root / "raw" / "ledger.jsonl").write_text("", encoding="utf-8")
    (root / "raw" / "trials.json").write_text(json.dumps({"cells": [
        {"cell": {"R0": 3.0}, "key": "R0=3.0", "metrics": {"final_size": {"values": values}}},
        {"cell": {"R0": 0.9}, "key": "R0=0.9", "metrics": {"final_size": {"values": [90.0]}}}]}), encoding="utf-8")
    (root / ".fi" / "trials" / "run.json").write_text(json.dumps({"files": tr._file_hashes(root)}), encoding="utf-8")


def test_the_answer_is_read_from_fis_record_and_an_edited_record_is_unscorable(tmp_path: Path) -> None:
    _trial_record(tmp_path, [300.4, 300.6])
    task = {"task": "Q99", "answers": [{"metric": "final_size", "cell": {"R0": 3.0}, "statistic": "mean",
                                        "expected": 300.5, "tolerance": 0.2}]}
    got = score.check_answers(tmp_path, task)
    assert got["correct"] is True and got["values"][0]["got"] == pytest.approx(300.5)
    (tmp_path / "raw" / "trials.json").write_text(json.dumps({"cells": []}), encoding="utf-8")
    edited = score.check_answers(tmp_path, task)
    assert edited["correct"] is None and "changed after FI wrote it" in edited["values"][0]["why"]


def _outcome(**over: Any) -> dict[str, Any]:
    base = {"run": "r", "task": "Q99", "error": None, "role": "clean", "mode": "record", "answer": {"correct": True},
            "evidence_level": "statistically_adequate", "would_publish": True, "stopped": "",
            "infrastructure_failure": False, "first_gate": None, "valid": None, "divergences": 0, "tokens": 10,
            "calls": 2, "seconds": 1.0}
    return {**base, **over}


def test_the_summary_counts_only_valid_planted_runs_and_right_clean_ones() -> None:
    outcomes = [
        _outcome(),
        _outcome(would_publish=False, first_gate="review"),  # a right answer held back
        _outcome(answer={"correct": False}),                  # a wrong answer published
        _outcome(role="planted", error="R1", valid=True, would_publish=False, first_gate="numbers",
                 evidence_level="executed", answer={"correct": True}),
        _outcome(role="planted", error="N3", valid=True, would_publish=True, answer={"correct": False}),
        _outcome(role="planted", error="N3", valid=False, would_publish=True),  # an equivalent change: not counted
    ]
    s = score.summarize(outcomes)
    assert (s["false_pass"]["total"]["k"], s["false_pass"]["total"]["n"]) == (1, 2)
    assert s["false_pass"]["by_error"]["N3"]["k"] == 1 and s["false_pass"]["not_valid"] == 1
    assert (s["false_block"]["k"], s["false_block"]["n"]) == (1, 2)
    assert (s["error_after_publication"]["k"], s["error_after_publication"]["n"]) == (1, 2)
    assert s["detection"]["R1"] == {"numbers": 1, "any": 1} and s["detection"]["N3"] == {"none": 1}
    assert "rate" not in s["calibration"]["statistically_adequate"], "under 10 runs: counts only"


def test_the_report_writes_one_json_and_one_page(tmp_path: Path) -> None:
    outcomes = [_outcome(answer={"correct": True, "values": [{"metric": "m", "cell": {"x": 1}, "expected": 1.0,
                                                              "got": 1.01, "why": ""}]}),
                _outcome(run="p", role="planted", error="R2", valid=True, would_publish=False, first_gate="numbers",
                         answer={"correct": True, "values": []})]
    js, md = report.write(outcomes, tmp_path)
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["schema"] == report.RESULTS_SCHEMA and len(data["runs"]) == 2
    text = md.read_text(encoding="utf-8")
    assert "Wrong results let through" in text and "0/1 = 0%" in text and "| R2 |" in text


# --- the moved-quest fix the benchmark found, and the CLI ------------------------------------------------------------

def test_a_path_of_a_copied_quest_is_taken_to_the_same_place_in_the_copy(tmp_path: Path) -> None:
    old = tmp_path / "a" / "1234-q" / "paper" / "paper.md"
    new_root = tmp_path / "b" / "1234-q"
    assert Path(in_quest(old, new_root)) == new_root / "paper" / "paper.md"
    assert in_quest(new_root / "x.md", new_root) == str(new_root / "x.md")
    assert in_quest(tmp_path / "elsewhere.md", new_root) == str(tmp_path / "elsewhere.md")
    assert in_quest(None, new_root) == ""


def test_fi_tools_bench_check_runs_from_the_cli(capsys: pytest.CaptureFixture[str]) -> None:
    from launch import _expand_tools_argv

    with pytest.raises(SystemExit) as done:
        _expand_tools_argv(["tools", "bench", "check"])
    assert done.value.code == 0
    assert "no problem found" in capsys.readouterr().out


def test_math_is_finite_in_the_answer_files() -> None:
    for path in answers.files():
        for a in answers.load(path).get("answers") or []:
            assert math.isfinite(a["expected"]) and a["tolerance"] > 0
