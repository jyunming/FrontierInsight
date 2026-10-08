"""The evaluation-suite harness (dev/evaluation/suite): configs made from a template, the held-back group, the runner with
processes mocked, the collector over fake quest folders, the failure taxonomy, the trend arithmetic and the grader's side.
No quest is launched and no model is called."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from dev.evaluation.suite import adopt, cli, collect, grade, runner, taxonomy, trend
from dev.evaluation.suite.suite import Suite, SuiteError

PROVIDER = {"name": "prov", "model": "main-m", "node_providers": {"implement": {"name": "cli", "model": "code-m"}}}


def _w(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


def _suite(tmp_path: Path) -> Suite:
    root = tmp_path / "suite"
    for name in ("alpha", "beta", "gamma", "delta"):
        _w(root / "topics" / f"{name}.yaml", yaml.safe_dump(
            {"topic": f"TOPIC. {name} question.\nSCOPE. one thing.\n", "title": f"t-{name}",
             "provider": {"name": "openai"}, "engine": {"max_iterations": 2}, "knowledge": {"top_k": 5, "write_back_quests": True},
             "output": {"kinds": ["paper_md"], "output_dir": "./outputs"}}))
    _w(root / "template.yaml", yaml.safe_dump({"provider": PROVIDER, "knowledge": {"write_back_quests": False},
                                               "pauses": {"papers": False}}))
    _w(root / ".env", "# c\nOLLAMA_API_KEY=secret-key-123\nexport OTHER='quoted'\n")
    _w(root / "suite.yaml", yaml.safe_dump({
        "rounds_dir": "rounds", "env_file": ".env", "config_template": "template.yaml",
        "groups": {"dev": {"topics": ["alpha", "beta", "gamma"]}, "held": {"acceptance_only": True, "topics": ["delta"]}},
        "topics": {n: f"topics/{n}.yaml" for n in ("alpha", "beta", "gamma", "delta")}}))
    return Suite(root / "suite.yaml")


def _fi(tmp_path: Path) -> Path:
    fi = tmp_path / "fi"
    _w(fi / "launch.py", "print('x')\n")
    return fi


# ------------------------------------------------------------------------------------------------ configs

def test_every_topic_gets_the_same_provider_block_and_its_own_output_folder(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    paths = suite.write_configs("dev", tmp_path / "r1")
    cfgs = {t: yaml.safe_load(p.read_text(encoding="utf-8")) for t, p in paths.items()}
    assert len(cfgs) == 3
    assert all(c["provider"] == PROVIDER for c in cfgs.values())
    assert all(c["knowledge"] == {"top_k": 5, "write_back_quests": False} for c in cfgs.values())
    assert all(c["pauses"] == {"papers": False} for c in cfgs.values())
    assert cfgs["alpha"]["output"]["output_dir"].endswith("r1/alpha/out")
    assert cfgs["alpha"]["output"]["kinds"] == ["paper_md"]  # the topic's own keys stay
    assert "alpha question" in cfgs["alpha"]["topic"]


def test_unknown_group_and_topic_in_a_group_are_named(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    with pytest.raises(SuiteError, match="no group"):
        suite.group_topics("nope")
    bad = tmp_path / "bad.yaml"
    _w(bad, yaml.safe_dump({"groups": {"g": {"topics": ["zzz"]}}, "topics": {"a": "a.yaml"}}))
    with pytest.raises(SuiteError, match="zzz"):
        Suite(bad)


# --------------------------------------------------------------------------------------------- held-back group

def test_held_back_group_needs_acceptance(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    fi = _fi(tmp_path)
    with pytest.raises(SuiteError, match="acceptance"):
        runner.prepare_round(suite, "held", "r1", fi, max_parallel=1, git=lambda p: {"sha": "a", "short": "a", "dirty": False})
    assert not (suite.rounds_dir / "r1").exists()  # nothing was written
    rdir = runner.prepare_round(suite, "held", "r1", fi, max_parallel=1, acceptance=True,
                                git=lambda p: {"sha": "a", "short": "a", "dirty": False})
    assert (rdir / runner.MANIFEST).is_file()


def test_cli_refuses_held_back_group_without_the_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite = _suite(tmp_path)
    rc = cli.main(["run", "--suite", str(suite.path), "--group", "held", "--round", "x", "--fi", str(_fi(tmp_path))])
    assert rc == 1
    assert "--acceptance" in capsys.readouterr().err


def test_round_needs_a_frozen_fi_and_a_new_name(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    git = lambda p: {"sha": "a", "short": "a", "dirty": False}  # noqa: E731
    with pytest.raises(SuiteError, match="launch.py"):
        runner.prepare_round(suite, "dev", "r1", tmp_path / "empty", max_parallel=2, git=git)
    runner.prepare_round(suite, "dev", "r1", _fi(tmp_path), max_parallel=2, git=git)
    with pytest.raises(SuiteError, match="already exists"):
        runner.prepare_round(suite, "dev", "r1", _fi(tmp_path), max_parallel=2, git=git)


# ----------------------------------------------------------------------------------------------- the runner

class _Proc:
    _next = 1000

    def __init__(self, cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
        _Proc._next += 1
        self.pid, self.cmd, self.cwd, self.env, self.code = _Proc._next, cmd, cwd, env, None

    def poll(self) -> int | None:
        return self.code


def test_runner_never_exceeds_max_parallel_and_records_pids(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    rdir = runner.prepare_round(suite, "dev", "r1", _fi(tmp_path), max_parallel=2,
                                git=lambda p: {"sha": "abc", "short": "abc", "dirty": False})
    procs: list[_Proc] = []
    peak = []

    def spawn(cmd: list[str], *, cwd: Path, env: dict[str, str], stdout: Path, stderr: Path) -> _Proc:
        p = _Proc(cmd, cwd, env)
        procs.append(p)
        return p

    def sleep(_s: float) -> None:  # one running topic ends at every tick
        live = [p for p in procs if p.code is None]
        peak.append(len(live))
        if live:
            live[0].code = 0

    m = runner.drive(suite, rdir, spawn=spawn, alive=lambda pid: False, sleep=sleep, interval=0)
    assert len(procs) == 3 and max(peak) <= 2
    assert all(e["state"] == "done" and e["exit_code"] == 0 and e["pid"] and e["started_at"] and e["ended_at"]
               for e in m["topics"].values())
    first = procs[0]
    assert first.cmd[1:3] == ["launch.py", "--config"] and first.cwd == _fi(tmp_path).resolve()
    assert first.env["AXON_STORE_BASE"].endswith("axon_store") and first.env["FI_AXON_PROJECT"] == "suite-r1-alpha"
    assert first.env["OLLAMA_API_KEY"] == "secret-key-123" and first.env["OTHER"] == "quoted"
    # the key is passed to the process only: never written to the manifest or to the status text
    saved = (rdir / runner.MANIFEST).read_text(encoding="utf-8")
    assert "secret-key-123" not in saved and "secret-key-123" not in runner.render_status(m)
    assert len({p.env["AXON_STORE_BASE"] for p in procs}) == 3  # independent stores


def test_status_follows_pids_and_wait_times_out(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    rdir = runner.prepare_round(suite, "dev", "r1", _fi(tmp_path), max_parallel=3, git=lambda p: {"sha": "a", "short": "a", "dirty": None})
    runner.drive(suite, rdir, spawn=lambda cmd, **kw: _Proc(cmd, kw["cwd"], kw["env"]), alive=lambda pid: True,
                 sleep=lambda s: None, max_loops=1)
    assert all(e["state"] == "running" for e in runner.status(rdir, alive=lambda pid: True)["topics"].values())
    done, _ = runner.wait(rdir, alive=lambda pid: True, sleep=lambda s: None, interval=0, timeout=0)
    assert done is False
    done, m = runner.wait(rdir, alive=lambda pid: False, sleep=lambda s: None, interval=0, timeout=5)
    assert done is True and all(e["state"] == "done" for e in m["topics"].values())


def test_a_launch_that_fails_ends_the_topic_not_the_round(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    rdir = runner.prepare_round(suite, "dev", "r1", _fi(tmp_path), max_parallel=3, git=lambda p: {"sha": "a", "short": "a", "dirty": None})

    def spawn(cmd: list[str], **kw: Any) -> _Proc:
        raise FileNotFoundError("no python")

    m = runner.drive(suite, rdir, spawn=spawn, alive=lambda pid: False, sleep=lambda s: None)
    assert {e["state"] for e in m["topics"].values()} == {"failed_to_start"}


def test_parse_env_file(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    _w(f, "A=1\n#x\nB = \"two words\"\nbad line\nexport C=3\n")
    assert runner.parse_env_file(f) == {"A": "1", "B": "two words", "C": "3"}
    assert runner.parse_env_file(tmp_path / "missing") == {}


# ---------------------------------------------------------------------------- collector over fake quests

def _quest(root: Path, kind: str, **extra: Any) -> Path:
    q = root / f"1790000000-{kind}-abc123"
    _w(q / ".fi" / "audit.jsonl", json.dumps({"seq": 1, "ts": "2026-01-01T00:00:00+00:00", "kind": "quest_started", "quest_id": q.name}) + "\n")
    _w(q / ".fi" / "model_calls.jsonl", json.dumps({"node": "plan", "provider": "p", "served_model": "m", "requested_model": "m",
                                                   "outcome": "ok", "usage": {"prompt_tokens": 100, "completion_tokens": 40}}) + "\n")
    _w(q / "needs" / "x.json", "{}")
    if kind == "paper":
        _w(q / "paper" / "paper.md", "# paper\n")
    elif kind == "stop":
        _w(q / "needs" / "STUCK.json", {"kind": "no_findings", "problem": "the experiment could not run to the end after repairs",
                                         "tried": ["asked the model to repair the script when it failed (3 repairs)"], "say": "FI stopped"})
        _w(q / ".fi" / "audit.jsonl", json.dumps({"seq": 1, "ts": "2026-01-01T00:00:00+00:00", "kind": "node_started", "node": "execute_reflect"}) + "\n")
    elif kind == "crash":
        _w(q / "quest_failed.md", "# stopped\n\n**What happened:** The model takes longer to answer than the time limit.\n"
                                  "**Failing node:** `implement`\n**Kind of failure:** setup\n\n```\nhttpx.ReadTimeout\n```\n")
    elif kind == "pause":
        _w(q / ".fi" / "pause.json", {"kind": "plan", "headline": "the settings to vary could not be read", "decide": "Is the plan right?"})
    for rel, data in extra.items():
        _w(q / rel.replace("__", "/"), data)
    return q


def _round(tmp_path: Path, kinds: dict[str, str], **extra: Any) -> tuple[Path, Suite]:
    suite = _suite(tmp_path)
    seed = {"rounds": {"r1": {"fi_sha": "deadbeef", "topics": {}}}}
    for topic, kind in kinds.items():
        out = tmp_path / "runs" / topic / "out"
        _quest(out, kind, **extra.get(topic, {}))
        seed["rounds"]["r1"]["topics"][topic] = {"quest_root": str(out)}
    seed_file = tmp_path / "seed.yaml"
    _w(seed_file, yaml.safe_dump(seed))
    return adopt.adopt_file(suite, seed_file)[0], suite


def test_collector_sorts_outcomes_and_counts_the_suite_metrics(tmp_path: Path) -> None:
    rdir, _ = _round(tmp_path, {"a": "paper", "b": "stop", "c": "crash", "d": "pause"})
    data = collect.collect_round(rdir)
    by = {t["topic"]: t for t in data["topics"]}
    assert {k: v["outcome"] for k, v in by.items()} == {"a": "paper", "b": "honest_stop", "c": "crash", "d": "pause"}
    m = data["metrics"]
    assert (m["papers"], m["honest_stops"], m["crashes"], m["pauses_needing_a_person"], m["topics"]) == (1, 1, 1, 1, 4)
    assert by["a"]["paper_status"] == "ungraded" and by["a"]["class"] is None
    assert by["b"]["class"]["class"] == "model_capability" and by["b"]["stage"] == "run"
    assert by["c"]["class"]["class"] == "provider_infra"
    assert by["d"]["class"]["class"] == "spec_plan_gap"
    assert by["a"]["prompt_tokens"] == 100 and by["a"]["models"]
    assert (rdir / "round.json").is_file() and "Suite metrics" in (rdir / "round.md").read_text(encoding="utf-8")
    assert data["fi_git"]["short"] == "deadbeef"


def test_collector_reads_a_stop_for_a_manifest_mismatch_as_integration(tmp_path: Path) -> None:
    rdir, _ = _round(tmp_path, {"a": "stop"}, a={".fi__run_manifest_stop.json": {"problems": ["the manifest reports 80 trials but the data has 17"]}})
    t = collect.collect_round(rdir)["topics"][0]
    assert t["class"]["class"] == "integration_contract" and t["class"]["rule"].startswith("R3")
    assert "80 trials" in t["class"]["evidence"]


def test_collector_does_not_change_the_quest_and_survives_a_missing_quest(tmp_path: Path) -> None:
    rdir, suite = _round(tmp_path, {"a": "paper"})
    q = next((tmp_path / "runs" / "a" / "out").iterdir())
    before = sorted(str(p.relative_to(q)) for p in q.rglob("*"))
    collect.collect_round(rdir)
    assert sorted(str(p.relative_to(q)) for p in q.rglob("*")) == before
    m = runner.load_manifest(rdir)
    m["topics"]["ghost"] = {"config": None, "dir": str(tmp_path / "g"), "quest_root": str(tmp_path / "g" / "out"), "state": "done",
                            "pid": 1, "started_at": None, "ended_at": None, "exit_code": 1}
    runner.save_manifest(rdir, m)
    ghost = next(t for t in collect.collect_round(rdir)["topics"] if t["topic"] == "ghost")
    assert ghost["outcome"] == "crash" and ghost["class"]["class"] == "unclassified"


def test_old_code_review_record_without_final_read_is_judged_from_its_versions(tmp_path: Path) -> None:
    rdir, _ = _round(tmp_path, {"a": "stop", "b": "stop"},
                     a={".fi__code_review.json": {"version": "v1", "read_version": "v1", "readings": [{"n": 1, "open": 2, "high": 0}]}},
                     b={".fi__code_review.json": {"version": "v2", "read_version": "v1", "readings": [{"n": 1, "open": 2, "high": 1}]}})
    by = {t["topic"]: t for t in collect.collect_round(rdir)["topics"]}
    assert by["a"]["code"]["final_read"] is True and by["b"]["code"]["final_read"] is False


# ------------------------------------------------------------------------------------------------ taxonomy

def _facts(**kw: Any) -> dict[str, Any]:
    base = {"outcome": "honest_stop", "stage": "", "chain": [], "headline": "h", "stuck": {}, "pause": {}, "failed_md": "",
            "run_manifest_stop": "", "typed_results_used": 0, "static_fired": {}, "evidence_gaps": [], "evidence_flaw": False,
            "code": {}, "coverage_left": [], "grade": {}}
    return {**base, **kw}


def test_each_taxonomy_rule_fires_on_its_record_and_names_it() -> None:
    cases = [
        (_facts(outcome="crash", stage="code", failed_md="**Kind of failure:** setup\nhttpx.ReadTimeout"), "provider_infra", "R1"),
        (_facts(outcome="crash", failed_md="HTTP 429 rate limit"), "provider_infra", "R1"),
        (_facts(outcome="pause", pause={"kind": "plan", "headline": "no grid"}), "spec_plan_gap", "R2"),
        (_facts(outcome="honest_stop", stage="plan", stuck={"problem": "x"}), "spec_plan_gap", "R2"),
        (_facts(run_manifest_stop="80 vs 17", stage="analysis"), "integration_contract", "R3"),
        (_facts(stage="run", stuck={"problem": "the experiment could not run to the end"}), "model_capability", "R4"),
        (_facts(outcome="paper", typed_results_used=1), "spec_gaming", "R5"),
        (_facts(outcome="paper", stage="code", chain=["equation test not fixed"]), "gate_bypassed", "R6"),
        (_facts(outcome="paper", evidence_flaw=True, evidence_gaps=["the claim check did not pass: 2 unsupported"]), "gate_bypassed", "R7"),
        (_facts(outcome="paper", grade={"g1": "fail", "g2": "pass", "g3": "pass"}), "no_gate", "R8"),
        (_facts(outcome="honest_stop", grade={"class": "fi_bug", "stage": "checks", "class_note": "grader"}), "fi_bug", "R0"),
    ]
    for facts, cls, rule in cases:
        got = taxonomy.classify(facts)
        assert got and got["class"] == cls and got["rule"].startswith(rule), (cls, got)
        assert got["evidence"]


def test_nothing_is_guessed_and_a_clean_paper_is_not_a_failure() -> None:
    got = taxonomy.classify(_facts(outcome="honest_stop", stage="design", stuck={"problem": "something odd"}))
    assert got["class"] == "unclassified" and got["stage"] == "design"
    assert taxonomy.classify(_facts(outcome="paper")) is None
    clean = {"g1": "pass", "g2": "pass", "g3": "pass"}
    assert taxonomy.classify(_facts(outcome="paper", grade=clean, evidence_flaw=True, evidence_gaps=["the claim check did not pass"])) is None
    assert taxonomy.classify(_facts(outcome="running")) is None
    # a rule never infers an FI bug
    assert all(taxonomy.classify(f)["class"] != "fi_bug" for f in (_facts(outcome="crash", failed_md="Traceback KeyError"),))


def test_contributing_flags_are_listed() -> None:
    f = _facts(outcome="pause", pause={"kind": "plan"}, static_fired={"grid_keys": 2}, coverage_left=["x"],
               code={"final_version_read": False, "repairs_converged": False},
               evidence_gaps=["FI went on by itself although the known-answer check 'chk' failed (expected 1)"])
    also = " | ".join(taxonomy.classify(f)["also"])
    for needle in ("known-answer check 'chk'", "not read", "did not converge", "grid_keys", "left uncovered"):
        assert needle in also


# --------------------------------------------------------------------------------------------- trend / pareto

def _m(**kw: int) -> dict[str, int]:
    base = {"topics": 0, "papers": 0, "honest_stops": 0, "pauses_needing_a_person": 0, "crashes": 0, "unfinished": 0,
            "papers_trustworthy": 0, "papers_fake": 0, "papers_flawed": 0, "papers_ungraded": 0, "stops_false": 0,
            "stops_correct": 0, "stops_unjudged": 0}
    return {**base, **kw}


def test_criteria_are_decided_only_when_the_data_decide_them() -> None:
    assert trend.criteria(_m(topics=0)) == {"trustworthy": "not yet known", "fake": "not yet known", "false_stops": "not yet known"}
    # no papers at all: nothing is trustworthy, and a stop nobody has judged is not "0 false stops"
    c = trend.criteria(_m(topics=4, honest_stops=2, pauses_needing_a_person=2, stops_unjudged=2))
    assert c["trustworthy"] == "not met" and c["false_stops"] == "not yet known"
    # ungraded papers keep a "0 fake" open
    c = trend.criteria(_m(topics=4, papers=4, papers_ungraded=4))
    assert c == {"trustworthy": "not yet known", "fake": "not yet known", "false_stops": "met"}
    c = trend.criteria(_m(topics=10, papers=8, papers_trustworthy=7, stops_correct=2))
    assert c == {"trustworthy": "met", "fake": "met", "false_stops": "met"}
    c = trend.criteria(_m(topics=10, papers=5, papers_trustworthy=4, papers_fake=1, honest_stops=2, stops_false=2))
    assert c == {"trustworthy": "not met", "fake": "not met", "false_stops": "not met"}
    # a topic still running can turn a met into unknown
    assert trend.criteria(_m(topics=10, papers_trustworthy=6, unfinished=2))["trustworthy"] == "not yet known"


def test_paper_status_and_stop_status() -> None:
    ok = {"g1": "pass", "g2": "pass", "g3": "pass"}
    assert collect.paper_status("paper", ok, "") == "trustworthy"
    assert collect.paper_status("paper", ok, "the experiment was not executed") == "fake"
    assert collect.paper_status("paper", {"g1": "fail (narrow)", "g2": "pass", "g3": "pass"}, "") == "fake"
    assert collect.paper_status("paper", {"g1": "pass", "g2": "fail", "g3": "pass"}, "") == "flawed"
    assert collect.paper_status("paper", None, "") == "ungraded"
    assert collect.paper_status("crash", ok, "") is None
    assert collect.stop_status("honest_stop", {"stop_correct": False}, None) == "false"
    assert collect.stop_status("honest_stop", {"class": "fi_bug"}, None) == "false"
    assert collect.stop_status("honest_stop", {"stop_correct": True}, None) == "correct"
    assert collect.stop_status("honest_stop", None, None) == "unjudged"
    assert collect.stop_status("pause", None, None) is None


def test_trend_and_pareto_from_collected_rounds(tmp_path: Path) -> None:
    rdir, suite = _round(tmp_path, {"a": "paper", "b": "stop", "c": "crash"})
    collect.collect_round(rdir)
    grade_report = tmp_path / "ga.md"
    _w(grade_report, "| run | date | build | model | D1 | D2 | D3 | D4 | D5 | D6 | Mean | G1 | G2 | G3 | pages | tok | note |\n"
                     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
                     "| a | d | b | m | 100 | 100 | N/A | 100 | 90 | 100 | 98.0 | pass | pass | pass | 4 | 1 | fine |\n")
    grade.import_grade(rdir, "a", grade_report)
    grade.import_grade(rdir, "b", _w_and(tmp_path / "gb.md", "STOP: wrong\nthe cause was untrue\n"))
    data = collect.collect_round(rdir)
    m = data["metrics"]
    assert (m["papers_trustworthy"], m["stops_false"], m["crashes"]) == (1, 1, 1)
    trend.write_reports(suite.rounds_dir)
    t = (suite.rounds_dir / "trend.md").read_text(encoding="utf-8")
    assert "| r1 |" in t and "deadbee" in t and "1/3 (33%)" in t
    p = (suite.rounds_dir / "pareto.md").read_text(encoding="utf-8")
    assert "provider / infrastructure" in p and "Stage by cause class" in p and "r1/c" in p
    row = trend.trend_row(data)
    assert row["criteria"]["trustworthy"] == "not met" and row["criteria"]["false_stops"] == "not met"


def _w_and(path: Path, text: str) -> Path:
    _w(path, text)
    return path


def test_pareto_with_no_failures() -> None:
    assert "No failures recorded yet" in trend.render_pareto([])


# ------------------------------------------------------------------------------------------------- grading

V4 = ("| run_a | 2026-10-06 | fix/x 497fc38b | gemma4:31b, thinking on | 100 | 100 | N/A | 100 | 63 | 100 | 92.6 | fail (narrow) | pass | pass | 4 | "
      "386,514 / 46 | all sweep numbers right |")
V5 = ("| run_b | 2026-10-06 | main faecf3be (#533) | gemma + codex | 78* | 67* | N/A | 0 | 17 | 83 | 49.0 (over 5) | fail | fail | fail | 4 | "
      "1,068,066 / 83 | no model |")


def test_grade_row_is_read_from_a_report() -> None:
    g = grade.parse_row("| run | date |\n|---|---|\n" + V4 + "\n")
    assert (g["g1"], g["g2"], g["g3"], g["mean"]) == ("fail (narrow)", "pass", "pass", 92.6)
    assert g["d"]["D1"] == "100" and g["d"]["D3"] is None and g["d"]["D5"] == "63"
    g5 = grade.parse_row(V5)
    assert g5["mean"] == 49.0 and g5["d"]["D1"] == "78" and g5["g3"] == "fail"
    assert grade.parse_row("no table here") == {}


def test_grade_import_needs_something_to_read(tmp_path: Path) -> None:
    f = _w_and(tmp_path / "x.md", "nothing\n")
    with pytest.raises(ValueError):
        grade.import_grade(tmp_path / "r", "t", f)
    g = grade.import_grade(tmp_path / "r", "t", _w_and(tmp_path / "y.md", "STOP: correct\n"), cause_class="fi_bug", stage="checks", note="n")
    assert g["stop_correct"] is True and g["class"] == "fi_bug"


def test_grader_prompt_carries_the_topics_own_goals_and_no_topic_words(tmp_path: Path) -> None:
    rdir, suite = _round(tmp_path, {"alpha": "paper", "beta": "stop"})
    collect.collect_round(rdir)
    paths = grade.write_templates(rdir, {t: suite.topic_text(t) for t in suite.topic_files})
    assert [p.parent.name for p in paths] == ["alpha"]  # only a delivered paper is graded
    text = paths[0].read_text(encoding="utf-8")
    assert "alpha question" in text and "SCOPE. one thing." in text and "rubric.md" in text
    assert "G1" in text and "STOP:" in text and "{" not in text.replace("{}", "")


# --------------------------------------------------------------------------------------------- the CLI

def test_cli_collect_and_report_on_an_adopted_round(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rdir, suite = _round(tmp_path, {"a": "paper"})
    assert cli.main(["collect", "--suite", str(suite.path), "--round", "r1"]) == 0
    assert "1 paper(s)" in capsys.readouterr().out
    assert (suite.rounds_dir / "trend.md").is_file() and (suite.rounds_dir / "pareto.md").is_file()
    assert cli.main(["status", "--suite", str(suite.path), "--round", "r1"]) == 0
    assert cli.main(["status", "--suite", str(suite.path), "--round", "nope"]) == 1


def test_detached_driver_is_started_with_the_driver_command(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def spawn(cmd: list[str], **kw: Any) -> _Proc:
        seen["cmd"], seen["cwd"] = cmd, kw["cwd"]
        return _Proc(cmd, kw["cwd"], kw["env"])

    (tmp_path / "round").mkdir()
    runner.start_detached_driver(tmp_path / "suite.yaml", tmp_path / "round", spawn=spawn)
    assert seen["cmd"][1:4] == ["-m", "dev.evaluation.suite", "drive"] and (seen["cwd"] / "dev" / "evaluation" / "suite").is_dir()
    assert time.time() > 0


def test_a_paper_with_a_missing_grade_is_undecided_not_flawed() -> None:
    # G1 passes and G2/G3 are not graded yet: nothing has failed, so it is neither trustworthy nor flawed
    assert collect.paper_status("paper", {"g1": "pass"}, "") == "ungraded"
    assert collect.paper_status("paper", {"g1": "pass", "g2": "pass"}, "") == "ungraded"
    assert collect.paper_status("paper", {"g1": "pass", "g2": "", "g3": None}, "") == "ungraded"
    assert collect.paper_status("paper", {"g2": "pass", "g3": "pass"}, "") == "ungraded"
    # a grade that is there and fails (or is anything but a pass) is a flaw, even with another grade still missing
    assert collect.paper_status("paper", {"g1": "pass", "g2": "fail (x)"}, "") == "flawed"
    assert collect.paper_status("paper", {"g1": "pass", "g3": "partial"}, "") == "flawed"
    assert collect.paper_status("paper", {"g2": "fail"}, "") == "flawed"
    assert collect.paper_status("paper", {"g1": "fail"}, "") == "fake"


def test_the_criterion_is_not_reported_not_met_before_the_missing_grades_decide_it() -> None:
    def topics(partial: int, flawed: int) -> list[dict[str, Any]]:
        rows = [{"outcome": "paper", "paper_status": "trustworthy"} for _ in range(5)]
        rows += [{"outcome": "paper", "paper_status": collect.paper_status("paper", {"g1": "pass"}, "")} for _ in range(partial)]
        rows += [{"outcome": "paper", "paper_status": collect.paper_status("paper", {"g1": "pass", "g2": "fail"}, "")}
                 for _ in range(flawed)]
        rows += [{"outcome": "honest_stop", "stop_status": "correct"} for _ in range(10 - 5 - partial - flawed)]
        return rows

    # 5 trustworthy + 2 waiting on G2/G3 of 10: 70% is still reachable
    c = trend.criteria(collect.metrics(topics(2, 0)))
    assert c["trustworthy"] == "not yet known"
    # the same two papers actually failing G2: 70% can no longer be reached
    assert trend.criteria(collect.metrics(topics(0, 2)))["trustworthy"] == "not met"
    # the missing grades arrive and pass: met
    done = [{"outcome": "paper", "paper_status": "trustworthy"} for _ in range(7)]
    done += [{"outcome": "honest_stop", "stop_status": "correct"} for _ in range(3)]
    assert trend.criteria(collect.metrics(done))["trustworthy"] == "met"
