"""The quality report (core/quality_report.py): QUALITY.md / .fi/quality.json read from a quest's own records, the rule-based
suggestions, robustness to missing or old files, and the cross-quest table (`fi tools quality`)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

import launch
from core import quality_report as qr


def _w(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")


def _call(node: str, model: str = "model-a", outcome: str = "ok", prompt: int = 100, completion: int = 50,
          provider: str = "prov") -> dict[str, Any]:
    return {"node": node, "provider": provider, "served_model": model, "requested_model": model, "outcome": outcome,
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def _calls(root: Path, rows: list[dict[str, Any]]) -> None:
    _w(root / ".fi" / "model_calls.jsonl", "\n".join(json.dumps(r) for r in rows) + "\n")


def _audit(root: Path, rows: list[dict[str, Any]]) -> None:
    out = []
    for i, r in enumerate(rows, 1):
        out.append(json.dumps({"seq": i, "ts": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00", **r}))
    _w(root / ".fi" / "audit.jsonl", "\n".join(out) + "\n")


def _node(name: str, start: int, end: int) -> list[dict[str, Any]]:
    return [{"kind": "node_started", "node": name, "ts": f"2026-01-01T00:00:{start:02d}+00:00"},
            {"kind": "node_completed", "node": name, "ts": f"2026-01-01T00:00:{end:02d}+00:00"}]


def _finished(root: Path) -> Path:
    """A finished quest with one problem fixed at the checks and one review round."""
    _calls(root, [_call("plan"), _call("oracle_review", "model-b"), _call("design"), _call("design_self_critique"),
                  _call("implement"), _call("analyze"), _call("cross_check", "model-b"), _call("write"),
                  _call("claim_check", "model-b"), _call("review", "model-b")])
    _w(root / ".fi" / "oracle_review.json", {"reviewer": "model-b", "planner": "model-a", "same_model": False,
                                              "written": True, "findings": ["no check tests equation E2"]})
    _w(root / ".fi" / "cost.summary.json", {"total_requests": 10, "total_cost_usd": 0.5})
    _w(root / "paper" / "paper.md", "# A paper\n")
    _w(root / "needs" / "EVIDENCE.json", {"status": "executed", "gaps": []})
    _audit(root, [{"kind": "quest_started", "quest_id": "q-fin"}, *_node("plan", 1, 5), *_node("design", 6, 9)])
    return root


def _stage(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(s for s in report["stages"] if s["stage"] == name)


def test_finished_quest_rows_and_cost(tmp_path: Path) -> None:
    root = _finished(tmp_path / "q-fin")
    r = qr.build(root)
    assert r["outcome"]["kind"] == "finished"
    assert [s["stage"] for s in r["stages"]] == list(qr.STAGES)
    checks = _stage(r, "checks")
    assert checks["reviewed_by"] == "model-b" and checks["same_model"] is False
    assert checks["problems"]["total"] == 1 and checks["problems"]["fixed"] == 1
    assert _stage(r, "plan")["seconds"] == 4
    assert _stage(r, "design")["wrote"] == ["prov/model-a"]
    assert r["failure"] is None
    assert r["cost"]["calls"] == 10 and r["cost"]["usd"] == 0.5
    assert r["cost"]["prompt_tokens"] == 1000 and r["cost"]["completion_tokens"] == 500
    assert r["problems"]["fixed_rate"] == 1.0


def test_same_model_reviewer_is_named_and_suggested(tmp_path: Path) -> None:
    root = tmp_path / "q-same"
    _calls(root, [_call("plan"), _call("oracle_review"), _call("design"), _call("design_self_critique")])
    _w(root / ".fi" / "oracle_review.json", {"reviewer": "model-a", "planner": "model-a", "same_model": True,
                                              "written": False, "findings": ["the check may not test the model"]})
    _w(root / "paper" / "paper.md", "x")
    r = qr.build(root)
    assert _stage(r, "checks")["same_model"] is True
    assert _stage(r, "design")["same_model"] is True
    texts = " ".join(s["text"] for s in r["suggestions"])
    assert "provider.node_providers.oracle_review" in texts
    assert all(s["source"] for s in r["suggestions"])
    assert "(same model)" in qr.render(r)


def test_honest_stop_failure_chain_and_suggestion(tmp_path: Path) -> None:
    root = tmp_path / "q-stop"
    _calls(root, [_call("implement"), _call("execute_reflect"), _call("execute_reflect")])
    _w(root / "needs" / "STUCK.json", {
        "kind": "no_findings", "problem": "the experiment could not run to the end, even after FI repaired the script "
        "2 times, so FI has nothing measured to write up", "tried": ["ran the experiment: it stopped with an error"]})
    _audit(root, [{"kind": "route_decision", "node": "execute_reflect", "chosen": "retry"},
                  {"kind": "route_decision", "node": "execute_reflect", "chosen": "retry"}])
    r = qr.build(root)
    assert r["outcome"]["kind"] == "honest_stop"
    assert r["failure"]["stage"] == "run"
    assert r["failure"]["chain"][0].startswith("FI tried:")
    assert "the script stopped with an error" in " ".join(r["failure"]["chain"])
    run = _stage(r, "run")
    assert run["problems"]["not_fixed"] == 1 and run["problems"]["fixed"] == 1
    assert any("provider.node_models" in s["text"] for s in r["suggestions"])


def test_known_answer_check_that_keeps_failing(tmp_path: Path) -> None:
    root = tmp_path / "q-oracle"
    bad = ["the oracle 'limit_case' failed: FI measured 3, the protocol expects 1 within 0.01 (absolute tolerance 0.01)"]
    _audit(root, [{"kind": "check_result", "check": "oracle", "status": "stopped", "problems": bad},
                  {"kind": "check_result", "check": "oracle", "status": "stopped", "problems": bad}])
    _w(root / ".fi" / "pause.json", {"kind": "oracle", "headline": "the script has not passed its known-answer checks",
                                      "decide": "Which side is wrong?", "recommended": "Look at the formula."})
    r = qr.build(root)
    assert r["outcome"]["kind"] == "paused" and r["outcome"]["needs_person"] is True
    assert r["failure"]["stage"] == "run"
    items = _stage(r, "run")["problems"]["items"]
    assert items[0]["status"] == "not fixed" and "failed 2 times" in items[0]["text"]
    texts = [s["text"] for s in r["suggestions"]]
    assert "Look at the formula." not in texts  # advice to look at code is not quoted (people do not debug)
    assert any("kept failing" in t for t in texts)


def test_check_fixed_by_a_later_attempt(tmp_path: Path) -> None:
    root = tmp_path / "q-fixed"
    bad = ["the oracle 'limit_case' failed: FI measured 3, the protocol expects 1 within 0.01 (absolute tolerance 0.01)"]
    _audit(root, [{"kind": "check_result", "check": "oracle", "status": "repairing", "problems": bad},
                  {"kind": "check_result", "check": "oracle", "status": "ok", "problems": []}])
    _w(root / "paper" / "paper.md", "x")
    assert _stage(qr.build(root), "run")["problems"]["fixed"] == 1


def test_a_reading_of_the_code_that_could_not_decide_is_listed_as_a_note_not_a_stop(tmp_path: Path) -> None:
    root = _finished(tmp_path / "q-note")
    note = "FI could not tell from the code whether the settings of `cell` (simulate.py, line 6) draw their own random numbers"
    _w(root / "needs" / "PROTOCOL_CHECK.json", {"status": "ok", "attempts": [], "differences": [], "notes": [note]})
    checks = _stage(qr.build(root), "checks")["problems"]
    assert checks["left_out"] == 1 and checks["not_fixed"] == 0
    qr.write(root)
    assert "could not tell from the code" in (root / "QUALITY.md").read_text(encoding="utf-8")


def test_a_precision_target_no_run_could_judge_is_said(tmp_path: Path) -> None:
    """A study that ran each setting once has no interval; the plan's precision target was then judged by nothing, and
    the report said nothing about it. The fact is the precision check's own, kept in the evidence record."""
    root = _finished(tmp_path / "q-once")
    _w(root / "needs" / "EVIDENCE.json", {"status": "executed",
                                          "precision_not_judged": {"target": 0.05, "metric": "peak_width"}})
    run = _stage(qr.build(root), "run")["problems"]
    assert run["left_out"] == 1 and run["not_fixed"] == 0
    assert "`peak_width`" in run["items"][0]["text"] and "was not judged" in run["items"][0]["text"]
    _w(root / "needs" / "EVIDENCE.json", {"status": "executed"})  # the target was held to an interval
    assert _stage(qr.build(root), "run")["problems"]["left_out"] == 0
    for bad in ({"target": "0.05"}, {"target": True}, {"target": float("inf")}, {"target": 10 ** 400}, "not a record", {}):
        _w(root / "needs" / "EVIDENCE.json", {"status": "executed", "precision_not_judged": bad})
        assert _stage(qr.build(root), "run")["problems"]["left_out"] == 0, bad
    _w(root / "needs" / "EVIDENCE.json", "{not json")
    assert _stage(qr.build(root), "run")["problems"]["left_out"] == 0


def test_crash_and_timeout_suggestions(tmp_path: Path) -> None:
    root = tmp_path / "q-crash"
    _calls(root, [_call("implement", outcome="ReadTimeout"), _call("implement", outcome="ReadTimeout"),
                  _call("implement", outcome="ReadTimeout")])
    _w(root / "quest_failed.md", "# The quest stopped\n\n**What happened:** The model takes longer to answer the step "
       "`implement` than the time limit.\n\n**What to do:** Raise `provider.http_timeout_s`.\n\n"
       "**Failing node:** `implement`\n")
    r = qr.build(root)
    assert r["outcome"]["kind"] == "crashed" and "takes longer" in r["outcome"]["line"]
    assert r["failure"]["stage"] == "code"
    code = _stage(r, "code")
    assert code["failed_calls"] == 3 and code["problems"]["not_fixed"] == 1
    texts = [s["text"] for s in r["suggestions"]]
    assert texts[0] == "Raise `provider.http_timeout_s`."
    assert any("timed out 3 times" in t and "node_http_timeout_s" in t for t in texts)


def test_review_rounds_and_unfixed_last_round(tmp_path: Path) -> None:
    root = tmp_path / "q-review"
    hit = {"must_flag_hits": ["unsupported_claim"], "verdict": "revise"}
    _audit(root, [{"kind": "route_decision", "node": "review", "chosen": "rewrite", "facts": hit},
                  {"kind": "route_decision", "node": "review", "chosen": "done", "facts": hit}])
    _w(root / "paper" / "paper.md", "x")
    r = qr.build(root)
    review = _stage(r, "review")["problems"]
    assert review["total"] == 2 and review["not_fixed"] == 2
    assert r["failure"]["stage"] == "review"


def test_code_review_skipped_names_the_same_model(tmp_path: Path) -> None:
    root = tmp_path / "q-code"
    _calls(root, [_call("implement", "model-a")])
    _w(root / ".fi" / "code_review.json", {"skipped": "the reader is the model that wrote the code (model-a)"})
    _w(root / ".fi" / "function_steps.json", {"functions": {
        "f1": {"equation": "E1", "status": "failed", "attempts": 3, "error": "TypeError"},
        "f2": {"equation": "E2", "status": "done", "attempts": 2}}})
    _w(root / "paper" / "paper.md", "x")
    code = _stage(qr.build(root), "code")
    assert code["same_model"] is True and code["reviewed_by"] == "model-a"
    assert code["problems"]["not_fixed"] == 1 and code["problems"]["fixed"] == 1


def test_missing_and_old_format_files_do_not_break(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    r = qr.build(empty)
    assert r["outcome"]["kind"] == "unknown" and r["problems"]["total"] == 0
    assert "Quality report" in qr.render(r)
    # Garbage and old shapes: a list where a dict is expected, text where JSON is expected, a truncated line.
    bad = tmp_path / "bad"
    _w(bad / ".fi" / "oracle_review.json", ["not", "a", "dict"])
    _w(bad / ".fi" / "model_calls.jsonl", '{"node": "plan"}\nnot json\n{"node":\n')
    _w(bad / ".fi" / "audit.jsonl", "garbage\n")
    _w(bad / ".fi" / "equation_tests.json", {"failed": "oops", "equations": []})
    _w(bad / "needs" / "DESIGN_CRITIQUE.json", {"not": "a list"})
    _w(bad / "needs" / "EVIDENCE.json", "{")
    _w(bad / "frontier_insight_summary.json", [])
    r = qr.build(bad)
    assert _stage(r, "plan")["calls"] == 1
    qr.render(r)


def test_write_puts_files_at_root_or_out_dir_and_never_raises(tmp_path: Path) -> None:
    root = _finished(tmp_path / "q-w")
    qr.write(root)
    assert (root / "QUALITY.md").is_file()
    stored = json.loads((root / ".fi" / "quality.json").read_text(encoding="utf-8"))
    assert stored["schema"] == qr.SCHEMA and stored["quest_id"] == "q-fin"
    out = tmp_path / "scratch"
    other = _finished(tmp_path / "q-w2")
    before = sorted(p.name for p in other.rglob("*"))
    qr.write(other, out)
    assert (out / "QUALITY.md").is_file() and (out / "quality.json").is_file()
    assert sorted(p.name for p in other.rglob("*")) == before  # the quest is untouched
    qr.write_quietly(tmp_path / "no-such-folder" / "x")  # must not raise


def test_engine_run_ends_by_writing_the_report() -> None:
    src = (Path(__file__).resolve().parent.parent / "core" / "engine.py").read_text(encoding="utf-8")
    assert "_quality_report.write_quietly(self.quest_root)" in src
    # The hook sits in run()'s outermost finally, before the quest logger is closed.
    assert src.index("_quality_report.write_quietly(self.quest_root)") < src.index("_close_quest_logger(self.quest_id)",
                                                                                  src.index("_quality_report.write_quietly"))


def test_table_over_several_quests(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    a = _finished(runs / "a" / "out" / "q-a")
    b = runs / "b" / "out" / "q-b"
    _calls(b, [_call("implement", "model-x", outcome="ReadTimeout"), _call("implement", "model-x", outcome="ReadTimeout")])
    _w(b / "quest_failed.md", "**What happened:** it timed out.\n**Failing node:** `implement`\n")
    _w(b / "config.yaml", "provider:\n  name: prov\n  model: model-x\n  node_models:\n    design: model-y\n")
    qr.write(a)  # one has a stored report, the other is built from its records
    rows, md, csv_text = qr.table([runs])
    assert {r["outcome"] for r in rows} == {"finished", "crashed"}
    row_b = next(r for r in rows if r["outcome"] == "crashed")
    assert row_b["failing_stage"] == "code" and row_b["main_model"] == "prov/model-x"
    assert row_b["routed_steps"] == "design=model-y"
    parsed = list(csv.DictReader(csv_text.splitlines()))
    assert len(parsed) == 2 and "problems_checks" in parsed[0]
    assert md.count("\n") == 4
    qr.write_table([runs], tmp_path / "out")
    assert (tmp_path / "out" / "quality_table.md").is_file() and (tmp_path / "out" / "quality_table.csv").is_file()


def test_cli_table_and_single_quest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = tmp_path / "runs"
    _finished(runs / "q-one")
    _finished(runs / "q-two")
    out = tmp_path / "tbl"
    assert launch._show_quality([str(runs)], tmp_path, str(out)) == 0
    assert "q-one" in (out / "quality_table.md").read_text(encoding="utf-8") or "q-fin" in capsys.readouterr().out
    assert launch._show_quality([str(runs / "q-one")], tmp_path, str(tmp_path / "one")) == 0
    assert "# Quality report" in capsys.readouterr().out
    assert (tmp_path / "one" / "QUALITY.md").is_file()
    assert launch._show_quality([str(tmp_path / "nothing-here")], tmp_path, None) == 1


def test_tools_subcommand_is_registered() -> None:
    assert launch._expand_tools_argv(["tools", "quality", "a", "b"]) == ["--quality", "a", "b"]


def test_quest_dirs_finds_nested_quests(tmp_path: Path) -> None:
    _finished(tmp_path / "x" / "out" / "q1")
    _finished(tmp_path / "y" / "q2")
    names = sorted(p.name for p in qr.quest_dirs([tmp_path]))
    assert names == ["q1", "q2"]


def test_advice_to_fix_code_or_files_is_never_quoted(tmp_path: Path) -> None:
    root = tmp_path / "q-advice"
    _w(root / ".fi" / "pause.json", {"kind": "manifest", "headline": "h", "recommended": "Fix the script as listed, then go on."})
    assert not any("Fix the script" in s["text"] for s in qr.build(root)["suggestions"])
    _w(root / ".fi" / "pause.json", {"kind": "plan", "headline": "h", "recommended": "Try another model for the plan step."})
    assert qr.build(root)["suggestions"][0]["text"] == "Try another model for the plan step."
    assert qr._is_choice("Raise `provider.http_timeout_s`, or use a faster model.")
    assert not qr._is_choice("Edit plan.md and add the missing check.")


def test_every_suggested_key_routes_the_step_it_names(tmp_path: Path) -> None:
    """Each `provider.node_models.<step>` / `node_providers.<step>` names a step the engine really calls (`_chat(..., node=)`),
    and the engine's own key matching (exact, then the part before the first dot) finds it."""
    import re
    from core import provider as prov

    engine = (Path(__file__).resolve().parent.parent / "core" / "engine.py").read_text(encoding="utf-8")
    oracle_review = (Path(__file__).resolve().parent.parent / "core" / "oracle_review.py").read_text(encoding="utf-8")
    steps = set(re.findall(r'node="([a-z_]+)', engine)) | set(re.findall(r'^NODE = "([a-z_]+)"', oracle_review, re.M))
    root = tmp_path / "q-keys"
    _calls(root, [_call("implement", outcome="ReadTimeout")] * 3 + [_call("design", outcome="HTTPStatusError")] * 3
           + [_call(n) for n in ("plan", "oracle_review", "design", "design_review", "implement", "analyze",
                                 "cross_check", "write", "claim_check", "review", "execute_reflect")])
    _w(root / ".fi" / "oracle_review.json", {"reviewer": "model-a", "planner": "model-a", "same_model": True,
                                              "findings": ["x"]})
    _w(root / ".fi" / "code_review.json", {"skipped": "the reader is the model that wrote the code (model-a)"})
    keys = set()
    for sug in qr.build(root)["suggestions"]:
        keys |= set(re.findall(r"provider\.node_(?:models|providers)\.([a-z_]+)", sug["text"]))
    assert {"oracle_review", "implement", "design", "design_review", "code_review"} <= keys
    from core import review_pairs as pairs

    # A reviewer's own step is a key a person routes with; the engine calls it under that key when it is set (and under
    # the step it falls back to when it is not), so the key must resolve to a node the engine really calls.
    reviewer_steps = set(pairs.READER_STEP.values())
    for key in keys:
        assert key in steps or key in reviewer_steps, key
        if key in reviewer_steps:
            assert pairs.node_for(key, lambda step, key=key: step == key) == key
        assert prov.node_provider_for({key: 1}, key) == (key, 1)
        assert prov.model_for_node({key: "m"}, key + ".sub") == "m"
    for stage_key in qr._KEY.values():
        assert stage_key in steps, stage_key
    for stage, step in pairs.READER_STEP.items():
        assert step in steps or step in reviewer_steps, (stage, step)
        assert any(qr._reads(stage, n) for n in (step, pairs.FALLBACK_NODE.get(step, step))), (stage, step)


def test_the_same_model_on_two_providers_is_the_same_model(tmp_path: Path) -> None:
    """FI's own check (`oracle_review.same_model`) ignores the provider; so does the report."""
    root = tmp_path / "q-prov"
    _calls(root, [_call("design", "model-a", provider="prov-one"), _call("design_self_critique", "model-a", provider="prov-two")])
    _w(root / "paper" / "paper.md", "x")
    assert _stage(qr.build(root), "design")["same_model"] is True


def test_a_running_quest_is_not_shown_with_an_earlier_runs_outcome(tmp_path: Path) -> None:
    root = tmp_path / "q-run"
    _calls(root, [_call("plan")])
    _w(root / "needs" / "STUCK.json", {"kind": "no_findings", "problem": "an earlier run stopped"})
    _w(root / "paper" / "paper.md", "old paper")
    _audit(root, [{"kind": "quest_started", "quest_id": "q-run", "resumed": True}])
    _w(root / ".fi" / "run.log", "[run] working")  # fresh log: it is running
    first = qr.write(root, running=False)
    assert first["outcome"]["kind"] != "running" and first["written_at"] and first["running"] is False
    live = qr.build(root)  # detects the running quest
    assert live["outcome"]["kind"] == "running" and live["failure"] is None and live["running"] is True
    assert "previous run of" in live["outcome"]["line"] and "running now" in live["outcome"]["line"]
    assert live["run_started_at"]
    text = qr.render(live)
    assert "running now" in text and "for the run that started" in text
    assert qr.load(root)["outcome"]["kind"] == "running"  # the stored report is not presented as current
    _, md, _csv = qr.table([tmp_path])
    assert "running" in md
    # The engine's own call ends a run: it never writes "running".
    qr.write_quietly(root)
    assert json.loads((root / ".fi" / "quality.json").read_text(encoding="utf-8"))["outcome"]["kind"] != "running"
    # Stopped for a person: not running.
    _w(root / ".fi" / "pause.json", {"kind": "plan", "headline": "h"})
    assert qr.build(root)["outcome"]["kind"] == "paused"


def test_the_report_says_how_many_functions_were_written_and_tested(tmp_path: Path) -> None:
    root = tmp_path / "q-fn"
    _calls(root, [_call("implement", "model-a")])
    _w(root / ".fi" / "function_steps.json", {"status": "partial", "functions": {
        "f1": {"equation": "E1", "status": "done", "attempts": 1},
        "f2": {"equation": "E2", "status": "failed", "attempts": 3, "error": "TypeError"},
        "f3": {"equation": "E3", "status": "done", "attempts": 1}}})
    _w(root / "paper" / "paper.md", "x")
    r = qr.build(root)
    assert r["code"]["functions_done"] == 2 and r["code"]["functions_total"] == 3
    assert "Functions written and tested: 2 of 3." in qr.render(r)
    none = qr.build(tmp_path / "q-none")
    assert none["code"]["functions_total"] == 0 and "Functions written and tested" not in qr.render(none)


def test_the_report_says_whether_the_pdf_is_typeset_and_which_characters_changed(tmp_path: Path) -> None:
    root = _finished(tmp_path / "q")
    _w(root / "paper.pdf", "%PDF-1.5\n")
    _w(root / ".fi" / "paper_render.json", {
        "engine": "xelatex", "typeset": True, "rc": 0, "reason": "", "pages": 6, "format": "generic",
        "glyphs_replaced": [{"glyph": "⅓", "code_point": "U+2153", "became": "\tfrac{1}{3}", "where": "text", "count": 2}],
        "glyphs_not_set": [],
    })
    report = qr.build(root, running=False)
    assert report["paper_pdf"]["typeset"] is True and report["paper_pdf"]["pages"] == 6
    assert not [p for p in _stage(report, "paper")["problems"]["items"] if "typeset" in p["text"]]
    md = qr.render(report)
    assert "paper.pdf is typeset by xelatex, 6 pages." in md and "(U+2153) printed as \tfrac{1}{3}, 2 time(s)" in md

    _w(root / ".fi" / "paper_render.json", {
        "engine": "edge", "typeset": False, "rc": 0, "pages": 7, "format": "generic",
        "reason": "xelatex: no font here has Ɐ (U+2C6F)", "glyphs_replaced": [],
        "glyphs_not_set": [{"glyph": "Ɐ", "code_point": "U+2C6F"}],
    })
    report = qr.build(root, running=False)
    items = _stage(report, "paper")["problems"]["items"]
    assert [p["status"] for p in items if "not typeset" in p["text"]] == ["not fixed"]
    assert report["failure"] and report["failure"]["stage"] == "paper"
    md = qr.render(report)
    assert "paper.pdf is **not typeset**" in md and "`Ɐ` (U+2C6F): no font here has it" in md


def test_a_quest_without_a_render_record_has_no_pdf_section(tmp_path: Path) -> None:
    report = qr.build(_finished(tmp_path / "q"), running=False)
    assert report["paper_pdf"] is None and "## The PDF" not in qr.render(report)
