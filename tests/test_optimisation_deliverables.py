"""What a search for the best design hands over: the paper's "Best design found" section (FI's numbers, which the
number-provenance check traces), a refine that searches further toward a person's target, and the files listed in the
three interfaces.

The search and its check run through FI's own harness on a toy simulation (each evaluation a process of its own), as
in a quest; no model is called.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core import best_design_report as report
from core import number_provenance
from core import optimisation_plan as op
from core import optimise
from core import optimise_refine as refine
from core import optimum_check as oc
from tests.test_optimise_runner import LocalExecutor

# A real optimum at x = 3 (f = 10 there), a mesh error that shrinks with the mesh, and a limit on g the best design
# meets. From the baseline x = 0, six evaluations do not get there: a continued search does.
SIMULATE = '''\
def run_cell(cell):
    with open("count.txt", "a", encoding="utf-8") as fh:
        fh.write("1\\n")
    x, y = cell["x"], cell["y"]
    return {"f": (x - 3.0) ** 2 + 2.0 * (y - 1.5) ** 2 + 10.0 + 0.5 * cell["mesh"] ** 2, "g": x}
'''
ANALYSIS = '''\
import json, os
best = json.load(open(os.environ["FI_BEST_DESIGN"], encoding="utf-8"))
print("RESULT_JSON: " + json.dumps({"evaluated": best["evaluations"]["search"]}))
'''


def _block(**changes: Any) -> dict[str, Any]:
    block = {
        "objective": {"quantity": "f", "direction": "minimise", "unit": "K", "meaning": "the peak temperature"},
        "design_variables": [{"name": "x", "low": 0.0, "high": 6.0, "kind": "continuous", "unit": "mm"},
                             {"name": "y", "low": 0.0, "high": 6.0, "kind": "continuous", "unit": "mm"}],
        "constraints": [{"quantity": "g", "limit": "<= 5.5"}],
        "baseline": {"values": {"x": 0.0, "y": 0.0}, "source": "the design in use"},
        "numerical_settings": {"mesh": {"search": 0.4}},
        "evaluation_budget": {"starts": 1, "per_start": 6},
        "search_method": "bounded_local",
    }
    block.update(copy.deepcopy(changes))
    fixed, why = op.normalize(block)
    assert why is None, why
    return fixed


def _quest(tmp_path: Path) -> Path:
    root = tmp_path / "quest"
    (root / "code").mkdir(parents=True)
    (root / ".fi").mkdir()
    (root / "code" / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    return root


def _runner(root: Path, block: dict[str, Any]) -> optimise.OptimisationRunner:
    return optimise.OptimisationRunner(LocalExecutor(), quest_root=root, protocol={"optimisation": block},
                                       simulate=root / "code" / "simulate.py",
                                       analysis=root / "code" / "experiment.py")


async def _run(runner: optimise.OptimisationRunner, root: Path) -> dict[str, Any]:
    result = await runner.execute([sys.executable, str(root / "code" / "experiment.py")], cwd=root, timeout_s=300)
    assert result.returncode == 0, result.stderr[-2000:]
    line = next(x for x in result.stdout.splitlines() if x.startswith("RESULT_JSON:"))
    return json.loads(line[len("RESULT_JSON:"):])


def _count(root: Path) -> int:
    return len((root / "count.txt").read_text(encoding="utf-8").splitlines())


PAPER = "# A study\n\n## Methods\n\nWe searched.\n\n## Results\n\nThe search ran.\n\n## Discussion\n\nIt worked.\n"


@pytest.mark.asyncio
async def test_the_papers_section_prints_fis_numbers_and_number_provenance_traces_every_one(tmp_path: Path) -> None:
    block = _block()
    root = _quest(tmp_path)
    result_json = await _run(_runner(root, block), root)
    best = json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8"))
    check = oc.read(root)
    assert check is not None and best["check"]["record_sha256"]

    text = report.for_paper(root, block, PAPER)
    paper = report.mark_paper(PAPER, text)
    # The section is FI's, between its markers, before the discussion.
    assert paper.index(report.BEGIN) < paper.index("## Discussion") and paper.count("## Best design found") == 1
    # Its numbers are the records': the best design and the baseline at the finest settings, the improvement and its
    # numerical error, the search's own values, the budget used.
    fmt = report._fmt
    finest_best = check["best"]["objective"]["check"][-1]
    finest_base = check["baseline"]["objective"]["check"][-1]
    assert f"| {fmt(finest_base)} | {fmt(finest_best)} |" in text
    imp = check["improvement"]
    assert f"{fmt(imp['value'])} K ± {fmt(imp['numerical_error'])} K" in text
    assert f"{fmt(best['best']['objective'])} | {fmt(best['improvement']['value'])} K" in text
    assert f"{best['evaluations']['search']} of {best['evaluations']['budget']} evaluations" in text
    assert fmt(check["best"]["design"]["x"]) in text and "0 to 6" in text
    # The check's verdict and each check, in plain words; the limits of the result.
    assert check["says"] in text and "- Finer settings:" in text and "**Limits of this result.**" in text
    # A second pass puts it in once, whatever the writer left between the markers.
    again = report.mark_paper(paper.replace("| 0 to 6 |", "| 0 to 60 |"), text)
    assert again.count(report.BEGIN) == 1 and "0 to 60" not in again

    # Every number in the paper traces to this run: the section's to FI's two records.
    rep = number_provenance.check(paper, result_json=result_json, design={"protocol": {"optimisation": block}},
                                  optimisation=report.records(root))
    assert not rep.skipped and rep.ok, [f.describe() for f in rep.findings]
    assert rep.paper_numbers >= 8
    # A number in the section that is not FI's is caught like any other.
    wrong = paper.replace(f"| {fmt(finest_base)} | {fmt(finest_best)} |", f"| {fmt(finest_base)} | 7.913 |")
    assert wrong != paper
    bad = number_provenance.check(wrong, result_json=result_json, design={"protocol": {"optimisation": block}},
                                  optimisation=report.records(root))
    assert not bad.ok and any("7.913" in f.describe() for f in bad.findings)

    # The claim check leaves the section out only when it is exactly FI's.
    assert report.BEGIN not in report.strip_for_checks(paper, report.for_paper(root, block, paper))
    assert report.BEGIN in report.strip_for_checks(paper.replace("It worked.", "x"), "something else")
    # The readable file is the same section, and the files a person opens are listed.
    assert (root / "results" / "best_design.md").read_text(encoding="utf-8").startswith("# Best design found")
    listed = [f["path"] for f in report.files(root)]
    assert listed[:4] == ["results/best_design.md", "results/best_design.json", "raw/optimisation_ledger.jsonl",
                          "needs/OPTIMUM_CHECK.json"]
    line = report.summary_line(root)
    d = check["best"]["design"]
    assert line.startswith(f"x = {fmt(d['x'])}, y = {fmt(d['y'])} — f {fmt(abs(imp['value']))} K better than the "
                           f"baseline (± {fmt(imp['numerical_error'])} K); ")
    assert report._VERDICT_SHORT[check["verdict"]] in line


def test_the_section_goes_before_the_discussion_that_follows_the_results() -> None:
    paper = ("# T\n\n## Summary\n\nS.\n\n## Limitations of prior work\n\nL.\n\n## Results\n\nR.\n\n## Discussion\n\nD.\n"
             "\n---\n\n## References\n\n1. x\n")
    out = report.mark_paper(paper, "## Best design found\n\nB.\n")
    assert out.index("## Results") < out.index(report.BEGIN) < out.index("## Discussion")
    # A paper whose sections are first-level headings, with no discussion: before the references' rule line.
    flat = "# Title\n\n# Results\n\nR.\n\n---\n\n# References\n\n1. x\n"
    assert report.heading_level(flat) == 1
    out = report.mark_paper(flat, "# Best design found\n\nB.\n")
    assert out.index("R.") < out.index(report.BEGIN) < out.index("---")


@pytest.mark.asyncio
async def test_a_refine_toward_a_target_continues_the_search_with_the_added_budget_and_checks_again(
        tmp_path: Path) -> None:
    block = _block()
    root = _quest(tmp_path)
    runner = _runner(root, block)
    await _run(runner, root)
    before = json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8"))
    ledger_before = (root / optimise.LEDGER_PATH).read_text(encoding="utf-8").splitlines()
    check_before = oc.read(root)
    count_before = _count(root)
    assert before["best"]["objective"] > 10.2, "the first search stops short of the optimum"

    request = refine.read_request("Not good enough: push further, I want f at most 10.2", block, before)
    assert request["kind"] == refine.SEARCH
    round_asked = request["round"]
    assert {k: round_asked[k] for k in ("added", "target", "asked", "read_as")} == {
        "added": 6, "target": 10.2, "asked": "Not good enough: push further, I want f at most 10.2",
        "read_as": "f at most 10.2 K"}
    # The search compares designs at its own settings: the target there is moved by the gap the check measured.
    gap = before["check"]["objective"] - before["best"]["objective"]
    assert round_asked["search_target"] == pytest.approx(10.2 - gap)
    refine.add_round(root, block, {**request["round"], "added": 30})
    await _run(runner, root)
    after = json.loads((root / "results" / "best_design.json").read_text(encoding="utf-8"))
    ledger_after = (root / optimise.LEDGER_PATH).read_text(encoding="utf-8").splitlines()
    check_after = oc.read(root)
    round_ = after["continued"][0]

    # The search went on from the best design so far, with the added budget, and stopped at the target.
    assert round_["from"] == before["best"]["design"] and round_["added"] == 30
    assert after["evaluations"]["budget"] == 6 + 30 and after["evaluations"]["added_budget"] == 30
    assert round_["stopped_because"] == "target" and round_["target_reached"] is True
    assert after["best"]["objective"] <= 10.2 < before["best"]["objective"]
    # The search before the round is the same search: its evaluations were taken from FI's record, not run again.
    # (The first line is the search's head: its budget is now the plan's and the added.)
    assert ledger_after[1:len(ledger_before)] == ledger_before[1:] and len(ledger_after) > len(ledger_before)
    assert json.loads(ledger_after[0])["budget"] == 36
    assert _count(root) - count_before == round_["evaluations"] + check_after["evaluations"]["check"]
    # The check ran again, on the new best design, and the target is judged at the finest settings too.
    assert check_after["best"]["design"] != check_before["best"]["design"]
    assert after["target"]["from"] == "refine" and after["target"]["reached_at_finest_settings"] is True
    # The plan's budget plus what the person added: the record is within it.
    gaps = oc.evidence_gaps(root, {"optimisation": block})
    assert not any("more than the" in g for g in gaps.get("protocol_runtime_matched", []))
    # The paper and the one line say whether the target was reached.
    text = report.for_paper(root, block, PAPER)
    assert "**Target.** The target asked for after the first result: f at most 10.2 K; reached at the finest " \
           "settings" in text
    assert "Continued at the person's request (round 1)" in text
    assert report.summary_line(root).endswith("target 10.2 K: reached")
    # code/run.py continues the search the same way.
    from core import code_project

    study = code_project.study_of(root / "code", {"optimisation": block})
    assert study is not None and study["rounds"] == [{"added": 30, "target": 10.2,
                                                      "search_target": round_asked["search_target"]}]


def test_what_a_refine_asks_of_a_search_is_read_and_said_back() -> None:
    block = _block()
    best = {"baseline": {"objective": 19.08}, "best": {"objective": 12.5}}
    plain = refine.read_request("push it further", block, best)
    assert plain["kind"] == refine.SEARCH and plain["round"]["added"] == 6 and plain["round"]["target"] is None
    assert refine.read_request("keep searching, 40 more evaluations", block, best)["round"]["added"] == 40
    better = refine.read_request("I want it 8 K better than the baseline", block, best)
    assert better["round"]["target"] == pytest.approx(11.08) and "8 K better than the baseline" in better["round"]["read_as"]
    # "at least 8 K" of a quantity made as low as possible is an improvement of at least that much.
    assert refine.read_request("I want at least 8 K", block, best)["round"]["target"] == pytest.approx(11.08)
    assert refine.read_request("do not change the limits, just push further", block, best)["kind"] == refine.SEARCH
    # A target the best design already reaches changes nothing and is said so.
    assert refine.read_request("f at most 13", block, best)["kind"] == refine.UNCLEAR
    # An ordinary note goes to the writer.
    assert refine.read_request("Fix the typo in the abstract.", block, best) == {}
    # For a quantity made as high as possible, "at most 5" cannot be read: asked again, nothing changed.
    high = _block(objective={"quantity": "f", "direction": "maximise", "unit": "K"})
    assert refine.read_request("f at most 5", high, best)["kind"] == refine.UNCLEAR
    # A comparison written with a symbol, and the objective's name with spaces.
    assert refine.read_request("f<=11", block, best)["round"]["target"] == 11
    # A thousands separator, and an improvement in percent.
    assert refine.read_request("f at most 1,500 K", block, {"best": {"objective": 2000.0}})["round"]["target"] == 1500
    assert refine.read_request("at least 5% better than the baseline", high,
                               {"baseline": {"objective": 30.0}, "best": {"objective": 31.0}}
                               )["round"]["target"] == pytest.approx(31.5)


@pytest.mark.parametrize("note", [
    "Compare the 3 designs in one table",
    "Average over the 5 runs in the text",
    "Search more literature on fin cooling",
    "Explain why this is a better design than the baseline",
    "Add a sentence explaining the constraints",
    "Change the y-axis range of Figure 2",
    "Minimise jargon in the discussion",
    "Keep the abstract under 200 words",
    "Use up to 3 significant figures",
    "Add at least 2 more references",
    "Discuss why 12 evaluations were enough",
    "Keep the temperature discussion below 200 words",
])
def test_a_note_about_the_paper_is_an_ordinary_refine(note: str) -> None:
    best = {"baseline": {"objective": 19.08}, "best": {"objective": 12.5}}
    assert refine.read_request(note, _block(), best) == {}
    high = _block(objective={"quantity": "f", "direction": "maximise", "unit": "K"})
    assert refine.read_request(note, high, best) == {}


@pytest.mark.parametrize("note", [
    "minimise g instead",
    "Maximise f",
    "change the objective to the mass",
    "relax the limit on g",
    "g at most 7",
    "widen the range of x",
])
def test_changing_the_objective_or_its_limits_in_a_refine_is_a_new_study(note: str) -> None:
    request = refine.read_request(note, _block(), {"baseline": {"objective": 19.0}, "best": {"objective": 12.0}})
    assert request["kind"] == refine.NEW_STUDY, request
    assert "That is a new study" in request["says"] and "Nothing was changed" in request["says"]


def _engine(tmp_path: Path):  # noqa: ANN202
    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
    from core.engine import Engine

    return Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(clarify_mode="off", review_loop=True, max_iterations=2,
                            human_feedback_gate="after_review"),  # type: ignore[arg-type]
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))


def test_the_write_step_puts_fis_section_in_the_paper_and_its_numbers_pass_the_provenance_check(
        tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    block = _block()
    root = eng.quest_root
    (root / "code").mkdir(parents=True, exist_ok=True)
    (root / "paper").mkdir(parents=True, exist_ok=True)
    (root / "code" / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    result_json = asyncio.run(_run(_runner(root, block), root))
    eng._protocol_block = lambda state: {"optimisation": block}  # type: ignore[method-assign]

    async def no_code_check() -> None:
        return None

    eng._check_code_project = no_code_check  # type: ignore[method-assign]
    prompts: list[str] = []
    # The writer's own copy of the section, with a number of its own, is replaced by FI's.
    reply = (PAPER.replace("## Discussion", f"{report.BEGIN}\n## Best design found\n\nf fell by 9.5 K.\n{report.END}\n\n"
                                            "## Discussion"))

    async def chat(prompt: str, *, node: str = "") -> str:
        prompts.append(prompt)
        return reply

    eng._chat = chat  # type: ignore[method-assign]
    state = {"topic": "t", "title": "t", "iteration": 1, "result_json": result_json,
             "design": {"study_type": "find_best_design", "protocol": {"optimisation": block}}}
    out = asyncio.run(eng._node_write(state))  # type: ignore[arg-type]
    paper = Path(out["paper_md"]).read_text(encoding="utf-8")
    assert paper.count(report.BEGIN) == 1 and "9.5 K" not in paper
    assert report.for_paper(root, block, paper).strip() in paper
    assert "FI puts a section titled \"Best design found\"" in prompts[0]
    # The paper's numbers, the section's included, all trace to the run.
    assert eng._number_provenance_hits(paper, {**state, **out}) == []  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_the_three_interfaces_show_the_best_design_and_list_its_files(tmp_path: Path, capsys: Any) -> None:
    import re

    from fastapi.testclient import TestClient

    import launch
    from web.server import make_app

    block = _block()
    out_root = tmp_path / "outputs"
    root = out_root / "1700000000-best-aaaa11"
    (root / "code").mkdir(parents=True)
    (root / ".fi").mkdir()
    (root / "code" / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    (root / "code" / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    await _run(_runner(root, block), root)
    line = report.summary_line(root)
    files = report.files(root)
    assert line and [f["path"] for f in files][:4] == ["results/best_design.md", "results/best_design.json",
                                                       "raw/optimisation_ledger.jsonl", "needs/OPTIMUM_CHECK.json"]

    # CLI: two lines, and the summary file's entry.
    summary: dict[str, Any] = {}
    launch._report_best_design(root, summary)
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == f"[FI] best design: {line}"
    assert printed[1] == "[FI] best design files: " + ", ".join(f["path"] for f in files)
    assert summary["best_design"] == {"says": line, "files": files}

    # Web: the quest API carries the same line, the files and FI's two records as written.
    body = TestClient(make_app(out_root)).get(f"/api/quests/{root.name}").json()
    assert body["best_design"]["says"] == line and body["best_design"]["files"] == files
    assert body["best_design"]["optimum_check"] == oc.read(root)
    page = (Path(__file__).resolve().parents[1] / "web" / "static" / "quest.html").read_text(encoding="utf-8")
    assert "renderBestDesign(data.best_design)" in page and "refine_not_done" in page

    # VS Code: the chat renders exactly the lines the CLI prints.
    ext_dir = Path(__file__).resolve().parents[1] / "vscode-frontier-insight"
    source = (ext_dir / "src" / "extension.ts").read_text(encoding="utf-8")
    for pattern in (r"^\[FI\] best design files: (.+)$", r"^\[FI\] best design: (.+)$"):
        assert f"/{pattern}/" in source
    assert re.match(r"^\[FI\] best design: (.+)$", printed[0]) and re.match(r"^\[FI\] best design files: (.+)$",
                                                                             printed[1])
    assert "refine_not_done" in (ext_dir / "src" / "bridge.ts").read_text(encoding="utf-8")
    assert "best design" in (ext_dir / "README.md").read_text(encoding="utf-8").lower()


def _review_answer(eng: Any, state: dict[str, Any], feedback: str) -> tuple[dict[str, Any], dict[str, Any]]:
    seen: dict[str, Any] = {}

    def pause(**kw: Any) -> dict[str, str]:
        seen.update(kw)
        return {"action": "refine", "feedback": feedback}

    eng._pause_for_human = pause
    out = asyncio.run(eng._node_human_feedback(state))
    return out, seen


def test_the_review_pause_routes_a_search_refine_to_the_search_and_refuses_a_new_study(tmp_path: Path) -> None:
    from core.engine import _auto_accepts

    eng = _engine(tmp_path)
    block = _block()
    eng._protocol_block = lambda state: {"optimisation": block}  # type: ignore[method-assign]
    root = eng.quest_root
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "best_design.json").write_text(json.dumps(
        {"objective": {"quantity": "f", "direction": "minimise", "unit": "K"},
         "baseline": {"objective": 19.0, "design": {"x": 0.0}}, "best": {"objective": 12.0, "design": {"x": 1.0}}}),
        encoding="utf-8")
    state = {"topic": "t", "title": "t", "iteration": 1, "review": {"verdict": "accept", "score": 8, "status": "ok"},
             "feedback_history": []}

    # A new study: nothing kept, nothing changed, asked again with the reason.
    out, seen = _review_answer(eng, dict(state), "minimise g instead")
    assert out["optimise_refine"]["kind"] == refine.NEW_STUDY
    assert "feedback_history" not in out and "iteration" not in out
    assert eng._route_after_human_feedback({**state, **out}) == "ask_again"
    assert not (root / optimise.ROUNDS_PATH).exists()
    # The next pause says why, and it is not accepted for the person even when the review passed.
    out2, seen2 = _review_answer(eng, {**state, **out}, "push further")
    assert any("That is a new study" in s for s in seen2["steps"])
    assert seen2["payload"]["human_review"]["refine_not_done"]
    assert not _auto_accepts(seen2["payload"]["human_review"])
    # Every pause of a search says how to ask for more.
    assert any("push further" in s and "Changing what is optimised" in s for s in seen["steps"])

    # A request to search further: recorded as a round, routed to the search, not to the writer.
    assert out2["optimise_refine"]["kind"] == refine.SEARCH
    assert eng._route_after_human_feedback({**state, **out2}) == "search"
    assert out2["feedback_history"][-1] == {"iteration": 1, "text": "push further", "answered_by": "search"}
    assert out2["refine_written_for"] == 1 and out2["feedback_rounds_from"] == 1
    # The run after it starts with fresh repair counters; the writer and the review do not read the note as a change.
    assert out2["exec_reflect_iter"] == 0 and out2["exec_give_up_reason"] == ""
    from core.engine import _format_review_for_writer, _user_feedback_review_block

    after = {**state, **out2}
    assert "push further" not in _format_review_for_writer(after) and not _user_feedback_review_block(after)
    # The person's next, ordinary refine is shown to the review (the marked note is left out, not the rounds after it).
    later = {**after, "feedback_history": [*after["feedback_history"], {"iteration": 2, "text": "Fix the typo"}],
             "feedback_rounds_from": 1}
    assert "Fix the typo" in _user_feedback_review_block(later)
    rounds = optimise.read_rounds(root, block)
    assert rounds == [{"added": 6, "target": None, "asked": "push further", "read_as": "", "refine": 1}]
    # The review step run again for the same refine (a resume) does not add the round twice.
    _review_answer(eng, {**state, **out}, "push further")
    assert len(optimise.read_rounds(root, block)) == 1
    # An ordinary refine of any other study goes to the writer as before.
    eng._protocol_block = lambda state: None  # type: ignore[method-assign]
    out3, _ = _review_answer(eng, dict(state), "push further")
    assert out3["optimise_refine"] == {} and eng._route_after_human_feedback({**state, **out3}) == "rewrite"
