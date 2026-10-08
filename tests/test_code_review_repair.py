"""The code review is a real gate and its repair channel is its own (core/code_review.py, Engine._code_review_gate): the
findings are sent in full, to the file they name, in batches; the changed version is read again; repairs stop when the open
high-severity findings do not drop; the last reading is kept for the version that will run; an open high-severity finding
about the main quantity that the results do not contradict ends in the honest stop. Fake models, a mass on a damped spring."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import pytest

from core import code_review as cr
from tests.test_code_review import (
    ANALYSIS, DESIGN, REPAIR_HEAD, SIM_FULL, STATE, WITH_PLAN, _engine,
)

MODEL = "# E1\ndef quality(m, w, c):\n    return m * w / c\n"
DESIGN_WITH_HEADLINE = DESIGN
HEADLINE_FINDING = {"where": "simulate.py:2", "what": "quality_factor ignores the damping", "why": "the plan's E1 divides by c",
                    "fix": "divide by cell['c']", "severity": "high", "quantity": "quality_factor"}


def _fence(rel: str, text: str) -> str:
    return f"```python\n# file: {rel}\n{text}```"


class _Script:
    """A fake reader and a fake code-writing model in one: ``decide(reading number, code shown)`` gives the findings of a
    reading; ``repair(file, number of repair requests)`` gives the whole file back (or ``None`` for no file)."""

    def __init__(self, decide: Callable[[int, str], list[dict[str, Any]]],
                 repair: Callable[[str, int], str | None] | None = None) -> None:
        self.decide, self.repair = decide, repair
        self.prompts: list[str] = []
        self.repair_prompts: list[str] = []

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if prompt.lstrip().startswith(REPAIR_HEAD):
            self.repair_prompts.append(prompt)
            rel = re.search(r"^### (\S+)", prompt, re.M).group(1)  # type: ignore[union-attr]
            text = self.repair(rel, len(self.repair_prompts)) if self.repair else None
            return "" if text is None else _fence(rel, text)
        assert prompt.lstrip().startswith("# Code Against the Plan"), prompt[:80]
        self.prompts.append(prompt)
        reqs = json.loads(prompt.split("```json", 1)[1].split("```", 1)[0])
        code = prompt.split("# The code", 1)[1]
        findings = self.decide(len(self.prompts), code)
        return json.dumps({"requirements": [{"id": r["id"], "status": "implemented", "file": "simulate.py", "line": 2}
                                            for r in reqs],
                           "findings": findings, "checked": ["E1: the factors and signs"], "summary": "s"})

    async def aclose(self) -> None:
        return None


def _with_package(engine: Any) -> None:
    pkg = engine.quest_root / "code" / "spring"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("from . import model\n", encoding="utf-8")
    (pkg / "model.py").write_text(MODEL, encoding="utf-8")


# --- where a finding is repaired -----------------------------------------------------------------------------------------


def test_a_finding_is_repaired_in_the_file_it_names_the_model_s_package_included(tmp_path: Path) -> None:
    code = tmp_path / "code"
    (code / "spring").mkdir(parents=True)
    (code / "spring" / "__init__.py").write_text("", encoding="utf-8")
    (code / "spring" / "model.py").write_text(MODEL, encoding="utf-8")
    (code / "simulate.py").write_text(SIM_FULL, encoding="utf-8")
    (code / "experiment.py").write_text(ANALYSIS, encoding="utf-8")

    def logic(where: str, severity: str = "medium") -> dict[str, Any]:
        got = cr.parse_logic({"findings": [{"where": where, "what": f"fault at {where}", "why": "w", "fix": "f",
                                            "severity": severity}]}, [])[0]
        return got

    def req(kind: str, file: str = "") -> dict[str, Any]:
        return {"id": "R1", "kind": kind, "text": f"the {kind}", "status": "missing", "file": file, "line": None,
                "note": "", "headline": False}

    routed = cr.files_for([logic("model.py:3"), logic("simulate.py:9"), logic("experiment.py:4"), logic("spring/model.py:7")], code)
    assert set(routed) == {"spring/model.py", "simulate.py", "experiment.py"}
    assert [v["line"] for v in routed["spring/model.py"]] == [3, 7], "model.py is the package's file, whatever the path written"
    by_kind = cr.files_for([req("sweep"), req("check"), req("equation"), logic(""), req("figure"), req("dependent variable"),
                            req("metric")], code)
    assert [v["kind"] for v in by_kind["simulate.py"]] == ["sweep", "check", "equation", "logic"]
    assert [v["kind"] for v in by_kind["experiment.py"]] == ["figure", "dependent variable", "metric"]
    # A file that is not one of the code's falls back by what the finding is about; the most serious come first.
    odd = cr.files_for([logic("run.py:1"), logic("simulate.py:2", "high"), req("figure", "tests.py")], code)
    assert [v["severity"] if v.get("severity") else "-" for v in odd["simulate.py"]] == ["high", "medium"]
    assert list(odd) == ["simulate.py", "experiment.py"]


@pytest.mark.asyncio
async def test_a_finding_that_names_the_model_s_package_rewrites_that_file_and_not_simulate_py(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _with_package(engine)
    on_model = {"where": "model.py:3", "what": "E1 divides by the wrong quantity", "why": "the plan's E1 divides by c",
                "fix": "divide by c", "severity": "medium"}
    fixed = "# E1\ndef quality(m, w, c):\n    return m * w / c  # divided by c\n"
    script = _Script(lambda n, code: [on_model] if "# divided by c" not in code else [],
                     lambda rel, n: fixed if rel == "spring/model.py" else None)
    engine._client = script
    assert await engine._code_review_gate(WITH_PLAN) is None
    assert len(script.repair_prompts) == 1 and "### spring/model.py" in script.repair_prompts[0]
    assert "def run_cell" not in script.repair_prompts[0], "only the file the finding names is shown"
    assert (engine.quest_root / "code" / "spring" / "model.py").read_text(encoding="utf-8") == fixed
    assert (engine.quest_root / "code" / "simulate.py").read_text(encoding="utf-8") == SIM_FULL
    assert (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8") == ANALYSIS
    assert cr.load(engine.quest_root)["final"] is True


# --- findings in full, in batches, with the opening sentence --------------------------------------------------------------


@pytest.mark.asyncio
async def test_findings_are_sent_in_full_in_batches_and_every_request_opens_with_what_is_wrong(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    many = [{"where": f"simulate.py:{i + 2}", "what": f"finding number {i} " + "w" * 120,
             "why": f"reason {i} " + "y" * 200, "fix": f"remedy {i} " + "z" * 200, "severity": "medium"} for i in range(12)]
    script = _Script(lambda n, code: many if n == 1 else [], lambda rel, n: SIM_FULL + f"# repaired {n}\n")
    engine._client = script
    assert await engine._code_review_gate(WITH_PLAN) is None
    assert len(script.repair_prompts) == 2, "12 findings in batches of six: two requests, each its own bounded repair"
    for prompt in script.repair_prompts:
        assert prompt.lstrip().startswith(REPAIR_HEAD) and "has NOT crashed" in prompt.split("# Topic")[0]
        assert len(prompt) > 2500
    together = "\n".join(script.repair_prompts)
    for i in range(12):
        for part in (f"finding number {i} " + "w" * 120, f"reason {i} " + "y" * 200, f"remedy {i} " + "z" * 200):
            assert part in together, f"finding {i} is sent whole"
    first = script.repair_prompts[0]
    assert first.count("What is wrong:") == 6 and "How to fix it:" in first and "Why it is wrong against the plan:" in first
    assert "has NOT crashed" in first and "[medium]" in first
    # The second batch sees the first batch's change: the file is read again from disk for each request.
    assert "# repaired 1" in script.repair_prompts[1]
    record = cr.load(engine.quest_root)
    assert record["repair_calls"] == 2 and record["send_backs"] == 1


@pytest.mark.asyncio
async def test_the_repair_request_names_the_topic_and_the_equations_but_never_an_expected_value(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    leaky = {**HEADLINE_FINDING, "fix": "divide by cell['c']; the result should be 0.123456789"}
    script = _Script(lambda n, code: [leaky] if n == 1 else [], lambda rel, n: SIM_FULL + "# repaired\n")
    engine._client = script
    await engine._code_review_gate(WITH_PLAN)
    asked = script.repair_prompts[0]
    assert "a mass on a damped spring" in asked and "Q = m * w / c" in asked
    assert "0.123456789" not in asked and "divide by cell['c']" in asked
    assert "just failed" not in asked and "returncode" not in asked and "RESULT_JSON" in asked


# --- no oscillation ---------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_repair_that_does_not_reduce_the_open_high_findings_stops_the_repairs_and_leaves_the_budget_unspent(
        tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    script = _Script(lambda n, code: [HEADLINE_FINDING], lambda rel, n: SIM_FULL + f"# attempt {n}\n")
    engine._client = script
    assert await engine._code_review_gate(WITH_PLAN) is None
    record = cr.load(engine.quest_root)
    assert len(script.prompts) == 2 and len(script.repair_prompts) == 1, "read, repaired once, read again, and no second repair"
    assert [(r["open"], r["high"]) for r in record["readings"]] == [(1, 1), (1, 1)]
    assert record["converged"] is False and record["final"] is True
    assert record["send_backs"] == 1 < cr.MAX_SEND_BACKS and record["reviews_total"] == 2 < cr.MAX_REVIEWS
    assert "did not reduce" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    note = cr.write_note(engine.quest_root, engine.quest_root / "code")
    assert "stopped because a round did not reduce" in note and "quality_factor ignores the damping" in note
    # The version that will run is the one read last; the late call asks and repairs nothing more.
    assert await engine._code_review_gate(WITH_PLAN, final=True) is None
    assert len(script.prompts) == 2 and len(script.repair_prompts) == 1
    assert cr.load(engine.quest_root)["final_read"] is True


@pytest.mark.asyncio
async def test_repairs_that_keep_reducing_the_findings_go_on_up_to_their_bound(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    three = [{**HEADLINE_FINDING, "what": f"fault {i}", "severity": "high", "quantity": "quality_factor"} for i in range(3)]

    def decide(n: int, code: str) -> list[dict[str, Any]]:
        return three[: max(0, 3 - code.count("# attempt"))]

    script = _Script(decide, lambda rel, n: SIM_FULL + "".join(f"# attempt {i}\n" for i in range(n)))
    engine._client = script
    await engine._code_review_gate(WITH_PLAN)
    record = cr.load(engine.quest_root)
    assert [r["high"] for r in record["readings"]] == [3, 2], "the third reading is the reserved one: not made by this call"
    assert record["converged"] is True and record["send_backs"] == 2 == cr.MAX_SEND_BACKS, "bounded at two rounds"
    assert len(script.repair_prompts) == 2 and record["reviews_total"] == cr.MAX_REVIEWS - cr.RESERVED_FOR_FINAL


# --- the last reading is for the version that will run --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_final_version_is_always_read_even_after_repairs_used_the_other_readings(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    three = [{**HEADLINE_FINDING, "what": f"fault {i}", "quantity": "quality_factor"} for i in range(3)]

    def decide(n: int, code: str) -> list[dict[str, Any]]:
        return three[: max(0, 3 - code.count("# attempt"))]

    script = _Script(decide, lambda rel, n: SIM_FULL + "".join(f"# attempt {i}\n" for i in range(n)))
    engine._client = script
    await engine._code_review_gate(WITH_PLAN)
    # Two intermediate readings were made; the repaired version after the second is not read by this early call.
    assert len(script.prompts) == cr.MAX_REVIEWS - 1
    assert cr.load(engine.quest_root)["final_read"] is False
    # An oracle or equation repair may change it again; whatever version will run is read with the reserved reading.
    (engine.quest_root / "code" / "simulate.py").write_text(SIM_FULL + "# attempt 0\n# attempt 1\n# gate\n", encoding="utf-8")
    assert await engine._code_review_gate(WITH_PLAN, final=True) is None
    record = cr.load(engine.quest_root)
    assert len(script.prompts) == cr.MAX_REVIEWS and len(script.repair_prompts) == 2, "no repair of a version nobody reads"
    assert record["final_read"] is True and record["read_version"] == cr.version(engine.quest_root / "code")


@pytest.mark.asyncio
async def test_a_reader_that_fails_twice_is_recorded_and_the_final_version_is_said_unread(tmp_path: Path) -> None:
    calls = {"n": 0}

    class _Fails:
        async def chat(self, messages, **kw):  # noqa: ANN001
            calls["n"] += 1
            raise TimeoutError("no answer")

        async def aclose(self) -> None:
            return None

    engine = _engine(tmp_path)
    engine._client = _Fails()
    assert await engine._code_review_gate(WITH_PLAN, final=True) is None
    assert calls["n"] == 2, "a failed request is made once more, and no more"
    record = cr.load(engine.quest_root)
    assert "twice" in record["error"] and record["final_read"] is False
    note = cr.write_note(engine.quest_root, engine.quest_root / "code")
    assert "not compared with the plan by another model" in note and "twice" in note


# --- after the run --------------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_high_finding_about_the_main_quantity_that_the_results_cannot_settle_is_a_limitation_never_a_stop(
        tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine._client = _Script(lambda n, code: [HEADLINE_FINDING])
    await engine._code_review_gate(WITH_PLAN, final=True)
    results = {**STATE, "result_json": {"cells": [{"c": 0.2, "quality_factor": 2.5}, {"c": 0.5, "quality_factor": 1.0}]}}
    assert engine._no_results_verdict(results) is None
    note = cr.write_note(engine.quest_root, engine.quest_root / "code")
    assert "quality_factor ignores the damping" in note and "not resolved" in note and "[high]" in note


@pytest.mark.asyncio
async def test_a_constant_claim_the_results_cannot_settle_does_not_stop_but_one_they_confirm_does(tmp_path: Path) -> None:
    constant = {**HEADLINE_FINDING, "what": "quality_factor does not depend on the damping", "constant": True}
    engine = _engine(tmp_path)
    engine._client = _Script(lambda n, code: [constant])
    await engine._code_review_gate(WITH_PLAN, final=True)
    assert engine._no_results_verdict({**STATE, "result_json": {"quality_factor": 1.0}}) is None, "it appears once: undecided"
    flat = {**STATE, "result_json": {"cells": [{"quality_factor": 1.0}, {"quality_factor": 1.0}]}}
    verdict = engine._no_results_verdict(flat)
    assert verdict and verdict["stuck"] and verdict["stuck_reason"] == "headline_missing"


@pytest.mark.asyncio
async def test_a_finding_that_names_no_quantity_is_not_a_headline_finding_whatever_its_text_says() -> None:
    reqs = cr.requirements(DESIGN_WITH_HEADLINE, None)
    got = cr.parse_logic({"findings": [{"where": "simulate.py:2", "what": "quality_factor is wrong", "why": "quality_factor",
                                        "fix": "f", "severity": "high"}]}, reqs)
    assert got[0]["headline"] is False


def test_the_directive_of_a_layout_problem_is_sent_whole_to_the_repair(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    message = "problem line. " * 400
    patch = engine._code_review_failed_run(message)
    assert patch["exec_result"]["stderr_tail"] == message and patch["exec_result"]["directive"] is True


@pytest.mark.asyncio
async def test_a_finding_that_the_results_contradict_lets_the_paper_go_on_with_the_disagreement_recorded(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    constant = {**HEADLINE_FINDING, "what": "quality_factor does not depend on the damping", "constant": True}
    script = _Script(lambda n, code: [constant])
    engine._client = script
    await engine._code_review_gate(WITH_PLAN, final=True)
    varies = {**STATE, "result_json": {"cells": [{"c": 0.2, "quality_factor": 2.5}, {"c": 0.5, "quality_factor": 1.0}]}}
    assert engine._no_results_verdict(varies) is None, "the results show it varies: the reviewer was wrong"
    record = cr.load(engine.quest_root)
    assert record["disagreement"] == ["quality_factor does not depend on the damping"]
    assert cr.write_note(engine.quest_root, engine.quest_root / "code") == ""


@pytest.mark.asyncio
async def test_a_lower_severity_finding_or_one_about_another_quantity_is_a_limitation_not_a_stop(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    minor = {**HEADLINE_FINDING, "severity": "medium", "what": "a normalisation differs"}
    other = {**HEADLINE_FINDING, "severity": "high", "quantity": "settling_time", "what": "settling_time uses the wrong band"}
    script = _Script(lambda n, code: [minor, other])
    engine._client = script
    await engine._code_review_gate(WITH_PLAN, final=True)
    results = {**STATE, "result_json": {"quality_factor": 1.0, "settling_time": 3.0}}
    assert engine._no_results_verdict(results) is None
    note = cr.write_note(engine.quest_root, engine.quest_root / "code")
    assert "a normalisation differs" in note and "settling_time uses the wrong band" in note and "[high]" in note


def test_the_quality_report_shows_whether_the_final_version_was_read_and_whether_the_repairs_converged(tmp_path: Path) -> None:
    from core import quality_report as qr

    root = tmp_path / "q"
    (root / ".fi").mkdir(parents=True)
    (root / ".fi" / "code_review.json").write_text(json.dumps({
        "read_version": "a", "final_read": True, "converged": False, "repair_calls": 2,
        "readings": [{"n": 1, "open": 31, "high": 9}, {"n": 2, "open": 30, "high": 9}]}), encoding="utf-8")
    report = qr.build(root)
    code = report["code"]
    assert code["final_version_read"] is True and code["open_per_reading"] == [31, 30]
    assert code["open_high_per_reading"] == [9, 9] and code["repairs_converged"] is False and code["repair_requests"] == 2
    text = qr.render(report)
    assert "The final version of the code was read by a second model." in text
    assert "Open findings per reading: 31 (9 high-severity), 30 (9 high-severity)." in text
    assert "The repairs did not converge" in text


def test_the_result_line_must_still_be_printed_by_an_executable_statement(tmp_path: Path) -> None:
    printed = "import json\nr = {}\nprint('RESULT_JSON: ' + json.dumps(r))\n"
    assert cr.prints_result_json(printed)
    assert cr.prints_result_json("import json\nr = {}\nprint(f'RESULT_JSON: {json.dumps(r)}')\n")
    assert cr.prints_result_json("line = 'RESULT_JSON: ' + '{}'\nimport sys\nsys.stdout.write(line)\n")
    assert not cr.prints_result_json("import json\nr = {}\n# print('RESULT_JSON: ' + json.dumps(r))\n")
    assert not cr.prints_result_json("note = 'RESULT_JSON: is printed elsewhere'\n")
    assert not cr.prints_result_json("print('done')  # RESULT_JSON\n")
    assert cr.repair_problem(printed, printed + "x = 1\n", "experiment.py", equations=[]) == ""
    commented = "import json\nr = {}\n# print('RESULT_JSON: ' + json.dumps(r))\nTEXT = 'RESULT_JSON: '\n"
    assert "RESULT_JSON" in cr.repair_problem(printed, commented, "experiment.py", equations=[])
