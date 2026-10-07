"""A different model reads the generated code against the plan's required outputs before the run
(core/code_review.py). Fake models, neutral topics (a mass on a damped spring)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core import code_review as cr
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine

DESIGN = {"hypothesis": "h", "variables": {"independent": ["c"], "dependent": ["quality_factor", "settling_time"]},
          "figures_planned": ["decay.png"]}
PROTOCOL = {
    "grid": {"c": [0.2, 0.5, 1.0]},
    "metrics": [{"id": "settling_time", "estimand": "time to settle", "kind": "mean"}],
    "precision": {"metric": "quality_factor"},
    "model": {"summary": "a damped spring", "equations": [
        {"id": "E1", "formula": "Q = m * w / c", "role": "generates", "source": "derivation"}]},
    "oracles": [{"name": "q_check", "kind": "special_case", "check": "Q at one setting", "expected": 0.123456789,
                 "tolerance": 1e-9, "case": {"c": 0.5}, "measure": "quality", "reference": "derivation: by hand"}],
}
STATE: dict[str, Any] = {"design": DESIGN, "exec_result": {"returncode": 0}}
SIM_FULL = "# E1\ndef run_cell(cell):\n    return {'quality': 1.0}\n"
SIM_OTHER = "# E1\ndef run_cell(cell):\n    return {'quality': 2.0}\n"
ANALYSIS = "print('RESULT_JSON: {}')\n"


def _engine(tmp_path: Path, *, other: bool = True) -> Engine:
    cfg = Config(topic="a mass on a damped spring", title="spring",
                 provider=ProviderConfig(name="openai", model="planner-model",
                                         node_models={"oracle_review": "reader-model"} if other else {}),
                 engine=EngineConfig(max_iterations=1, review_loop=False, exec_reflect_max_iterations=3),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
                 output=OutputConfig(output_dir=tmp_path / "outputs"))
    engine = Engine(cfg)
    code = engine.quest_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "simulate.py").write_text(SIM_FULL, encoding="utf-8")
    (code / "experiment.py").write_text(ANALYSIS, encoding="utf-8")
    from core import plan

    text = plan.render(engine.config.topic, {}, {**DESIGN, "protocol": PROTOCOL})
    plan.plan_path(engine.quest_root).write_text(text, encoding="utf-8")
    plan.record_version(engine.quest_root, text, by="model")
    return engine


class _Reader:
    def __init__(self, missing: tuple[str, ...] = (), headline: bool = False) -> None:
        self.missing, self.headline, self.prompts = missing, headline, []

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        assert prompt.lstrip().startswith("# Code Against the Plan"), prompt[:80]
        self.prompts.append(prompt)
        reqs = json.loads(prompt.split("```json", 1)[1].split("```", 1)[0])
        out = []
        for r in reqs:
            gone = any(m in r["text"] for m in self.missing)
            out.append({"id": r["id"], "status": "missing" if gone else "implemented",
                        "file": "" if gone else "simulate.py", "line": None if gone else 2,
                        "note": "nothing in the code computes it" if gone else ""})
        return json.dumps({"requirements": out, "summary": "s"})

    async def aclose(self) -> None:
        return None


def test_the_requirements_are_names_and_never_a_number_and_the_headline_is_marked() -> None:
    reqs = cr.requirements(DESIGN, PROTOCOL)
    kinds = [r["kind"] for r in reqs]
    assert kinds == ["dependent variable", "dependent variable", "metric", "figure", "sweep", "check", "equation"]
    assert [r["text"] for r in reqs if r["headline"]] == [
        "the dependent variable quality_factor is computed and reported in the result"]
    blob = json.dumps(reqs)
    assert "0.123456789" not in blob and "expected" not in blob


def test_the_reply_is_read_strictly() -> None:
    reqs = cr.requirements(DESIGN, PROTOCOL)
    ok = {"requirements": [{"id": "r1", "status": "Implemented", "file": "simulate.py", "line": 3},
                           {"id": "R2", "status": "missing", "note": "not there"},
                           {"id": "R99", "status": "missing"}, {"id": "R3", "status": "maybe"}]}
    got = cr.parse(ok, reqs)
    assert [(v["id"], v["status"]) for v in got] == [("R1", "implemented"), ("R2", "missing")]
    assert cr.parse({"requirements": "all fine"}, reqs) is None and cr.parse("text", reqs) is None
    assert [v["id"] for v in cr.problems(got)] == ["R2"]
    text = cr.directive(cr.problems(got))
    assert "is missing from the code" in text and "R2" not in text


@pytest.mark.asyncio
async def test_a_missing_quantity_is_sent_to_the_existing_repair_as_a_plain_directive(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    reader = _Reader(missing=("settling_time",))
    engine._client = reader
    patch = await engine._code_review_gate(STATE)
    assert len(reader.prompts) == 1 and "0.123456789" not in reader.prompts[0]
    assert patch is not None and patch["exec_result"]["returncode"] == 1 and patch["result_json"] == {}
    said = patch["exec_result"]["stderr_tail"]
    assert "settling_time" in said and "is missing from the code" in said and "0.123456789" not in said
    record = cr.load(engine.quest_root)
    assert record["send_backs"] == 1 and record["same_model"] is False and not record.get("final")
    # The same code is read once; sent back and not changed, it stands as it is (nothing more is asked).
    assert await engine._code_review_gate(STATE) is None
    assert len(reader.prompts) == 1 and cr.load(engine.quest_root)["final"] is True
    # The repair changed the code: a new version is read again.
    (engine.quest_root / "code" / "simulate.py").write_text(SIM_OTHER, encoding="utf-8")
    reader.missing = ()
    assert await engine._code_review_gate(STATE) is None and len(reader.prompts) == 2
    assert cr.load(engine.quest_root)["final"] is True and cr.not_computed(engine.quest_root, engine.quest_root / "code") == []


@pytest.mark.asyncio
async def test_the_model_that_wrote_the_code_is_not_a_second_reader_and_nothing_is_asked(tmp_path: Path) -> None:
    engine = _engine(tmp_path, other=False)
    reader = _Reader(missing=("settling_time",))
    engine._client = reader
    assert await engine._code_review_gate(STATE) is None and reader.prompts == []
    record = cr.load(engine.quest_root)
    assert "not read by another model" in cr.plain_lines(record) and "planner-model" in record["skipped"]
    assert "the code was not read by another model" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_resume_does_not_read_the_same_code_again(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine._client = _Reader()
    assert await engine._code_review_gate(STATE) is None
    again = Engine(engine.config, resume_quest_id=engine.quest_id)
    reader = _Reader(missing=("settling_time",))
    again._client = reader
    assert await again._code_review_gate(STATE) is None and reader.prompts == []


@pytest.mark.asyncio
async def test_what_is_still_missing_after_the_repairs_is_not_reported_and_the_paper_says_so(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine._client = _Reader(missing=("planned figure",))
    # The repair attempts are spent: this is what stands, and the run goes on.
    assert await engine._code_review_gate({**STATE, "exec_reflect_iter": 3}) is None
    left = cr.not_computed(engine.quest_root, engine.quest_root / "code")
    assert [v["kind"] for v in left] == ["figure"]
    note = cr.write_note(engine.quest_root, engine.quest_root / "code")
    assert "decay.png" in note and "not computed" in note and "limitations" in note
    # Other code, other record: the note follows the code that ran.
    (engine.quest_root / "code" / "simulate.py").write_text(SIM_OTHER, encoding="utf-8")
    assert "changed afterwards" in cr.write_note(engine.quest_root, engine.quest_root / "code")


HEADLINE = "the dependent variable quality_factor is computed and reported in the result"


@pytest.mark.asyncio
async def test_a_reviewer_s_word_alone_never_stops_a_quest_the_results_decide(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine._client = _Reader(missing=("quality_factor",))
    out = await engine._code_review_gate({**STATE, "exec_reflect_iter": 3})
    assert out is None, "nothing stops before the run, and no repair is given up on"
    code = engine.quest_root / "code"
    assert cr.headline_missing(engine.quest_root, code) == [HEADLINE]
    # The run's results do not hold the quantity either: now the honest stop.
    lacking = {**STATE, "result_json": {"settling_time_mean": 1.2}}
    verdict = engine._no_results_verdict(lacking)
    assert verdict and verdict["stuck"] and verdict["stuck_reason"] == "headline_missing"
    stuck = await engine._node_stuck_no_findings({**lacking, "evidence_assessment": verdict})
    problem = stuck["stuck"]["problem"]
    assert "does not compute the study's main quantity" in problem and "quality_factor" in problem
    assert (engine.quest_root / "needs" / "STUCK.json").is_file()
    # The results do hold it (the reviewer was wrong): the quest goes on and the disagreement is recorded.
    holding = {**STATE, "result_json": {"by_c": {"quality_factor": 3.4}}}
    assert engine._no_results_verdict(holding) is None
    record = cr.load(engine.quest_root)
    assert record["disagreement"] == [HEADLINE] and cr.headline_missing(engine.quest_root, code) == []
    assert "quality_factor" not in cr.write_note(engine.quest_root, code), "the paper is not told it was not computed"


def test_only_the_quantity_s_own_name_counts_as_having_it() -> None:
    assert cr.in_results("settling_time", {"a": {"Settling_Time": 1.0}})
    assert cr.in_results("quality_factor", {"qualityFactor": 2})
    assert cr.in_results("error", {"Error": 0.1}) and cr.in_results("quality_factor", {"quality_factor": [1, 2]})
    assert not cr.in_results("error", {"error_count": 3, "error_rate_nm": 1.0}), "a different quantity is not an alias"
    assert not cr.in_results("settling_time", {"settling_time_mean": 1.0})
    assert not cr.in_results("quality_factor", {"quality_factor": None}) and not cr.in_results("quality_factor", {"x": 1})
    assert not cr.in_results("quality_factor", {"quality_factor": True}) and not cr.in_results("", {"a": 1})


@pytest.mark.asyncio
async def test_a_result_with_only_a_similar_name_does_not_stand_in_for_the_headline(tmp_path: Path) -> None:
    design = {**DESIGN, "variables": {"independent": ["c"], "dependent": ["error"]}}
    engine = _engine(tmp_path)
    state = {**STATE, "design": design}
    engine._client = _Reader(missing=("error",))
    assert await engine._code_review_gate({**state, "exec_reflect_iter": 3}) is None
    verdict = engine._no_results_verdict({**state, "result_json": {"error_count": 4}})
    assert verdict and verdict["stuck_reason"] == "headline_missing", "error_count is not error"
    assert engine._no_results_verdict({**state, "result_json": {"Error": 0.2}}) is None


@pytest.mark.asyncio
async def test_a_code_version_that_ran_without_a_successful_reading_is_disclosed(tmp_path: Path) -> None:
    class _Fails(_Reader):
        async def chat(self, messages, **kw):  # noqa: ANN001
            raise TimeoutError("no answer")

    engine = _engine(tmp_path)
    engine._client = _Fails()
    assert await engine._code_review_gate(STATE) is None
    note = cr.write_note(engine.quest_root, engine.quest_root / "code")
    assert "not compared with the plan by another model" in note and "the call failed" in note
    # The same model as the writer, and the cap: also disclosed.
    same = _engine(tmp_path / "b", other=False)
    same._client = _Reader()
    await same._code_review_gate(STATE)
    assert "the reader is the model that wrote the code" in cr.write_note(same.quest_root, same.quest_root / "code")
    capped = _engine(tmp_path / "c")
    capped._client = _Reader()
    for i in range(cr.MAX_REVIEWS):
        (capped.quest_root / "code" / "simulate.py").write_text(SIM_FULL + f"# v{i}" + chr(10), encoding="utf-8")
        await capped._code_review_gate(STATE)
    # A successful reading discloses nothing.
    fine = _engine(tmp_path / "d")
    fine._client = _Reader()
    await fine._code_review_gate(STATE)
    assert cr.write_note(fine.quest_root, fine.quest_root / "code") == ""


@pytest.mark.asyncio
async def test_code_changed_after_the_reading_is_read_again_before_the_run(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    reader = _Reader()
    engine._client = reader
    assert await engine._code_review_gate(STATE) is None and len(reader.prompts) == 1
    # A repair of a gate rewrote the code: the version that will run has not been read.
    (engine.quest_root / "code" / "simulate.py").write_text(SIM_OTHER, encoding="utf-8")
    reader.missing = ("planned figure",)
    out = await engine._code_review_gate(STATE)
    assert len(reader.prompts) == 2 and out is not None and "decay.png" in out["exec_result"]["stderr_tail"]
    assert cr.load(engine.quest_root)["read_version"] == cr.version(engine.quest_root / "code")


@pytest.mark.asyncio
async def test_after_the_cap_the_final_version_is_said_not_to_have_been_read_again(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine._client = _Reader()
    for i in range(cr.MAX_REVIEWS):
        (engine.quest_root / "code" / "simulate.py").write_text(SIM_FULL + f"# v{i}" + chr(10), encoding="utf-8")
        await engine._code_review_gate(STATE)
    (engine.quest_root / "code" / "simulate.py").write_text(SIM_FULL + "# final" + chr(10), encoding="utf-8")
    await engine._code_review_gate(STATE)
    assert len(engine._client.prompts) == cr.MAX_REVIEWS == 3
    record = cr.load(engine.quest_root)
    assert "allowed in a quest" in record["skipped"]
    note = cr.write_note(engine.quest_root, engine.quest_root / "code")
    assert "final version was not read again by another model" in note
    assert "final version of the code was not read again" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    again = Engine(engine.config, resume_quest_id=engine.quest_id)
    assert cr.load(again.quest_root)["reviews_total"] == 3, "a resume keeps the count"


class _Partial(_Reader):
    """Answers only the first requirement of a request, unless told to answer the rest when asked again."""

    def __init__(self, answers_again: bool) -> None:
        super().__init__()
        self.answers_again = answers_again

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        reqs = json.loads(prompt.split("```json", 1)[1].split("```", 1)[0])
        shown = reqs[:1] if (len(self.prompts) == 1 or not self.answers_again) else reqs
        return json.dumps({"requirements": [{"id": r["id"], "status": "implemented", "file": "simulate.py", "line": 1}
                                            for r in shown]})


@pytest.mark.asyncio
async def test_requirements_a_reply_leaves_unanswered_are_asked_once_more(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    reader = _Partial(answers_again=True)
    engine._client = reader
    assert await engine._code_review_gate(STATE) is None
    assert len(reader.prompts) == 2
    first = json.loads(reader.prompts[0].split("```json", 1)[1].split("```", 1)[0])
    again = json.loads(reader.prompts[1].split("```json", 1)[1].split("```", 1)[0])
    assert [r["id"] for r in again] == [r["id"] for r in first[1:]], "only what was not answered is asked again"
    record = cr.load(engine.quest_root)
    assert len(record["verdicts"]) == len(first) and "not_reviewed" not in record


@pytest.mark.asyncio
async def test_what_is_still_unanswered_is_recorded_per_requirement_and_an_unreviewed_headline_is_said(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    reader = _Partial(answers_again=False)
    engine._client = reader
    await engine._code_review_gate(STATE)
    assert len(reader.prompts) == 2, "asked once more and never a third time"
    record = cr.load(engine.quest_root)
    judged = {v["text"] for v in record["verdicts"]}
    assert record["not_reviewed"] and judged and not judged & set(record["not_reviewed"])
    assert len(judged) + len(record["not_reviewed"]) == record["requirements"]
    assert record["not_reviewed_headline"] is False  # the headline (R1) was answered
    # An unreviewed headline is said plainly.
    engine2 = _engine(tmp_path / "b")
    class _SkipHeadline(_Reader):
        async def chat(self, messages, **kw):  # noqa: ANN001
            prompt = messages[-1]["content"]
            self.prompts.append(prompt)
            reqs = json.loads(prompt.split("```json", 1)[1].split("```", 1)[0])
            return json.dumps({"requirements": [{"id": r["id"], "status": "implemented", "file": "simulate.py", "line": 1}
                                                for r in reqs if "quality_factor" not in r["text"]]})
    engine2._client = _SkipHeadline()
    await engine2._code_review_gate(STATE)
    note = cr.write_note(engine2.quest_root, engine2.quest_root / "code")
    assert "did not check these requirements" in note and "quality_factor" in note and "main quantity among them" in note


@pytest.mark.asyncio
async def test_the_readings_in_a_quest_are_capped(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine._client = _Reader()
    for i in range(cr.MAX_REVIEWS + 2):
        (engine.quest_root / "code" / "simulate.py").write_text(SIM_FULL + f"# version {i}\n", encoding="utf-8")
        await engine._code_review_gate(STATE)
    assert len(engine._client.prompts) == cr.MAX_REVIEWS
    assert "readings of the code allowed in a quest" in cr.load(engine.quest_root)["skipped"]
