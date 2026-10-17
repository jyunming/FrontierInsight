"""A different model reads the generated code against the plan's required outputs before the run
(core/code_review.py). Fake models, neutral topics (a mass on a damped spring)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from core import code_review as cr
from core import finding_repair as fr
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
WITH_PLAN: dict[str, Any] = {**STATE, "design": {**DESIGN, "protocol": PROTOCOL}}
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


REPAIR_HEAD = "# Code That Does Not Do What the Plan Requires"
TEST_HEAD = "# A Test That Shows One Finding"
pytestmark = pytest.mark.usefixtures("this_python")

#: A test a fake code model writes: it fails on the code as it is and passes once the file it was asked about has changed.
#: (The first run notes the file's hash beside the test and fails; later runs fail while the file is unchanged.)
CHANGED_TEST = '''import hashlib, pathlib
FILE = pathlib.Path(__file__).resolve().parents[2] / "code" / "{rel}"
NOTE = pathlib.Path(__file__).with_suffix(".orig")


def test_the_file_was_changed():
    now = hashlib.sha256(FILE.read_bytes()).hexdigest()
    if not NOTE.exists():
        NOTE.write_text(now)
    assert now != NOTE.read_text(), "{rel} is as it was when the problem was found"
'''
PASSING_TEST = "def test_nothing_is_wrong():\n    assert True\n"


def code_model_answer(owner: Any, prompt: str) -> str | None:
    """What a fake code-writing model answers: ``None`` when ``prompt`` is not one of its requests; a test for a request for a
    test (one that passes when ``owner.undemonstrated`` holds the file); the file or function from ``owner.repaired`` for a
    request for a repair (nothing when the test gave none, which is a reply with no changed file)."""
    head = prompt.lstrip()
    if not (head.startswith(TEST_HEAD) or head.startswith(REPAIR_HEAD)):
        return None
    rel = re.search(r"^### (\S+)", prompt, re.M).group(1)  # type: ignore[union-attr]
    scoped = "(the function `" in prompt.split("### ", 1)[1].splitlines()[0]
    if head.startswith(TEST_HEAD):
        owner.test_prompts.append(prompt)
        body = PASSING_TEST if rel in getattr(owner, "undemonstrated", ()) else CHANGED_TEST.format(rel=rel)
        return f"```python\n{body}```"
    owner.repair_prompts.append(prompt)
    count = len(owner.repair_prompts)
    text = owner.repaired.get(rel)
    text = text(count) if callable(text) else text
    if text is None:
        return ""
    return f"```python\n{text}```" if scoped else f"```python\n# file: {rel}\n{text}```"


def repair_answer(owner: Any, prompt: str) -> str:
    return code_model_answer(owner, prompt) or ""


class _Reader:
    def __init__(self, missing: tuple[str, ...] = (), headline: bool = False, fixed_by: str = "") -> None:
        self.missing, self.headline, self.prompts = missing, headline, []
        self.repair_prompts: list[str] = []
        self.test_prompts: list[str] = []
        self.repaired: dict[str, Any] = {}
        self.fixed_by = fixed_by  # a text whose presence in the code the reader is shown means the problem is repaired

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        answered = code_model_answer(self, prompt)
        if answered is not None:
            return answered
        assert prompt.lstrip().startswith("# Code Against the Plan"), prompt[:80]
        self.prompts.append(prompt)
        reqs = json.loads(prompt.split("```json", 1)[1].split("```", 1)[0])
        mended = bool(self.fixed_by) and self.fixed_by in prompt.split("# The code", 1)[1]
        out = []
        for r in reqs:
            gone = any(m in r["text"] for m in self.missing) and not mended
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
async def test_a_missing_quantity_goes_to_a_repair_request_of_its_own_and_the_changed_code_is_read_again(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    reader = _Reader(missing=("settling_time",), fixed_by="# settling_time is computed")
    reader.repaired["experiment.py"] = ANALYSIS + "# settling_time is computed\n"
    engine._client = reader
    assert await engine._code_review_gate(WITH_PLAN) is None, "the code goes on: nothing is returned as a failed run"
    assert len(reader.prompts) == 2 and len(reader.repair_prompts) == 1
    asked = reader.repair_prompts[0]
    assert asked.lstrip().startswith(REPAIR_HEAD) and "has NOT crashed" in asked
    assert "a mass on a damped spring" in asked, "the writer of the repair sees the topic"
    assert "Q = m * w / c" in asked, "and the plan's equations"
    assert "settling_time" in asked and "is missing from the code" in asked and "### experiment.py" in asked
    assert "0.123456789" not in asked and "expected" not in asked.split("# The plan's equations", 1)[1].split("# The findings", 1)[0]
    assert "just failed" not in asked and "crash" not in asked.replace("has NOT crashed", "").replace("not crashed", "")
    record = cr.load(engine.quest_root)
    assert record["send_backs"] == 1 and record["repair_calls"] == 1 and record["same_model"] is False
    assert [r["open"] for r in record["readings"]] == [1, 0], "both requirements are one issue"
    assert record["converged"] is True
    assert record["final"] is True and record["final_read"] is True and cr.not_computed(engine.quest_root, engine.quest_root / "code") == []
    assert "# settling_time is computed" in (engine.quest_root / "code" / "experiment.py").read_text(encoding="utf-8")
    assert "open issues per reading" in cr.metrics_line(record) and "1 issue(s) were repaired" in cr.metrics_line(record)
    # The version that will run is already read: the late call asks nothing.
    assert await engine._code_review_gate(WITH_PLAN, final=True) is None and len(reader.prompts) == 2


@pytest.mark.asyncio
async def test_a_repair_that_changes_nothing_ends_the_repairs_and_what_is_open_stands(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    reader = _Reader(missing=("settling_time",))  # the repair answers with no file
    engine._client = reader
    assert await engine._code_review_gate(STATE) is None
    assert len(reader.prompts) == 1 and len(reader.test_prompts) == 1, "one test shows it"
    assert len(reader.repair_prompts) == fr.REPAIR_CALLS + fr.OTHER_CALLS, "the code model's tries, then the other model's one"
    record = cr.load(engine.quest_root)
    assert record["final"] is True and record["send_backs"] == 1
    assert [v["kind"] for v in cr.not_computed(engine.quest_root, engine.quest_root / "code")] == ["dependent variable"]
    # The same code is read once; asked again, it is not read or repaired again.
    assert await engine._code_review_gate(STATE) is None and len(reader.prompts) == 1
    assert len(reader.repair_prompts) == fr.REPAIR_CALLS + fr.OTHER_CALLS


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
    assert await engine._code_review_gate(STATE, final=True) is None
    left = cr.not_computed(engine.quest_root, engine.quest_root / "code")
    assert [v["kind"] for v in left] == ["figure"]
    note = cr.write_note(engine.quest_root, engine.quest_root / "code")
    assert "decay.png" in note and "not computed" in note and "limitations" in note
    # Other code, other record: the note follows the code that ran.
    (engine.quest_root / "code" / "simulate.py").write_text(SIM_OTHER, encoding="utf-8")
    q, c = engine.quest_root, engine.quest_root / "code"
    assert cr.write_note(q, c) == "", "the writer is not told the code was read before it changed"
    assert "changed afterwards" in cr.findings_block(q, c), "the quality report says it"


HEADLINE = "the dependent variable quality_factor is computed and reported in the result"


@pytest.mark.asyncio
async def test_a_reviewer_s_word_alone_never_stops_a_quest_the_results_decide(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine._client = _Reader(missing=("quality_factor",))
    out = await engine._code_review_gate(STATE, final=True)
    assert out is None, "nothing stops before the run"
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
    assert await engine._code_review_gate(state, final=True) is None
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
    assert note == "", "the writer is not told that no second model read the code"
    block = cr.findings_block(engine.quest_root, engine.quest_root / "code")
    assert "not compared with the plan by another model" in block and "the call failed" in block, "the quality report says it"
    # The same model as the writer, and the cap: also disclosed.
    same = _engine(tmp_path / "b", other=False)
    same._client = _Reader()
    await same._code_review_gate(STATE)
    assert "the reader is the model that wrote the code" in cr.findings_block(same.quest_root, same.quest_root / "code")
    capped = _engine(tmp_path / "c")
    capped._client = _Reader()
    for i in range(cr.MAX_REVIEWS):
        (capped.quest_root / "code" / "simulate.py").write_text(SIM_FULL + f"# v{i}" + chr(10), encoding="utf-8")
        await capped._code_review_gate(STATE, final=True)
    # A successful reading discloses nothing.
    fine = _engine(tmp_path / "d")
    fine._client = _Reader()
    await fine._code_review_gate(STATE)
    assert cr.write_note(fine.quest_root, fine.quest_root / "code") == ""


@pytest.mark.asyncio
async def test_code_changed_after_the_reading_is_read_again_before_the_run_and_nothing_is_repaired_then(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    reader = _Reader()
    engine._client = reader
    assert await engine._code_review_gate(STATE) is None and len(reader.prompts) == 1
    # A repair of a gate rewrote the code: the version that will run has not been read.
    (engine.quest_root / "code" / "simulate.py").write_text(SIM_OTHER, encoding="utf-8")
    reader.missing = ("planned figure",)
    assert await engine._code_review_gate(STATE, final=True) is None
    assert len(reader.prompts) == 2 and reader.repair_prompts == [], "the version that runs is not repaired: nobody would read it"
    record = cr.load(engine.quest_root)
    assert record["read_version"] == cr.version(engine.quest_root / "code") and record["final_read"] is True
    assert [v["kind"] for v in cr.not_computed(engine.quest_root, engine.quest_root / "code")] == ["figure"]


@pytest.mark.asyncio
async def test_the_last_reading_is_kept_for_the_version_that_will_run(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    reader = _Reader()
    engine._client = reader
    sim = engine.quest_root / "code" / "simulate.py"
    for i in range(cr.MAX_REVIEWS):  # versions that are not the last: the early call never spends the reserved reading
        sim.write_text(SIM_FULL + f"# v{i}" + chr(10), encoding="utf-8")
        await engine._code_review_gate(STATE)
    assert len(reader.prompts) == cr.MAX_REVIEWS - cr.RESERVED_FOR_FINAL
    record = cr.load(engine.quest_root)
    assert record["final_read"] is False, "the last version was not read yet"
    # The version that will run is read, with the reserved reading, however many versions came before.
    assert await engine._code_review_gate(STATE, final=True) is None
    assert len(reader.prompts) == cr.MAX_REVIEWS
    assert cr.load(engine.quest_root)["final_read"] is True and cr.load(engine.quest_root)["read_version"] == cr.version(sim.parent)


@pytest.mark.asyncio
async def test_after_the_cap_the_final_version_is_said_not_to_have_been_read_again(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine._client = _Reader()
    for i in range(cr.MAX_REVIEWS):
        (engine.quest_root / "code" / "simulate.py").write_text(SIM_FULL + f"# v{i}" + chr(10), encoding="utf-8")
        await engine._code_review_gate(STATE, final=True)
    (engine.quest_root / "code" / "simulate.py").write_text(SIM_FULL + "# final" + chr(10), encoding="utf-8")
    await engine._code_review_gate(STATE, final=True)
    assert len(engine._client.prompts) == cr.MAX_REVIEWS == 3
    record = cr.load(engine.quest_root)
    assert "allowed in a quest" in record["skipped"] and record["final_read"] is False
    assert cr.write_note(engine.quest_root, engine.quest_root / "code") == ""
    assert "not read by another model" in cr.findings_block(engine.quest_root, engine.quest_root / "code")
    assert "final version of the code was not read again" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    again = Engine(engine.config, resume_quest_id=engine.quest_id)
    assert cr.load(again.quest_root)["reviews_total"] == 3, "a resume keeps the count"


@pytest.mark.asyncio
async def test_what_the_last_reading_left_open_stays_open_for_a_version_no_one_read(tmp_path: Path) -> None:
    # Live: p2 kinetics (1792232341) was repaired after the run with the readings spent; the 18 findings the last reading
    # left open, a high-severity one among them, were dropped from the record, and the paper's limitations named none.
    engine = _engine(tmp_path)
    reader = _Reader()
    reader.missing = ("planned figure",)
    engine._client = reader
    sim = engine.quest_root / "code" / "simulate.py"
    for i in range(cr.MAX_REVIEWS):
        sim.write_text(SIM_FULL + f"# v{i}" + chr(10), encoding="utf-8")
        await engine._code_review_gate(STATE, final=True)
    read = [v["text"] for v in cr.not_computed(engine.quest_root, sim.parent)]
    assert read, "the last reading leaves the figure open"
    sim.write_text(SIM_FULL + "# repaired after the run" + chr(10), encoding="utf-8")
    await engine._code_review_gate(STATE, final=True)
    record = cr.load(engine.quest_root)
    assert "allowed in a quest" in record["skipped"] and record["final_read"] is False
    assert [v["text"] for v in cr.not_computed(engine.quest_root, sim.parent)] == read
    assert read[0] in cr.write_note(engine.quest_root, sim.parent)
    block = cr.findings_block(engine.quest_root, sim.parent)
    assert read[0] in block and "final code was not read again" in block
    assert "stays open" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


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
    assert cr.write_note(engine2.quest_root, engine2.quest_root / "code") == ""
    block = cr.findings_block(engine2.quest_root, engine2.quest_root / "code")
    assert "did not check these requirements" in block and "quality_factor" in block and "main quantity among them" in block


@pytest.mark.asyncio
async def test_the_readings_in_a_quest_are_capped(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine._client = _Reader()
    for i in range(cr.MAX_REVIEWS + 2):
        (engine.quest_root / "code" / "simulate.py").write_text(SIM_FULL + f"# version {i}\n", encoding="utf-8")
        await engine._code_review_gate(STATE, final=True)
    assert len(engine._client.prompts) == cr.MAX_REVIEWS
    assert "readings of the code allowed in a quest" in cr.load(engine.quest_root)["skipped"]


# ---- what is left open is stated, in full, in the writer's note and in the paper's own block ---------------------------


def _record(tmp_path: Path, *, missing: int = 2, logic: bool = True, **extra: Any) -> tuple[Path, Path]:
    """A quest folder with a code folder and a finished record for that code: ``missing`` quantities not computed, one
    planned figure computed differently, and (with ``logic``) one still-flagged high-severity finding, one the tests
    could not show, one plan conflict and one the results contradict."""
    root = tmp_path / "q"
    code = root / "code"
    code.mkdir(parents=True)
    (code / "simulate.py").write_text(SIM_FULL, encoding="utf-8")
    problems: list[dict[str, Any]] = [
        {"id": f"R{i}", "kind": "metric", "text": f"the result reports metric_{i} (a quantity)", "status": "missing",
         "headline": False, "fid": f"F-m{i}"} for i in range(missing)]
    problems.append({"id": "R90", "kind": "figure", "text": "the script draws the planned figure shape.png",
                     "status": "different", "file": "experiment.py", "line": 5, "fid": "F-d"})
    if logic:
        problems += [
            {"id": "L1", "kind": "logic", "text": "A statistic is set to None when every repeat agrees", "status": "different",
             "file": "experiment.py", "line": 74, "severity": "high", "fid": "F-l1"},
            {"id": "L2", "kind": "logic", "text": "The generator ignores the seed of the case", "status": "different",
             "file": "simulate.py", "line": 46, "severity": "low", "fid": "F-l2"},
            {"id": "L3", "kind": "logic", "text": "Two definitions of one thing disagree", "status": "different",
             "plan_conflict": True, "note": "the plan defines it twice", "fid": "F-l3"},
            {"id": "L4", "kind": "logic", "text": "The results are declared constant", "status": "different",
             "contradicted": True, "fid": "F-l4"},
        ]
    rec = {"version": cr.version(code), "read_version": cr.version(code), "final": True, "problems": problems,
           "issues": {"F-l2": {"outcome": "not demonstrated"}}, "converged": False, **extra}
    cr.save(root, rec)
    return root, code


def test_the_note_names_every_kind_of_open_finding_not_only_the_missing_quantities(tmp_path: Path) -> None:
    root, code = _record(tmp_path)
    note = cr.write_note(root, code)
    assert "metric_0" in note and "metric_1" in note and "not computed" in note
    assert "planned figure shape.png" in note and "computed differently from the plan" in note
    assert "A statistic is set to None" in note and "[high]" in note, "a flagged logic problem next to missing quantities"
    assert "The generator ignores the seed" in note and "unconfirmed" in note, "a finding no test could show"
    assert "Two definitions of one thing disagree" in note and "contradicts itself" in note
    assert "declared constant" not in note, "a finding the results contradict is not said"
    assert "stopped because a round did not reduce" not in note, "an internal state the writer is not told"
    assert "ended because a round did not reduce" in cr.findings_block(root, code), "the quality report says it"
    # The same facts without the requirement findings still all appear.
    root2, code2 = _record(tmp_path / "b", missing=0)
    assert "A statistic is set to None" in cr.write_note(root2, code2)


def test_a_long_list_in_the_note_says_how_many_more_there_are_and_where(tmp_path: Path) -> None:
    root, code = _record(tmp_path, missing=14)
    note = cr.write_note(root, code)
    shown = [i for i in range(14) if f"metric_{i} " in note]
    assert len(shown) == cr.NOTE_CAP - 1 or len(shown) == cr.NOTE_CAP, "the note keeps a cap for the prompt's size"
    assert f"and {14 + 1 - cr.NOTE_CAP} more (all are listed in .fi/code_review.json)" in note
    block = cr.findings_block(root, code)
    assert all(f"metric_{i} " in block for i in range(14)), "the paper's block has room for all of them"
    assert "more (all are listed" not in block
    big, bcode = _record(tmp_path / "big", missing=cr.BLOCK_CAP + 5)
    assert "and 5 more (all are listed in .fi/code_review.json)" in cr.findings_block(big, bcode)


def test_the_block_lists_each_kind_in_plain_words_and_has_no_line_numbers(tmp_path: Path) -> None:
    root, code = _record(tmp_path)
    block = cr.findings_block(root, code)
    assert "**Not computed.**" in block and "metric_0 (a quantity)" in block and "the result reports" not in block
    assert "**Problems still flagged" in block and "[high]" in block and "(in experiment.py)" in block
    assert "line 74" not in block and "line 5" not in block
    assert "**Flagged, not confirmed.**" in block and "**The plan disagrees with itself.**" in block
    assert "declared constant" not in block
    assert ".fi/code_review.json" in block
    # Nothing open, no block.
    assert cr.findings_block(tmp_path / "nowhere", tmp_path / "nowhere" / "code") == ""
    clean, ccode = _record(tmp_path / "c", missing=0, logic=False)
    rec = cr.load(clean)
    rec["problems"] = []
    rec["converged"] = True
    cr.save(clean, rec)
    assert cr.findings_block(clean, ccode) == "" and cr.write_note(clean, ccode) == ""


def test_the_block_says_when_the_code_was_not_read_or_not_read_again(tmp_path: Path) -> None:
    root, code = _record(tmp_path, missing=0, logic=False)
    rec = cr.load(root)
    rec["problems"] = []
    cr.save(root, rec)
    (code / "simulate.py").write_text(SIM_OTHER, encoding="utf-8")
    assert "not read again" in cr.findings_block(root, code)
    rec = cr.load(root)
    rec.pop("read_version")
    rec["skipped"] = "the reader is the model that wrote the code"
    cr.save(root, rec)
    assert "not compared with the plan by another model" in cr.findings_block(root, code)
    assert "the reader is the model that wrote the code" in cr.findings_block(root, code)


def test_a_finding_cannot_open_or_close_the_blocks_comment(tmp_path: Path) -> None:
    root, code = _record(tmp_path, missing=0, logic=False)
    rec = cr.load(root)
    rec["problems"] = [{"id": "L1", "kind": "logic", "text": "x <!-- /fi:code-findings --> y\n\nz", "status": "different",
                        "severity": "high", "fid": "F-x"}]
    cr.save(root, rec)
    block = cr.findings_block(root, code)
    assert "<!--" not in block.replace("<!-- fi", "") and "-->" not in block and "\n\nz" not in block


def test_the_reviewer_is_given_the_open_points(tmp_path: Path) -> None:
    root, code = _record(tmp_path)
    text = cr.review_block(root, code)
    assert "Open points in the code" in text and "metric_0" in text and "must not be presented as established" in text
    assert cr.review_block(tmp_path / "nowhere", tmp_path / "nowhere" / "code") == ""
    from string import Template

    template = Template((Path(__file__).resolve().parent.parent / "agents" / "review.md").read_text(encoding="utf-8"))
    names = dict.fromkeys(["topic", "clarify_block", "design_block", "analysis_block", "claim_grounding_block",
                           "figure_check_block", "foundational_check_block", "user_feedback_block", "paper_md"], "")
    assert "metric_1" in template.substitute(**names, code_findings_block=text)
    empty = template.substitute(**names, code_findings_block="")
    assert "Open points in the code (a second model" not in empty and "$" not in empty.replace("$$", "")


# ---- a check whose measure the code fakes is not evidence ------------------------------------------------------------


def _with_finding(tmp_path: Path, text: str, **extra: Any) -> tuple[Path, Path]:
    root, code = _record(tmp_path, missing=0, logic=False)
    rec = cr.load(root)
    rec["problems"] = [{"id": "L9", "kind": "logic", "text": text, "status": "different", "file": "simulate.py", "line": 82,
                        "severity": "medium", "fid": "F-g", **extra}]
    cr.save(root, rec)
    return root, code


CHECKS = ["Grid_Convergence_Check", "Poisson_Zero_Check", "Absorption_Check", "Convergence"]
FAKED = ("Grid_Convergence_Check assigns coarse and fine critical dimensions to identical values from an analytical formula "
         "without executing simulations on coarse and fine grids.")


def test_a_check_named_in_an_open_finding_is_flagged_and_the_others_are_not(tmp_path: Path) -> None:
    root, code = _with_finding(tmp_path, FAKED)
    flagged = cr.flagged_checks(root, code, CHECKS)
    assert list(flagged) == ["Grid_Convergence_Check"]
    assert "analytical formula" in flagged["Grid_Convergence_Check"] and "82" not in flagged["Grid_Convergence_Check"]


def test_the_name_is_matched_whatever_its_case_punctuation_or_underscores(tmp_path: Path) -> None:
    root, code = _with_finding(tmp_path, "The grid-convergence CHECK returns a constant instead of running the grids.")
    assert list(cr.flagged_checks(root, code, CHECKS[:3])) == ["Grid_Convergence_Check"]


def test_a_finding_that_only_shares_common_words_with_a_check_does_not_match(tmp_path: Path) -> None:
    root, code = _with_finding(tmp_path, "The grid is never refined, no check of the zero count is made; absorption is a closed form.")
    assert cr.flagged_checks(root, code, CHECKS) == {}, "words of a multi-word name far apart, or only some of them, are not a name"
    root2, code2 = _with_finding(tmp_path / "b", "The grid size never changes, so the convergence check is vacuous.")
    assert cr.flagged_checks(root2, code2, ["Grid_Convergence_Check"]) == {}, "two of the three words, not in order"


def test_a_one_word_check_name_matches_only_as_a_whole_word_whatever_its_case(tmp_path: Path) -> None:
    root, code = _with_finding(tmp_path, "The Convergence test returns a constant instead of running the grids.")
    assert list(cr.flagged_checks(root, code, ["convergence", "Absorption"])) == ["convergence"], "case does not matter"
    # The rule is the exact word: a one-word check named "Convergence" IS named by "Grid convergence ..." (the word is there).
    root2, code2 = _with_finding(tmp_path / "b", "Grid convergence is computed from a closed form.")
    assert list(cr.flagged_checks(root2, code2, ["Convergence"])) == ["Convergence"]
    root3, code3 = _with_finding(tmp_path / "c", "The Grid_Convergence_Check is a constant.")
    assert cr.flagged_checks(root3, code3, ["Convergence"]) == {}, "a longer name does not hold the one-word name as a word"


def test_a_fixed_or_contradicted_finding_does_not_flag_the_check(tmp_path: Path) -> None:
    root, code = _with_finding(tmp_path, FAKED, contradicted=True)
    assert cr.flagged_checks(root, code, CHECKS) == {}, "the run's results contradict the finding"
    root2, code2 = _with_finding(tmp_path / "b", FAKED)
    rec = cr.load(root2)
    rec["problems"] = []  # the final reading no longer flags it
    cr.save(root2, rec)
    assert cr.flagged_checks(root2, code2, CHECKS) == {}



def test_a_flagged_check_is_not_counted_by_the_note_the_criteria_or_the_evidence_helpers() -> None:
    from core import criteria as crit
    from core import oracle_check as oc

    judged = [{"name": "Grid_Convergence_Check", "value": 0.0, "expected": 0.0, "limit": 0.01, "passed_by_engine": True,
               "measured_by": "engine"},
              {"name": "Absorption_Check", "value": 0.16, "expected": 0.16, "limit": 0.01, "passed_by_engine": True,
               "measured_by": "engine"}]
    marked = oc.with_reader_flags(judged, {"Grid_Convergence_Check": "the grid sizes are not simulated"})
    assert "flagged_by_reader" not in judged[0], "the list given is not changed"
    assert oc.engine_passed(marked) == ["Absorption_Check"] and oc.not_passed(marked) == ["Grid_Convergence_Check"]
    note = oc.analysis_note(marked)
    assert "Grid_Convergence_Check: measured 0, expected 0 within 0.01: not shown: a second model found" in note
    assert "Absorption_Check: measured 0.16, expected 0.16 within 0.01: passed" in note
    items = [{"name": "grid", "oracle": "Grid_Convergence_Check", "use": "error", "direction": "lower", "tolerance": 0.01,
              "target": 0.0},
             {"name": "absorption", "oracle": "Absorption_Check", "use": "error", "direction": "lower", "tolerance": 0.01,
              "target": 0.0}]
    rows = {r["name"]: r for r in crit.evaluate(items, judged=marked, series={})}
    assert rows["grid"]["counts"] is True and rows["grid"]["met"] is False and "not shown" in rows["grid"]["why"]
    from core import improve as imp

    assert not imp.all_met(list(rows.values())), "a flagged check is an unmet check, so the improve loop does not stop as done"
    only = [rows["grid"]]
    assert not imp.all_met(only) and imp.counted(only) == only
    assert "not shown" in crit.summary_line(list(rows.values()))
    assert rows["absorption"]["counts"] is True and rows["absorption"]["met"] is True


def test_a_flag_sticks_to_later_versions_until_a_newer_reading_clears_it(tmp_path: Path) -> None:
    root, code = _with_finding(tmp_path, FAKED)
    (code / "simulate.py").write_text(SIM_FULL + "# version B, an improve round, nobody read it\n", encoding="utf-8")
    assert list(cr.flagged_checks(root, code, CHECKS)) == ["Grid_Convergence_Check"], "no new reading: still flagged"
    rec = cr.load(root)  # a newer reading of version B that no longer flags it
    rec.update(version=cr.version(code), read_version=cr.version(code), problems=[], final=True)
    cr.save(root, rec)
    assert cr.flagged_checks(root, code, CHECKS) == {}
    # a reading that failed keeps the earlier reading's findings
    root2, code2 = _with_finding(tmp_path / "b", FAKED)
    rec2 = cr.load(root2)
    rec2["read_problems"] = rec2.pop("problems")
    cr.save(root2, rec2)
    assert list(cr.flagged_checks(root2, code2, CHECKS)) == ["Grid_Convergence_Check"]


def test_an_improve_round_cannot_keep_a_version_by_passing_a_flagged_check(tmp_path: Path) -> None:
    from core import criteria as crit
    from core import improve as imp
    from core.engine import _oracle_with_reader_flags

    root, code = _with_finding(tmp_path, FAKED)
    (code / "simulate.py").write_text(SIM_FULL + "# version B\n", encoding="utf-8")
    judged = [{"name": "Grid_Convergence_Check", "value": 0.0, "expected": 0.0, "limit": 0.01, "passed_by_engine": True,
               "measured_by": "engine"}]
    items = [{"name": "grid", "oracle": "Grid_Convergence_Check", "use": "error", "direction": "lower", "tolerance": 0.01,
              "target": 0.0}]
    rows = crit.evaluate(items, judged=_oracle_with_reader_flags(root, judged), series={})
    assert rows[0]["counts"] is True and rows[0]["met"] is False and not imp.all_met(rows)


def test_a_finding_a_repair_settled_no_longer_flags_its_check(tmp_path: Path) -> None:
    from core import code_findings

    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text("def run_cell(cell):\n    return {}\n", encoding="utf-8")
    finding = {"kind": "logic", "text": "The Grid_Convergence_Check sets coarse and fine to one value", "fid": "F-1"}
    record = {"version": cr.version(code), "read_version": cr.version(code), "final": True, "problems": [finding],
              "issues": {"F-1": {"outcome": code_findings.OPEN}}}
    cr.save(tmp_path, record)
    assert "Grid_Convergence_Check" in cr.flagged_checks(tmp_path, code, ["Grid_Convergence_Check"])
    for settled in (code_findings.FIXED, code_findings.GONE):
        record["issues"]["F-1"]["outcome"] = settled
        cr.save(tmp_path, record)
        assert cr.flagged_checks(tmp_path, code, ["Grid_Convergence_Check"]) == {}, settled
    for unsettled in (code_findings.NOT_DEMONSTRATED, code_findings.GAVE_UP, code_findings.REOPENED):
        record["issues"]["F-1"]["outcome"] = unsettled
        cr.save(tmp_path, record)
        assert "Grid_Convergence_Check" in cr.flagged_checks(tmp_path, code, ["Grid_Convergence_Check"]), unsettled


def test_a_one_word_check_name_with_punctuation_is_still_matched_as_its_word() -> None:
    assert cr._names_check("The convergence test returns a constant.", "Convergence!")
    assert cr._names_check("The convergence test returns a constant.", " CONVERGENCE ")
