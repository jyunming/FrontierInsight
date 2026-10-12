"""The code-writing side of "correct at source" in the engine: the model's functions written from their step cards and the
literature passages, the call cap, the simulate.py request that tells the writer the steps are already in the package, and
the second, independently written version of the key functions (core/second_impl.py) inside the gate pass. Fake models,
neutral models (a mass on a damped spring)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core import code_layout, code_review, function_steps as fs, quality_report, second_impl
from core.config import Config, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from core.provider import node_provider_for
from tests.test_equation_gate import SPRING, _engine, _plan_md

STEPS = [
    {"id": "S1", "name": "mean quality", "does": "average two quality factors", "inputs": {"q1": "dimensionless", "q2": "dimensionless"},
     "output": "dimensionless", "uses": ["E2"], "source": "Smith 2020",
     "example": {"inputs": {"q1": 2.0, "q2": 4.0}, "expected_formula": "(q1 + q2) / 2"}},
    {"id": "S3", "name": "cycle count", "does": "the number of whole cycles, the integer part of n", "inputs": {"n": "1"},
     "output": "count"},
]
PROTOCOL = {"grid": {"c": [0.5, 1.0]}, "model": {**SPRING, "steps": STEPS}}
STATE: dict[str, Any] = {"design": {"protocol": PROTOCOL}}
OUTLINE = {"scaffold": "x = 1", "functions": [
    {"name": "frequency", "signature": "def frequency(k, m) -> float:", "one_line_purpose": "natural frequency", "implements": "E1"},
    {"name": "quality", "signature": "def quality(m, w, c) -> float:", "one_line_purpose": "quality factor", "implements": "E2",
     "depends_on": ["frequency"]},
    {"name": "mean_quality", "signature": "def mean_quality(q1, q2) -> float:", "one_line_purpose": "average",
     "implements": "S1"},
    {"name": "count_cycles", "signature": "def count_cycles(n=7.5) -> int:", "one_line_purpose": "count", "implements": "S3"},
    {"name": "run_cell", "signature": "def run_cell(cell) -> dict:", "one_line_purpose": "one setting"},
], "constants": []}
STATE["implement_outline"] = OUTLINE
GOOD = {
    "frequency": "import math\n\n\n# E1\ndef frequency(k, m):\n    return math.sqrt(k / m)\n",
    "quality": "# E2\ndef quality(m, w, c):\n    return m * w / c\n",
    "mean_quality": "# S1\ndef mean_quality(q1, q2):\n    return (q1 + q2) / 2\n",
    "count_cycles": "# S3\ndef count_cycles(n):\n    return int(n)\n",
}
# What the second writer gives: the same maths, written separately (no label, no import of the package).
REFERENCE = {
    "frequency": "import math\n\n\ndef frequency(k, m):\n    return math.sqrt(k / m)\n",
    "quality": "def quality(m, w, c):\n    return m * w / c\n",
    "mean_quality": "def mean_quality(q1, q2):\n    return 0.5 * (q1 + q2)\n",
    "count_cycles": "def count_cycles(n):\n    return int(n)\n",
}
_FUNCTION = re.compile(r"# The function[^\n]*\n\n```python\n(?:#[^\n]*\n)*def (\w+)\(")


def fenced(code: str) -> str:
    return "```python\n" + code + "```"


class _Client:
    """A fake model: the main writer by function name, the second writer by function name, the judge in order."""

    def __init__(self, main: dict[str, list[str]] | None = None, second: dict[str, list[str]] | None = None,
                 judge: list[str] | None = None) -> None:
        self.main = {k: list(v) for k, v in (main or {}).items()}
        self.second = {k: list(v) for k, v in (second or {}).items()}
        self.judge = list(judge or [])
        self.calls: list[tuple[str, str]] = []

    def count(self, node: str) -> int:
        return sum(1 for n, _ in self.calls if n == node)

    async def chat(self, messages, **kw):  # noqa: ANN001
        node, prompt = kw.get("node", ""), messages[-1]["content"]
        self.calls.append((node, prompt))
        if node == "second_impl.adjudicate":
            assert self.judge, "unexpected request to the judge"
            return self.judge.pop(0)
        found = _FUNCTION.search(prompt)
        assert found, (node, prompt[:300])
        queue = (self.second if node == "second_impl" else self.main)[found.group(1)]
        assert queue, f"unexpected request to {node} for {found.group(1)}"
        return fenced(queue.pop(0)) if len(queue) > 1 or True else ""

    async def aclose(self) -> None:
        return None


class _Knowledge:
    enabled = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def asearch(self, query: str, **kw: Any) -> list[Any]:
        self.calls.append(query)
        return [SimpleNamespace(content="The mean quality factor is the average of the two quality factors. " * 4,
                                metadata={"title": "On quality", "authors": ["Smith, J"], "year": 2020, "doi": "10.1/x"})]


def _setup(tmp_path: Path, **kw: Any) -> tuple[Engine, _Client]:
    engine = _engine(tmp_path)
    _plan_md(engine, PROTOCOL)
    engine._plan_code_layout(STATE)  # what the plan step does: the quest is laid out as a package
    client = _Client(main={k: [v] for k, v in GOOD.items()}, **kw)
    engine._client = client
    return engine, client


async def _written(tmp_path: Path, **kw: Any) -> tuple[Engine, _Client]:
    engine, client = _setup(tmp_path, **kw)
    assert await engine._fill_model_functions(STATE, OUTLINE) is True
    (engine.quest_root / "code" / "simulate.py").write_text("from spring import model\n\n\ndef run_cell(cell):\n    return {}\n",
                                                            encoding="utf-8")
    return engine, client


# --- 1. the bug found offline -------------------------------------------------------------------------------------------


def test_the_description_of_an_equation_function_can_be_built() -> None:
    spec = fs.Spec(name="quality", signature="def quality(m, w, c):", purpose="quality factor", equation="E2")
    text = second_impl._spec_block(spec, PROTOCOL)
    assert "E2: Q = m * w / c" in text and "purpose: quality factor" in text


# --- 2. cards, passages and the call cap into the filler ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_filler_gets_the_cards_the_passages_and_the_call_cap(tmp_path: Path) -> None:
    engine, client = _setup(tmp_path)
    knowledge = _Knowledge()
    engine.knowledge = knowledge  # type: ignore[assignment]
    assert await engine._fill_model_functions(STATE, OUTLINE) is True
    by = {_FUNCTION.search(p).group(1): p for n, p in client.calls if n == "implement"}
    assert set(by) == {"frequency", "quality", "mean_quality", "count_cycles"}
    assert "what it does: average two quality factors" in by["mean_quality"], "the step card is in the request"
    assert "# What the literature says" in by["mean_quality"] and "average of the two quality factors" in by["mean_quality"]
    assert "On quality" in by["mean_quality"], "the paper's citation line leads the passage"
    assert "# What the literature says" not in by["frequency"], "an equation whose source names nothing searchable gets none"
    assert len(knowledge.calls) == 1, "only the card with a source was searched for"
    cache = json.loads((engine.fi_dir / "step_passages.json").read_text(encoding="utf-8"))
    assert "S1" in cache
    specs = fs.model_functions(OUTLINE, PROTOCOL)[0]
    filler = engine._new_function_filler(OUTLINE, "spring", PROTOCOL, specs, {}, [])
    assert filler.max_calls == fs.max_calls_for(len(specs)) == 16
    # a resume or a later pass does not search again
    again = await engine._step_passages(PROTOCOL, specs)
    assert "S1" in again and len(knowledge.calls) == 1


@pytest.mark.asyncio
async def test_a_failed_search_leaves_the_functions_to_be_written_without_passages(tmp_path: Path) -> None:
    engine, client = _setup(tmp_path)

    class Down(_Knowledge):
        async def asearch(self, query: str, **kw: Any) -> list[Any]:
            raise RuntimeError("down")

    engine.knowledge = Down()  # type: ignore[assignment]
    assert await engine._fill_model_functions(STATE, OUTLINE) is True
    assert all("# What the literature says" not in p for n, p in client.calls)


@pytest.mark.asyncio
async def test_the_simulate_request_says_the_steps_are_in_the_package_and_to_call_them(tmp_path: Path) -> None:
    engine, _client = await _written(tmp_path)
    block = engine._split_block(STATE)
    assert "ALREADY WRITTEN AND TESTED" in block
    assert "CALL them from simulate.py; do not rewrite them" in block
    assert "`mean_quality` (S1: mean quality)" in block and "`count_cycles` (S3: cycle count)" in block
    assert "`frequency` (" not in block.split("CALL them")[1], "an equation's function is not listed as a step"
    # a quest whose package has no step says nothing more than it did
    assert code_layout.steps_block(None) == "" and code_layout.steps_block([]) == ""
    assert code_layout.prefilled_block("p", "x = 1") == code_layout.prefilled_block("p", "x = 1", steps=[])


# --- 3. the gate, after the equation gate and before the oracle gate -----------------------------------------------------


@pytest.mark.asyncio
async def test_the_second_version_gate_runs_after_the_equation_gate_and_before_the_oracle_gate(tmp_path: Path) -> None:
    engine, _client = _setup(tmp_path)
    order: list[str] = []

    def stub(name: str, value: Any = None):
        async def f(*a: Any, **k: Any) -> Any:
            order.append(name)
            return value
        return f

    for name in ("_results_from_computation", "_static_gate", "_code_review_gate", "_equation_gate", "_second_version_gate"):
        setattr(engine, name, stub(name, False if name == "_second_version_gate" else None))
    engine._oracle_gate = stub("_oracle_gate", None)  # type: ignore[method-assign]
    engine._cannot_repair_stop = lambda: None  # type: ignore[method-assign]
    engine._code_problems_stop = lambda state: None  # type: ignore[method-assign]
    seed = engine.quest_root / "code" / "simulate.py"
    stop, code, repaired = await engine._gate_pass({}, None, None, seed)
    assert (stop, code, repaired) == (None, None, False)
    assert order.index("_equation_gate") < order.index("_second_version_gate") < order.index("_oracle_gate")
    assert order[order.index("_second_version_gate") + 1] == "_results_from_computation", "a repair is read again at once"


# --- 4. the check itself ---------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_versions_that_agree_are_logged_plainly_and_recorded(tmp_path: Path) -> None:
    engine, client = await _written(tmp_path)
    client.second = {k: [v] for k, v in REFERENCE.items()}
    assert await engine._second_version_gate(STATE, None, None) is False
    record = second_impl.load(engine.quest_root)
    assert {n: e["status"] for n, e in record["functions"].items()} == {
        "frequency": "agree", "quality": "agree", "mean_quality": "agree", "count_cycles": "agree"}
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert re.search(r"a second, independently written version of mean_quality agrees on \d+ inputs", log)
    assert client.count("second_impl") == 4 and client.count("second_impl.adjudicate") == 0
    assert second_impl.limitations(record) == [] and code_review.open_findings(engine.quest_root, engine.quest_root / "code")["second_version"] == []
    for node, prompt in client.calls:
        if node == "second_impl":
            body = GOOD[_FUNCTION.search(prompt).group(1)].split("\n", 1)[-1]
            assert body.strip().splitlines()[-1] not in prompt, "the writer never sees the first version"


@pytest.mark.asyncio
async def test_a_resume_reuses_the_finished_functions_and_asks_nothing(tmp_path: Path) -> None:
    engine, client = await _written(tmp_path)
    client.second = {k: [v] for k, v in REFERENCE.items()}
    await engine._second_version_gate(STATE, None, None)
    asked = client.count("second_impl")
    assert asked == 4
    again = Engine(engine.config)
    again._client = client
    assert await again._second_version_gate(STATE, None, None) is False
    assert client.count("second_impl") == asked, "a finished function is not asked again"
    assert second_impl.load(engine.quest_root)["calls"] == 4


@pytest.mark.asyncio
async def test_unresolved_is_a_finding_in_the_writer_note_the_quality_report_and_the_papers_limitations(tmp_path: Path) -> None:
    engine, client = await _written(tmp_path)
    wrong = "def mean_quality(q1, q2):\n    return q1 - q2\n"
    client.second = {"frequency": [REFERENCE["frequency"]], "quality": [REFERENCE["quality"]],
                     "count_cycles": [REFERENCE["count_cycles"]], "mean_quality": [wrong]}
    client.judge = [json.dumps({"verdict": "card_ambiguous", "reason": "the description allows both"})]
    assert await engine._second_version_gate(STATE, None, None) is False, "nothing is repaired, nothing stops"
    record = second_impl.load(engine.quest_root)
    assert record["functions"]["mean_quality"]["status"] == "unresolved"
    assert "mean_quality" in " ".join(second_impl.limitations(record)) or "mean quality" in " ".join(second_impl.limitations(record))
    found = code_review.open_findings(engine.quest_root, engine.quest_root / "code")
    assert len(found["second_version"]) == 1
    note = code_review.write_note(engine.quest_root, engine.quest_root / "code")
    assert "independently written" in note and "mean quality" in note
    block = code_review.findings_block(engine.quest_root, engine.quest_root / "code")
    assert "second, independent version of a key function was not settled" in block and "mean quality" in block
    # No second model read this code: the paper must not say one did
    assert "A second model read the code" not in block and "FI compared two independently written versions" in block
    from core.engine import _claim_findings_parts

    claim = " ".join(_claim_findings_parts(engine.quest_root))
    assert claim.startswith("TWO INDEPENDENTLY WRITTEN VERSIONS") and "SECOND MODEL'S READING" not in claim
    problems = quality_report._code_problems(quality_report._Quest(engine.quest_root))
    assert any(p["source"].endswith("second_impl.json") and p["status"] == "not fixed" for p in problems)
    assert "could not settle which is right" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_wrong_first_version_is_repaired_only_when_the_repair_still_passes_its_own_checks(tmp_path: Path) -> None:
    # count_cycles has no example, so its own checks are the import and the look at the code
    for repair, kept in (("# S3\ndef count_cycles(n):\n    return int(n)\n", True),
                         ("# S3\ndef count_cycles(n):\n    return 3\n", False)):
        root = tmp_path / ("kept" if kept else "dropped")
        engine, client = await _written(root)
        model = engine.quest_root / "code" / "spring" / "model.py"
        wrong = model.read_text(encoding="utf-8").replace("return int(n)", "return int(n) + 1")
        model.write_text(wrong, encoding="utf-8")
        client.second = {k: [v] for k, v in REFERENCE.items() if k == "count_cycles"}
        client.main = {"count_cycles": [repair]}
        client.judge = [json.dumps({"verdict": "main_wrong", "reason": "it adds one"})]
        key_only = [s for s in fs.model_functions(OUTLINE, PROTOCOL)[0] if s.name == "count_cycles"]
        assert key_only
        # only the function under test is compared (the others are already finished in a record of their own)
        pre = {"functions": {n: {"status": "agree", "rounds": 0, "inputs_tried": 5, "disagreements": []}
                             for n in ("frequency", "quality", "mean_quality")},
               "calls": 0, "code_sha": second_impl._sha(model)}
        second_impl.save(engine.quest_root, pre)
        for name in ("frequency", "quality", "mean_quality"):
            second_impl_ref = engine.quest_root / second_impl.DIR / f"{name}.py"
            second_impl_ref.parent.mkdir(parents=True, exist_ok=True)
            second_impl_ref.write_text(REFERENCE[name], encoding="utf-8")
        changed = await engine._second_version_gate(STATE, None, None)
        entry = second_impl.load(engine.quest_root)["functions"]["count_cycles"]
        text = model.read_text(encoding="utf-8")
        if kept:
            assert changed is True and entry["status"] == "agree" and "int(n) + 1" not in text
            assert "showed the first was wrong: the first was repaired" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")
        else:
            assert changed is False and entry["status"] == "unresolved" and text == wrong, "the repair is not kept"
            assert "return 3" not in text


@pytest.mark.asyncio
async def test_a_quest_with_no_cards_or_equations_skips_the_gate_without_a_word(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    bare = {"grid": {"c": [0.5]}}
    _plan_md(engine, bare)
    client = _Client()
    engine._client = client
    state = {"design": {"protocol": bare}, "implement_outline": {"scaffold": "x", "functions": []}}
    assert await engine._second_version_gate(state, None, None) is False
    assert client.calls == [] and not (engine.quest_root / second_impl.RECORD).exists()
    assert "second" not in (engine.fi_dir / "run.log").read_text(encoding="utf-8") if (engine.fi_dir / "run.log").is_file() else True
    # a package that was not written function by function has no cards to write a second version from
    other, client2 = _setup(tmp_path / "b")
    assert await other._second_version_gate(STATE, None, None) is False and client2.calls == []


@pytest.mark.asyncio
async def test_a_request_that_fails_never_stops_the_quest(tmp_path: Path) -> None:
    engine, client = await _written(tmp_path)

    async def boom(messages, **kw):  # noqa: ANN001
        raise RuntimeError("the model is down")

    client.chat = boom  # type: ignore[method-assign]
    assert await engine._second_version_gate(STATE, None, None) is False
    record = second_impl.load(engine.quest_root)
    assert all(e["status"] == "unconfirmed" for e in record["functions"].values())
    assert len(code_review.open_findings(engine.quest_root, engine.quest_root / "code")["second_version"]) == 4


# --- 5. the providers -----------------------------------------------------------------------------------------------------


def test_each_node_resolves_its_own_provider_and_the_dot_name_falls_back(tmp_path: Path) -> None:
    def engine_with(steps: dict[str, dict[str, str]]) -> Engine:
        cfg = Config(topic="a mass on a damped spring", title="spring",
                     provider=ProviderConfig(name="openai", node_providers=steps),  # type: ignore[arg-type]
                     execution=ExecutionConfig(sandbox="venv", timeout_s=120), knowledge=KnowledgeConfig(enabled=False),
                     output=OutputConfig(output_dir=tmp_path / "outputs"))
        return Engine(cfg)

    both = {"second_impl": {"name": "gemini_cli", "model": "gemini-3.8-flash-high"},
            "second_impl.adjudicate": {"name": "claude_cli", "model": "claude-haiku"}}
    engine = engine_with(both)
    nodes = engine.config.provider.node_providers
    assert node_provider_for(nodes, "second_impl")[0] == "second_impl"
    assert node_provider_for(nodes, "second_impl.adjudicate")[0] == "second_impl.adjudicate"
    assert engine._family_of_node("second_impl") == "google"
    assert engine._family_of_node("second_impl.adjudicate") == "anthropic"
    assert engine._family_of_node("implement") == "openai", "the main provider answers a node with no entry"
    only = engine_with({"second_impl": both["second_impl"]})
    assert node_provider_for(only.config.provider.node_providers, "second_impl.adjudicate")[0] == "second_impl"
    assert only._family_of_node("second_impl.adjudicate") == "google"
