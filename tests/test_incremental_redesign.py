"""A revised plan re-writes only the functions whose plan part changed; the repairs of the rest are kept
(core/function_steps.py::function_hash, core/code_delta.py)."""
from __future__ import annotations

import asyncio
import copy
import logging
import logging
from pathlib import Path

import pytest

from core import code_delta as cd
from core import function_steps as fs
from tests.test_function_steps import GOOD, OUTLINE, PROMPTS, PROTOCOL, _Model, _run, _specs


def _filler(root: Path, model: _Model, protocol: dict) -> fs.FunctionFiller:
    async def run(argv: list[str]) -> tuple[int, str, str]:
        return await _run(argv, root)

    return fs.FunctionFiller(quest_root=root, package="spring", protocol=protocol, summary="a mass on a damped spring",
                             prompts=PROMPTS, chat=model.chat, run=run, log=logging.getLogger("t"))


def _model() -> _Model:
    return _Model({k: [v] for k, v in GOOD.items()})


def test_a_changed_equation_rewrites_only_its_function_and_a_repair_of_another_is_kept(tmp_path: Path) -> None:
    first = _model()
    outcome = asyncio.run(_filler(tmp_path, first, PROTOCOL).run_all(_specs()))
    assert outcome.status == "done" and sorted(first.count) == ["decay", "frequency", "quality"]
    record = fs.load(tmp_path)
    assert all(e["built_from"] for e in record["functions"].values()), "each function records what it was built from"
    path = tmp_path / "code" / "spring" / "model.py"
    # a repair that was accepted after the function was written (a finding repair edits the file in place)
    repaired = path.read_text(encoding="utf-8").replace("return m * w / c", "return (m * w) / c  # repaired")
    path.write_text(repaired, encoding="utf-8")
    revised = copy.deepcopy(PROTOCOL)
    revised["model"]["equations"][2]["formula"] = "x(t) = x0 * exp(-g * t) + b"
    specs, _ = fs.model_functions(OUTLINE, revised)
    second = _Model({"decay": [GOOD["decay"]]})
    outcome = asyncio.run(_filler(tmp_path, second, revised).run_all(specs))
    assert outcome.status == "done"
    assert second.count == {"decay": 1}, "only the function whose equation changed is asked for"
    assert "(m * w) / c  # repaired" in path.read_text(encoding="utf-8"), "an accepted repair is kept"


def test_a_record_without_what_the_functions_were_built_from_writes_every_function_again(tmp_path: Path) -> None:
    asyncio.run(_filler(tmp_path, _model(), PROTOCOL).run_all(_specs()))
    record = fs.load(tmp_path)
    for e in record["functions"].values():
        e.pop("built_from")
    fs.save(tmp_path, record)
    revised = copy.deepcopy(PROTOCOL)
    revised["model"]["equations"][0]["formula"] = "w = sqrt(k / m) * 1"
    specs, _ = fs.model_functions(OUTLINE, revised)
    again = _model()
    asyncio.run(_filler(tmp_path, again, revised).run_all(specs))
    assert sorted(again.count) == ["decay", "frequency", "quality"]


def test_the_plan_parts_the_scripts_read_are_compared_from_the_record(tmp_path: Path) -> None:
    design = {"protocol": {"optimisation": {"objective": "a"}, "metrics": [1], "definitions": [1], "model": {"x": 1}}, "note": "n"}
    assert cd.compare(tmp_path, cd.parts(design)).known is False
    cd.save(tmp_path, cd.parts(design, {"f": "def f(x):"}), package="p")
    same = {**design, "protocol": {**design["protocol"], "definitions": [2], "model": {"x": 2}}}
    assert cd.compare(tmp_path, cd.parts(same, {"f": "def f(x):"})).changed == [], "definitions and the model are not read by the scripts"
    moved = {**design, "protocol": {**design["protocol"], "optimisation": {"objective": "b"}}}
    delta = cd.compare(tmp_path, cd.parts(moved, {"f": "def f(x):"}))
    assert delta.known and delta.changed == ["protocol.optimisation"]
    assert '"objective": "b"' in cd.describe(delta) and '"objective": "a"' in cd.describe(delta)


def test_the_shared_splice_keeps_every_other_byte_and_says_why_not() -> None:
    old = "import os\n\nX = 1\n\n\ndef a(p):\n    return 1\n\n\ndef b():\n    return 2\n"
    new, names, why = cd.splice_functions(old, "def b():\n    return 3\n")
    assert new == old.replace("return 2", "return 3") and names == ["b"] and not why
    new, _n, why = cd.splice_functions(old, "def a(q):\n    return 1\n")
    assert new is None and "a" in why and "parameters" in why
    assert cd.splice_functions(old, "no code here")[0] is None


@pytest.mark.asyncio
async def test_the_scripts_of_a_revised_plan_change_only_the_functions_the_revision_touched(tmp_path: Path, monkeypatch) -> None:
    from tests.test_code_layout import PKG, _engine_with_tool

    engine, code, state = _engine_with_tool(tmp_path)
    outline = {"scaffold": "x = 1", "functions": []}
    fs.save(engine.quest_root, {"package": PKG, "status": "done", "functions": {}})
    asked: list[str] = []

    async def chat(prompt, **kw):  # noqa: ANN001
        asked.append(prompt)
        return reply[0]

    monkeypatch.setattr(engine, "_chat", chat)
    reply = [""]
    # nothing recorded: the scripts are written whole as before
    assert await engine._incremental_scripts(state, outline) is None and not asked
    cd.save(engine.quest_root, engine._scripts_built_from(state, outline), package=PKG)
    sim = (code / "simulate.py").read_text(encoding="utf-8")
    # the revision changes nothing the scripts read: kept, no request
    kept = await engine._incremental_scripts({**state, "design": {"protocol": {**state["design"]["protocol"], "definitions": ["words only"]}}}, outline)
    assert kept is not None and kept["simulate"] == sim and not asked
    # the revision changes a part the scripts read: only the returned function is replaced
    import ast
    name = next(n.name for n in ast.parse(sim).body if isinstance(n, ast.FunctionDef))
    node = next(n for n in ast.parse(sim).body if getattr(n, "name", "") == name)
    old_fn = "\n".join(sim.splitlines()[node.lineno - 1:node.end_lineno])
    reply[0] = f"```python\n# file: simulate.py\n{old_fn}\n    _revised = 1\n```\n"
    revised = {**state, "design": {"protocol": {**state["design"]["protocol"], "optimisation": {"objective": "new"}}}}
    got = await engine._incremental_scripts(revised, outline)
    assert got is not None and len(asked) == 1 and '"objective": "new"' in asked[0]
    assert got["simulate"] == sim.replace(old_fn, old_fn + "\n    _revised = 1")
    assert got["analysis"] == (code / "experiment.py").read_text(encoding="utf-8")
    # a reply that needs a module-level change falls back to writing the scripts whole
    reply[0] = "FULL REWRITE NEEDED: a constant changes"
    assert await engine._incremental_scripts(revised, outline) is None


def test_a_function_is_rewritten_when_a_function_it_calls_changed_under_the_same_signature() -> None:
    specs, _ = fs.model_functions(OUTLINE, PROTOCOL)
    revised = copy.deepcopy(PROTOCOL)
    revised["model"]["equations"][0]["formula"] = "w = 1000 * sqrt(k / m)"  # E1, which `quality` calls
    after, _ = fs.model_functions(OUTLINE, revised)
    by = lambda ss, n: next(s for s in ss if s.name == n)  # noqa: E731
    assert "frequency" in by(specs, "quality").depends_on
    assert fs.function_hash(by(specs, "quality"), specs, PROTOCOL) != fs.function_hash(by(after, "quality"), after, revised)
    assert fs.function_hash(by(specs, "decay"), specs, PROTOCOL) == fs.function_hash(by(after, "decay"), after, revised)


def test_a_check_a_package_and_a_function_hash_are_parts_the_scripts_are_built_from() -> None:
    design = {"protocol": {"oracles": [{"name": "a"}]}}
    changed = {"protocol": {"oracles": [{"name": "a"}, {"name": "needs_a_new_output"}]}}
    assert cd.parts(design) != cd.parts(changed), "a new check can need new code"
    assert cd.parts(design, {"f": "h1"}, "pkg_a") != cd.parts(design, {"f": "h1"}, "pkg_b"), "another package is a change"
    assert cd.parts(design, {"f": "h1"}, "p") != cd.parts(design, {"f": "h2"}, "p"), "a function built from another equation"


@pytest.mark.asyncio
async def test_the_scripts_record_the_functions_hashes_not_their_signatures(tmp_path: Path, monkeypatch) -> None:
    from tests.test_code_layout import _engine_with_tool

    engine, _code, state = _engine_with_tool(tmp_path)
    monkeypatch.setattr(engine, "_protocol_block", lambda s: PROTOCOL)
    first = engine._scripts_built_from(state, OUTLINE)
    revised = copy.deepcopy(PROTOCOL)
    revised["model"]["equations"][2]["formula"] = "x(t) = x0 * exp(-g * t) + b"
    monkeypatch.setattr(engine, "_protocol_block", lambda s: revised)
    second = engine._scripts_built_from(state, OUTLINE)
    assert first["functions"] != second["functions"] and first["package"] == second["package"] != ""


def test_the_hashes_of_a_deep_chain_and_a_wide_diamond_cost_one_pass() -> None:
    import time

    def spec(name: str, deps: list[str]) -> fs.Spec:
        return fs.Spec(name=name, signature=f"def {name}(x):", purpose=name, equation="E1", depends_on=deps)

    diamond = [spec("f0", [])]
    for level in range(1, 31):  # each level calls both functions of the level below: 2**30 paths
        diamond += [spec(f"a{level}", [diamond[-1].name]), spec(f"b{level}", [diamond[-1].name])]
        diamond.append(spec(f"f{level}", [f"a{level}", f"b{level}"]))
    chain = [spec("c0", [])] + [spec(f"c{i}", [f"c{i - 1}"]) for i in range(1, 1500)]
    cycle = [spec("p", ["q"]), spec("q", ["p"])]
    started = time.time()
    for specs in (diamond, chain, cycle):
        got = fs.function_hashes(specs, PROTOCOL)
        assert len(got) == len(specs)
    assert time.time() - started < 5


def test_a_record_of_another_version_writes_every_function_again_and_says_so(tmp_path: Path, caplog) -> None:
    asyncio.run(_filler(tmp_path, _model(), PROTOCOL).run_all(_specs()))
    record = fs.load(tmp_path)
    assert record["built_from_schema"] == fs.BUILT_FROM_SCHEMA
    record.pop("built_from_schema")  # a record from before the version was written
    fs.save(tmp_path, record)
    revised = copy.deepcopy(PROTOCOL)
    revised["model"]["equations"][2]["formula"] = "x(t) = x0 * exp(-g * t) + b"
    specs, _ = fs.model_functions(OUTLINE, revised)
    again = _model()
    with caplog.at_level(logging.INFO):
        asyncio.run(_filler(tmp_path, again, revised).run_all(specs))
    assert sorted(again.count) == ["decay", "frequency", "quality"]
    assert "another version of FI" in caplog.text
    cd.save(tmp_path, {"p": "1"})
    path = tmp_path / cd.RECORD
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("schema")
    path.write_text(json.dumps(data), encoding="utf-8")
    delta = cd.compare(tmp_path, {"p": "1"})
    assert not delta.known and "another version" in delta.reason
