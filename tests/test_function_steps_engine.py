"""The model's package written one function at a time, and a failing equation test repaired alone, in the engine
(core/function_steps.py). Fake models, neutral models (a mass on a damped spring)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import equation_tests as eqt, function_steps as fs
from core.config import Config, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from tests.test_equation_gate import (
    GOOD, OUTLINE, STATE, WRONG_Q, _engine, _plan_md, _Writer, _write_code,
)


@pytest.mark.asyncio
async def test_the_package_is_written_one_function_at_a_time_and_the_code_request_asks_for_the_scripts_only(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _plan_md(engine)
    writer = _Writer({"frequency": [GOOD["frequency"]], "quality": [WRONG_Q, GOOD["quality"]]})
    engine._client = writer
    assert await engine._fill_model_functions(STATE, OUTLINE) is True
    # Three requests in all for two functions (one repair): the cost is one request per function, plus the repairs.
    assert len(writer.prompts) == 3 and writer.seen == {"frequency": 1, "quality": 2}
    model = (engine.quest_root / "code" / "spring" / "model.py").read_text(encoding="utf-8")
    assert "return m * w / c" in model and "return w / c" not in model
    block = engine._split_block(STATE)
    assert "ALREADY WRITTEN AND TESTED" in block and "def frequency(k, m)" in block
    assert "# file: spring/model.py" not in block, "the code request does not ask for the package again"
    assert "9.81" in writer.prompts[0] and "G = 9.81" not in model
    assert "repairing ONE function" in writer.prompts[2] and "frequency" not in writer.prompts[2].split("# The equation")[0]
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "function quality passes its checks" in log and "the model's package is written: 2 function(s)" in log
    # The same plan again: nothing is written again (a resume or a later pass).
    again = _Writer({})
    engine._client = again
    assert await engine._fill_model_functions(STATE, OUTLINE) is True and again.prompts == []


@pytest.mark.asyncio
async def test_a_function_that_stays_wrong_falls_back_to_writing_the_code_whole(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _plan_md(engine)
    writer = _Writer({"frequency": [GOOD["frequency"]], "quality": [WRONG_Q]})
    engine._client = writer
    assert await engine._fill_model_functions(STATE, OUTLINE) is False
    assert writer.seen["quality"] == 3, "one request and the two repairs of a function, never more"
    assert not (engine.quest_root / "code" / "spring").exists()
    block = engine._split_block(STATE)
    assert "# file: spring/model.py" in block, "the code request asks for the package as it always did"
    assert "`quality` (equation E2) still failed its checks" in block
    # The fallback is the old path: a reply that holds the package is used as before.
    assert not fs.is_ready(engine.quest_root, "spring")


@pytest.mark.asyncio
async def test_a_quest_whose_outline_names_no_function_per_equation_or_with_one_script_is_unchanged(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _plan_md(engine)
    writer = _Writer({})
    engine._client = writer
    bare = {"scaffold": "x = 1", "functions": [OUTLINE["functions"][2]]}
    assert await engine._fill_model_functions(STATE, bare) is False and writer.prompts == []
    assert "# file: spring/model.py" in engine._split_block(STATE)
    # Not a package: nothing is asked, and nothing is written.
    cfg = engine.config.model_copy(deep=True)
    cfg.execution.code_package = False
    other = Engine(cfg)
    other._client = writer
    assert await other._fill_model_functions(STATE, OUTLINE) is False and writer.prompts == []
    off = engine.config.model_copy(deep=True)
    off.execution.code_function_steps = False
    third = Engine(off)
    third._client = writer
    assert await third._fill_model_functions(STATE, OUTLINE) is False and writer.prompts == []
    assert not (engine.quest_root / "code").exists() or not (engine.quest_root / "code" / "spring").exists()


@pytest.mark.asyncio
async def test_an_equation_implemented_wrongly_is_caught_before_the_run_and_repaired_alone(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _plan_md(engine)
    _write_code(engine, WRONG_Q)
    writer = _Writer({"quality": [GOOD["quality"]]})
    engine._client = writer
    assert await engine._equation_gate(STATE, None, None) is None
    assert len(writer.prompts) == 1, "the wrong function was caught and repaired with one request"
    prompt = writer.prompts[0]
    assert "`quality`" in prompt and "equation E2" in prompt and "def frequency" not in prompt.split("# The equation")[0]
    for secret in ("12.0", "3.0", "'m': 2.0", "0.5"):
        assert secret not in prompt, f"the worked example's numbers never reach the repair ({secret})"
    assert "return m * w / c" in (engine.quest_root / "code" / "spring" / "model.py").read_text(encoding="utf-8")
    record = json.loads((engine.fi_dir / "equation_tests.json").read_text(encoding="utf-8"))
    assert record["failed"] == [] and record["repairs"] == {"E2": 1}
    assert "agree with the plan's worked examples" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert (engine.quest_root / "code" / eqt.TEST_PATH).is_file()


@pytest.mark.asyncio
async def test_a_function_still_wrong_after_its_repairs_is_sent_to_the_existing_whole_script_repair(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _plan_md(engine)
    _write_code(engine, WRONG_Q)
    writer = _Writer({"quality": [WRONG_Q]})
    engine._client = writer
    patch = await engine._equation_gate(STATE, None, None)
    assert len(writer.prompts) == 2, "at most function_steps.REPAIRS repairs of one function"
    assert patch is not None and patch["exec_result"]["returncode"] == 1 and patch["result_json"] == {}
    assert patch["exec_result"]["failed_script"] == "simulate.py"
    stderr = patch["exec_result"]["stderr_tail"]
    assert "`quality`" in stderr and "equation E2" in stderr and "Q = m * w / c" in stderr
    for secret in ("12.0", "3.0", "0.5", "'m'"):
        assert secret not in stderr
    assert json.loads((engine.fi_dir / "equation_tests.json").read_text(encoding="utf-8"))["failed"][0]["id"] == "E2"




@pytest.mark.asyncio
async def test_no_request_is_spent_checking_functions_in_an_environment_that_is_not_there(tmp_path: Path) -> None:
    cfg = Config(topic="a mass on a damped spring", title="spring", provider=ProviderConfig(name="openai"),
                 execution=ExecutionConfig(sandbox="venv", timeout_s=120, shared_interpreter=False,
                                           system_site_packages=False),
                 knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "outputs"))
    engine = Engine(cfg)  # no run(): the executor's setup has not made the quest's venv
    engine.quest_root.mkdir(parents=True, exist_ok=True)
    _plan_md(engine)
    writer = _Writer({})
    engine._client = writer
    assert not engine.executor.python_path(engine.quest_root).is_file()
    assert await engine._fill_model_functions(STATE, OUTLINE) is False and writer.prompts == []
    assert "environment is not there" in (engine.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_a_failed_equation_test_leaves_no_earlier_script_s_seeds_behind(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _plan_md(engine)
    _write_code(engine, WRONG_Q)
    engine._client = _Writer({"quality": [WRONG_Q]})
    patch = await engine._equation_gate(STATE, None, None)
    assert patch["result_json_replicates"] == [] and patch["result_json_deterministic"] is False
    assert patch["result_json_trials"] is False


# --- the plan's own numbers never reach a request that writes or repairs code -----------------------------------------

_SECRET_PROTOCOL = {
    "grid": {"c": [0.5, 1.0]},
    "model": {"summary": "a mass on a damped spring", "equations": [
        {"id": "E1", "formula": "w = sqrt(k / m)", "role": "generates", "source": "derivation", "derivation": "Newton",
         "example": {"inputs": {"k": 7.3137, "m": 4.2719}, "expected_formula": "sqrt(k / m) * 1.0000001"}},
        {"id": "E2", "formula": "Q = m * w / c", "role": "generates", "source": "derivation",
         "example": {"inputs": {"m": 2.0, "w": 3.0, "c": 0.5}, "expected_formula": "m * w / c * 1.0000002"}},
    ]},
    "oracles": [{"name": "w_check", "kind": "special_case", "check": "w at one setting", "expected": 0.123456789,
                 "expected_formula": "0.123456789 + 0.0000000001", "tolerance": 1e-9, "case": {"c": 0.5}, "measure": "w",
                 "reference": "derivation: by hand"}],
}
_SECRETS = ("7.3137", "4.2719", "1.0000001", "1.0000002", "0.0000000001", "expected_formula")


@pytest.mark.asyncio
async def test_no_request_that_writes_or_repairs_code_holds_an_example_or_an_expected_value(tmp_path: Path) -> None:
    from core.engine import _parse_json_lenient  # noqa: F401  (the helper the nodes use)

    engine = _engine(tmp_path)
    protocol = _SECRET_PROTOCOL
    _plan_md(engine, protocol)
    state: dict = {"design": {"hypothesis": "h", "protocol": protocol}, "code": "x = 1", "deps": [],
                   "implement_outline": OUTLINE,
                   "exec_result": {"returncode": 1, "stderr_tail": "boom", "stdout_tail": "", "failed_script": "simulate.py"}}
    prompts: list[tuple[str, str]] = []

    class _Spy:
        async def chat(self, messages, **kw):  # noqa: ANN001
            prompts.append((str(kw.get("node") or ""), messages[-1]["content"]))
            return "{}"  # nothing usable: the repairs and writes all end without a change

        async def aclose(self) -> None:
            return None

    engine._client = _Spy()
    code = engine.quest_root / "code"
    _write_code(engine, WRONG_Q)
    (code / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    steps = [
        lambda: engine._node_implement_outline({**state, "implement_outline": {}}),
        lambda: engine._node_implement(state),
        lambda: engine._fill_model_functions(state, OUTLINE),
        lambda: engine._equation_gate({"design": state["design"]}, None, None),
        lambda: engine._repair_script_for_oracle(state, code / "simulate.py", protocol["oracles"], ["w_check is off"]),
        lambda: engine._repair_script_for_protocol(state, code / "simulate.py", "x = 1", [], protocol, []),
        lambda: engine._node_execute_reflect(state),
    ]
    for step in steps:
        _write_code(engine, WRONG_Q)  # an earlier step may have removed or changed the scripts
        try:
            await step()
        except Exception as e:  # noqa: BLE001 -- only what was asked of the model matters here
            print("step raised", repr(e))
    nodes = {n for n, _p in prompts}
    assert {"implement", "implement_outline", "implement_oracle", "implement_protocol", "execute_reflect"} <= nodes, nodes
    for node, prompt in prompts:
        # (A check's own `expected` stays in the oracle repair: its `oracle_change` reply argues about that number.)
        for secret in _SECRETS:
            assert secret not in prompt, f"{secret!r} reached a request of the step {node!r}"
    outline = next(p for n, p in prompts if n == "implement_outline")
    assert '"input_names"' in outline and '"k"' in outline, "the outline may know the NAMES of an example's inputs"
