"""Explore, then confirm (``engine.phased``, core/phased.py): the record, the data held back, the seeds, the evidence
ladder and the paper's note, then real fake-LLM quests with the setting off and on.

No real model is called: ``core.engine.LLMClient.chat`` is patched."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from core import evidence as _evidence
from core import phased


def _csv(path: Path, rows: int) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ("x,y\n" + "".join(f"{i},{i * i}\n" for i in range(rows))).encode()
    path.write_bytes(data)
    return data


def _data_rows(raw: bytes) -> list[str]:
    return [line for line in raw.decode().splitlines()[1:] if line.strip()]


# ---- the data held back -------------------------------------------------------------------------------------------


def test_no_data_means_fresh_seeds_and_says_why(tmp_path: Path) -> None:
    record, lines = phased.prepare(tmp_path, "q1")
    assert record["stage"] == phased.EXPLORE and record["strategy"] == phased.FRESH_SEEDS
    assert record["why_no_data"] == "no data was supplied in inputs/data/"
    assert any("exploration stage" in line for line in lines)
    assert any("new random seeds because no held-back data is available" in line for line in lines)
    assert phased.load(tmp_path)["stage"] == phased.EXPLORE


def test_enough_rows_are_split_and_exploration_sees_only_its_part(tmp_path: Path) -> None:
    original = _csv(tmp_path / "inputs" / "data" / "d.csv", 100)
    record, lines = phased.prepare(tmp_path, "q1")
    assert record["strategy"] == phased.HELD_BACK
    (info,) = record["files"]
    assert (info["rows"], info["explore_rows"], info["held_back_rows"]) == (100, 70, 30)
    explore = (tmp_path / "inputs" / "data" / "d.csv").read_bytes()
    held = (tmp_path / ".fi" / "phased" / "held_back" / "d.csv").read_bytes()
    assert explore.startswith(b"x,y\n") and held.startswith(b"x,y\n")
    assert len(_data_rows(explore)) == 70 and len(_data_rows(held)) == 30
    assert not set(_data_rows(explore)) & set(_data_rows(held)), "no row is in both parts"
    assert sorted(_data_rows(explore) + _data_rows(held)) == sorted(_data_rows(original))
    assert (tmp_path / ".fi" / "phased" / "original" / "d.csv").read_bytes() == original
    assert any("held back 30 of 100 rows of inputs/data/d.csv" in line for line in lines)
    # Idempotent: the part exploration sees is not split again.
    again, more = phased.prepare(tmp_path, "q1")
    assert more == [] and again["files"] == record["files"]
    assert (tmp_path / "inputs" / "data" / "d.csv").read_bytes() == explore


def test_too_few_rows_or_a_file_that_cannot_be_split_falls_back_to_new_seeds(tmp_path: Path) -> None:
    small = _csv(tmp_path / "inputs" / "data" / "small.csv", 12)
    record, _ = phased.prepare(tmp_path, "q1")
    assert record["strategy"] == phased.FRESH_SEEDS and "has 12 rows" in record["why_no_data"]
    assert (tmp_path / "inputs" / "data" / "small.csv").read_bytes() == small, "left whole"

    other = tmp_path / "other"
    big = _csv(other / "inputs" / "data" / "big.csv", 100)
    (other / "inputs" / "data" / "meta.json").write_text("{}", encoding="utf-8")
    record, _ = phased.prepare(other, "q2")
    assert record["strategy"] == phased.FRESH_SEEDS and "meta.json cannot be split" in record["why_no_data"]
    assert (other / "inputs" / "data" / "big.csv").read_bytes() == big, "exploration would see the JSON whole: nothing held back"


def test_a_file_that_arrives_later_and_cannot_be_split_puts_the_whole_files_back(tmp_path: Path) -> None:
    original = _csv(tmp_path / "inputs" / "data" / "d.csv", 100)
    phased.prepare(tmp_path, "q1")
    (tmp_path / "inputs" / "data" / "notes.parquet").write_bytes(b"PAR1")
    record, lines = phased.prepare(tmp_path, "q1")
    assert record["strategy"] == phased.FRESH_SEEDS and record["files"] == []
    assert (tmp_path / "inputs" / "data" / "d.csv").read_bytes() == original
    assert any("no longer held back" in line for line in lines)


# ---- seeds and the boundary ---------------------------------------------------------------------------------------


def _env(seed: int, index: int) -> dict[str, str]:
    return {"PATH": "/bin", "FI_REPLICATE_SEED": str(seed), "FI_REPLICATE_INDEX": str(index)}


def test_the_confirm_run_gets_a_seed_base_exploration_never_used(tmp_path: Path) -> None:
    stride = 1_000_000
    phased.prepare(tmp_path, "q1")
    for i in range(3):
        env = _env(i * stride, i)
        assert phased.seed_env(tmp_path, env) is env, "exploration: the seed is kept as it is"
    assert phased.load(tmp_path)["explore_seed_bases"] == [0, stride, 2 * stride]
    record, lines = phased.enter_confirm(tmp_path, explore_result={"score": 1.0}, frozen_sha256="abc", stride=stride,
                                         replicates=3, explore_runs=2)
    base = record["confirm_seed_base"]
    assert record["stage"] == phased.CONFIRM and base % stride == 0 and base + 3 * stride < 2**31
    assert base >= 3 * stride, "apart from every exploration range"
    assert any("new random seeds" in line and "no held-back data was available" in line for line in lines)
    moved = [int(phased.seed_env(tmp_path, _env(i * stride, i))["FI_REPLICATE_SEED"]) for i in range(3)]
    assert moved == [base, base + stride, base + 2 * stride]
    assert not set(moved) & {0, stride, 2 * stride}
    # Idempotent once in the confirm stage.
    again, more = phased.enter_confirm(tmp_path, explore_result=None, frozen_sha256="x", stride=stride, replicates=3,
                                       explore_runs=9)
    assert more == [] and again["confirm_seed_base"] == base


def test_the_confirm_run_sees_only_the_held_back_part_and_the_whole_file_comes_back(tmp_path: Path) -> None:
    original = _csv(tmp_path / "inputs" / "data" / "d.csv", 50)
    phased.prepare(tmp_path, "q1")
    held = (tmp_path / ".fi" / "phased" / "held_back" / "d.csv").read_bytes()
    record, lines = phased.enter_confirm(tmp_path, explore_result={"a": 1}, frozen_sha256=None, stride=10,
                                         replicates=1, explore_runs=1)
    assert (tmp_path / "inputs" / "data" / "d.csv").read_bytes() == held
    assert any("only the data held back" in line for line in lines)
    record, lines = phased.record_confirm(tmp_path, {"a": 2})
    assert record["stage"] == phased.CONFIRMED and phased.status(record) == phased.CONFIRMED
    assert (tmp_path / "inputs" / "data" / "d.csv").read_bytes() == original
    # The same result again is not a second look; another one is.
    assert phased.record_confirm(tmp_path, {"a": 2})[1] == []
    record, lines = phased.record_confirm(tmp_path, {"a": 3})
    assert phased.status(record) == "confirm_reused" and any("no longer from one untouched" in x for x in lines)


def test_a_confirm_run_without_a_result_is_not_confirmed(tmp_path: Path) -> None:
    phased.prepare(tmp_path, "q1")
    phased.enter_confirm(tmp_path, explore_result={"a": 1}, frozen_sha256=None, stride=1, replicates=1, explore_runs=1)
    record, lines = phased.record_confirm(tmp_path, None)
    assert phased.status(record) == "confirm_failed" and any("produced no result" in x for x in lines)


# ---- the evidence ladder and the paper ----------------------------------------------------------------------------


def _ready_gaps(tmp_path: Path, settings: dict) -> list[str]:
    record = _evidence.assess(tmp_path, {"result_json": {"a": 1}}, settings=settings)
    return next(level["gaps"] for level in record["ladder"] if level["level"] == "publication_ready")


def test_the_ladder_reads_only_a_confirmed_result_as_more_than_preliminary(tmp_path: Path) -> None:
    base = _ready_gaps(tmp_path, {})
    assert not any("exploration" in g or "confirm" in g for g in base)
    for state, words in (("explore", "exploration stage"), ("confirming", "has not finished"),
                         ("confirm_failed", "produced no result"), ("confirm_reused", "run again after")):
        gaps = _ready_gaps(tmp_path, {"phased": state, "phased_strategy": "fresh_seeds"})
        assert any(words in g for g in gaps), (state, gaps)
    assert _ready_gaps(tmp_path, {"phased": "confirmed", "phased_strategy": "fresh_seeds"}) == base


def test_the_paper_note_says_which_numbers_are_confirmed_and_holds_no_numbers(tmp_path: Path) -> None:
    _csv(tmp_path / "inputs" / "data" / "small.csv", 12)
    phased.prepare(tmp_path, "q1")
    paper = "# Title\n\nBody.\n"
    explore_note = phased.mark_paper(paper, phased.load(tmp_path))
    assert "exploratory" in explore_note.lower() and explore_note.startswith("# Title\n")
    phased.enter_confirm(tmp_path, explore_result={"a": 1}, frozen_sha256=None, stride=1, replicates=1, explore_runs=1)
    phased.record_confirm(tmp_path, {"a": 2})
    note = phased.mark_paper(explore_note, phased.load(tmp_path))
    assert note.count("<!-- fi:phased -->") == 1, "replaced, not added"
    assert "confirmed on new random seeds because no held-back data was available" in note
    block = note.split("<!-- fi:phased -->")[1].split("<!-- /fi:phased -->")[0]
    assert not any(ch.isdigit() for ch in block), "the number audits hold every number in the paper to the results"
    assert "numbers reported as results are that confirm run's" in block


def test_phased_reads_no_attempt_record_or_shadow_recommendation() -> None:
    tree = ast.parse(Path(phased.__file__).read_text(encoding="utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not {m for m in imported if "attempt" in m}, imported


def test_off_by_default() -> None:
    from core.config import EngineConfig

    assert EngineConfig().phased is False


# ---- real quests (fake model, real venv) --------------------------------------------------------------------------

#: A script that says what it was given: the seed, and how many rows of the supplied data it could read.
_SEEING_SCRIPT = """\
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

seed = int(os.environ.get('FI_REPLICATE_SEED', '0'))
path = os.path.join('inputs', 'data', 'd.csv')
rows = 0
if os.path.exists(path):
    with open(path) as f:
        rows = sum(1 for line in f if line.strip()) - 1
if not os.environ.get('FI_PILOT'):
    with open('seen.jsonl', 'a') as f:
        f.write(json.dumps({'seed': seed, 'index': os.environ.get('FI_REPLICATE_INDEX'), 'rows': rows}) + '\\n')
os.makedirs('figures', exist_ok=True)
plt.figure(); plt.plot([0, 1, 2], [0, 1, 4]); plt.savefig('figures/result.png', dpi=72)
print('RESULT_JSON: ' + json.dumps({'score': 0.5, 'rows': rows}))
"""


def _config(root: Path, *, phased_on: bool | None):
    from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig

    engine_kwargs = dict(max_iterations=1, review_loop=False, auto_accept_on_pass=True, execute_replicates=2)
    if phased_on is not None:
        engine_kwargs["phased"] = phased_on
    return Config(topic="smoke-test topic for the engine", title="phased-smoke", provider=ProviderConfig(name="openai"),
                  engine=EngineConfig(**engine_kwargs), execution=ExecutionConfig(sandbox="venv", timeout_s=120),
                  knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=root))


def _fake(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_engine_smoke import _classify, _fake_response_for

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if _classify(prompt) == "Implementation":
            return json.dumps({"code": _SEEING_SCRIPT, "deps": ["matplotlib"]})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)


def _seen(engine) -> list[dict]:  # noqa: ANN001
    path = engine.quest_root / "seen.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []


def _routes(engine) -> list[tuple]:  # noqa: ANN001
    from core import audit_log

    return [(e.get("kind"), e.get("node"), e.get("chosen")) for e in audit_log.read(engine.fi_dir / "audit.jsonl")
            if e.get("kind") in ("node_completed", "route_decision")]


@pytest.mark.slow
@pytest.mark.asyncio
async def test_off_the_quest_never_touches_the_phased_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Off (the default and written out): every function of core/phased.py but ``enabled`` raises if called, the graph
    has no confirm edge, the data and the seeds are as they were, and both quests take the same steps and end in the
    same state."""
    from core.engine import Engine
    from tests.test_attempt_memory import _normalized_state

    _fake(monkeypatch)
    for name in ("prepare", "seed_env", "enter_confirm", "record_confirm", "evidence_settings", "write_note",
                 "mark_paper", "load"):
        def boom(*a, _name=name, **k):  # noqa: ANN001, ANN002, ANN003
            raise AssertionError(f"core.phased.{_name} called with engine.phased off")
        monkeypatch.setattr(phased, name, boom)
    runs = {}
    for label, flag in (("default", None), ("off", False)):
        root = tmp_path / label
        engine = Engine(_config(root, phased_on=flag))
        data = _csv(engine.quest_root / "inputs" / "data" / "d.csv", 100)
        artifacts = await engine.run()
        assert (engine.quest_root / "inputs" / "data" / "d.csv").read_bytes() == data
        assert not (engine.fi_dir / phased.RECORD).exists() and not (engine.fi_dir / phased.DIR).exists()
        assert [s["seed"] for s in _seen(engine)] == [0, 1_000_000] and {s["rows"] for s in _seen(engine)} == {100}
        paper = artifacts.paper_md.read_text(encoding="utf-8")
        assert "fi:phased" not in paper and "Exploratory and confirmed" not in paper
        assert "phased" not in (engine.quest_root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8")
        assert "[phased]" not in (engine.fi_dir / "run.log").read_text(encoding="utf-8")
        assert "confirm" not in _evidence_gate_ends(engine)
        runs[label] = (engine, artifacts)
    (a, a_art), (b, b_art) = runs["default"], runs["off"]
    assert _routes(a) == _routes(b)
    sa, sb = _normalized_state(a_art, a.quest_root), _normalized_state(b_art, b.quest_root)
    assert sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k)) == []


def _evidence_gate_ends(engine) -> dict:  # noqa: ANN001
    branches = engine._build_graph().branches["evidence_gate"]
    (spec,) = branches.values()
    return dict(spec.ends or {})


def test_the_confirm_edge_exists_only_when_on(tmp_path: Path) -> None:
    from core.engine import Engine

    off = Engine(_config(tmp_path / "off", phased_on=None))
    on = Engine(_config(tmp_path / "on", phased_on=True))
    assert "confirm" not in _evidence_gate_ends(off)
    assert _evidence_gate_ends(on).get("confirm") == "execute"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_on_without_data_the_frozen_design_is_confirmed_on_new_seeds(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine
    from core import frozen_protocol as _frozen

    _fake(monkeypatch)
    engine = Engine(_config(tmp_path / "out", phased_on=True))
    artifacts = await engine.run()
    record = phased.load(engine.quest_root)
    assert record["stage"] == phased.CONFIRMED and record["strategy"] == phased.FRESH_SEEDS
    seen = _seen(engine)
    explore = [s["seed"] for s in seen[:2]]
    confirm = [s["seed"] for s in seen[2:]]
    assert explore == [0, 1_000_000], seen
    base = record["confirm_seed_base"]
    assert confirm == [base, base + 1_000_000] and not set(confirm) & set(explore)
    routes = _routes(engine)
    assert ("route_decision", "evidence_gate", "confirm") in routes
    assert [r for r in routes if r[:2] == ("node_completed", "execute")].__len__() == 2
    frozen = _frozen.load(engine.quest_root)
    assert frozen is not None and "exploration" in frozen["source"]
    assert record["frozen_sha256"] == frozen["sha256"]
    log = (engine.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "[phased] exploration stage" in log and "[phased] confirm stage" in log
    assert "[protocol] frozen at the end of exploration" in log
    paper = artifacts.paper_md.read_text(encoding="utf-8")
    assert "confirmed on new random seeds because no held-back data was available" in paper
    ev = json.loads((engine.quest_root / "needs" / "EVIDENCE.json").read_text(encoding="utf-8"))
    assert not any("exploration stage" in g for g in ev["ladder"][-1]["gaps"]), "a confirmed result is not preliminary"
    assert ev.get("phased", {}).get("status") == "confirmed"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_on_with_enough_data_the_confirm_run_sees_only_the_held_back_rows(tmp_path: Path,
                                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from core.engine import Engine

    _fake(monkeypatch)
    engine = Engine(_config(tmp_path / "out", phased_on=True))
    original = _csv(engine.quest_root / "inputs" / "data" / "d.csv", 100)
    artifacts = await engine.run()
    record = phased.load(engine.quest_root)
    assert record["strategy"] == phased.HELD_BACK and record["stage"] == phased.CONFIRMED
    assert [s["rows"] for s in _seen(engine)] == [70, 70, 30, 30]
    assert artifacts.raw_state["result_json"]["rows"] == 30, "the paper is written from the confirm run"
    assert (engine.quest_root / "inputs" / "data" / "d.csv").read_bytes() == original, "the whole file is back"
    paper = artifacts.paper_md.read_text(encoding="utf-8")
    assert "held back at random before exploration began" in paper


def test_in_the_confirm_stage_a_redesign_is_not_followed(tmp_path: Path) -> None:
    from core.engine import Engine

    engine = Engine(_config(tmp_path / "out", phased_on=True))
    state = {"cross_check": [{"finding": "x"}], "analysis": {"next_step": "re_experiment"}, "iteration": 0}
    assert engine._route_after_cross_check(state) == "redesign", "exploration may redesign"
    phased.prepare(engine.quest_root, engine.quest_id)
    phased.enter_confirm(engine.quest_root, explore_result={"a": 1}, frozen_sha256=None, stride=1, replicates=1,
                         explore_runs=1)
    assert engine._route_after_cross_check(state) == "write"
    gate = {"evidence_assessment": {"route": "redesign"}, "result_json": {"a": 2}, "exec_result": {"returncode": 0}}
    assert engine._route_after_evidence_gate(gate) == "write"
    assert phased.status(phased.load(engine.quest_root)) == phased.CONFIRMED


def test_changing_the_protocol_after_the_confirm_stage_began_is_an_amendment(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    from core import frozen_protocol as _frozen
    from core.engine import Engine

    engine = Engine(_config(tmp_path / "out", phased_on=True))
    phased.prepare(engine.quest_root, engine.quest_id)
    protocol = {"grid": {"n": [1, 2]}, "runs_per_setting": 5}
    engine._draft_protocol = lambda state: protocol  # type: ignore[method-assign]
    engine._freeze_protocol_if_due({})
    assert _frozen.load(engine.quest_root) is None, "not frozen during exploration"
    engine._freeze_protocol_if_due({}, at_confirm=True)
    assert _frozen.load(engine.quest_root)["protocol"] == protocol
    asked: list = []
    monkeypatch.setattr(engine, "_pause_for_amendment", lambda pending: asked.append(pending))
    changed = {"grid": {"n": [1, 2, 3]}, "runs_per_setting": 5}
    engine._hold_design_to_frozen({"iteration": 1}, {"protocol": changed,
                                                     "protocol_amendment": {"protocol": changed, "reason": "more"}})
    assert asked and asked[0]["changes"], "a change after the freeze goes through an amendment"
