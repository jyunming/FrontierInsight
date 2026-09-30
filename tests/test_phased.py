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
    assert info["rows"] == 100 and info["explore_rows"] + info["held_back_rows"] == 100
    assert 15 <= info["held_back_rows"] <= 45, "about 30% held back"
    n_held, n_explore = info["held_back_rows"], info["explore_rows"]
    explore = (tmp_path / "inputs" / "data" / "d.csv").read_bytes()
    held = (phased.store_dir(tmp_path) / "held_back" / "d.csv").read_bytes()
    assert explore.startswith(b"x,y\n") and held.startswith(b"x,y\n")
    assert len(_data_rows(explore)) == n_explore and len(_data_rows(held)) == n_held
    assert not set(_data_rows(explore)) & set(_data_rows(held)), "no row is in both parts"
    assert sorted(_data_rows(explore) + _data_rows(held)) == sorted(_data_rows(original))
    assert (phased.store_dir(tmp_path) / "original" / "d.csv").read_bytes() == original
    assert any(f"held back {n_held} of 100 rows of inputs/data/d.csv" in line for line in lines)
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
    held = (phased.store_dir(tmp_path) / "held_back" / "d.csv").read_bytes()
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
import random
print('RESULT_JSON: ' + json.dumps({'score': 0.5, 'rows': rows, 'draw': random.Random(seed).random()}))
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
                 "mark_paper", "load", "turned_off", "restore_inputs", "note_job_pending", "note_confirm_result",
                 "mark_compromised"):
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
        assert not phased.store_dir(engine.quest_root).exists()
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
    (info,) = record["files"]
    n_explore, n_held = info["explore_rows"], info["held_back_rows"]
    assert n_explore + n_held == 100 and n_held > 0
    assert [s["rows"] for s in _seen(engine)] == [n_explore, n_explore, n_held, n_held]
    assert artifacts.raw_state["result_json"]["rows"] == n_held, "the paper is written from the confirm run"
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
    phased.seed_env(engine.quest_root, _env(0, 0))  # the confirm run was made
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


# ---- a start cut short, a stop, and a quest that goes on ----------------------------------------------------------


def _held(tmp_path: Path) -> bytes:
    return (phased.store_dir(tmp_path) / "held_back" / "d.csv").read_bytes()


def _inputs(tmp_path: Path) -> bytes:
    return (tmp_path / "inputs" / "data" / "d.csv").read_bytes()


def test_a_start_cut_short_before_the_record_was_written_keeps_the_whole_file(tmp_path: Path) -> None:
    original = _csv(tmp_path / "inputs" / "data" / "d.csv", 100)
    phased.prepare(tmp_path, "q1")
    explore, held = _inputs(tmp_path), _held(tmp_path)
    phased.record_path(tmp_path).unlink()  # the files were written, the record was not
    record, _ = phased.prepare(tmp_path, "q1")
    assert (phased.store_dir(tmp_path) / "original" / "d.csv").read_bytes() == original, "the whole file is not lost"
    assert _inputs(tmp_path) == explore and _held(tmp_path) == held
    assert record["files"][0]["original_sha256"] == phased._sha(original)


def test_a_stop_while_the_confirm_stage_began_never_hands_the_held_back_rows_to_exploration(tmp_path: Path) -> None:
    original = _csv(tmp_path / "inputs" / "data" / "d.csv", 100)
    phased.prepare(tmp_path, "q1")
    explore, held = _inputs(tmp_path), _held(tmp_path)
    # Held-back rows copied in, but the stage not yet saved: the next start takes it for what it is, not as new data.
    (tmp_path / "inputs" / "data" / "d.csv").write_bytes(held)
    record, _ = phased.prepare(tmp_path, "q1")
    assert record["stage"] == phased.EXPLORE and _inputs(tmp_path) == explore and _held(tmp_path) == held
    assert (phased.store_dir(tmp_path) / "original" / "d.csv").read_bytes() == original
    # The stage saved, the held-back rows not yet copied in: the next start gives the confirm run the held-back rows.
    phased.enter_confirm(tmp_path, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1,
                         explore_runs=1)
    (tmp_path / "inputs" / "data" / "d.csv").write_bytes(explore)
    phased.prepare(tmp_path, "q1")
    assert _inputs(tmp_path) == held


def test_outside_a_run_the_whole_file_is_back_and_the_next_start_holds_the_same_rows_back(tmp_path: Path) -> None:
    original = _csv(tmp_path / "inputs" / "data" / "d.csv", 100)
    phased.prepare(tmp_path, "q1")
    explore, held = _inputs(tmp_path), _held(tmp_path)
    phased.restore_inputs(tmp_path)
    assert _inputs(tmp_path) == original
    _record, lines = phased.prepare(tmp_path, "q1")
    assert _inputs(tmp_path) == explore and _held(tmp_path) == held and lines == []
    # Rows added while the quest was paused: every row exploration saw stays on exploration's side.
    phased.restore_inputs(tmp_path)
    with (tmp_path / "inputs" / "data" / "d.csv").open("ab") as f:
        f.write("".join(f"{i},{i * i}\n" for i in range(100, 130)).encode())
    phased.prepare(tmp_path, "q1")
    assert set(_data_rows(explore)) <= set(_data_rows(_inputs(tmp_path)))
    assert set(_data_rows(held)) <= set(_data_rows(_held(tmp_path)))
    # In the confirm stage a stop also puts the whole file back, and the next start the held-back rows again.
    phased.enter_confirm(tmp_path, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1,
                         explore_runs=1)
    held_now = _held(tmp_path)
    assert _inputs(tmp_path) == held_now
    phased.restore_inputs(tmp_path)
    assert _data_rows(_inputs(tmp_path))[-1] == "129,16641"
    phased.prepare(tmp_path, "q1")
    assert _inputs(tmp_path) == held_now


def test_turned_on_after_the_experiment_ran_holds_nothing_back(tmp_path: Path) -> None:
    original = _csv(tmp_path / "inputs" / "data" / "d.csv", 100)
    record, lines = phased.prepare(tmp_path, "q1", already_ran=True)
    assert record["strategy"] == phased.FRESH_SEEDS and record["files"] == []
    assert _inputs(tmp_path) == original and any("already run" in line for line in lines)
    assert phased.prepare(tmp_path, "q1")[0]["files"] == [], "stays so on the next start"
    phased.enter_confirm(tmp_path, explore_result={"a": 1}, frozen_sha256=None, stride=1, replicates=1, explore_runs=1)
    phased.record_confirm(tmp_path, {"a": 2})
    assert "already run before explore-then-confirm was turned on" in phased.summary(phased.load(tmp_path))


def test_the_gate_run_again_before_the_confirm_run_sends_the_quest_to_it(tmp_path: Path) -> None:
    """A stop after the confirm stage began but before the gate's step was saved: the gate runs again with exploration's
    result in hand, and must send the quest to the confirm run, not record exploration's result as confirmed."""
    from core.engine import Engine

    engine = Engine(_config(tmp_path / "out", phased_on=True))
    phased.prepare(engine.quest_root, engine.quest_id)
    phased.enter_confirm(engine.quest_root, explore_result={"a": 1}, frozen_sha256=None, stride=1, replicates=1,
                         explore_runs=1)
    gate = {"evidence_assessment": {"route": "write"}, "result_json": {"a": 1}, "exec_result": {"returncode": 0}}
    assert engine._route_after_evidence_gate(gate) == "confirm"
    assert phased.load(engine.quest_root)["stage"] == phased.CONFIRM
    phased.seed_env(engine.quest_root, _env(0, 0))  # the confirm run starts
    assert engine._route_after_evidence_gate({**gate, "result_json": {"a": 2}}) == "write"
    assert phased.status(phased.load(engine.quest_root)) == phased.CONFIRMED


# ---- the confirm data is looked at once, or the result is not confirmed ------------------------------------------


def _confirm(tmp_path: Path) -> None:
    phased.prepare(tmp_path, "q1")
    phased.enter_confirm(tmp_path, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1,
                         explore_runs=1)


def test_a_confirm_run_repaired_and_run_again_is_not_confirmed(tmp_path: Path) -> None:
    _confirm(tmp_path)
    phased.seed_env(tmp_path, _env(0, 0))
    phased.seed_env(tmp_path, _env(0, 0))  # the script was changed after the first confirm run and run again
    record, lines = phased.record_confirm(tmp_path, {"a": 2})
    assert phased.status(record) == "confirm_reused" and any("run again" in x for x in lines)


def test_any_later_run_on_the_confirm_seeds_makes_the_result_preliminary_even_a_failed_one(tmp_path: Path) -> None:
    _confirm(tmp_path)
    phased.seed_env(tmp_path, _env(0, 0))
    record, _ = phased.record_confirm(tmp_path, {"a": 2})
    assert phased.status(record) == phased.CONFIRMED
    phased.seed_env(tmp_path, _env(0, 0))  # a re-run the review asked for, which crashed
    record, lines = phased.record_confirm(tmp_path, None)
    assert phased.status(record) == "confirm_reused" and lines
    assert phased.record_confirm(tmp_path, None)[1] == [], "said once"


def test_an_empty_result_is_not_a_confirmed_result(tmp_path: Path) -> None:
    _confirm(tmp_path)
    phased.seed_env(tmp_path, _env(0, 0))
    record, _ = phased.record_confirm(tmp_path, {})
    assert phased.status(record) == "confirm_failed"


def test_data_that_becomes_splittable_after_exploration_ran_on_it_whole_is_not_held_back(tmp_path: Path) -> None:
    original = _csv(tmp_path / "inputs" / "data" / "d.csv", 100)
    (tmp_path / "inputs" / "data" / "codebook.json").write_text("{}", encoding="utf-8")
    phased.prepare(tmp_path, "q1")
    phased.seed_env(tmp_path, _env(0, 0))  # exploration ran on every row
    (tmp_path / "inputs" / "data" / "codebook.json").unlink()
    record, _ = phased.prepare(tmp_path, "q1")
    assert record["strategy"] == phased.FRESH_SEEDS and _inputs(tmp_path) == original


def test_more_than_one_data_file_holds_nothing_back(tmp_path: Path) -> None:
    a = _csv(tmp_path / "inputs" / "data" / "patients.csv", 100)
    _csv(tmp_path / "inputs" / "data" / "outcomes.csv", 100)
    record, _ = phased.prepare(tmp_path, "q1")
    assert record["strategy"] == phased.FRESH_SEEDS and "more than one data file" in record["why_no_data"]
    assert (tmp_path / "inputs" / "data" / "patients.csv").read_bytes() == a
    phased.enter_confirm(tmp_path, explore_result={"a": 1}, frozen_sha256=None, stride=1, replicates=1, explore_runs=1)
    phased.seed_env(tmp_path, _env(0, 0))
    phased.record_confirm(tmp_path, {"a": 2})
    assert "more than one file" in phased.summary(phased.load(tmp_path))


def test_too_few_rows_held_back_means_new_seeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(phased, "MIN_HELD_BACK_ROWS", 1000)
    original = _csv(tmp_path / "inputs" / "data" / "d.csv", 100)
    record, _ = phased.prepare(tmp_path, "q1")
    assert record["strategy"] == phased.FRESH_SEEDS and "fewer than the 1000" in record["why_no_data"]
    assert _inputs(tmp_path) == original


def test_a_file_the_person_deleted_is_not_brought_back(tmp_path: Path) -> None:
    _csv(tmp_path / "inputs" / "data" / "d.csv", 100)
    phased.prepare(tmp_path, "q1")
    (tmp_path / "inputs" / "data" / "d.csv").unlink()
    phased.restore_inputs(tmp_path)
    assert not (tmp_path / "inputs" / "data" / "d.csv").exists()


def test_a_part_edited_after_a_hard_stop_never_loses_the_kept_whole_file(tmp_path: Path) -> None:
    original = _csv(tmp_path / "inputs" / "data" / "d.csv", 100)
    phased.prepare(tmp_path, "q1")  # a hard stop: no restore, exploration's part stays in inputs/data/
    with (tmp_path / "inputs" / "data" / "d.csv").open("ab") as f:
        f.write("".join(f"{i},{i + 1}\n" for i in range(1000, 1080)).encode())
    record, _ = phased.prepare(tmp_path, "q1")
    assert record["strategy"] == phased.HELD_BACK, "enough new rows to hold some back"
    kept = list((phased.store_dir(tmp_path) / "original").glob("d.csv.replaced-*"))
    assert len(kept) == 1 and kept[0].read_bytes() == original


def test_held_back_rows_that_could_not_be_kept_apart_mean_nothing_is_confirmed(tmp_path: Path) -> None:
    _confirm(tmp_path)
    phased.mark_compromised(tmp_path, "a data file could not be written")
    phased.seed_env(tmp_path, _env(0, 0))
    record, _ = phased.record_confirm(tmp_path, {"a": 2})
    assert phased.status(record) == "compromised"
    assert phased.evidence_settings(tmp_path)["phased"] == "compromised"
    gaps = _ready_gaps(tmp_path, {"phased": "compromised", "phased_strategy": "fresh_seeds"})
    assert any("could not be kept apart" in g for g in gaps)
    assert "nothing here is confirmed" in phased.summary(record)


# ---- follow-ups: held-back rows out of reach, one confirm run, background jobs -----------------------------------

_GLOBBING_SCRIPT = """\
import glob, sys
seen = set()
for path in glob.glob('**/*', recursive=True) + glob.glob('.*/**/*', recursive=True):
    try:
        with open(path, encoding='utf-8') as f:
            seen.update(line.strip() for line in f)
    except (OSError, UnicodeDecodeError):
        pass
sys.stdout.write('\\n'.join(sorted(seen)))
"""


def _quest(tmp_path: Path, rows: int = 200) -> tuple[Path, bytes]:
    root = tmp_path / "out" / "q1"
    return root, _csv(root / "inputs" / "data" / "d.csv", rows)


def test_an_exploration_script_globbing_the_quest_folder_never_finds_a_held_back_row(tmp_path: Path) -> None:
    import subprocess
    import sys

    root, original = _quest(tmp_path)
    phased.prepare(root, "q1")
    # The held-back rows: the supplied rows exploration's part does not hold (wherever FI keeps them).
    held = set(_data_rows(original)) - set(_data_rows((root / "inputs" / "data" / "d.csv").read_bytes()))
    assert len(held) >= phased.MIN_HELD_BACK_ROWS
    # The experiment runs in the quest folder (a container sees only it, at /work): what can a script there read?
    out = subprocess.run([sys.executable, "-c", _GLOBBING_SCRIPT], cwd=root, capture_output=True, text=True,
                         check=True).stdout
    seen = set(out.splitlines())
    assert _data_rows((root / "inputs" / "data" / "d.csv").read_bytes())[0] in seen, "the script did read the folder"
    assert not held & seen, "a held-back row is readable from the quest folder"


def test_a_quest_that_kept_its_files_inside_the_quest_folder_has_them_moved_out(tmp_path: Path) -> None:
    root, original = _quest(tmp_path)
    phased.prepare(root, "q1")
    store = phased.store_dir(root)
    # The layout before the move: .fi/phased/{original,held_back}/ inside the quest folder.
    old = root / ".fi" / phased.DIR
    for sub in ("original", "held_back"):
        (old / sub).mkdir(parents=True)
        (store / sub / "d.csv").replace(old / sub / "d.csv")
    held = (old / "held_back" / "d.csv").read_bytes()
    phased.prepare(root, "q1")
    assert not old.exists()
    assert (store / "held_back" / "d.csv").read_bytes() == held
    assert (store / "original" / "d.csv").read_bytes() == original


def test_a_renamed_file_keeps_every_row_exploration_saw_on_exploration_s_side(tmp_path: Path) -> None:
    root, _original = _quest(tmp_path)
    phased.prepare(root, "q1")
    explored = set(_data_rows((root / "inputs" / "data" / "d.csv").read_bytes()))
    held = set(_data_rows((phased.store_dir(root) / "held_back" / "d.csv").read_bytes()))
    phased.seed_env(root, _env(0, 0))  # exploration ran on its part
    phased.restore_inputs(root)
    (root / "inputs" / "data" / "d.csv").replace(root / "inputs" / "data" / "d_v2.csv")
    record, _ = phased.prepare(root, "q1")
    assert record["strategy"] == phased.HELD_BACK
    now_held = set(_data_rows((phased.store_dir(root) / "held_back" / "d_v2.csv").read_bytes()))
    assert now_held == held and not now_held & explored


def test_a_file_re_exported_after_exploration_ran_holds_nothing_back(tmp_path: Path) -> None:
    root, _original = _quest(tmp_path)
    phased.prepare(root, "q1")
    phased.seed_env(root, _env(0, 0))
    phased.restore_inputs(root)
    # The same rows written another way (1 -> 1.0): every row's text changed, so a new split would hold back rows
    # exploration saw.
    (root / "inputs" / "data" / "d.csv").write_bytes(
        ("x,y\n" + "".join(f"{i}.0,{i * i}.0\n" for i in range(200))).encode())
    record, lines = phased.prepare(root, "q1")
    assert record["strategy"] == phased.FRESH_SEEDS and "changed since" in record["why_no_data"]
    assert any("new random seeds" in line for line in lines)


def test_rows_added_after_exploration_ran_go_to_exploration_not_to_the_confirm_part(tmp_path: Path) -> None:
    root, _original = _quest(tmp_path)
    phased.prepare(root, "q1")
    held = (phased.store_dir(root) / "held_back" / "d.csv").read_bytes()
    phased.seed_env(root, _env(0, 0))
    phased.restore_inputs(root)
    with (root / "inputs" / "data" / "d.csv").open("ab") as f:
        f.write("".join(f"{i},{i * i}\n" for i in range(200, 260)).encode())
    phased.prepare(root, "q1")
    assert (phased.store_dir(root) / "held_back" / "d.csv").read_bytes() == held
    assert "259,67081" in _data_rows((root / "inputs" / "data" / "d.csv").read_bytes())


def test_a_rerun_that_brings_back_exploration_s_result_is_not_recorded_as_confirmed(tmp_path: Path) -> None:
    """``--from evidence`` after the confirm run started replays the newest checkpoint before the gate: exploration's."""
    root, _original = _quest(tmp_path)
    phased.prepare(root, "q1")
    phased.enter_confirm(root, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1, explore_runs=1)
    phased.seed_env(root, _env(0, 0))  # the confirm run started, and stopped before its result
    record, lines = phased.record_confirm(root, {"a": 1})
    assert phased.status(record) == "compromised" and any("not the one the confirm run produced" in x for x in lines)


def test_only_the_result_the_confirm_run_produced_is_recorded_as_confirmed(tmp_path: Path) -> None:
    for got, want in (({"a": 2}, phased.CONFIRMED), ({"a": 3}, "compromised")):
        root = tmp_path / str(got["a"]) / "q1"
        _csv(root / "inputs" / "data" / "d.csv", 200)
        phased.prepare(root, "q1")
        phased.enter_confirm(root, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1,
                             explore_runs=1)
        phased.seed_env(root, _env(0, 0))
        phased.note_confirm_result(root, {"a": 2})
        assert phased.status(phased.record_confirm(root, got)[0]) == want


def test_a_changed_file_in_the_confirm_stage_means_nothing_is_confirmed(tmp_path: Path) -> None:
    root, _original = _quest(tmp_path)
    phased.prepare(root, "q1")
    phased.enter_confirm(root, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1, explore_runs=1)
    phased.restore_inputs(root)  # a stop before the confirm run: the whole file is back
    with (root / "inputs" / "data" / "d.csv").open("ab") as f:
        f.write(b"999,998001\n")  # saved again in a spreadsheet, say
    record, lines = phased.prepare(root, "q1")
    assert phased.status(record) == "compromised" and any("could not be put in place" in x for x in lines)


def test_a_part_edited_during_exploration_means_the_confirm_run_cannot_be_given_the_held_back_rows(
        tmp_path: Path) -> None:
    root, _original = _quest(tmp_path)
    phased.prepare(root, "q1")
    with (root / "inputs" / "data" / "d.csv").open("ab") as f:
        f.write(b"999,998001\n")
    record, lines = phased.enter_confirm(root, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1,
                                         explore_runs=1)
    assert phased.status(record) == "compromised" and any("could not be put in place" in x for x in lines)


def test_a_file_that_cannot_be_put_back_is_named_not_said_to_be_back(tmp_path: Path) -> None:
    root, _original = _quest(tmp_path)
    phased.prepare(root, "q1")
    with (root / "inputs" / "data" / "d.csv").open("ab") as f:
        f.write(b"999,998001\n")
    lines = phased.restore_inputs(root)
    assert lines and "was changed" in lines[0] and str(phased.store_dir(root)) in lines[0]


def test_a_first_start_that_failed_leaves_a_record_that_nothing_is_confirmed(tmp_path: Path) -> None:
    phased.mark_compromised(tmp_path, "a data file could not be written at a start")
    record = phased.load(tmp_path)
    assert phased.status(record) == "compromised"
    assert phased.prepare(tmp_path, "q1")[0]["compromised"], "the next start holds nothing back"


def test_turned_off_while_exploring_puts_the_whole_file_back_and_holds_nothing_back_later(tmp_path: Path) -> None:
    root, original = _quest(tmp_path)
    phased.prepare(root, "q1")  # a hard stop: exploration's part is still in inputs/data/
    lines = phased.turned_off(root)
    assert (root / "inputs" / "data" / "d.csv").read_bytes() == original and lines
    record, _ = phased.prepare(root, "q1")  # turned on again
    assert record["strategy"] == phased.FRESH_SEEDS and record["late_start"]
    assert (root / "inputs" / "data" / "d.csv").read_bytes() == original


def test_turned_off_during_the_confirm_stage_means_nothing_is_confirmed(tmp_path: Path) -> None:
    root, original = _quest(tmp_path)
    phased.prepare(root, "q1")
    phased.enter_confirm(root, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1, explore_runs=1)
    phased.turned_off(root)
    assert (root / "inputs" / "data" / "d.csv").read_bytes() == original
    assert phased.status(phased.load(root)) == "compromised"


def test_a_paused_experiment_step_counts_as_having_run_before_it_was_turned_on(tmp_path: Path) -> None:
    from core.engine import Engine

    engine = Engine(_config(tmp_path / "out", phased_on=True))
    _csv(engine.quest_root / "inputs" / "data" / "d.csv", 200)
    engine.audit.append("node_started", node="execute")
    engine.audit.append("node_paused", node="execute", pause="results")  # a background job read the whole file
    engine._phased_prepare()
    record = phased.load(engine.quest_root)
    assert record["late_start"] and record["files"] == []


def test_checking_on_the_confirm_run_s_own_background_job_is_not_a_second_run(tmp_path: Path) -> None:
    root, _original = _quest(tmp_path)
    phased.prepare(root, "q1")
    phased.enter_confirm(root, explore_result={"a": 1}, frozen_sha256=None, stride=10, replicates=1, explore_runs=1)
    phased.seed_env(root, _env(0, 0))  # the confirm run submits its job
    phased.note_job_pending(root)  # ... which is pending: the quest waits
    phased.seed_env(root, _env(0, 0))  # a resume checks on it: still pending
    phased.note_job_pending(root)
    phased.seed_env(root, _env(0, 0))  # done
    phased.note_confirm_result(root, {"a": 2})
    record, _ = phased.record_confirm(root, {"a": 2})
    assert record["confirm_executions"] == 1 and phased.status(record) == phased.CONFIRMED


def _bg_config(root: Path):
    cfg = _config(root, phased_on=True)
    cfg.execution.background_jobs = True
    return cfg


def test_a_background_job_s_confirm_run_submits_a_new_job(tmp_path: Path) -> None:
    from core.engine import Engine

    engine = Engine(_bg_config(tmp_path / "out"))
    code = engine.quest_root / "code"
    code.mkdir(parents=True)
    (code / "experiment.py").write_text("import os\nseed = int(os.environ.get('FI_REPLICATE_SEED', '0'))\n",
                                        encoding="utf-8")
    job = engine.quest_root / "job" / "state.json"
    job.parent.mkdir(parents=True)
    job.write_text('{"id": "explore-job"}', encoding="utf-8")
    phased.prepare(engine.quest_root, engine.quest_id)
    engine._runs_code = lambda state: True  # type: ignore[method-assign]
    gate = {"evidence_assessment": {"route": "write"}, "result_json": {"a": 1}, "exec_result": {"returncode": 0}}
    assert engine._route_after_evidence_gate(gate) == "confirm"
    assert not job.exists(), "exploration's job is not reported again as the confirm run's"
    assert (job.parent / "state.explore.json").read_text(encoding="utf-8") == '{"id": "explore-job"}'


def test_the_background_job_prompt_passes_the_seed_on_only_when_on(tmp_path: Path) -> None:
    from core.engine import Engine

    on = Engine(_bg_config(tmp_path / "on"))._job_block()
    assert "Ignore FI_PILOT and FI_REPLICATE_SEED" not in on and "pass it to the job" in on
    off_cfg = _bg_config(tmp_path / "off")
    off_cfg.engine.phased = False
    off = Engine(off_cfg)._job_block()
    # Off: exactly the prompt it always was.
    assert off.endswith("Never sleep-wait for the job. Ignore FI_PILOT and FI_REPLICATE_SEED: no pilot or replicate "
                        "run is made for a background job.\n")
    assert "pass it to the job" not in off


def test_a_confirm_run_on_new_seeds_that_cannot_take_the_seed_is_not_confirmed(tmp_path: Path) -> None:
    from core.engine import Engine

    engine = Engine(_bg_config(tmp_path / "out"))
    code = engine.quest_root / "code"
    code.mkdir(parents=True)
    (code / "experiment.py").write_text("import random\nrandom.seed(42)\nprint(random.random())\n", encoding="utf-8")
    phased.prepare(engine.quest_root, engine.quest_id)
    engine._runs_code = lambda state: True  # type: ignore[method-assign]
    gate = {"evidence_assessment": {"route": "write"}, "result_json": {"a": 1}, "exec_result": {"returncode": 0}}
    assert engine._route_after_evidence_gate(gate) == "write"
    record = phased.load(engine.quest_root)
    assert phased.status(record) == "not_confirmable" and "FI_REPLICATE_SEED" in record["not_confirmable"]
    assert "could only repeat exploration's run" in phased.summary(record)


def _gate_after(tmp_path: Path, files: dict[str, str], state: dict | None = None, *, background: bool = False,
                label: str = "q") -> tuple:
    from core import frozen_protocol as _frozen
    from core.engine import Engine

    engine = Engine(_bg_config(tmp_path / label) if background else _config(tmp_path / label, phased_on=True))
    code = engine.quest_root / "code"
    code.mkdir(parents=True)
    for name, text in files.items():
        (code / name).write_text(text, encoding="utf-8")
    phased.prepare(engine.quest_root, engine.quest_id)
    engine._runs_code = lambda state: True  # type: ignore[method-assign]
    engine._draft_protocol = lambda state: {"grid": {"n": [1]}}  # type: ignore[method-assign]
    gate = {"evidence_assessment": {"route": "write"}, "result_json": {"a": 1}, "exec_result": {"returncode": 0},
            **(state or {})}
    return engine._route_after_evidence_gate(gate), phased.load(engine.quest_root), _frozen.load(engine.quest_root)


def test_a_study_without_randomness_and_without_held_back_data_cannot_be_confirmed(tmp_path: Path) -> None:
    seeded = "import os, random\nrng = random.Random(int(os.environ.get('FI_REPLICATE_SEED', '0')))\n"
    # The runs agreed whatever the seed (recorded by the experiment step), or FI's trial contract runs each setting once.
    for label, files, state in (
            ("ran", {"experiment.py": seeded}, {"result_json_deterministic": True}),
            ("cell", {"simulate.py": "def run_cell(cell):\n    return {'v': 1.0}\n", "experiment.py": ""}, {}),
            ("ode", {"experiment.py": "import math\nprint(math.sin(1.0))\n"}, {})):
        route, record, frozen = _gate_after(tmp_path, files, state, label=label)
        assert route == "write" and phased.status(record) == "not_confirmable", label
        assert frozen is not None, "the protocol is still frozen before the paper is written"


def test_a_study_that_takes_its_seed_or_draws_fresh_numbers_goes_to_the_confirm_run(tmp_path: Path) -> None:
    for label, files in (
            # The seed is read in a helper module the experiment imports.
            ("module", {"rng.py": "import os, random\ngen = random.Random(int(os.environ['FI_REPLICATE_SEED']))\n",
                        "experiment.py": "from rng import gen\nprint(gen.random())\n"}),
            # No seed at all: every run draws new numbers from the operating system.
            ("entropy", {"experiment.py": "import numpy as np\nrng = np.random.default_rng()\nprint(rng.random())\n"}),
            ("trials", {"simulate.py": "def run_trial(cell, trial_id, seed):\n    return {'v': 1.0}\n",
                        "experiment.py": ""})):
        route, record, _frozen = _gate_after(tmp_path, files, label=label)
        assert route == "confirm" and phased.status(record) == "confirming", label


def test_a_temporary_copy_never_stays_in_the_quest_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _original = _quest(tmp_path)
    phased.prepare(root, "q1")
    phased.restore_inputs(root)
    (root / "inputs" / "data" / ".d.csv.fi-tmp").write_bytes(b"x,y\n1,1\n")  # left by an earlier FI's hard stop
    phased.prepare(root, "q1")
    assert [p.name for p in (root / "inputs" / "data").iterdir()] == ["d.csv"]

    def locked(self, target):  # noqa: ANN001, ANN202 -- the file is open in a spreadsheet program
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "replace", locked)
    with pytest.raises(PermissionError):
        phased.restore_inputs(root)
    monkeypatch.undo()
    assert [p.name for p in (root / "inputs" / "data").iterdir()] == ["d.csv"]
    assert not list(phased.store_dir(root).rglob("*.fi-tmp"))


def test_a_quest_approved_before_the_setting_was_listed_is_held_to_it_from_now_on(tmp_path: Path) -> None:
    from core import plan_settings

    root = tmp_path / "q"
    root.mkdir()
    (root / "config.yaml").write_text(f"{plan_settings.INTERVIEW_MARK} FI\nengine:\n  phased: true\n", encoding="utf-8")
    plan_settings.record(root / ".fi", _config(tmp_path / "out", phased_on=True), root)
    path = root / ".fi" / plan_settings.NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["settings"]["engine.phased"]  # a record an older FI wrote
    data["explicit"].remove("engine.phased")
    path.write_text(json.dumps(data), encoding="utf-8")
    assert plan_settings.check(root, root / ".fi", _config(tmp_path / "out", phased_on=True)) == []
    assert json.loads(path.read_text(encoding="utf-8"))["settings"]["engine.phased"] is True
    changed = plan_settings.check(root, root / ".fi", _config(tmp_path / "out", phased_on=False))
    assert any("`engine.phased`" in line for line in changed), changed


@pytest.mark.asyncio
async def test_the_cluster_job_array_of_the_confirm_run_draws_seeds_exploration_never_used(tmp_path: Path) -> None:
    from core import trial_runner
    from core.execution import ExecutionResult

    root, _original = _quest(tmp_path)
    code = root / "code"
    code.mkdir(parents=True)
    (code / "simulate.py").write_text("def run_trial(cell, trial_id, seed):\n    return {'v': 1.0}\n", encoding="utf-8")
    (code / "experiment.py").write_text("print('RESULT_JSON: {}')\n", encoding="utf-8")
    (code / "submit.py").write_text("print('submit')\n", encoding="utf-8")

    class Cluster:
        async def execute(self, cmd, *, cwd, timeout_s, env=None):  # noqa: ANN001, ANN201
            (root / "job").mkdir(exist_ok=True)
            (root / "job" / "state.json").write_text('{"id": "j"}', encoding="utf-8")
            pending = '{"fi_job": {"status": "pending", "id": "j", "note": "queued", "poll_s": 60}}'
            return ExecutionResult(returncode=0, stdout=f"RESULT_JSON: {pending}", stderr="", duration_s=0.1)

    def seeds() -> set[int]:
        plan = json.loads((root / trial_runner.CLUSTER_RECORD).read_text(encoding="utf-8"))["plan"]
        return {t["seed"] for task in plan for t in task["trials"]}

    protocol = {"grid": {"n": [1, 2, 3]}, "runs_per_setting": 4}

    async def submit(env: dict[str, str]) -> None:
        runner = trial_runner.TrialsRunner(Cluster(), quest_root=root, protocol=protocol, deterministic=False,
                                           simulate=code / "simulate.py", analysis=code / "experiment.py",
                                           submit=code / "submit.py")
        await runner.execute(["python", str(code / "experiment.py")], cwd=root, timeout_s=10, env=env)

    phased.prepare(root, "q1")
    await submit(phased.seed_env(root, _env(0, 0)))
    explored = seeds()
    phased.enter_confirm(root, explore_result={"a": 1}, frozen_sha256=None, stride=1_000_000, replicates=1,
                         explore_runs=1)
    await submit(phased.seed_env(root, _env(0, 0)))
    confirmed = seeds()
    assert len(confirmed) == 12 and not confirmed & explored
    assert (root / "job" / "state.previous.json").is_file(), "exploration's job is set aside, a new one submitted"


# ---- a setting that changes what a result means: approved, and asked on every interface -------------------------


def test_explore_then_confirm_is_an_approved_setting_a_hand_edit_cannot_flip(tmp_path: Path) -> None:
    from core import plan_settings

    root = tmp_path / "q"
    root.mkdir()
    (root / "config.yaml").write_text(f"{plan_settings.INTERVIEW_MARK} FI\nengine:\n  phased: true\n", encoding="utf-8")
    plan_settings.record(root / ".fi", _config(tmp_path / "out", phased_on=True), root)
    changed = plan_settings.check(root, root / ".fi", _config(tmp_path / "out", phased_on=False))
    assert any("`engine.phased`" in line and "approved on, now off" in line for line in changed), changed


def test_the_interview_writes_explore_then_confirm_and_keeps_it_on_update(tmp_path: Path) -> None:
    from core.config import Config
    from core.interview import QUESTIONS, InterviewAnswers, answers_to_yaml
    from core.interview_update import load_current_answers

    (question,) = [q for q in QUESTIONS if q.id == "phased"]
    assert question.mid_quest_editable is False and {"cli", "serve", "vscode"} <= set(question.frontends)
    answers = InterviewAnswers(topic="t", title="t", output_kinds=["paper_md"], paper_format="generic",
                               no_simulation=False, study_depth="journal-length", comparative_baseline="",
                               success_metric="", budget="", clarify_mode="auto", review_panel=[],
                               knowledge_enabled=False, provider="openai", provider_model="gpt-4o", phased=True)
    root = tmp_path / "q"
    root.mkdir()
    (root / "config.yaml").write_text(answers_to_yaml(answers), encoding="utf-8")
    assert Config.from_yaml(root / "config.yaml").engine.phased is True
    assert load_current_answers(root)[0].phased is True
    answers.phased = False
    assert "phased" not in answers_to_yaml(answers)
