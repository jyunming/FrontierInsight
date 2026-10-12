"""Tests for core/probes_run.py with a fake executor (no subprocess): the budget, the confirmation rule, the record and resume.

The fake executor stands in for the quest's Python: it reads the spec the harness would read and writes the rows the
harness would write, calling a plain Python function for each trial. A slow-tier replay of a stored quest is at the end.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import probes as P
from core import probes_run as PR

GRID = {"dt": [0.4, 0.2, 0.1, 0.05], "n": [10, 100, 1000]}
PROTOCOL = {"grid": GRID, "metrics": [{"id": "err"}, {"id": "rate"}, {"id": "dof"}]}


class FakeExecutor:
    """Runs ``fn(cell, seed) -> dict`` per trial the way trial_runner's harness does."""

    def __init__(self, fn, delay=0.0):
        self.fn, self.delay, self.calls = fn, delay, []

    async def execute(self, cmd, *, cwd, timeout_s, env=None):
        spec_path, out_path = Path(cwd) / cmd[2], Path(cwd) / cmd[3]
        assert (Path(cwd) / cmd[1]).is_file()          # the harness copy exists while the cells run
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec_path.unlink()
        if self.delay:
            time.sleep(self.delay)
        self.calls.append({"cell": spec["cell"], "seeds": [t["seed"] for t in spec["trials"]], "entry": spec["entry"]})
        rows = []
        for t in spec["trials"]:
            row = {"nonce": spec["nonce"], "trial": t["trial"], "seed": t["seed"]}
            try:
                row.update(status="ok", values={k: float(v) for k, v in self.fn(spec["cell"], t["seed"]).items()})
            except Exception as e:  # noqa: BLE001
                row.update(status="failed", reason=f"{type(e).__name__}: {e}")
            rows.append(row)
        out_path.write_text("\n".join(json.dumps(r, allow_nan=True) for r in rows) + "\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="", duration_s=0.0, timed_out=False)


def quest(tmp_path: Path, entry="run_trial") -> Path:
    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(f"def {entry}(cell, *a):\n    return {{'err': 1.0}}\n", encoding="utf-8")
    return root


def noise(seed: int) -> float:
    return ((seed % 1000) / 1000.0 - 0.5) * 0.02


def good(cell, seed):
    return {"err": cell["dt"] ** 2 * 3 + noise(seed), "rate": 5.0 + noise(seed), "dof": 40.0 * cell["n"] ** 0.5 / 10 + cell["dt"] * 10 + noise(seed)}


def rel(kind, **kw):
    base = dict(id="R1", kind=kind, metric="err", setting="dt", quote="q", strength="core")
    base.update(kw)
    return P.Relation(**base)


def go(coro):
    return asyncio.run(coro)


def run(root, fn, relations, **kw):
    ex = kw.pop("executor", None) or FakeExecutor(fn)
    rec = go(PR.run(ex, "python", root, PROTOCOL, relations, **kw))
    return rec, ex


def mine(rec, root):
    return rec["results"][PR.code_sha(root)]


def test_held_relations_write_the_record_and_remove_the_harness(tmp_path):
    root = quest(tmp_path)
    rels = [rel("sensitivity"), rel("monotone", id="R2", direction="up"), rel("invariance", id="R3", metric="rate")]
    rec, ex = run(root, good, rels)
    res = mine(rec, root)
    assert {r: res[r]["verdict"] for r in ("R1", "R2", "R3")} == {"R1": "held", "R2": "held", "R3": "held"}
    assert res["saturation"]["verdict"] == "held"
    saved = json.loads((root / PR.RECORD).read_text(encoding="utf-8"))
    assert set(saved) >= {"plan_sha", "asked", "relations", "dropped", "results", "repairs"} and saved["results"] == rec["results"]
    plain = json.loads((root / PR.SUMMARY).read_text(encoding="utf-8"))
    assert {i["id"] for i in plain["held"]} >= {"R1", "R2", "R3"} and plain["broken"] == [] and plain["code_sha256"] == PR.code_sha(root)
    assert not (root / ".fi" / "probes_run").exists()           # FI's harness copy is gone
    assert all(len(c["seeds"]) == P.DEFAULT_SEEDS for c in ex.calls)
    assert all(c["seeds"] == ex.calls[0]["seeds"] for c in ex.calls)   # paired: the same seeds in every cell


def test_a_setting_the_code_ignores_is_broken_only_after_a_second_run_agrees(tmp_path):
    root = quest(tmp_path)

    def ignores_dt(cell, seed):
        return {"err": 1.0 + noise(seed) * 0.0, "rate": 5.0}

    rec, ex = run(root, ignores_dt, [rel("sensitivity")])
    r = mine(rec, root)["R1"]
    assert r["verdict"] == "broken" and "exactly the same" in r["says"] and "second run" in r["says"]
    first, second = ex.calls[0], ex.calls[-1]
    assert first["seeds"] != second["seeds"]                    # fresh seeds
    assert first["cell"] != second["cell"] or len(ex.calls) > 2  # and a different size of the change
    assert any(c["cell"]["dt"] == 0.2 for c in ex.calls)         # (the dearest end of the axis is dropped)


def test_a_first_run_that_looks_wrong_but_is_not_confirmed_stays_unconfirmed(tmp_path):
    root = quest(tmp_path)
    first_seeds = {PR._tr.trial_seed(0, "", i) for i in range(P.DEFAULT_SEEDS)}

    def flaky(cell, seed):          # only wrong on the first set of seeds
        return {"err": 1.0 if seed in first_seeds else cell["dt"] * 7.0}

    rec, _ = run(root, flaky, [rel("sensitivity")])
    r = mine(rec, root)["R1"]
    assert r["verdict"] == "unconfirmed" and "second run" in r["says"] and "first_run" in r["numbers"]


def test_budget_marks_the_relations_it_cannot_afford_too_costly(tmp_path):
    root = quest(tmp_path)
    rels = [rel("monotone", direction="up"), rel("sensitivity", id="R2", setting="n"), rel("invariance", id="R3", setting="n", metric="rate")]
    rec, ex = run(root, good, rels, executor=FakeExecutor(good, delay=0.05), budget_s=0.12)
    res = mine(rec, root)
    assert res["R1"]["verdict"] in ("held", "unconfirmed")      # core relations first, as far as the budget goes
    costly = [r for r, v in res.items() if r.startswith("R") and PR.TOO_COSTLY in v["says"]]
    assert costly, res
    assert all(res[r]["verdict"] == "unconfirmed" for r in costly)   # never broken for want of time
    assert len(ex.calls) < 8


def test_resume_reuses_a_result_for_the_same_code_and_runs_again_for_new_code(tmp_path):
    root = quest(tmp_path)
    rels = [rel("sensitivity")]
    rec, ex = run(root, good, rels)
    n = len(ex.calls)
    rec2, ex2 = run(root, good, rels)
    assert ex2.calls == [] and rec2["results"] == rec["results"] and n > 0
    (root / "code" / "simulate.py").write_text("def run_trial(cell, *a):\n    return {'err': 2.0}\n", encoding="utf-8")
    rec3, ex3 = run(root, good, rels)
    assert len(ex3.calls) == n and len(rec3["results"]) == 2    # both code versions are kept
    # a relation added later is run alone
    more = rels + [rel("monotone", id="R2", direction="up")]
    rec4, ex4 = run(root, good, more)
    assert rec4["results"] != rec3["results"] and ex4.calls != []


def test_relations_the_grid_cannot_test_and_failing_cells_are_unconfirmed(tmp_path):
    root = quest(tmp_path)
    one = {"grid": {"dt": [0.1], "n": [1, 2, 3]}, "metrics": [{"id": "err"}]}
    rec = go(PR.run(FakeExecutor(good), "python", root, one, [rel("sensitivity")]))
    assert mine(rec, root)["R1"]["verdict"] == "unconfirmed"

    def crashes(cell, seed):
        if cell["dt"] < 0.1:
            raise ValueError("step too small")
        return good(cell, seed)

    root2 = quest(tmp_path / "b")
    rec2, _ = run(root2, crashes, [rel("monotone", direction="up")])
    r = mine(rec2, root2)["R1"]
    assert r["verdict"] == "unconfirmed" and "step too small" in r["says"]
    rec3, _ = run(quest(tmp_path / "c"), good, [rel("sensitivity", metric="rate", setting="n")], metric_alias={"rate": "not_returned"})
    assert "does not return" in list(rec3["results"].values())[0]["R1"]["says"]


def test_a_value_pinned_at_a_bound_is_found_with_no_relation_at_all(tmp_path):
    root = quest(tmp_path)

    def zero_dof(cell, seed):
        return {"err": cell["dt"] * 2, "dof": 0.0}

    rec, _ = run(root, zero_dof, [])
    res = mine(rec, root)
    assert res["saturation:dof"]["verdict"] == "broken" and res["saturation:dof"]["strength"] == "core"
    assert res["saturation"]["verdict"] == "broken"


def test_a_deterministic_simulation_runs_one_seed_per_cell(tmp_path):
    root = quest(tmp_path, entry="run_cell")
    rec, ex = run(root, lambda c, s: {"err": c["dt"] * 2}, [rel("monotone", direction="up")])
    assert all(c["entry"] == "run_cell" and c["seeds"] == [None] for c in ex.calls)
    assert mine(rec, root)["R1"]["verdict"] == "held"


def test_no_simulation_to_call_leaves_every_relation_unconfirmed(tmp_path):
    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text("x = 1\n", encoding="utf-8")
    rec, ex = run(root, good, [rel("sensitivity")])
    assert ex.calls == [] and mine(rec, root)["R1"]["verdict"] == "unconfirmed"


def test_stored_cells_skip_settings_that_never_finished(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "trials.json").write_text(json.dumps({"cells": [
        {"cell": {"a": 1}, "metrics": {"x": {"values": [1.0, 2.0]}}}, {"cell": {"a": 2}, "metrics": {}}]}), encoding="utf-8")
    assert PR.stored_cells(tmp_path) == [{"cell": {"a": 1}, "metrics": {"x": [1.0, 2.0]}}]


# ------------------------------------------------------------------------------------------------ slow replay of a stored quest

KINETICS = Path("C:/fi_runs/suite/rounds/r6t/first_order_kinetics/out/1791722337-kinetics-parameter-recovery-dfc46d")
FIXTURES = Path(__file__).parent / "fixtures" / "probes"


@pytest.mark.slow
@pytest.mark.skipif(not KINETICS.is_dir(), reason="the stored r6t kinetics quest is not on this machine")
def test_replay_of_the_known_good_kinetics_quest_has_no_broken_relation():
    import importlib.util

    spec = importlib.util.spec_from_file_location("probe_replay", Path(__file__).parent.parent / "dev" / "probe_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fixture = json.loads((FIXTURES / "kinetics.json").read_text(encoding="utf-8"))
    rec = go(mod._replay(KINETICS, fixture, P.DEFAULT_SEEDS, 120))
    (res,) = rec["results"].values()
    broken = [k for k, v in res.items() if v["verdict"] == "broken" and k != "saturation"]
    assert broken == []
    assert sum(v["verdict"] == "held" for k, v in res.items() if k.startswith("R")) >= 6


@pytest.mark.slow
@pytest.mark.skipif(not (KINETICS / "code" / "simulate.py").is_file(), reason="the stored r6t kinetics quest is not on this machine")
def test_replay_catches_a_setting_the_code_no_longer_reads():
    import importlib.util

    spec = importlib.util.spec_from_file_location("probe_replay", Path(__file__).parent.parent / "dev" / "probe_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fixture = json.loads((FIXTURES / "kinetics.json").read_text(encoding="utf-8"))
    text = (KINETICS / "code" / "simulate.py").read_text(encoding="utf-8")
    broken_text = P.ignore_setting(text, "n_time", 12)
    assert broken_text
    rec = go(mod._replay(KINETICS, fixture, P.DEFAULT_SEEDS, 120, edits={"simulate.py": broken_text}))
    (res,) = rec["results"].values()
    assert res["R3"]["verdict"] == "broken"
