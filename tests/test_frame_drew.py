"""The frame measures whether a trial drew random numbers (``Ctx.rng``, ``Ctx.rngs()``, ``Ctx.seed``) and says so in the row FI's
harness writes for the trial (``drew``); the seed sentence of the paper is built from it. Neutral made-up trials."""
from __future__ import annotations

import json
import types
from pathlib import Path

from core import paper_facts as pf
from core import skeleton


def _frame() -> types.ModuleType:
    mod = types.ModuleType("fi_frame_under_test")
    exec(compile(skeleton.RUNTIME.replace("@VERSION@", str(skeleton.VERSION)), "fi_frame.py", "exec"), mod.__dict__)
    return mod


def _run(frame: types.ModuleType, body) -> dict:
    """What the harness does for one trial: a ``main()`` whose local ``row`` is the trial's row."""
    def main():
        row = {"nonce": "n", "trial": 0, "seed": 7}
        body(frame, frame.Ctx(seed=7, trial=0))
        return row
    return main()


def test_a_trial_that_asks_for_its_generator_or_its_seed_drew_and_one_that_does_not_did_not() -> None:
    frame = _frame()
    assert _run(frame, lambda f, ctx: None)["drew"] is False
    assert _run(frame, lambda f, ctx: ctx.rng)["drew"] is True
    assert _run(frame, lambda f, ctx: ctx.rngs())["drew"] is True
    assert _run(frame, lambda f, ctx: ctx.seed)["drew"] is True


def test_every_trial_of_a_batch_is_told_whether_the_batch_drew() -> None:
    frame = _frame()
    seen = []

    def trial_batch(cell, ctx):
        seen.append(1)
        gens = ctx.rngs()
        return [r for r in ctx.results()]

    def one(trial):
        def main():
            row = {"nonce": "n", "trial": trial, "seed": 10 + trial}
            spec = {"module": "m", "entry": "run_trial", "trials": [{"trial": 0, "seed": 10}, {"trial": 1, "seed": 11}]}
            ctx = frame.Ctx(seed=10 + trial, trial=trial)
            frame.batch_of_trials(trial_batch, {"a": 1}, ctx)
            return row
        return main()

    assert one(0)["drew"] is True and one(1)["drew"] is True and len(seen) == 1  # the second is the cached batch


def test_outside_the_harness_nothing_is_written_and_nothing_fails() -> None:
    frame = _frame()
    ctx = frame.Ctx(seed=3, trial=0)
    assert ctx.rng is not None and ctx.seed == 3


def _quest(tmp_path: Path, rows: list[dict]) -> Path:
    (tmp_path / ".fi" / "trials").mkdir(parents=True)
    (tmp_path / ".fi" / "trials" / "cell0.out.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path


def test_the_seed_sentence_is_built_from_what_the_trials_measured(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pf._run_facts, "seed_verdict", lambda root: {"verdict": pf._run_facts.UNKNOWN})
    monkeypatch.setattr(pf._disclosure, "recorded_seeds", lambda root: (0, frozenset()))
    ok = {"status": "ok"}
    drew = _quest(tmp_path / "a", [{**ok, "drew": True}, {**ok, "drew": True}])
    assert pf._seed_text(drew) == "each trial drew its random numbers from its own seed, derived from base seed 0"
    none = _quest(tmp_path / "b", [{**ok, "drew": False}])
    assert pf._seed_text(none) == "none: no trial drew random numbers (each setting's result does not depend on a seed)"
    old = _quest(tmp_path / "c", [{**ok}])  # a quest from before the record: today's sentence
    assert "FI could not tell from the code" in pf._seed_text(old)
    mixed = _quest(tmp_path / "d", [{**ok, "drew": True}, {**ok, "drew": False}])
    assert "FI could not tell from the code" in pf._seed_text(mixed)
