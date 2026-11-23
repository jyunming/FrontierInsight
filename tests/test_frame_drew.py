"""The frame measures whether a trial drew random numbers (``Ctx.rng``, ``Ctx.rngs()``, ``Ctx.seed``) and says so in the row FI's
harness writes for the trial (``drew``); what the writer is told a seed means is built from it. Neutral made-up trials."""
from __future__ import annotations

import types

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
