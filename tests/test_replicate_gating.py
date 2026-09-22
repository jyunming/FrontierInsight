"""Replication should cost what it is worth, and say what it found.

Three changes, all of them things a real quest exposed that the existing
flat-JSON fixtures could not:

* Replicates ran BEFORE the plausibility gate, so every futile repair
  iteration paid for three runs instead of one.
* A deterministic experiment was replicated in full. Honouring
  ``FI_REPLICATE_SEED`` is not the same as consuming randomness: the observed
  script seeded numpy and then integrated an ODE, so all three runs were byte
  identical and the only possible aggregate was ``std=0`` everywhere.
* The aggregator scanned top-level keys only, so a crossed design
  (``by_h`` x integrator) aggregated **nothing** -- reported as
  ``0 numeric keys``, which reads like a broken aggregator.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
    ProviderConfig,
)
from core.engine import (
    Engine, _aggregate_result_json_replicates, _replicate_env,
    _replicate_seed_count, _script_reads_replicate_seed, replicate_seed_reaches_rng, unseeded_rng_calls,
)

# The fixtures' scripts are never executed (the executor is mocked); they are
# only ever SCANNED, for whether they can respond to the seed at all.
FAKE_SEEDED_SCRIPT = 'import os\nseed = int(os.environ.get("FI_REPLICATE_SEED", 0))\n'
# The shape three graded quests actually shipped: an RNG seeded from a constant
# the script wrote down, with the engine's variable nowhere in the file.
TERRA_SHAPED_SCRIPT = (
    "import numpy as np\n"
    "RNG_SEED = 20260917\n"
    "def main() -> None:\n"
    "    rng = np.random.default_rng(RNG_SEED)\n"
    "    print(rng.random())\n"
)


def _er(stdout: str, returncode: int = 0):
    return type("ER", (), {
        "returncode": returncode, "duration_s": 0.1, "timed_out": False,
        "stdout": stdout, "stderr": "",
    })()


def _rj(payload: str) -> str:
    return f"RESULT_JSON: {payload}\n"


def _engine(tmp_path: Path, *, replicates: int) -> Engine:
    cfg = Config(
        topic="replication", title="rep",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="off",
            execute_replicates=replicates, pilot_run=False,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng.executor.install = AsyncMock(  # type: ignore[method-assign]
        return_value=type("IR", (), {"returncode": 0, "stderr": ""})())
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "experiment.py").write_text(
        FAKE_SEEDED_SCRIPT, encoding="utf-8")
    return eng


# --- ordering: the gate before the error bars --------------------------------

@pytest.mark.asyncio
async def test_no_replicates_when_the_primary_result_is_implausible(
    tmp_path: Path,
) -> None:
    """A result about to be regenerated is not worth an error bar."""
    eng = _engine(tmp_path, replicates=3)
    eng.executor.execute = AsyncMock(return_value=_er(_rj('{"rmse": 999.0}')))  # type: ignore[method-assign]
    design = {"result_assertions": [
        {"path": "rmse", "min": 0.0, "max": 10.0, "unit": "unitless"},
    ]}
    await eng._node_execute({"deps": [], "design": design})
    assert eng.executor.execute.await_count == 1, "replicated a rejected result"


@pytest.mark.asyncio
async def test_replicates_still_run_when_the_result_is_plausible(
    tmp_path: Path,
) -> None:
    """The gate must not become a blanket opt-out."""
    eng = _engine(tmp_path, replicates=3)
    eng.executor.execute = AsyncMock(side_effect=[  # type: ignore[method-assign]
        _er(_rj('{"rmse": 0.10}')),
        _er(_rj('{"rmse": 0.20}')),
        _er(_rj('{"rmse": 0.30}')),
    ])
    design = {"result_assertions": [
        {"path": "rmse", "min": 0.0, "max": 10.0, "unit": "unitless"},
    ]}
    patch = await eng._node_execute({"deps": [], "design": design})
    assert eng.executor.execute.await_count == 3
    assert len(patch["result_json_replicates"]) == 3


@pytest.mark.asyncio
async def test_no_assertions_means_no_gating(tmp_path: Path) -> None:
    """Silence in the design means nothing was claimed, not that everything
    failed — replication proceeds as before."""
    eng = _engine(tmp_path, replicates=2)
    eng.executor.execute = AsyncMock(side_effect=[  # type: ignore[method-assign]
        _er(_rj('{"rmse": 0.1}')), _er(_rj('{"rmse": 0.2}')),
    ])
    await eng._node_execute({"deps": []})
    assert eng.executor.execute.await_count == 2


@pytest.mark.asyncio
async def test_replicates_run_when_no_repair_attempt_is_left(tmp_path: Path) -> None:
    """At the cap nothing regenerates the result: the paper is written from
    it, so it gets its error bars. Skipping them left a real SIR quest with one
    seed and no confidence intervals."""
    eng = _engine(tmp_path, replicates=3)
    eng.executor.execute = AsyncMock(side_effect=[  # type: ignore[method-assign]
        _er(_rj('{"rmse": 999.0}')),
        _er(_rj('{"rmse": 998.0}')),
        _er(_rj('{"rmse": 997.0}')),
    ])
    design = {"result_assertions": [
        {"path": "rmse", "min": 0.0, "max": 10.0, "unit": "unitless"},
    ]}
    spent = eng.config.engine.exec_reflect_max_iterations
    patch = await eng._node_execute({"deps": [], "design": design, "exec_reflect_iter": spent})
    assert eng.executor.execute.await_count == 3
    assert len(patch["result_json_replicates"]) == 3
    assert patch["exec_patch_pending"] is False


# --- determinism: stop once two seeds agree ----------------------------------

@pytest.mark.asyncio
async def test_identical_seeds_stop_replication_early(tmp_path: Path) -> None:
    eng = _engine(tmp_path, replicates=5)
    eng.executor.execute = AsyncMock(  # type: ignore[method-assign]
        return_value=_er(_rj('{"rmse": 0.25}')))
    patch = await eng._node_execute({"deps": []})
    assert eng.executor.execute.await_count == 2, "should stop after seeds 0 and 1"
    assert patch["result_json_deterministic"] is True


@pytest.mark.asyncio
async def test_differing_seeds_replicate_in_full(tmp_path: Path) -> None:
    eng = _engine(tmp_path, replicates=3)
    eng.executor.execute = AsyncMock(side_effect=[  # type: ignore[method-assign]
        _er(_rj('{"rmse": 0.10}')),
        _er(_rj('{"rmse": 0.11}')),
        _er(_rj('{"rmse": 0.12}')),
    ])
    patch = await eng._node_execute({"deps": []})
    assert eng.executor.execute.await_count == 3
    assert patch["result_json_deterministic"] is False


@pytest.mark.asyncio
async def test_determinism_is_judged_on_values_not_the_seed_tag(
    tmp_path: Path,
) -> None:
    """``_seed`` differs by construction; it must not mask agreement."""
    eng = _engine(tmp_path, replicates=4)
    eng.executor.execute = AsyncMock(  # type: ignore[method-assign]
        return_value=_er(_rj('{"a": {"b": 1.0}}')))
    patch = await eng._node_execute({"deps": []})
    assert eng.executor.execute.await_count == 2
    seeds = [r["_seed"] for r in patch["result_json_replicates"]]
    assert seeds == [0, 1]


# --- aggregation: nested results ---------------------------------------------

def test_nested_results_aggregate_under_dotted_paths() -> None:
    reps = [
        {"_seed": 0, "by_h": {"0.5": {"RK4": {"err": 1.0}}}},
        {"_seed": 1, "by_h": {"0.5": {"RK4": {"err": 1.2}}}},
    ]
    agg = _aggregate_result_json_replicates(reps)
    assert "by_h.0.5.RK4.err" in agg
    assert agg["by_h.0.5.RK4.err"]["mean"] == pytest.approx(1.1)
    assert agg["by_h.0.5.RK4.err"]["n"] == 2


def test_flat_and_nested_coexist() -> None:
    reps = [
        {"_seed": 0, "top": 5.0, "grp": {"inner": 1.0}},
        {"_seed": 1, "top": 7.0, "grp": {"inner": 3.0}},
    ]
    agg = _aggregate_result_json_replicates(reps)
    assert set(agg) == {"top", "grp.inner"}
    assert agg["top"]["mean"] == pytest.approx(6.0)
    assert agg["grp.inner"]["mean"] == pytest.approx(2.0)


def test_a_path_missing_from_one_replicate_is_skipped() -> None:
    """Aggregating over a path only some seeds produced would silently change
    n between metrics."""
    reps = [
        {"_seed": 0, "grp": {"a": 1.0, "b": 2.0}},
        {"_seed": 1, "grp": {"a": 3.0}},
    ]
    agg = _aggregate_result_json_replicates(reps)
    assert "grp.a" in agg
    assert "grp.b" not in agg


def test_non_numeric_leaves_are_ignored() -> None:
    reps = [
        {"_seed": 0, "grp": {"name": "RK4", "err": 1.0, "ok": True, "xs": [1, 2]}},
        {"_seed": 1, "grp": {"name": "RK4", "err": 2.0, "ok": True, "xs": [1, 2]}},
    ]
    agg = _aggregate_result_json_replicates(reps)
    assert set(agg) == {"grp.err"}, "bools, strings and lists are not measurements"


def test_seed_key_is_not_aggregated_but_a_nested_seed_is_kept() -> None:
    """``_seed`` is synthetic at the top level only; a metric that happens to
    be called `_seed` deeper in the payload is the experiment's own data."""
    reps = [
        {"_seed": 0, "cfg": {"_seed": 11.0}},
        {"_seed": 1, "cfg": {"_seed": 11.0}},
    ]
    agg = _aggregate_result_json_replicates(reps)
    assert "_seed" not in agg
    assert "cfg._seed" in agg


# --- disjoint seed streams ---------------------------------------------------

def test_every_run_including_the_first_is_handed_its_own_seed() -> None:
    """Leaving the primary run unseeded let the script fall back to whatever
    base it had written down (42 in the graded quest) while the replicates were
    handed 1 and 2 — three bases a few integers apart."""
    envs = [_replicate_env({}, i, 1_000_000) for i in range(3)]
    assert [e["FI_REPLICATE_SEED"] for e in envs] == ["0", "1000000", "2000000"]
    # The ordinal is NOT the seed: the figure records are keyed on this.
    assert [e["FI_REPLICATE_INDEX"] for e in envs] == ["0", "1", "2"]


def test_the_seed_streams_are_disjoint_at_this_benchmark_s_trial_count() -> None:
    """The defect on its real numbers, then the fix.

    The graded quest swept 3 reproduction numbers x 3 population sizes and drew
    300 trajectories in each cell, seeding every trajectory ``base + counter``
    off one base per run. Three bases a few apart therefore repeat almost every
    trajectory, and the spread across those runs is not sampling error.
    """
    trials = 3 * 3 * 300  # 2,700 draws per run

    def stream(base: int) -> set[int]:
        return {base + i for i in range(trials)}

    # What shipped: bases 42 (the script's own default) and 1 and 2.
    shipped = [stream(b) for b in (42, 1, 2)]
    assert len(shipped[0] & shipped[1]) == 2659
    assert len(shipped[0] & shipped[2]) == 2660
    assert len(shipped[1] & shipped[2]) == 2699

    # With the stride, no two runs share a single draw. The bases are taken
    # from the engine's OWN env builder rather than recomputed here, so this
    # goes red if the striding is ever dropped from the code that ships.
    stride = EngineConfig().replicate_seed_stride
    bases = [int(_replicate_env({}, i, stride)["FI_REPLICATE_SEED"]) for i in range(3)]
    strided = [stream(b) for b in bases]
    assert all(
        not (strided[i] & strided[j])
        for i in range(3) for j in range(i + 1, 3)
    )
    # And that is a guarantee, not an accident of 2,700: it holds for any run
    # drawing fewer than a stride of seeds.
    assert trials < stride


# --- a script that cannot respond to its seed --------------------------------

def test_a_script_that_reads_the_seed_is_recognised(tmp_path: Path) -> None:
    script = tmp_path / "experiment.py"
    script.write_text(FAKE_SEEDED_SCRIPT, encoding="utf-8")
    assert _script_reads_replicate_seed(script) is True


def test_a_script_that_hardcodes_its_seed_is_recognised(tmp_path: Path) -> None:
    script = tmp_path / "experiment.py"
    script.write_text(TERRA_SHAPED_SCRIPT, encoding="utf-8")
    assert _script_reads_replicate_seed(script) is False


def test_an_unreadable_script_gets_the_benefit_of_the_doubt(tmp_path: Path) -> None:
    """Silence is not evidence of a fault, and the runtime check still runs."""
    assert _script_reads_replicate_seed(tmp_path / "does_not_exist.py") is True


@pytest.mark.asyncio
async def test_identical_runs_of_a_seed_ignoring_script_are_not_replicates(
    tmp_path: Path,
) -> None:
    """The case that reached production. The runs are identical because the
    script cannot see the seed, so they are ONE run repeated. Publishing no
    replicate list is what stops every downstream path from describing a single
    run as "the mean over N seeds" with a confidence interval."""
    eng = _engine(tmp_path, replicates=3)
    (eng.quest_root / "code" / "experiment.py").write_text(
        TERRA_SHAPED_SCRIPT, encoding="utf-8")
    eng.executor.execute = AsyncMock(  # type: ignore[method-assign]
        return_value=_er(_rj('{"major_outbreak_probability": 0.667}')))
    patch = await eng._node_execute({"deps": []})

    assert eng.executor.execute.await_count == 2, "one extra run settles it"
    assert not patch.get("result_json_replicates"), "a repeated run is not replicates"
    assert patch["result_json_deterministic"] is False, "unreplicated, not deterministic"
    assert patch["result_json_replicate_seed_ignored"] is True
    # The downstream consequence that matters: nothing can say "seed 0 of N".
    assert _replicate_seed_count(patch) is None


@pytest.mark.asyncio
async def test_a_seeded_script_whose_runs_agree_is_still_deterministic(
    tmp_path: Path,
) -> None:
    """Do not weaken the honest case. A script that DOES read the seed and
    still computes the same answer every time (integrating an ODE, say) keeps
    exactly its previous behaviour."""
    eng = _engine(tmp_path, replicates=5)
    eng.executor.execute = AsyncMock(  # type: ignore[method-assign]
        return_value=_er(_rj('{"rmse": 0.25}')))
    patch = await eng._node_execute({"deps": []})

    assert eng.executor.execute.await_count == 2
    assert patch["result_json_deterministic"] is True
    assert len(patch["result_json_replicates"]) == 2
    assert patch["result_json_replicate_seed_ignored"] is False


@pytest.mark.asyncio
async def test_runs_that_legitimately_agree_to_many_digits_still_report(
    tmp_path: Path,
) -> None:
    """The guard is about runs that COULD NOT have differed, never about runs
    that merely happened not to differ much. These three agree to three decimal
    places and must still get their full aggregate."""
    eng = _engine(tmp_path, replicates=3)
    eng.executor.execute = AsyncMock(side_effect=[  # type: ignore[method-assign]
        _er(_rj('{"final_size": 0.5812670}')),
        _er(_rj('{"final_size": 0.5810260}')),
        _er(_rj('{"final_size": 0.5810780}')),
    ])
    patch = await eng._node_execute({"deps": []})

    assert eng.executor.execute.await_count == 3
    assert len(patch["result_json_replicates"]) == 3
    assert patch["result_json_deterministic"] is False
    assert patch["result_json_replicate_seed_ignored"] is False
    assert _replicate_seed_count(patch) == 3
    agg = _aggregate_result_json_replicates(patch["result_json_replicates"])
    assert agg["final_size"]["n"] == 3
    assert agg["final_size"]["ci_lower"] < agg["final_size"]["ci_upper"]


# --- and analyze is told why there are no error bars -------------------------

async def _analyze_prompt(eng: Engine, state: dict) -> str:
    prompts: list[str] = []

    async def fake_chat(prompt: str, *, node: str | None = None) -> str:
        prompts.append(prompt)
        return json.dumps({"summary": "s", "key_findings": [], "next_step": "publish"})

    eng._chat = fake_chat  # type: ignore[assignment]
    await eng._node_analyze(state)  # type: ignore[arg-type]
    return "\n".join(prompts)


@pytest.mark.asyncio
async def test_analyze_is_told_the_run_was_never_replicated(tmp_path: Path) -> None:
    """Withholding the replicate list is not enough by itself: without a reason
    analyze sees an ordinary single-seed run and never mentions that replication
    was asked for and could not happen."""
    eng = _engine(tmp_path, replicates=3)
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    prompt = await _analyze_prompt(eng, {
        "result_json": {"major_outbreak_probability": 0.667},
        "result_json_replicate_seed_ignored": True,
        "exec_result": {"returncode": 0}, "figures": [], "design": {},
    })
    assert "never reads FI_REPLICATE_SEED" in prompt
    assert "ONE measurement here, not several" in prompt
    assert "do NOT report a mean over seeds" in prompt


@pytest.mark.asyncio
async def test_an_ordinary_run_gets_no_such_note(tmp_path: Path) -> None:
    eng = _engine(tmp_path, replicates=3)
    eng.quest_root = tmp_path  # type: ignore[attr-defined]
    prompt = await _analyze_prompt(eng, {
        "result_json": {"major_outbreak_probability": 0.667},
        "exec_result": {"returncode": 0}, "figures": [], "design": {},
    })
    assert "never reads FI_REPLICATE_SEED" not in prompt


# --- and the verdict does not outlive the script it was about ----------------

@pytest.mark.asyncio
async def test_a_repaired_seed_ignoring_script_clears_the_earlier_replicates(
    tmp_path: Path,
) -> None:
    """``_node_execute`` runs again on a repair and on a re_experiment, and the
    state's fields are last-value channels. A pass that merely WITHHELD the
    replicate list would leave the previous script's seeds sitting on the
    state, and analyze would aggregate those against this script's result."""
    eng = _engine(tmp_path, replicates=3)
    (eng.quest_root / "code" / "experiment.py").write_text(
        TERRA_SHAPED_SCRIPT, encoding="utf-8")
    eng.executor.execute = AsyncMock(  # type: ignore[method-assign]
        return_value=_er(_rj('{"p": 0.667}')))
    patch = await eng._node_execute({  # the state an earlier, seeded pass left
        "deps": [],
        "result_json_replicates": [{"_seed": k, "p": 0.1 * k} for k in range(3)],
        "result_json_deterministic": False,
    })
    assert patch["result_json_replicates"] == [], "the earlier seeds must not survive"
    assert patch["result_json_replicate_seed_ignored"] is True
    assert _replicate_seed_count(patch) is None


@pytest.mark.asyncio
async def test_a_repaired_seeded_script_clears_the_earlier_verdict(
    tmp_path: Path,
) -> None:
    """The other direction: a quest whose earlier script ignored the seed must
    not carry "there is one measurement here" into a pass that replicated
    properly, or analyze gets the aggregate and the banner together."""
    eng = _engine(tmp_path, replicates=3)
    eng.executor.execute = AsyncMock(side_effect=[  # type: ignore[method-assign]
        _er(_rj('{"p": 0.10}')), _er(_rj('{"p": 0.20}')), _er(_rj('{"p": 0.30}')),
    ])
    patch = await eng._node_execute({
        "deps": [], "result_json_replicate_seed_ignored": True,
    })
    assert len(patch["result_json_replicates"]) == 3
    assert patch["result_json_replicate_seed_ignored"] is False


# --- a script the seed cannot reach ------------------------------------------
#
# The shape a graded quest shipped: the script reads FI_REPLICATE_SEED and
# seeds numpy's legacy global with it, then draws every trajectory from a
# Generator built ninety lines earlier with no seed at all. It passes
# ``_script_reads_replicate_seed`` -- the name is right there -- and its
# replicates differ, so nothing downstream notices that the seed reached none
# of the randomness and no run can be reproduced.

GILLESPIE_SHAPED_SCRIPT = (
    "import os\n"
    "import numpy as np\n"
    "def trial() -> float:\n"
    "    rng = np.random.default_rng()\n"
    "    return float(rng.random())\n"
    "def main() -> None:\n"
    "    np.random.seed(int(os.environ.get('FI_REPLICATE_SEED', '0')))\n"
    "    print(trial())\n"
)


@pytest.mark.parametrize("code, line", [
    (GILLESPIE_SHAPED_SCRIPT, 4),
    ("import numpy as np\nrng = np.random.default_rng(None)\n", 2),
    ("import numpy as np\nrng = np.random.default_rng(seed=None)\n", 2),
    ("import numpy as np\nnp.random.seed()\n", 2),
    ("from numpy.random import default_rng\nrng = default_rng()\n", 2),
    ("import numpy.random as nr\nrng = nr.default_rng()\n", 2),
    ("from numpy import random as npr\nrng = npr.RandomState()\n", 2),
    ("import random\nr = random.Random()\n", 2),
])
def test_a_generator_built_without_a_seed_is_found(code: str, line: int) -> None:
    assert [n for n, _ in unseeded_rng_calls(code)] == [line]


@pytest.mark.parametrize("code", [
    # Seeded, in every spelling.
    "import numpy as np\nrng = np.random.default_rng(12)\n",
    "import numpy as np\nrng = np.random.default_rng(seed=12)\n",
    "import os\nimport numpy as np\n"
    "rng = np.random.default_rng(int(os.environ['FI_REPLICATE_SEED']))\n",
    "import numpy as np\nkw = {}\nrng = np.random.default_rng(**kw)\n",  # unknowable
    "import numpy as np\nnp.random.seed(7)\n",
    # The two fallback idioms the archived quests write, where every call site
    # hands in a seeded generator and the branch is never taken.
    "import numpy as np\n"
    "def trial(rng=None):\n"
    "    if rng is None:\n"
    "        rng = np.random.default_rng()\n"
    "    return rng.random()\n"
    "trial(np.random.default_rng(3))\n",
    "import numpy as np\n"
    "def trial(seed=None):\n"
    "    if seed is not None:\n"
    "        rng = np.random.RandomState(seed)\n"
    "    else:\n"
    "        rng = np.random.RandomState()\n"
    "    return rng.random()\n"
    "trial(3)\n",
    # A name that is not numpy's or the stdlib's, however much it looks it.
    "from mypkg import Random\nr = Random()\n",
    "class Random:\n    pass\nr = Random()\n",
    # Not valid Python: it has a louder problem than its seeding.
    "import numpy as np\nrng = np.random.default_rng(\n",
])
def test_a_reproducible_script_is_left_alone(code: str) -> None:
    assert unseeded_rng_calls(code) == []


# --- P1-5: does FI_REPLICATE_SEED actually flow into a seeding call, not merely appear in the file -----------------


def test_a_seed_read_directly_into_the_generator_reaches_it() -> None:
    code = "import numpy as np\nimport os\nrng = np.random.default_rng(int(os.environ.get('FI_REPLICATE_SEED', 0)))\n"
    assert replicate_seed_reaches_rng(code) is True


def test_a_seed_read_into_a_variable_and_then_passed_reaches_it() -> None:
    code = (
        "import numpy as np\nimport os\n"
        "seed = int(os.environ.get('FI_REPLICATE_SEED', 0))\n"
        "rng = np.random.default_rng(seed)\n"
    )
    assert replicate_seed_reaches_rng(code) is True


def test_a_seed_read_via_os_getenv_or_a_subscript_reaches_it() -> None:
    assert replicate_seed_reaches_rng(
        "import numpy as np\nimport os\nrng = np.random.default_rng(int(os.getenv('FI_REPLICATE_SEED', '0')))\n",
    ) is True
    assert replicate_seed_reaches_rng(
        "import numpy as np\nimport os\nrng = np.random.default_rng(int(os.environ['FI_REPLICATE_SEED']))\n",
    ) is True


def test_a_seed_read_and_used_only_to_reseed_the_legacy_global_reaches_it() -> None:
    code = "import numpy as np\nimport os\nnp.random.seed(int(os.environ.get('FI_REPLICATE_SEED', 0)))\n"
    assert replicate_seed_reaches_rng(code) is True


def test_a_seed_read_into_an_unused_name_does_not_reach_a_hardcoded_generator() -> None:
    """The exact blind spot P1-5 closes: the file mentions FI_REPLICATE_SEED (so
    ``_script_reads_replicate_seed`` cannot rule it out), but the value never reaches the generator that is
    actually seeded -- from a hardcoded constant instead."""
    code = (
        "import numpy as np\nimport os\n"
        "_unused = os.environ.get('FI_REPLICATE_SEED')\n"
        "RNG_SEED = 20260917\n"
        "rng = np.random.default_rng(RNG_SEED)\n"
    )
    assert replicate_seed_reaches_rng(code) is False


def test_a_seed_reaching_one_generator_and_a_decoy_hardcoded_generator_does_not_pass() -> None:
    """The reviewed loophole a lenient "any one connects" version would have: a script that correctly seeds ONE
    generator from FI_REPLICATE_SEED and a SECOND, unused for anything but decoration, from a hardcoded constant,
    while the actual measured randomness comes from a THIRD generator seeded the same hardcoded way. Every
    recognised call must connect, not just one, or this script would read as reproducible when its real result
    is not."""
    code = (
        "import numpy as np\nimport os\n"
        "seed = int(os.environ.get('FI_REPLICATE_SEED', 0))\n"
        "decoy = np.random.default_rng(seed)  # correctly seeded, never used for anything that matters\n"
        "RNG_SEED = 20260917\n"
        "rng = np.random.default_rng(RNG_SEED)  # this is what the measured result actually comes from\n"
    )
    assert replicate_seed_reaches_rng(code) is False


def test_a_seed_split_across_a_tuple_assignment_is_still_tracked() -> None:
    """``seed, label = int(os.environ.get('FI_REPLICATE_SEED', 0)), "run"`` binds two names at once; only the one
    whose element of the tuple actually reads the environment is marked, not both."""
    code = (
        "import numpy as np\nimport os\n"
        "seed, label = int(os.environ.get('FI_REPLICATE_SEED', 0)), 'run'\n"
        "rng = np.random.default_rng(seed)\n"
    )
    assert replicate_seed_reaches_rng(code) is True
    # `label` was NOT seeded from the environment; a generator seeded from it must not be granted either.
    mislabelled = code.replace("np.random.default_rng(seed)", "np.random.default_rng(label)")
    assert replicate_seed_reaches_rng(mislabelled) is False


def test_a_script_that_never_names_the_seed_at_all_does_not_reach_it() -> None:
    assert replicate_seed_reaches_rng(TERRA_SHAPED_SCRIPT) is False


def test_a_script_with_no_rng_call_at_all_has_nothing_to_be_disconnected_from() -> None:
    """A purely numerical script (an ODE solve, say) that reads the seed but seeds nothing -- there is no
    randomness for the value to reach, so this check has nothing to say and defers to the old "it reads the seed"
    reasoning. FAKE_SEEDED_SCRIPT is exactly this shape: it is the fixture several tests below build a real
    ``Engine`` run around and expect ``result_json_deterministic`` to still be settled by ``reads_seed`` alone."""
    assert replicate_seed_reaches_rng(FAKE_SEEDED_SCRIPT) is True


def test_an_unreadable_script_gets_the_benefit_of_the_doubt_here_too(tmp_path: Path) -> None:
    from core.engine import _replicate_seed_reaches_rng
    assert _replicate_seed_reaches_rng(tmp_path / "does_not_exist.py") is True


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::SyntaxWarning")
def test_every_archived_generated_script_but_the_known_one_is_left_alone() -> None:
    """The sweep that decided this check was worth shipping, kept as a test.

    Over every script the stored quests generated, the only finding is the
    Gillespie quest whose Generator drew from entropy; the two fallback idioms
    that earlier drafts flagged are in this corpus and must stay silent. Skips
    when the outputs are not in the checkout (CI has no quest archive).
    """
    outputs = Path(__file__).resolve().parents[1] / "outputs"
    scripts = sorted(outputs.glob("*/code/*.py"))
    if len(scripts) < 50:
        pytest.skip("no archived quest outputs in this checkout")
    flagged = {
        path.parent.parent.name: unseeded_rng_calls(
            path.read_text(encoding="utf-8", errors="replace"))
        for path in scripts
        if unseeded_rng_calls(path.read_text(encoding="utf-8", errors="replace"))
    }
    assert flagged == {"1789886314-fi-trend-sir-fin-fm2-3eb3a8": [(43, "np.random.default_rng()")]}
