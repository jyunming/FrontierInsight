"""Whether the seed FI gives each trial reaches the code that draws the trial's random numbers (core/run_facts.py).

The cases are the review's probes (2026-10-15): code that only echoes the seed into its result, overwrites it, hands it to
a model function that ignores it, imports ``run_trial`` from another file, takes it as a keyword-only argument, or draws
from a generator made when the module is loaded. Each was called "reaches" by the earlier check, which only asked whether
the name was read. Where FI cannot follow the seed (a library call, ``**kwargs``), the verdict is "unknown", never "reaches".
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from core import run_facts

FIXTURES = Path(__file__).parent / "fixtures" / "paper_record"

MODEL_IGNORES = "import numpy as np\ndef run_trial(cell, trial_id, seed):\n    return np.random.default_rng(99).normal()\n"

CASES = {
    # name: (simulate.py, other files, verdict, a driving seed or None)
    "uses_it": ("import numpy as np\ndef run_trial(cell, trial_id, seed):\n    rng = np.random.default_rng(seed)\n"
                "    return {'x': rng.normal()}\n", {}, "reaches", None),
    "spawn": ("import numpy as np\ndef run_trial(cell, trial_id, seed):\n    ss = np.random.SeedSequence(seed).spawn(4)\n"
              "    return 1\n", {}, "reaches", None),
    "class": ("import numpy as np\nclass M:\n    def __init__(self, seed):\n        self.rng = np.random.default_rng(seed)\n"
              "    def go(self):\n        return self.rng.normal()\n"
              "def run_trial(cell, trial_id, seed):\n    return M(seed).go()\n", {}, "reaches", None),
    "through_model": ("from model import simulate\ndef run_trial(cell, trial_id, seed):\n    return simulate(cell, seed)\n",
                      {"model.py": "import numpy as np\ndef simulate(cell, s):\n    return np.random.default_rng(s).normal()\n"},
                      "reaches", None),
    "logged_only": ("import numpy as np\ndef run_trial(cell, trial_id, seed):\n    rng = np.random.default_rng(20261015)\n"
                    "    return {'seed': seed, 'x': rng.normal()}\n", {}, "does_not_reach", 20261015),
    "del_seed": ("import numpy as np\ndef run_trial(cell, trial_id, seed):\n    del seed\n    rng = np.random.default_rng(7)\n"
                 "    return {'x': rng.normal()}\n", {}, "does_not_reach", 7),
    "overwritten": ("import numpy as np\ndef run_trial(cell, trial_id, seed=None):\n    seed = 42\n"
                    "    return np.random.default_rng(seed).normal()\n", {}, "does_not_reach", 42),
    "wrapper": ("from model import run_trial as _rt\ndef run_trial(cell, trial_id, seed):\n    return _rt(cell, trial_id, seed)\n",
                {"model.py": MODEL_IGNORES}, "does_not_reach", 99),
    "imported": ("from model import run_trial\n", {"model.py": MODEL_IGNORES}, "does_not_reach", 99),
    "kwonly_ignored": ("import numpy as np\ndef run_trial(cell, trial_id, *, seed):\n    rng = np.random.default_rng(5)\n"
                       "    return rng.normal()\n", {}, "does_not_reach", 5),
    "module_rng": ("import numpy as np\nRNG = np.random.default_rng(123)\ndef run_trial(cell, trial_id, seed):\n"
                   "    np.random.seed(seed)\n    return RNG.normal()\n", {}, "does_not_reach", 123),
    "unseeded_in_model": ("from model import draw\ndef run_trial(cell, trial_id, seed):\n    return draw(cell, seed)\n",
                          {"model.py": "import numpy as np\ndef draw(cell, seed):\n    rng = np.random.default_rng()\n"
                                       "    return rng.normal()\n"}, "does_not_reach", None),
    "kwargs": ("import numpy as np\ndef run_trial(cell, trial_id, seed):\n    return model(**{'seed': seed}, **cell)\n",
               {}, "unknown", None),
    "library": ("import functools, numpy as np\nimport somelib\ndef run_trial(cell, trial_id, seed):\n"
                "    return somelib.simulate(cell, seed=seed)\n", {}, "unknown", None),
    "outside": ("from somelib import run_trial\n", {}, "unknown", None),
    # Second review (P0-1): FI's seed seeds a generator, but a number typed into the code seeds one on the trial's path
    # too. That is "partly", never "reaches", and {{seed}} names both.
    "side_rng_and_literal_rng": ("import numpy as np\ndef run_trial(cell, trial_id, seed):\n"
                                 "    jitter = np.random.default_rng(seed)\n    rng = np.random.default_rng(20261015)\n"
                                 "    return {'x': rng.normal(size=100).mean(), 'j': jitter.random()*0}\n",
                                 {}, "partly", 20261015),
    "model_ignores_the_rng_it_is_given": ("from model import sim\nimport numpy as np\ndef run_trial(cell, trial_id, seed):\n"
                                          "    rng = np.random.default_rng(seed)\n    return sim(cell, rng)\n",
                                          {"model.py": "import numpy as np\ndef sim(cell, rng):\n"
                                                       "    r2 = np.random.default_rng(7)\n    return r2.normal()\n"},
                                          "partly", 7),
    "seed_only_on_a_branch": ("import numpy as np\ndef run_trial(cell, trial_id, seed):\n    if cell.get('debug'):\n"
                              "        rng = np.random.default_rng(seed)\n    else:\n        rng = np.random.default_rng(3)\n"
                              "    return rng.normal()\n", {}, "partly", 3),
    "seed_times_zero": ("import numpy as np\ndef run_trial(cell, trial_id, seed):\n"
                        "    rng = np.random.default_rng(seed * 0 + 5)\n    return rng.normal()\n", {}, "does_not_reach", 5),
    "str_split_is_no_seeder": ("import numpy as np\ndef run_trial(cell, trial_id, seed):\n"
                               "    parts = str(cell).split(str(seed))\n    rng = np.random.default_rng(1)\n"
                               "    return rng.normal()\n", {}, "does_not_reach", 1),
    # Second review (P1-1): a draw from numpy's global generator, stdlib random, or scipy's .rvs with no random_state is
    # unseeded unless FI's seed reaches that global's seed call.
    "numpy_global_draw": ("import numpy as np\ndef run_trial(cell, trial_id, seed):\n    rng = np.random.default_rng(seed)\n"
                          "    return rng.normal() * 0 + np.random.normal()\n", {}, "does_not_reach", None),
    "stdlib_random_draw": ("import random, numpy as np\ndef run_trial(cell, trial_id, seed):\n"
                           "    rng = np.random.default_rng(seed)\n    return random.random() + 0 * rng.random()\n",
                           {}, "does_not_reach", None),
    "scipy_rvs_without_random_state": ("import numpy as np\nfrom scipy import stats\ndef run_trial(cell, trial_id, seed):\n"
                                       "    _ = np.random.default_rng(seed)\n    return float(stats.norm.rvs(size=3).mean())\n",
                                       {}, "does_not_reach", None),
    "global_draw_in_model": ("from model import draw\nimport numpy as np\ndef run_trial(cell, trial_id, seed):\n"
                             "    rng = np.random.default_rng(seed)\n    return draw(cell, rng)\n",
                             {"model.py": "import numpy as np\ndef draw(cell, rng):\n    return rng.normal() + np.random.poisson(3)\n"},
                             "does_not_reach", None),
    "numpy_seeded_global": ("import numpy as np\ndef run_trial(cell, trial_id, seed):\n    np.random.seed(seed)\n"
                            "    return np.random.normal()\n", {}, "reaches", None),
    "stdlib_seeded_global": ("import random\ndef run_trial(cell, trial_id, seed):\n    random.seed(seed)\n"
                             "    return random.random()\n", {}, "reaches", None),
    "scipy_rvs_with_random_state": ("import numpy as np\nfrom scipy import stats\ndef run_trial(cell, trial_id, seed):\n"
                                    "    rng = np.random.default_rng(seed)\n"
                                    "    return float(stats.norm.rvs(size=3, random_state=rng).mean())\n", {}, "reaches", None),
    "torch_manual_seed": ("import torch\ndef run_trial(cell, trial_id, seed):\n    torch.manual_seed(seed)\n"
                          "    return float(torch.rand(1))\n", {}, "reaches", None),
}


def _verdict(tmp_path: Path, src: str, extra: dict[str, str]) -> dict:
    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "code" / "simulate.py").write_text(src, encoding="utf-8")
    for name, text in extra.items():
        (root / "code" / name).write_text(text, encoding="utf-8")
    return run_facts.seed_verdict(root)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_seed_verdict_follows_the_seed_to_a_generator(tmp_path: Path, name: str) -> None:
    src, extra, verdict, driving = CASES[name]
    got = _verdict(tmp_path, src, extra)
    assert got["verdict"] == verdict, (name, got)
    assert got["fi_seed_reaches"] is {"reaches": True, "does_not_reach": False, "partly": False, "unknown": None}[verdict]
    if driving is not None:
        assert driving in [d["value"] for d in got["driving_seeds"]], (name, got["driving_seeds"])
    if verdict != "reaches":
        assert got["why"], "a verdict that is not 'reaches' says why"


def test_no_run_trial_is_no_entry(tmp_path: Path) -> None:
    got = _verdict(tmp_path, "def run_cell(cell):\n    return {'x': 1.0}\n", {})
    assert got["verdict"] == "no_entry" and got["fi_seed_reaches"] is None


def test_e1_the_seed_that_drives_the_trials_is_the_one_in_run_trial_not_the_self_check_s(tmp_path: Path) -> None:
    """E1: run_trial never reads its seed; its generator is seeded from SeedSequence([20261015, ...]) at simulate.py:106.
    The helper it calls for a self-check seeds its own generator with 17 (simulate.py:88): that is not the run's seed. The
    literal 20261015 at simulate.py:176 is in run_cell, which FI's harness does not call (the study's entry is run_trial)."""
    root = tmp_path / "fdr_e1"
    shutil.copytree(FIXTURES / "fdr_e1", root)
    got = run_facts.seed_verdict(root)
    assert got["verdict"] == "does_not_reach" and "never uses it" in got["why"]
    assert got["entry"] == "code/simulate.py:96"
    assert got["driving_seeds"] == [{"where": "code/simulate.py:106", "value": 20261015}]
    assert not got["unseeded"], "model.py:52's unseeded generator is in a function run_trial never calls"
    # Every seed written in the code is still listed (the quality report shows them), 17 among them.
    assert 17 in {s["value"] for s in got["code_seeds"]}


def test_partly_names_fi_s_seed_and_the_fixed_one_in_the_paper(tmp_path: Path) -> None:
    from core import paper_facts

    src, extra, _v, _d = CASES["model_ignores_the_rng_it_is_given"]
    _verdict(tmp_path, src, extra)
    text = paper_facts._seed_text(tmp_path / "q")
    assert "FI's own seed for each trial" in text and "7, at code/model.py:3" in text, text
    assert "derived from base seed" not in text or "fixed in the code" in text


def test_kinetics_seed_reaches(tmp_path: Path) -> None:
    root = tmp_path / "kinetics_s1"
    shutil.copytree(FIXTURES / "kinetics_s1", root)
    got = run_facts.seed_verdict(root)
    assert got["verdict"] == "reaches" and got["fi_seed_reaches"] is True, got
