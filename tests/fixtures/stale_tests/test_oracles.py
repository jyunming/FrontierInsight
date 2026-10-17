"""The plan's checks against known answers, as unit tests. Written by Frontier Insight from the plan; it follows the
plan each time the code changes.

    python -m pytest tests          (or, without pytest: python tests/test_oracles.py)

Each test runs the simulation on the check's small case the way Frontier Insight does before the main run, and compares
the number with the value the plan fixed (and the tolerance it allows).
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("FI_THRESHOLDS", '{"alpha": 0.05, "effect_size": 2.5}')

import simulate  # noqa: E402

CHECKS = json.loads('[{"name": "normal_tail_second_implementation", "case": {"m": 20, "pi1": 0.05, "procedure": "uncorrected", "run_index": 0}, "measure": "max(abs(pvalue_i - 0.5*erfc(z_i/sqrt(2))))", "expected": 0.0, "limit": 1e-12, "per_trial": false}, {"name": "bh_step_up_reference", "case": {"m": 20, "pi1": 0.2, "procedure": "bh", "run_index": 0}, "measure": "sum(BH_rejections != reference_rejections)", "expected": 0.0, "limit": 0.0, "per_trial": false}]')


def _seed(case):
    key = ",".join(f"{axis}={repr(v) if isinstance(v, float) and v.is_integer() else v}" for axis, v in case.items())
    base = int(os.environ.get("FI_REPLICATE_SEED") or 0)
    return int(hashlib.sha256(f"{base}|{key}|0".encode("utf-8")).hexdigest()[:8], 16)


def _measure(check):
    if check["case"] is None:
        found = simulate.oracle()
        want = check["name"].strip().lower()
        value = next((v for k, v in found.items() if str(k).strip().lower() == want), None)
        assert value is not None, f"{check['name']}: simulate.oracle() does not return this check"
        return value
    case = dict(check["case"])
    # With both functions the study runs only run_trial: a check that holds in every single trial runs on it, and so does any
    # other check unless run_cell really calls run_trial (its mean then comes from the study's own trials), as Frontier
    # Insight does.
    if hasattr(simulate, "run_trial") and hasattr(simulate, "run_cell") and not check["per_trial"]:
        # Counted at run time, as Frontier Insight does: run_cell counts only if it really called run_trial.
        real, calls, sums = simulate.run_trial, [0], {}

        def counted(*args, **kwargs):
            calls[0] += 1
            got = real(*args, **kwargs)
            for key, v in (got.items() if isinstance(got, dict) else ()):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    total, n = sums.get(key, (0.0, 0))
                    sums[key] = (total + v, n + 1)
            return got

        simulate.run_trial = counted
        try:
            got = simulate.run_cell(case)
        finally:
            simulate.run_trial = real
        # ... and what it returns agrees, within the check's limit, with the mean of what those trials returned.
        if calls[0] and all(abs(v - sums[k][0] / sums[k][1]) <= check["limit"]
                            for k, v in got.items() if k in sums and isinstance(v, (int, float))):
            return got[check["measure"]]
        return real(case, 0, _seed(case))[check["measure"]]
    if hasattr(simulate, "run_trial") and (not hasattr(simulate, "run_cell") or check["per_trial"]):
        return simulate.run_trial(case, 0, _seed(case))[check["measure"]]
    return simulate.run_cell(case)[check["measure"]]


def _check(index):
    check = CHECKS[index]
    value = float(_measure(check))
    assert math.isfinite(value) and abs(value - check["expected"]) <= check["limit"], (
        f"{check['name']}: measured {value}, the plan expects {check['expected']} within {check['limit']}")


def test_normal_tail_second_implementation():
    'normal_tail_second_implementation'
    _check(0)


def test_bh_step_up_reference():
    'bh_step_up_reference'
    _check(1)


if __name__ == "__main__":
    failed = 0
    for name, fn in [('test_normal_tail_second_implementation', test_normal_tail_second_implementation), ('test_bh_step_up_reference', test_bh_step_up_reference)]:
        try:
            fn()
            print("ok  ", name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", name, e)
    sys.exit(1 if failed else 0)
