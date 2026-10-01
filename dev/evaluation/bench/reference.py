"""The benchmark's own reference computations: where an expected value in an ``answer.json`` comes from when it is not a
closed form printed in a textbook. Each function is small, uses numpy only, and is what the answer's ``source`` names,
so a reader can compute the value again (``fi tools bench reference <task>``).

Nothing here is FI: the scorer never runs these during a benchmark; they only re-derive the fixed answers.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# --- Q1: convergence order of three integrators on a damped oscillator ------------------------------------------------

_C = 0.1  # m = k = 1, c = 0.1, x(0) = 1, x'(0) = 0 (examples/integrator_bakeoff)
Q1_STEPS = (0.1, 0.05, 0.025)
Q1_T = 10.0


def _exact(t: float) -> float:
    zeta = _C / 2.0
    wd = math.sqrt(1.0 - zeta * zeta)
    return math.exp(-zeta * t) * (math.cos(wd * t) + zeta / wd * math.sin(wd * t))


def _accel(x: float, v: float) -> float:
    return -_C * v - x


def _euler(h: float, t: float) -> float:
    x, v = 1.0, 0.0
    for _ in range(int(round(t / h))):
        x, v = x + h * v, v + h * _accel(x, v)
    return x


def _verlet(h: float, t: float) -> float:
    # Velocity-Verlet; the damping term makes the closing half-step implicit in v, solved exactly (it is linear).
    x, v = 1.0, 0.0
    a = _accel(x, v)
    for _ in range(int(round(t / h))):
        x = x + h * v + 0.5 * h * h * a
        half = v + 0.5 * h * a
        v = (half - 0.5 * h * x) / (1.0 + 0.5 * h * _C)
        a = _accel(x, v)
    return x


def _rk4(h: float, t: float) -> float:
    x, v = 1.0, 0.0
    for _ in range(int(round(t / h))):
        k1 = (v, _accel(x, v))
        k2 = (v + h / 2 * k1[1], _accel(x + h / 2 * k1[0], v + h / 2 * k1[1]))
        k3 = (v + h / 2 * k2[1], _accel(x + h / 2 * k2[0], v + h / 2 * k2[1]))
        k4 = (v + h * k3[1], _accel(x + h * k3[0], v + h * k3[1]))
        x += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return x


def q1_orders() -> dict[str, float]:
    """The slope of log(|x(10) - exact|) against log(h) over h in 0.1, 0.05, 0.025, per method."""
    out = {}
    for name, fn in (("euler", _euler), ("verlet", _verlet), ("rk4", _rk4)):
        errors = [abs(fn(h, Q1_T) - _exact(Q1_T)) for h in Q1_STEPS]
        out[name] = float(np.polyfit(np.log(Q1_STEPS), np.log(errors), 1)[0])
    return out


# --- Q4: the stochastic SIR epidemic's final size --------------------------------------------------------------------

def final_size_z(r0: float) -> float:
    """The root of z = 1 - exp(-R0 z) in (0, 1] (0 when R0 <= 1), by bisection."""
    if r0 <= 1.0:
        return 0.0
    lo, hi = 1e-12, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if mid - 1.0 + math.exp(-r0 * mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def q4_sir(n: int = 1000, r0: float = 2.0, runs: int = 20000, seed: int = 12345, cutoff: int = 10) -> dict[str, float]:
    """The Markov SIR epidemic of the task (one starting case; recovery at rate 1, infection at rate R0 S I / N) run
    ``runs`` times through its jump chain: the mean final fraction ever infected over all outbreaks, and the share of
    outbreaks in which fewer than ``cutoff`` people were ever infected."""
    rng = np.random.default_rng(seed)
    s = np.full(runs, n - 1, dtype=np.int64)
    i = np.ones(runs, dtype=np.int64)
    ever = np.ones(runs, dtype=np.int64)
    while True:
        alive = i > 0
        if not alive.any():
            break
        rate_inf = r0 * s * i / n
        with np.errstate(divide="ignore", invalid="ignore"):  # a finished outbreak has no event left: never drawn
            p_inf = np.where(alive, rate_inf / (rate_inf + i), 0.0)
        infect = alive & (rng.random(runs) < p_inf)
        recover = alive & ~infect
        s = s - infect
        i = i + infect - recover
        ever = ever + infect
    return {"mean_final_fraction": float(np.mean(ever / n)), "early_die_out_share": float(np.mean(ever < cutoff)),
            "theory_mean_final_fraction": (1.0 - 1.0 / r0) * final_size_z(r0) if r0 > 1 else 0.0,
            "theory_early_die_out_share": min(1.0, 1.0 / r0)}


# --- Q5: Benjamini-Hochberg's false discovery rate --------------------------------------------------------------------

def q5_bh_fdr(pi1: float, alpha: float = 0.05) -> float:
    """For independent, continuous p-values Benjamini-Hochberg's false discovery rate is exactly pi0 * alpha
    (Benjamini & Hochberg 1995; the equality for independent tests, Benjamini & Yekutieli 2001)."""
    return (1.0 - pi1) * alpha


def compute(task: str) -> dict[str, Any]:
    """The reference values for one task id (what ``fi tools bench reference`` prints)."""
    task = task.upper()
    if task == "Q1":
        return q1_orders()
    if task == "Q4":
        return q4_sir()
    if task == "Q5":
        return {"fdr_bh at pi1=0.2": q5_bh_fdr(0.2), "fdr_bh at pi1=0.05": q5_bh_fdr(0.05)}
    raise KeyError(f"no reference computation for {task}: its answers are closed forms or values the task sets")
