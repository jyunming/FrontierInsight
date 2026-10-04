"""Before the known-answer checks stop a quest, FI finds out for itself whether the check or the script is wrong.

The oracle gate (``Engine._oracle_gate``) judges each known-answer check and repairs the script a bounded number of
times. Many real stops were the check's own fault: an expected value worked out wrongly (an RK4 error expected at
1.637e-08 where the true one is 3.33e-07), a tolerance tighter than the method's own error at its step (Verlet's energy
expected to be conserved to 1e-12 where the scheme's error at h=0.01 is h**2/8 = 1.25e-05), a single random trial judged
against a tolerance smaller than its own noise. A repair spent on a correct script rewrites it towards the wrong number.
So before the first repair of a failing check FI looks itself, each look bounded and recorded under
``attempts[].triage`` in ``needs/ORACLE_CHECK.json``:

- **the expected value recomputed** by another model from the check's statement, case and reference alone, never
  shown the measured value (:func:`recompute_verdict`): one call per check, kept in ``.fi/oracle_triage.json`` so a
  resume never asks again. A recomputed value that disagrees with the plan's and lies near the measurement marks the
  expected value *disputed*: the script is not rewritten for it, and the card offers the recomputed value as a
  proposal a person accepts or not;
- **half the step** for a deterministic check whose case has one (``dt``, ``h``, ``dx``, ``n_steps`` ...): the case
  again at half the step (and a quarter when the check declares no ``order``), and the value extrapolated to step 0
  (Richardson, :func:`step_verdict`). When that limit agrees with the expected value, the gap is the method's own
  error at the check's step, and the tolerance is tighter than it;
- **more seeds** for a random check, which the gate runs as one trial: three more trials (:func:`seeds_verdict`).
  When their spread is larger than the tolerance and the gap is within it, the tolerance is smaller than one trial's
  noise;
- (the gate itself) the run's last exception carried into the repair request and the card, an early stop when the
  same exception comes back after a repair, and one retry at twice the time limit when a run of the checks runs out
  of time.

Nothing here changes a verdict: a failing check still fails, and the evidence level still says so. What changes is
whether the script is rewritten for it (not when FI's own look points to the check) and what the card says.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import oracle_check as _oracle
from . import oracle_forms as _forms
from .oracle_card import last_error

#: FI's record of its own looks at the checks (under the quest's ``.fi``): a recomputed expected value per check, so a
#: resume never asks the model again.
RECORD = "oracle_triage.json"
#: How many more trials a random check gets.
EXTRA_SEEDS = 3
#: Case keys that count steps rather than size one: doubling them halves the step.
COUNT_KEYS = ("n_steps", "nsteps", "num_steps", "steps", "n_step", "nt", "n_t")
#: The prompt's node: ``provider.node_models.oracle_review`` answers it (a dot name falls back to its first part).
NODE = "oracle_review.recompute"


def num(value: Any) -> float | None:
    """A finite number, or ``None`` (a bool, text, nan, inf)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


_num = num


def _fmt(value: float | None) -> str:
    return _oracle.fmt_digits(value) if value is not None else "?"


def fingerprint(oracle: dict[str, Any]) -> str:
    """What a check's verdict depends on (its expected value, tolerance, case, measure and statement), hashed: a dispute
    or a recomputed value about one version of a check never carries over to another."""
    keep = {k: oracle.get(k) for k in ("expected", "tolerance", "tolerance_mode", "case", "measure", "check")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def recompute_key(oracle: dict[str, Any]) -> str:
    """What a recomputed expected value depends on: the check's version and everything the prompt shows (its kind and
    reference too)."""
    keep = {"check": fingerprint(oracle), "kind": oracle.get("kind"), "reference": oracle.get("reference")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def read(fi_dir: Path) -> dict[str, Any]:
    try:
        data = json.loads((Path(fi_dir) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write(fi_dir: Path, data: dict[str, Any]) -> None:
    try:
        Path(fi_dir).mkdir(parents=True, exist_ok=True)
        (Path(fi_dir) / RECORD).write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    except OSError:
        pass  # a record that cannot be written never stops a quest; the next run asks again


def value_of(values: dict[str, Any] | None, measure: str) -> float | None:
    """The number a check reads from what the simulation returned: one returned name, or the check's formula of them."""
    if not isinstance(values, dict):
        return None
    if measure in values:
        return _num(values[measure])
    return _forms.evaluate(measure, values).value


# --- the expected value, recomputed ------------------------------------------------------------------------------------


def recompute_prompt(template: Any, *, topic: str, oracle: dict[str, Any]) -> str:
    """The request to work the expected value out again: the check's statement, kind, case, how its number is computed
    and its reference. Never the measured value, and never the plan's expected value as a number to agree with."""
    case = oracle.get("case")
    return template.substitute(
        topic=" ".join(str(topic or "").split())[:1500],
        name=str(oracle.get("name") or "").strip(),
        kind=_oracle.kind_of(oracle) or _oracle.kind_written(oracle) or "(not given)",
        check=" ".join(str(oracle.get("check") or "").split())[:1200] or "(not given)",
        case=json.dumps(case, default=str) if isinstance(case, dict) and case else "(not given)",
        measure=str(oracle.get("measure") or "").strip() or "(not given)",
        reference=" ".join(str(_oracle.reference_of(oracle) or "").split())[:2000] or "(not given)",
    )


def parse_recompute(parsed: Any) -> tuple[float | None, str]:
    """``(the recomputed expected value, how it was worked out)`` from the model's JSON; ``(None, "")`` for anything
    else (no number, a reply that is not JSON): never guessed at."""
    if not isinstance(parsed, dict):
        return None, ""
    value = parsed.get("expected")
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            value = None
    return _num(value), " ".join(str(parsed.get("how") or "").split())[:600]


def recompute_verdict(oracle: dict[str, Any], recomputed: float | None, value: float | None) -> str:
    """What a recomputed expected value says: ``agrees`` (within the check's tolerance of the plan's: the expected
    value holds up, so the simulation is the likely cause), ``disputed`` (it disagrees with the plan's and lies near the
    measurement: within the tolerance of it, or within 1.5 times of it where the plan's is an order of magnitude or
    more away, or of the other sign; never for the worst violation of a rule, which expects 0 by its kind's form),
    ``differs`` (it disagrees with both), or ``""`` (nothing to judge)."""
    expected, limit, _mode = _oracle.limit_of(oracle)
    if recomputed is None or expected is None or limit is None:
        return ""
    if abs(recomputed - expected) <= limit:
        return "agrees"
    if value is None:
        return "differs"
    if expected == 0 and _oracle.kind_of(oracle) in _forms.VIOLATION_KINDS:
        # The worst violation of a rule expects 0 by the kind's form: a non-zero "expected" violation is the method's
        # error at its step, which the run at a smaller step judges (with numbers), not a model's estimate.
        return "differs"
    if abs(recomputed - value) <= limit:
        return "disputed"
    near = (recomputed != 0 and value != 0 and (recomputed > 0) == (value > 0)
            and abs(math.log10(abs(recomputed / value))) < math.log10(1.5))
    plan_off = (expected == 0 or (expected > 0) != (value > 0) or value == 0
                or abs(math.log10(abs(expected / value))) >= 1)
    return "disputed" if near and plan_off else "differs"


def recompute_entry(oracle: dict[str, Any], verdict: str, recomputed: float | None, value: float | None, how: str, *,
                    model: str, same_model: bool) -> dict[str, Any]:
    """The record of one recomputation, with the sentences the card shows."""
    name = str(oracle.get("name") or "").strip()
    expected, _limit, _mode = _oracle.limit_of(oracle)
    who = (f"the model that wrote the plan ({model}; no other model is named for `oracle_review`)" if same_model
           else f"another model ({model})")
    if recomputed is None:
        tried = (f"FI asked {who} to work out the expected value of `{name}` again from the check's statement, case and "
                 "reference, without the measured value; it gave no usable number.")
        return {"check": name, "kind": "recompute", "verdict": "", "points_to": "", "model": model,
                "same_model": same_model, "tried": tried, "cause": None}
    tried = (f"FI asked {who} to work out the expected value of `{name}` again from the check's statement, case and "
             f"reference, without showing it the measured value: it got {_fmt(recomputed)} (the plan says "
             f"{_fmt(expected)}; measured {_fmt(value)}).")
    cause: dict[str, str] | None = None
    points = ""
    slip = arithmetic_slip(how, recomputed)
    if slip is not None:
        # Its own working does not give the number it reports: the recheck is no evidence either way.
        said = (f"{slip['expression']} comes to {_fmt(slip['computes'])}, not the {_fmt(slip['written'])} it "
                "writes")
        return {"check": name, "kind": "recompute", "verdict": verdict, "points_to": "", "recomputed": recomputed,
                "expected": expected, "measured": value, "how": how, "model": model, "same_model": same_model,
                "how_slip": slip, "tried": tried + f" Its own arithmetic does not add up ({said}), so it is not used.",
                "cause": None}
    if same_model and verdict in ("agrees", "disputed"):
        # The model that wrote the plan, asked again, is not an independent check of it: said, never counted.
        return {"check": name, "kind": "recompute", "verdict": verdict, "points_to": "", "recomputed": recomputed,
                "expected": expected, "measured": value, "how": how, "model": model, "same_model": same_model,
                "tried": tried + " It is the same model that wrote the plan, so this is not an independent check.",
                "cause": None}
    if verdict == "agrees":
        points = "script"
        cause = {"text": f"The expected value of `{name}` holds up, so the simulation is the likely cause: worked out "
                         "again without the measured value, it comes out as the plan says.",
                 "evidence": f"Recomputed {_fmt(recomputed)}, the plan's {_fmt(expected)}"
                             + (f"; its working: {how}" if how else "") + "."}
    elif verdict == "disputed":
        points = "check"  # the card shows it through the proposal (the cause is said there)
    elif verdict == "differs":
        cause = {"text": f"The expected value of `{name}` is uncertain: worked out again without the measured value, it "
                         "matches neither the plan's value nor the measurement.",
                 "evidence": f"Recomputed {_fmt(recomputed)}, the plan's {_fmt(expected)}, measured {_fmt(value)}"
                             + (f"; its working: {how}" if how else "") + "."}
    return {"check": name, "kind": "recompute", "verdict": verdict, "points_to": points, "recomputed": recomputed,
            "expected": expected, "measured": value, "how": how, "model": model, "same_model": same_model,
            "tried": tried, "cause": cause}


def recompute_proposal(oracle: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """A disputed expected value as a proposal, in the shape :func:`core.oracle_check.proposals` gives (the card's
    one-step "accept" reads it): the recomputed value with the check's own tolerance. Never the measured value."""
    tolerance = _num(oracle.get("tolerance")) or 0.0
    mode = "relative" if str(oracle.get("tolerance_mode") or "").strip().lower() == "relative" else "absolute"
    who = ("the model that wrote the plan" if entry.get("same_model") else "another model") + f" ({entry.get('model')})"
    reason = (f"FI asked {who} to work the expected value out again from the check's statement, case and reference, "
              f"without showing it the measured value, and it got {_fmt(entry.get('recomputed'))}"
              + (f": {entry['how']}" if entry.get("how") else "") + ".")
    return {"name": str(oracle.get("name") or "").strip(), "expected": entry["recomputed"], "tolerance": tolerance,
            "tolerance_mode": mode, "reason": reason[:600], "source": "recompute",
            # Whether the model that wrote the plan answered: never counted as independent evidence when it did.
            "same_model": bool(entry.get("same_model"))}


# --- half the step -----------------------------------------------------------------------------------------------------


def step_of(case: Any) -> tuple[str, float, bool] | None:
    """``(key, value, counts_steps)`` of the step a check's case sets: a step size (``dt``, ``h``, ``dx`` ...) or a
    count of steps (``n_steps`` ...); ``None`` when it sets neither."""
    if not isinstance(case, dict):
        return None
    sizes = [(k, _num(case.get(k))) for k in _oracle._STEP_KEYS]  # noqa: SLF001 -- the one list of step names
    sizes = [(k, v) for k, v in sizes if v is not None and v > 0]
    counts = [(k, _num(case.get(k))) for k in COUNT_KEYS]
    counts = [(k, v) for k, v in counts if v is not None and v >= 1 and float(v).is_integer()]
    grid = any(_num(case.get(k)) is not None for k in ("nx", "n_x", "ny", "n_y", "n_points", "npoints", "n_grid"))
    if len(sizes) + len(counts) != 1 or (sizes and grid):
        # Both a step and a count of steps or points (halving one changes the span the other covers), or neither: not
        # run.
        return None
    if sizes:
        return sizes[0][0], sizes[0][1], False
    return counts[0][0], counts[0][1], True


def refined(case: dict[str, Any], key: str, counts: bool, times: int) -> dict[str, Any]:
    """``case`` with its step halved ``times`` times (a count of steps doubled instead)."""
    v = float(case[key])
    out = dict(case)
    out[key] = int(round(v * 2 ** times)) if counts else v / 2 ** times
    return out


#: How far the order seen at three steps may be from the order a check declares.
ORDER_SLACK = 0.5
#: The lowest order seen at three steps that counts as a higher-order scheme when the check declares none: a
#: first-order scheme where a higher one was meant (Euler for RK4 or Verlet) is the classic bug, so a first-order
#: convergence is said, never set aside.
MIN_ORDER = 1.5
#: How close the value extrapolated to step 0 must come to the expected one, relative to it (or, for an expected 0,
#: to the gap at the check's step), beyond the tolerance itself.
LIMIT_SLACK = 1e-3


def step_verdict(oracle: dict[str, Any], values: list[float], *, order: float | None) -> dict[str, Any] | None:
    """What the check's value at its step, half of it and a quarter of it says (``values`` in that order). The three
    give the observed order; the value at step 0 is extrapolated from the last two (Richardson) with the declared
    ``order`` when there is one, else with the observed one. ``verdict``:

    - ``method_error``: the values shrink like a scheme of the declared order (or, with none declared, of order 1.5 or
      more on a rule's worst violation expecting 0) and the step-0 value agrees with the expected one (within the
      tolerance, or within the smaller of a tenth of the last step's change and a thousandth of the expected value, or
      of the gap when 0 is expected) while the value at the check's step does not: the tolerance is tighter than the
      method's own error there;
    - ``wrong_order``: they shrink at another order than the declared one; ``low_order``: at about first order with
      none declared (a first-order step where a higher one is meant is a classic bug, so this is said, not set aside);
      ``converges_unconfirmed``: they converge to the expected value with no order declared, on a check other than a
      rule's worst violation expecting 0 (which order the scheme should have is not known: said, not set aside);
    - ``converges_elsewhere``: they settle on another value; ``steady_elsewhere``: they hardly move at all (within the
      check's tolerance) and stay far from the expected value; ``not_converging``: they do not shrink like an error.

    ``None`` when the check passes or nothing can be said (fewer than three values)."""
    expected, limit, _mode = _oracle.limit_of(oracle)
    if expected is None or limit is None or len(values) < 3 or any(v is None for v in values):
        return None
    v0 = values[0]
    gap = abs(v0 - expected)
    if gap <= limit:
        return None
    out: dict[str, Any] = {"values": list(values), "gap": gap}
    spread = max(values) - min(values)
    if spread <= max(limit, 1e-9 * abs(v0)) and gap > 10 * spread:
        # The value hardly moves as the step shrinks and stays far from the expected one: whatever is wrong, it is not
        # the step (said as "stays the same", never as "does not settle").
        return {**out, "spread": spread, "verdict": "steady_elsewhere"}
    d1, d2 = v0 - values[1], values[1] - values[2]
    if d1 == 0 or d2 == 0 or (d1 > 0) != (d2 > 0):
        return {**out, "verdict": "not_converging"}
    seen = math.log2(abs(d1 / d2))
    out["order"], out["order_from"] = seen, "observed"
    if not 0.5 <= seen <= 10:
        return {**out, "verdict": "not_converging"}
    declared = order is not None and order > 0
    if declared:
        out["declared_order"] = float(order)  # type: ignore[arg-type]
    p = float(order) if declared else seen  # type: ignore[arg-type]
    limit_value = values[2] + (values[2] - values[1]) / (2 ** p - 1)
    out["extrapolated"] = limit_value
    out["method_error"] = abs(v0 - limit_value)
    if declared and abs(seen - p) > ORDER_SLACK:
        return {**out, "verdict": "wrong_order"}
    scale = abs(expected) if expected != 0 else gap
    # The step-0 value must agree within the tolerance, or within what the extrapolation itself can resolve (a tenth of
    # the last step's change) when that is finer than a thousandth of the scale.
    if abs(limit_value - expected) > max(limit, min(0.1 * abs(values[1] - values[2]), LIMIT_SLACK * scale)):
        return {**out, "verdict": "converges_elsewhere"}
    if not declared and seen < MIN_ORDER:
        return {**out, "verdict": "low_order"}
    if not declared and not (expected == 0 and _oracle.kind_of(oracle) in _forms.VIOLATION_KINDS):
        # Which order the scheme should have is not known (RK2 where RK4 is meant converges too): said, not set aside.
        # A rule that holds exactly in the limit (a conserved quantity's worst violation, expecting 0) is the exception.
        return {**out, "verdict": "converges_unconfirmed"}
    return {**out, "verdict": "method_error"}


def step_entry(oracle: dict[str, Any], key: str, steps: list[float], result: dict[str, Any]) -> dict[str, Any]:
    name = str(oracle.get("name") or "").strip()
    expected, limit, _mode = _oracle.limit_of(oracle)
    values = result.get("values") or []
    runs = "; ".join(f"{key}={_fmt(s)}: {_fmt(v)}" for s, v in zip(steps, values))
    order = result.get("order")
    order_text = (f"observed order about {order:.2g}" if isinstance(order, (int, float)) else "") + (
        f" (the check declares {_fmt(result['declared_order'])})" if result.get("declared_order") is not None else "")
    tried = f"FI ran the case of `{name}` again at a smaller step ({runs})."
    cause: dict[str, str] | None = None
    points = ""
    verdict = result.get("verdict")
    if verdict == "method_error":
        points = "tolerance"
        cause = {"text": f"The tolerance of `{name}` is tighter than the method's own error at this step: the value moves "
                         "towards the expected one as the step shrinks, as a correct scheme's error does.",
                 "evidence": f"{runs}; {order_text}; extrapolated to step 0 it is {_fmt(result.get('extrapolated'))}, "
                             f"which agrees with the expected {_fmt(expected)}. The method's own error at "
                             f"{key}={_fmt(steps[0])} is about {_fmt(result.get('method_error'))}, the tolerance "
                             f"{_fmt(limit)}."}
    elif verdict == "converges_elsewhere":
        points = "script"
        cause = {"text": f"At a smaller step `{name}` settles on another value than the expected one, so the gap is "
                         "not the step's error: the simulation (or the expected value) is wrong.",
                 "evidence": f"{runs}; {order_text}; extrapolated to step 0 it is {_fmt(result.get('extrapolated'))}, "
                             f"the expected value {_fmt(expected)}."}
    elif verdict == "wrong_order":
        points = "script"
        cause = {"text": f"At a smaller step `{name}` shrinks at another order than the check declares: the scheme the "
                         "simulation runs is not the one the check is about.",
                 "evidence": f"{runs}; {order_text}."}
    elif verdict == "low_order":
        cause = {"text": f"At a smaller step `{name}` moves towards the expected value, but only at about first order: if "
                         "the method should be of higher order, the simulation is likely wrong; if first order is "
                         "right, the tolerance is tighter than its error at this step.",
                 "evidence": f"{runs}; {order_text}; extrapolated to step 0 it is {_fmt(result.get('extrapolated'))}, "
                             f"the expected value {_fmt(expected)}."}
    elif verdict == "converges_unconfirmed":
        cause = {"text": f"At a smaller step `{name}` moves towards the expected value: if the scheme's order is the one "
                         "it shows, the tolerance is tighter than its error at this step; declare the check's `order` so "
                         "FI can tell.",
                 "evidence": f"{runs}; {order_text}; extrapolated to step 0 it is {_fmt(result.get('extrapolated'))}, "
                             f"the expected value {_fmt(expected)}."}
    elif verdict == "steady_elsewhere":
        points = "script"
        cause = {"text": f"At a smaller step `{name}` stays at the same value, so the gap is not the step's error: the "
                         "simulation (or the expected value) is wrong.",
                 "evidence": f"{runs}; the expected value {_fmt(expected)}."}
    elif verdict == "not_converging":
        points = "script"
        cause = {"text": f"At a smaller step `{name}` does not shrink like a method's error, so the gap is not the "
                         "step's error.",
                 "evidence": runs + "."}
    return {"check": name, "kind": "half_step", "verdict": verdict or "", "points_to": points, "key": key,
            "steps": list(steps), **{k: v for k, v in result.items() if k != "verdict"}, "tried": tried,
            "cause": cause}


# --- more seeds --------------------------------------------------------------------------------------------------------


def seeds_verdict(oracle: dict[str, Any], values: list[float]) -> dict[str, Any] | None:
    """What trial 0 and the extra trials of a random check say: ``noise`` when their spread (standard deviation) is
    larger than the tolerance, trial 0's gap is within three of it and their mean within three standard errors of the
    expected value (one trial cannot meet the tolerance, right or wrong); ``beyond_noise`` otherwise; ``varies`` for a
    rule every trial must keep exactly (an invariant, a worst violation expecting 0) whose trials differ, which is the
    fault itself; ``None`` when nothing can be said."""
    expected, limit, _mode = _oracle.limit_of(oracle)
    values = [v for v in values if v is not None]
    if expected is None or limit is None or len(values) < 3:
        return None
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    gap = abs(values[0] - expected)
    if gap <= limit:
        return None
    out = {"values": list(values), "mean": mean, "sd": sd, "gap": gap}
    if expected == 0 or _oracle.kind_of(oracle) in _forms.VIOLATION_KINDS:
        # A rule every trial must keep (an invariant, a symmetry, a worst violation expecting 0): one trial shows it
        # exactly, so trials that differ are the fault, not noise.
        return {**out, "verdict": "varies" if sd > limit else "beyond_noise"}
    if sd > limit and gap <= 3 * sd and abs(mean - expected) <= 3 * sd / math.sqrt(len(values)):
        return {**out, "verdict": "noise"}
    return {**out, "verdict": "beyond_noise"}


def seeds_entry(oracle: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    name = str(oracle.get("name") or "").strip()
    expected, limit, _mode = _oracle.limit_of(oracle)
    values = result.get("values") or []
    shown = ", ".join(_fmt(v) for v in values)
    tried = (f"FI ran the case of `{name}` with {len(values) - 1} more seeds: {shown} (mean {_fmt(result.get('mean'))}, "
             f"spread {_fmt(result.get('sd'))}).")
    cause: dict[str, str] | None = None
    points = ""
    if result.get("verdict") == "noise":
        points = "tolerance"
        cause = {"text": f"The tolerance of `{name}` is smaller than one trial's noise: the check judges a single random "
                         "trial, and trials of the same case differ by more than the tolerance.",
                 "evidence": f"{len(values)} trials gave {shown}: spread {_fmt(result.get('sd'))} against a tolerance of "
                             f"{_fmt(limit)}, and the first trial's gap ({_fmt(result.get('gap'))}) is within that noise "
                             f"of the expected {_fmt(expected)}."}
    elif result.get("verdict") == "varies":
        points = "script"
        cause = {"text": f"`{name}` is a rule every trial must keep exactly, and it differs from trial to trial: the "
                         "simulation breaks it by a random amount.",
                 "evidence": f"{len(values)} trials gave {shown}; the expected value is {_fmt(expected)}."}
    elif result.get("verdict") == "beyond_noise":
        points = "script"
        cause = {"text": f"The gap of `{name}` is larger than the trials' own noise, so it is not chance.",
                 "evidence": f"{len(values)} trials gave {shown} (spread {_fmt(result.get('sd'))}); the expected value "
                             f"is {_fmt(expected)}."}
    return {"check": name, "kind": "seeds", "verdict": str(result.get("verdict") or ""), "points_to": points,
            **{k: v for k, v in result.items() if k != "verdict"}, "tried": tried, "cause": cause}


# --- the arithmetic a derivation writes out ------------------------------------------------------------------------------
#
# A derivation often ends in numbers: "T = T0 * (2/pi) * K(sin(45deg)) = 2*pi*sqrt(1/9.81) * (2/pi) * 1.85407 => 1.18034".
# When the numbers it multiplies out do not give the answer it writes (2.368, not 1.180), the expected value is a slip of
# the plan, whatever any model says about it (a real quest: the same model, asked again, repeated the same slip). FI works
# each written-out step out itself, with the small calculator the checks' formulas already use
# (core/oracle_forms.py::evaluate: numbers, + - * / ** and a short list of named functions, walked, never run as code).

import re as _re

_DEGREES = _re.compile(r"(\d+(?:\.\d+)?)\s*(?:°|deg\b|degrees?\b)")


def calculate(text: str) -> float | None:
    """The value of a written-out calculation (``2*pi*sqrt(1/9.81) * (2/pi) * 1.85407``), or ``None`` when ``text`` is
    not plain arithmetic: a symbol (``T0``), a function not on the calculator's list, anything else."""
    text = str(text or "").strip().rstrip(".,;").strip()
    if not text or len(text) > 300 or not _re.search(r"\d", text):
        return None
    text = _DEGREES.sub(lambda m: f"({m.group(1)}*pi/180)", text)
    text = text.replace("^", "**").replace("×", "*").replace("·", "*").replace("−", "-").replace("π", "pi")
    return _forms.evaluate(text, {}, special=True).value


def _digits(number: str) -> int:
    """The significant digits a written number states: ``1.85407`` 6, ``0.0595`` 3, ``2.000`` 4, ``100`` 1 (the
    trailing zeros of a whole number say nothing), ``3.3e-7`` 2."""
    mantissa = _re.sub(r"[eE][-+]?\d+$", "", number.strip().lstrip("+-"))
    if "." not in mantissa:
        mantissa = mantissa.rstrip("0") or "0"
    return len(_re.sub(r"[^0-9]", "", mantissa).lstrip("0")) or 1


#: After an "approximately" sign the comparison is never tighter than this (a truncated series written "≈ 1.0400" is the
#: author saying the series is close, not a slip); a real slip, "(2/pi)*ellipk(sin(pi/8)**2) approx 1.031" for 1.0400,
#: is 0.9% off.
APPROX_FLOOR = 5e-3


# A decimal number inside a step (``1.6336``, ``12.3``, ``3.3e-7``); a whole number (and ``pi``) is exact.
_DECIMAL = _re.compile(r"(?<![\w.])(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|(?<![\w.])\d+[eE][-+]?\d+")
#: At most this many rounded constants are varied together (every corner); more are varied one at a time.
_CORNERS_UP_TO = 6


def _half_unit(number: str) -> float:
    """Half a unit in the last place a written number states: ``1.03`` 0.005, ``12.3`` 0.05, ``3.3e-7`` 5e-9."""
    mantissa, _, exponent = number.lower().partition("e")
    decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    return 0.5 * 10.0 ** (-decimals) * 10.0 ** (int(exponent) if exponent else 0)


def calculated_range(expression: str) -> tuple[float, float] | None:
    """The least and greatest value ``expression`` can take when each decimal constant it uses is anywhere within half
    a unit of its last written digit (a constant rounded to the digits shown), worked out with the same calculator;
    ``None`` when some such value cannot be computed (a division that can reach zero, say: then nothing can be said).
    A step with only whole numbers gives one value. Every corner is tried for up to six constants; beyond that each
    is varied alone and the spreads are added (a wider range: a slip is called less often, never more)."""
    import itertools

    text = str(expression or "")
    found = list(_DECIMAL.finditer(text))
    base = calculate(text)
    if base is None:
        return None
    if not found:
        return base, base

    def with_(shifts: tuple[int, ...]) -> float | None:
        out, last = [], 0
        for m, s in zip(found, shifts):
            out.append(text[last:m.start()])
            out.append(repr(float(m.group()) + s * _half_unit(m.group())))
            last = m.end()
        out.append(text[last:])
        return calculate("".join(out))

    if len(found) <= _CORNERS_UP_TO:
        values = [with_(c) for c in itertools.product((-1, 1), repeat=len(found))]
        if any(v is None for v in values):
            return None
        return min(values + [base]), max(values + [base])  # type: ignore[type-var]
    spread_lo = spread_hi = 0.0
    for i in range(len(found)):
        for s in (-1, 1):
            shifts = tuple(s if j == i else 0 for j in range(len(found)))
            v = with_(shifts)
            if v is None:
                return None
            spread_lo, spread_hi = max(spread_lo, base - v), max(spread_hi, v - base)
    return base - len(found) * spread_lo, base + len(found) * spread_hi


def _close(a: float, b: float, written: str, *, approximate: bool = False, limit: float | None = None) -> bool:
    """Whether ``a`` (worked out) and ``b`` (the number written) agree to what the written number's digits can say
    (five times looser after an "approximately" sign), so a last digit rounded differently is never a slip. Without
    ``limit`` never tighter than 1%. With ``limit`` (the check's own tolerance, as an absolute amount) the 1% floor
    gives way to it: a gap the check itself can tell apart is a slip (a real quest's "(2/pi)*ellipk(sin(pi/8)**2)
    approx 1.031" comes to 1.0400, 0.9% off, nine times its tolerance) -- never tighter than :data:`APPROX_FLOOR`
    after an "approximately" sign. Rounded constants inside the step are allowed for by :func:`calculated_range`."""
    relative = max(5 * 10.0 ** (-_digits(written)) * (5 if approximate else 1), APPROX_FLOOR if approximate else 0.0)
    if limit is None or limit <= 0:
        slack_abs = max(1e-2 * (5 if approximate else 1), relative) * max(abs(a), abs(b))
    else:
        slack_abs = max(relative * max(abs(a), abs(b)), limit)
    return abs(a - b) <= slack_abs or abs(a - b) < 1e-12


_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
# What may follow the number a step writes: nothing, or one unit word (letters, a degree sign, letters per letters).
# Never an operator, a bracket, another number or a percent sign (a percent rescales the number).
_UNIT = _re.compile(r"^(?:[A-Za-zµ°]{1,8}(?:/[A-Za-zµ°]{1,8})*)?$")


def _is_arithmetic(text: str) -> bool:
    """Whether ``text`` is a calculation (an operator or a function call), not a lone number like ``g = 9.81``."""
    import ast

    text = _DEGREES.sub(lambda m: f"({m.group(1)}*pi/180)", str(text or "").strip().rstrip(".,;"))
    text = text.replace("^", "**").replace("×", "*").replace("·", "*").replace("−", "-").replace("π", "pi")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return False
    return any(isinstance(n, (ast.BinOp, ast.Call)) for n in ast.walk(tree))


# A bare ``log(`` can mean base 10 or base e, and a trig function whose argument names no angle unit can mean degrees
# or radians: the calculator would pick one and could call a correct derivation a slip (``log(1000) = 3``,
# ``sin(30) = 0.5``, ``cos(2*30) = 0.5``). Such a step is never counted as a slip. ``log10(`` and an angle written with
# ``pi``, ``deg`` or a degree sign are unambiguous and are still worked out. (``ln(`` is not on the calculator's list,
# so a step that uses it is never worked out at all.)
_AMBIGUOUS_LOG = _re.compile(r"(?<![\w.])log\s*\(")
_TRIG = _re.compile(r"(?<![\w.])(?:sin|cos|tan|asin|acos|atan|atan2)\s*\(")


def _call_argument(text: str, start: int) -> str:
    """The text inside the brackets of the call whose ``(`` is at ``start - 1``."""
    depth, i = 1, start
    while i < len(text) and depth:
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        i += 1
    return text[start:i - 1]


def ambiguous(expression: str) -> bool:
    """Whether ``expression`` uses a bare ``log(`` or a trig function whose argument names no angle unit (no ``pi``,
    ``deg``, a degree sign or ``rad``)."""
    text = str(expression or "")
    if _AMBIGUOUS_LOG.search(text):
        return True
    for m in _TRIG.finditer(text):
        if not _re.search(r"pi|π|deg|°|rad", _call_argument(text, m.end()), _re.IGNORECASE):
            return True
    # ellipk takes the parameter m = k**2 (SciPy's convention); a step that passes the modulus k itself means the
    # other convention, so only an argument written as a square is worked out.
    for m in _ELLIPK.finditer(text):
        if not _re.search(r"(\*\*|\^)\s*2\s*$", _call_argument(text, m.end()).strip()):
            return True
    return False


_ELLIPK = _re.compile(r"(?<![\w.])ellipk\s*\(")


def arithmetic_slip(text: Any, expected: Any = None, limit: float | None = None) -> dict[str, Any] | None:
    """A written-out step of ``text`` whose arithmetic does not give the number written after it, as
    ``{"expression", "computes", "written"}``; ``None`` when there is none.

    Only a step that is a calculation (an operator or a function call, never a lone number: "g = 9.81 = 2.006" is a
    parameter and a result, not a slip) followed by a bare number (with at most a unit word) counts. With
    ``expected`` (the check's expected value), only a step that writes that number counts: the slip is in the working
    that produces the expected value, never in an intermediate step. ``limit``: the check's tolerance as an absolute
    amount (:func:`_close`)."""
    want = num(expected)
    pieces = _APPROX_SPLIT.split(str(text or ""))
    for i in range(0, len(pieces) - 2, 2):
        left, sign, right = pieces[i].strip(), pieces[i + 1].strip().lower(), pieces[i + 2].strip()
        found = _re.match(rf"^({_NUMBER})", right)
        if found is None:
            continue
        rest = right[found.end():].strip().strip(".,;").strip()
        if not _UNIT.match(rest.split(";")[0].split(",")[0].strip() if rest else ""):
            continue
        written = found.group(1)
        if want is not None and not _close(want, float(written), written):
            continue
        expression = left.split(":")[-1].strip()
        if not _is_arithmetic(expression) or ambiguous(expression):
            continue
        computes = calculate(expression)
        approximate = sign in _APPROXIMATE
        if computes is None or _close(computes, float(written), written, approximate=approximate, limit=limit):
            continue
        # The constants the step uses are rounded to the digits shown: the written number counts as a slip only
        # outside every value they allow (worked out at each end), never because a rounding was carried through.
        span = calculated_range(expression)
        if span is None or span[0] <= float(written) <= span[1] or any(
                _close(edge, float(written), written, approximate=approximate, limit=limit) for edge in span):
            continue
        return {"expression": expression, "computes": computes, "written": float(written)}
    return None


# The signs a derivation writes its result after: "=", "=>", and the approximate ones ("≈", "~=", "~", "approx",
# "approximately", "about"), after which the written number is compared more loosely (its own digits, five times over).
_APPROXIMATE = ("≈", "~=", "~", "approx", "approx.", "approximately", "about")
_APPROX_SPLIT = _re.compile(r"(=>|⇒|≈|~=|(?<![<>=!~\w])~(?!=)|\bapprox(?:imately|\.)?(?=\s|$)|\babout\b|(?<![<>=!~])=(?!=))",
                            _re.IGNORECASE)


def plan_slip(oracle: dict[str, Any]) -> dict[str, Any] | None:
    """The slip in the arithmetic of ``oracle``'s own derivation that produces its expected value (:func:`arithmetic_slip`
    tied to the check's ``expected``), or ``None``."""
    _expected, limit, _mode = _oracle.limit_of(oracle)
    return arithmetic_slip(_oracle.reference_of(oracle), oracle.get("expected"), limit)


def correctable(oracle: dict[str, Any]) -> bool:
    """Whether FI may correct ``oracle``'s expected value: only a check that compares a measured quantity with its
    expected value (a special case, a published value ...). Not a check whose number is a worst violation of a rule or
    a formula built from what the simulation returns: its expected value is 0 by its form, and what a slip changes
    lives inside its ``measure``, which FI never edits."""
    measure = str(oracle.get("measure") or "").strip()
    simple = not measure or _re.fullmatch(r"[A-Za-z_]\w*", measure) is not None
    return simple and _oracle.kind_of(oracle) not in _forms.VIOLATION_KINDS


def arithmetic_entry(oracle: dict[str, Any], slip: dict[str, Any], *, where: str = "plan") -> dict[str, Any]:
    """A triage entry for a slip in the plan's own working (``where`` ``plan``): it points to the check."""
    name = str(oracle.get("name") or "").strip()
    said = (f"the plan's own working for the expected value does not add up: {slip['expression']} comes to "
            f"{_fmt(slip['computes'])}, not the {_fmt(slip['written'])} it writes")
    return {"check": name, "kind": "arithmetic", "verdict": "slip", "points_to": "check", **slip, "where": where,
            "tried": f"FI worked out the arithmetic in the plan's derivation of `{name}` itself: {said}.",
            "cause": {"text": f"The expected value of `{name}` is a slip in the plan's arithmetic, not a fault of the "
                              "simulation.",
                      "evidence": said[0].upper() + said[1:] + "."}}


def arithmetic_proposal(oracle: dict[str, Any], slip: dict[str, Any]) -> dict[str, Any]:
    """The expected value the plan's own derivation gives, worked out by FI, as a proposal in the shape
    :func:`core.oracle_check.proposals` gives (the card offers it; a person's yes applies it through the plan). It comes
    from the plan's own working, never from the measured value; the check's tolerance is kept."""
    tolerance = _num(oracle.get("tolerance")) or 0.0
    mode = "relative" if str(oracle.get("tolerance_mode") or "").strip().lower() == "relative" else "absolute"
    reason = (f"FI worked out the plan's own derivation: {slip['expression']} comes to {_fmt(slip['computes'])}, not the "
              f"{_fmt(slip['written'])} the plan writes.")
    return {"name": str(oracle.get("name") or "").strip(), "expected": slip["computes"], "tolerance": tolerance,
            "tolerance_mode": mode, "reason": reason[:600], "source": "arithmetic"}


# --- a measured value that is a simple multiple of a value FI worked out itself -----------------------------------------
#
# A real quest measured 0.519923 for a period ratio whose value, worked out from the plan's own derivation, is 1.0400:
# half of it, the classic sign of a simulation that measures half a period. Said to the repair (where to look) and on
# the card; never a verdict, and only against a value FI worked out itself (never the plan's number alone).

_MULTIPLES: tuple[tuple[float, str, str], ...] = (
    (0.5, "half", "half of the quantity the check means (half a cycle, say, or half the range)"),
    (2.0, "twice", "twice the quantity the check means (two cycles, say, or something counted twice)"),
    (0.25, "a quarter of", "a quarter of the quantity the check means"),
    (4.0, "four times", "four times the quantity the check means"),
)


def multiple_of(value: Any, reference: Any, limit: float | None) -> tuple[float, str, str] | None:
    """``(factor, word, what)`` when ``value`` is ``factor`` times ``reference`` (half, twice, a quarter, four times),
    within ``limit`` (the check's tolerance, as an absolute amount, at the reference's scale); ``None`` otherwise, and
    when the value already agrees with the reference."""
    v, ref = num(value), num(reference)
    if v is None or ref is None or v == 0 or ref == 0 or limit is None:
        return None
    if abs(v - ref) <= limit:
        return None
    for factor, word, what in _MULTIPLES:
        if abs(v / factor - ref) <= limit:
            return factor, word, what
    return None


def multiple_entry(oracle: dict[str, Any], value: float, reference: float, found: tuple[float, str, str],
                   source: str) -> dict[str, Any]:
    """A triage entry for a measured value that is a simple multiple of a value FI worked out itself: it points to the
    script, and its ``hint`` goes into the repair request."""
    name = str(oracle.get("name") or "").strip()
    _factor, word, what = found
    said = (f"the measured {_fmt(value)} is {word} the {_fmt(reference)} FI worked out itself ({source}): the simulation "
            f"likely computes {what}, not the quantity the check means")
    return {"check": name, "kind": "multiple", "verdict": "factor", "points_to": "script", "factor": found[0],
            "measured": value, "reference": reference,
            "tried": f"FI compared what `{name}` measured with the value it worked out itself: {said}.",
            "hint": f"The check '{name}': {said}.",
            "cause": {"text": f"The simulation likely computes {what} for `{name}`.",
                      "evidence": said[0].upper() + said[1:] + "."}}


# --- the gate's own: an exception that comes back, a run out of time ---------------------------------------------------


def exception_of(stderr_tail: str, found: list[str]) -> str:
    """The last exception a run of the checks raised (``KeyError: 'FI_RAW_DIR'``), or ``""``."""
    return last_error(stderr_tail or "") or last_error("; ".join(found or []))


def same_exception(attempts: list[dict[str, Any]]) -> str:
    """The exception when the last run raised the same one as the run before it, and a repair changed the script in
    between (the repair did not reach the fault); ``""`` otherwise."""
    if len(attempts) < 2 or attempts[-2].get("repair") != "applied":
        return ""
    now, before = str(attempts[-1].get("exception") or ""), str(attempts[-2].get("exception") or "")
    return now if now and now == before else ""


def same_exception_entry(error: str, left: int) -> dict[str, Any]:
    return {"check": "", "kind": "same_exception", "verdict": "stopped", "points_to": "script", "error": error,
            "repairs_left": left,
            "tried": (f"The same error came back after FI's repair ({error}), so FI stopped repairing"
                      + (f" and kept the {left} repair(s) left" if left else "") + ": another rewrite of the same kind "
                      "would not reach it."),
            "cause": None}


def timeout_entry(seconds: int, again_timed_out: bool) -> dict[str, Any]:
    return {"check": "", "kind": "timeout_retry", "verdict": "timed_out" if again_timed_out else "finished",
            "points_to": "script" if again_timed_out else "", "seconds": seconds * 2,
            "tried": (f"The run of the checks ran out of time ({seconds} s); FI ran it once more with twice the time "
                      f"({seconds * 2} s), and " + ("it ran out of time again." if again_timed_out else "it finished.")),
            "cause": ({"text": "The checks' cases take too long to run: a case should take seconds.",
                       "evidence": f"It ran out of time at {seconds} s and again at {seconds * 2} s."}
                      if again_timed_out else None)}


def cost_line(model_calls: int, runs: int) -> str:
    """plan.md's line on what FI's own look at the failing checks cost this run."""
    return (f"Before stopping or repairing, FI looked at the failing known-answer checks itself (at most one model call "
            f"per check to work its expected value out again, two more runs of a deterministic check's case at smaller "
            f"steps, three more seeds of a random check, one retry at twice the time limit): this run spent "
            f"{model_calls} model call(s) and {runs} extra run(s) of the simulation on it.")
