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
            "tolerance_mode": mode, "reason": reason[:600], "source": "recompute"}


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
    - ``converges_elsewhere``: they settle on another value; ``not_converging``: they do not shrink like an error.

    ``None`` when the check passes or nothing can be said (fewer than three values)."""
    expected, limit, _mode = _oracle.limit_of(oracle)
    if expected is None or limit is None or len(values) < 3 or any(v is None for v in values):
        return None
    v0 = values[0]
    gap = abs(v0 - expected)
    if gap <= limit:
        return None
    out: dict[str, Any] = {"values": list(values), "gap": gap}
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
