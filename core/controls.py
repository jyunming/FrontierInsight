"""The study's own controls: settings of the grid where the plan knows the answer before the run, judged by FI after it.

A plan often names a control in words ("a positive control must give the known value", "a negative control must show
no effect"), and the words were judged by nobody: the analysis's model read the results, said the control failed and
asked for a redesign, and FI, which goes back to design only on a fact of its own record, wrote the paper. A control the
plan types in ``protocol.controls`` is a fact FI can judge:

* ``name``: what the control is, in the plan's own plain words (``positive control: the exactly solvable case``);
* ``setting``: which cells of ``protocol.grid`` it is, a mapping from each named setting to one of its values (a setting
  left out means every value of it);
* ``quantity``: the declared result (``protocol.metrics`` or ``protocol.derived``) it is read from;
* ``expected`` and ``tolerance`` (``tolerance_mode`` ``absolute``, the default, or ``relative``): what a working study
  gives there.

:func:`entry_fault` is the plan's check of one entry (a reason a person can act on, or ``None``); a plan with no
``controls`` is valid, and an entry that cannot be read is left out with that reason, never a stop. :func:`judge` reads
FI's own record after the full run, with no model, at every setting of the study's grid the control covers. Where the
trials' record (``raw/trials.json``) holds the quantity, a setting is judged on the trials alone by the mean check's own
rule (:func:`core.mc_error.statistic`, :func:`core.mc_error.mean_verdict`: three standard errors, the tolerance never
widened; a declared proportion on its binomial interval; fewer than :data:`core.mc_error.MIN_TRIALS` trials, or no
standard error, decide nothing), except that a study that does not repeat its trials is judged on its value, as a
deterministic check is; where the trials do not hold it, on the value FI's analysis gives (``data/results/derived.json``),
which has no interval. A control is ``failed`` when one of its settings is outside the tolerance, ``passed`` when every
setting it covers is inside it, ``not decided`` when a setting is not decided or has no value in FI's record (those
settings are named), when FI did not list every setting it covers (the study has more than FI lists), or when several
records of one setting differ in settings the plan does not vary and the plan's constants do not pick one, and ``not
judged`` when FI's record holds no value at any of them. A control is a check, never a
route: a failed one is listed in the quality report and named by the paper's closing sentence (core/quality_pointer.py),
and the study is written up as it ran.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

#: The protocol part.
PART = "controls"
#: FI's record of every judgement of the controls (one entry per pass of the cross-check step).
RECORD_PATH = Path(".fi") / "controls.json"
SCHEMA = "fi.controls/v1"
PASSED, FAILED, NOT_DECIDED, NOT_JUDGED = "passed", "failed", "not decided", "not judged"

#: The plan's rule for the part (the plan directive in core/engine_base.py carries it).
PLAN_FIELD = ('"controls": [{"name": "<what the control is, in plain words, e.g. positive control: the exactly solvable '
              'case>", "setting": {"<a setting of grid>": <one of its values>}, "quantity": "<the id of a declared '
              'result, in metrics or derived>", "expected": <the value a working study gives there>, "tolerance": '
              '<how far from it still counts as working>, "tolerance_mode": "<absolute (default) | relative>"}]')
PLAN_RULE = (
    "`controls` are the study's own controls: settings of `grid` where the answer is known before the run, so the full "
    "run itself shows whether the measurement works (a positive control gives a known value; a negative control gives "
    "the value of no effect, such as 0). Whenever the topic or your plan names a control, write each one here, not "
    "only in words: `setting` picks the cells of `grid` it is (each setting named at one of its values; a setting left "
    "out means every value of it), `quantity` is the declared result it is read from, and `expected` with `tolerance` "
    "is what a working study gives there. After the full run FI judges each one from its own record (on the mean of "
    "the trials within its sampling error, when the trials hold the result); a control that fails is reported in the "
    "quality report and the paper. This is not `variables.controls` (what the study holds fixed). Leave `controls` out when the "
    "study has no control.")


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _same_value(a: Any, b: Any) -> bool:
    """Whether a value of a setting written in two ways is the same value (``4096`` and ``"4096"``, ``1`` and ``1.0``)."""
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.0)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def _declared(protocol: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("metrics", "derived"):
        rows = protocol.get(key)
        for row in rows if isinstance(rows, list) else [rows] if isinstance(rows, dict) else []:
            if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"].strip():
                out.append(row["id"].strip())
    return out


def _grid(protocol: dict[str, Any]) -> dict[str, list[Any]]:
    """The settings the study's runs differ in, as the runner and the judge below list them: a measurement's
    ``protocol.grid``, a search's coarse scan (``protocol.optimisation.grid``; the plan's own ``protocol.grid`` is removed
    when the search block is taken in, so it is never the one to read there)."""
    from .contract import settings_between_runs

    grid = settings_between_runs(protocol)[0]
    if not isinstance(grid, dict):
        return {}
    return {str(k): (list(v) if isinstance(v, (list, tuple)) else [v]) for k, v in grid.items()}


def _scan_name(protocol: dict[str, Any]) -> str:
    """Where the settings a control may name are written, for a message."""
    return "`protocol.optimisation.grid`" if isinstance(protocol.get("optimisation"), dict) else "`protocol.grid`"


def left_out_note(index: int, item: Any, why: str) -> str:
    """The sentence the plan says for a control left out because it cannot be judged."""
    label = (str(item.get("name") or "").strip() if isinstance(item, dict) else "") or f"number {index}"
    return (f"the control {label!r} was left out of `protocol.controls` because it could not be judged "
            f"({why}); FI does not judge it")


def setting_fault(index: int, item: Any, protocol: Any) -> str | None:
    """Why the control's ``setting`` names something the study does not run (a setting its grid, or its search's scan, lacks,
    or a value of one it does not have), in words a person can act on; ``None`` when it names only what the study runs, and
    for an entry that is not a mapping with a name (the other checks say so). Nothing else about the entry is judged."""
    protocol = protocol if isinstance(protocol, dict) else {}
    if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
        return None
    head = f"`protocol.controls` entry {index} (`{item['name'].strip()}`)"
    grid = _grid(protocol)
    where = _scan_name(protocol)
    setting = item.get("setting")
    if not isinstance(setting, dict) or not setting:
        return (f"{head}: `setting` must map each setting of {where} that picks the control's cells to one of its "
                f"values ({', '.join(grid) or 'the grid is empty'})")
    for axis, value in setting.items():
        if str(axis) not in grid:
            return f"{head}: `{axis}` is not a setting of {where} ({', '.join(grid) or 'the grid is empty'})"
        if isinstance(value, (list, dict)) or not any(_same_value(value, v) for v in grid[str(axis)]):
            return (f"{head}: `{axis}` is {value!r}, which is not one of its values in {where} "
                    f"({', '.join(str(v) for v in grid[str(axis)])})")
    return None


def entry_fault(index: int, item: Any, protocol: Any) -> str | None:
    """Why the ``index``-th entry of ``protocol.controls`` cannot be judged, in words a person can act on; ``None`` when
    it can. Held to the grid and the declared results the protocol has; nothing is read from the control's name."""
    protocol = protocol if isinstance(protocol, dict) else {}
    head = f"`protocol.controls` entry {index}"
    if not isinstance(item, dict):
        return f"{head} must be a mapping with a `name`, a `setting`, a `quantity`, an `expected` value and a `tolerance`"
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        return f"{head} has no `name` (what the control is, in plain words)"
    head = f"{head} (`{name.strip()}`)"
    declared = _declared(protocol)
    quantity = item.get("quantity")
    if not isinstance(quantity, str) or quantity.strip() not in declared:
        return (f"{head}: `quantity` must be the id of a declared result ({', '.join(declared) or 'the plan declares none'})")
    why = setting_fault(index, item, protocol)
    if why:
        return why
    if not _number(item.get("expected")):
        return f"{head}: `expected` must be a number (the value a working study gives at the control)"
    if not _number(item.get("tolerance")) or float(item["tolerance"]) <= 0:
        return f"{head}: `tolerance` must be a number above 0 (how far from `expected` still counts as working)"
    mode = item.get("tolerance_mode", "absolute")
    if mode is None:
        mode = "absolute"
    if not isinstance(mode, str) or mode.strip().lower() not in ("absolute", "relative"):
        return f"{head}: `tolerance_mode` must be `absolute` or `relative`"
    if mode.strip().lower() == "relative" and float(item["expected"]) == 0:
        return f"{head}: an `expected` of 0 takes an absolute `tolerance`; a relative one around 0 cannot be judged"
    return None


def normalize(controls: Any, protocol: Any) -> tuple[list[dict[str, Any]] | None, int | None, str | None]:
    """``(controls, None, None)`` read as FI judges them (a single mapping is a list of one; ``tolerance_mode`` written
    out), or ``(None, index, why)`` for the first entry that cannot be (``index`` 0 when the part is not a list)."""
    if isinstance(controls, dict):
        controls = [controls]
    if not isinstance(controls, list):
        return None, 0, ("`protocol.controls` must be a list of controls, each with a `name`, a `setting`, a `quantity`, "
                         "an `expected` value and a `tolerance`")
    out: list[dict[str, Any]] = []
    for index, item in enumerate(controls, start=1):
        why = entry_fault(index, item, protocol)
        if why:
            return None, index, why
        mode = str(item.get("tolerance_mode") or "absolute").strip().lower()
        out.append({**item, "name": str(item["name"]).strip(), "quantity": str(item["quantity"]).strip(),
                    "tolerance_mode": mode})
    return out, None, None


def of_protocol(protocol: Any) -> list[dict[str, Any]]:
    """The controls of ``protocol`` FI can judge (an entry that cannot be is not judged; the plan already said why)."""
    if not isinstance(protocol, dict) or protocol.get(PART) in (None, [], {}):
        return []
    raw = protocol[PART] if isinstance(protocol[PART], list) else [protocol[PART]]
    kept = []
    for index, item in enumerate(raw, start=1):
        if entry_fault(index, item, protocol) is None:
            kept.append(normalize([item], protocol)[0][0])  # type: ignore[index]
    return kept


# ---- judging ------------------------------------------------------------------------------------------------------------


def _matches(selector: dict[str, Any], setting: dict[str, Any]) -> bool:
    return all(str(axis) in setting and _same_value(value, setting[str(axis)]) for axis, value in selector.items())


def _same_setting(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return len(a) == len(b) and _matches(a, b)


def _limit(control: dict[str, Any]) -> float:
    tol = float(control["tolerance"])
    return tol * abs(float(control["expected"])) if control.get("tolerance_mode") == "relative" else tol


def _key_words(setting: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in setting.items())


def _derived_values(derived: dict[str, Any], quantity: str) -> list[tuple[dict[str, str], float]]:
    """``[(setting, value)]`` for each setting FI's analysis gives ``quantity`` a finite value at."""
    from .rerun_check import setting_of

    rec = ((derived.get("derived") if isinstance(derived, dict) else None) or {}).get(quantity)
    if not isinstance(rec, dict) or rec.get("status") != "ok" or not isinstance(rec.get("by_setting"), dict):
        return []
    out = []
    for key, value in rec["by_setting"].items():
        setting = setting_of(str(key))
        if setting is not None and _number(value):
            out.append((setting, float(value)))
    return out


def _planned(protocol: Any, selector: dict[str, Any]) -> tuple[list[dict[str, Any]], bool, bool]:
    """``(settings, expanded, cut)``: the settings of the study's grid the control covers, as the trial runner lists them
    (sub-studies included); ``expanded`` is ``False`` when the protocol's grid does not name every setting the control
    picks, and ``cut`` is ``True`` when the study has more settings than FI lists
    (:data:`core.reference_values.MAX_CHECKED_SETTINGS`), so some the control covers may not be among them."""
    from . import reference_values as _refs

    axes = _refs.study_axes(protocol if isinstance(protocol, dict) else None)
    if not axes or any(str(a) not in axes for a in selector):
        return [], False, False
    cap = int(_refs.MAX_CHECKED_SETTINGS)
    listed = _refs.grid_settings(protocol, limit=cap + 1)
    return [s for s in listed[:cap] if _matches(selector, s)], True, len(listed) > cap


def _constants_agree(found: dict[str, Any], setting: dict[str, Any], held: dict[str, Any]) -> bool:
    """Whether every value ``found`` lists beyond ``setting`` is the plan's own constant (``protocol.constants``)."""
    return all(str(a) in held and _same_value(v, held[str(a)]) for a, v in found.items() if a not in setting)


def _one(found: list[tuple[dict[str, Any], Any]], setting: dict[str, Any], held: dict[str, Any],
         ) -> tuple[Any, str]:
    """``(payload, "")`` of the one record of ``setting`` among ``found`` (records whose settings match it, with any
    constants beside), ``(None, why)`` when several differ in settings the plan does not vary and the plan's constants
    do not pick exactly one of them (never the first one listed), ``(None, "")`` when there is none."""
    distinct: list[tuple[dict[str, Any], list[Any]]] = []
    for s, payload in found:
        same = next((d for d in distinct if _same_setting(d[0], s)), None)
        if same is None:
            distinct.append((s, [payload]))
        else:
            same[1].append(payload)
    if not distinct:
        return None, ""
    if len(distinct) > 1:
        agree = [d for d in distinct if _constants_agree(d[0], setting, held)]
        if len(agree) != 1:
            names = "; ".join(_key_words({a: v for a, v in d[0].items() if a not in setting}) for d in distinct[:3])
            return None, (f"FI's record holds {len(distinct)} records of this setting that differ in settings the plan "
                          f"does not vary ({names}), and the plan's constants do not pick one")
        distinct = agree
    return distinct[0][1], ""


def _judge_setting(control: dict[str, Any], values: list[float], derived_value: float | None,
                   protocol: Any, repeated: bool | None) -> dict[str, Any]:
    """The verdict at one setting. When the trials hold the quantity it is judged on them alone, by the mean check's own
    rule (:func:`core.mc_error.statistic` and :func:`core.mc_error.mean_verdict`: a declared proportion on its binomial
    interval; fewer than :data:`core.mc_error.MIN_TRIALS` trials, or no standard error, decide nothing), except that a
    study that does not repeat its trials (``repeated`` ``False``: the quest's own reading,
    :func:`core.mc_error.quest_repeats_trials`) is judged on its value, as a deterministic check is. Else it is judged on
    the value FI's analysis gives (no interval)."""
    from . import mc_error

    quantity, expected, limit = control["quantity"], float(control["expected"]), _limit(control)
    protocol = protocol if isinstance(protocol, dict) else None
    if values:
        stat = mc_error.statistic([{quantity: v} for v in values], quantity, expected=expected, limit=limit,
                                  proportions=mc_error.proportion_names(protocol), repeated=repeated)
        if stat is not None:
            n = int(stat.get("trials") or 0)
            row = {"value": stat["value"], "se": stat.get("se"), "trials": n, "on": "trials",
                   **({"successes": stat["successes"]} if stat.get("binary") else {})}
            if repeated is False and not stat.get("varies"):
                verdict, why = mc_error.decide(float(stat["value"]), expected, limit, 0.0), ""
                row["se"] = None
            else:
                verdict = mc_error.mean_verdict(stat, expected, limit)
                why = ("" if verdict else
                       f"{n} trials are fewer than the {mc_error.MIN_TRIALS} FI needs to judge a mean"
                       if n < mc_error.MIN_TRIALS else
                       "the trials' standard error is not known" if not _number(stat.get("se")) else
                       "the interval of the trials' share reaches both sides of the tolerance" if stat.get("binary") else
                       "the interval of the trials' mean reaches both sides of the tolerance")
            return {**row, "verdict": PASSED if verdict == "pass" else FAILED if verdict == "fail" else NOT_DECIDED,
                    "why": why}
    if derived_value is not None:
        inside = abs(derived_value - expected) <= limit
        return {"value": derived_value, "se": None, "trials": None, "on": "analysis",
                "verdict": PASSED if inside else FAILED, "why": ""}
    return {"value": None, "se": None, "trials": None, "on": "", "verdict": NOT_JUDGED,
            "why": f"FI's record holds no value of `{quantity}` at this setting"}


def _cell_values(cells: list[dict[str, Any]], setting: dict[str, Any], quantity: str,
                 held: dict[str, Any] | None = None) -> tuple[list[float], str]:
    """``(values, why)``: the trials' values of ``quantity`` in the trial cell of ``setting`` (a cell matching every
    setting it names; several cells that differ in other settings are told apart by the plan's constants, else
    ``why`` says so and no value is taken)."""
    found = [(c["cell"], [float(v) for v in (c.get("metrics") or {}).get(quantity) or [] if _number(v)])
             for c in cells if _matches(setting, c["cell"])]
    picked, why = _one([(s, v) for s, v in found if v], setting, held or {})
    return ([v for vals in picked for v in vals] if picked else []), why


def judge(controls: list[dict[str, Any]], derived: dict[str, Any] | None, cells: list[dict[str, Any]] | None,
          protocol: Any = None, *, repeated: bool | None = None) -> list[dict[str, Any]]:
    """Each control's verdict from FI's own record: ``derived`` (``data/results/derived.json``) and ``cells`` (the trials'
    record as :func:`core.reported_settings.read_record` gives it), at every setting of ``protocol``'s grid the control
    covers. ``repeated``: whether the study repeats its trials (the engine passes
    :func:`core.mc_error.quest_repeats_trials`; ``None`` reads the protocol's run counts alone). See the module docstring
    for the rules."""
    from . import mc_error

    if repeated is None:
        repeated = mc_error.repeats_trials(protocol if isinstance(protocol, dict) else None)
    from . import reference_values as _refs

    held = _refs.held_values(protocol if isinstance(protocol, dict) else None)
    derived = derived if isinstance(derived, dict) else {}
    cells = [c for c in cells or [] if isinstance(c, dict) and isinstance(c.get("cell"), dict)]
    out: list[dict[str, Any]] = []
    for control in controls:
        quantity, selector = control["quantity"], control["setting"]
        from_analysis = [(s, v) for s, v in _derived_values(derived, quantity) if _matches(selector, s)]
        planned, expanded, cut = _planned(protocol, selector)
        if not expanded:  # no grid to expand: the settings FI's record lists
            planned = [dict(c["cell"]) for c in cells if _matches(selector, c["cell"])]
            planned += [dict(s) for s, _ in from_analysis if not any(_same_setting(s, p) for p in planned)]
        settings: list[dict[str, Any]] = []
        for setting in planned:
            # FI's record names a setting by every value of its cell, the study's constants too; when several records
            # match, the plan's constants pick the one, never the order they are listed in
            values, why_trials = _cell_values(cells, setting, quantity, held)
            given, why_analysis = (None, "") if values or why_trials else _one(
                [(s, v) for s, v in from_analysis if _matches(setting, s)], setting, held)
            if why_trials or why_analysis:
                settings.append({"setting": dict(setting), "value": None, "se": None, "trials": None, "on": "",
                                 "verdict": NOT_DECIDED, "why": why_trials or why_analysis})
                continue
            settings.append({"setting": dict(setting),
                             **_judge_setting(control, values, given[0] if given else None, protocol, repeated)})
        verdicts = [s["verdict"] for s in settings]
        missing = [s["setting"] for s in settings if s["verdict"] == NOT_JUDGED]
        verdict = (FAILED if FAILED in verdicts else
                   NOT_DECIDED if cut and (not settings or len(missing) < len(settings)) else
                   NOT_JUDGED if not settings or len(missing) == len(settings) else
                   PASSED if all(v == PASSED for v in verdicts) else NOT_DECIDED)
        why = ""
        if verdict == NOT_DECIDED and cut:
            why = (f"the study has more settings than the {_refs.MAX_CHECKED_SETTINGS} FI lists, so FI did not list "
                   "every setting the control covers")
        elif verdict == NOT_JUDGED:
            why = (f"FI's record holds no value of `{quantity}` at {_key_words(selector)}" if settings
                   else f"FI's record holds no setting with {_key_words(selector)}")
        elif verdict == NOT_DECIDED and missing:
            apart = _differing(settings, selector) or list(missing[0])
            names = "; ".join(_key_words({a: m.get(a) for a in apart}) for m in missing[:3])
            more = f" and {len(missing) - 3} more" if len(missing) > 3 else ""
            why = f"FI's record holds no value of `{quantity}` at {len(missing)} of its settings ({names}{more})"
        elif verdict == NOT_DECIDED:
            why = next(s["why"] for s in settings if s["verdict"] == NOT_DECIDED)
        out.append({"name": control["name"], "quantity": quantity, "setting": dict(selector),
                    "expected": float(control["expected"]), "tolerance": float(control["tolerance"]),
                    "tolerance_mode": control.get("tolerance_mode", "absolute"), "limit": _limit(control),
                    "verdict": verdict, "why": why, "missing": missing, "settings": settings})
    return out


def not_judged(controls: list[dict[str, Any]], why: str) -> list[dict[str, Any]]:
    """Every control as not judged, for ``why`` (the run gave no results to judge them on)."""
    return [{"name": c["name"], "quantity": c["quantity"], "setting": dict(c["setting"]), "expected": float(c["expected"]),
             "tolerance": float(c["tolerance"]), "tolerance_mode": c.get("tolerance_mode", "absolute"),
             "limit": _limit(c), "verdict": NOT_JUDGED, "why": why, "missing": [], "settings": []} for c in controls]


def failed(verdicts: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [v for v in verdicts or [] if isinstance(v, dict) and v.get("verdict") == FAILED]


# ---- words ----------------------------------------------------------------------------------------------------------------


def _num(x: Any) -> str:
    return f"{float(x):.4g}" if _number(x) else "no value"


def _measured(s: dict[str, Any]) -> str:
    if s.get("on") == "trials":
        if "successes" in s:
            return f"{_num(s['value'])} ({s['successes']} of {s['trials']} trials)"
        se = s.get("se")
        return (f"{_num(s['value'])} (mean of {s['trials']} trials"
                + (f", standard error {float(se):.2g}" if _number(se) and se else "") + ")")
    if s.get("on") == "analysis":
        return f"{_num(s['value'])} (no interval)"
    return "no value"


def _differing(settings: list[dict[str, Any]], selector: dict[str, Any]) -> list[str]:
    """The settings, beyond the control's own, whose values differ between the cells shown (the ones that tell them
    apart)."""
    axes: list[str] = []
    for s in settings:
        for a in s.get("setting") or {}:
            if a not in selector and a not in axes:
                axes.append(a)
    return [a for a in axes if len({str((s.get("setting") or {}).get(a)) for s in settings}) > 1]


def words(v: dict[str, Any]) -> str:
    """One plain sentence for a control's verdict (short: the quality report keeps about 220 characters)."""
    tol = (f"within {v['tolerance']:g} (relative)" if v.get("tolerance_mode") == "relative"
           else f"within {v['tolerance']:g}")
    head = f"{v['name']}: {v['quantity']} at {_key_words(v['setting'])} should be {v['expected']:.4g} {tol}"
    if v["verdict"] == NOT_JUDGED:
        return f"{head}; not judged: {v['why']}"
    measured = [s for s in v.get("settings") or [] if s.get("verdict") != NOT_JUDGED]
    shown = [s for s in measured if v["verdict"] != FAILED or s.get("verdict") == FAILED]
    apart = _differing(v.get("settings") or [], v["setting"])
    where = "; ".join(_measured(s) + (" at " + _key_words({a: s["setting"].get(a) for a in apart}) if apart else "")
                      for s in shown[:3])
    more = f" (and {len(shown) - 3} more settings)" if len(shown) > 3 else ""
    if v["verdict"] == NOT_DECIDED:
        return f"{head}; not decided: {v['why']}" + (f"; FI measured {where}{more}" if where else "")
    said = {PASSED: "passed", FAILED: "failed"}[v["verdict"]]
    return f"{head}; {said}: FI measured {where}{more}"


# ---- the record -------------------------------------------------------------------------------------------------------------


def read_record(quest_root: Path | str) -> dict[str, Any]:
    try:
        data = json.loads((Path(quest_root) / RECORD_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and data.get("schema") == SCHEMA else {}


def write_record(quest_root: Path | str, verdicts: list[dict[str, Any]], *, iteration: int) -> None:
    """Add this pass's verdicts to ``.fi/controls.json`` (every pass is kept, the last is the one that stands)."""
    record = read_record(quest_root) or {"schema": SCHEMA, "judgements": []}
    record["judgements"] = [*(record.get("judgements") or []), {"iteration": int(iteration), "controls": verdicts}]
    path = Path(quest_root) / RECORD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")


def last(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The verdicts of the last judgement in ``record`` (the one that stands)."""
    judgements = [j for j in (record or {}).get("judgements") or [] if isinstance(j, dict)]
    return [v for v in (judgements[-1].get("controls") or []) if isinstance(v, dict)] if judgements else []


def has_failed(quest_root: Path | str) -> bool:
    """Whether a control of the last judgement failed (the paper's closing sentence names it: core/quality_pointer.py)."""
    return bool(failed(last(read_record(quest_root))))


def quality_rows(record: dict[str, Any]) -> list[tuple[str, str]]:
    """``[(sentence, status)]`` for the quality report: each control of the last judgement that failed (``not fixed``)
    or was not decided or not judged (``left out``). A passed control is not a problem."""
    rows: list[tuple[str, str]] = []
    for v in last(record):
        # The verdict and the numbers come first: the quality report keeps the first 220 characters of a row.
        if v.get("verdict") == FAILED:
            rows.append((words(v), "not fixed"))
        elif v.get("verdict") in (NOT_JUDGED, NOT_DECIDED):
            rows.append((words(v), "left out"))
    return rows
