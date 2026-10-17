"""The values every run of a study goes through together (``protocol.in_each_run``), and how one run's numbers per value
are read and recorded.

A setting of the grid is run on its own: its own process, its own trials, its own seeds. A value of something the study
compares inside each run is not: one run fits all five temperatures together, or applies every procedure to the same
simulated data. Such a run gets the whole list of values in its cell (``cell["temperature_K"] == [290.2, 298.5, ...]``)
and may return, beside its plain numbers (one per run: an activation energy fitted across the temperatures, a difference
between two procedures on the same data), a number per value (``{"k_hat": {"290.2": ..., "298.5": ...}}``). FI records
the run once, and keeps each value's numbers as the record of a setting of its own (``temperature_K=290.2``) next to it,
so a number per value is counted once per value and a number per run once per run.

This file is standard library only and holds nothing but top-level functions: FI's trial harness and the ``code/run.py``
it writes carry its text, so the three places that read what a run returned read it the same way."""

import math


def fmt(value):
    """A value as a cell key writes it (the same as ``core.trial_runner._fmt``)."""
    if isinstance(value, float) and value.is_integer():
        return repr(value)
    if isinstance(value, list):
        return "[" + ";".join(fmt(x) for x in value) + "]"
    return str(value)


def level_cells(in_each_run):
    """Every combination of the values compared inside each run, factors in their order, values in theirs; ``[]`` when
    there are none."""
    factors = [(str(k), list(v)) for k, v in (in_each_run or {}).items()]
    if not factors:
        return []
    out = [{}]
    for name, values in factors:
        out = [dict(cell, **{name: value}) for cell in out for value in values]
    return out


def level_key(level):
    """``"procedure=bh"``: one combination of the values compared inside each run, written as a cell key is."""
    return ",".join("{}={}".format(name, fmt(value)) for name, value in level.items())


def record_key(run_key, level_key_text):
    """The key of a value's own record: the run's setting and the value (``"m=20,pi1=0.05,procedure=bh"``)."""
    return "{},{}".format(run_key, level_key_text) if run_key else level_key_text


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _close(a, b):
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(x) or math.isnan(y):
        return False
    return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)


def _same_level_value(written, value):
    if isinstance(value, str):
        return str(written).strip() == value
    return _close(written, value)


def _level_of(key, levels):
    """The level (a dict of factor -> value from ``levels``) a returned key names, or ``None``. With one factor the key may
    be the value alone (``"bh"``, ``290.2``, ``"290.2"``); any key may be written ``factor=value[,factor=value]``."""
    text = str(key).strip()
    pairs = {}
    if "=" in text:
        for part in text.split(","):
            name, _, raw = part.partition("=")
            pairs[name.strip()] = raw.strip()
        for level in levels:
            if set(pairs) == set(level) and all(_same_level_value(pairs[n], level[n]) for n in level):
                return level
        return None
    for level in levels:
        if len(level) == 1:
            (only,) = level.values()
            if (isinstance(key, str) and _same_level_value(key, only)) or (_number(key) and _close(key, only)):
                return level
    return None


def read_returned(value, levels, entry):
    """``(the values, "")`` or ``(None, why)`` for what the simulation function returned for one run.

    A number stays a number (a measurement of the run). With values compared inside each run (``levels``, from
    :func:`level_cells`), a mapping of each level to a number is a measurement per value, kept as ``{level_key: number}``;
    with one factor compared inside the run, a list of one number per value, in the order the run was given them, is the
    same. A per-value measurement must give every value, and nothing else."""
    if not isinstance(value, dict) or not value:
        return None, "{}() returned {}, not a dict of numbers".format(entry, type(value).__name__)
    out = {}
    bad = []
    for name, v in value.items():
        name = str(name)
        if _number(v):
            out[name] = float(v)
            continue
        if not isinstance(v, (str, bytes, dict, list, tuple)) and hasattr(v, "tolist"):
            v = v.tolist()  # an array, or a scalar of one (numpy): its plain Python form
            if _number(v):
                out[name] = float(v)
                continue
        if not levels:
            bad.append(name)
            continue
        keys = [level_key(level) for level in levels]
        if isinstance(v, (list, tuple)) and len(levels[0]) > 1:
            return None, ("{}() returned `{}` as a list, and a list is only read when one thing is compared inside the "
                          "run; with several, return a mapping keyed like {} (one number for each of the {} "
                          "combinations)".format(entry, name, "`" + keys[0] + "`", len(levels)))
        if isinstance(v, (list, tuple)) and len(levels[0]) == 1:
            if len(v) != len(levels) or not all(_number(x) for x in v):
                return None, ("{}() returned `{}` as a list of {} item(s): a value per level of what each run compares "
                              "must be one number for each of the {} value(s) the run was given ({}), in that order"
                              .format(entry, name, len(v), len(levels), ", ".join(keys[:6])))
            out[name] = {k: float(x) for k, x in zip(keys, v)}
            continue
        if isinstance(v, dict):
            got = {}
            for k, x in v.items():
                level = _level_of(k, levels)
                if level is None:
                    return None, ("{}() returned `{}` with the key {!r}, which is not one of the values the run was given "
                                  "({})".format(entry, name, k, ", ".join(keys[:6])))
                if not _number(x):
                    return None, "{}() returned `{}` with a non-number for {!r}".format(entry, name, k)
                lk = level_key(level)
                if lk in got:
                    return None, "{}() returned `{}` with {} given twice".format(entry, name, lk)
                got[lk] = float(x)
            missing = [k for k in keys if k not in got]
            if missing:
                return None, ("{}() returned `{}` without {}: a value per level must give every value the run was given "
                              "(NaN where it does not exist)".format(entry, name, ", ".join(missing[:6])))
            out[name] = {k: got[k] for k in keys}
            continue
        bad.append(name)
    if bad:
        return None, "{}() returned non-numbers for {}".format(entry, bad[:5])
    return out, ""


def split(values):
    """``(per_run, per_level)`` of what :func:`read_returned` kept: the plain numbers, and ``{level_key: {name: number}}``
    for the numbers per value."""
    per_run, per_level = {}, {}
    for name, v in (values or {}).items():
        if isinstance(v, dict):
            for lk, x in v.items():
                per_level.setdefault(lk, {})[name] = x
        else:
            per_run[name] = v
    return per_run, per_level


def flat(run_key, values):
    """``[(record key, name, number)]`` for one run: the plain numbers under the run's own key, each value's numbers under
    the key of that value's record (:func:`record_key`)."""
    per_run, per_level = split(values)
    out = [(run_key, name, v) for name, v in per_run.items()]
    for lk, named in per_level.items():
        out.extend((record_key(run_key, lk), name, v) for name, v in named.items())
    return out


def summary_cells(run_cell, run_key, planned, rows, levels):
    """The per-setting summary (``raw/trials.json`` cells) of one run setting: the setting itself with the numbers per run,
    then, for each value compared inside each run, a record of its own (its cell names the value too) with the numbers per
    value. Each holds ``values`` and ``trials`` (the trial each value came from), ``count``, ``total`` and ``non_finite``;
    every record of a setting has the setting's ``planned``, ``ok`` and ``failed`` (the planned trials that did not end
    well, reported or not), since a run that fails fails for every value. ``rows`` are the run's trials (``{"trial",
    "status" or "ok", "values"}``)."""
    def ok(row):
        return row.get("status") == "ok" if "status" in row else bool(row.get("ok"))

    good = [r for r in rows if ok(r)]
    done = len(good)

    def metrics_of(pick):
        metrics, trials_of = {}, {}
        for row in good:
            for name, v in pick(row).items():
                metrics.setdefault(name, []).append(v)
                trials_of.setdefault(name, []).append(row.get("trial"))
        return {
            name: {"values": vals, "trials": trials_of[name], "count": len(vals),
                   "total": math.fsum(v for v in vals if math.isfinite(v)),
                   "non_finite": sum(1 for v in vals if not math.isfinite(v))}
            for name, vals in metrics.items()
        }

    head = {"cell": dict(run_cell), "key": run_key, "planned": planned, "ok": done, "failed": planned - done}
    first: dict = dict(head, metrics=metrics_of(lambda r: split(r.get("values"))[0]))
    if levels:
        first["compared_in_each_run"] = [level_key(level) for level in levels]
    out: list = [first]
    if levels:
        for level in levels:
            lk = level_key(level)
            out.append(dict(head, cell=dict(run_cell, **level), key=record_key(run_key, lk), level=dict(level),
                            metrics=metrics_of(lambda r, lk=lk: split(r.get("values"))[1].get(lk, {}))))
    return out


def level_named(named, levels):
    """The level (one of ``levels``) whose values are those ``named`` gives for every factor compared inside the run, or
    ``None`` when ``named`` leaves one out or gives a value the run is not given."""
    if not levels or set(named) != set(levels[0]):
        return None
    for level in levels:
        if all(_same_level_value(named[n], level[n]) for n in level):
            return level
    return None


def for_case(values, levels, named):
    """What a check reads from one run on its case: the numbers per run as they are; each number per value at the value
    the case names (``named`` gives every factor compared inside the run), or, when it names none, the list of every
    value's number in the order the run was given them."""
    level = level_named(named, levels) if named else None
    lk = level_key(level) if level is not None else None
    out = {}
    for name, v in (values or {}).items():
        if isinstance(v, dict):
            out[name] = v.get(lk) if lk is not None else [v[level_key(lv)] for lv in levels if level_key(lv) in v]
        else:
            out[name] = v
    return out


def of(protocol):
    """``protocol.in_each_run`` (factor -> its values), or ``None`` when the protocol compares nothing inside its runs."""
    inside = protocol.get("in_each_run") if isinstance(protocol, dict) else None
    return inside if isinstance(inside, dict) and inside else None


def parse_level_key(text):
    """``{"temperature_K": 290.2, "procedure": "bh"}`` from a level's key (:func:`level_key`): a value that reads as a
    number is that number, any other is its text."""
    out = {}
    for part in str(text).split(","):
        name, _, raw = part.partition("=")
        try:
            number = float(raw)
            out[name.strip()] = number if math.isfinite(number) else raw.strip()  # "nan" and "inf" are names
        except ValueError:
            out[name.strip()] = raw.strip()
    return out


def groups(cells):
    """The records of ``raw/trials.json`` (each ``{"cell", "metrics", ...}``, a value's own record carrying its ``level``)
    as the groups a check judges a metric over: a number per run lives only in the records of the grid's settings and a
    number per value only in the records of a value, so each is judged over the records that hold it. A study that compares
    nothing inside its runs is one group, as it always was."""
    per_value = [c for c in cells if c.get("level")]
    if not per_value:
        return [list(cells)]
    return [g for g in ([c for c in cells if not c.get("level")], per_value) if g]
