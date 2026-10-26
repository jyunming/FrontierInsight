"""The values every run of a study goes through together (``protocol.in_each_run``), and how one run's numbers per value
are read and recorded.

A setting of the grid is run on its own: its own process, its own trials, its own seeds. A value of something the study
compares inside each run is not: one run fits all five temperatures together, or applies every procedure to the same
simulated data. Such a run gets the whole list of values in its cell (``cell["temperature_K"] == [290.2, 298.5, ...]``)
and may return, beside its plain numbers (one per run: an activation energy fitted across the temperatures, a difference
between two procedures on the same data), a number per value (``{"k_hat": {"290.2": ..., "298.5": ...}}``). FI records
the run once, and keeps each value's numbers as the record of a setting of its own (``temperature_K=290.2``) next to it,
so a number per value is counted once per value and a number per run once per run.

This file is standard library only and holds nothing but top-level definitions: FI's trial harness and the
``code/run.py`` it writes carry its text, so the three places that read what a run returned read it the same way."""

import itertools
import math


def fmt(value):
    """A value as a cell key writes it (the same as ``core.trial_runner._fmt``)."""
    if isinstance(value, float) and value.is_integer():
        return repr(value)
    if isinstance(value, list):
        return "[" + ";".join(fmt(x) for x in value) + "]"
    return str(value)


class Levels:
    """Every combination of the values compared inside each run, worked out when one is read, never held as a list: a
    plan that put seven lists in each run had 1,215,120 combinations, its runs returned none of them one by one, and FI
    wrote every one into the spec of each setting and kept them (a 30 GB process). The first factor varies slowest."""

    def __init__(self, factors):
        self.factors = [(str(name), list(values)) for name, values in factors]

    def __len__(self):
        n = 1
        for _name, values in self.factors:
            n *= len(values)
        return n if self.factors else 0

    def __bool__(self):
        return len(self) > 0

    def __iter__(self):
        names = [name for name, _values in self.factors]
        for combo in itertools.product(*[values for _name, values in self.factors]):
            yield dict(zip(names, combo))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        n = len(self)
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError("level index out of range")
        out = {}
        for name, values in reversed(self.factors):
            index, at = divmod(index, len(values))
            out[name] = values[at]
        return {name: out[name] for name, _values in self.factors}

    def __eq__(self, other):
        return list(self) == list(other) if isinstance(other, (Levels, list, tuple)) else NotImplemented

    def to_json(self):
        return {"factors": [[name, values] for name, values in self.factors]}


def level_cells(in_each_run):
    """Every combination of the values compared inside each run, factors in their order, values in theirs (a
    :class:`Levels`, read one combination at a time); ``[]`` when there are none."""
    factors = [(str(k), list(v)) for k, v in (in_each_run or {}).items()]
    return Levels(factors) if factors else []


def levels_json(levels):
    """``levels`` as a spec or a record writes them: the factors of a :class:`Levels`, a list as it is."""
    return levels.to_json() if isinstance(levels, Levels) else list(levels or [])


def levels_of(written):
    """The levels a spec or a record wrote (:func:`levels_json`): a :class:`Levels` again, or the list."""
    if isinstance(written, Levels):
        return written
    if isinstance(written, dict) and isinstance(written.get("factors"), list):
        return Levels(written["factors"])
    if isinstance(written, (str, bytes)):
        raise TypeError("the spec's levels are text, not the values compared inside each run: write them with levels_json")
    return list(written or [])


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


def factors_of(levels):
    """``[(factor, [its values])]`` of the levels compared inside each run (a :class:`Levels`, or a list of levels as an
    older record wrote them)."""
    if isinstance(levels, Levels):
        return levels.factors
    out = {}
    for level in levels or []:
        for name, value in level.items():
            seen = out.setdefault(name, [])
            if not any(x == value for x in seen):
                seen.append(value)
    return list(out.items())


def sub_levels(levels, axes):
    """Every combination of the values of just ``axes`` (factor names, any order), written in the factors' order: the
    values a result that varies over only those factors is given one number for."""
    chosen = [(name, values) for name, values in factors_of(levels) if name in set(axes)]
    return [dict(zip([n for n, _v in chosen], combo)) for combo in itertools.product(*[v for _n, v in chosen])] if chosen else []


def axes_of_key(key):
    """The factor names a returned key names (``("pitch", "gap")`` for ``"gap=80,pitch=74"``, in the order it wrote them),
    or ``None`` when the key is not written ``factor=value[,factor=value]``."""
    text = str(key).strip()
    if "=" not in text:
        return None
    names = tuple(part.partition("=")[0].strip() for part in text.split(",") if "=" in part)
    return names if len(set(names)) == len(names) else None  # one factor named twice is not a level


def _level_of(key, levels):
    """The level a returned key names, or ``None``: a dict of factor -> value. With one factor the key may be the value
    alone (``"bh"``, ``290.2``, ``"290.2"``). Any key may be written ``factor=value[,factor=value]``; one that names every
    factor is a combination of the run, one that names only some (``"pitch=74"``) is a value of just those, for a result that
    varies over those factors alone (:func:`read_returned`). With several factors a bare value names none of them."""
    text = str(key).strip()
    for level in levels:
        # FI's own key for the level first: a value that is a text with a comma in it ("mirror-symmetric, optimised")
        # cannot be split back into its pairs, and every key of every level was refused.
        if text == level_key(level):
            return level
    pairs = {}
    if "=" in text:
        for part in text.split(","):
            name, _, raw = part.partition("=")
            if name.strip() in pairs:
                return None  # one factor named twice ("pitch=0,pitch=74") is not a level; never keep just one of them
            pairs[name.strip()] = raw.strip()
        factors = factors_of(levels)
        names = [n for n, _v in factors]
        if not pairs or not set(pairs) <= set(names):
            return None
        if len(pairs) < len(names):
            for level in sub_levels(levels, set(pairs)):
                if all(_same_level_value(pairs[n], level[n]) for n in level):
                    return level
            return None
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


def key_forms(levels):
    """What a returned key may look like, with the run's own factor names: ``"pitch=74,gap=80"`` for a combination, and,
    with several factors, ``"pitch=74"`` for a result over one of them alone."""
    factors = factors_of(levels)
    if not factors:
        return ""
    full = level_key(levels[0])
    if len(factors) == 1:
        return "`{}` (or the value alone)".format(full)
    return ("`{}` for a number per combination of all {} ({}), or, for a number that varies over only some of them, "
            "those alone (`{}`): a bare value names none of them"
            .format(full, len(factors), ", ".join(n for n, _v in factors), level_key(sub_levels(levels, [factors[0][0]])[0])))


#: The one name a run gives the reasons for results it has no number for (``None``): ``{result: reason}``, a reason in a
#: few words ("no crossing in the focus range", "the line does not print"). It is not a result; FI keeps the reasons in
#: the record beside the missing values, so a value that cannot be measured is never a stand-in number.
WHY_MISSING = "why_missing"


def missing_reasons(value, kept):
    """``{result: reason}`` from what a run returned under :data:`WHY_MISSING`, for each result it has no number for
    (``kept`` from :func:`read_returned`: absent, not a number, or not a number for some value compared inside the run);
    a reason for a result that has its numbers is not kept."""
    raw = value.get(WHY_MISSING) if isinstance(value, dict) else None
    if not isinstance(raw, dict):
        return {}
    out = {}
    for name, why in raw.items():
        name = str(name)
        if not isinstance(why, str) or not why.strip():
            continue
        v = (kept or {}).get(name)
        lacks = v is None or (_number(v) and not math.isfinite(v)) or (
            isinstance(v, dict) and any(not math.isfinite(x) for x in v.values()))
        if lacks:
            out[name] = " ".join(why.split())[:200]
    return out


def none_as_nan(value):
    """``value`` with ``None`` read as NaN, a value that does not exist, also inside a list or a mapping of numbers. The
    code writer is told to return ``null`` for a value that is undefined (and a plan's failure policy says so too), and
    the harness reads that as no value, as it reads NaN, not as a broken run: a real study's settings where a value
    did not exist returned ``None`` as its plan said, and the run was refused for it."""
    nan = float("nan")
    if value is None:
        return nan
    if isinstance(value, (list, tuple)):
        return [nan if x is None else x for x in value]
    if isinstance(value, dict):
        return {k: (nan if x is None else x) for k, x in value.items()}
    return value


def read_returned(value, levels, entry):
    """``(the values, "")`` or ``(None, why)`` for what the simulation function returned for one run.

    A number stays a number (a measurement of the run). With values compared inside each run (``levels``, from
    :func:`level_cells`), a mapping of each level to a number is a measurement per value, kept as ``{level_key: number}``;
    with one factor compared inside the run, a list of one number per value, in the order the run was given them, is the
    same. A per-value measurement must give every value, and nothing else."""
    if not isinstance(value, dict) or not value:
        return None, "{}() returned {}, not a dict of numbers".format(entry, type(value).__name__)
    if isinstance(value.get(WHY_MISSING), dict):
        value = {k: v for k, v in value.items() if k != WHY_MISSING}  # the reasons, read by missing_reasons
        if not value:
            return None, "{}() returned only `{}`, no result".format(entry, WHY_MISSING)
    out = {}
    bad = []
    for name, v in value.items():
        name = str(name)
        v = none_as_nan(v)
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
                          "run; with several, return a mapping keyed {} (one number for each value of the settings it "
                          "varies over)".format(entry, name, key_forms(levels)))
        if isinstance(v, (list, tuple)) and len(levels[0]) == 1:
            if len(v) != len(levels) or not all(_number(x) for x in v):
                return None, ("{}() returned `{}` as a list of {} item(s): a value per level of what each run compares "
                              "must be one number for each of the {} value(s) the run was given ({}), in that order"
                              .format(entry, name, len(v), len(levels), ", ".join(keys[:6])))
            out[name] = {k: float(x) for k, x in zip(keys, v)}
            continue
        if isinstance(v, dict):
            got = {}
            axes = None
            for k, x in v.items():
                level = _level_of(k, levels)
                if level is None:
                    return None, ("{}() returned `{}` with the key {!r}, which is not one of the values the run was given: key it "
                                  "{}".format(entry, name, k, key_forms(levels)))
                if not _number(x):
                    return None, "{}() returned `{}` with a non-number for {!r}".format(entry, name, k)
                if axes is None:
                    axes = set(level)
                elif axes != set(level):
                    return None, ("{}() returned `{}` keyed by {} and by {} in one result: key all of one result over the "
                                  "same settings".format(entry, name, ", ".join(sorted(axes)), ", ".join(sorted(level))))
                lk = level_key(level)
                if lk in got:
                    return None, "{}() returned `{}` with {} given twice".format(entry, name, lk)
                got[lk] = float(x)
            expected = [level_key(lv) for lv in (sub_levels(levels, axes) if axes and len(axes) < len(levels[0]) else levels)]
            missing = [k for k in expected if k not in got]
            if missing:
                return None, ("{}() returned `{}` without {}: a value per level must give every value the run was given "
                              "(NaN where it does not exist)".format(entry, name, ", ".join(missing[:6])))
            out[name] = {k: got[k] for k in expected}
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


def record_levels(levels, rows):
    """The levels that get a record of their own: for each set of factors the run's numbers per value were keyed by (all of
    them for a number per combination, ``pitch`` alone for a number that varies over just that), every combination of
    their values, the full set first. ``rows`` are the run's trials (only the ones that ended well count)."""
    used = []
    for row in rows:
        if not (row.get("status") == "ok" if "status" in row else row.get("ok")):
            continue
        for lk in split(row.get("values"))[1]:
            axes = axes_of_key(lk)
            if axes is not None and set(axes) not in used:
                used.append(set(axes))
    names = [n for n, _v in factors_of(levels)]
    used.sort(key=lambda a: len(a) != len(names))  # stable: the full set first, the others as met
    out = []
    for axes in used:
        out.extend(list(levels) if len(axes) == len(names) else sub_levels(levels, axes))
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
        metrics, trials_of, why = {}, {}, {}
        for row in good:
            for name, v in pick(row).items():
                metrics.setdefault(name, []).append(v)
                trials_of.setdefault(name, []).append(row.get("trial"))
                reason = (row.get(WHY_MISSING) or {}).get(name) if not math.isfinite(v) else None
                if reason:  # why the run has no number here, as it said: counted by reason
                    why.setdefault(name, {})[reason] = why.get(name, {}).get(reason, 0) + 1
        return {
            name: {"values": vals, "trials": trials_of[name], "count": len(vals),
                   "total": math.fsum(v for v in vals if math.isfinite(v)),
                   "non_finite": sum(1 for v in vals if not math.isfinite(v)),
                   **({WHY_MISSING: why[name]} if name in why else {})}
            for name, vals in metrics.items()
        }

    head = {"cell": dict(run_cell), "key": run_key, "planned": planned, "ok": done, "failed": planned - done}
    first: dict = dict(head, metrics=metrics_of(lambda r: split(r.get("values"))[0]))
    out: list = [first]
    # A value gets a record of its own only when the run returned a number per value. A run that returned none (every
    # number is one per run) has nothing to record per value: an empty record would be a setting that never ran,
    # claiming the run's trials. The list of values compared goes with them; without them it would be every combination
    # of the lists, written for nothing.
    if levels and any(split(r.get("values"))[1] for r in good):
        recorded = record_levels(levels, rows)
        first["compared_in_each_run"] = [level_key(level) for level in recorded]
        for level in recorded:
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


def levels_named(named, levels):
    """Every level (in the run's order) that has the value ``named`` gives for each factor it names. A case that names
    each factor compared inside the run gives one level, a case that names only some gives the levels of the factors it
    leaves out, and a case that names none gives all of them; ``[]`` when a value it names is one the run is not
    given."""
    return [level for level in levels
            if all(n in level and _same_level_value(v, level[n]) for n, v in (named or {}).items())]


def for_case(values, levels, named):
    """What a check reads from one run on its case: the numbers per run as they are; each number per value at the value
    the case names, or, when the case names more than one value of what that number varies over (it names only some of the
    factors compared inside the run, or none), the list of the number at each of them in the order the run was given them.
    The case does not pick the value of a factor it leaves out, so the check reads every one. A number that varies over
    only some of the factors (keyed ``pitch=74``) is read at the values of those, whatever the case says of the others."""
    matching = levels_named(named, levels) if named else list(levels or [])
    out = {}
    for name, v in (values or {}).items():
        if isinstance(v, dict):
            axes = axes_of_key(next(iter(v), ""))
            keys = []
            for lv in matching:
                lk = level_key({n: lv[n] for n in lv if axes is None or n in axes})
                if lk not in keys:
                    keys.append(lk)
            out[name] = v.get(keys[0]) if (named and len(keys) == 1) else [v[k] for k in keys if k in v]
        else:
            out[name] = v
    return out


def of(protocol):
    """``protocol.in_each_run`` (factor -> its values), or ``None`` when the protocol compares nothing inside its runs."""
    inside = protocol.get("in_each_run") if isinstance(protocol, dict) else None
    return inside if isinstance(inside, dict) and inside else None


def of_study(protocol):
    """What each run of the study compares inside it, wherever the plan wrote it: a search for the best design
    (``protocol.optimisation``) holds its own ``in_each_run`` in that block, any other study in the protocol. Everything that
    calls the simulation on a case (the checks against known answers, a run again at finer settings) gives the run these
    values, as the study's own runs get them; without them the simulation is called without a list it needs and returns a
    number per value that FI reads as no number at all."""
    block = protocol.get("optimisation") if isinstance(protocol, dict) else None
    if isinstance(block, dict):
        inside = block.get("in_each_run") if isinstance(block.get("in_each_run"), dict) else {}
        inside = {str(k): list(v) for k, v in inside.items() if isinstance(v, list)}
        return inside or None
    return of(protocol)


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
