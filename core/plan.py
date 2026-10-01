"""The plan of a quest: one file a person can read, edit and ask to have changed.

Until now a quest's plan existed only as the ``design`` in its checkpoint (a hypothesis, variables, a method,
figures and result bounds) and, in ``needs/DESIGN_HISTORY.json``, the hypothesis of each version. Nothing
in the quest folder said what the literature had established, what the experiment was for, what would count
as support, or let a person change any of it before compute was spent. The ``plan`` step now writes
``plan.md`` in the quest folder, after the literature is in and before the experiment is designed.

The file has two kinds of section:

* prose for the reader and for the paper's methods: *In short*, *What the literature says* (each source
  named), *The gap this experiment addresses*, *The model behind the numbers* (shown from the protocol's ``model`` and
  ``oracles``: what model produces the numbers, its equations with their sources, and where each check's expected value
  comes from; only the design block is read back), *How we will judge whether the code got better* (the protocol's
  ``criteria``, :mod:`core.criteria`; also shown, not read back), *Success criteria*, *Risks*, *What this quest will not do*, and
  *Checks already made* (what the methodology audit objected to and how the design answers it);
* **the design**, a fenced YAML block under the heading *The design (used as written)*: the hypothesis,
  variables, method, expected outcome, planned figures, dependencies and result bounds. That block is the
  design, exactly. The design step reads it from the file, so what a person edits is what runs, and nothing
  re-derives it from prose. Its ``study_type`` says whether the study measures over settings chosen in advance or
  searches for the best design; a search's ``protocol.optimisation`` block is read by :mod:`core.optimisation_plan`,
  which also writes the *What is being optimised* section.

The file is the source of truth. A quest that pauses for the plan (``pauses.plan: ask``) stops once it is
written; the person edits it, or asks for a change (``--revise-plan``), and resumes. A block that cannot be
read stops the quest again with the reason rather than being guessed at.

Every version is kept under ``.fi/plan_versions/`` and listed in ``needs/PLAN_HISTORY.json`` with who wrote it
(the model, a request, the engine's oracle check, or the person) and its hash, and the first design entry of ``needs/DESIGN_HISTORY.json``
names the hash of the plan it came from: the plan is the pre-registered version of the hypothesis.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PLAN_FILE = "plan.md"
DESIGN_HEADING = "The design (used as written)"
_HEADING_RE = re.compile(r"^#{2,3}\s+the design\b.*$", re.IGNORECASE | re.MULTILINE)
_FENCE_RE = re.compile(r"```[ \t]*(?:ya?ml|json)?[ \t]*\n(.*?)\n[ \t]*```", re.DOTALL | re.IGNORECASE)
_OUTER_FENCE_RE = re.compile(r"\A\s*```[a-z]*[ \t]*\n(.*)\n```\s*\Z", re.DOTALL | re.IGNORECASE)

# The design's keys and what each is allowed to hold. Keys a design has beyond these (a survey's outline,
# say) are kept untouched.
_LIST_KEYS = ("figures_planned", "dependencies")


def plan_path(quest_root: Path) -> Path:
    return quest_root / PLAN_FILE


def sha256(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[\n;]+", value) if part.strip()]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = "; ".join(f"{k}: {v}" for k, v in item.items() if v not in (None, ""))
            else:
                text = str(item).strip()
            if text:
                out.append(text)
        return out
    return [str(value)]


def normalize_design(design: Any) -> tuple[dict[str, Any] | None, str | None]:
    """``(design, None)`` when ``design`` can drive the experiment, else ``(None, why)``.

    Lenient about form (a dependency list written as a comma string, a missing optional key) and strict about
    what would send the experiment somewhere else: no hypothesis, or a result bound that is not a bound."""
    if not isinstance(design, dict):
        return None, "the design block is not a mapping of names to values"
    out: dict[str, Any] = dict(design)
    hypothesis = out.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        return None, "the design has no `hypothesis` (one sentence)"
    out["hypothesis"] = hypothesis.strip()
    variables = out.get("variables")
    if variables is None:
        out["variables"] = {"independent": [], "dependent": [], "controls": []}
    elif not isinstance(variables, dict):
        return None, "`variables` must list `independent`, `dependent` and `controls`"
    else:
        out["variables"] = {
            **variables,
            **{key: _as_list(variables.get(key)) for key in ("independent", "dependent", "controls")},
        }
    for key in ("method", "expected_outcome"):
        if out.get(key) is not None and not isinstance(out[key], str):
            return None, f"`{key}` must be text"
    for key in _LIST_KEYS:
        out[key] = _as_list(out.get(key))
    protocol = out.get("protocol")
    if protocol is not None:
        out["protocol"], why = normalize_protocol(protocol)
        if out["protocol"] is None:
            return None, why
    if out.get("study_type") is not None:
        # Measure, or find the best design (core/optimisation_plan.py). A plan that says find_best_design without its
        # optimisation block is readable (the quest stops before anything runs and says what is missing); one that says
        # measure beside an optimisation block contradicts itself, and is refused rather than run as a sweep.
        from . import optimisation_plan

        study_type, why = optimisation_plan.normalize_study_type(out["study_type"])
        if study_type is None:
            return None, why
        if study_type == "measure" and optimisation_plan.has_block(out):
            return None, ("`study_type` is measure, but the protocol has an `optimisation` block (a search for the best "
                          "design): set `study_type: find_best_design`, or replace the block with a `grid`")
        out["study_type"] = study_type
    assertions = out.get("result_assertions")
    if assertions is None:
        out["result_assertions"] = []
    elif not isinstance(assertions, list):
        return None, "`result_assertions` must be a list of bounds"
    else:
        for index, item in enumerate(assertions, start=1):
            if not isinstance(item, dict) or not str(item.get("path") or "").strip():
                return None, f"result_assertions entry {index} has no `path` (the name of the result)"
            for bound in ("min", "max"):
                value = item.get(bound)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    return None, f"result_assertions entry {index}: `{bound}` must be a number"
            low, high = item.get("min"), item.get("max")
            if low is not None and high is not None and low > high:
                return None, f"result_assertions entry {index}: `min` is above `max`"
    return out, None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_protocol(protocol: Any) -> tuple[dict[str, Any] | None, str | None]:
    """``(protocol, None)`` when the experiment protocol block can be checked against a script, else ``(None, why)``.

    The protocol fixes what an experiment is not allowed to change on its own (:mod:`core.protocol_check`): ``grid`` (each
    parameter it sweeps, with every value), ``runs_per_setting``, ``thresholds``, and the prose ``seed_policy``,
    ``ci_method``, ``failure_policy`` (how a trial that fails is treated) and ``acceptance`` (what would count as support). Every key is optional, and keys beyond these are kept.
    Strict about the numbers, which are what a check reads."""
    if not isinstance(protocol, dict):
        return None, "`protocol` must be a mapping (`grid`, `runs_per_setting`, `thresholds`, ...)"
    out: dict[str, Any] = dict(protocol)
    grid = out.get("grid")
    if grid is not None:
        if not isinstance(grid, dict):
            return None, "`protocol.grid` must map each parameter to the list of values it takes"
        fixed: dict[str, list[Any]] = {}
        for axis, values in grid.items():
            if _number(values) or (isinstance(values, str) and values.strip()):
                values = [values]
            if (not isinstance(values, list) or not values
                    or not (all(_number(v) for v in values) or all(isinstance(v, str) and v.strip() for v in values))):
                return None, f"`protocol.grid.{axis}` must be a non-empty list of numbers, or a non-empty list of names (text), not a mix"
            fixed[str(axis)] = [v.strip() if isinstance(v, str) else v for v in values]
        out["grid"] = fixed
    runs = out.get("runs_per_setting")
    if runs is not None and (not _number(runs) or runs < 1 or int(runs) != runs):
        return None, "`protocol.runs_per_setting` must be a whole number of at least 1"
    thresholds = out.get("thresholds")
    if thresholds is not None:
        if not isinstance(thresholds, dict) or not all(_number(v) for v in thresholds.values()):
            return None, "`protocol.thresholds` must map each name to a number"
    for key in ("seed_policy", "ci_method", "failure_policy"):
        if out.get(key) is not None and not isinstance(out[key], str):
            return None, f"`protocol.{key}` must be text"
    precision = out.get("precision")
    if precision is not None:
        if not isinstance(precision, dict) or not _number(precision.get("target_half_width")) or not (
            0 < precision["target_half_width"] < 1
        ):
            return None, "`protocol.precision` must be a mapping with a `target_half_width` between 0 and 1 (for example 0.03)"
        for key in ("metric", "reason"):
            if precision.get(key) is not None and not isinstance(precision[key], str):
                return None, f"`protocol.precision.{key}` must be text"
    metrics = out.get("metrics")
    if metrics is not None:
        from . import metric_spec

        fixed_metrics, why = metric_spec.normalize(metrics)
        if fixed_metrics is None:
            return None, why
        out["metrics"] = fixed_metrics
    oracles = out.get("oracles")
    if oracles is not None:
        if isinstance(oracles, (str, dict)):
            oracles = [oracles]
        if not isinstance(oracles, list):
            return None, "`protocol.oracles` must be a list of checks (each with a `name` and a `check`)"
        fixed_oracles: list[dict[str, Any]] = []
        for index, item in enumerate(oracles, start=1):
            if isinstance(item, str) and item.strip():
                item = {"name": item.strip(), "check": item.strip()}
            if not isinstance(item, dict) or not str(item.get("name") or "").strip():
                return None, f"`protocol.oracles` entry {index} has no `name`"
            for key in ("check", "kind", "reference"):
                value = item.get(key)
                if isinstance(value, (list, tuple, dict)):
                    # A reference written as a list of sources, a kind as ["symmetry"], a check as its steps: one line
                    # of text, as written (a shape the plan refused before, so no frozen protocol holds it).
                    from .oracle_check import _text
                    item = {**item, key: _text(value)}
                elif value is not None and not isinstance(value, str):
                    return None, f"`protocol.oracles` entry {index}: `{key}` must be text"
            for key in ("expected", "tolerance"):
                if isinstance(item.get(key), (dict, list, bool)):
                    return None, f"`protocol.oracles` entry {index}: `{key}` must be a number (the engine judges the script's value against it)"
            # A `case`, `measure` or `order` that is not usable is left out (the oracle is then read from the script's
            # own oracle(), as before) instead of refusing a plan, or dropping every oracle of a draft, over one field.
            if not isinstance(item.get("case"), dict):
                item = {k: v for k, v in item.items() if k != "case"}
            if not isinstance(item.get("measure"), str):
                item = {k: v for k, v in item.items() if k != "measure"}
            if isinstance(item.get("order"), bool) or not isinstance(item.get("order"), (int, float)):
                item = {k: v for k, v in item.items() if k != "order"}
            mode = item.get("tolerance_mode")
            if mode is not None and str(mode).strip().lower() not in ("absolute", "relative"):
                return None, f"`protocol.oracles` entry {index}: `tolerance_mode` must be `absolute` or `relative`"
            fixed_oracles.append({**item, "name": str(item["name"]).strip()})
        out["oracles"] = fixed_oracles
    if out.get("acceptance") is not None:
        out["acceptance"] = _as_list(out["acceptance"])
    if out.get("criteria") is not None:
        # After the checks and the metrics: a criterion names a check, and may not name a headline number.
        from . import criteria as _criteria

        fixed_criteria, why = _criteria.normalize(out["criteria"], out)
        if fixed_criteria is None:
            return None, why
        out["criteria"] = fixed_criteria
    if out.get("optimisation") is not None:
        # A search for the best design (core/optimisation_plan.py). A coarse scan before the search is the block's own
        # `grid`: a protocol is either a sweep or a search, never both.
        from . import optimisation_plan

        if out.get("grid") is not None:
            return None, ("`protocol.optimisation` and a top-level `protocol.grid` cannot both be given: a search for the "
                          "best design puts a coarse scan run before it in `optimisation.grid`")
        block, why = optimisation_plan.normalize(out["optimisation"])
        if block is None:
            return None, why
        out["optimisation"] = block
    if "model" in out:
        model, why = normalize_model(out.pop("model"))
        if model is None:
            return None, why
        # Last, so a cut of the protocol (the claim check shows its first few thousand characters) keeps the grid and
        # the checks, which a long list of equations would push out.
        out["model"] = model
    return out, None


# A key of a mapping of equations by their ids: E1, e2, 3.
_EQ_KEY_RE = re.compile(r"^\s*E?\d+\s*$", re.IGNORECASE)


def _equations_by_id(equations: dict[Any, Any]) -> list[Any]:
    """A mapping of equation id to formula (or to the rest of the equation) as a list, each id written as E<n>."""
    out = []
    for key, value in equations.items():
        eid = str(key).strip().upper()
        eid = eid if eid.startswith("E") else f"E{eid}"
        out.append({**value, "id": eid} if isinstance(value, dict) else {"id": eid, "formula": value})
    return out


def _equations_from_text(equations: list[Any]) -> list[Any]:
    """``equations`` with each one written as a line of text ("E1: q = ...", or a formula alone) made a mapping with an
    ``id`` (the one it names, else the next free E<n>) and a ``formula``; the others as they were."""
    if not any(isinstance(item, str) for item in equations):
        return equations
    from .oracle_check import _EQ_ID_RE

    taken = {str(e.get("id")).strip().upper() for e in equations if isinstance(e, dict) and e.get("id") is not None}
    taken |= {m.group(1).upper() for m in (_EQ_ID_RE.match(e) for e in equations if isinstance(e, str)) if m}
    out: list[Any] = []
    for item in equations:
        if isinstance(item, str):
            if not item.strip():
                continue
            found = _EQ_ID_RE.match(item)
            if found:
                item = {"id": found.group(1).upper(), "formula": item[found.end():].strip()}
            else:
                n = 1
                while f"E{n}" in taken:
                    n += 1
                item = {"id": f"E{n}", "formula": item.strip()}
                taken.add(item["id"])
        out.append(item)
    return out

# The two things an equation of the model can be for, and the words a model writes for each.
ROLES = {"generates": "produces the data", "analyses": "analyses the results"}
_ROLE_WORDS = {
    "generates": "generates", "generate": "generates", "generating": "generates", "generation": "generates",
    "simulation": "generates", "simulates": "generates", "produces": "generates",
    "analyses": "analyses", "analyzes": "analyses", "analyse": "analyses", "analyze": "analyses",
    "analysis": "analyses", "analysing": "analyses", "analyzing": "analyses",
}


def normalize_model(model: Any) -> tuple[dict[str, Any] | None, str | None]:
    """``(model, None)`` when the plan's model block can be read, else ``(None, why)``.

    The model block (``protocol.model``) says what produces the numbers: ``summary`` (what model, in a sentence),
    ``assumptions``, ``holds_for`` (the range where it holds) and ``equations``, each with an ``id`` (E1, E2...), the
    ``formula`` as the source writes it, its ``role`` (``generates``: it produces the data; ``analyses``: it is used on
    the results) and its ``source`` (a source the quest retrieved, by its [n], or ``derivation`` with the steps in
    ``derivation``). Strict about form only: whether each source was really retrieved is for
    :func:`core.oracle_check.model_notes`, which needs the quest's sources."""
    if isinstance(model, str) and model.strip():
        # A sentence, as a plan written before the block had its parts could carry: kept as it was written (the frozen
        # protocol's hash depends on it), shown as the summary, and the plan notes that its equations are missing.
        return model, None
    if not isinstance(model, dict):
        return None, "`protocol.model` must be a mapping (`summary`, `assumptions`, `holds_for`, `equations`)"
    out: dict[str, Any] = dict(model)
    for key in ("summary", "holds_for"):
        if out.get(key) is not None and not isinstance(out[key], str):
            return None, f"`protocol.model.{key}` must be text"
        if isinstance(out.get(key), str):
            out[key] = out[key].strip()
    if out.get("assumptions") is not None:
        out["assumptions"] = _as_list(out["assumptions"])
    equations = out.get("equations")
    if equations is None:
        return out, None  # left out, not added: a block read back must hash as it was written
    # The shapes a model writes equations in besides a list of mappings, each refused before (so no frozen protocol holds
    # one) and read now instead of losing every equation: one text with a line per equation, a mapping of id to formula
    # ({"E1": "q = ...", "E2": "..."}), or a list of lines ("E1: q = ...").
    if isinstance(equations, str):
        from .oracle_check import _EQ_SPLIT_RE

        equations = [part for part in _EQ_SPLIT_RE.split(equations) if part.strip()]
    if isinstance(equations, dict) and equations and all(_EQ_KEY_RE.match(str(k)) for k in equations):
        equations = _equations_by_id(equations)
    if isinstance(equations, dict):
        equations = [equations]
    if not isinstance(equations, list):
        return None, "`protocol.model.equations` must be a list (each with an `id`, a `formula`, a `role` and a `source`)"
    equations = _equations_from_text(equations)
    fixed: list[dict[str, Any]] = []
    for index, item in enumerate(equations, start=1):
        if not isinstance(item, dict):
            return None, f"`protocol.model` equation {index} is not a mapping (`id`, `formula`, `role`, `source`)"
        eq = dict(item)
        if eq.get("formula") in (None, ""):
            # A formula written under another name (`equation`, `expression`): refused before, read now.
            other = next((k for k in ("equation", "expression", "eq", "latex") if isinstance(eq.get(k), str) and eq[k].strip()), None)
            if other:
                eq["formula"] = eq.pop(other)
        for key in ("id", "formula", "role", "source", "derivation"):
            value = eq.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int, float))):
                return None, f"`protocol.model` equation {index}: `{key}` must be text"
            if eq.get(key) is not None:
                eq[key] = str(eq[key]).strip()
        if not eq.get("id"):
            return None, f"`protocol.model` equation {index} has no `id` (E1, E2, ...)"
        if eq["id"].isdigit():
            eq["id"] = f"E{eq['id']}"  # a bare number is the equation's number: a check cites it as E<n>
        if not eq.get("formula"):
            return None, f"`protocol.model` equation {eq['id']} has no `formula`"
        role = str(eq.get("role") or "").strip().lower()
        if role:
            eq["role"] = _ROLE_WORDS.get(role, role)
        fixed.append(eq)
    ids = [eq["id"] for eq in fixed]
    twice = sorted({i for i in ids if ids.count(i) > 1})
    if twice:
        return None, f"`protocol.model` names equation {', '.join(twice)} more than once"
    out["equations"] = fixed
    return out, None


def repair_model(model: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """A drafted model block with each equation that cannot be read left out (and an equation with no ``id`` given the
    next free one), and a sentence per change, so the plan says what happened; ``(None, notes)`` when the block itself
    cannot be read."""
    if not isinstance(model, dict) or not isinstance(model.get("equations"), (list, dict)):
        fixed, why = normalize_model(model)
        return fixed, ([] if fixed is not None else [
            f"the model behind the numbers (`protocol.model`) was left out of the plan because it could not be read "
            f"({why}); write it here"
        ])
    equations = model["equations"]
    if isinstance(equations, dict) and equations and all(_EQ_KEY_RE.match(str(k)) for k in equations):
        equations = _equations_by_id(equations)
    equations = _equations_from_text(equations if isinstance(equations, list) else [equations])
    notes: list[str] = []
    taken = {str(e.get("id")).strip() for e in equations if isinstance(e, dict) and str(e.get("id") or "").strip()}
    kept: list[Any] = []
    for index, item in enumerate(equations, start=1):
        if isinstance(item, dict) and not str(item.get("id") or "").strip() and str(item.get("formula") or "").strip():
            n = 1
            while f"E{n}" in taken:
                n += 1
            item = {**item, "id": f"E{n}"}
            taken.add(f"E{n}")
            notes.append(f"equation {index} of the model had no id and is called E{n} in this plan")
        fixed, why = normalize_model({**model, "equations": [item]})
        label = str(item.get("id") or index).strip() if isinstance(item, dict) else str(index)
        if fixed is None:
            notes.append(f"equation {label} was left out of the model behind the numbers because it could not be read "
                         f"({why}); put it right here if it matters")
            continue
        if any(str(k.get("id")).strip() == label for k in kept):
            notes.append(f"a second equation called {label} was left out of the model behind the numbers; give it its "
                         "own id here if it matters")
            continue
        kept.append(item)
    fixed, why = normalize_model({**model, "equations": kept})
    if fixed is None:
        return None, notes + [f"the model behind the numbers (`protocol.model`) was left out ({why}); write it here"]
    return fixed, notes


_PROTOCOL_KEY = re.compile(r"`protocol\.([A-Za-z_]+)")


def repair_protocol(protocol: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """``(protocol, notes)``: ``protocol`` with each key that cannot be checked left out, and a sentence per key saying so.

    A model asked for a protocol writes what it means in the shape it likes (a real quest wrote the thresholds as prose
    where numbers were asked for), and one such key used to cost the whole plan: the design was refused and drafted again
    without a plan. A draft is repaired instead, key by key, and the plan says which keys were left out, so a person reads
    it and puts them right; a protocol that a person has edited in ``plan.md`` is still refused with the reason
    (:func:`normalize_protocol` is strict). ``(None, notes)`` when nothing usable is left."""
    if not isinstance(protocol, dict):
        return None, ["the protocol was not a mapping of names to values and was left out"]
    out = dict(protocol)
    notes: list[str] = []
    for _ in range(len(out) + 1):
        fixed, why = normalize_protocol(out)
        if fixed is not None:
            return fixed, notes
        match = _PROTOCOL_KEY.search(why or "")
        key = match.group(1) if match else ""
        if key not in out:
            return None, notes + [f"the protocol could not be repaired and was left out ({why})"]
        if key == "metrics":
            # One metric the check refuses is left out, not the list it is in.
            from . import metric_spec

            kept, dropped = metric_spec.repair(out["metrics"])
            if kept and dropped:
                out["metrics"] = kept
                notes.extend(
                    f"a `protocol.metrics` entry was left out of the plan because it could not be checked ({reason}); "
                    "put it right here if it matters"
                    for reason in dropped
                )
                continue
        if key == "thresholds" and isinstance(out["thresholds"], dict):
            # Likewise one threshold, not the block: a real plan listed a sweep of five cut-offs under one name beside
            # the numeric headline threshold, and lost both.
            kept_t = {name: v for name, v in out["thresholds"].items() if _number(v)}
            left_out = [name for name in out["thresholds"] if name not in kept_t]
            if kept_t and left_out:
                out["thresholds"] = kept_t
                notes.extend(
                    f"the threshold `{name}` was left out of `protocol.thresholds` because it is not one number "
                    f"({out_value!r}); put it right here if it matters"
                    for name, out_value in ((n, protocol["thresholds"][n]) for n in left_out)
                )
                continue
        if key == "criteria":
            # One criterion that cannot be computed is left out, not the list it is in.
            from . import criteria as _criteria

            kept_c, dropped_c = _criteria.repair(out["criteria"], {k: v for k, v in out.items() if k != "criteria"})
            notes.extend(dropped_c)
            if kept_c:
                out["criteria"] = kept_c
                continue
            del out[key]
            continue
        if key == "optimisation":
            # One part of a search for the best design that cannot be read is left out, not the block.
            from . import optimisation_plan

            if out.get("grid") is not None and "top-level `protocol.grid`" in (why or ""):
                block = out["optimisation"]
                if isinstance(block, dict) and block.get("grid") is None:
                    out["optimisation"] = {**block, "grid": out["grid"]}
                    notes.append("the sweep in `protocol.grid` was moved into `protocol.optimisation.grid`: a search for "
                                 "the best design runs it as a coarse scan first, then searches from its best point")
                else:
                    notes.append("`protocol.grid` was left out: a search for the best design has no separate sweep (its "
                                 "coarse scan is `protocol.optimisation.grid`)")
                del out["grid"]
                continue
            fixed_block, block_notes = optimisation_plan.repair(out["optimisation"])
            notes.extend(block_notes)
            if fixed_block is not None:
                out["optimisation"] = fixed_block
                continue
            del out[key]
            continue
        if key == "oracles" and isinstance(out["oracles"], list):
            # One check that cannot be read is left out, not every check of the plan: a real plan lost all three of its
            # checks over one field, and the engine then asked for new ones that said nothing of where their values came from.
            kept_o: list[Any] = []
            for index, item in enumerate(out["oracles"], start=1):
                one, why_one = normalize_protocol({"oracles": [item]})
                if one is not None:
                    kept_o.append(item)
                    continue
                label = (str(item.get("name") or "").strip() if isinstance(item, dict) else "") or f"number {index}"
                reason = (why_one or "").replace("`protocol.oracles` entry 1: ", "").replace("`protocol.oracles` entry 1 ", "")
                notes.append(f"the check {label!r} was left out of `protocol.oracles` because it could not be read "
                             f"({reason}); put it right here if it matters")
            if len(kept_o) < len(out["oracles"]):
                if kept_o:
                    out["oracles"] = kept_o
                else:
                    del out["oracles"]
                continue
        if key == "model":
            # One equation that cannot be read is left out, not the model it is part of.
            fixed_model, model_notes = repair_model(out["model"])
            notes.extend(model_notes)
            if fixed_model is not None:
                out["model"] = fixed_model
                continue
            del out[key]
            continue
        if key == "grid":
            notes.append(
                f"The settings to sweep (`protocol.grid`) could not be read ({why}), so the plan has NO settings to vary and the "
                "experiment would run one setting only. Write each parameter as a list of numbers, or a list of names such as "
                "[euler, rk4], in the protocol of this plan before the experiment runs."
            )
        else:
            notes.append(f"`protocol.{key}` was left out of the plan because it could not be checked ({why}); put it right here if it matters")
        del out[key]
    return None, notes


def _bullets(items: Any, empty: str) -> str:
    rows = _as_list(items)
    return "\n".join(f"- {row}" for row in rows) if rows else empty


def _literature_lines(items: Any) -> str:
    rows: list[str] = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                source = str(item.get("source") or item.get("title") or "").strip()
                says = str(item.get("says") or item.get("finding") or "").strip()
                if source and says:
                    rows.append(f"- **{source}**: {says}")
                elif source or says:
                    rows.append(f"- {source or says}")
            elif str(item).strip():
                rows.append(f"- {str(item).strip()}")
    elif isinstance(items, str) and items.strip():
        rows.append(items.strip())
    return "\n".join(rows) or "- (the model named no source that bears on this topic)"


MODEL_HEADING = "The model behind the numbers"
SOURCES_HEADING = "The sources this quest found"


def _sources_lines(sources: list[dict[str, str]] | None) -> list[str]:
    """The numbered list of the sources the quest retrieved, as a check or an equation cites them ([n]). Not read back."""
    from . import oracle_check

    rows = oracle_check.sources_block(sources or [])
    if not rows:
        return []
    return [f"## {SOURCES_HEADING}", "",
            "> A check's expected value or an equation of the model can cite one of these by its number ([2]); a source "
            "that is not listed here does not count.", "", *rows, ""]


def _flat(text: Any) -> str:
    """``text`` on one line with no backtick, so nothing in it can open a fence above the design block."""
    return " ".join(str(text if text is not None else "").replace("`", "'").split())


def _inline(text: Any) -> str:
    """``text`` on one line, as a code span (a formula's ``_`` and ``*`` are not emphasis)."""
    flat = _flat(text)
    return f"`{flat}`" if flat else ""


def _model_lines(protocol: Any) -> list[str]:
    """The readable form of the protocol's model and checks, for the section above the design block. Nothing here is
    read back: the block below is what runs, so an edit belongs there."""
    from . import oracle_check

    if not isinstance(protocol, dict):
        return []
    # Read as the engine reads it: a part written under another key (`description`, `governing_equations` ...) is shown.
    model = oracle_check.model_view(protocol.get("model"))
    oracles = oracle_check.declared(protocol)
    if model is None and not oracles:
        return []
    lines = [f"## {MODEL_HEADING}", "",
             f"> Shown from the design block below (`protocol.model` and `protocol.oracles`); edit it there.", ""]
    if model is None:
        lines += ["- (the plan does not say what model produces the numbers, what it assumes, or its equations: "
                  "add `model` to the protocol below)", ""]
    else:
        lines.append(f"**What produces the numbers:** {_flat(model.get('summary')) or '(not written)'}")
        lines.append("")
        assumptions = [_flat(a) for a in model.get("assumptions") or []]
        lines.append("**What it assumes:** " + ("; ".join(assumptions) if assumptions else "(not written)"))
        lines.append("")
        lines.append(f"**Where it holds:** {_flat(model.get('holds_for')) or '(not written)'}")
        lines.append("")
        equations = model.get("equations") or []
        lines.append("**Its equations:**")
        lines.append("")
        if not equations:
            lines.append("- (none written)")
        for eq in equations:
            role = ROLES.get(str(eq.get("role") or ""), f"role not given ({eq.get('role')})" if eq.get("role") else "role not given")
            source = _flat(eq.get("source")) or "no source given"
            if eq.get("derivation"):
                source += f": {_flat(eq['derivation'])}"
            lines.append(f"- **{eq.get('id')}** ({role}): {_inline(eq.get('formula'))}. Source: {source}")
        lines.append("")
    if oracles:
        lines.append("**The checks against known answers, and where each expected value comes from:**")
        lines.append("")
        for oracle in oracles:
            kind = oracle_check.KINDS.get(oracle_check.kind_of(oracle) or "", "")
            written = oracle_check.kind_written(oracle)
            kind = kind or (f"kind not recognised ({written})" if written else "kind not given")
            expected = oracle.get("expected")
            reference = _flat(oracle_check.reference_of(oracle)) or "(not said)"
            # How the number is computed: FI applies this formula to what the simulation returns on the check's case.
            case, measure = oracle.get("case"), oracle.get("measure")
            computed = (f"; computed as {_inline(measure)} from what the simulation returns on "
                        f"{_inline(', '.join(f'{k}={v}' for k, v in case.items()) or 'its default settings')}"
                        if isinstance(case, dict) and isinstance(measure, str) and measure.strip() else
                        "; its number is the script's own (it names no `case` and `measure` FI can run)")
            lines.append(f"- **{str(oracle['name']).strip()}**, {kind}: expects {expected if expected is not None else '(no value)'}"
                         f"{computed}; from: {reference}")
        lines.append("")
    return lines


CRITERIA_HEADING = "How we will judge whether the code got better"


def _criteria_lines(protocol: Any) -> list[str]:
    """The readable form of the protocol's criteria (``core/criteria.py``). Not read back: the block below is what runs."""
    from . import criteria as _criteria

    if not isinstance(protocol, dict):
        return []
    items = _criteria.declared(protocol)
    lines = [f"## {CRITERIA_HEADING}", "",
             "> Checks of correctness, never the study's own finding. FI computes each one itself after every run of the "
             "code and keeps the result in `.fi/criteria_history.jsonl`; they are fixed with the protocol before the "
             "first full run. Shown from `criteria` in the design block below; edit it there.",
             "> When one is not met after a run, FI changes the simulation one small step at a time (at most "
             "`engine.improve_rounds` times, 3 unless set) and keeps a change only when it makes a check better and "
             "none worse by more than that check's own tolerance; the study's own results never choose the version, "
             "and every round is in `code/CHANGELOG.md`.", ""]
    if not items:
        lines.append("- (none: FI records every run as having no criterion, so a later change to the code cannot be "
                     "shown to be better; add `criteria` to the protocol below)")
    lines += [f"- {_criteria.describe(c, protocol)}" for c in items]
    return lines + [""]


def render(topic: str, extra: dict[str, Any] | None, design: dict[str, Any], audit: list[str] | None = None,
           sources: list[dict[str, str]] | None = None, code_layout: list[str] | None = None,
           confirm: list[str] | None = None) -> str:
    """The text of ``plan.md``: the prose the model wrote around ``design``, and ``design`` itself in the block
    that is used as written. ``code_layout``: the lines saying how the code will be laid out and what that costs
    (``core/code_layout.py``). ``confirm``: the lines saying whether and how the result will be confirmed once more on
    data or seeds exploration never saw, and what that costs (``core/phased.py``)."""
    from . import optimisation_plan

    extra = extra if isinstance(extra, dict) else {}
    block = yaml.safe_dump(design, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100).rstrip()
    in_short = str(extra.get("in_short") or "").strip() or "(not written)"
    gap = str(extra.get("gap") or "").strip() or "(not written)"
    checks = _bullets(audit, "- (the methodology audit had nothing to add)")
    return "\n".join([
        f"# Plan: {topic.strip().splitlines()[0][:150] if topic.strip() else 'this quest'}",
        "",
        "> Written by FI from the topic and the literature it found, before any experiment is run. **Edit this file,",
        "> then resume the quest** (`--resume <quest_id>`), or ask for a change (`--revise-plan \"...\"`). The prose is for",
        f"> you and for the paper's methods; the block under *{DESIGN_HEADING}* is the design, exactly: what it says is",
        "> what runs. Every version is kept in `.fi/plan_versions/`.",
        "",
        "## In short",
        "",
        in_short,
        "",
        "## What the literature says",
        "",
        _literature_lines(extra.get("literature")),
        "",
        *_sources_lines(sources),
        "## The gap this experiment addresses",
        "",
        gap,
        "",
        *optimisation_plan.plan_lines(design),
        *_model_lines(design.get("protocol") if isinstance(design, dict) else None),
        *_criteria_lines(design.get("protocol") if isinstance(design, dict) else None),
        *(code_layout or []),
        *(confirm or []),
        "## Success criteria",
        "",
        _bullets(extra.get("success_criteria"), "- (not written)"),
        "",
        "## Risks",
        "",
        _bullets(extra.get("risks"), "- (not written)"),
        "",
        "## What this quest will not do",
        "",
        _bullets(extra.get("out_of_scope"), "- (not written)"),
        "",
        "## Checks already made",
        "",
        checks,
        "",
        f"## {DESIGN_HEADING}",
        "",
        "```yaml",
        block,
        "```",
        "",
    ])


class Parsed:
    def __init__(self, design: dict[str, Any] | None, error: str | None) -> None:
        self.design = design
        self.error = error


def parse(text: str) -> Parsed:
    """The design in a ``plan.md``: ``Parsed(design, None)`` or ``Parsed(None, why)``."""
    heading = _HEADING_RE.search(text or "")
    if heading is None:
        return Parsed(None, f"no `## {DESIGN_HEADING}` section was found")
    fenced = _FENCE_RE.search(text, heading.end())
    if fenced is None:
        return Parsed(None, f"the `{DESIGN_HEADING}` section has no fenced ```yaml block")
    try:
        loaded = yaml.safe_load(fenced.group(1))
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" (line {mark.line + 1} of the block)" if mark is not None else ""
        problem = str(getattr(e, "problem", "") or "invalid YAML").strip()
        return Parsed(None, f"the design block is not valid YAML{where}: {problem}")
    design, why = normalize_design(loaded)
    return Parsed(design, why)


def load_design(quest_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """The design in the quest's ``plan.md``. ``(None, None)`` when there is no plan file."""
    path = plan_path(quest_root)
    if not path.is_file():
        return None, None
    try:
        parsed = parse(path.read_text(encoding="utf-8"))
    except OSError as e:
        return None, f"plan.md could not be read: {e}"
    return parsed.design, parsed.error


def raw_design_block(text: str) -> dict[str, Any] | None:
    """The design block of ``text`` as written (not normalised), or ``None`` when it cannot be read."""
    heading = _HEADING_RE.search(text or "")
    fenced = _FENCE_RE.search(text, heading.end()) if heading else None
    if fenced is None:
        return None
    try:
        loaded = yaml.safe_load(fenced.group(1))
    except yaml.YAMLError:
        return None
    return loaded if isinstance(loaded, dict) else None


def edit_design_block(text: str, change: Any) -> str | None:
    """``text`` with its design block replaced by ``change(block)``, where ``block`` is the YAML of the block as written
    (not the normalised design, so no default key is added to what a person wrote); ``None`` when the block cannot be
    read or ``change`` returns ``None``. The block is written back as YAML (a comment in it is not kept; a block fenced
    as JSON is still read, YAML being a superset of JSON)."""
    heading = _HEADING_RE.search(text or "")
    fenced = _FENCE_RE.search(text, heading.end()) if heading else None
    loaded = raw_design_block(text)
    if fenced is None or loaded is None:
        return None
    changed = change(loaded)
    if not isinstance(changed, dict):
        return None
    block = yaml.safe_dump(changed, sort_keys=False, allow_unicode=True, default_flow_style=False, width=100).rstrip()
    return text[:fenced.start(1)] + block + text[fenced.end(1):]


def refresh_model_section(text: str) -> str:
    """``text`` with its *The model behind the numbers* section shown again from its design block, after the engine
    edited the block (the section is shown from it and never read back). Unchanged when either cannot be found."""
    parsed = parse(text)
    if parsed.design is None:
        return text
    found = re.search(rf"^##\s+{re.escape(MODEL_HEADING)}\s*$", text or "", re.MULTILINE)
    lines = _model_lines(parsed.design.get("protocol"))
    if not found:
        # Checks the plan had none of before: the section goes where render puts it (before the criteria, or the design).
        anchor = (re.search(rf"^##\s+{re.escape(CRITERIA_HEADING)}\s*$", text, re.MULTILINE)
                  or _HEADING_RE.search(text))
        if lines and anchor is not None:
            text = text[:anchor.start()] + "\n".join(lines).rstrip("\n") + "\n\n" + text[anchor.start():]
    else:
        if not lines:
            lines = [f"## {MODEL_HEADING}", "", "- (the plan no longer has a model or any check against a known answer)",
                     ""]
        after = re.search(r"^##\s+", text[found.end():], re.MULTILINE)
        end = found.end() + after.start() if after else len(text)
        text = text[:found.start()] + "\n".join(lines).rstrip("\n") + "\n\n" + text[end:]
    # The criteria are shown from the block too, and read the checks: shown again with them.
    criteria = re.search(rf"^##\s+{re.escape(CRITERIA_HEADING)}\s*$", text, re.MULTILINE)
    crit_lines = _criteria_lines(parsed.design.get("protocol"))
    if criteria and crit_lines:
        after = re.search(r"^##\s+", text[criteria.end():], re.MULTILINE)
        end = criteria.end() + after.start() if after else len(text)
        text = text[:criteria.start()] + "\n".join(crit_lines).rstrip("\n") + "\n\n" + text[end:]
    return text


def add_to_section(text: str, heading: str, lines: list[str]) -> str:
    """``text`` with ``lines`` added at the end of its ``## heading`` section, which is made (above the design block)
    when there is none. Prose only: nothing in it is read back."""
    lines = [line for line in lines if line is not None]
    if not lines:
        return text
    body = "\n".join(lines).rstrip() + "\n"
    found = re.search(rf"^##\s+{re.escape(heading)}\s*$", text or "", re.MULTILINE)
    if found:
        after = re.search(r"^##\s+", text[found.end():], re.MULTILINE)
        end = found.end() + after.start() if after else len(text)
        section = text[found.end():end].rstrip("\n")
        return text[:found.end()] + section + "\n" + body + "\n" + text[end:]
    design = _HEADING_RE.search(text or "")
    block = f"## {heading}\n\n{body}\n"
    if design is None:
        return (text.rstrip("\n") + "\n\n" + block) if text else block
    return text[:design.start()] + block + text[design.start():]


_SOURCE_LINE_RE = re.compile(r"^-\s*\[(W?\d+)\]\s*(.*?)(?:\s*\(DOI\s+(\S+)\))?\s*$", re.IGNORECASE)


def listed_sources(text: str) -> list[dict[str, str]]:
    """The sources a ``plan.md`` lists under *The sources this quest found* (FI writes that list), as the checks of where
    an expected value comes from match them: ``{label, title, doi, url}``. Empty when the plan lists none."""
    found = re.search(rf"^##\s+{re.escape(SOURCES_HEADING)}\s*$", text or "", re.MULTILINE)
    if not found:
        return []
    after = re.search(r"^##\s+", text[found.end():], re.MULTILINE)
    section = text[found.end():found.end() + after.start()] if after else text[found.end():]
    out = []
    for line in section.splitlines():
        m = _SOURCE_LINE_RE.match(line.strip())
        if m:
            out.append({"label": m.group(1).upper(), "title": " ".join(m.group(2).split()),
                        "doi": (m.group(3) or "").strip().lower(), "url": ""})
    return out


def strip_outer_fence(reply: str) -> str:
    """A model that was asked for the file itself sometimes wraps it in a fence."""
    match = _OUTER_FENCE_RE.match(reply or "")
    return (match.group(1) if match else (reply or "")).strip() + "\n"


# --- versions -----------------------------------------------------------------


def _history_path(quest_root: Path) -> Path:
    return quest_root / "needs" / "PLAN_HISTORY.json"


def history(quest_root: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(_history_path(quest_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def record_version(quest_root: Path, text: str, *, by: str, note: str = "") -> dict[str, Any]:
    """Keep this version of the plan (``.fi/plan_versions/plan.vN.md``) and list it in
    ``needs/PLAN_HISTORY.json``. ``by`` is ``"model"``, ``"request"``, ``"engine"`` (the oracle gate added
    or completed its checks) or ``"user"``."""
    rows = history(quest_root)
    entry = {
        "version": len(rows) + 1,
        "sha256": sha256(text),
        "by": by,
        "note": note,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    rows.append(entry)
    try:
        versions = quest_root / ".fi" / "plan_versions"
        versions.mkdir(parents=True, exist_ok=True)
        (versions / f"plan.v{entry['version']}.md").write_text(text, encoding="utf-8")
        _history_path(quest_root).parent.mkdir(parents=True, exist_ok=True)
        _history_path(quest_root).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass  # a history that cannot be written must never stop a quest
    return entry


def note_edit(quest_root: Path, text: str) -> dict[str, Any] | None:
    """When the plan on disk is not the version last recorded, a person edited it: keep that version too."""
    rows = history(quest_root)
    if rows and rows[-1].get("sha256") == sha256(text):
        return None
    return record_version(quest_root, text, by="user", note="edited on disk before the quest went on")
