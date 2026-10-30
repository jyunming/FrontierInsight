"""The study contract: one typed statement of what a study runs, returns, compares, draws and checks.

A plan today is free-form YAML, and many checks match names between the plan, the code and the results after the run
to find out what the study meant. The contract makes those facts declared and typed instead, so FI reads them from one
place: the settings and their values, every result the simulation returns with its *role*, the comparisons and derived
quantities FI computes, the figures and what each shows, the checks, the run budget, and free-text notes that no check
reads.

Roles of a result (:data:`ROLES`):

* ``result``: a measured quantity the study reports.
* ``reference``: a theory or reference curve computed alongside, to compare a result with.
* ``timing``: a time the study measures as a result. It is not used to judge whether a setting changed anything or
  whether a rerun gives the same numbers; FI's own measured time of every trial is shown beside it.
* ``physical_bound``: a result that by its physics may sit exactly on the edge of its declared ``range`` (a value that
  is 0 or 1 by construction): sitting there is not a fault. For any other role a value on the edge is said, and a value
  outside the range is a fault.
* ``diagnostic``: everything the simulation returns that the contract does not declare, and a declared result that is
  the measure of a check of a violation kind (an invariant, a symmetry or a second implementation: FI measures that
  check itself and the quantity is a size of disagreement, not a finding). Kept in the record, readable by the study's
  own derive functions, never reported and never checked.

Where a plan gives no ``role``, :func:`from_design` sets one from the plan's structure alone: a result with a
``reference.formula`` is a ``reference``; a result that is the ``measure`` (or ``metric``) of a check of a violation
kind is a ``diagnostic`` (the quantity of a special case, a published value or a convergence rate is the result itself,
measured on a small case, so it stays a ``result``); any other is a ``result``, said in ``gaps``. ``measured_by`` names
every check that measures a result, whatever its kind. No word of an estimand or a label is read.

This is not :class:`core.protocol.ResearchProtocol`, which says what *kind* of quest this is (a simulation, a literature
review); the contract says what one simulation study does.

:func:`from_design` reads a plan's design block (as :func:`core.plan.parse` gives it) into a contract by fixed rules,
with no model. A part it cannot place by those rules is left out and named in ``gaps`` (for the paper to say), never
guessed: an old plan that names its results only in prose, a precision target that is a sentence, or a figure given only
by its file name.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .oracle_forms import VIOLATION_KINDS

ROLES = ("result", "reference", "timing", "physical_bound", "diagnostic")
Role = Literal["result", "reference", "timing", "physical_bound", "diagnostic"]
#: The most panels one figure may have.
MAX_PANELS = 3


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> Any:
        """A copy with ``update`` applied and checked again (pydantic's own copy skips the checks)."""
        data = {**dict(self), **(update or {})}
        return type(self).model_validate(copy.deepcopy(data) if deep else data)


class Range(_Strict):
    """The values a result may take; outside is a fault (see the module docstring for a value on the edge)."""

    min: float | None = None
    max: float | None = None
    reason: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> "Range":
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"the range's min {self.min} is above its max {self.max}")
        return self


class Result(_Strict):
    """One quantity the simulation returns for each trial."""

    id: str
    role: Role = "result"
    #: The names of the known-answer checks whose measure is this result (whatever its role): FI's own measurement of
    #: each is in its record, beside what the simulation returned.
    measured_by: tuple[str, ...] = ()
    kind: Literal["mean", "proportion"] = "mean"
    estimand: str = ""
    observation: str = ""  # what one independent observation is (one trial, one cluster)
    unit: str = ""  # the physical unit of the value, if any
    label: str = ""
    depends_on: tuple[str, ...] | None = None  # the settings it should change with; None: not declared
    #: ``each_run``: every trial returns it; ``afterwards``: the study computes it from the trials; None: not declared.
    returned: Literal["each_run", "afterwards"] | None = None
    range: Range | None = None
    known_value: dict[str, Any] | None = None
    reference: dict[str, Any] | None = None
    given: str | None = None  # a proportion's id: this result is a mean over the trials where that one holds
    cluster: Any = None
    paired: bool = False
    family: str = ""
    undefined_when: dict[str, Any] = Field(default_factory=dict)


class Comparison(_Strict):
    """A difference or ratio of one result between values of one setting; FI computes it and its interval (paired when
    the trials share their seeds)."""

    id: str
    result: str
    across: str  # a compared setting or a value compared inside each run
    values: tuple[Any, ...] = ()  # the values compared; empty: every value against the first
    how: Literal["difference", "ratio"] = "difference"
    asked: bool = False  # the model's answer to FI's question after the plan (core/contract_parts.py): read by no check


class Derived(_Strict):
    """A quantity computed from the results of a setting: by FI from ``formula`` (FI's formula language) when one is
    given, else by the study's own ``derive_<id>`` function, which sees every returned value (diagnostics too)."""

    id: str
    inputs: tuple[str, ...] = ()
    formula: str | None = None
    role: Role = "result"
    kind: Literal["mean", "proportion"] = "mean"
    label: str = ""
    asked: bool = False  # the model's answer to FI's question after the plan (core/contract_parts.py): read by no check
    # the plan's own row, when FI asked about its formula (core/contract_parts.py): read by no check; what the audit judged
    as_planned: dict[str, Any] | None = None


class Figure(_Strict):
    """One figure: what it shows, along which setting and grouped by which; FI writes its caption from this. A figure an
    older plan gives only by file name has ``declared`` false and nothing else."""

    id: str
    file: str
    shows: tuple[str, ...] = ()
    x: str | None = None
    series: str | None = None
    panels: int = Field(default=1, ge=1, le=MAX_PANELS)
    declared: bool = True


def figure_file(planned: Any) -> str:
    """The file name of a planned figure, the one name every part of FI matches a saved figure by: the plan gives a figure as
    ``fig1.png`` or as ``figures/fig1.png (two panels: ...)``, and a figure is saved in the figures folder under its base name."""
    text = str(planned.get("file") or "") if isinstance(planned, dict) else str(planned or "")
    head = text.split("(")[0].strip()
    name = head if re.search(r"\.\w{2,4}$", head) else (text.split() or [""])[0].split("(")[0]
    return name.strip().strip(",;:").replace("\\", "/").rsplit("/", 1)[-1]


class Precision(_Strict):
    target: str  # a result or comparison id
    half_width: float = Field(gt=0)
    reason: str = ""


class Budget(_Strict):
    runs_per_setting: int | None = Field(default=None, ge=1)
    min_runs_per_setting: int | None = Field(default=None, ge=1)
    precision: Precision | None = None


class Settings(_Strict):
    compared: dict[str, tuple[Any, ...]] = Field(default_factory=dict)
    sub_studies: tuple[dict[str, Any], ...] = ()
    in_each_run: dict[str, tuple[Any, ...]] = Field(default_factory=dict)
    numerical: dict[str, Any] = Field(default_factory=dict)
    fixed: dict[str, Any] = Field(default_factory=dict)  # constants and thresholds

    def names(self) -> set[str]:
        return set(self.compared) | set(self.in_each_run) | set(self.numerical) | set(self.fixed)

    def varied(self) -> set[str]:
        """The settings whose values change between the runs of the study (a constant does not)."""
        return set(self.compared) | set(self.in_each_run)


class Checks(_Strict):
    oracles: tuple[dict[str, Any], ...] = ()
    criteria: tuple[dict[str, Any], ...] = ()
    assertions: tuple[dict[str, Any], ...] = ()  # result bounds on a path that is not a declared result
    topic_map: tuple[dict[str, Any], ...] = ()


class StudyContract(_Strict):
    #: ``contract``: a plan written in this form (``protocol.form: contract``), whose results are what the simulation
    #: returns for each trial and whose derived quantities are declared apart. ``older``: a plan from before, whose
    #: results mix values returned per trial with quantities its analysis computed afterwards (measured 2026-10-27: 315
    #: of 472 declared results in 51 recorded quests were returned per trial), so a result missing from a trial does not
    #: make that trial a failure.
    form: Literal["contract", "older"] = "contract"
    settings: Settings = Field(default_factory=Settings)
    results: tuple[Result, ...] = ()
    comparisons: tuple[Comparison, ...] = ()
    derived: tuple[Derived, ...] = ()
    figures: tuple[Figure, ...] = ()
    checks: Checks = Field(default_factory=Checks)
    model: dict[str, Any] = Field(default_factory=dict)
    budget: Budget = Field(default_factory=Budget)
    search: dict[str, Any] | None = None  # a find-the-best-design study's search block, as the plan gives it
    shared_draws: bool = False  # trial t uses the same seed at every setting, so comparisons are paired
    labels: dict[str, str] = Field(default_factory=dict)
    notes: dict[str, Any] = Field(default_factory=dict)  # free text: shown to the models, read by no check
    gaps: tuple[str, ...] = ()  # what the plan gave that could not be placed, for the paper to say

    @model_validator(mode="after")
    def _references(self) -> "StudyContract":
        ids = [r.id for r in self.results] + [d.id for d in self.derived] + [c.id for c in self.comparisons]
        twice = sorted({i for i in ids if ids.count(i) > 1})
        if twice:
            raise ValueError("these ids are given more than once: " + ", ".join(twice))
        values = {r.id for r in self.results} | {d.id for d in self.derived}
        settings = self.settings.varied()
        for c in self.comparisons:
            if c.result not in values:
                raise ValueError(f"comparison `{c.id}` compares `{c.result}`, which is not a declared result")
            if c.across not in settings:
                raise ValueError(f"comparison `{c.id}` is across `{c.across}`, which is not a compared setting")
        for d in self.derived:
            missing = [i for i in d.inputs if i not in values]
            if missing:
                raise ValueError(f"derived `{d.id}` uses {', '.join(missing)}, which are not declared results")
        for r in self.results:
            unknown = [s for s in (r.depends_on or ()) if s not in settings]
            if unknown:
                raise ValueError(f"result `{r.id}` depends on {', '.join(unknown)}, which are not varied settings")
            if r.given is not None and r.given not in values:
                raise ValueError(f"result `{r.id}` is given `{r.given}`, which is not a declared result")
        for f in self.figures:
            unknown = [s for s in f.shows if s not in set(ids)]
            if unknown:
                raise ValueError(f"figure `{f.id}` shows {', '.join(unknown)}, which are not declared")
            for axis in (f.x, f.series):
                if axis is not None and axis not in settings:
                    raise ValueError(f"figure `{f.id}` uses `{axis}`, which is not a varied setting")
        p = self.budget.precision
        if p is not None and p.target not in set(ids):
            raise ValueError(f"the precision target `{p.target}` is not a declared result or comparison")
        return self

    def result(self, name: str) -> Result | None:
        return next((r for r in self.results if r.id == name), None)

    def role_of(self, name: str) -> str:
        """The role of a returned name: its declared role, else ``diagnostic``."""
        r = self.result(name)
        return r.role if r is not None else "diagnostic"


# --- reading a plan ------------------------------------------------------------------------------------------------------

_NOTE_KEYS = ("seed_policy", "ci_method", "failure_policy", "acceptance", "simplifications", "definitions", "named_results")
_DESIGN_NOTES = ("hypothesis", "method", "expected_outcome", "rationale", "variables")
#: Protocol keys the reader places; any other key goes to the notes and is named in the gaps.
_PLACED = {"grid", "sub_studies", "in_each_run", "numerical_settings", "constants", "thresholds", "runs_per_setting",
           "min_runs_per_setting", "precision", "metrics", "oracles", "criteria", "topic_map", "optimisation", "model",
           "labels", "comparisons", "derived", "figures", "steps", "shared_draws", "form", "parts_left_out", "names_not_asked", "figure_declarations", *_NOTE_KEYS}


def oracle_measures(protocol: Any) -> tuple[dict[str, list[str]], set[str]]:
    """The names of the plan's checks by the id each measures (its ``measure`` or ``metric``), and the ids measured by a
    check of a violation kind (``oracle_forms.VIOLATION_KINDS``). Reads the ``oracles`` as written, with no validation."""
    by_id: dict[str, list[str]] = {}
    violation: set[str] = set()
    oracles = protocol.get("oracles") if isinstance(protocol, dict) else None
    for o in oracles if isinstance(oracles, list) else []:
        if not isinstance(o, dict):
            continue
        for key in ("measure", "metric"):
            target, name = _ident(o.get(key)), _ident(o.get("name"))
            if target is None:
                continue
            if name is not None and name not in by_id.setdefault(target, []):
                by_id[target].append(name)
            if o.get("kind") in VIOLATION_KINDS:
                violation.add(target)
    return by_id, violation


def structural_role(metric: dict[str, Any], violation_ids: set[str]) -> str | None:
    """The role the plan's structure gives a declared result that writes none: ``reference`` for a reference formula,
    ``diagnostic`` for the measure of a check of a violation kind; ``None`` when the structure says nothing."""
    ref = metric.get("reference")
    if isinstance(ref, dict) and str(ref.get("formula") or "").strip():
        return "reference"
    return "diagnostic" if str(metric.get("id")) in violation_ids else None


def raw_roles(protocol: Any) -> dict[str, str]:
    """The role of every declared result of a protocol, read without validating the entries: the role written, else the
    structural one, else ``result``. An entry the contract would leave out (a kind or a reference it cannot place) keeps
    the role it would have, so a check that must count it still does."""
    metrics = protocol.get("metrics") if isinstance(protocol, dict) else None
    _by_id, violation = oracle_measures(protocol)
    out: dict[str, str] = {}
    for m in metrics if isinstance(metrics, list) else []:
        if not isinstance(m, dict) or not m.get("id"):
            continue
        written = m.get("role")
        out.setdefault(str(m["id"]), written if written in ROLES else structural_role(m, violation) or "result")
    return out


def is_routine_gap(gap: str) -> bool:
    """A gap every older plan has today (it names no role, it is in the older form): kept in ``.fi/contract.json``, not
    shown in the run record."""
    return gap.startswith(("the plan gives no role", "an older plan"))


def _values(v: Any) -> tuple[Any, ...]:
    return tuple(v) if isinstance(v, (list, tuple)) else (v,)


def _mapping(v: Any, gaps: list[str] | None = None, what: str = "") -> dict[str, Any]:
    """``v`` as a table; a value that is there but not a table is left out (said in ``gaps`` when given)."""
    if isinstance(v, dict):
        return {str(k): x for k, x in v.items()}
    if v is not None and gaps is not None:
        gaps.append(f"{what} is not a table, so it is left out")
    return {}


def _rows(v: Any, gaps: list[str] | None = None, what: str = "") -> tuple[dict[str, Any], ...]:
    """The table entries of the list ``v``; what is not a list, or an entry that is not a table, is left out (said)."""
    if v is None:
        return ()
    if not isinstance(v, list):
        if gaps is not None:
            gaps.append(f"{what} is not a list, so it is left out")
        return ()
    rows = tuple(x for x in v if isinstance(x, dict))
    lost = len(v) - len(rows)
    if lost and gaps is not None:
        gaps.append(f"{lost} entr{'y' if lost == 1 else 'ies'} of {what} are not tables, so they are left out")
    return rows


def _ident(v: Any) -> str | None:
    """An id as a plan must give it: a non-empty text. A number or nothing is not an id."""
    return v.strip() if isinstance(v, str) and v.strip() else None


def _first_valid(cls: type[BaseModel], data: dict[str, Any], gaps: list[str], what: str) -> Any:
    try:
        return cls.model_validate(data)
    except ValueError as e:
        gaps.append(f"{what} could not be read ({_first_error(e)}); it is left out")
        return None


def _first_error(e: ValueError) -> str:
    errors = getattr(e, "errors", None)
    if callable(errors):
        first = errors()[0]
        return f"{'.'.join(str(x) for x in first.get('loc', ()))}: {first.get('msg')}"
    return str(e)


def _named_id(text: Any, ids: set[str]) -> str | None:
    """The declared id an older plan's precision target names: the text itself, or the longest id the text starts with
    when a space or a bracket follows it (``"rate_ratio (at the middle setting)"``). Anything else names none."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text in ids:
        return text
    named = [i for i in ids if text.startswith(i) and text[len(i):len(i) + 1] in (" ", "(", "[")]
    return max(named, key=len) if named else None


def from_design(design: dict[str, Any] | None, *, fi_analysis: bool = False,
                left_out: dict[str, str] | None = None) -> StudyContract:
    """The contract a plan's design block gives, by fixed rules (no model, no guess); see the module docstring.

    ``fi_analysis``: FI works out this plan's derived quantities itself (core/analysis_frame.py), so a formula it cannot work
    out as written is left out, with a gap that says why; without it the model's own analysis computes them, and every
    formula is kept as the plan writes it. ``left_out``, when given, is filled with the id of each such formula and why (not
    one left out only because it names another that is)."""
    design = design if isinstance(design, dict) else {}
    gaps: list[str] = []
    p = _mapping(design.get("protocol"), gaps, "the protocol")
    labels = {str(k): str(v) for k, v in _mapping(p.get("labels"), gaps, "the labels").items()}

    fixed = {**_mapping(p.get("constants"), gaps, "the constants"),
             **_mapping(p.get("thresholds"), gaps, "the thresholds")}
    settings = _first_valid(Settings, {
        "compared": {k: _values(v) for k, v in _mapping(p.get("grid"), gaps, "the grid").items()},
        "sub_studies": _rows(p.get("sub_studies"), gaps, "the sub-studies"),
        "in_each_run": {k: _values(v) for k, v in _mapping(p.get("in_each_run"), gaps, "in_each_run").items()},
        "numerical": _mapping(p.get("numerical_settings"), gaps, "the numerical settings"), "fixed": fixed},
        gaps, "the settings") or Settings()

    metrics = _rows(p.get("metrics"), gaps, "the metrics")
    ranges = {}
    assertions = []
    declared_ids = {_ident(m.get("id")) for m in metrics} - {None}
    derived_rows = _rows(p.get("derived"), gaps, "the derived quantities")
    derived_ids = {_ident(d.get("id")) for d in derived_rows} - {None}
    for a in _rows(design.get("result_assertions"), gaps, "the result bounds"):
        if str(a.get("path")) in declared_ids:
            ranges[str(a.get("path"))] = {"min": a.get("min"), "max": a.get("max"), "reason": str(a.get("reason") or "")}
        else:
            assertions.append(a)

    measures, violation_ids = oracle_measures(p)

    results = []
    no_role: list[str] = []
    for m in metrics:
        rid = _ident(m.get("id"))
        if rid is None:
            gaps.append(f"a result is declared with the id {m.get('id')!r}, which is not a name, so it is left out")
            continue
        if rid in {r.id for r in results}:
            gaps.append(f"result `{rid}` is declared more than once; only the first is kept")
            continue
        structural = structural_role({**m, "id": rid}, violation_ids)
        if not m.get("role") and structural is None:
            no_role.append(rid)
        if not m.get("role") and structural == "diagnostic":
            gaps.append(f"result `{rid}` is the measure of the check {', '.join('`' + n + '`' for n in measures[rid])}, "
                        "so a non-finite value of it is disclosed, not stopped")
        depends = m.get("depends_on")
        if isinstance(depends, list):
            unknown = [str(s) for s in depends if str(s) not in settings.varied()]
            if unknown:
                gaps.append(f"result `{rid}` is said to depend on {', '.join(unknown)}, which are not varied "
                            "settings; that is left out")
            depends = [str(s) for s in depends if str(s) in settings.varied()]
        given = m.get("given")
        if given is not None and _ident(given) not in declared_ids | derived_ids:
            gaps.append(f"result `{rid}` is given `{given}`, which is not a declared result; that is left out")
            given = None
        data = {"id": rid, "role": m.get("role") or structural or "result", "measured_by": tuple(measures.get(rid, ())),
                "kind": m.get("kind") or "mean",
                "estimand": str(m.get("estimand") or ""), "observation": str(m.get("unit") or ""),
                "unit": str(m.get("physical_unit") or ""), "label": str(m.get("label") or labels.get(rid, "")),
                "depends_on": depends, "returned": m.get("returned") if m.get("returned") in ("each_run", "afterwards") else None,
                "range": ranges.get(rid) or m.get("range"),
                "known_value": m.get("known_value"), "reference": m.get("reference"), "given": _ident(given),
                "cluster": m.get("cluster"), "paired": bool(m.get("paired")), "family": str(m.get("family") or ""),
                "undefined_when": _mapping(m.get("undefined_when"))}
        r = _first_valid(Result, {k: v for k, v in data.items() if v is not None}, gaps, f"result `{rid}`")
        if r is not None:
            results.append(r)
    if results and len(no_role) == len(results):
        gaps.append("the plan gives no role for its results; each is read as a measured result")
    elif no_role:
        gaps.append("the plan gives no role for " + ", ".join(f"`{i}`" for i in no_role)
                    + "; each is read as a measured result")

    ids = {r.id for r in results}
    derived = [d for d in (_first_valid(Derived, x, gaps, f"derived `{x.get('id')}`") for x in derived_rows)
               if d is not None]
    # what a formula may name when FI works it out (core/analysis_frame.py): a result, a derived quantity, a setting of the
    # grid, a constant or threshold, the plan's fixed settings and a numerical setting's one value; never a value compared
    # inside a run, which the analysis is not given
    if fi_analysis:
        derived = _computable(derived, ids | set(settings.compared) | set(settings.fixed), p, gaps,
                              not_given=set(settings.numerical) | set(settings.in_each_run),
                              left_out=left_out if left_out is not None else {})
    ids |= {d.id for d in derived}
    comparisons = [c for c in (_first_valid(Comparison, x, gaps, f"comparison `{x.get('id')}`")
                               for x in _rows(p.get("comparisons"), gaps, "the comparisons")) if c is not None]

    figures = []
    planned = p.get("figures") if "figures" in p else design.get("figures_planned")
    if planned is not None and not isinstance(planned, list):
        gaps.append("the figures are not a list, so they are left out")
    # What the plan's model said each figure shows (core/contract_parts.py) is kept by file name beside the plan's own list,
    # which it never replaces: a figure added or removed in the plan is the one drawn.
    said = p.get("figure_declarations") if isinstance(p.get("figure_declarations"), dict) else {}
    for i, f in enumerate(planned if isinstance(planned, list) else [], 1):
        name = figure_file(f)
        decl = said.get(name) if name else None
        if isinstance(decl, dict) and decl.get("drop") is True:
            continue  # the plan's model said it cannot be drawn from the study's numbers
        own = None
        if isinstance(f, dict):
            try:
                own = Figure(**{"id": f"fig{i}", **f})
            except ValueError:
                own = None
        if own is not None:
            fig = own
        elif isinstance(decl, dict):
            fig = _first_valid(Figure, {"id": f"fig{i}", **{k: v for k, v in decl.items() if k in ("shows", "x", "series", "panels")},
                                       "file": name}, gaps, f"figure {i}")
        elif isinstance(f, dict):
            fig = _first_valid(Figure, {"id": f"fig{i}", **f}, gaps, f"figure {i}")
        else:
            fig = Figure(id=f"fig{i}", file=str(f), declared=False)
        if fig is not None:
            figures.append(fig)
    if any(not f.declared for f in figures):
        gaps.append("the plan names " + str(sum(not f.declared for f in figures))
                    + " figure(s) only by file name, so what they show is not declared")

    budget = _first_valid(Budget, {k: v for k, v in {"runs_per_setting": p.get("runs_per_setting"),
                                                     "min_runs_per_setting": p.get("min_runs_per_setting")}.items()
                                   if v is not None}, gaps, "the run budget") or Budget()
    precision = p.get("precision")
    if isinstance(precision, dict):
        target = _named_id(precision.get("target", precision.get("metric")), ids | {c.id for c in comparisons})
        if target is not None:
            kept = _first_valid(Precision, {"target": target, "half_width": precision.get(
                "target_half_width", precision.get("half_width")), "reason": str(precision.get("reason") or "")},
                gaps, "the precision target")
            budget = budget.model_copy(update={"precision": kept})
        else:
            gaps.append("the precision target does not name a declared result, so the study has no precision target")

    notes = {k: p[k] for k in _NOTE_KEYS if k in p}
    notes.update({k: design[k] for k in _DESIGN_NOTES if k in design})
    notes.update({k: p[k] for k in p if k.endswith("_note")})
    other = sorted(k for k in p if k not in _PLACED and not k.endswith("_note"))
    if other:
        notes.update({k: p[k] for k in other})
        gaps.append("these parts of the protocol are kept only as notes: " + ", ".join(other))

    form = "contract" if p.get("form") == "contract" else "older"
    unasked = p.get("names_not_asked")
    if isinstance(unasked, list) and unasked:
        gaps.append("the plan uses more names than were asked about; not asked: " + ", ".join(str(x) for x in unasked))
    left_out = p.get("parts_left_out")
    if isinstance(left_out, list) and left_out:
        gaps.append("the plan's model did not give, in a form that could be used, " + ", ".join(str(x) for x in left_out)
                    + "; the plan stays in the older form")
    if form == "older":
        gaps.append("an older plan: its results may be computed by its analysis, not returned by each trial")
    base = {"form": form, "settings": settings, "budget": budget, "labels": labels, "notes": notes,
            "checks": Checks(oracles=_rows(p.get("oracles"), gaps, "the known-answer checks"),
                             criteria=_rows(p.get("criteria"), gaps, "the criteria"),
                             assertions=tuple(assertions), topic_map=_rows(p.get("topic_map"), gaps, "the topic map")),
            "model": _mapping(p.get("model")) if isinstance(p.get("model"), dict) else
            ({"summary": p["model"]} if isinstance(p.get("model"), str) else {}),
            "search": _mapping(p.get("optimisation")) or None, "shared_draws": p.get("shared_draws") is True}
    if isinstance(p.get("steps"), list) and "steps" not in base["model"]:
        base["model"] = {**base["model"], "steps": p["steps"]}  # steps written beside the model belong to it
    parts = {"results": tuple(results), "comparisons": tuple(comparisons), "derived": tuple(derived),
             "figures": tuple(figures)}
    try:
        return StudyContract(**base, **parts, gaps=tuple(gaps))
    except ValueError as e:
        # A reference between parts that does not hold (a comparison of an undeclared result, a figure along a name
        # that is not a setting): drop the part that holds it, say so, and keep the rest.
        return _without_broken_links(base, parts, gaps, e)


_ONE = {"derived": "derived", "comparisons": "comparison", "figures": "figure"}


def _computable(derived: list[Derived], known: set[str], protocol: dict[str, Any], gaps: list[str],
                not_given: set[str] = frozenset(), left_out: dict[str, str] | None = None) -> list[Derived]:
    """The derived quantities FI can work out as the plan writes them: a formula that cannot be read, that summarises one
    number (``max(growth)``: the analysis has one number of each result at a setting), that names what the plan does not
    declare (a result, a derived quantity, a setting, a constant or threshold), or that names other formulas round in a
    circle is left out, with a gap that says why, when the plan is read, and never found only after the long run."""
    from . import oracle_forms as _forms
    from . import reference_values as _reference_values

    functions = {**_forms.FUNCTIONS, **_forms.ANALYSIS_SPECIAL}
    known = set(known) | set(_reference_values.analysis_inputs(protocol))
    left_out = left_out if left_out is not None else {}
    kept = list(derived)
    while True:
        kept, circle = _without_circles(_with_known_names(kept, known, functions, gaps, set(not_given) - known, left_out),
                                        functions)
        if not circle:
            return kept
        why = "its formula names others in a circle, so none of them can be worked out first"
        left_out.update({i: why for i in circle})
        gaps.append("derived " + ", ".join(f"`{i}`" for i in circle) + " are left out: their formulas name each other in a "
                    "circle, so none of them can be worked out first")


def _with_known_names(kept: list[Derived], known: set[str], functions: dict[str, Any], gaps: list[str],
                      not_given: set[str], left_out: dict[str, str]) -> list[Derived]:
    from . import oracle_forms as _forms

    kept, first = list(kept), {d.id for d in kept}
    while True:
        ids = {d.id for d in kept}
        dropped = False
        for d in list(kept):
            if not d.formula:
                continue
            # read as FI works it out
            why = (_forms.one_number_summary(d.formula)
                   or _forms.evaluate(d.formula, {}, special=_forms.ANALYSIS_SPECIAL).unreadable or None)
            itself = False
            if why is None:
                unknown = [n for n in _forms.names(d.formula, functions) if n not in known and n not in ids]
                itself = bool(unknown) and set(unknown) <= first
                why = ("it uses " + ", ".join(f"`{n}`" for n in unknown)
                       + (", which is itself left out" if itself else
                          ", a value compared inside each run or a numerical setting with no one value, which the analysis "
                          "is not given" if set(unknown) <= not_given else ", which the plan does not declare")
                       if unknown else None)
            if why is not None:
                gaps.append(f"derived `{d.id}` is left out: its formula ({d.formula}) cannot be worked out as written ({why})")
                if not itself:
                    left_out[d.id] = why
                kept.remove(d)
                dropped = True
        if not dropped:
            return kept


def _without_circles(kept: list[Derived], functions: dict[str, Any]) -> tuple[list[Derived], list[str]]:
    """``(kept, circle)``: the formulas that name each other round in a circle (none of them can be worked out first) taken
    out."""
    from . import oracle_forms as _forms

    formulas = {d.id: set(_forms.names(d.formula, functions)) for d in kept if d.formula}
    done = {d.id for d in kept if not d.formula}
    while True:
        ready = [i for i, uses in formulas.items() if i not in done and uses & set(formulas) <= done]
        if not ready:
            break
        done.update(ready)
    circle = sorted(i for i in formulas if i not in done)
    return [d for d in kept if d.id not in circle], circle


def _without_broken_links(base: dict[str, Any], parts: dict[str, Any], gaps: list[str], error: ValueError) -> StudyContract:
    gaps = [*gaps, f"a part of the plan refers to something it does not declare ({_first_error(error)})"]
    budget = base["budget"]
    base = {**base, "budget": budget.model_copy(update={"precision": None})}  # put back last: it may name any part
    # The parts are added back one at a time, each kind after the kinds it may refer to, so one broken item does not
    # take the good ones with it.
    built = {**parts, "comparisons": (), "derived": (), "figures": ()}
    for key in ("derived", "comparisons", "figures"):
        waiting = list(range(len(parts[key])))  # by position: two equal entries are two entries
        while waiting:  # again until nothing more fits: a derived quantity may use one listed after it
            kept = []
            for n in waiting:
                trial = {**built, key: (*built[key], parts[key][n])}
                try:
                    StudyContract(**base, **trial)
                    built = trial
                    kept.append(n)
                except ValueError:
                    pass
            if not kept:
                break
            waiting = [n for n in waiting if n not in kept]
        gaps.extend(f"{_ONE[key]} `{parts[key][n].id}` is left out: it refers to something the plan does not declare"
                    for n in waiting)
    try:
        return StudyContract(**{**base, "budget": budget}, **built, gaps=tuple(gaps))
    except ValueError:
        gaps.append("the precision target is left out: it refers to something the plan does not declare")
        return StudyContract(**base, **built, gaps=tuple(gaps))


# --- asking the plan for the parts the contract needs ----------------------------------------------------------------------

#: The four answers for a name the plan relies on but does not declare as a result.
NAME_ANSWERS = ("result", "reference", "diagnostic", "derived")
#: The most undeclared names asked about in one question.
MAX_NAMES = 15
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KNOWN_WORDS = {"pi", "e", "inf", "nan", "true", "false", "none", "and", "or", "not", "if", "else", "for", "in"}


def varied_settings(protocol: Any) -> list[str]:
    """The settings whose values change between the runs of a study: the grid and what each run goes through."""
    p = protocol if isinstance(protocol, dict) else {}
    out: list[str] = []
    for key in ("grid", "in_each_run"):
        block = p.get(key)
        out += [str(k) for k in block if str(k) not in out] if isinstance(block, dict) else []
    return out


def _declared_names(protocol: dict[str, Any]) -> set[str]:
    from . import metric_spec

    names: set[str] = set()
    for key in ("metrics", "derived"):
        for m in protocol.get(key) if isinstance(protocol.get(key), list) else []:
            if isinstance(m, dict) and _ident(m.get("id")):
                names.update(metric_spec.result_forms(str(m["id"]).strip()))
    for key in ("grid", "in_each_run", "numerical_settings", "constants", "thresholds"):
        if isinstance(protocol.get(key), dict):
            names.update(str(k) for k in protocol[key])
    for sub in protocol.get("sub_studies") if isinstance(protocol.get("sub_studies"), list) else []:
        if isinstance(sub, dict) and _ident(sub.get("name")):
            names.add(str(sub["name"]).strip())
    for o in protocol.get("oracles") if isinstance(protocol.get("oracles"), list) else []:
        if isinstance(o, dict) and isinstance(o.get("case"), dict):
            names.update(str(k) for k in o["case"])
    return names


def undeclared_names(design: Any, not_asked: list[str] | None = None) -> dict[str, str]:
    """The names the plan relies on without declaring them as results, each with where the plan uses it: a name in the
    formula of a check's ``measure``, a key of ``labels``, the path of a ``result_assertions`` bound, or a name in
    backticks in the method. FI finds them by shape; a model's answer is then asked about each. In the order the plan
    uses them, at most :data:`MAX_NAMES`; the names past that are added to ``not_asked`` when it is given."""
    design = design if isinstance(design, dict) else {}
    p = design.get("protocol") if isinstance(design.get("protocol"), dict) else {}
    known = {n.lower() for n in _declared_names(p)} | _KNOWN_WORDS
    found: dict[str, str] = {}

    def take(name: str, where: str) -> None:
        if name.lower() not in known and name not in found and len(name) > 1:
            found[name] = where

    for o in p.get("oracles") if isinstance(p.get("oracles"), list) else []:
        if not isinstance(o, dict):
            continue
        for key in ("measure", "metric"):
            text = str(o.get(key) or "")
            for m in _WORD.finditer(text):
                if text[m.end():m.end() + 1] != "(" and not text[max(0, m.start() - 1):m.start()].isdigit():
                    take(m.group(0), f"the measure of the check `{o.get('name')}`")
    for name in p.get("labels") if isinstance(p.get("labels"), dict) else []:
        take(str(name), "the labels")
    for a in design.get("result_assertions") if isinstance(design.get("result_assertions"), list) else []:
        if isinstance(a, dict) and isinstance(a.get("path"), str) and _WORD.fullmatch(a["path"].strip()):
            take(a["path"].strip(), "a bound on its value")
    for m in re.finditer(r"`([A-Za-z_][A-Za-z0-9_]*)`", str(design.get("method") or "")):
        take(m.group(1), "the method")
    if not_asked is not None:
        not_asked.extend(list(found)[MAX_NAMES:])
    return dict(list(found.items())[:MAX_NAMES])


def _model_equation_ids(protocol: dict[str, Any]) -> set[str]:
    model = protocol.get("model")
    eqs = model.get("equations") if isinstance(model, dict) else None
    return {str(e.get("id")).strip() for e in eqs if isinstance(e, dict) and e.get("id")} if isinstance(eqs, list) else set()


def check_part(part: str, value: Any, design: Any) -> list[str]:
    """Why ``value``, a model's answer for ``part``, cannot be used; empty when it can. Structural only: every name is
    held to the ids and settings FI already has in ``design`` (nothing is judged by its words).

    ``roles``: ``{result id: {"role", "depends_on"}}`` for every declared result. ``names``: ``{name: {"answer", ...}}`` for
    every undeclared name :func:`undeclared_names` found, with ``formula`` or ``equation`` for a ``reference`` and
    ``inputs`` (declared results) for a ``derived``. ``comparisons``: ``{"comparisons": [{"result", "across", "values",
    "how"}]}``, a result being a declared result or a name answered ``derived``."""
    design = design if isinstance(design, dict) else {}
    p = design.get("protocol") if isinstance(design.get("protocol"), dict) else {}
    varied = varied_settings(p)
    declared = [str(m["id"]).strip() for m in p.get("metrics") or [] if isinstance(m, dict) and _ident(m.get("id"))]
    if not isinstance(value, dict):
        return [f"the answer for `{part}` must be a mapping"]
    out: list[str] = []
    if part == "roles":
        for rid in declared:
            row = value.get(rid)
            if not isinstance(row, dict):
                out.append(f"`{rid}` is not answered")
                continue
            if row.get("role") not in ROLES:
                out.append(f"`{rid}`: role must be one of {', '.join(ROLES)}")
            if row.get("returned") not in ("each_run", "afterwards"):
                out.append(f"`{rid}`: returned must be each_run or afterwards")
            dep = row.get("depends_on")
            if not isinstance(dep, list) or not all(isinstance(s, str) for s in dep):
                out.append(f"`{rid}`: depends_on must be a list of settings (empty when it changes with none)")
            else:
                bad = [s for s in dep if s not in varied]
                if bad:
                    out.append(f"`{rid}`: {', '.join(bad)} is not a setting that varies (the settings are "
                               f"{', '.join(varied) or 'none'})")
        given = {str(m.get("given")) for m in p.get("metrics") or [] if isinstance(m, dict) and m.get("given")}
        out += [f"`{g}` is the subset another result is given, so each run returns it (returned: each_run)"
                for g in sorted(given) if isinstance(value.get(g), dict) and value[g].get("returned") == "afterwards"]
        out += [f"`{k}` is not a declared result" for k in value if k not in declared]
    elif part == "names":
        wanted = undeclared_names(design)
        for name in wanted:
            row = value.get(name)
            if not isinstance(row, dict):
                out.append(f"`{name}` is not answered")
                continue
            answer = row.get("answer")
            if answer not in NAME_ANSWERS:
                out.append(f"`{name}`: answer must be one of {', '.join(NAME_ANSWERS)}")
            elif answer == "reference":
                eq, formula = row.get("equation"), row.get("formula")
                if not (isinstance(formula, str) and formula.strip()) and not (
                        isinstance(eq, str) and eq.strip() in _model_equation_ids(p)):
                    out.append(f"`{name}`: a reference needs a `formula` of the settings, or an `equation` that is in the "
                               f"plan's model ({', '.join(sorted(_model_equation_ids(p))) or 'it has none'})")
            elif answer == "derived":
                inputs = row.get("inputs")
                if not isinstance(inputs, list) or not inputs or not all(i in declared for i in inputs):
                    out.append(f"`{name}`: a quantity computed afterwards needs `inputs`, the declared results it is computed "
                               f"from ({', '.join(declared)})")
        out += [f"`{k}` was not asked about" for k in value if k not in wanted]
    elif part == "comparisons":
        items = value.get("comparisons")
        if not isinstance(items, list):
            return ["`comparisons` must be a list (empty when the study compares nothing between settings)"]
        names = set(declared) | {str(x) for x in value.get("_derived", ())}
        grid = p.get("grid") if isinstance(p.get("grid"), dict) else {}
        seen: set[tuple[str, str]] = set()
        for i, c in enumerate(items, 1):
            if not isinstance(c, dict):
                out.append(f"comparison {i} is not a mapping")
                continue
            if c.get("result") not in names:
                out.append(f"comparison {i}: `result` must be a declared result")
            if c.get("across") not in varied:
                out.append(f"comparison {i}: `across` must be a setting that varies ({', '.join(varied) or 'none'})")
            if c.get("how", "difference") not in ("difference", "ratio"):
                out.append(f"comparison {i}: `how` must be difference or ratio")
            vals = c.get("values", [])
            if not isinstance(vals, list) or (vals and c.get("across") in grid
                                              and any(v not in list(grid[c["across"]]) for v in vals)):
                out.append(f"comparison {i}: `values` must be values of `{c.get('across')}` in the grid")
            key = (str(c.get("result")), str(c.get("across")))
            if key in seen:
                out.append(f"comparison {i} repeats {key[0]} across {key[1]}")
            seen.add(key)
    else:
        out.append(f"unknown part `{part}`")
    return out
