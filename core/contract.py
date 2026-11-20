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
    #: The bound is one the physics of the quantity fixes (a fraction cannot be above 1), as the plan's model answered FI's
    #: question after the plan (core/contract_parts.py); a value on it is allowed, a value beyond it is a fault.
    physical: bool = False
    unit: str = ""  # the unit the bound is given in, where the answer states one

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
    #: ``True``: the plan says this number is a count or a share of counts (the same in every run), so the paper
    #: writes it as the record holds it; ``False``: a measurement; None: the plan does not say.
    exact: bool | None = None
    family: str = ""
    undefined_when: dict[str, Any] = Field(default_factory=dict)
    #: What one run returns for it: one ``number`` (the default), a ``curve`` along one name of :attr:`along`, or a ``map``
    #: over two. A name is ``time``, a position (``x``, ``y``, ``z``) in the studied system, or another quantity one run steps
    #: through named in a word or two; never a setting of the plan (FI records a number for each of its values itself).
    shape: Literal["number", "curve", "map"] = "number"
    along: tuple[str, ...] = ()


#: The names a curve or a map may run along that FI offers by name: time, and a position in the studied system.
SHAPE_AXES = ("time", "x", "y", "z")
SHAPES = ("number", "curve", "map")


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
    for_question: str | None = None  # the question of the topic the names step made this result for (core/contract_parts.py): read by no check
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
    x_scale: Literal["linear", "log"] | None = None
    y_scale: Literal["linear", "log"] | None = None
    declared: bool = True
    #: The reference lines the figure draws beside what it shows: each the id of a reference quantity of the plan, or of a
    #: result it shows whose plan gives a reference or a known value (:func:`reference_ids`).
    references: tuple[str, ...] = ()


#: The kinds of entry of the studied system: a ``part`` (a body, a region, a component), a ``boundary`` (a wall, an edge,
#: an interface where a condition is set) and a ``source`` (what drives the system).
SYSTEM_KINDS = ("part", "boundary", "source")
#: The most entries a description of the studied system may have (a set-up drawing with more is not readable).
MAX_SYSTEM_PARTS = 16


class SystemPart(_Strict):
    """One entry of the studied system. ``carries``: the ids of the plan's settings, constants and results that belong to it;
    every number shown for it is read from the plan by those ids, never typed beside its name."""

    id: str
    name: str
    kind: Literal["part", "boundary", "source"] = "part"
    carries: tuple[str, ...] = ()


class SystemLink(_Strict):
    """Two entries of the studied system that touch or feed each other, with what passes between them in plain words."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    source: str = Field(alias="from")
    to: str
    what: str = ""


class System(_Strict):
    """The studied system as the plan states it (``protocol.system``), the structure a set-up drawing is made from."""

    parts: tuple[SystemPart, ...] = ()
    links: tuple[SystemLink, ...] = ()


def figure_file(planned: Any) -> str:
    """The file name of a planned figure, the one name every part of FI matches a saved figure by: the plan gives a figure as
    ``fig1.png`` or as ``figures/fig1.png (two panels: ...)``, and a figure is saved in the figures folder under its base name."""
    text = str(planned.get("file") or "") if isinstance(planned, dict) else str(planned or "")
    head = text.split("(")[0].strip()  # the name, then (a description)
    first = (head.split() or [""])[0]
    # "fig1.png shows x against y": the first word is the file (an extension is a dot and a letter, not the decimal of "0.25");
    # "Figure 1" or "my plot.png": no such word, so the whole head is the name
    name = first if re.search(r"\.[A-Za-z][A-Za-z0-9]{1,3}$", first.rstrip(",;:")) else head
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
    system: System | None = None  # the studied system (protocol.system); None: the plan does not state one
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
            if f.series is not None and f.series not in settings:
                raise ValueError(f"figure `{f.id}` uses `{f.series}`, which is not a varied setting")
            if f.x is not None and f.x not in settings:
                # a result every trial returns may be the x axis: the figure then plots each trial's pair of values
                if f.x not in self.per_trial_ids() or not set(f.shows) <= set(self.per_trial_ids()):
                    raise ValueError(f"figure `{f.id}` uses `{f.x}`, which is not a varied setting")
        p = self.budget.precision
        if p is not None and p.target not in set(ids):
            raise ValueError(f"the precision target `{p.target}` is not a declared result or comparison")
        return self

    def per_trial_ids(self) -> list[str]:
        """The results every trial returns (``returned: each_run``): the record holds each trial's value of each."""
        return [r.id for r in self.results if r.returned == "each_run"]

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
           "labels", "comparisons", "derived", "figures", "steps", "shared_draws", "form", "parts_left_out", "names_not_asked", "figure_declarations", "dependent_map", "system", *_NOTE_KEYS}


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


def search_in_each_run(p: dict[str, Any]) -> dict[str, Any]:
    """What each evaluation of a search goes through together, where the plan wrote it: inside the search block
    (``protocol.optimisation.in_each_run``), as :func:`core.in_run.of_study` reads it; ``{}`` when the protocol has no search
    block or the block compares nothing inside its evaluations."""
    block = p.get("optimisation")
    inside = block.get("in_each_run") if isinstance(block, dict) else None
    return inside if isinstance(inside, dict) else {}


def settings_between_runs(p: dict[str, Any]) -> tuple[Any, Any]:
    """``(grid, sub_studies)``: the settings the runs of the study differ in, each with its values, and the parts that
    divide them, where the plan wrote them. A measurement: ``protocol.grid`` and ``protocol.sub_studies``. A search for the
    best design: the coarse scan it runs first (``protocol.optimisation.grid`` and ``.sub_studies``), whose settings FI
    runs and records one by one as it does a measurement's (core/optimise_search.py ``scan_grid``). A search that
    writes no scan has none: a grid beside its block is never run (the engine removes it when it takes the block). As
    written: nothing is checked here."""
    block = p.get("optimisation")
    if isinstance(block, dict):
        return block.get("grid"), block.get("sub_studies")
    return p.get("grid"), p.get("sub_studies")


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
    grid, parts = settings_between_runs(p)
    settings = _first_valid(Settings, {
        "compared": {k: _values(v) for k, v in _mapping(grid, gaps, "the grid").items()},
        "sub_studies": _rows(parts, gaps, "the sub-studies"),
        "in_each_run": {k: _values(v) for k, v in _mapping(search_in_each_run(p) or p.get("in_each_run"), gaps,
                                                            "in_each_run").items()},
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
                "range": _joined_range(ranges.get(rid), m.get("range")),
                **_shape_of(m, p, gaps),
                "known_value": m.get("known_value"), "reference": m.get("reference"), "given": _ident(given),
                "cluster": m.get("cluster"), "paired": bool(m.get("paired")), "exact": m.get("exact") if isinstance(m.get("exact"), bool) else None,
                "family": str(m.get("family") or ""),
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
                own = Figure(**{"id": f"fig{i}", **f, **({"file": figure_file(f)} if f.get("file") else {})})
            except ValueError:
                own = None
        if own is not None:
            fig = own
        elif isinstance(decl, dict):
            fig = _first_valid(Figure, {"id": f"fig{i}", **{k: v for k, v in decl.items() if k in ("shows", "x", "series", "panels", "x_scale", "y_scale", "references")},
                                       "file": name}, gaps, f"figure {i}")
        elif isinstance(f, dict):
            fig = _first_valid(Figure, {"id": f"fig{i}", **f}, gaps, f"figure {i}")
        else:
            fig = Figure(id=f"fig{i}", file=str(f), declared=False)
        if fig is not None:
            figures.append(_with_known_references(fig, results, derived, gaps))
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
            "search": _mapping(p.get("optimisation")) or None, "shared_draws": p.get("shared_draws") is True,
            "system": _system_of(p, gaps)}
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


def _joined_range(asserted: Any, own: Any) -> Any:
    """The range of a result: the plan's bound on it (``result_assertions``) and the range on its own entry (a physical bound
    FI asked after the plan, or one the plan wrote there), both held: the values inside both. Either alone is as given."""
    if not isinstance(asserted, dict) or not isinstance(own, dict):
        return asserted or own
    lows = [v for v in (asserted.get("min"), own.get("min")) if _is_number(v)]
    highs = [v for v in (asserted.get("max"), own.get("max")) if _is_number(v)]
    reason = "; ".join(str(r).strip() for r in (asserted.get("reason"), own.get("reason")) if str(r or "").strip())
    if lows and highs and max(lows) > min(highs):
        return asserted  # the two do not overlap: the plan's own bound is the one read (FI's check refuses such an answer)
    return {"min": max(lows) if lows else None, "max": min(highs) if highs else None, "reason": reason,
            "physical": own.get("physical") is True, "unit": str(own.get("unit") or asserted.get("unit") or "")}


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _shape_of(m: dict[str, Any], protocol: dict[str, Any], gaps: list[str]) -> dict[str, Any]:
    """``{"shape", "along"}`` of a declared result as its entry states it (a number when it states none). A shape that does not
    hold to the plan (:func:`check_shape`) is read as a number, said in ``gaps``; the result itself is kept."""
    if m.get("shape") in (None, "", "number") and not m.get("along"):
        return {}
    problems = check_shape(str(m.get("id")), {"shape": m.get("shape"), "along": m.get("along")}, protocol)
    if problems:
        gaps.append(f"the shape of result `{m.get('id')}` could not be used ({problems[0]}); it is read as one number")
        return {}
    return {"shape": m["shape"], "along": tuple(m.get("along") or ())}


def reference_ids(results: Any, derived: Any, shows: Any = None) -> list[str]:
    """What a figure may draw as a reference line: a reference quantity of the plan (a result or derived quantity of role
    ``reference``), and, among the results it ``shows`` (all of them when ``None``), each whose plan gives a reference or a
    known value."""
    shown = None if shows is None else set(shows)
    out = [r.id for r in results if r.role == "reference"] + [d.id for d in derived if d.role == "reference"]
    out += [r.id for r in results if (shown is None or r.id in shown) and r.role != "reference"
            and (isinstance(r.reference, dict) and r.reference or isinstance(r.known_value, dict) and r.known_value)]
    return list(dict.fromkeys(out))


def _with_known_references(fig: Figure, results: list[Result], derived: list[Derived], gaps: list[str]) -> Figure:
    """``fig`` with only the reference lines FI holds for it (:func:`reference_ids`); one it does not is left out, said in
    ``gaps``, and the figure is kept."""
    if not fig.references:
        return fig
    known = reference_ids(results, derived, fig.shows)
    lost = [r for r in fig.references if r not in known]
    if not lost:
        return fig
    gaps.append(f"figure `{fig.file}` names {', '.join('`' + r + '`' for r in lost)} as a reference line, which the plan "
                "declares as no reference of what it shows; that line is left out")
    return fig.model_copy(update={"references": tuple(r for r in fig.references if r in known)})


def _system_of(protocol: dict[str, Any], gaps: list[str]) -> System | None:
    """The studied system the plan states (``protocol.system``), held to the plan's ids (:func:`check_system`); one that does
    not hold is left out, said in ``gaps``. ``None`` when the plan states none, or states that there is no set-up to draw."""
    value = protocol.get("system")
    if value is None or (isinstance(value, dict) and not value.get("parts")):
        return None
    problems = check_system(value, protocol)
    if problems:
        gaps.append(f"the studied system the plan states could not be used ({problems[0]}); it is left out")
        return None
    return _first_valid(System, value, gaps, "the studied system")


# --- what a plan states about a result's shape, its physical bound and the studied system -------------------------------

_CODE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: A number written as its own word in a name ("layer 2", "0.5 mm gap"); a digit inside a word ("H2O") is not one.
_NUMBER_IN_TEXT = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])")


def _grid_and_inside(protocol: dict[str, Any]) -> tuple[list[str], list[str]]:
    grid = settings_between_runs(protocol)[0]
    inside = search_in_each_run(protocol) or protocol.get("in_each_run")
    return ([str(k) for k in grid] if isinstance(grid, dict) else [],
            [str(k) for k in inside] if isinstance(inside, dict) else [])


def check_shape(rid: str, row: Any, protocol: Any) -> list[str]:
    """Why the shape an answer states for the result ``rid`` (``row``'s ``shape`` and ``along``) cannot be used; empty when it
    can, or when it states none (a number). Held to the plan's own names: a ``curve`` runs along one thing a single run steps
    through and a ``map`` over two: ``time``, a position (:data:`SHAPE_AXES`), or another such quantity named in a word or two
    (a frequency). No setting of the plan is one of them: a setting the runs differ in has one value in a run, and for a value
    every run goes through together FI already records a number for each value when the result changes with it
    (``depends_on``); nor is a result or a constant of the plan."""
    p = protocol if isinstance(protocol, dict) else {}
    row = row if isinstance(row, dict) else {}
    shape, along = row.get("shape"), row.get("along")
    if shape in (None, "", "number"):
        return [] if along in (None, [], ()) else [f"`{rid}`: a number runs along nothing; give `along` only for a curve or a map"]
    if shape not in SHAPES:
        return [f"`{rid}`: `shape` must be one of {', '.join(SHAPES)}"]
    grid, inside = _grid_and_inside(p)
    allowed = list(SHAPE_AXES)
    want = 1 if shape == "curve" else 2
    if not isinstance(along, list) or len(along) != want or not all(isinstance(a, str) for a in along) or len(set(along)) != want:
        return [f"`{rid}`: a {shape} needs `along`, a list of {'one name' if want == 1 else 'two different names'} of "
                f"{', '.join(allowed)}"]
    out = []
    fixed = [str(k) for key in ("constants", "thresholds", "numerical_settings") if isinstance(p.get(key), dict) for k in p[key]]
    results = [str(m["id"]).strip() for key in ("metrics", "derived") for m in p.get(key) or []
               if isinstance(m, dict) and _ident(m.get("id"))]
    for a in along:
        if a in grid:
            out.append(f"`{rid}`: `{a}` is a setting the runs differ in, so one run has one value of it and returns one number "
                       f"for `{rid}` there (FI draws such numbers along `{a}` itself): answer `number`, or the thing it runs "
                       f"along inside one run ({', '.join(allowed)}, or another named in a word or two)")
        elif a in allowed:
            continue
        elif a in inside:
            out.append(f"`{rid}`: every run goes through the values of `{a}` together, and FI records a number of `{rid}` for "
                       f"each of them when it changes with `{a}` (list `{a}` in its depends_on): that is a number, not a curve; "
                       f"a curve runs along what one run steps through ({', '.join(allowed)}, or another named in a word or two)")
        elif a in results or a in fixed:
            out.append(f"`{rid}`: `{a}` is a {'number the study reports' if a in results else 'constant of the plan'}, not "
                       f"something a run steps through; use {', '.join(allowed)}, or another such thing named in a word or two")
        elif not _plain_word(a):
            out.append(f"`{rid}`: `along` names what a run steps through in a word or two, with no number (`{a}` is not)")
    return out


def _plain_word(text: str) -> bool:
    """A name of a quantity a run steps through: one to three words of letters (underscores allowed), no number of its own."""
    words = text.replace("_", " ").split()
    return 1 <= len(words) <= 3 and len(text) <= 40 and all(w.isalpha() for w in words)


def _plan_bound(design: dict[str, Any], rid: str) -> dict[str, Any] | None:
    """The plan's own bound on the result ``rid``: its ``result_assertions`` entry by that path, else a range on its entry that
    FI did not ask for."""
    for a in design.get("result_assertions") if isinstance(design.get("result_assertions"), list) else []:
        if isinstance(a, dict) and str(a.get("path") or "").strip() == rid and (_is_number(a.get("min")) or _is_number(a.get("max"))):
            return a
    p = design.get("protocol") if isinstance(design.get("protocol"), dict) else {}
    for m in p.get("metrics") or []:
        if isinstance(m, dict) and m.get("id") == rid and isinstance(m.get("range"), dict) and m["range"].get("physical") is not True:
            return m["range"]
    return None


def _known_numbers(design: dict[str, Any], rid: str) -> list[tuple[str, float]]:
    """The values the plan itself says the result ``rid`` takes: its known true value, and the expected value of each check
    that measures exactly it, each with where it is written."""
    from . import metric_spec

    p = design.get("protocol") if isinstance(design.get("protocol"), dict) else {}
    fixed = metric_spec.fixed_numbers(p)
    out: list[tuple[str, float]] = []
    for m in p.get("metrics") or []:
        if isinstance(m, dict) and m.get("id") == rid and isinstance(m.get("known_value"), dict):
            v = metric_spec.fixed_value(m["known_value"].get("true_value"), fixed)
            if v is not None:
                out.append(("its known true value", v))
    for o in p.get("oracles") if isinstance(p.get("oracles"), list) else []:
        if isinstance(o, dict) and _ident(o.get("measure")) == rid and _is_number(o.get("expected")):
            out.append((f"the expected value of the check `{o.get('name')}`", float(o["expected"])))
    return out


def check_bound(rid: str, bound: Any, design: Any) -> list[str]:
    """Why the physical bound an answer states for the result ``rid`` cannot be used; empty when it can, or when it states
    none (``None``). A bound gives ``min`` and/or ``max`` (numbers, ``min`` not above ``max``) and ``why``. It is held to the
    plan's own facts about the result: the values the plan says it takes lie inside it, it overlaps the plan's own bound on
    it, and its ``unit``, where both are given, is the result's declared ``physical_unit``. Nothing is judged by its words."""
    if bound is None:
        return []
    design = design if isinstance(design, dict) else {}
    if not isinstance(bound, dict):
        return [f"`{rid}`: `bound` must be a mapping with `min` and/or `max`, or null"]
    lo, hi = bound.get("min"), bound.get("max")
    out = []
    if (lo is not None and not _is_number(lo)) or (hi is not None and not _is_number(hi)):
        return [f"`{rid}`: the bound's `min` and `max` must be numbers (or null)"]
    if lo is None and hi is None:
        return [f"`{rid}`: a bound gives `min`, `max` or both; answer `bound`: null when physics fixes none"]
    if lo is not None and hi is not None and lo > hi:
        return [f"`{rid}`: the bound's `min` {lo} is above its `max` {hi}"]
    if not (isinstance(bound.get("why"), str) and bound["why"].strip()):
        out.append(f"`{rid}`: say in `why` what in the physics fixes the bound")
    p = design.get("protocol") if isinstance(design.get("protocol"), dict) else {}
    declared = next((m.get("physical_unit") for m in p.get("metrics") or [] if isinstance(m, dict) and m.get("id") == rid
                     and str(m.get("physical_unit") or "").strip()), None)
    unit = bound.get("unit")
    if declared is not None and isinstance(unit, str) and unit.strip() and "".join(unit.split()) != "".join(str(declared).split()):
        out.append(f"`{rid}`: the plan gives `{rid}` in `{declared}`, but the bound is in `{unit}`; give the bound in `{declared}`")
    for where, v in _known_numbers(design, rid):
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            out.append(f"`{rid}`: {where} is {v:g}, outside the bound; a physical bound holds every value the quantity can take, "
                       "so correct the bound, or answer null if physics fixes none")
    own = _plan_bound(design, rid)
    if own is not None:
        olo, ohi = own.get("min"), own.get("max")
        if (lo is not None and _is_number(ohi) and ohi < lo) or (hi is not None and _is_number(olo) and olo > hi):
            out.append(f"`{rid}`: the plan's own bound on it is {olo if _is_number(olo) else '-inf'} to "
                       f"{ohi if _is_number(ohi) else 'inf'}, which lies outside the physical bound; correct it, or answer null")
    return out


def is_code_name(name: Any) -> bool:
    """Whether ``name`` is written as a name in code rather than in words: one token of letters, digits and underscores
    with an underscore or a digit in it (``full_grid``, ``sweep2``)."""
    return isinstance(name, str) and bool(_CODE_NAME.fullmatch(name)) and ("_" in name or any(c.isdigit() for c in name))


def titles_to_ask(protocol: Any) -> dict[str, str]:
    """The sub-studies and the groups of results (a result's ``family``) whose name is a name in code (:func:`is_code_name`)
    and that have no reader name in ``protocol.labels`` yet, each with what it is: a figure or a paper would print the code
    name. In the plan's order."""
    p = protocol if isinstance(protocol, dict) else {}
    labels = p.get("labels") if isinstance(p.get("labels"), dict) else {}
    out: dict[str, str] = {}
    parts = settings_between_runs(p)[1]
    for sub in parts if isinstance(parts, list) else []:
        name = sub.get("name") if isinstance(sub, dict) else None
        if is_code_name(name) and not str(labels.get(name) or "").strip():
            out.setdefault(name, "a sub-study")
    for m in p.get("metrics") or []:
        name = m.get("family") if isinstance(m, dict) else None
        if is_code_name(name) and not str(labels.get(name) or "").strip():
            out.setdefault(name, "a group of results")
    return out


def group_names(protocol: Any) -> set[str]:
    """The names of the plan's sub-studies and of its groups of results (a result's ``family``)."""
    p = protocol if isinstance(protocol, dict) else {}
    parts = settings_between_runs(p)[1]
    out = {str(s["name"]) for s in parts if isinstance(s, dict) and isinstance(s.get("name"), str)} if isinstance(parts, list) else set()
    return out | {str(m["family"]) for m in p.get("metrics") or [] if isinstance(m, dict) and isinstance(m.get("family"), str)}


def check_title(name: str, title: Any) -> list[str]:
    """Why ``title``, the plain title an answer gives the sub-study or group ``name``, cannot be used; empty when it can:
    words a scientist would write, not a name in code and not the name itself written again."""
    if not isinstance(title, str) or not title.strip():
        return [f"`{name}` needs a plain title"]
    text = " ".join(title.split())
    if len(text) > 80:
        return [f"`{name}`: the title is longer than 80 characters; shorten it"]
    if "_" in text or is_code_name(text) or text.lower().replace(" ", "_") == name.lower():
        return [f"`{name}`: \"{text}\" is the name in code again; give the title a scientist would write in a paper"]
    return []


def system_names(protocol: Any) -> list[str]:
    """The ids an entry of the studied system may carry: the plan's settings (those the runs differ in, those every run goes
    through together, the numerical settings), its constants and thresholds, its declared results and derived quantities."""
    p = protocol if isinstance(protocol, dict) else {}
    out: list[str] = []
    grid, inside = _grid_and_inside(p)
    out += grid + inside
    for key in ("numerical_settings", "constants", "thresholds"):
        if isinstance(p.get(key), dict):
            out += [str(k) for k in p[key]]
    for key in ("metrics", "derived"):
        out += [str(m["id"]).strip() for m in p.get(key) or [] if isinstance(m, dict) and _ident(m.get("id"))]
    return list(dict.fromkeys(out))


def _plain_name(text: Any) -> str | None:
    """Why ``text`` is not a plain name for an entry or a link (``None`` when it is): text, short, with no number in it."""
    if not isinstance(text, str) or not text.strip():
        return "needs a plain `name`"
    if len(text) > 80:
        return "has a name longer than 80 characters"
    if _NUMBER_IN_TEXT.search(text):
        return (f"has a number in its words (\"{text.strip()}\"); every number comes from the plan by the id in `carries`, so "
                "write the name without it")
    return None


def check_system(value: Any, protocol: Any) -> list[str]:
    """Why ``value``, the studied system an answer or the plan states, cannot be used; empty when it can, or when it states
    there is none (``None``, or no parts). ``parts``: each an ``id`` (a name in code, once), a plain ``name`` with no number
    in it, a ``kind`` of :data:`SYSTEM_KINDS` and ``carries``, ids of the plan (:func:`system_names`). ``links``: each
    ``from`` and ``to``, two different entries, and ``what`` passes between them in plain words."""
    if value is None:
        return []
    if not isinstance(value, dict):
        return ["`system` must be a mapping with `parts` and `links`, or null when the study has no set-up to draw"]
    parts, links = value.get("parts"), value.get("links", [])
    unknown_keys = [k for k in value if k not in ("parts", "links")]
    out = [f"`system` has `{k}`, which is not `parts` or `links`" for k in unknown_keys]
    if parts in (None, []):
        return out
    if not isinstance(parts, list) or not all(isinstance(x, dict) for x in parts):
        return out + ["`system.parts` must be a list of entries, each with `id`, `name`, `kind` and `carries`"]
    if len(parts) > MAX_SYSTEM_PARTS:
        out.append(f"`system.parts` has {len(parts)} entries; at most {MAX_SYSTEM_PARTS}, so join the smaller ones")
    names = system_names(protocol)
    ids: list[str] = []
    for i, part in enumerate(parts, 1):
        pid = part.get("id")
        if not isinstance(pid, str) or not _CODE_NAME.fullmatch(pid):
            out.append(f"part {i} needs an `id`, a name in code (letters, digits and underscores)")
            continue
        if pid in ids:
            out.append(f"`{pid}` is given twice; each part has its own id")
        ids.append(pid)
        why = _plain_name(part.get("name"))
        if why:
            out.append(f"part `{pid}` {why}")
        if part.get("kind", "part") not in SYSTEM_KINDS:
            out.append(f"part `{pid}`: `kind` must be one of {', '.join(SYSTEM_KINDS)}")
        carries = part.get("carries", [])
        if not isinstance(carries, list) or not all(isinstance(c, str) for c in carries):
            out.append(f"part `{pid}`: `carries` must be a list of the plan's names")
        else:
            bad = [c for c in carries if c not in names]
            if bad:
                out.append(f"part `{pid}` carries {', '.join('`' + c + '`' for c in bad)}, which the plan does not declare "
                           f"(it declares {', '.join(names[:40]) or 'nothing'})")
        extra = [k for k in part if k not in ("id", "name", "kind", "carries")]
        if extra:
            out.append(f"part `{pid}` has {', '.join('`' + k + '`' for k in extra)}; an entry has only `id`, `name`, `kind` and "
                       "`carries` (a number goes in by its plan id in `carries`)")
    if not isinstance(links, list) or not all(isinstance(x, dict) for x in links):
        return out + ["`system.links` must be a list of entries, each with `from`, `to` and `what`"]
    for i, link in enumerate(links, 1):
        a, b = link.get("from"), link.get("to")
        if a not in ids or b not in ids:
            out.append(f"link {i} must join two part ids ({', '.join(ids)})")
        elif a == b:
            out.append(f"link {i} joins `{a}` to itself")
        what = link.get("what")
        if what not in (None, "") and _plain_name(what):
            out.append(f"link {i} {_plain_name(what)}")
        extra = [k for k in link if k not in ("from", "to", "what")]
        if extra:
            out.append(f"link {i} has {', '.join('`' + k + '`' for k in extra)}; a link has only `from`, `to` and `what`")
    return out


def range_assertions(design: Any) -> list[dict[str, Any]]:
    """The bounds the ranges on the plan's declared results give, written as ``result_assertions`` entries (``path``, ``min``,
    ``max``, ``unit``, ``reason``), for the range check (core/plausibility.py) to hold every run to: a physical bound FI asked
    after the plan, or a range the plan wrote on the entry. A physical bound may be reached (``may_equal_bound``). An entry
    the plan's own ``result_assertions`` already has, with the same numbers, is not repeated."""
    design = design if isinstance(design, dict) else {}
    p = design.get("protocol") if isinstance(design.get("protocol"), dict) else {}
    have = {(str(a.get("path")), a.get("min"), a.get("max")) for a in design.get("result_assertions") or [] if isinstance(a, dict)}
    out = []
    for m in p.get("metrics") or []:
        r = m.get("range") if isinstance(m, dict) else None
        rid = _ident(m.get("id")) if isinstance(m, dict) else None
        if rid is None or not isinstance(r, dict):
            continue
        lo, hi = (r.get("min") if _is_number(r.get("min")) else None), (r.get("max") if _is_number(r.get("max")) else None)
        if (lo is None and hi is None) or (rid, lo, hi) in have or (lo is not None and hi is not None and lo > hi):
            continue
        out.append({"path": rid, "min": lo, "max": hi, "unit": str(m.get("physical_unit") or r.get("unit") or ""),
                    "reason": str(r.get("reason") or ""), "may_equal_bound": r.get("physical") is True})
    return out


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

#: The answers for a name the plan relies on but does not declare as a result. ``label``: only another word for several
#: results already declared (one per method or setting); no number is computed under it.
NAME_ANSWERS = ("result", "reference", "diagnostic", "derived", "label")
#: The most undeclared names asked about in one question.
MAX_NAMES = 15
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KNOWN_WORDS = {"pi", "e", "inf", "nan", "true", "false", "none", "and", "or", "not", "if", "else", "for", "in"}


def varied_settings(protocol: Any) -> list[str]:
    """The settings whose values change between the runs of a study: the grid and what each run goes through."""
    p = protocol if isinstance(protocol, dict) else {}
    out: list[str] = []
    for block in (settings_between_runs(p)[0], search_in_each_run(p) or p.get("in_each_run")):
        out += [str(k) for k in block if str(k) not in out] if isinstance(block, dict) else []
    return out


def _declared_names(protocol: dict[str, Any]) -> set[str]:
    from . import metric_spec

    names: set[str] = set()
    for key in ("metrics", "derived"):
        for m in protocol.get(key) if isinstance(protocol.get(key), list) else []:
            if isinstance(m, dict) and _ident(m.get("id")):
                names.update(metric_spec.result_forms(str(m["id"]).strip()))
    grid, parts = settings_between_runs(protocol)
    for block in (grid, search_in_each_run(protocol) or protocol.get("in_each_run"), protocol.get("numerical_settings"),
                  protocol.get("constants"), protocol.get("thresholds")):
        if isinstance(block, dict):
            names.update(str(k) for k in block)
    for sub in parts if isinstance(parts, list) else []:
        if isinstance(sub, dict) and _ident(sub.get("name")):
            names.add(str(sub["name"]).strip())
    for o in protocol.get("oracles") if isinstance(protocol.get("oracles"), list) else []:
        if isinstance(o, dict) and isinstance(o.get("case"), dict):
            names.update(str(k) for k in o["case"])
    return names


def is_family_label(row: Any) -> bool:
    """An answer ``label``: the name is only another word for the declared results in ``inputs``. FI never freezes it as a
    result (nothing defines one number out of several results of one quantity); each member stays as declared and the name
    refers to all of them (core/contract_parts.py ``_apply_names``)."""
    return isinstance(row, dict) and row.get("answer") == "label"


def family_labels(protocol: Any) -> dict[str, list[str]]:
    """The names the plan's model answered as another word for several declared results: ``{name: [those results]}``."""
    rows = protocol.get("named_results") if isinstance(protocol, dict) else None
    return {str(n["id"]): [str(i) for i in n["of"]] for n in rows or []
            if isinstance(n, dict) and n.get("id") and n.get("answer") == "label" and isinstance(n.get("of"), list)}


def undeclared_names(design: Any, not_asked: list[str] | None = None) -> dict[str, str]:
    """The names the plan relies on without declaring them as results, each with where the plan uses it: a name in the
    formula of a check's ``measure``, a key of ``labels``, the path of a ``result_assertions`` bound, or a name in
    backticks in the method. FI finds them by shape; a model's answer is then asked about each. In the order the plan
    uses them, at most :data:`MAX_NAMES`; the names past that are added to ``not_asked`` when it is given."""
    design = design if isinstance(design, dict) else {}
    p = design.get("protocol") if isinstance(design.get("protocol"), dict) else {}
    # a sub-study's or a group's reader name in the labels is a title, not a number
    known = {n.lower() for n in _declared_names(p) | group_names(p)} | _KNOWN_WORDS
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


def unknown_measure_names(old_protocol: Any, new_protocol: Any, returned: Any) -> dict[str, list[str]]:
    """After a test run of the plan's checks: the checks of ``new_protocol`` that were added or whose ``measure`` was changed
    since ``old_protocol``, each with the names that measure reads and no part of FI defines: ``{check name: [names]}``.
    ``returned`` is what the simulation returned on the test run (the names, or the values by name). A name is defined when
    the plan declares it (:func:`_declared_names`: a result, a derived quantity, a setting, a constant, a threshold, a key
    of a check's case) or the simulation returned it on the test run. Any other is a number nobody computes: a check that
    reads it can never be judged, and the code is not rewritten after this test run. ``{}`` when ``returned`` is empty (the
    test run returned nothing to hold a name to). Pure: no model, no run."""
    have = {str(n).lower() for n in returned or ()}
    if not have or not isinstance(new_protocol, dict):
        return {}
    old = old_protocol if isinstance(old_protocol, dict) else {}
    known = {n.lower() for n in _declared_names(new_protocol)} | _KNOWN_WORDS | have
    before: dict[str, str] = {}
    for o in old.get("oracles") if isinstance(old.get("oracles"), list) else []:
        if isinstance(o, dict):
            before[str(o.get("name") or "").strip()] = " ".join(str(o.get(k) or "") for k in ("measure", "metric")).strip()
    out: dict[str, list[str]] = {}
    for o in new_protocol.get("oracles") if isinstance(new_protocol.get("oracles"), list) else []:
        if not isinstance(o, dict):
            continue
        name = str(o.get("name") or "").strip()
        text = " ".join(str(o.get(k) or "") for k in ("measure", "metric")).strip()
        if not text or before.get(name) == text:
            continue
        found = [m.group(0) for m in _WORD.finditer(text)
                 if text[m.end():m.end() + 1] != "(" and not text[max(0, m.start() - 1):m.start()].isdigit()
                 and m.group(0).lower() not in known]
        if found:
            out[name] = list(dict.fromkeys(found))
    return out


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
            elif answer == "label":
                inputs = row.get("inputs")
                if not isinstance(inputs, list) or len(inputs) < 2 or not all(i in declared for i in inputs):
                    out.append(f"`{name}`: a label needs `inputs`, two or more declared results it is another word for "
                               f"({', '.join(declared)})")
        out += [f"`{k}` was not asked about" for k in value if k not in wanted]
    elif part == "comparisons":
        items = value.get("comparisons")
        if not isinstance(items, list):
            return ["`comparisons` must be a list (empty when the study compares nothing between settings)"]
        names = set(declared) | {str(x) for x in value.get("_derived", ())}
        grid = settings_between_runs(p)[0] if isinstance(settings_between_runs(p)[0], dict) else {}
        inside = search_in_each_run(p) or p.get("in_each_run")
        listed = {**(inside if isinstance(inside, dict) else {}), **grid}  # each setting that varies, with the values the plan gives it
        seen: set[tuple[str, str]] = set()
        for i, c in enumerate(items, 1):
            if not isinstance(c, dict):
                out.append(f"comparison {i} is not a mapping")
                continue
            if c.get("result") not in names:
                of = (value.get("_labels") or {}).get(c.get("result"))
                out.append(f"comparison {i}: `result` must be a declared result"
                           + (f"; `{c.get('result')}` is only another name for {', '.join(of)}, so name one of them" if of else ""))
            if c.get("across") not in varied:
                out.append(f"comparison {i}: `across` must be a setting that varies ({', '.join(varied) or 'none'})")
            if c.get("how", "difference") not in ("difference", "ratio"):
                out.append(f"comparison {i}: `how` must be difference or ratio")
            vals = c.get("values", [])
            across = c.get("across")
            if not isinstance(vals, list) or (vals and across in listed and isinstance(listed[across], (list, tuple))
                                              and any(v not in list(listed[across]) for v in vals)):
                own = listed.get(across) if across not in grid else None  # a setting every run goes through together
                out.append(f"comparison {i}: `values` must be values of `{across}` "
                           + (f"({', '.join(str(v) for v in own)})" if isinstance(own, (list, tuple)) else "in the grid"))
            key = (str(c.get("result")), str(c.get("across")))
            if key in seen:
                out.append(f"comparison {i} repeats {key[0]} across {key[1]}")
            seen.add(key)
    else:
        out.append(f"unknown part `{part}`")
    return out
