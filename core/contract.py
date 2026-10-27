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
* ``diagnostic``: everything the simulation returns that the contract does not declare. Kept in the record, readable by
  the study's own derive functions, never reported and never checked.

This is not :class:`core.protocol.ResearchProtocol`, which says what *kind* of quest this is (a simulation, a literature
review); the contract says what one simulation study does.

:func:`from_design` reads a plan's design block (as :func:`core.plan.parse` gives it) into a contract by fixed rules,
with no model. A part it cannot place by those rules is left out and named in ``gaps`` (for the paper to say), never
guessed: an old plan that names its results only in prose, a precision target that is a sentence, or a figure given only
by its file name.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROLES = ("result", "reference", "timing", "physical_bound", "diagnostic")
Role = Literal["result", "reference", "timing", "physical_bound", "diagnostic"]
#: The most panels one figure may have.
MAX_PANELS = 3


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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
    kind: Literal["mean", "proportion"] = "mean"
    estimand: str = ""
    observation: str = ""  # what one independent observation is (one trial, one cluster)
    unit: str = ""  # the physical unit of the value, if any
    label: str = ""
    depends_on: tuple[str, ...] | None = None  # the settings it should change with; None: not declared
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


class Derived(_Strict):
    """A quantity computed from the results of a setting: by FI from ``formula`` (FI's formula language) when one is
    given, else by the study's own ``derive_<id>`` function, which sees every returned value (diagnostics too)."""

    id: str
    inputs: tuple[str, ...] = ()
    formula: str | None = None
    role: Role = "result"
    kind: Literal["mean", "proportion"] = "mean"
    label: str = ""


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


class Checks(_Strict):
    oracles: tuple[dict[str, Any], ...] = ()
    criteria: tuple[dict[str, Any], ...] = ()
    assertions: tuple[dict[str, Any], ...] = ()  # result bounds on a path that is not a declared result
    topic_map: tuple[dict[str, Any], ...] = ()


class StudyContract(_Strict):
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
        settings = self.settings.names()
        for c in self.comparisons:
            if c.result not in values:
                raise ValueError(f"comparison `{c.id}` compares `{c.result}`, which is not a declared result")
            if c.across not in set(self.settings.compared) | set(self.settings.in_each_run):
                raise ValueError(f"comparison `{c.id}` is across `{c.across}`, which is not a compared setting")
        for d in self.derived:
            missing = [i for i in d.inputs if i not in values]
            if missing:
                raise ValueError(f"derived `{d.id}` uses {', '.join(missing)}, which are not declared results")
        for r in self.results:
            unknown = [s for s in (r.depends_on or ()) if s not in settings]
            if unknown:
                raise ValueError(f"result `{r.id}` depends on {', '.join(unknown)}, which are not settings")
        for f in self.figures:
            unknown = [s for s in f.shows if s not in set(ids)]
            if unknown:
                raise ValueError(f"figure `{f.id}` shows {', '.join(unknown)}, which are not declared")
            for axis in (f.x, f.series):
                if axis is not None and axis not in settings:
                    raise ValueError(f"figure `{f.id}` uses `{axis}`, which is not a setting")
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

_NOTE_KEYS = ("seed_policy", "ci_method", "failure_policy", "acceptance", "simplifications", "definitions")
_DESIGN_NOTES = ("hypothesis", "method", "expected_outcome", "rationale", "variables")
#: Protocol keys the reader places; any other key goes to the notes and is named in the gaps.
_PLACED = {"grid", "sub_studies", "in_each_run", "numerical_settings", "constants", "thresholds", "runs_per_setting",
           "min_runs_per_setting", "precision", "metrics", "oracles", "criteria", "topic_map", "optimisation", "model",
           "labels", "comparisons", "derived", "figures", "steps", "shared_draws", *_NOTE_KEYS}


def _values(v: Any) -> tuple[Any, ...]:
    return tuple(v) if isinstance(v, (list, tuple)) else (v,)


def _mapping(v: Any) -> dict[str, Any]:
    return {str(k): x for k, x in v.items()} if isinstance(v, dict) else {}


def _rows(v: Any) -> tuple[dict[str, Any], ...]:
    return tuple(x for x in v if isinstance(x, dict)) if isinstance(v, list) else ()


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


def from_design(design: dict[str, Any] | None) -> StudyContract:
    """The contract a plan's design block gives, by fixed rules (no model, no guess); see the module docstring."""
    design = design if isinstance(design, dict) else {}
    p = _mapping(design.get("protocol"))
    gaps: list[str] = []
    labels = {str(k): str(v) for k, v in _mapping(p.get("labels")).items()}

    fixed = {**_mapping(p.get("constants")), **_mapping(p.get("thresholds"))}
    settings = _first_valid(Settings, {
        "compared": {k: _values(v) for k, v in _mapping(p.get("grid")).items()},
        "sub_studies": _rows(p.get("sub_studies")),
        "in_each_run": {k: _values(v) for k, v in _mapping(p.get("in_each_run")).items()},
        "numerical": _mapping(p.get("numerical_settings")), "fixed": fixed}, gaps, "the settings") or Settings()

    ranges = {}
    assertions = []
    declared_ids = {str(m.get("id")) for m in _rows(p.get("metrics"))}
    for a in _rows(design.get("result_assertions")):
        if str(a.get("path")) in declared_ids:
            ranges[str(a.get("path"))] = {"min": a.get("min"), "max": a.get("max"), "reason": str(a.get("reason") or "")}
        else:
            assertions.append(a)

    results = []
    for m in _rows(p.get("metrics")):
        rid = str(m.get("id") or "")
        depends = m.get("depends_on")
        if isinstance(depends, list):
            unknown = [str(s) for s in depends if str(s) not in settings.names()]
            if unknown:
                gaps.append(f"result `{rid}` is said to depend on {', '.join(unknown)}, which are not settings; "
                            "that is left out")
            depends = [str(s) for s in depends if str(s) in settings.names()]
        data = {"id": rid, "role": m.get("role") or "result", "kind": m.get("kind") or "mean",
                "estimand": str(m.get("estimand") or ""), "observation": str(m.get("unit") or ""),
                "unit": str(m.get("physical_unit") or ""), "label": str(m.get("label") or labels.get(rid, "")),
                "depends_on": depends, "range": ranges.get(rid) or m.get("range"),
                "known_value": m.get("known_value"), "reference": m.get("reference"), "given": m.get("given"),
                "cluster": m.get("cluster"), "paired": bool(m.get("paired")), "family": str(m.get("family") or ""),
                "undefined_when": _mapping(m.get("undefined_when"))}
        r = _first_valid(Result, {k: v for k, v in data.items() if v is not None}, gaps, f"result `{rid}`")
        if r is not None:
            results.append(r)
    if results and not any("role" in m for m in _rows(p.get("metrics"))):
        gaps.append("the plan gives no role for its results; each is read as a measured result")

    ids = {r.id for r in results}
    derived = [d for d in (_first_valid(Derived, x, gaps, f"derived `{x.get('id')}`") for x in _rows(p.get("derived")))
               if d is not None]
    ids |= {d.id for d in derived}
    comparisons = [c for c in (_first_valid(Comparison, x, gaps, f"comparison `{x.get('id')}`")
                               for x in _rows(p.get("comparisons"))) if c is not None]

    figures = []
    planned = p.get("figures") if isinstance(p.get("figures"), list) else design.get("figures_planned")
    for i, f in enumerate(planned if isinstance(planned, list) else [], 1):
        if isinstance(f, dict):
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

    base = {"settings": settings, "budget": budget, "labels": labels, "notes": notes,
            "checks": Checks(oracles=_rows(p.get("oracles")), criteria=_rows(p.get("criteria")),
                             assertions=tuple(assertions), topic_map=_rows(p.get("topic_map"))),
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


def _without_broken_links(base: dict[str, Any], parts: dict[str, Any], gaps: list[str], error: ValueError) -> StudyContract:
    gaps = [*gaps, f"a part of the plan refers to something it does not declare ({_first_error(error)})"]
    budget = base["budget"]
    base = {**base, "budget": budget.model_copy(update={"precision": None})}  # put back last: it may name any part
    # The parts are added back one at a time, each kind after the kinds it may refer to, so one broken item does not
    # take the good ones with it.
    built = {**parts, "comparisons": (), "derived": (), "figures": ()}
    for key in ("derived", "comparisons", "figures"):
        for item in parts[key]:
            trial = {**built, key: (*built[key], item)}
            try:
                StudyContract(**base, **trial)
                built = trial
            except ValueError:
                gaps.append(f"{key[:-1]} `{item.id}` is left out: it refers to something the plan does not declare")
    try:
        return StudyContract(**{**base, "budget": budget}, **built, gaps=tuple(gaps))
    except ValueError:
        gaps.append("the precision target is left out: it refers to something the plan does not declare")
        return StudyContract(**base, **built, gaps=tuple(gaps))
