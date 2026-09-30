"""An oracle the experiment must pass before its main run: something independent of the script's own numbers.

Every check that ran before this looked at internal consistency. The bounds a design declares (``result_assertions``) only
say a number is legal; the numeric, statistics and provenance audits only say the paper copied the script's output
faithfully; the reviewer reads prose. A simulator that uses the wrong model, the wrong parameter or the wrong estimator
still prints legal numbers, and every one of those checks is green. What was missing is a check against something the
script did not produce: a closed form, a limiting case, an invariant that must hold, a small case whose exact answer is
known, or a second implementation.

The plan's protocol therefore declares its **oracles** (``protocol.oracles``: a ``name``, a ``check`` that says what is
compared with what, a numeric ``expected`` and ``tolerance`` that the check is judged by, optionally a ``tolerance_mode``
(``absolute``, the default, or ``relative``), a ``kind`` (one of :data:`KINDS`) and the ``reference`` the expected value
comes from, which the engine reads: :func:`source_gaps`), and the script is
written to *measure* them. When the environment variable ``FI_ORACLE`` is ``1`` the script does not run its sweep: it computes
the value of each declared check on a small fast case, prints one line

    ORACLE_JSON: {"checks": [{"name": "...", "value": 0.98, "diagnostics": {"solver_success": true}}]}

and exits 0. **The engine judges**: it takes ``expected`` and ``tolerance`` from the (frozen) protocol, never from the script,
and computes ``abs(value - expected) <= tolerance`` (or ``tolerance * abs(expected)``) itself. A ``passed`` the script prints,
and any ``expected`` or ``tolerance`` it prints, are not read as a verdict: a script that writes its own pass/fail, its own
expected value and its own tolerance can always be made to pass, which is what an oracle is for not being. (A script that
reports a check as failed itself is still a problem.) For an invariant (conservation, monotonicity) the value is the worst
violation observed and the expected value is 0.

An oracle may also name a **case** (the settings of one run, e.g. ``{"dt": 0.1}``) and a **measure** (which number the
simulation returns for it). The engine then calls the simulation function (``run_trial``/``run_cell``) itself on that case
and reads the measure from what it returns: the number never passes through anything the script wrote for the check, so a
script whose own ``oracle()`` returns the closed form without simulating cannot pass. An oracle without a case falls back to
the script's ``oracle()``; the record says which of the two produced each value, and a value the script reported is not
counted as independent evidence.

A **problem** is any of: the design declares no oracle; a declared oracle fixes no numeric ``expected`` and ``tolerance`` (the
engine cannot judge it); the script printed no ``ORACLE_JSON`` line; a declared oracle does not appear among the checks or
reports no finite numeric ``value``; the value is outside the tolerance; the script reports the check as failed itself; the
script exited non-zero.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

_LINE_RE = re.compile(r"^\s*ORACLE_JSON:\s*(\{.*\})\s*$", re.MULTILINE)


def declared(protocol: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The oracles the protocol declares, each as a mapping with at least a ``name``."""
    items = protocol.get("oracles") if isinstance(protocol, dict) else None
    out: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and str(item.get("name") or "").strip():
            out.append(item)
        elif isinstance(item, str) and item.strip():
            out.append({"name": item.strip(), "check": item.strip()})
    return out


def parse(stdout: str) -> dict[str, Any] | None:
    """The last ``ORACLE_JSON`` line of ``stdout`` as a mapping; ``None`` when there is none or it is not JSON."""
    found = None
    for match in _LINE_RE.finditer(stdout or ""):
        try:
            value = json.loads(match.group(1))
        except ValueError:
            continue
        if isinstance(value, dict):
            found = value
    return found


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def limit_of(oracle: dict[str, Any]) -> tuple[float | None, float | None, str]:
    """``(expected, limit, mode)`` an oracle is judged by: ``limit`` is the largest difference from ``expected`` that counts as
    agreeing (``None`` when the oracle cannot be judged: no numeric expected or tolerance, or a relative tolerance around 0)."""
    expected, tolerance = _num(oracle.get("expected")), _num(oracle.get("tolerance"))
    mode = "relative" if str(oracle.get("tolerance_mode") or "").strip().lower() == "relative" else "absolute"
    if expected is None or tolerance is None or tolerance < 0:
        return None, None, mode
    if mode == "relative":
        if expected == 0:
            return expected, None, mode
        return expected, tolerance * abs(expected), mode
    return expected, tolerance, mode


def case_of(oracle: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    """``(case, measure)`` when the oracle names a run of the simulation the engine can make itself (a ``case`` mapping and a
    ``measure``, the name of the number that run returns); ``None`` when it does not."""
    case, measure = oracle.get("case"), str(oracle.get("measure") or "").strip()
    if isinstance(case, dict) and measure:
        return dict(case), measure
    return None


_STEP_KEYS = ("dt", "h", "step", "dx", "delta_t", "timestep", "step_size")


def loose_tolerance(oracle: dict[str, Any]) -> str | None:
    """A sentence when the oracle claims an order of accuracy (``order``, 2 or more) on a case with a step size below 1 and its
    tolerance could not tell that method from a first-order one: an error of about ``step ** order`` is what the claimed
    method gives and about ``step`` what a first-order one gives, and a tolerance above the geometric middle of the two,
    ``step ** ((1 + order) / 2)``, lets both through. ``None`` otherwise. A warning, never a verdict: the constant in front of
    the error is not known."""
    order, expected, limit, _ = (_num(oracle.get("order")),) + limit_of(oracle)
    case = oracle.get("case")
    if order is None or order < 2 or limit is None or expected is None or not isinstance(case, dict):
        return None
    step = next((_num(case[k]) for k in _STEP_KEYS if k in case and _num(case[k]) is not None), None)
    if step is None or not 0 < step < 1:
        return None
    scale = abs(expected) if expected != 0 else 1.0
    middle = step ** ((1 + order) / 2)
    if limit / scale < middle:
        return None
    return (
        f"the oracle {str(oracle['name']).strip()!r} claims order {_fmt(order)} at step {_fmt(step)}, but its tolerance "
        f"({_fmt(limit / scale)}) is above {_fmt(middle)}: a first-order method would pass it too"
    )

# --- what kind of check, and where its expected value comes from ---------------------------------------------------
#
# A check is only as good as its expected value, and the plan (a model) writes that value. A real quest expected an RK4
# error of 1.637e-08 where the true one is 3.33241e-07, and nothing said where the number came from, so nobody could
# tell whether the simulation or the check was wrong. Each oracle therefore names its kind (one of six) and a
# ``reference`` the engine reads: a derivation with its steps written, a source this quest retrieved (by the [n] the
# plan's literature uses, its title or its DOI), an equation of the plan's model (E1, E2...) whose own source counts, or,
# for a second implementation, what code it does not share. A reference to anything else (a source recalled from
# memory) is not a source a reader can check.

#: The six kinds of check against a known answer, and how the plan says each in words.
KINDS: dict[str, str] = {
    "special_case": "a special or limiting case with a known answer",
    "invariant": "a conserved quantity or other invariant",
    "symmetry": "a symmetry or scaling law",
    "second_implementation": "an independent second implementation",
    "convergence_rate": "a convergence rate",
    "published_value": "a published benchmark value",
}
# The names used before these six (the plan asked for them), and other words a model writes for one of the six.
_KIND_WORDS = {
    "closed_form": "special_case", "limiting_case": "special_case", "exact_small_case": "special_case",
    "exact_solution": "special_case", "analytic": "special_case", "analytical": "special_case", "limit": "special_case",
    "known_case": "special_case", "small_case": "special_case",
    "conservation": "invariant", "conservation_law": "invariant", "conserved_quantity": "invariant",
    "scaling": "symmetry", "symmetry_scaling": "symmetry", "scaling_law": "symmetry",
    "independent_implementation": "second_implementation", "second_method": "second_implementation",
    "convergence": "convergence_rate", "convergence_order": "convergence_rate", "order_of_accuracy": "convergence_rate",
    "benchmark": "published_value", "published_benchmark": "published_value", "benchmark_value": "published_value",
    "literature_value": "published_value",
}


def kind_of(oracle: dict[str, Any]) -> str | None:
    """The oracle's kind as one of :data:`KINDS` (an older or looser name read as the kind it means); ``None`` when it
    names none of them. The plan keeps what was written: only what the engine reads is mapped."""
    word = re.sub(r"[\s\-/]+", "_", str(oracle.get("kind") or "").strip().lower()).strip("_")
    if not word:
        return None
    return word if word in KINDS else _KIND_WORDS.get(word)


def _doi(text: Any) -> str:
    """A DOI in one form: lower case, without a ``https://doi.org/`` or ``doi:`` prefix or trailing punctuation."""
    doi = str(text or "").strip().lower()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi)
    return doi.rstrip(".,;:”’\"'")


def retrieved_sources(labelled: list[tuple[str, dict[str, Any]]]) -> list[dict[str, str]]:
    """The quest's retrieved sources as the checks below match them: ``(label, metadata)`` pairs, the labels the plan's
    literature list and the paper use ([1], [W1]), become ``{label, title, doi, url}``."""
    out = []
    for label, meta in labelled:
        out.append({
            "label": str(label), "title": " ".join(str(meta.get("title") or "").split()),
            "doi": _doi(meta.get("doi")), "url": str(meta.get("url") or "").strip().lower(),
        })
    return out


def sources_block(sources: list[dict[str, str]]) -> list[str]:
    """The numbered list of the sources this quest retrieved, one line each, as a check cites them ([n])."""
    rows = []
    for s in sources if isinstance(sources, list) else []:
        if isinstance(s, dict) and s.get("label"):
            doi = f" (DOI {s['doi']})" if s.get("doi") else ""
            rows.append(f"- [{s['label']}] {' '.join(str(s.get('title') or '(untitled)').replace('`', chr(39)).split())}{doi}")
    return rows


# A citation label, alone or in a group or range: [2], [1, 3], [1-3], [W1]. Not an index such as y[10].
_LABEL_RE = re.compile(r"(?<![\w\)])\[\s*(W?\d+(?:\s*[,–\-]\s*W?\d+)*)\s*\]", re.IGNORECASE)
# A DOI up to whitespace; parentheses inside it are kept (10.1016/0021-9991(76)90041-3), and a closing bracket or
# punctuation that ends the sentence is not.
_DOI_RE = re.compile(r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?\b10\.\d{4,9}/\S+", re.IGNORECASE)
_DERIVATION_RE = re.compile(r"^\s*(?:derivation|derived)\b[\s:.\-–—]*", re.IGNORECASE)
# A source named by author and year (Butcher 2008, Hairer et al. (1993)): a source, not a derivation.
_AUTHOR_YEAR_RE = re.compile(
    r"\b(?!(?:January|February|March|April|May|June|July|August|September|October|November|December)\b)"
    r"[A-Z][a-zÀ-ſ'\-]{2,}(?: et al\.?| (?:and|&) [A-Z][a-zÀ-ſ'\-]{2,})?,? \(?(?:1[6-9]\d\d|20\d\d)\b\)?")
# Words that say a value was taken from somewhere rather than worked out.
_RECALLED_RE = re.compile(
    r"\b(?:text ?book|handbook|recalled|from memory|well[- ]known value|known value|the literature)\b"
    r"|\b[A-Z][a-z]+[’']s (?:book|text|table|paper|monograph)\b",
    re.IGNORECASE,
)
# Steps that only point elsewhere ("from Butcher 2008", "see the handbook") are not steps.
_POINTER_RE = re.compile(r"^(?:from|see|in|as in|according to|following|per|cf\.?)\b", re.IGNORECASE)
# A step of a derivation states a relation (y(1) = exp(-1) = 0.3679); a number alone can be the recalled value.
_MATH_RE = re.compile(r"[=<>≈≤≥→]")
# What a second implementation says about the code it does not share with the simulation: a negation about code.
_NOT_SHARED_RE = re.compile(
    r"shares? no (?:code|function|module|routine|part)|(?:does not|doesn[’']t|do not|don[’']t) share\b"
    r"|no (?:shared|common) code|no code (?:in common|with|shared)"
    r"|(?:does not|doesn[’']t|without) (?:use|call|reuse|import|using|calling|reusing|importing)\b[^.;]{0,60}"
    r"\b(?:code|simulation|function|module|stepper|solver|routine|simulate\.py|run_trial|run_cell)\b"
    r"|independent(?:ly)? of the simulation(?:'s)?(?: code)?|written from scratch",
    re.IGNORECASE,
)
# A derivation shorter than this names one; it does not write its steps.
_MIN_STEPS = 20
# A title shorter than this (in words) is too generic to count as naming a source ("Monte Carlo Methods").
_MIN_TITLE_WORDS = 5


def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _labels_in(group: str) -> list[str]:
    """The labels of one bracket: ``1, 3`` gives 1 and 3; ``1-3`` gives 1, 2 and 3."""
    out: list[str] = []
    for part in re.split(r"\s*,\s*", group.upper()):
        bounds = re.split(r"\s*[–\-]\s*", part)
        if len(bounds) == 2 and bounds[0].lstrip("W").isdigit() and bounds[1].lstrip("W").isdigit():
            web = bounds[0].startswith("W")
            low, high = int(bounds[0].lstrip("W")), int(bounds[1].lstrip("W"))
            if 0 < high - low <= 50:
                out.extend(f"{'W' if web else ''}{n}" for n in range(low, high + 1))
                continue
        out.extend(b for b in bounds if b)
    return out


def _cites(text: str, sources: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    """``(what text cites that the quest retrieved, what it cites that it did not)``: a [n] label (alone, grouped or a
    range), a DOI, or the title of a retrieved source (five words or more) written into it."""
    sources = [s for s in sources if isinstance(s, dict)] if isinstance(sources, list) else []
    labels = {str(s.get("label") or "").upper() for s in sources}
    found, missing = [], []
    for group in _LABEL_RE.findall(text):
        for label in _labels_in(group):
            (found if label in labels else missing).append(f"[{label}]")
    dois = {_doi(s.get("doi")) for s in sources if s.get("doi")}
    for raw in _DOI_RE.findall(text):
        doi = _doi(raw)
        while doi.endswith(")") and doi.count(")") > doi.count("("):
            doi = _doi(doi[:-1])
        doi = doi.rstrip("]")
        (found if doi in dois else missing).append(doi)
    flat = f" {_words(text)} "
    for s in sources:
        title = _words(str(s.get("title") or ""))
        if len(title.split()) >= _MIN_TITLE_WORDS and f" {title} " in flat:
            found.append(f"[{s.get('label')}]")
    return list(dict.fromkeys(found)), list(dict.fromkeys(missing))


def _equations(protocol: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    model = protocol.get("model") if isinstance(protocol, dict) else None
    items = model.get("equations") if isinstance(model, dict) else None
    items = items if isinstance(items, list) else []
    return {str(e.get("id")).strip().upper(): e for e in items if isinstance(e, dict) and str(e.get("id") or "").strip()}


def _label_texts(source: str) -> list[str]:
    """The comments and the function and class docstrings of a Python source: where a label such as ``# E1`` is written.
    Code and other string literals are not labels (``E1`` as a variable name, or inside a formula string, says nothing
    about where E1 is), and neither is the module's docstring, which marks no code."""
    import ast
    import io
    import tokenize

    texts: list[str] = []
    try:
        texts += [tok.string for tok in tokenize.generate_tokens(io.StringIO(source).readline)
                  if tok.type == tokenize.COMMENT]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        texts += [line.split("#", 1)[1] for line in source.splitlines() if "#" in line]
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return texts
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                texts.append(doc)
    return texts


def generating_equations(protocol: dict[str, Any] | None) -> list[str]:
    """The ids of the plan's equations whose role is ``generates`` (the simulation computes the data with them), as the
    plan writes them."""
    model = protocol.get("model") if isinstance(protocol, dict) else None
    items = model.get("equations") if isinstance(model, dict) else None
    out = []
    for eq in items if isinstance(items, list) else []:
        if isinstance(eq, dict) and str(eq.get("role") or "").strip().lower() == "generates":
            eid = str(eq.get("id") or "").strip()
            if eid and eid not in out:
                out.append(eid)
    return out


_LABEL_RANGE_RE = re.compile(r"(?<![\w.])([A-Za-z]+)(\d+)\s*(?:-|–|—|\.\.|to)\s*\1?(\d+)(?!\w|\.\d)", re.IGNORECASE)


def _labelled_in(eid: str, texts: str) -> bool:
    """Whether ``eid`` is named in ``texts``, alone (``# E1``) or inside a range (``# E1-E3``, ``# E1 to 3``)."""
    if re.search(rf"(?<![\w.]){re.escape(eid)}(?!\w|\.\d)", texts, re.IGNORECASE):
        return True
    parts = re.fullmatch(r"([A-Za-z]+)(\d+)", eid)
    if not parts:
        return False
    prefix, number = parts.group(1).lower(), int(parts.group(2))
    return any(m.group(1).lower() == prefix and int(m.group(2)) <= number <= int(m.group(3))
               for m in _LABEL_RANGE_RE.finditer(texts))


def unlabelled_equations(protocol: dict[str, Any] | None, source: str | dict[str, str]) -> list[str]:
    """The ``generates`` equations of the plan whose id no comment or function docstring of the simulation carries (a
    label such as ``# E1`` on or above the code that implements it, or a range ``# E1-E3``). ``source`` is the
    simulation's text, or ``{file name: text}`` for a simulation with helper modules. Only the labelled mapping is
    read, not the mathematics: a label says where to look, not that the code there is right. Ids match in any case."""
    wanted = generating_equations(protocol)
    if not wanted:
        return []
    sources = source.values() if isinstance(source, dict) else [source]
    texts = "\n".join(t for s in sources for t in _label_texts(s or ""))
    return [eid for eid in wanted if not _labelled_in(eid, texts)]


def label_gaps(protocol: dict[str, Any] | None, source: str | dict[str, str], script: str) -> list[str]:
    """One sentence for the equations of the plan the simulation (``script``, its file name) does not say it implements
    (empty when every ``generates`` equation is labelled, or the plan lists none)."""
    missing = unlabelled_equations(protocol, source)
    if not missing:
        return []
    names = ", ".join(missing)
    return [f"{script} does not mark where it implements equation{'s' if len(missing) > 1 else ''} {names} of the model "
            f"behind the numbers (a comment such as `# {missing[0]}` on or above the code that computes it), so a wrong "
            "number cannot be traced to the equation or to the code"]


def _derivation_steps(text: str) -> str | None:
    """The steps after a leading "derivation" (``None`` when the text does not start with one)."""
    match = _DERIVATION_RE.match(text)
    return text[match.end():].strip() if match else None


def _steps_problem(steps: str, sources: list[dict[str, str]]) -> str | None:
    """Why the steps of a derivation are not steps a reader can follow (too short, only a pointer elsewhere, nothing to
    compute with, or a source the quest did not retrieve), or ``None``."""
    if len(steps) < _MIN_STEPS or not _MATH_RE.search(steps):
        return "does not write its steps (a relation such as y(1) = exp(-1) = 0.3679)"
    found, missing = _cites(steps, sources)
    if missing:
        return f"rests on {', '.join(missing)}, which this quest did not retrieve"
    named = [m.group(0).strip().rstrip(")") for m in _AUTHOR_YEAR_RE.finditer(steps)]
    if not found and (_RECALLED_RE.search(steps) or (_POINTER_RE.match(steps) and named)):
        return "takes its value from a source this quest did not retrieve, not from steps written out"
    if named and not found:
        return f"rests on {', '.join(dict.fromkeys(named))}, which is not a source this quest retrieved"
    return None


def equation_problem(eq: dict[str, Any], sources: list[dict[str, str]]) -> str | None:
    """Why an equation of the model has no source a reader can check, or ``None`` when it has one."""
    eid = str(eq.get("id") or "?").strip()
    source = " ".join(str(eq.get("source") or "").split())
    if not source:
        return (f"equation {eid} names no source: give the [n] of a source this quest retrieved, or `derivation` with "
                "the steps written in `derivation`")
    steps = _derivation_steps(source)
    if steps is not None:
        why = _steps_problem(" ".join(f"{steps} {eq.get('derivation') or ''}".split()), sources)
        return f"equation {eid} is marked as a derivation, but it {why}" if why else None
    found, missing = _cites(source, sources)
    if missing:
        return f"equation {eid} cites {', '.join(missing)}, which this quest did not retrieve"
    if found:
        return None
    return (f"equation {eid} cites a source this quest did not retrieve (“{source[:80]}”): a source recalled "
            "from memory does not count; cite a retrieved source by its [n], or write the derivation")


def reference_problem(oracle: dict[str, Any], protocol: dict[str, Any] | None,
                      sources: list[dict[str, str]]) -> str | None:
    """Why the oracle's expected value has no source a reader can check, or ``None`` when it has one."""
    name = str(oracle.get("name") or "?").strip()
    ref = " ".join(str(oracle.get("reference") or "").split())
    if not ref:
        return f"the check {name!r} does not say where its expected value comes from (its `reference` is empty)"
    steps = _derivation_steps(ref)
    if steps is not None:
        why = _steps_problem(steps, sources)
        return f"the check {name!r} names a derivation for its expected value, but it {why}" if why else None
    found, missing = _cites(ref, sources)
    if missing:
        return (f"the check {name!r} takes its expected value from {', '.join(missing)}, which this quest did not "
                "retrieve")
    equations = _equations(protocol)
    cited = [eid for eid in equations
             if re.search(rf"(?<![\w.]){re.escape(eid)}(?!\w|\.\d)", ref, re.IGNORECASE)]
    for eid in cited:
        why = equation_problem(equations[eid], sources)
        if why:
            return f"the check {name!r} rests on equation {eid}, and {why}"
    if cited or found:
        return None
    if kind_of(oracle) == "second_implementation" or re.search(r"(second|independent) (implementation|solver|method)", ref, re.I):
        if _NOT_SHARED_RE.search(ref):
            return None
        return (f"the check {name!r} compares with a second implementation but does not say what code it does not share "
                "with the simulation (write, for example, “shares no code with the simulation: it uses "
                "scipy's solve_ivp”)")
    unknown = [f"E{n}" for n in dict.fromkeys(re.findall(r"\bE(\d+)\b", ref))]
    if unknown:
        return (f"the check {name!r} rests on equation {', '.join(unknown)}, which the model behind the numbers does not "
                "list")
    return f"the check {name!r} takes its expected value from a source this quest did not retrieve (“{ref[:80]}”)"


def empty_references(protocol: dict[str, Any] | None) -> list[str]:
    """The sentence :func:`reference_problem` gives each declared oracle whose ``reference`` is empty: the one judgement
    that needs no list of the quest's sources."""
    return [reference_problem(o, protocol, []) or "" for o in declared(protocol)
            if not " ".join(str(o.get("reference") or "").split())]


#: How an expected value may be sourced, for the sentences a person reads when one is not.
SOURCE_FORMS = (
    "give its `reference` as `derivation: <the steps>`, a source this quest retrieved by its number in the plan's list of retrieved sources ([2]), an "
    "equation of the model behind the numbers (E1), or, for a second implementation, what code it does not share"
)


def source_gaps(protocol: dict[str, Any] | None, sources: list[dict[str, str]]) -> list[str]:
    """One sentence per declared oracle whose expected value has no source a reader can check (empty when each has one)."""
    return [why for why in (reference_problem(o, protocol, sources) for o in declared(protocol)) if why]


def model_notes(protocol: dict[str, Any] | None, sources: list[dict[str, str]]) -> list[str]:
    """What the plan's *Checks already made* says about the model behind the numbers and the kinds of its checks: no
    model, no equations, an equation with no role or no source a reader can check, a check whose kind is none of the six."""
    if not isinstance(protocol, dict):
        return []
    notes: list[str] = []
    model = protocol.get("model")
    if isinstance(model, str) and model.strip():
        notes.append("The model behind the numbers is one sentence: write its core equations too (`equations`), each "
                     "with its source.")
    elif not isinstance(model, dict):
        notes.append(
            "The plan does not say what model produces the numbers (`model` in the protocol: what it is, what it assumes, "
            "where it holds, and its equations, each with its source). Without it nobody can tell whether a wrong number "
            "comes from the model or from the code."
        )
    else:
        if not str(model.get("summary") or "").strip():
            notes.append("The model behind the numbers does not say, in a sentence, what model produces them (`summary`).")
        items = model.get("equations")
        equations = [e for e in items if isinstance(e, dict)] if isinstance(items, list) else []
        if not equations:
            notes.append("The model behind the numbers lists no equation: write its core equations, each with its source.")
        for eq in equations:
            eid = str(eq.get("id") or "?")
            if str(eq.get("role") or "") not in ("generates", "analyses"):
                notes.append(
                    f"Equation {eid} does not say its role: `generates` (it produces the data, in the simulation code) or "
                    "`analyses` (it is used on the results, in the analysis code)."
                )
            why = equation_problem(eq, sources)
            if why:
                notes.append(why[0].upper() + why[1:] + ".")
    for oracle in declared(protocol):
        if kind_of(oracle) is None:
            written = str(oracle.get("kind") or "").strip()
            notes.append(
                f"The check {str(oracle['name']).strip()!r} "
                + (f"has the kind {written!r}, which is none of the six" if written else "does not say its kind")
                + ": " + ", ".join(KINDS.values()) + "."
            )
    return notes


def unjudgeable(oracles: list[dict[str, Any]]) -> list[str]:
    """The declared oracles the engine cannot judge, one sentence each: they fix no numeric ``expected`` and ``tolerance``."""
    out: list[str] = []
    for oracle in oracles:
        expected, limit, mode = limit_of(oracle)
        name = str(oracle["name"]).strip()
        if limit is None and expected is not None and mode == "relative":
            out.append(f"the oracle {name!r} has a relative tolerance around an expected value of 0 (use an absolute tolerance)")
        elif limit is None:
            out.append(
                f"the oracle {name!r} fixes no numeric `expected` and `tolerance` in the protocol, so the engine cannot judge it "
                "(the script's own pass/fail is not evidence)"
            )
    return out


def judged(oracles: list[dict[str, Any]], reported: dict[str, Any] | None) -> list[dict[str, Any]]:
    """What the engine made of each declared oracle: its value as the script reported it, what the protocol expects and allows,
    and the verdict computed here. For the record in ``needs/ORACLE_CHECK.json``."""
    checks = (reported or {}).get("checks")
    by_name = {str(c.get("name") or "").strip().lower(): c for c in checks if isinstance(c, dict)} if isinstance(checks, list) else {}
    out: list[dict[str, Any]] = []
    for oracle in oracles:
        name = str(oracle["name"]).strip()
        check = by_name.get(name.lower())
        expected, limit, mode = limit_of(oracle)
        value = _num(check.get("value")) if check else None
        verdict = None if (value is None or limit is None) else abs(value - expected) <= limit  # type: ignore[operator]
        out.append({
            "name": name, "value": value, "expected": expected, "tolerance": oracle.get("tolerance"), "mode": mode,
            "limit": limit, "passed_by_engine": verdict, "script_said": (check or {}).get("passed"),
            "measured_by": "engine" if (reported or {}).get("engine_measured") and (check or {}).get("measured_by") == "engine" else "script",
        })
    return out


def script_measured(judged_list: list[dict[str, Any]]) -> list[str]:
    """The names of the judged oracles whose value the script reported (not one the engine measured by running the simulation)."""
    return [str(j["name"]) for j in judged_list if j.get("measured_by") != "engine" and j.get("value") is not None]


def engine_passed(judged_list: list[dict[str, Any]]) -> list[str]:
    """The names of the judged oracles the engine measured itself (by running the simulation on the oracle's case), got a
    number for, and passed. An oracle check counts as independent evidence only when this is not empty."""
    return [
        str(j["name"]) for j in judged_list
        if j.get("measured_by") == "engine" and _num(j.get("value")) is not None and j.get("passed_by_engine") is True
    ]


def not_passed(judged_list: list[dict[str, Any]]) -> list[str]:
    """The names of the judged oracles that got no verdict or failed (``passed_by_engine`` not ``True``)."""
    return [str(j["name"]) for j in judged_list if j.get("passed_by_engine") is not True]


def last_judged(record: Any) -> list[dict[str, Any]]:
    """The engine's verdicts from the last attempt of a ``needs/ORACLE_CHECK.json`` record that let the main run go on
    (``ok`` or ``warned``); empty for anything else."""
    if not isinstance(record, dict) or record.get("status") not in ("ok", "warned"):
        return []
    attempts = record.get("attempts")
    last = attempts[-1] if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict) else {}
    return [j for j in last.get("judged") or [] if isinstance(j, dict) and j.get("name")]


def analysis_note(judged_list: list[dict[str, Any]]) -> str:
    """What the analysis is told about the oracles the engine judged before the main run ('' when there were none).

    A script often re-judges its own oracles in its results with its own copy of each tolerance, and that copy can be
    out of date: in one real quest a person corrected a tolerance in the plan, the engine's check passed under it, and
    the analysis still reported the oracle as failed because the script's results carried the old value."""
    lines = []
    for j in judged_list:
        value, expected, limit = _num(j.get("value")), _num(j.get("expected")), _num(j.get("limit"))
        verdict = {True: "passed", False: "failed"}.get(j.get("passed_by_engine"), "not judged")
        verdict += "" if j.get("measured_by") == "engine" else " (the value is the script's own)"
        disputed = _num(j.get("disputed_expected"))
        if disputed is not None and j.get("passed_by_engine") is False:
            verdict += (
                f"; a repair says the expected value the protocol declares is itself wrong and proposes {_fmt(disputed)}, "
                "but nobody has approved that change, so the check counts as failed (say so: the check failed and its "
                "expected value is disputed)"
            )
        if value is None or expected is None or limit is None:
            lines.append(f"- {j['name']}: {verdict}")
        else:
            lines.append(f"- {j['name']}: measured {_fmt(value)}, expected {_fmt(expected)} within {_fmt(limit)}: {verdict}")
    if not lines:
        return ""
    return (
        "[FI NOTE] Before the main run the engine checked the protocol's oracles against the expected values and "
        "tolerances the protocol fixes:\n" + "\n".join(lines) + "\n"
        "These verdicts are the ones that count. If the results below also judge these oracles (a pass/fail flag, or the "
        "script's own copy of an expected value or tolerance), that copy can be out of date: report the engine's verdicts "
        "and numbers above, and do not report an oracle as failed or passed on the script's word.\n\n"
    )


def problems(oracles: list[dict[str, Any]], reported: dict[str, Any] | None, returncode: int, timed_out: bool = False) -> list[str]:
    """What is wrong with the oracle run, one sentence each; empty when every declared oracle ran and passed."""
    if not oracles:
        return [
            "the protocol declares no oracle, so nothing independent of the script's own numbers checks that they are right "
            "(add `oracles` to the protocol: a special or limiting case with a known answer, an invariant, a symmetry or "
            "scaling law, a convergence rate, a published benchmark value, or a second implementation)"
        ]
    if timed_out:
        return ["the script did not finish the oracle checks in time (run with FI_ORACLE=1: each check must be small and fast)"]
    if reported is None:
        return [
            "the script printed no `ORACLE_JSON:` line when run with FI_ORACLE=1 "
            f"(exit code {returncode}); it must run the declared oracle checks and print one"
        ]
    checks = reported.get("checks")
    checks = checks if isinstance(checks, list) else []
    by_name = {str(c.get("name") or "").strip().lower(): c for c in checks if isinstance(c, dict)}
    out: list[str] = []
    for oracle in oracles:
        name = str(oracle["name"]).strip()
        check = by_name.get(name.lower())
        expected, limit, mode = limit_of(oracle)
        if check is None:
            out.append(f"the declared oracle {name!r} was not checked (the script reported: {', '.join(sorted(by_name)) or 'nothing'})")
        elif limit is None:
            out.append(unjudgeable([oracle])[0])
        elif _num(check.get("value")) is None:
            out.append(f"the oracle {name!r} reported no finite numeric `value` (it reported {check.get('value')!r}): the script measures, the engine judges")
        else:
            value = _num(check.get("value"))
            if abs(value - expected) > limit:  # type: ignore[operator]
                out.append(
                    f"the oracle {name!r} failed: the script measured {_fmt(value)}, the protocol expects {_fmt(expected)} "
                    f"within {_fmt(limit)} ({mode} tolerance {_fmt(_num(oracle.get('tolerance')))})"
                )
            elif check.get("passed") is False:
                out.append(f"the oracle {name!r} is within its tolerance, but the script reports the check as failed itself: find out why")
    if not out and returncode != 0:
        out.append(f"every declared oracle passed, but the script exited with code {returncode}")
    return out


def with_run_problems(run_problems: list[str], found: list[str], oracles: list[dict[str, Any]] | None = None) -> list[str]:
    """``found`` led by the reasons a value could not be measured, without the sentences that only repeat them: a missing
    value or one that is not a number for an oracle a run problem already names (all the oracles that have no case when
    the script's own ``oracle()`` gave nothing), and the non-zero exit / timeout of the run. What is about a value that was
    measured (a failed check) or about another oracle stays."""
    named = " ".join(run_problems)
    covered = [str(o["name"]).strip() for o in (oracles or []) if repr(str(o["name"]).strip()) in named]
    if any(p.startswith("simulate.py's oracle()") for p in run_problems):
        covered += [str(o["name"]).strip() for o in (oracles or []) if case_of(o) is None]

    def repeats(sentence: str) -> bool:
        if "exited with code" in sentence or "did not finish" in sentence:
            return True
        return ("was not checked" in sentence or "no finite numeric" in sentence) and any(repr(n) in sentence for n in covered)

    return list(run_problems) + [f for f in found if not repeats(f)]


def undisputed(found: list[str], disputed: Any) -> list[str]:
    """The problems in ``found`` that a repair may still fix: all but a disputed check's value judged outside its
    tolerance (named in ``disputed``, a check a repair has called wrong). A dispute is about the expected value, so any
    other problem with a disputed check (not measured, not a number, the simulation failing on its case) and a problem
    that names no oracle (a crash, no ORACLE_JSON line) is the script's to fix."""
    heads = tuple(f"the oracle {str(n).strip()!r} failed:" for n in disputed or [])
    return [f for f in found if not (heads and f.startswith(heads))]


def disputed_failing(judged_list: list[dict[str, Any]], disputed: Any) -> list[str]:
    """The disputed checks among ``judged_list`` whose measured value the engine judged outside the tolerance (a check
    not measured at all is not one: nothing was judged against the disputed expected value)."""
    names = {str(n).strip() for n in disputed or []}
    return [str(j["name"]) for j in judged_list if str(j.get("name")) in names and j.get("passed_by_engine") is False]


def directive(oracles: list[dict[str, Any]], found: list[str], disputed: list[str] | None = None) -> str:
    """What stands where a traceback would in the repair request. ``disputed``: the checks an earlier repair called wrong
    (their proposals wait for a person); they are left out of what to fix, and the script is not to be changed for them."""
    declared_block = json.dumps(oracles, indent=2)
    set_aside = (
        "Checks an earlier repair already said are wrong themselves: " + ", ".join(repr(str(n)) for n in disputed) + ". "
        "Their proposed changes are recorded and wait for a person. Do not change the script for these checks (do not "
        "change how their values are computed to reach the declared expected value, skip them or hard-code their "
        "values): they go on failing until a person decides. If one of them is listed below as not checked or not a "
        "number, restore its honest measurement, nothing more. Fix only what is listed below.\n\n"
    ) if disputed else ""
    return (
        "This script has NOT run its experiment yet: it was run with the environment variable FI_ORACLE=1 to check its "
        "oracles, and that check did not pass. The account of a crash above does not apply.\n\n"
        "The oracles the design declares (independent of the script's own numbers):\n" + declared_block + "\n\n"
        + set_aside +
        "What went wrong:\n" + "\n".join(f"- {p}" for p in found) + "\n\n"
        "The contract: when FI_ORACLE is 1 the script must NOT run its sweep. It MEASURES each declared oracle on a small, fast "
        "case (seconds) and prints ONE line `ORACLE_JSON: {\"checks\": [{\"name\": <the declared name>, \"value\": <the "
        "number it measured>, \"diagnostics\": {...}}, ...]}`, then exits 0. It does NOT decide pass or fail and it does not "
        "state the expected value or the tolerance: the engine judges the value against the `expected` and `tolerance` the "
        "protocol fixes above (for an invariant the value is the worst violation observed, and the expected value is 0).\n\n"
        "If a value is outside its tolerance, find out which is wrong before changing anything: the simulator or estimator (fix "
        "it) or the way the value is measured (fix that), and say which in `patch_summary`. Never make a check pass by "
        "measuring something else, skipping it or hard-coding its value: a check the script can always pass is not an oracle. If "
        "the checks were missing, add them for every declared oracle.\n\n"
        "The check itself can be what is wrong: an `expected` or `tolerance` the method cannot reach on that case (below its "
        "known error at that step or sample size), or a measurement that is not well defined (a convergence order read far "
        "from the asymptotic regime, two methods compared on different quantities), or an `expected` value that is simply "
        "miscalculated. You cannot change the protocol, and you must not bend the script to hide it: if the declared "
        "expected value is wrong and the script measures the right one, do NOT change the code to match it. Instead return "
        "`oracle_change`: a list of {\"name\": <the declared "
        "name>, \"expected\": <number>, \"tolerance\": <number>, \"tolerance_mode\": \"absolute\" | \"relative\", \"check\": "
        "<the corrected check, only if the measurement itself must change>, \"reason\": <the method's known error or the flaw, "
        "with the numbers>}, and leave `code` empty: the script is kept as it is. Code returned together with an "
        "`oracle_change` for a check not already listed as disputed is not used (it cannot be told apart from code bent to "
        "that check); when other problems need a fix too, you are asked for it again without the disputed check. A person "
        "decides whether to accept the change; nothing changes without them.\n\n"
        "Keep everything else unchanged: the same functions, outputs and figures, the handling of FI_PILOT and "
        "FI_REPLICATE_SEED, and the same final RESULT_JSON line. When you fix the script, return the whole script in "
        "`code`; always give one sentence in `patch_summary`, and leave `give_up_reason` empty."
    )


def proposals(raw: Any, oracles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The usable entries of a repair's ``oracle_change``: each names a declared oracle, gives a finite ``expected`` and a
    non-negative ``tolerance``, and says why. Anything else is dropped, never guessed at."""
    names = {str(o.get("name")).strip().lower(): str(o.get("name")).strip() for o in oracles}
    out: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = names.get(str(item.get("name") or "").strip().lower())
        expected, tolerance = _num(item.get("expected")), _num(item.get("tolerance"))
        reason = str(item.get("reason") or "").strip()
        if name is None or expected is None or tolerance is None or tolerance < 0 or not reason:
            continue
        mode = "relative" if str(item.get("tolerance_mode") or "").strip().lower() == "relative" else "absolute"
        entry: dict[str, Any] = {"name": name, "expected": expected, "tolerance": tolerance, "tolerance_mode": mode, "reason": reason[:600]}
        check = str(item.get("check") or "").strip()
        if check:
            entry["check"] = check[:600]
        out.append(entry)
    return out


def proposal_request(proposal: dict[str, Any]) -> str:
    """The ``--revise-plan`` request that applies one proposal to the plan, word for word.

    It is shown inside a double-quoted command a person copies into a shell, and the check and the reason are the model's
    own words: a quote, ``$``, a backtick or a backslash in them could end the argument or run something when pasted
    (``$(...)`` in bash, a backtick escape in PowerShell). Those become plain characters, so the pasted command only
    ever carries text."""
    change = f"expected {_fmt(proposal['expected'])}, tolerance {_fmt(proposal['tolerance'])} ({proposal['tolerance_mode']})"
    if proposal.get("check"):
        change += f", and its check reads: {proposal['check']}"
    return _shell_safe(f"Change the oracle '{proposal['name']}' to {change}. Reason: {proposal['reason']} Change nothing else.")


def _shell_safe(text: str) -> str:
    """``text`` with nothing a shell acts on inside double quotes: ``"`` and the curly double quotes PowerShell also
    ends a string on become ``'``, ``!`` (bash history) becomes ``.``, ``$``, backticks and backslashes are dropped, and
    line breaks become spaces."""
    text = re.sub(r'["“”„‟]', "'", text).replace("!", ".")
    text = re.sub(r"[$`\\]", "", text)
    return re.sub(r"\s+", " ", text).strip()
