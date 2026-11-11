"""Two checks made on the simulation before and while it runs, as methods the engine mixes in.

* **A reported result is computed** (``core/typed_results.py``): the simulation is read, never run, and a returned value that
  is a number typed into the code is sent back to be computed (or removed), at most twice; one that is still typed in
  afterwards is left out of the results the paper may use, and the paper says so.
* **The run fits the time it is allowed** (``core/run_estimate.py``): a few real trials are timed before the freeze and
  added up against the topic's own time limit when it states one (``execution.timeout_s`` stays FI's hard stop) or else
  ``execution.timeout_s``; a study that would not finish gets fewer runs per setting from FI itself (no model is asked;
  every setting is kept; the runs go down to the fewest the plan allows, and below it only for the topic's own limit,
  said in the paper with the precision those runs give); one that would not fit FI's own limit even so stops the quest
  plainly with no paper, and one that cannot meet the topic's limit runs at the plan's fewest and says it was over.
* **The simulation measures what the protocol names** (the same trials, ``RunChecksMixin._pilot_gate``): what the timed trials
  returned is read before the full run; a metric that is not a number in any pilot trial at any pilot setting (after the
  settings with few values were tried in every combination), one that never moves while others do, or an output that is the
  same everywhere goes to the existing repair of the simulation with the exact finding, and what the repairs leave ends in
  the honest stop in minutes instead of after the study.

They live here and not in ``engine.py`` so the node only calls them. Every method uses the engine's own attributes
(``quest_root``, ``fi_dir``, ``config``, ``_log``, ``_chat`` ...).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import replace as _replace
from pathlib import Path
from typing import Any

from . import cost_model as _cost_model
from . import execution as _execution
from . import frozen_protocol as _frozen
from . import numeric_warnings as _numeric
from . import in_run as _in_run
from . import optimise_search as _optimise_search
from . import oracle_forms as _forms
from . import plan as _plan
from . import run_estimate as _estimate
from .result_checks import TYPED_RECORD
from . import skeleton as _skeleton
from . import split_run as _split_run
from . import stop_power as _stop_power
from . import trial_runner as _trial_runner
from . import typed_results as _typed

#: How many times, in all, a simulation that returns typed-in numbers is sent back.
TYPED_RESULT_REPAIRS = 2
#: In ``.fi/``: the settings FI took out of the grid because the simulation never reads them.
NO_DIFFERENCE_RECORD = "NO_DIFFERENCE_OBSERVED.json"
_COUNTER = "typed_results.json"
#: How many times the gate sends a fault of the analysis on the recorded results back to be repaired in a script's life. The
#: repairs have a counter of their own (``gate_repairs``): crash repairs of the simulation never leave them with none, and
#: they spend none of the crash repairs.
GATE_REPAIRS = 2

# A number is written precisely enough to be recognised as a value (so that a note never repeats a left-out one): three
# significant digits, or two written to four decimals ("0.0068"). Below that a number is a round figure, a setting or a label.
_MIN_SIG_DIGITS = 3
_LOW_PRECISION_SIG_DIGITS = 2
_LOW_PRECISION_MIN_DECIMALS = 4


def _carries_enough_precision(token: str) -> bool:
    """Is this number written precisely enough to be recognised as a value?"""
    t = token.lstrip("-+").replace(",", "")
    decimals = len(t.split(".", 1)[1]) if "." in t else 0
    if "." in t:
        whole, frac = t.split(".", 1)
        whole = whole.lstrip("0")
        sig = len(whole) + len(frac) if whole else len(frac.lstrip("0")) or len(frac)
    else:
        sig = len(t.strip("0")) or 1
    return sig >= _MIN_SIG_DIGITS or (sig >= _LOW_PRECISION_SIG_DIGITS and decimals >= _LOW_PRECISION_MIN_DECIMALS)


def _one(text: str, limit: int = 400) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit - 3] + "..."


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def typed_result_exempt(protocol: dict[str, Any] | None, design: dict[str, Any] | None = None) -> set[str]:
    """Names the plan fixes (its fixed settings, the grid's settings, what each run compares, the thresholds): echoed back,
    not results. The one place that list is made (the engine's check and :mod:`core.code_checks` both read it)."""
    protocol = protocol if isinstance(protocol, dict) else {}
    design = design if isinstance(design, dict) else {}
    names = set(_forms.fixed_settings_of_design({**design, "protocol": protocol}))
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    names |= {str(k) for k in grid} | {str(k) for k in _in_run.of(protocol) or {}}
    thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else {}
    return names | {str(k) for k in thresholds}


#: The metric keys that hold a model's answer about the contract, not the plan's own statement of the study.
_CONTRACT_ANSWERS = ("role", "depends_on", "returned")


def settings_the_study_compares(fi_dir: Path, design: Any, protocol: dict[str, Any], axes: list[str]) -> dict[str, str]:
    """``{setting: why it is part of the study}`` for each of ``axes`` that the study compares, from FI's records: a
    factor the topic compares (:func:`core.topic_requirements.compared_axes`, read from ``fi_dir``), a key of a check's
    case, of ``in_each_run``, or a name a metric or the plan's comparisons use."""
    from . import in_run as _in_run
    from . import oracle_check as _oracle
    from . import topic_requirements as _tr

    design = design if isinstance(design, dict) else {}
    out: dict[str, str] = {}
    req = _read_json(Path(fi_dir) / _tr.FILE)
    if isinstance(req, dict) and isinstance(req.get("factors"), list):
        for axis, factor in _tr.compared_axes(req, design, protocol).items():
            out.setdefault(axis, f"the topic compares {factor}")
    for oracle in _oracle.declared(protocol):
        own = _oracle.case_of(oracle)
        for axis in (own[0] or {}) if own else {}:
            out.setdefault(str(axis), f"the check `{oracle.get('name')}` runs at a value of it")
    for axis in _in_run.of(protocol) or {}:
        out.setdefault(str(axis), "each run compares its values")

    def names(value: Any) -> set[str]:
        if isinstance(value, str):
            return {value}
        if isinstance(value, dict):
            return {n for v in value.values() for n in names(v)}
        if isinstance(value, list):
            return {n for v in value for n in names(v)}
        return set()

    for metric in protocol.get("metrics") or []:
        # `depends_on` and `role` are a model's answer to a question about the contract (core/contract_parts.py): what a
        # result "may change with" is a superset, read by no check, so it never makes a setting one the study compares.
        for axis in names({k: v for k, v in metric.items() if k not in ("id", "description", "kind", *_CONTRACT_ANSWERS)}
                          if isinstance(metric, dict) else {}):
            out.setdefault(axis, f"the metric `{metric.get('id')}` is given per value of it")
    # A comparison marked `asked` is the model's answer to FI's question after the plan, read by no check.
    planned = [c for v in (protocol.get("comparisons"), design.get("comparisons"))
               for c in (v if isinstance(v, list) else [v]) if not (isinstance(c, dict) and c.get("asked"))]
    for axis in names(planned):
        out.setdefault(axis, "the plan compares its values")
    return {axis: why for axis, why in out.items() if axis in axes}


def _literal_keys(files: dict[str, str]) -> set[str]:
    """The names the code's text looks up by a literal key (``x["name"]``, ``x.get("name")``) anywhere. Used only to hold a
    repair back: a setting FI's small run saw no read of, whose name the code does look up somewhere, is read under a
    condition the small run did not reach, not ignored."""
    import ast

    out: set[str] = set()
    for text in files.values():
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                key = node.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    out.add(key.value)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr in ("get", "pop", "setdefault") and node.args
                  and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                out.add(node.args[0].value)
    return out


def unread_findings(files: dict[str, str], protocol: dict[str, Any], design: Any, fi_dir: Path) -> tuple[
        list[Any], dict[str, str], list[str], str]:
    """FI's reading of which settings of ``protocol.grid`` the code never reads: ``(misread, compared, unread, why)``.
    ``misread``: reads of a name the plan does not give (a fault the static check owns: nothing else is said then).
    ``compared``: ``{setting: why}`` for a swept setting the code never reads that the study compares (a fault of the
    code; :func:`settings_the_study_compares`). ``unread``: the swept settings nothing reads (the compared ones among
    them), or ``[]`` with ``why`` when the reading is not certain. Nothing is run and no model is asked."""
    from . import grid_keys as _grid_keys
    from . import unread_settings as _unread

    misread = _grid_keys.problems(files, protocol, design)
    if misread:
        return misread, {}, [], ""
    grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
    # A setting the study compares and the simulation itself never reads is the simulation's fault, even when the
    # analysis reads it (then it groups results that differ in nothing but their seeds).
    swept = [str(a) for a, v in grid.items() if isinstance(v, list) and len(v) >= 2]
    simulated, _why = _unread.setting_reads(files, analysis=False)
    if simulated is not None and any(a in simulated for a in swept):
        compared = settings_the_study_compares(fi_dir, design, protocol, [a for a in swept if a not in simulated])
        if compared:
            return [], compared, [], ""
    unread, why = _unread.unread_axes(files, protocol)
    return [], (settings_the_study_compares(fi_dir, design, protocol, unread) if unread else {}), unread, why


#: How many designs of a search's own space (besides its baseline) a speed-up is held to.
SEARCH_SAMPLE_DESIGNS = 3


def _search_rounds(block: dict[str, Any], search_n: int, at_once: int) -> tuple[int, int, int]:
    """How the search's ``search_n`` evaluations wait, from the search itself rehearsed on a stand-in objective
    (:func:`core.optimise_search.rehearse`, with the run's own ``batch``): ``(alone, batches, together)``: the evaluations
    asked for one at a time, the rounds of ``at_once`` the designs asked for together run in, and how many designs those
    are. Evaluations the rehearsal did not spend (a start that converged early on the stand-in) are counted alone, and a
    search that cannot be rehearsed (a library method) is counted one at a time."""
    widths = _optimise_search.rehearse(block, batch=at_once > 1) if at_once > 1 else None
    if not widths:
        return search_n, 0, 0
    alone = sum(w for w in widths if w == 1)
    together = sum(w for w in widths if w > 1)
    batches = sum(math.ceil(w / at_once) for w in widths if w > 1)
    return alone + max(0, search_n - alone - together), batches, together


def _search_sample(block: dict[str, Any], grid: Any) -> list[dict[str, Any]]:
    """The settings a search's speed-up is held to besides its coarse scan's: the baseline and
    :data:`SEARCH_SAMPLE_DESIGNS` designs spread over the search's space (fixed points of it, so the sample is the same on
    every run), each with the scan's other settings at their base values. A design the scan never computes (an array
    of free values, say) is where a search spends its time, so a change that alters a number there is caught too."""
    try:
        space = _optimise_search.Space(block["design_variables"])
        base = dict(next(iter(_estimate.probe_cells(grid)))[2])
        baseline = dict((block.get("baseline") or {}).get("values") or {})
    except (KeyError, TypeError, ValueError, StopIteration):
        return []
    out = [{**base, **baseline}] if baseline else []
    n = SEARCH_SAMPLE_DESIGNS
    for i in range(n):
        # points of the unit cube spread over it by the golden ratio (each coordinate its own offset), off its centre,
        # where a baseline often sits
        u = [(0.5 + 0.618034 * (i + 1) + 0.37 * k) % 1.0 for k in range(space.dims)]
        try:
            cell = {**base, **space.design(u)}
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if cell not in out:
            out.append(cell)
    return out


def _whole_number(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0



# Rules of the single-script request that a frame written by FI makes untrue (see :meth:`_frame_template`).
_FRAME_DROP_BULLETS = ("- Single Python file.", "- **Honour `FI_PILOT`", "- **Honour `FI_REPLICATE_SEED`",
                       "- **Answer the design's oracles when asked.**")
#: Bullets of the single-script request that are the analysis's, when FI writes the analysis too (``bodies`` below): FI makes and
#: saves the figures and prints the results, so the model is not told how.
_ANALYSIS_DROP_BULLETS = ("- Save figures to", "- Save the tables and arrays", "- **A figure holds at most 3 panels",
                          "- Write a one-line JSON", "- **Report the trials behind a probability", "- **Stratify when",
                          "- The last line of stdout MUST", "- **Keep `RESULT_JSON`")


_ANALYSIS_SWAPS = (
    ("- **Figure styling is automatic",
     "- **Figure styling is automatic - don't fight it.** FI applies a house style to every figure. Do NOT call "
     "`plt.style.use(...)`, touch `rcParams`/`rcdefaults()`, call `seaborn.set_*`, or hard-code colors/colormaps. Make a figure "
     "read well instead: label every axis in words with its unit (the plan's `label` for the quantity when `protocol.labels` or "
     "the metric gives one), never with a variable name from the code, give it a short descriptive title, prefer direct series "
     "labels over a boxed legend, and annotate the single number that matters. A figure with several panels draws one on each "
     "of `fig.axes`."),
    ("- **Draw every figure from the run's results",
     "- **Draw every figure from `table`, never from numbers typed into the plotting code.** A plotting call (`plot`, "
     "`scatter`, `bar`, `errorbar`, ...) is given values read from `table` (`table.mean`, `table.across`, ...), not a list written "
     "out in the function such as `ax.plot(xs, [4, 5, 6, 8])`. FI reads the function before it runs and sends such a figure back. "
     "Axis limits, ticks and reference lines (a limit, a target) may be written in the code."),
    ("- **Report what the method produced",
     "- **Report what the method produced, even when it is bad.** Never clamp, cap or clip a result into the range the design "
     "expects. A quantity that is undefined at a setting is `table.missing(\"<a few words why>\")` in a function of derive.py "
     "(in `trial`, return `None` and give the reason beside it as `out[\"why_missing\"]`); never a stand-in number (0, -1, a bound, a marker value) - a "
     "result pinned to a bound is rejected."),
)

#: The two figure bullets of the code template when the figure functions are given FI's drawing calls (a frame of version 5), by the
#: start of the bullet they stand for: FI writes every label, the title and the legend, and a drawing call is given what is read
#: from `table`.
_CALLS_SWAPS = {
    "- **Figure styling is automatic": (
        "- **Figure styling is automatic - don't fight it.** FI applies a house style to every figure and writes every axis label, "
        "the title and the legend itself. Do NOT call `plt.style.use(...)`, touch `rcParams`/`rcdefaults()`, call `seaborn.set_*`, "
        "or hard-code colors/colormaps. A figure with several panels draws each in its own panel (the `panel` of a drawing call)."),
    "- **Draw every figure from the run's results": (
        "- **Draw every figure from `table`, never from numbers typed into the plotting code.** A drawing call (`fig.line`, "
        "`fig.points`, `fig.bars`, ...) is given values read from `table` (`table.mean`, `table.across`, ...), not a list written "
        "out in the function such as `fig.line(0, series, [1, 2, 3, 4], [4, 5, 6, 8], what)`. FI reads the function before it runs "
        "and sends such a figure back. Reference lines (a limit, a target) are drawn with `fig.ref`."),
}

_OUTLINE_SWAPS = (
    ("- FI writes the frame of simulate.py", None),
    ("- Your scaffold MUST already include",
     "- Your scaffold holds the imports and a STUB for `trial` and for each function you list. It sets up no figures, prints no "
     "results and has no main: FI writes those. The body stage only fills in function bodies; it does NOT rewrite the contract."),
    ("- Plan every figure as at most 3 panels",
     "- A figure is a function `figure_<id>(fig, table)` that draws on the axes FI made for it, at most 3 panels side by side. A "
     "comparison that needs more panels (a grid over two factors, more than three strata) becomes SEVERAL figures of at most 3 "
     "panels each, one function per figure, because a figure is shown at a fixed width and every extra panel shrinks the text of "
     "all of them."),
    ("- Stratification:", ""),
    ('  "scaffold":',
     '  "scaffold": "<Python source: the imports and the function STUBS that raise NotImplementedError, with no figure setup and no '
     'printed results. ~10-40 lines.>",'),
    ('  "data_flow":',
     '  "data_flow": "<one-paragraph plan: input variables, function order, what each function returns. The body stage uses this to '
     'keep the code linear.>",'),
    ('  "result_json_template":', '  "result_json_template": "{}",'),
)


def _swap_lines(text: str, swaps: Sequence[tuple[str, str | None]]) -> str:
    """``text`` with each line that starts with a swap's first words replaced by its text (``""``: the line goes; a bullet's
    continuation lines go with it)."""
    out, skipping = [], False
    for line in text.split("\n"):
        hit = next((s for s in swaps if line.startswith(s[0]) and s[1] is not None), None)
        if hit is not None:
            skipping = line.startswith("- ")
            if hit[1]:
                out.append(hit[1])
            continue
        if skipping and line.startswith(("  ", "\t")):
            continue
        skipping = False
        out.append(line)
    return "\n".join(out)


def _cluster_bodies() -> str:
    """The cluster section of the request when FI writes the analysis (the engine's text, for the reply of that request)."""
    from . import engine as _e

    text = _e._CLUSTER_PROTOCOL
    for old, new in (
            ("Reply with a THIRD fenced block, starting with the line `# file: submit.py`, after the other two.",
             "Reply with one more fenced block, starting with the line `# file: submit.py`, after the other blocks."),
            ("FI then reads every task's results itself, writes the record and runs experiment.py.",
             "FI then reads every task's results itself, writes the record and runs the analysis."),
            ("submit.py never reads or writes the tasks' results, and experiment.py does not submit anything: it reads FI_TRIALS as\n"
             "above.", "submit.py never reads or writes the tasks' results.")):
        text = text.replace(old, new)
    return text


def panel_statement(panels: int, calls: bool = False) -> str:
    """What FI tells the model of the axes it makes for a figure: how many, and that the count is the plan's, not the number of
    values of a setting or of results the figure shows. ``calls``: the figure is drawn with FI's drawing calls, whose first
    argument is the panel."""
    if calls:
        if panels <= 1:
            return ("FI makes ONE panel for it, panel 0: give `panel=0` to every drawing call. Draw everything the figure shows on "
                    "it (several settings, series or results are lines in that one panel, not more panels).")
        return (f"FI makes exactly {panels} panels for it, numbered 0 to {panels - 1} (the `panel` of a drawing call), no more. "
                "The count is the plan's, not the number of values of a setting. When a panel shows one setting value, say so "
                "with `fig.panel(i, setting=value)`.")
    if panels <= 1:
        return ("FI makes ONE panel for it: `fig.axes` holds exactly one axes, `fig.axes[0]`. Draw everything the figure shows on "
                "it (several settings, series or results are lines on that one axes, not more panels).")
    return (f"FI makes exactly {panels} panels for it: `fig.axes[0]` to `fig.axes[{panels - 1}]`, no more. The count is the plan's, "
            "not the number of values of a setting: index `fig.axes` by panel, and when a panel shows one setting value, say so "
            "with `fig.panel(i, setting=value)`.")


def _bodies_reply(bodies: Sequence[str], cluster: bool, panels: dict[str, int] | None = None, calls: bool = False) -> str:
    """The reply format when FI writes the analysis as well (experiment.py is FI's): ``trial``, then the bodies of derive.py
    that the plan needs (none: no block), then the driver of a cluster, then the dependencies. ``panels``: the panels FI
    makes for each figure body (its name to the count), stated under that body's signature."""
    blocks = ["A fenced Python block starting with the line `# file: simulate.py`: the function `trial` and the imports and "
              "helper functions it needs (nothing of FI's frame)."]
    if bodies:
        listed = "\n".join(f"   - `{_skeleton.body_signature(b) or 'def ' + b + '(...)'}`"
                           + (f" {panel_statement(panels[b], calls)}" if panels and b in panels else "") for b in bodies)
        blocks.append("A fenced Python block starting with the line `# file: derive.py`: ONLY these functions, each given "
                      "whole with exactly the parameters shown, and the imports and helper functions they need (nothing of "
                      "FI's frame; FI writes experiment.py, you do not):\n" + listed)
    if cluster:
        blocks.append("A fenced Python block starting with the line `# file: submit.py`: the driver that submits the job array, "
                      "as the section on the cluster says.")
    blocks.append("A single line `DEPS:` followed by a comma-separated list of pip-installable package names. Use the outline's "
                  "`deps` as the baseline; add only if you genuinely needed something more.")
    return ("# Output format\nRespond with EXACTLY these, in this order, and nothing else:\n\n"
            + "\n".join(f"{i}. {b}" for i, b in enumerate(blocks, 1)) + "\n\n")


def _frame_reply(cluster: bool = False, bodies: Sequence[str] | None = None, panels: dict[str, int] | None = None,
                 calls: bool = False) -> str:
    """The reply format of a request where FI writes the frame of simulate.py; with ``cluster`` (the trials run as a job
    array) the driver `submit.py` the cluster section asks for is a block of the format too. With ``bodies`` (the functions
    of derive.py the plan needs, possibly none) FI writes experiment.py too and it is not asked for."""
    if bodies is not None:
        return _bodies_reply(bodies, cluster, panels, calls)
    return ("# Output format\nRespond with EXACTLY these, in this order, and nothing else:\n\n"
            "1. A fenced Python block starting with the line `# file: simulate.py`: the function `trial` and the imports and "
            "helper functions it needs (nothing of FI's frame).\n"
            "2. A fenced Python block starting with the line `# file: experiment.py`: the analysis.\n"
            + ("3. A fenced Python block starting with the line `# file: submit.py`: the driver that submits the job array, as "
               "the section on the cluster says.\n4." if cluster else "3.")
            + " A single line `DEPS:` followed by a comma-separated list of pip-installable package names. Use the outline's "
            "`deps` as the baseline; add only if you genuinely needed something more.\n\n")
_FRAME_BODY_INTRO_BODIES = (
    "You are the **Implementation Body** stage of an automated research pipeline.\n\n"
    "A prior stage produced a structural outline for the experiment. Your job: write the code the output format below asks for. "
    "You do NOT change the signatures of the outline's functions that you keep, and you write nothing of FI's frame or of "
    "FI's analysis. The outline locked those decisions.\n\n")
_FRAME_BODY_INTRO = (
    "You are the **Implementation Body** stage of an automated research pipeline.\n\n"
    "A prior stage produced a structural outline for the experiment. Your job: write the code the output format below asks for. "
    "You do NOT change the signatures of the outline's functions that you keep, you do NOT rewrite the RESULT_JSON contract, "
    "you do NOT alter the imports or the figures-dir setup of the scaffold. The outline locked those decisions.\n\n")


def frame_template_text(text: str, cluster: bool = False, bodies: Sequence[str] | None = None,
                        panels: dict[str, int] | None = None, calls: bool = False) -> str:
    """``text`` (a code-writing template) for a request where FI writes the frame of simulate.py: the bullets about a single
    script, the pilot, the seed and the oracle run are taken out, the reply format is the frame's, and the template does not
    say the whole scaffold is filled. ``bodies``: FI writes the analysis too (see :func:`_frame_reply`); the bullets about
    figures and ``RESULT_JSON`` are then taken out as well. ``calls``: the figure functions are given FI's drawing calls
    (a frame of version 5), so the texts about figures give the rule of :data:`core.skeleton._FIGURE_RULE`, not ``fig.axes``."""
    drop = _FRAME_DROP_BULLETS if bodies is None else _FRAME_DROP_BULLETS + _ANALYSIS_DROP_BULLETS
    out, skipping = [], False
    for line in text.split("\n"):
        swap = next((s for s in _ANALYSIS_SWAPS if line.startswith(s[0])), None) if bodies is not None else None
        if swap is not None:
            out.append(_CALLS_SWAPS.get(swap[0], swap[1]) if calls else swap[1])
            skipping = True
            continue
        if line.startswith("- "):
            skipping = line.startswith(drop)
        elif skipping and (not line.strip() or line.startswith("#")):
            skipping = False
        if not skipping:
            out.append(line)
    text = "\n".join(out)
    text = text.replace("; the single-file constraint still applies, since an import is not a second file you author",
                        "; an import is not a second file you author")
    a, b = text.find("# Output format"), text.find("# Inputs")
    if 0 <= a < b:
        text = text[:a] + _frame_reply(cluster, bodies, panels, calls) + text[b:]
    c = text.find("# The topic, as the person wrote it")
    if c > 0 and text.startswith("You are the **Implementation Body**"):
        text = (_FRAME_BODY_INTRO if bodies is None else _FRAME_BODY_INTRO_BODIES) + text[c:]
    return text


def frame_outline_text(text: str, bodies: Sequence[str] | None = None) -> str:
    """``text`` (the outline template) for a request where FI writes the frame of simulate.py. ``bodies``: FI writes the
    analysis (experiment.py) too, so the outline lists the functions of derive.py instead of the analysis."""
    text = text.replace(
        "- The next stage will write a single Python file constrained to:",
        "- FI writes the frame of simulate.py (its entry points, the loop over settings and trials, the seeds and the main). "
        "The outline lists the function `trial` (ONE trial: it takes the settings and returns the plan's results) and the "
        "helper functions it uses, which are the equation and step functions below, and, for experiment.py, the functions of "
        "the analysis (statistics, figures, `RESULT_JSON`). It lists no entry point, no loop over settings or trials and no "
        "main for the simulation. The next stage writes `trial` and its helpers, and experiment.py, constrained to:")
    text = text.replace("(those belong to the scenario, in the other functions)",
                        "(those belong to FI's frame, or to the analysis)")
    if bodies is not None:
        first = ("- FI writes the frame of simulate.py (its entry points, the loop over settings and trials, the seeds and the "
                 "main) and all of experiment.py (the statistics, the figure files and the printed results). The outline lists the "
                 "function `trial` (ONE trial: it takes the settings and returns the plan's results) and the helper functions it "
                 "uses, which are the equation and step functions below"
                 + (", and the functions of derive.py: " + ", ".join(f"`{b}`" for b in bodies)
                    + " (each reads only the table of FI's record)" if bodies else "; the plan needs no function of derive.py")
                 + ". It lists no entry point, no loop over settings or trials, no main and nothing of the analysis. The next "
                 "stage writes these functions, constrained to: standard library + dependencies you list, no network access, no "
                 "reads outside the working directory, wall-time under the limit given in the Inputs section.")
        text = _swap_lines(text, [(s[0], first if s[1] is None else s[1]) for s in _OUTLINE_SWAPS])
        text = text.replace("; the single-file constraint still applies, since an import is not a second file you author",
                            "; an import is not a second file you author")
    return text


# What the request says of the analysis when FI writes it (execution.code_skeleton, a plan in the contract form): the model writes
# only the functions of derive.py that FI cannot work out from its own record. The names are those of core/frame_table.py.
_ANALYSIS_BODIES_RULES = """\
**The analysis is FI's.** FI writes experiment.py and runs it once, after all the trials. It reads its own record of the trials,
works out each setting's count, mean, 95% interval and failed trials, computes every quantity the plan gives a formula for,
draws the figures, and prints the results. You write only the functions of derive.py that the output format lists. A function
runs on the table of FI's record and can change nothing in it:
- `derive_<id>(table)` works out the number `<id>` of the plan. It returns one number for the study, or a mapping of setting
  keys to numbers (a key is one of `table.keys()`), or `table.missing("why")` where the number does not exist (never a stand-in
  such as 0, -1 or a bound). It never returns a text, a list or True/False.
- `reference_<id>(setting)` returns the reference value of `<id>` at one setting (`{axis: value}`), or `None` where there is none;
  FI calls it once for each setting.
- `figure_<id>(fig, table)` draws what the plan says the figure shows on `fig.axes` (one axes for each panel of the plan), from
  `table`. FI makes the figure, saves it and closes it: never call `savefig`, `show`, `close`, `figure` or `subplots`.
  When `fig.x` is a result id (not a setting), draw one point for each trial of a setting: `table.values(fig.x, key)` against
  `table.values(<shown id>, key)`, paired by `table.trials(...)` (a trial with no finite value in one is left out of both).
  When panels show different settings, say which on each: `fig.panel(0, m=200)` (the panel's place in `fig.axes`, then the setting
  and its value). FI writes that panel's title from it (`m = 200`); do not type a title for such a panel.
- The table gives, for a result `id` at a setting `key`: `table.mean(id, key)`, `table.interval(id, key)`, `table.n(id, key)`,
  `table.values(id, key)` (one number for each trial that ran), `table.failed(key)`, and, for a result the plan gives a true
  value and an interval, `table.coverage(id, key)` (1.0 for each trial whose interval holds the true value, else 0.0; the
  interval's own ends are not results to read; give a factor compared inside each run as `table.coverage(id, key, factor="value")`)
  and `table.bias(id, key)` (the mean of its values minus the true value); `table.known()` lists the ids that have a true value. FI prints
  the coverage and the bias of every such result itself, one for each parameter, as `<id>_coverage` and `<id>_bias`. When a plan
  quantity is the coverage or the bias of ONE named result, work it out from that result: `table.coverage` gives 1.0 and 0.0 values, so
  take their mean for the one number a setting needs. When it spans several results, a function returns one number per setting and cannot
  say which result it is: do not pick one or pool them, leave the function as FI wrote it (`NotYet`), and FI's per-result
  values are the answer. Statistics across the trials of a setting are the table's too:
  `table.paired(a, b, key)` (the values of two results that the same trial gave, paired by the record's trial numbers, never by
  position), `table.correlation(a, b, key)` (their Pearson correlation), `table.sd(id, key)` and `table.quantile(id, key, q)`; a
  correlation, spread or quantile of the trials is one of these, never `NotYet`; and `table.keys()`, `table.setting(key)`,
  `table.at(**settings)` (a LIST of the keys of the settings that hold those values: pass one of them, `table.at(**settings)[0]`,
  to a read, never the list), `table.across(axis, **fixed)` (the values of one setting with their keys),
  `table.reference(id, key)` and, for a sub-study, `table.part(name)`. Name only ids the plan declares: any other id raises an error
  that lists the ones FI holds. `table.mean` is `None` and `table.interval` is `(None, None)` at a setting where no trial gave
  the result a value (a result the run records only at some settings): test for `None` before any arithmetic, `float()` or
  drawing, and skip that setting (or return `table.missing("why")`).
- Take no number from your own head: a value a function returns comes from `table`, from the plan's thresholds or from a formula
  written out in the function. A function reads no file, prints nothing and keeps nothing between calls.
- If you cannot work a function out, leave it as FI wrote it (it raises `NotYet`): FI then says that number is not computed.

"""

#: The figure bullet of :data:`_ANALYSIS_BODIES_RULES` for a figure function given FI's drawing calls (a frame of version 5): the
#: rule is the one the repair gives (agents/figure_primitives.md), not a second copy of it.
_FIGURE_BULLET_CALLS = (
    "- `figure_<id>(fig, table)` draws what the plan says the figure shows (one panel for each panel of the plan), from `table`.\n"
    "  When `fig.x` is a result id (not a setting), draw one point for each trial of a setting: `table.values(fig.x, key)` against\n"
    "  `table.values(<shown id>, key)`, paired by `table.trials(...)` (a trial with no finite value in one is left out of both).\n"
    + "\n".join(("  " + line).rstrip() for line in _skeleton._FIGURE_RULE.split("\n")) + "\n")


def analysis_bodies_rules(calls: bool = False) -> str:
    """:data:`_ANALYSIS_BODIES_RULES`; with ``calls`` its figure bullet is the rule of FI's drawing calls."""
    if not calls:
        return _ANALYSIS_BODIES_RULES
    a = _ANALYSIS_BODIES_RULES.index("- `figure_<id>(fig, table)` draws")
    b = _ANALYSIS_BODIES_RULES.index("- The table gives, for a result `id`")
    return _ANALYSIS_BODIES_RULES[:a] + _FIGURE_BULLET_CALLS + _ANALYSIS_BODIES_RULES[b:]
_NO_ANALYSIS_BODIES = """\
**The analysis is FI's.** FI writes experiment.py and runs it once, after all the trials: it reads its own record of the trials,
works out every number and figure the plan declares and prints the results. There is nothing of it for you to write.

"""
_IN_EACH_RUN_BODIES = (
    "FI keeps each value's own results: `table.levels(id, factor, key)` gives the number of the value of the compared setting "
    "`factor`. " + _skeleton.DERIVE_LEVEL_RULE + " A number the run gives once is one number.")


class RunChecksMixin:
    """Methods of :class:`core.engine.Engine`; see the module docstring."""

    # ---- a reported result is computed --------------------------------------------------------------------------------

    def _typed_result_exempt(self, state: Any) -> set[str]:
        """Names the plan fixes (its fixed settings, the grid's settings, what each run compares, the thresholds): echoed back,
        not results."""
        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        design = state.get("design") if isinstance(state.get("design"), dict) else {}
        return typed_result_exempt(protocol, design)

    def _typed_results_in_simulation(self, state: Any) -> tuple[Path, list[_typed.TypedResult]]:
        path = self.quest_root / "code" / _split_run.SIMULATE_NAME  # type: ignore[attr-defined]
        if not path.is_file():
            return path, []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return path, []
        return path, _typed.typed_results(text, path.name, exempt=self._typed_result_exempt(state))

    def _typed_counter(self, text: str) -> int:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        record = _read_json(self.fi_dir / _COUNTER)  # type: ignore[attr-defined]
        return int(record.get("used") or 0) if record.get("sha") == sha else 0

    def _save_typed_counter(self, text: str, used: int) -> None:
        try:
            self.fi_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
            (self.fi_dir / _COUNTER).write_text(  # type: ignore[attr-defined]
                json.dumps({"sha": hashlib.sha256(text.encode("utf-8")).hexdigest(), "used": used}), encoding="utf-8")
        except OSError:
            pass

    async def _results_from_computation(self, state: Any) -> bool:
        """Send a simulation that returns numbers typed into its code back to compute them, then leave out what is still
        typed in. ``True`` when the simulation was rewritten. The left-out quantities are this pass's alone: the record of
        an earlier pass goes first."""
        record_path = self.quest_root / "needs" / TYPED_RECORD  # type: ignore[attr-defined]
        simulate = self.quest_root / "code" / _split_run.SIMULATE_NAME  # type: ignore[attr-defined]
        try:
            sha = hashlib.sha256(simulate.read_bytes()).hexdigest() if simulate.is_file() else ""
        except OSError:
            sha = ""
        # Asked again for every later text of the script (a gate's repair, a repair of the run, a new round), never for
        # one that was read already: its answer (the left-out quantities and their record) stands.
        if sha and getattr(self, "_typed_checked_sha", None) == sha:
            return False
        record_path.unlink(missing_ok=True)
        self._typed_result_keys: set[str] = set()
        path, found = self._typed_results_in_simulation(state)
        if not found:
            self._typed_checked_sha = sha
            return False
        rewrote = False
        written: str | None = None  # the text a try of this pass wrote and no run has used yet
        text = path.read_text(encoding="utf-8")
        used = self._typed_counter(text)
        while found and used < TYPED_RESULT_REPAIRS:
            used += 1
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] %s returns numbers typed into the code (%s); asking for them to be computed (%d of %d)",
                path.name, "; ".join(f.says() for f in found[:4]), used, TYPED_RESULT_REPAIRS)
            earlier = path.read_text(encoding="utf-8") if written is not None else ""
            wrote = await self._repair_typed_results(state, path, found)
            rewrote = wrote or rewrote
            text = path.read_text(encoding="utf-8")
            if written is not None and wrote and earlier and earlier != text:
                # the version the first try wrote was replaced by this one before any run of it
                await self._keep_overwritten_try(path, used - 1, earlier, "typed-results repair replaced before it ran: "
                                                 "a later try wrote over it")
            written = text if wrote else written
            self._save_typed_counter(text, used)
            path, found = self._typed_results_in_simulation(state)
        self._typed_checked_sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
        if not found:
            return rewrote
        keys = sorted({f.key for f in found})
        self._typed_result_keys = set(keys)
        note = _typed.plain_note(keys, found)
        try:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(_typed.record(keys, found), indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not write needs/%s (%r)", TYPED_RECORD, exc)  # type: ignore[attr-defined]
        self._log.warning(  # type: ignore[attr-defined]
            "[execute] %s The paper does not use %s and says so in its limitations.", note,
            "it" if len(keys) == 1 else "them")
        return rewrote

    async def _keep_overwritten_try(self, path: Path, number: int, text: str, message: str) -> None:
        """Keep a version written into ``code/`` and overwritten before any run, on a side reference of the folder's git
        history. Best effort: never changes what the check does."""
        try:
            from . import code_project as _project

            code_dir = self.quest_root / "code"  # type: ignore[attr-defined]
            rel = path.relative_to(code_dir).as_posix()
            await asyncio.to_thread(_project.keep_refused, self.quest_root, f"typed-{path.stem}-{number}",  # type: ignore[attr-defined]
                                    {rel: text}, message, log=self._log)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 -- the record is a convenience; the check goes on whatever happens to it
            pass

    async def _repair_typed_results(self, state: Any, path: Path, found: list[_typed.TypedResult]) -> bool:
        """ONE repair of the simulation: kept only if it parses, keeps what FI reads from it, and has fewer typed-in
        results than before."""
        import ast

        from . import engine as _e
        from .static_checks import reply_script

        code = path.read_text(encoding="utf-8")
        prompt = self._prompts["execute_reflect"].substitute(  # type: ignore[attr-defined]
            topic_block=self._topic_block(), own_failure_block="",
            previous_code=code, returncode="(not run yet)", stdout_tail=_typed.directive(found), stderr_tail="",
            duration_s="0.00", figures_count="0", result_json_present="no (not run yet)",
            reflect_history_block=_e._format_reflect_history([]),
            design_block=json.dumps(state.get("design") or {}, indent=2), clarify_block=_e._format_clarify(state),
        )
        try:
            text = await self._chat(prompt, node="execute_reflect")  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 -- a repair is best-effort
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] the call to compute the typed-in results failed (%r); keeping %s as written", exc, path.name)
            return False
        new_code, parsed = reply_script(self, text, code, path.name)
        try:
            ast.parse(new_code)
            usable = bool(new_code.strip()) and all(mark in new_code for mark in _typed.KEEP_MARKS if mark in code)
        except (SyntaxError, ValueError):
            usable = False
        if not usable:
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] the rewrite of %s is not the same simulation with its results computed; keeping it as written",
                path.name)
            return False
        left = _typed.typed_results(new_code, path.name, exempt=self._typed_result_exempt(state))
        if len(left) >= len(found):
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] the rewritten %s still returns %d number(s) typed into the code; keeping it as written",
                path.name, len(left))
            return False
        package_was = self._package_snapshot()  # type: ignore[attr-defined]
        path.write_text(new_code, encoding="utf-8")
        if await self._simulation_kept_or_back(state, path, code, package_was, "execute"):  # type: ignore[attr-defined]
            return False
        self._log.info(  # type: ignore[attr-defined]
            "[execute] rewrote %s so its results are computed (%d typed-in left): %s", path.name, len(left),
            str(parsed.get("patch_summary") or "no summary")[:120])
        return True

    #: The most times the oracle gate and the typed-results check hand the simulation to each other in one pass.
    TYPED_GATE_ROUNDS = 3

    async def _typed_results_stable(self, state: Any, py: Any, env: Any, seed_path: Path,
                                    oracle_code: str | None) -> tuple[str | None, dict[str, Any] | None]:
        """After an oracle repair rewrote the simulation: read the text it left for typed-in results, and when that repair
        rewrote it again run the oracle gate on the result, and read what that left, until the text is one the check has
        read (each step is bounded by its own budget per text). ``(the repaired code for the state, None)``; a text still
        unread when the rounds are spent is not run: ``(code, the quest's stop)``."""
        for _ in range(self.TYPED_GATE_ROUNDS):
            if not await self._results_from_computation(state):
                return oracle_code, None  # the text is read: nothing to rewrite, or nothing it could rewrite
            again = await self._oracle_gate(state, py, env, seed_path)  # type: ignore[attr-defined]
            oracle_code = again if again is not None else seed_path.read_text(encoding="utf-8")
            self._check_equation_labels(state)  # type: ignore[attr-defined]
            self._check_code_layout(state)  # type: ignore[attr-defined]
        if self._typed_text_unread(seed_path):
            text = ("the simulation kept being rewritten by the known-answer checks' repair and by the repair of its typed-in "
                    "results, and the text FI would run was not checked for typed-in results")
            return oracle_code, self._stop_before_run(text, "not_run")
        return oracle_code, None

    def _typed_text_unread(self, seed_path: Path) -> bool:
        try:
            sha = hashlib.sha256(seed_path.read_bytes()).hexdigest() if seed_path.is_file() else ""
        except OSError:
            return True
        return bool(sha) and getattr(self, "_typed_checked_sha", None) != sha

    def _left_out_quantities(self) -> set[str]:
        """The quantities this pass's analysis and results must not use (still typed into the code after the repairs)."""
        typed = getattr(self, "_typed_result_keys", None)
        if typed is None:
            # A fresh process (a resume at a later node): the pass that left them out wrote its record, which is what stands.
            from . import result_checks as _checks

            return set(_checks.left_out_by_record(self.quest_root))  # type: ignore[attr-defined]
        return set(typed) | self._result_check_keys()  # type: ignore[attr-defined]

    def _without_typed_results(self, result_json: Any) -> Any:
        """``result_json`` without the left-out quantities."""
        keys = self._left_out_quantities()
        return _typed.strip_keys(result_json, keys) if keys and result_json else result_json

    def _lines_without_left_out(self, lines: list[str]) -> tuple[list[str], int, list[str]]:
        """``lines`` without those that name a left-out quantity: the kept lines, how many were taken out, and the names."""
        keys = sorted(self._left_out_quantities())
        patterns = [re.compile(r"(?<![A-Za-z0-9_])" + re.escape(k) + r"(?![A-Za-z0-9_])") for k in keys]
        kept = [line for line in lines if not any(p.search(line) for p in patterns)]
        return kept, len(lines) - len(kept), keys

    def _taken_out_note(self, dropped: int, keys: list[str], where: str) -> str:
        """The plain note that says how many lines of ``where`` were taken out, and why."""
        names = ", ".join(f"`{k}`" for k in keys)
        return (f"[FI NOTE] {dropped} line(s) of the run's {where} that name {names} were taken out: "
                f"{'that quantity was' if len(keys) == 1 else 'those quantities were'} left out of the results "
                "(typed into the code, or failed a check of what ran) and must not be used, derived from or renamed.")

    def _stdout_for_models(self, stdout: str) -> str:
        """What a model may read of a run's output: the output without its ``RESULT_JSON`` line (the verified results reach a
        model once, as FI's accepted projection of them, with the left-out quantities taken out) and without any line that
        names a left-out quantity, with a plain note saying what was taken out and why. A bare left-out value is replaced
        too. The raw output stays in the records (``.fi/last_run_stdout.txt``)."""
        text = stdout or ""
        lines = text.splitlines()
        body = [line for line in lines if not line.lstrip().startswith("RESULT_JSON:")]
        kept, dropped, keys = self._lines_without_left_out(body)
        notes: list[str] = []
        if len(body) != len(lines):
            notes.append("[FI NOTE] The run's RESULT_JSON line is not repeated here: the results block is FI's accepted "
                         "projection of the results, and the only place to read them.")
        if dropped:
            notes.append(self._taken_out_note(dropped, keys, "output"))
        out = "\n".join([*kept, *notes]) if notes else text
        return self._withhold_left_out_values(out)

    def _stderr_for_models(self, stderr: str) -> str:
        """What a model may read of a run's error output: every line except those that name a left-out quantity (the
        traceback and the rest stay readable), with a plain note saying how many were taken out. A bare left-out value is
        replaced too."""
        text = stderr or ""
        lines = text.splitlines()
        # FI's analysis (core/analysis_frame.py) says each step as it begins it: only the last one (what was running when it
        # stopped) is the model's to read.
        from .analysis_frame import RUN_MARK, STEP_MARK

        steps = [i for i, line in enumerate(lines) if line.startswith(STEP_MARK)]
        runs = {i for i, line in enumerate(lines) if line.startswith(RUN_MARK)}  # the run's id: FI's, never the model's
        if len(steps) > 1 or runs:
            lines = [line for i, line in enumerate(lines) if i not in set(steps[:-1]) | runs]
            text = "\n".join(lines)
        # FI's own messages in the error output (why a run was refused: "FI ran no trial successfully: ...", "FI's run gave
        # a result that is not usable: ...") name a quantity to say what to repair; they are FI's diagnosis, not a value the
        # script printed, so they stay. Only what the script itself printed is filtered (and any bare value, below).
        kept: list[str] = []
        dropped, keys = 0, []
        for line in lines:
            if line.lstrip().startswith(("FI ", "FI's ", "[FI")):
                kept.append(line)
                continue
            ok, gone, keys = self._lines_without_left_out([line])
            kept.extend(ok)
            dropped += gone
        if not dropped:
            return self._withhold_left_out_values(text)
        out = "\n".join([*kept, self._taken_out_note(dropped, keys, "error output")])
        return self._withhold_left_out_values(out)

    def _keep_raw_stdout(self, stdout: str) -> None:
        """The run's output exactly as it was, for the records (no model reads it)."""
        try:
            path = self.fi_dir / "last_run_stdout.txt"  # type: ignore[attr-defined]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((stdout or "")[-200000:], encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not keep the run's raw output (%r)", exc)  # type: ignore[attr-defined]

    def _note_typed_values(self, raw_result_json: Any) -> None:
        """The values of the quantities left out for being typed into the code, kept in this pass's record so the writer's
        text can be checked for quoting one (core/record_claims.py). Nothing is added when none was left out."""
        from . import result_checks as _checks

        keys = self._typed_result_keys if getattr(self, "_typed_result_keys", None) is not None else set()
        if not keys or not raw_result_json:
            return
        path = self.quest_root / "needs" / TYPED_RECORD  # type: ignore[attr-defined]
        record = _read_json(path)
        if not record:
            return
        record["removed_values"] = _checks._values_left_out(raw_result_json, set(keys))
        try:
            path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not write needs/%s (%r)", TYPED_RECORD, exc)  # type: ignore[attr-defined]

    # ---- a result that did not change is recorded as observed ----------------------------------------------------------

    def _record_no_difference(self, note: str | None) -> None:
        """Every setting gave the same numbers and the code does read the settings (core/flat_output.py): the observation is
        kept in ``needs/NO_DIFFERENCE_OBSERVED.json`` (this run's alone) for the analysis and the writer. A run without one
        removes an earlier run's record."""
        path = self.quest_root / "needs" / NO_DIFFERENCE_RECORD  # type: ignore[attr-defined]
        if not note:
            path.unlink(missing_ok=True)
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"note": note}, indent=1), encoding="utf-8")
        except OSError as exc:
            self._log.warning("[execute] could not write needs/%s (%r)", NO_DIFFERENCE_RECORD, exc)  # type: ignore[attr-defined]

    def _no_difference_note(self) -> str:
        """For the analysis and the writer: a study whose numbers did not change with its settings, said as an observation
        (never as proof of no effect, and never as a broken simulation). Empty when there is none."""
        record = _read_json(self.quest_root / "needs" / NO_DIFFERENCE_RECORD)  # type: ignore[attr-defined]
        note = record.get("note") if isinstance(record, dict) else None
        if not isinstance(note, str) or not note.strip():
            return ""
        return (f"Observed: {note} Report it as stated here, not as an effect the study found; do not call the "
                "simulation broken, and do not claim that the setting can have no effect in general.")

    # ---- the note for the writer ---------------------------------------------------------------------------------------

    def _withhold_left_out_values(self, text: str) -> str:
        """``text`` (a note a model reads) with any number written precisely enough to be recognised that equals a value of a
        quantity left out of the results replaced by plain words: a note that says why a quantity is out never repeats it."""
        values: list[float] = []
        for name in (TYPED_RECORD, "RESULT_CHECK.json"):
            removed = _read_json(self.quest_root / "needs" / name).get("removed_values")  # type: ignore[attr-defined]
            if isinstance(removed, dict):
                values += [float(v) for vs in removed.values() if isinstance(vs, list) for v in vs
                           if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not values:
            return text

        def swap(m: re.Match[str]) -> str:
            token = m.group(0)
            if not _carries_enough_precision(token):
                return token
            number = float(token)
            return "(a value that is not given)" if any(abs(number - v) <= 1e-9 * max(1.0, abs(v)) for v in values) else token

        return re.sub(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w])", swap, text)

    def _run_checks_note(self) -> str:
        """For the writer: quantities left out because they were typed into the code, and a run FI made smaller to fit the
        time allowed, so the limitations say so."""
        parts: list[str] = []
        record = _read_json(self.quest_root / "needs" / TYPED_RECORD)  # type: ignore[attr-defined]
        note = record.get("note")
        if isinstance(note, str) and note.strip():
            names = ", ".join(f"`{k}`" for k in record.get("removed_quantities") or [])
            parts.append(f"{self._withhold_left_out_values(note)} Do not report, describe or quote any value for {names or 'them'}: say plainly, in the "
                         "limitations, that this quantity was not computed by the simulation and is left out.")
        if checked := self._result_checks_note():  # type: ignore[attr-defined]  # core/result_checks.py
            parts.append(self._withhold_left_out_values(checked))
        if static := self._static_checks_note():  # type: ignore[attr-defined]  # core/static_checks.py
            parts.append(static)
        from . import code_review as _code_review

        now = _code_review.version(self.quest_root / "code")  # type: ignore[attr-defined]
        for p in self._cannot_repair_problems():  # type: ignore[attr-defined]  # core/engine.py
            if _stop_power.may_stop(p.get("source")):
                continue
            if p.get("code_version") and p["code_version"] != now:
                # The code was repaired again after this was found, and no check looked at it again.
                parts.append(f"FI's checks of an earlier version of the code found this: {p['text']}. The code was "
                             "repaired again after that and this was not checked again. Say it in the limitations as "
                             "not known to be fixed, not as a fact of the results.")
                continue
            if p.get("check") == "figure_fi_fault":
                # a figure FI itself could not draw: the results stand; say it plainly and as FI's fault, not as a finding about the code
                parts.append(f"{p['text']} Say this plainly in the limitations, in these words or very close to them: the figure is "
                             "missing because of a fault of FI's own, not because of the study, and the numbers it would have "
                             "shown are in the results table. Do not describe, cite or draw conclusions from that figure.")
                continue
            if p.get("check") == "figure_pass_fault":
                parts.append(f"{p['text']}. Say this once, plainly, in the limitations: the figures' sentences were not checked by FI.")
                continue
            parts.append(f"FI's checks of the code before the run found this, and the repairs did not remove it: {p['text']}. "
                         f"It is {_stop_power.source_words(p.get('source'))}, so the run went on. Say it in the "
                         "limitations as " + ("what FI saw on a small copy of the results" if p.get("source") == _stop_power.COPY
                                              else "FI's reading of the code") + ", not as a fact of the results.")
        if observed := self._no_difference_note():
            parts.append(observed)
        sized = _read_json(self.quest_root / _estimate.RECORD)  # type: ignore[attr-defined]
        reduced = sized.get("reduced")
        if isinstance(reduced, dict) and reduced.get("note"):
            parts.append(f"{reduced['note']} Say this in the paper with the numbers (it is among FI's facts about this run, "
                         "given below), and write no result for what was left out.")
        over = sized.get("over_topic_limit")
        if isinstance(over, dict) and over.get("note"):
            parts.append(f"{over['note']} Say this in the paper (it is among FI's facts about this run, given below).")
        if scope := self._scope_gap_note():
            parts.append(scope)
        return "\n\n".join(parts)

    def _scope_gap_note(self) -> str:
        """For the writer: each result or setting of the study the person chose that the frozen plan has no name for
        (core/scope_cards.py ``gaps``, a reading by names). It is told in the limitations and never stops the quest."""
        from . import scope_cards as _sc

        record = _sc.read_record(self.quest_root)  # type: ignore[attr-defined]
        card = record.get("chosen") if isinstance(record, dict) else None
        protocol = _frozen.protocol_of(self.quest_root)  # type: ignore[attr-defined]
        if not isinstance(card, dict) or not card or protocol is None:
            return ""
        design = {"protocol": protocol}
        names = [r for r in _sc._strings(card.get("main_results")) if _sc.gaps({"main_results": [r]}, design)]
        names += [s for s in _sc._strings(card.get("compared_settings"))
                  if _sc.gaps({"compared_settings": [s]}, design)]
        return "\n".join(f"The person chose a study that names {x}; the plan does not cover it and gives no reason. "
                         "Say so in the limitations." for x in names)

    # ---- the frame of simulate.py (core/skeleton.py) ----------------------------------------------------------------------

    def _skeleton_frame(self, state: Any) -> str:
        """FI's frame of ``simulate.py`` for this plan, or ``""`` when the frame is off (``execution.code_skeleton``) or does not
        fit the plan (the reason is said in run.log): the model then writes the whole script as before."""
        from . import contract as _contract
        from . import skeleton as _skeleton

        if not self._frame_on():
            return ""
        design = state.get("design")
        why = _skeleton.applies(design)
        if why:
            if getattr(self, "_frame_why_logged", None) != why:  # one line, not one for every request that asks
                self._frame_why_logged = why
                self._log.info("[implement] FI writes no frame of simulate.py: %s", why)  # type: ignore[attr-defined]
            return ""
        # A quest keeps the frame it was written with (a resume, an update of FI): the frame on disk is the one used while the
        # plan has not changed; a plan that changed gets a new one.
        try:
            existing = (self.quest_root / "code" / _split_run.SIMULATE_NAME).read_text(encoding="utf-8")  # type: ignore[attr-defined]
        except (OSError, UnicodeDecodeError):
            existing = ""
        return _skeleton.frame_for(design, stochastic=_split_run.design_is_stochastic(design),
                                   strict=_contract.from_design(design).form == "contract", existing=existing)

    def _strict_frame(self, state: Any) -> bool:
        """Whether the frame refuses a name outside its list (a plan in the contract form)."""
        from . import contract as _contract

        return _contract.from_design(state.get("design")).form == "contract"

    def _frame_on(self) -> bool:
        """Whether FI writes the frame of simulate.py, and so the analysis of a plan in the contract form: the settings
        (``execution.code_skeleton`` with ``split_analysis``, and not the agent coder), read before any code exists."""
        ex = self.config.execution  # type: ignore[attr-defined]
        return bool(ex.code_skeleton and ex.split_analysis and not self._agent_mode())  # type: ignore[attr-defined]

    def _fis_analysis_planned(self, design: Any) -> bool:
        """Whether FI will work out this plan's analysis itself (core/analysis_frame.py) once the plan is in the contract
        form: its formulas are then held, when the plan is read, to what FI can work out. Read at the plan step, before
        any code exists; with the frame off the model's own analysis computes them, and nothing of this is asked."""
        from . import skeleton as _skeleton

        return self._frame_on() and not _skeleton.applies(design)

    def _frame_variant(self, state: Any) -> str:
        """``"prompt"`` when FI writes the frame of simulate.py (the code request shows the frame first and asks for ``trial``
        beside experiment.py), else ``""``."""
        return "prompt" if self._skeleton_frame(state) else ""

    def _analysis_bodies(self, state: Any) -> list[str] | None:
        """The functions of derive.py the model is asked for when FI writes the analysis too (core/analysis_frame.py), possibly
        none; ``None`` when it does not (no frame of simulate.py, or a plan that is not in the contract form): the model then
        writes experiment.py as before. The requests below take this as their ``bodies``."""
        if not self._skeleton_frame(state):
            return None
        try:
            _analysis = importlib.import_module(f"{__package__}.analysis_frame")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__package__}.analysis_frame":
                raise  # an import that fails inside the module must show in the run's log, not give the old texts
            return None
        design = state.get("design")
        if _analysis.applies(design):
            return None
        # A quest whose frame an older FI wrote keeps the analysis the model writes (the frame is not rewritten under a run).
        # Read from the frame FI writes now, not the file on disk: a changed plan gets a new frame, of this version, before
        # the reply is put into it, so the request and what is written of the reply agree.
        if _analysis.frame_problem(self._skeleton_frame(state)):
            return None
        return _analysis.body_names(_analysis.plan_of(design))

    def _bodies_kept(self, state: Any, bodies: list[str] | None, outline: dict[str, Any]) -> Any:
        """The functions of the study that already ran and passed, kept when the code step is entered again
        (core/kept_bodies.py); ``None`` when FI does not write the analysis here, or nothing ran. They are put back as they ran,
        and the model is asked only for the rest. A function stays unless the reason for the pass names it: a function the last
        run failed in, a quantity or figure a measured finding names (a number no function of derive.py makes is the
        simulation's), or a part of the plan the function reads that the design changed."""
        from . import analysis_frame as _analysis
        from . import kept_bodies as _kept

        if bodies is None:
            return None
        design = state.get("design")
        try:
            trial_reads = _kept._sha(self._scripts_built_from(state, outline)) if outline else ""  # type: ignore[attr-defined]
            _kept.snapshot(self.quest_root, design)  # type: ignore[attr-defined]
            named = {str((state.get("exec_result") or {}).get("failed_body") or "")} - {""}
            from . import engine as _e

            sends = getattr(_e, "_review_sends_the_experiment_back", None)
            said = sends(state.get("review") or {}, state) if sends else ""
            if said:
                record = _analysis.read_record(self.quest_root)  # type: ignore[attr-defined]
                named.add(_analysis.made_by(record, quantity=said) or _analysis.made_by(record, figure=said) or _kept.TRIAL)
            kept = _kept.decide(self.quest_root, design, list(bodies), trial_reads, named)  # type: ignore[attr-defined]
            written = _kept.restore(self.quest_root, kept)  # type: ignore[attr-defined]
            _kept.mark_written(self.quest_root, design, trial_reads, [b for b in bodies if b not in kept.names],  # type: ignore[attr-defined]
                               not kept.trial)
        except Exception as exc:  # noqa: BLE001 -- a record FI cannot read keeps nothing: every function is asked for as before
            self._log.warning("[implement] could not tell which functions already ran and passed (%r); all are written again", exc)  # type: ignore[attr-defined]
            return None
        if kept.names or kept.trial or kept.again:
            self._log.info("[implement] %s", _kept.say(kept))  # type: ignore[attr-defined]
        if written:
            self._log.info("[implement] put back as they ran: %s", ", ".join(written))  # type: ignore[attr-defined]
        return kept if (kept.names or kept.trial) else None

    def _keep_note(self, kept: Any) -> str:
        """The start of the code request when functions are kept: what FI keeps, and what is still asked for."""
        names = [*(["the simulation (`trial`)"] if kept.trial else []), *(f"`{n}`" for n in kept.names)]
        return ("FI keeps " + ", ".join(names) + (" exactly as it ran and passed" if len(names) == 1 else " exactly as they ran and passed")
                + ": do not write " + ("it again" if len(names) == 1 else "them again") + ", and do not change anything they use."
                + (" Leave the `# file: simulate.py` block out." if kept.trial else "")
                + " Give only the functions this request asks for.\n\n")

    def _fis_analysis_packages(self, state: Any) -> list[str]:
        """The packages FI's own analysis needs whatever the reply lists: matplotlib, when the plan declares a figure or a
        search (FI draws those itself, in fi_figures.py; a clean venv or a container has no matplotlib of its own)."""
        from . import analysis_frame as _analysis

        design = state.get("design")
        if _analysis.applies(design):
            return []
        plan = _analysis.plan_of(design)
        return ["matplotlib"] if plan["figures"] or plan["search"] else []

    def _analysis_written(self, state: Any, derive_reply: str, simulate_text: str, keep: Any = ()) -> str:
        """Write the analysis FI writes for this plan into ``code/`` (core/analysis_frame.py): ``derive.py`` with the study's
        functions from ``derive_reply`` put into FI's frame, and ``experiment.py``, ``fi_table.py``, ``fi_figures.py`` and
        ``fi_core/`` beside it. Returns the text of ``experiment.py`` (FI's)."""
        from . import analysis_frame as _analysis

        code_dir = self.quest_root / "code"  # type: ignore[attr-defined]
        code_dir.mkdir(parents=True, exist_ok=True)
        try:
            existing = (code_dir / _analysis.DERIVE_FILE).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            existing = ""
        design = state.get("design")
        text, note = _analysis.compose(design, derive_reply, existing, keep, calls=self._figure_calls_on())
        if note:
            self._log.warning("[implement] derive.py: %s", note)  # type: ignore[attr-defined]
        if text != existing:
            (code_dir / _analysis.DERIVE_FILE).write_text(text, encoding="utf-8")
        _analysis.ensure_files(code_dir, design, simulate_text)
        names = ", ".join(f"`{b}`" for b in _analysis.body_names(_analysis.plan_of(design)))
        self._log.info("[implement] FI wrote the analysis (experiment.py, the frame of derive.py and the files they use); "  # type: ignore[attr-defined]
                       "the model's part of it is %s", names or "nothing (the plan needs no function of derive.py)")
        return _analysis.EXPERIMENT

    def _keep_fis_analysis(self, state: Any) -> None:
        """Before a run, when FI writes the analysis of this quest (core/analysis_frame.py) and derive.py has FI's frame: FI's
        own files are as FI wrote them. A step that wrote other text over experiment.py (or a carried file) is undone here,
        whatever path it came by, and the run says so; the study's functions in derive.py are kept."""
        from . import analysis_frame as _analysis

        code_dir = self.quest_root / "code"  # type: ignore[attr-defined]
        if not _analysis.analysis_file(code_dir, _analysis.DERIVE_FILE) or self._analysis_bodies(state) is None:
            return
        try:
            simulate = (code_dir / _split_run.SIMULATE_NAME).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            simulate = ""
        try:
            written = _analysis.ensure_files(code_dir, state.get("design"), simulate)
        except Exception as exc:  # noqa: BLE001 -- the run then shows what is wrong with the files
            self._log.warning("[execute] FI's analysis files could not be checked: %r", exc)  # type: ignore[attr-defined]
            return
        if written:
            self._log.warning("[execute] FI wrote its own analysis files again before the run (%s): they are FI's, and "  # type: ignore[attr-defined]
                              "they differed from what FI writes for this plan", ", ".join(written))

    def _fis_stub_sendback(self, state: Any) -> dict[str, Any] | None:
        """Before anything runs, when FI writes the analysis (core/analysis_frame.py): a function of derive.py that is still
        FI's stub, for a declared quantity of a counting role (a result, a reference, a physical bound), goes to the repair
        of that function while a repair is left. A stub gives no value on any number of trials, so the long run would only
        say so; read from the file FI wrote, not guessed. ``None`` to go on (the run then lists what it did not work out)."""
        from . import analysis_frame as _analysis

        code_dir = self.quest_root / "code"  # type: ignore[attr-defined]
        if not self._analysis_on() or self._analysis_bodies(state) is None:
            return None
        if int(state.get("exec_reflect_iter") or 0) >= self.config.engine.exec_reflect_max_iterations:  # type: ignore[attr-defined]
            return None
        try:
            left = set(_analysis.stubs((code_dir / _analysis.DERIVE_FILE).read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            return None
        plan = _analysis.plan_of(state.get("design"))
        from .numeric_warnings import COUNTING_ROLES

        items = {d["id"]: d for d in plan.get("derived") or []}
        # A quantity of a counting role worked out by a formula of the plan from a stub's quantity needs the stub as well.
        counting = {i for i, d in items.items() if (d.get("role") or "result") in COUNTING_ROLES}
        needed, todo = set(counting), list(counting)
        from . import oracle_forms as _forms

        while todo:
            d = items.get(todo.pop()) or {}
            uses = [*(d.get("inputs") or []), *_forms.names(d.get("formula") or "")] if d.get("by") == "formula" else []
            for n in uses:
                if n in items and n not in needed:
                    needed.add(n)
                    todo.append(n)
        wanted = [(i, d["by"]) for i, d in items.items() if d.get("by") in left and i in needed]
        if not wanted:
            return None
        said = "; ".join(f"`{i}` is computed by no trial and no function of the study's yet (`{b}` is still FI's stub)"
                         for i, b in wanted[:4])
        self._log.warning("[execute] %s: sending derive.py back for it before the run", said)  # type: ignore[attr-defined]
        failed = self._code_review_failed_run(  # type: ignore[attr-defined]
            f"FI read derive.py before the run: {said}. Write `{wanted[0][1]}`, so the run can give the value the plan "
            "asks for.", script=_analysis.DERIVE_FILE)
        failed["exec_result"]["failed_body"] = wanted[0][1]
        return failed

    def _analysis_on(self) -> bool:
        """Whether this quest's analysis is the one FI writes: ``code/experiment.py`` holds FI's text."""
        from . import analysis_frame as _analysis

        return _analysis.is_fis_file(self.quest_root / "code" / _analysis.EXPERIMENT_FILE)  # type: ignore[attr-defined]

    def _frame_rules(self, state: Any) -> str:
        """What the code request says about the function ``trial`` when FI writes the frame, else ``""``."""
        if not self._skeleton_frame(state):
            return ""
        protocol = (state.get("design") or {}).get("protocol")
        names = ", ".join(f"`{i}`" for i in _skeleton.declared_ids(protocol))
        more = ""
        if isinstance(protocol, dict) and isinstance(protocol.get("optimisation"), dict):
            more += ("- This study searches for the best design: `cell` is ONE design (every setting by name), and `ctx.kind` says "
                     "what FI asks for (`\"search\"`: a design the search tries; `\"scan\"`: a row of the coarse scan; "
                     "`\"check\"`: a case of a check). The objective and every constrained quantity are results: return them by "
                     "name. `trial` never searches, optimises or loops over designs: one call computes one design.\n")
        inside_map = _in_run.of_study(protocol) or {}
        inside = list(inside_map)
        if inside:
            more += (f"- `cell[\"{inside[0]}\"]`" + (" and the others of " + ", ".join(f"`{n}`" for n in inside[1:]) if inside[1:] else "")
                     + " compared inside each run are the LIST of their values, not one value: use every value of it. Give a "
                     "result per value with `out.each(\"name\", {value: number})` (or a list in the order of the values when one "
                     "thing is compared), a number for EVERY value (NaN or None where the result does not exist for one). A "
                     "number the run gives once (fitted across all the values) is returned as a number. "
                     + "Key a mapping " + _in_run.key_forms(_in_run.level_cells(inside_map)) + ". "
                     + _in_run.one_number_rule(protocol) + "\n")
        if getattr(_optimise_search.study_grid(protocol), "sub_studies", None):
            more += ("- The study has sub-studies: a `cell` holds every setting; those the sub-study does not vary are at the value "
                     "it holds.\n")
        silent = _skeleton.case_less_checks(protocol)
        if silent:
            more += ("- The plan's check(s) " + ", ".join(f"`{n}`" for n in silent) + " name no case FI can run: also give the "
                     "function `oracle()` (no arguments) that returns the numbers those checks measure, by name, computed with the "
                     "same simulation code.\n")
        more += ("- Only when one call can compute every trial of a setting at once (vectorised code): give `trial_batch(cell, ctx)` "
                 "instead of `trial`. It returns a list with one result for each seed in `ctx.seeds`, in that order "
                 "(`ctx.results()` gives them ready, `ctx.rngs()` one generator per seed); FI checks the length and the order.\n")
        lead = ("FI writes the frame of simulate.py (the entry point, the seed, the thresholds). In the `# file: simulate.py` block "
                "give ONLY the function `trial(cell, ctx)` and the imports and helper functions it needs.\n")
        return (
            lead +
            "- The frame already imports `TrialResult` and `Ctx`: add no import of `fi_frame`.\n"
            "- `cell[\"name\"]`: the settings, read by name.\n"
            "- `ctx.rng`: the random generator FI seeded for this trial. Draw every random number from it. `ctx.seed` is the "
            "same seed as an integer for a library that wants one; `ctx.trial` is the trial's number; `ctx.thresholds` holds the "
            "plan's thresholds.\n"
            f"- Return the results the plan declares, by name ({names}; a proportion may be returned as `<name>_count` and "
            "`<name>_total`): `out = TrialResult()`, `out[\"name\"] = value`, `return out`. The frame lists in RESULTS every name "
            "`trial` may return (the plan's results, the names of a `known_value`, the names a check's `measure` uses, "
            "`why_missing`); " + ("a name outside that list fails EVERY trial with `UnknownResult`, so keep any other number in "
                                  "a local variable" if self._strict_frame(state) else "any other name is kept as a diagnostic, "
                                  "not as a result of the plan") + ".\n"
            "- A quantity the method cannot give at this setting (it diverged, there is no crossing in the range) is `None`, with "
            "the reason in `out[\"why_missing\"] = {\"name\": \"reason\"}`, never a stand-in number (0, -1, a bound). A trial whose "
            "code fails raises an exception with the reason instead of returning a made-up value. `trial` writes no files, "
            "prints no results and keeps no state between calls.\n"
            "- A cut-off that decides something (what counts as an outbreak, a success) is read from `ctx.thresholds`, never "
            "written into the code as a number of its own.\n"
            "- When a protocol metric is a mean over the trials a proportion counts (`\"given\": \"p_outbreak\"`), also set, under "
            "that proportion's own id, 1 (this trial is in the subset) or 0 for EVERY trial, and the value being averaged under "
            "the mean metric's own id: FI picks the subset trial by trial from its own record.\n"
            "- When a protocol metric has a `known_value`, set, under EXACTLY the names it gives (`estimate`, `standard_error`, "
            "the two ends of `interval`), the trial's own estimate and its own uncertainty, computed from that trial's data, on "
            "the scale of the true value; never whether the interval covered the truth (FI counts that itself). Where a trial has "
            "no estimate it raises with the reason, unless the metric's `undefined_when` names that setting.\n"
            "- A check of the plan that names a `case` and a `measure` is run by FI: it calls `trial` on that case and computes "
            "the `measure` (one returned name, or a formula of them) from the result, so `trial` must return every name a "
            "`measure` uses, computed by the real simulation (never a check's number already worked out in another form).\n"
            "- Each equation of the design's `model` whose `role` is `generates` carries its id in a comment (`# E1`) on or above "
            "the code that computes it (in simulate.py, or in the model's package when the code has one).\n" + more + "\n")

    def _frame_note(self, state: Any) -> str:
        """The start of the code request when FI writes the frame, else ``""``: what to give for simulate.py, and the frame FI
        writes around it."""
        if not self._frame_variant(state):
            return ""
        return (self._frame_rules(state)
                + "FI's frame of simulate.py, as FI writes it around your function (you write none of it and change none of it):\n\n"
                + "```python\n" + self._skeleton_frame(state).rstrip("\n") + "\n```\n\n")

    def _frame_protocols(self, state: Any, search: bool, inside: dict[str, Any], bodies: Sequence[str] | None = None) -> str:
        """The two-script contract of the code request when FI writes the frame, else ``""``: the experiment.py parts of the
        protocol texts, with nothing that asks simulate.py for an entry point, a seed or a loop (those are the frame's).
        ``bodies`` (see :meth:`_analysis_bodies`): FI writes experiment.py too; the request then has the rules of the
        functions of derive.py in place of the rules of the analysis."""
        if not self._frame_variant(state):
            return ""
        from . import engine as _e

        if bodies is not None:
            return self._frame_protocols_bodies(state, search, inside, bodies)
        split = _e._SPLIT_PROTOCOL
        analysis = split[split.index("**experiment.py is the analysis.**"):split.index("If an outline is given")]
        head = ("## The experiment is two scripts, and FI runs the trials (execution.split_analysis is on)\n\n"
                "FI itself runs every trial and keeps their record, so the record of what ran is never the simulation's own "
                "statement. Reply with the fenced Python blocks the output format names: `# file: simulate.py` (the function "
                "`trial`, as the start of this request says), then `# file: experiment.py`, then the `DEPS:` line.\n\n")
        outline = ("If an outline is given, it describes the whole experiment as one program: put the code of one trial in "
                   "`trial` and the statistics, figure and `RESULT_JSON` code in experiment.py, keeping every name and "
                   "signature of the outline's functions that you keep. simulate.py holds no loop over settings or trials and no "
                   "main: the frame has them, so leave out whatever of the outline does that.\n")
        block = head + analysis + outline
        if self.config.execution.background_jobs:  # type: ignore[attr-defined]
            block += _e._CLUSTER_PROTOCOL
        if search:
            block += ("\n## This study is a search for the best design, and FI runs the search (the plan's `optimisation` block)\n\n"
                      "FI, not the scripts, searches for the design that makes the objective as low or as high as possible: it "
                      "decides which design to try next, runs one trial of the simulation for each, counts every call against the "
                      "plan's evaluation budget and keeps the record. The rules above still hold, with these differences:\n"
                      + _e._SEARCH_PROTOCOL[_e._SEARCH_PROTOCOL.index("- experiment.py does not read `FI_TRIALS` for the search."):])
        if inside:
            block += ("\n**What each run compares inside it** (`protocol.in_each_run`: "
                      + ", ".join(f"`{k}`" for k in inside) + "). These are not settings of their own: every run goes through ALL "
                      "their values together, on the same random numbers.\n"
                      + _e._IN_EACH_RUN_PROTOCOL[_e._IN_EACH_RUN_PROTOCOL.index("In FI_TRIALS each setting is followed"):])
            rule = _in_run.one_number_rule((state.get("design") or {}).get("protocol"))
            block += f"\n{rule}\n" if rule else ""
        return block

    def _figure_calls_on(self) -> bool:
        """Whether the quest's figure functions are given FI's drawing calls (a frame of version 5), as :func:`core.skeleton.
        reflect_derive` decides it for a repair: a quest whose runtime or frame an older FI wrote keeps drawing on ``fig.axes``;
        one with no runtime yet is written with this version's."""
        code_dir = self.quest_root / "code"  # type: ignore[attr-defined]
        have = _skeleton.quest_frame_version(code_dir)
        if have:
            return have >= _skeleton.VERSION
        try:
            framed = _skeleton.frame_version((code_dir / _split_run.SIMULATE_NAME).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            framed = None
        return framed is None or framed >= _skeleton.VERSION

    def _frame_protocols_bodies(self, state: Any, search: bool, inside: dict[str, Any], bodies: Sequence[str]) -> str:
        """:meth:`_frame_protocols` when FI writes the analysis: the functions of derive.py are the model's only other task."""
        from . import engine as _e

        names = ", ".join(f"`{b}`" for b in bodies)
        asked = (f"then `# file: derive.py` (only {names}), " if bodies else "")
        head = ("## The experiment is a simulation and an analysis, and FI writes both around your code (execution.split_analysis is on)\n\n"
                "FI itself runs every trial and keeps their record, so the record of what ran is never the simulation's own "
                "statement. Reply with the fenced Python blocks the output format names: `# file: simulate.py` (the function "
                f"`trial`, as the start of this request says), {asked}then the `DEPS:` line.\n\n")
        block = head + (analysis_bodies_rules(self._figure_calls_on()) if bodies else _NO_ANALYSIS_BODIES)
        block += ("If an outline is given, it describes the experiment as one program: put the code of one trial in `trial`"
                  + (" and the code of the quantities and figures in the functions of derive.py" if bodies else "")
                  + ", keeping every name and signature of the outline's functions that you keep. simulate.py holds no loop over "
                  "settings or trials and no main: the frame has them, so leave out whatever of the outline does that. FI "
                  "writes experiment.py, so leave out the statistics, the figures' files and the printed results too.\n")
        if self.config.execution.background_jobs:  # type: ignore[attr-defined]
            block += _cluster_bodies()
        if search:
            block += ("\n## This study is a search for the best design, and FI runs the search (the plan's `optimisation` block)\n\n"
                      "FI, not the scripts, searches for the design that makes the objective as low or as high as possible: it "
                      "decides which design to try next, runs one trial of the simulation for each, counts every call against the "
                      "plan's evaluation budget and keeps the record. FI also draws the search's own figures (the best objective "
                      "found so far against the evaluation number, the baseline against the best design, the coarse scan) and "
                      "reports the best design, the baseline, the improvement and FI's check of the best design at finer "
                      "settings from its own record, so you write none of that.\n")
        if inside:
            block += ("\n**What each run compares inside it** (`protocol.in_each_run`: "
                      + ", ".join(f"`{k}`" for k in inside) + "). These are not settings of their own: every run goes through ALL "
                      "their values together, on the same random numbers. " + _IN_EACH_RUN_BODIES + "\n")
            rule = _in_run.one_number_rule((state.get("design") or {}).get("protocol"))
            block += f"{rule}\n" if rule else ""
        return block

    def _frame_template(self, state: Any, tmpl: Any, bodies: Sequence[str] | None = None) -> Any:
        """The code-writing template with what contradicts the frame taken out (``tmpl`` itself when there is no frame): the
        rules of the single script about seeding, pilots, the oracle run and the one-block reply, and the reply format."""
        import string as _string

        if not self._frame_variant(state):
            return tmpl
        return _string.Template(frame_template_text(tmpl.template, bool(self.config.execution.background_jobs), bodies,
                                                    self._figure_panels(state, bodies), self._figure_calls_on()))  # type: ignore[attr-defined]

    def _figure_panels(self, state: Any, bodies: Sequence[str] | None) -> dict[str, int] | None:
        """The panels FI makes for each figure body the request asks for (``figure_<id>`` to the plan's count), else ``None``."""
        if not bodies:
            return None
        from . import analysis_frame as _analysis

        try:
            figures = _analysis.plan_of(state.get("design"))["figures"]
            return {f"figure_{i}": int(f["panels"]) for i, f in figures.items() if f"figure_{i}" in bodies} or None
        except Exception:  # noqa: BLE001 -- a plan FI cannot read gives the request as before, without the statement
            return None

    def _frame_outline_template(self, state: Any, tmpl: Any, bodies: Sequence[str] | None = None) -> Any:
        """The outline request with what contradicts the frame taken out (``tmpl`` itself when there is no frame): the outline
        lists `trial` and its helpers for simulate.py (and the analysis for experiment.py), never an entry point, a loop over
        settings or trials, or a main of the simulation, which are the frame's."""
        import string as _string

        if not self._frame_variant(state):
            return tmpl
        return _string.Template(frame_outline_text(tmpl.template, bodies))
    def _frame_outline(self, state: Any, outline: dict[str, Any]) -> dict[str, Any]:
        """The outline without the entry points of simulate.py (``run_trial``, ``run_cell``, ``oracle``): their signatures are
        the frame's, not the model's. ``outline`` itself when there is no frame."""
        import ast as _ast

        if not self._frame_variant(state) or not isinstance(outline, dict):
            return outline
        out = dict(outline)
        gone = {"run_trial", "run_cell", "oracle"}
        if isinstance(out.get("functions"), list):
            out["functions"] = [f for f in out["functions"] if not (isinstance(f, dict) and f.get("name") in gone)]
        scaffold = out.get("scaffold")
        if isinstance(scaffold, str):
            try:
                tree = _ast.parse(scaffold)
            except SyntaxError:
                return out
            lines = scaffold.splitlines(keepends=True)
            drop = [n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name in gone]
            for n in sorted(drop, key=lambda n: -n.lineno):
                del lines[n.lineno - 1:n.end_lineno]
            out["scaffold"] = "".join(lines)
        return out

    def _heal_frame(self, state: Any) -> bool:
        """Write the ``RESULTS`` line of the frame in the quest's simulate.py again when it is short of the names a trial may return
        now (:func:`core.skeleton.heal_results`): a quest resumed after its code was written keeps the frame it was written with,
        and a repair that puts functions into that file is compared with it. Only that line changes, and only for the same plan.
        Returns whether the file was written."""
        from . import skeleton as _skeleton

        design = state.get("design") if isinstance(state, dict) else None
        if not isinstance(design, dict) or not self._frame_on() or _skeleton.applies(design):
            return False
        path = self.quest_root / "code" / _split_run.SIMULATE_NAME  # type: ignore[attr-defined]
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False
        if not _skeleton.has_frame(text):
            return False
        healed = _skeleton.heal_results(text, design, stochastic=_split_run.design_is_stochastic(design),
                                        strict=self._strict_frame(state))
        if healed == text:
            return False
        if self._study_results_recorded():  # type: ignore[attr-defined]
            # A trial is recorded: the file's text is part of what the run's keys and records were made from, so it is not changed.
            if not getattr(self, "_frame_older_said", False):
                self._frame_older_said = True
                self._log.info("[implement] simulate.py: its frame is older than the plan's list of names a trial may return; "  # type: ignore[attr-defined]
                               "left as it is, because trials of the study are already recorded")
            return False
        import os

        temp = path.with_name(path.name + ".fi-tmp")
        temp.write_text(healed, encoding="utf-8")
        os.replace(temp, path)
        self._log.info("[implement] simulate.py: the frame's list of names a trial may return was written again "  # type: ignore[attr-defined]
                       "(it now holds the names FI's checks read from a trial)")
        return True

    def _framed(self, state: Any, simulate_code: str) -> str:
        """The text of ``simulate.py`` once the model's reply is put into FI's frame (and ``fi_frame.py`` written beside it);
        the reply itself when there is no frame or the reply is a whole simulation of its own."""
        from . import skeleton as _skeleton

        frame = self._skeleton_frame(state)
        if not frame or not simulate_code.strip():
            return simulate_code
        code_dir = self.quest_root / "code"  # type: ignore[attr-defined]
        try:
            previous = (code_dir / _split_run.SIMULATE_NAME).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            previous = ""
        text, note = _skeleton.compose(frame, simulate_code, previous)
        if note:
            # what was left out of the reply is information; a refusal is a warning
            (self._log.info if note.startswith("left out") else self._log.warning)(  # type: ignore[attr-defined]
                "[implement] simulate.py: %s", note)
        if _skeleton.has_frame(text):
            # The runtime is written whenever the file imports it, so a changed frame is never met as a missing module; a quest
            # whose frame an older FI wrote keeps the runtime that frame was written with.
            code_dir.mkdir(parents=True, exist_ok=True)
            _skeleton.ensure_runtime(code_dir, text)
            problem = _skeleton.frame_intact(frame, text)
            if problem:
                # the run then says what changed and the simulation is repaired; the file keeps FI's frame, never the damaged one
                healed, lost = _skeleton.heal_ex(frame, text)
                if lost:  # the reply's own function could not be kept either: say that too
                    problem += f"; the reply's function `trial` could not be used either: {lost}"
                text = _skeleton.refuse(healed, problem)
            else:
                self._log.info("[implement] simulate.py is FI's frame with the model's trial function")  # type: ignore[attr-defined]
        return text

    def _level_rule(self, state: Any, simulate: Any) -> str:
        """FI's one rule for a result compared inside each run (:func:`core.skeleton.level_rule`) when simulate.py has FI's
        frame, else ``""``; a repair of a failed run says it only when a failure is of that kind."""
        from . import skeleton as _skeleton

        try:
            source = Path(simulate).read_text(encoding="utf-8")
        except (OSError, TypeError, ValueError, UnicodeDecodeError):
            return ""
        return _skeleton.level_rule(self._protocol_block(state), source)  # type: ignore[attr-defined]

    # ---- a setting of the grid the code never reads ----------------------------------------------------------------------

    def _check_unread_settings(self, state: Any, simulate: Path) -> dict[str, Any] | None:
        """After the checks of the code before the run (the static gate is clean) and before the run is timed and the
        protocol frozen: a setting of ``protocol.grid`` (two values or more) that FI's reading of the code
        (:func:`core.unread_settings.unread_axes`) finds no use of. FI never takes a setting out of the grid on its own
        reading: a reading of code can be wrong, and a setting taken out is a study that was not run.

        * The study compares the setting (a factor the topic compares, as the plan's own topic map or a name match places
          it; a key of a check's case, of ``in_each_run``, or named by a metric or a comparison) and the simulation never
          reads it: a fault of the code. The simulation goes back to be repaired (a failed run, at most
          ``engine.exec_reflect_max_iterations`` times), naming the setting; after that the grid is kept as written.
        * Otherwise: one plain line in plan.md and run.log says FI's reading finds no use of it, and every setting is kept
          and run.

        Nothing is said when the reading is not certain or the code reads a name the plan does not give (a misspelt read
        is a fault of the code: core/grid_keys.py sends it back), after the freeze, or for a redesign. No model is asked.
        Returns the failed run for the repair, or ``None`` to go on."""
        from . import grid_keys as _grid_keys

        if _frozen.load(self.quest_root) is not None or int(state.get("iteration", 0) or 0) > 0:  # type: ignore[attr-defined]
            return None
        protocol = self._draft_protocol(state) or {}  # type: ignore[attr-defined]
        grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
        if len(grid) < 1:
            return None
        files = {rel: text for rel, text in _grid_keys.code_files(simulate.parent).items()
                 if rel not in ("run.py", _trial_runner.SUBMIT_NAME)}
        misread, compared, unread, why = unread_findings(files, protocol, state.get("design"), self.fi_dir)  # type: ignore[attr-defined]
        if misread:
            self._log.info("[execute] the code reads a name the plan does not give (%s); left to the check of the code",  # type: ignore[attr-defined]
                           ", ".join(f"`{k}`" for r in misread for k in r.keys) or "a fixed number of runs")
            return None
        if compared:
            return self._unread_compared_setting(state, grid, compared, simulate)
        if not unread:
            if why:
                self._log.info("[execute] FI could not tell whether the code reads every setting of the grid (%s); "  # type: ignore[attr-defined]
                               "every setting is kept and run", why)
            return None
        self._note_unread_setting(f"FI's reading of the code finds no use of {', '.join(f'`{k}`' for k in unread)}; every "
                                  "setting is kept and run.")
        return None

    def _note_unread_setting(self, line: str, why: str = "FI found no use of a setting; every setting is kept") -> None:
        """One line about a setting the code does not use (or another fact of FI's small run: ``why`` names it in the
        plan's history), in run.log and in plan.md (once)."""
        from . import engine as _e

        self._log.warning("[execute] %s", line)  # type: ignore[attr-defined]
        path = _plan.plan_path(self.quest_root)  # type: ignore[attr-defined]
        try:
            text = path.read_text(encoding="utf-8")
            if line not in text:
                edited = _plan.add_to_section(text, _e._PART_HEADING, ["", f"- {line}"])
                _plan.note_edit(self.quest_root, text)  # type: ignore[attr-defined]
                path.write_text(edited, encoding="utf-8")
                _plan.record_version(self.quest_root, edited, by="engine",  # type: ignore[attr-defined]
                                     note=why)
        except OSError as e:
            self._log.warning("[execute] FI could not add that line to plan.md: %r", e)  # type: ignore[attr-defined]

    def _unread_in_small_run(self, state: Any, record: dict[str, Any], simulate: Path) -> dict[str, Any] | None:
        """FI's small run saw the code never ask for a setting of the grid (:meth:`core.small_run.SmallRun.reads`: a fact
        FI observed, not a reading of the text). The same consequence as the reading of the code above, which runs later and
        still runs: a setting the study compares sends the simulation back (a failed run, while a repair is left, before the
        freeze); any other is one line in plan.md and run.log. The grid is never changed. ``None`` to go on."""
        reads = record.get("reads") if isinstance(record.get("reads"), dict) else {}
        never = [str(a) for a in reads.get("never") or []]
        if not never or _frozen.load(self.quest_root) is not None or int(state.get("iteration", 0) or 0) > 0:  # type: ignore[attr-defined]
            return None
        protocol = self._draft_protocol(state) or {}  # type: ignore[attr-defined]
        grid = protocol.get("grid") if isinstance(protocol.get("grid"), dict) else {}
        never = [a for a in never if isinstance(grid.get(a), list) and len(grid[a]) >= 2]
        if not never:
            return None
        asked = [str(a) for a in reads.get("unknown") or []]
        compared = settings_the_study_compares(self.fi_dir, state.get("design"), protocol, never)  # type: ignore[attr-defined]
        # The small run covers the middle and the two corners of the grid, so a setting the code reads only under a condition
        # the small run did not reach (a source of one family, a method of one kind) looks unread. Where the text of the code
        # does read it, it is a note and nothing is sent back; the whole study reads it.
        from . import grid_keys as _grid_keys
        from . import unread_settings as _unread

        files = {rel: text for rel, text in _grid_keys.code_files(simulate.parent).items()
                 if rel not in ("run.py", _trial_runner.SUBMIT_NAME)}
        # (the analysis script reads FI's record of the trials, not the cell: its lookups say nothing about the simulation)
        seen = set(_unread.setting_reads(files, analysis=False)[0] or ()) | _literal_keys(
            {rel: text for rel, text in files.items() if rel != "experiment.py"})
        sent = {a: why for a, why in compared.items() if a not in seen}
        if sent:
            return self._unread_compared_setting(state, grid, sent, simulate, ran=True, asked=asked)
        conditional = [a for a in never if a in seen]
        self._note_unread_setting(
            f"FI's small run: the simulation never asked for {', '.join(f'`{k}`' for k in never)}"
            + (f" (the code does read {', '.join(f'`{k}`' for k in conditional)}, under a condition the small run did not reach)"
               if conditional else "") + "; every setting is kept and run.")
        return None

    def _settings_the_study_compares(self, state: Any, protocol: dict[str, Any], axes: list[str]) -> dict[str, str]:
        """``{setting: why it is part of the study}`` for each of ``axes`` that the study compares, from FI's records
        (:func:`settings_the_study_compares`)."""
        return settings_the_study_compares(self.fi_dir, state.get("design"), protocol, axes)  # type: ignore[attr-defined]

    def _unread_compared_setting(self, state: Any, grid: dict[str, Any], compared: dict[str, str],
                                 simulate: Path, *, ran: bool = False, asked: list[str] | None = None) -> dict[str, Any] | None:
        """A setting the study compares and the code never reads: a fault of the code. Sent back to be repaired a bounded
        number of times; after that the grid is kept as written (the run then shows the setting changed nothing)."""
        said = "; ".join(f"`{axis}` ({why})" for axis, why in compared.items())
        if int(state.get("exec_reflect_iter") or 0) >= self.config.engine.exec_reflect_max_iterations:  # type: ignore[attr-defined]
            self._log.warning("[execute] the code still never reads %s; the repairs are used up, so FI keeps the grid as the "  # type: ignore[attr-defined]
                              "plan wrote it", said)
            return None
        self._log.warning("[execute] nothing in the code reads %s, a setting the study compares: sending the simulation back "  # type: ignore[attr-defined]
                          "to be repaired, not taking the setting out of the plan", said)
        print("[FI] the code never reads a setting the study compares; sending the simulation back to be repaired")
        values = "; ".join(f"`{axis}` takes {', '.join(str(v) for v in grid[axis][:6])}" for axis in compared)
        wrong = (f" It asked for {', '.join(f'`{k}`' for k in asked)}, which the plan's grid does not give." if asked else "")
        return self._code_review_failed_run(  # type: ignore[attr-defined]
            f"FI {'ran the simulation on a small case' if ran else 'read the code'} before the run: nothing in it reads "
            f"{said}.{wrong} The plan's grid gives each run its own value "
            f"({values}), and the study compares the results across those values, so the simulation must use the value it is "
            "given (`cell[\"<name>\"]`, the name exactly as the grid spells it) where the plan's model uses it. Do not fix "
            "the value inside the code.", script=simulate.name)

    # ---- the run fits the time it is allowed ---------------------------------------------------------------------------

    def _grid_and_runs(self, state: Any) -> tuple[dict[str, list[Any]], int, dict[str, Any]]:
        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        whole = _optimise_search.study_grid(protocol)
        grid = _optimise_search.with_axes(whole, {str(k): list(v) for k, v in whole.items() if isinstance(v, list) and v})
        thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else {}
        return grid, max(1, int(protocol.get("runs_per_setting") or 1)), thresholds

    def _sizing_key(self, runner: Any, grid: dict[str, list[Any]], runs: int, limit_s: float,
                    inside: dict[str, list[Any]] | None = None) -> str:
        what = {"grid": grid, **({"sub_studies": grid.sub_studies} if getattr(grid, "sub_studies", None) else {}),
                **({"in_each_run": inside} if inside else {})}
        # How many settings run at once changes how long the run takes: a different number is timed again.
        at_once = int(getattr(runner, "at_once", 1) or 1)
        return (_trial_runner._run_key(runner.simulate, what, runs, 0, runner.deterministic) + f"|{int(limit_s)}"
                + (f"|{at_once}" if at_once > 1 else ""))

    async def _size_the_run(self, state: Any, runner: Any, python: Any, env: Any) -> dict[str, Any] | None:
        """Before the protocol is frozen and the study starts: time a few real trials, add them up, and compare the total with
        ``execution.timeout_s``. Fits (or cannot be timed): go on, saying so in one line. Does not fit: FI lowers the runs per
        setting itself (no model is asked, every setting is kept, never below the fewest runs the plan states), times the
        shorter run again, and when it still would not fit returns the quest's stop (no paper). ``None`` to go on.

        Only the trial contract is timed, and only before the first freeze: a frozen protocol cannot be changed, a search for
        the best design or a cluster job has its own limits, and a resume of a run that already fits is not timed again."""
        self._sized_protocol: dict[str, Any] | None = None
        self._pilot_spent = 0.0  # seconds the pilot's extra trials took in this sizing: they count against the same budget
        # Every sizing starts with no cut: what an earlier sizing set (the settings planned, the run's id) is decided again below.
        if isinstance(runner, _trial_runner.TrialsRunner):
            runner.first_cells = None
            runner.run_id = ""
            runner.per_process = 1
            runner.shared_groups = None
            runner.group_allowance_s = None
            if runner.group_reuse is not None:
                runner.group_reuse.discard()
            runner.group_reuse = None
        # A run_trial that returns the mean of many trials (it calls run_cell) is a fault of the code, not of the plan's size:
        # it goes back to the code step, never to the plan to shrink the grid (which dropped settings the topic required).
        if isinstance(runner, _trial_runner.TrialsRunner):
            try:
                text = runner.simulate.read_text(encoding="utf-8")
                aggregate = _trial_runner.trial_aggregates(text, self.quest_root)  # type: ignore[attr-defined]
                first = self._first_study_cell(self._protocol_block(state) or {}) if aggregate else None  # type: ignore[attr-defined]
                if aggregate and first is not None and await _trial_runner.settle_aggregate(
                        self.executor, python, self.quest_root, runner.simulate, cell=first[0], inside=first[1],  # type: ignore[attr-defined]
                        timeout_s=int(self.config.execution.timeout_s), env=env, log=self._log, kind=first[2]) is not False:  # type: ignore[attr-defined]
                    aggregate = ""  # two seeds gave the same numbers, or the call could not be measured (said in the log)
            except (OSError, ValueError):
                aggregate = ""
            if aggregate:
                self._log.warning("[execute] %s; sending simulate.py back to be repaired before the experiment is timed",  # type: ignore[attr-defined]
                                  aggregate)
                return self._sizing_failed(aggregate, text=f"FI read simulate.py before timing the experiment: {aggregate}. "
                                                           "Make run_trial compute one trial (one random draw of the setting) "
                                                           "on its own; keep any mean over trials in run_cell, which may "
                                                           "call run_trial, never the other way round.")
        from . import optimise as _optimise

        if isinstance(runner, _optimise.OptimisationRunner):
            return await self._size_the_scan(state, runner, python, env)
        if not isinstance(runner, _trial_runner.TrialsRunner) or self.config.execution.background_jobs:  # type: ignore[attr-defined]
            return None
        # A frozen protocol (or a redesign) cannot be cut, but its run is still timed and the estimate said before it starts:
        # a redesign's run took four hours with nothing said beforehand because only the first design was timed.
        if int(state.get("iteration", 0) or 0) > 0:  # type: ignore[attr-defined]
            await self._say_estimate_only(state, runner, python, env)
            return None
        if _frozen.load(self.quest_root) is not None:  # type: ignore[attr-defined]
            return None  # a resume of a run that was timed before it was frozen
        # The time the run is sized to: the topic's own limit when it states one (never more than FI's), else FI's
        # ``execution.timeout_s``, which stays the hard stop of a run in either case.
        hard_s = float(self.config.execution.timeout_s)  # type: ignore[attr-defined]
        topic_s = self._topic_limit_s(state)
        by_topic = topic_s is not None and topic_s < hard_s
        limit_s = float(topic_s) if by_topic else hard_s  # type: ignore[arg-type]
        budget_s = max(hard_s * float(self.config.engine.pilot_timeout_frac), 30.0)  # type: ignore[attr-defined]
        cached: dict[str, _estimate.Probe] = {}
        prior = _estimate.read_record(self.quest_root)  # type: ignore[attr-defined]
        # The note of a cut made before a resume is kept, whether the sizing finished or not.
        reduced = prior.get("reduced") if prior.get("fits") is not None else None
        # FI lowers the runs once; the lowered run is timed again, and a resume after the cut does not cut a second time.
        cut_done = bool(reduced) and prior.get("fits") is False
        speed_done = False
        grouping_done = False
        at_once = int(getattr(runner, "at_once", 1) or 1)
        slowdown: float | None = None  # measured once, only when the settings one after another would not fit
        # The timing with the most paid once: a later timing runs on the cache the earlier one filled (run_estimate.with_once).
        once_seen: _estimate.Estimate | None = None
        if float(prior.get("once_s") or 0.0) > 0:  # a resume times on the cache the earlier session filled
            once_seen = _estimate.Estimate(seconds=0.0, at_least=False, complete=True, cells=0, trials=0,
                                           once_s=float(prior["once_s"]), once_axes=list(prior.get("once_axes") or []))
        for _round in range(3):
            grid, runs, thresholds = self._grid_and_runs(state)
            inside = _in_run.of(self._protocol_block(state) or {})  # type: ignore[attr-defined]
            key = self._sizing_key(runner, grid, runs, limit_s, inside)
            record = _estimate.read_record(self.quest_root)  # type: ignore[attr-defined]
            if record.get("key") == key and record.get("fits"):
                self._log.info("[execute] %s (timed before; nothing in the experiment has changed)",  # type: ignore[attr-defined]
                               record.get("says") or "the experiment fits")
                if int(record.get("per_process") or 1) > 1 and record.get("shared_groups"):
                    # the groups the check ran are kept: a resumed run shares processes the same way, with no new timing
                    runner.per_process = int(record["per_process"])  # type: ignore[attr-defined]
                    runner.shared_groups = [list(g) for g in record["shared_groups"]]  # type: ignore[attr-defined]
                    runner.group_allowance_s = record.get("group_allowance_s")  # type: ignore[attr-defined]
                return None
            if record.get("key") == key and record.get("truncated"):
                self._log.info("[execute] %s (timed before; nothing in the experiment has changed)",  # type: ignore[attr-defined]
                               record.get("says") or "the experiment is run up to the time limit")
                runner.cost = None  # type: ignore[attr-defined]  # the plan's order
                runner.run_id = str((record.get("truncated") or {}).get("run_id") or "")  # type: ignore[attr-defined]
                runner.first_cells = (record.get("truncated") or {}).get("planned_settings")  # type: ignore[attr-defined]
                return None
            if record.get("key") == key and record.get("stopped"):
                return self._stop_too_long(str(record["stopped"]), fit=record.get("fit"))
            probes, why = await _estimate.measure(
                self.executor, python, self.quest_root, runner.simulate.relative_to(self.quest_root).as_posix(), grid,  # type: ignore[attr-defined]
                runs=runs, deterministic=runner.deterministic, limit_s=limit_s,
                budget_s=max(0.0, budget_s - self._pilot_spent), env=env,
                thresholds=thresholds, cached=cached, in_each_run=inside)
            if probes is None:
                # A setting that fails when timed is a setting the simulation fails at: the same repair as a failed run, naming
                # it (the error only; the repair is never given an expected value).
                self._log.warning("[execute] the simulation failed while FI timed the experiment %s", why)  # type: ignore[attr-defined]
                return self._sizing_failed(why)
            for p in probes:
                cached[_estimate.probe_key(p.cell, runs)] = p
            # The checks of the pilot's trials get the time they always had (FI's own limit, not the topic's).
            if (broken := await self._pilot_gate(state, runner, python, env, grid, probes, thresholds, hard_s, budget_s)) is not None:
                return broken
            if isinstance(runner, _trial_runner.TrialsRunner):
                runner.cost = _estimate.setting_cost(grid, probes)  # the run starts the costliest settings first
            est = _estimate.with_once(_estimate.estimate(grid, runs, runner.deterministic, probes), once_seen)
            if est is None:
                return None
            if est.once_s > (once_seen.once_s if once_seen is not None else 0.0):
                once_seen = est
            if at_once > 1 and _estimate.known_too_long(est, limit_s):
                if slowdown is None:
                    slowdown = await self._slowdown_at_once(runner, python, env, probes, runs, thresholds, inside,
                                                            at_once, budget_s)
                if slowdown is not None:
                    est = _estimate.with_once(_estimate.estimate(grid, runs, runner.deterministic, probes, at_once=at_once,
                                                                 slowdown=slowdown), once_seen)
            # The timings are kept term by term (core/cost_model.py), not only their total; nothing reads them back yet.
            _cost_model.keep_measured(self.quest_root, probes=probes, est=est, slowdown=slowdown, limit_s=limit_s, key=key)  # type: ignore[attr-defined]
            if not speed_done and not cut_done and isinstance(runner, _trial_runner.TrialsRunner):
                # After the correctness checks and before the study is sized: a study that does not fit is sped up before it
                # is cut, and one that fits only when it is long (core/improve_speed.py); a kept speed-up is timed again.
                speed_done = True
                # a grid larger than FI lists does not fit either way: it is cut too
                too_large = _optimise_search.grid_size(grid) > _optimise_search.MAX_LISTED_SETTINGS
                if (sped := await self._speed_up_if_long(
                        state, runner, python, env, grid, runs, thresholds, inside, est, limit_s,
                        fits=None if not too_large else False, budget_s=max(0.0, budget_s - self._pilot_spent), probes=probes)) is not None \
                        and sped.get("kept"):
                    # The simulation is a different one now: what was measured on the slow one (the slowdown at once, whether
                    # settings may share processes, with the time that check was given) is measured again on the faster one.
                    cached.clear()
                    slowdown = None
                    grouping_done = False
                    # the slower of two timings of the faster simulation is the one the next decision is made on
                    for p in sped.get("probes") or []:
                        cached[_estimate.probe_key(p.cell, runs)] = p
                    continue
            line = _estimate.says(est, limit_s, topic=by_topic)
            base = {"key": key, "estimate_s": est.seconds * _estimate.SAFETY, "limit_s": limit_s, "says": line,
                    "reduced": reduced, "at": time.time(), "timed": est.timed, "cells": est.cells, "trials": est.trials,
                    "limit": "topic" if by_topic else "execution.timeout_s", "topic_limit_s": topic_s, "hard_limit_s": hard_s,
                    "once_s": est.once_s, "once_axes": list(est.once_axes),
                    "at_once": est.at_once}
            # The smaller run FI chose is timed again: it goes on unless it is over the limit even without the margin
            # (the margin was spent choosing it; a second timing of the same run differs by about as much).
            # More settings than FI lists: the whole grid is never built. The settings the time allowed can reach are planned,
            # in the plan's order, and run up to the limit (a study that fits would still be cut at what FI lists).
            oversized = _optimise_search.grid_size(grid) > _optimise_search.MAX_LISTED_SETTINGS
            if not oversized and not _estimate.known_too_long(est, limit_s, timed_again=cut_done):
                if _estimate.known_too_long(est, limit_s):
                    line += (" (timed again with the fewer runs; without FI's margin of one half it is "
                             f"{_estimate.plain_duration(est.seconds)}, within the limit)")
                    base["says"] = line
                self._log.info("[execute] %s", line)  # type: ignore[attr-defined]
                print(f"[FI] {line}")
                self._say_headroom(est, hard_s)
                _estimate.write_record(self.quest_root, {**base, "fits": True})  # type: ignore[attr-defined]
                return None
            if oversized:
                fits_in = self._what_fits(est, hard_s)
                reach = int(fits_in["settings_that_fit"]) if fits_in else _optimise_search.MAX_LISTED_SETTINGS
                first = max(1, min(_optimise_search.MAX_LISTED_SETTINGS, reach))
                if self._run_up_to_the_limit(runner, est, hard_s, _estimate.cheapest_setting_s(probes), kind="grid",
                                             base=base, detail=self._too_long_text(est, hard_s), fit=fits_in,
                                             first_cells=first):
                    return None
                text = _optimise_search.grid_too_large(_optimise_search.study_grid(self._protocol_block(state) or {})) \
                    or self._too_long_text(est, hard_s)
                _estimate.write_record(self.quest_root, {**base, "fits": False, "stopped": text, "fit": fits_in})  # type: ignore[attr-defined]
                return self._stop_too_long(text, fit=fits_in)
            self._log.warning("[execute] %s, so it would not finish within it", line)  # type: ignore[attr-defined]
            if not grouping_done and isinstance(runner, _trial_runner.TrialsRunner):
                # Over the time allowed: before anything is cut, settings may share processes when a sample shows the
                # very same numbers (core/trial_runner.py ``measure_grouping``); below the limit nothing here runs.
                grouping_done = True
                shared = await self._group_settings(state, runner, python, env, grid, runs, thresholds, inside, est, limit_s,
                                                    budget_s=max(0.0, budget_s - self._pilot_spent))
                if shared is not None:
                    said, grouped_s, grouped_est = shared
                    line += f" {said}"
                    base.update(says=line, ungrouped_estimate_s=base["estimate_s"], estimate_s=grouped_s * _estimate.SAFETY)
                    print(f"[FI] {said}")
                    self._say_headroom(grouped_est, hard_s)
                    _estimate.write_record(self.quest_root, {  # type: ignore[attr-defined]
                        **base, "fits": True, "per_process": runner.per_process, "shared_groups": runner.shared_groups,
                        "group_allowance_s": runner.group_allowance_s})
                    return None
            protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
            if cut_done:
                # The shorter run, timed again, is still over: over the topic's limit but within FI's own, it runs and the
                # paper says it was over; over FI's own, the quest stops (below).
                if by_topic and not _estimate.known_too_long(est, hard_s):
                    over = (f"Timed again with the fewer runs, the experiment is expected to take {self._padded(est)}, over the "
                            f"topic's limit of {_estimate.plain_duration(limit_s)}; FI runs it, and the paper reports the "
                            "measured run time against that limit.")
                    self._log.warning("[execute] %s", over)  # type: ignore[attr-defined]
                    if isinstance(reduced, dict):
                        reduced = {**reduced, "note": f"{reduced.get('note', '')} {over}".strip(),
                                   "over_topic_limit": "cut_still_over"}
                    self._say_headroom(est, hard_s)
                    _estimate.write_record(self.quest_root, {**base, "reduced": reduced, "fits": False})  # type: ignore[attr-defined]
                    return None
                fit = None
            else:
                # Fewer runs per setting is the only cut FI makes: every setting the study compares is kept. The runs go down
                # to the fewest the plan states, or, for the topic's own limit, below it (said in the paper). The shorter run
                # is timed again before it is used. No model is asked.
                fit = _estimate.fit_runs(grid, runs, runner.deterministic, probes, protocol, limit_s=limit_s,
                                         topic_limit=by_topic, spreads=est.spreads, at_once=est.at_once,
                                         slowdown=est.slowdown)
            over_topic = ""
            if fit is None and by_topic and not cut_done:
                # Not even the fewest runs meet the topic's limit: going below the plan's minimum would not meet it either,
                # so the plan's minimum stands and the run is nearest the limit there, over it (said in the paper). Only a
                # run over FI's own limit stops.
                floor = _estimate.runs_floor(protocol, runs, est.spreads) if _estimate.floor_is_stated(protocol) else runs
                floor = min(runs, max(floor, _estimate.runs_checks_need(protocol, grid)))  # what the plan's checks need
                least = _estimate.fewest_for_topic(protocol, grid, runs)
                fewest = _estimate.seconds_at(grid, least, runs, runner.deterministic, probes, at_once=est.at_once,
                                              slowdown=est.slowdown) or 0.0
                at_floor = _estimate.seconds_at(grid, floor, runs, runner.deterministic, probes, at_once=est.at_once,
                                                slowdown=est.slowdown) or est.seconds
                if at_floor * _estimate.SAFETY <= hard_s:
                    over_topic = (f"The topic's limit of {_estimate.plain_duration(limit_s)} cannot be met by fewer runs per "
                                  f"setting: even with {least} runs per setting the {est.cells:,} settings would take about "
                                  f"{_estimate.plain_duration(fewest * _estimate.SAFETY)} with FI's margin (each setting runs "
                                  "in a process of its own).")
                    if floor < runs:
                        fit = _estimate.Fit(runs=floor, seconds=at_floor, plan_floor=floor, below_plan=False)
                    else:
                        note = (f"{over_topic} FI ran the plan's {runs} runs per setting and all {est.cells:,} settings; the "
                                f"experiment is expected to take {self._padded(est)}, over the topic's limit.")
                        self._log.warning("[execute] %s", note)  # type: ignore[attr-defined]
                        print(f"[FI] {note}")
                        self._say_headroom(est, hard_s)
                        _estimate.write_record(self.quest_root, {**base, "fits": False, "over_topic_limit": {"note": note}})  # type: ignore[attr-defined]
                        return None
            if fit is None:
                text = self._too_long_text(est, hard_s)
                fits_in = self._what_fits(est, hard_s)
                if self._run_up_to_the_limit(runner, est, hard_s, _estimate.cheapest_setting_s(probes), kind="grid", base=base,
                                             detail=text, fit=fits_in):
                    return None
                _estimate.write_record(self.quest_root, {**base, "fits": False, "stopped": text, "fit": fits_in})  # type: ignore[attr-defined]
                return self._stop_too_long(text, fit=fits_in)
            note = self._lower_the_runs(state, runs, fit, est, limit_s, grid, protocol, topic=by_topic, over_topic=over_topic)
            if note is None:
                text = self._too_long_text(est, hard_s)
                fits_in = self._what_fits(est, hard_s)
                if self._run_up_to_the_limit(runner, est, hard_s, _estimate.cheapest_setting_s(probes), kind="grid", base=base,
                                             detail=text, fit=fits_in):
                    return None
                _estimate.write_record(self.quest_root, {**base, "fits": False, "stopped": text, "fit": fits_in})  # type: ignore[attr-defined]
                return self._stop_too_long(text, fit=fits_in)
            cut_done = True
            reduced = note
            self._sized_protocol = {**(self._sized_protocol or {}), "runs_per_setting": fit.runs}
            _estimate.write_record(self.quest_root, {**base, "fits": False, "reduced": reduced})  # type: ignore[attr-defined]
            await self._settle_plan_sources(state, protocol=self._draft_protocol(state))  # type: ignore[attr-defined]
            if over_topic:
                return None  # sized to the plan's minimum, knowingly over the topic's limit: nothing to gain by timing again
        return None

    async def _say_estimate_only(self, state: Any, runner: Any, python: Any, env: Any) -> None:
        """The estimate of a run whose protocol cannot be changed (frozen, or a redesign): time the same settings as for a first
        design, say the one plain line, and go on. Nothing is cut, stopped or recorded as sized, and a timing that fails or finds
        nothing says nothing (a failed timing never prints an estimate)."""
        try:
            hard_s = float(self.config.execution.timeout_s)  # type: ignore[attr-defined]
            topic_s = self._topic_limit_s(state)
            by_topic = topic_s is not None and topic_s < hard_s
            limit_s = float(topic_s) if by_topic else hard_s  # type: ignore[arg-type]
            budget_s = max(hard_s * float(self.config.engine.pilot_timeout_frac), 30.0)  # type: ignore[attr-defined]
            grid, runs, thresholds = self._grid_and_runs(state)
            inside = _in_run.of(self._protocol_block(state) or {})  # type: ignore[attr-defined]
            probes, why = await _estimate.measure(
                self.executor, python, self.quest_root, runner.simulate.relative_to(self.quest_root).as_posix(), grid,  # type: ignore[attr-defined]
                runs=runs, deterministic=runner.deterministic, limit_s=limit_s, budget_s=budget_s, env=env,
                thresholds=thresholds, in_each_run=inside)
            if probes is None:
                self._log.info("[execute] the run was not timed before it started (%s)", why)  # type: ignore[attr-defined]
                return
            est = _estimate.estimate(grid, runs, runner.deterministic, probes)
            if est is None:
                return
            line = _estimate.says(est, limit_s, topic=by_topic)
            self._log.info("[execute] %s (this design is not cut: its plan is already fixed)", line)  # type: ignore[attr-defined]
            print(f"[FI] {line}")
            self._say_headroom(est, hard_s)
        except Exception as exc:  # noqa: BLE001 - a number said for information must never stop the run
            self._log.warning("[execute] the run could not be timed before it started (%r)", exc)  # type: ignore[attr-defined]

    async def _slowdown_at_once(self, runner: Any, python: Any, env: Any, probes: list[Any], runs: int,
                                thresholds: dict[str, Any], inside: Any, at_once: int, budget_s: float,
                                deterministic: bool | None = None) -> float | None:
        """How many times slower a setting runs when ``at_once`` run at the same time on this machine
        (:func:`core.run_estimate.slowdown_at_once`), or ``None`` when it could not be measured in the time left for timing;
        then the estimate counts the settings one after another, as before. It is measured on the costliest timed setting
        that finished and can be timed again in the time left: the costly settings are most of the run, and running beside
        others slows them the most (a short setting is mostly its process start, which hardly slows)."""
        left = budget_s - self._pilot_spent

        def timing_of(p: Any) -> float:
            # What timing the setting took: its timed trials, not the whole setting.
            if p.per_trial_s is not None and p.timed_trials > 0:
                return min(p.seconds, (p.fixed_s or 0.0) + p.timed_trials * p.per_trial_s)
            return p.seconds

        finished = [p for p in probes if not p.at_least and p.seconds > 0]
        # Alone once, then beside the others: at worst as slow as all of them one after another.
        fits = [p for p in finished if left >= timing_of(p) * (1 + at_once) + 10]
        base = max(fits, key=timing_of, default=None)
        if base is None:
            why = "no timed setting finished" if not finished else "no time was left for it"
            self._log.info("[execute] FI did not measure how much slower settings run at the same time (%s); the "  # type: ignore[attr-defined]
                           "estimate counts them one after another", why)
            return None
        timing = timing_of(base)
        started = time.monotonic()
        factor, why = await _estimate.slowdown_at_once(
            self.executor, python, self.quest_root, runner.simulate.relative_to(self.quest_root).as_posix(),  # type: ignore[attr-defined]
            dict(base.cell), at_once=at_once, runs=runs,
            deterministic=runner.deterministic if deterministic is None else deterministic,
            timeout_s=int(max(30, 3 * timing + 10)), env=env, thresholds=thresholds, in_each_run=inside)
        self._pilot_spent += time.monotonic() - started
        if factor is None:
            self._log.info("[execute] FI could not measure how much slower settings run at the same time (%s); the "  # type: ignore[attr-defined]
                           "estimate counts them one after another", why or "no reason given")
            return None
        self._log.info("[execute] with %d settings running at the same time, each took %.1f times as long as alone (timed "  # type: ignore[attr-defined]
                       "on a setting that took %s alone)", at_once, factor, _estimate.plain_duration(timing))
        return factor

    def _topic_limit_s(self, state: Any) -> float | None:
        """The one run-time limit the topic states (in seconds), read as the paper's run record reads it, or ``None``."""
        from . import disclosure as _disclosure

        limit = _disclosure.topic_limit(self.quest_root, str(state.get("topic") or ""))  # type: ignore[attr-defined]
        return float(limit) if isinstance(limit, (int, float)) and limit > 0 else None

    @staticmethod
    def _padded(est: _estimate.Estimate) -> str:
        return ("at least " if est.at_least else "about ") + _estimate.plain_duration(est.seconds * _estimate.SAFETY)

    def _say_headroom(self, est: _estimate.Estimate, hard_s: float) -> None:
        """One line when FI's hard stop (``execution.timeout_s``) is close to what the run is expected to take: a run slower
        than estimated would be stopped before it ends."""
        if est.seconds * _estimate.SAFETY * 2 > hard_s:
            self._log.warning(  # type: ignore[attr-defined]
                "[execute] FI stops a run that takes longer than execution.timeout_s (%s); this one is expected to take %s, "
                "so a run much slower than the estimate would be stopped before it ends",
                _estimate.plain_duration(hard_s), self._padded(est))

    async def _size_the_scan(self, state: Any, runner: Any, python: Any, env: Any,
                             speed_done: bool = False) -> dict[str, Any] | None:
        """The coarse scan of a search for the best design runs every setting of the plan's grid, so its size is counted
        (the product of the axes' lengths, nothing listed) and its time added up BEFORE anything lists or runs the settings.
        More settings than FI lists: the quest's plain stop. Otherwise the scan's time is the larger of the plan's own estimate
        and FI's timing of the scan's settings as a measurement's are timed (a plan's claim alone is not trusted), and the
        search's evaluations, which share the scan's time, are added at what one new design (the search's baseline) costs;
        over ``execution.timeout_s``: the same stop, as for a measurement (FI does not run a smaller scan than the plan's),
        with what fits for the plan. ``None`` to go on. Before the study is sized, one that does not fit, or fits and is
        long (:meth:`_speed_up_if_long`), has its simulation made faster with the same numbers as a measurement's is
        (:meth:`_speed_up`, once; ``speed_done``), on the scan's settings and a few designs of the search's own space, and
        is timed again.

        Not skipped for a frozen protocol or a background job: the search runs here either way and sizing never changes the
        protocol; the timing is bounded by ``engine.pilot_timeout_frac`` of the time allowed, and a resume repeats it."""
        from . import optimisation_plan as _optimisation_plan
        from . import optimise as _optimise
        from . import optimise_search as _optimise_search

        if self._search_already_run(runner, env):
            return None
        earlier = (_estimate.read_record(self.quest_root).get("truncated") or {})  # type: ignore[attr-defined]
        self._clear_cut(runner)  # what an earlier sizing of this study decided is decided again below
        raw = _optimise.block_of(self._protocol_block(state))  # type: ignore[attr-defined]
        block, _why = _optimisation_plan.normalize(raw) if raw is not None else (None, "")
        if block is None:
            return None  # the run reports a block it cannot read
        scan = _optimisation_plan.evaluations(block)[1]
        if scan <= 1:
            return None
        try:
            _optimise_search.check_listable(_optimise_search.scan_grid(block))
        except _optimise_search.TooManySettings as exc:
            return self._stop_too_long(str(exc))
        except (KeyError, TypeError, ValueError):
            pass  # a block the scan cannot be built from is reported by the run
        deterministic = "run_trial" not in _trial_runner.entries(runner.simulate)
        runs = 1 if deterministic else max(1, int(block.get("runs_per_evaluation") or 1))
        limit_s = float(self.config.execution.timeout_s)  # type: ignore[attr-defined]
        claimed = (block.get("evaluation_budget") or {}).get("seconds_per_evaluation")
        per_setting = float(claimed) * runs if isinstance(claimed, (int, float)) and not isinstance(claimed, bool)             and claimed > 0 else None
        thresholds = (self._protocol_block(state) or {}).get("thresholds")  # type: ignore[attr-defined]
        # The scan's settings are timed as a measurement's are (core/run_estimate.py::measure): the base twice, each setting
        # at its ends (and quarter points), every part of the scan, so a cost that depends on the settings, or is paid once
        # per new value and reused, is counted as the scan will pay it; one evaluation of the baseline alone was not.
        est: _estimate.Estimate | None = None
        first_s = 0.0  # what the scan's first timed setting took the first time it ran
        probes: list[Any] | None = None
        at_once = int(getattr(runner, "at_once", 1) or 1)
        slowdown: float | None = None  # how much slower a setting runs beside ``at_once - 1`` others, once measured
        timing_started = time.monotonic()
        budget_s = max(limit_s * float(self.config.engine.pilot_timeout_frac), 30.0)  # type: ignore[attr-defined]
        try:
            grid = _optimise_search.scan_grid(block)
            inside = _optimise_search.in_each_run(block)
        except (KeyError, TypeError, ValueError):
            grid, inside = None, {}
        if grid is not None:
            probes, why = await _estimate.measure(
                self.executor, python, self.quest_root, runner.simulate.relative_to(self.quest_root).as_posix(),  # type: ignore[attr-defined]
                grid, runs=runs, deterministic=deterministic, limit_s=limit_s, budget_s=budget_s, env=env,
                thresholds=thresholds, in_each_run=inside or None)
            if probes is None:
                # A setting that fails when timed is a setting the simulation fails at, as for a measurement: the same repair
                # as a failed run, before the protocol is frozen. Going on, the scan was sized from the plan's claim alone.
                self._log.warning("[execute] the simulation failed while FI timed the search's coarse scan %s", why)  # type: ignore[attr-defined]
                return self._sizing_failed(why)
            else:
                est = _estimate.estimate(grid, runs, deterministic, probes)
                if probes:
                    first_s = float(probes[0].first_s or probes[0].seconds)
                # The scan runs ``execution.settings_at_once`` settings at the same time (core/optimise.py::run_search), so
                # it is counted that way, with how much slower a setting runs beside the others on this machine, as a
                # measurement is (_size_the_run); one after another it was counted many times too long.
                if at_once > 1 and probes and _estimate.known_too_long(est, limit_s):
                    spent = time.monotonic() - timing_started
                    slowdown = await self._slowdown_at_once(runner, python, env, probes, runs, thresholds or {},
                                                            inside or None, at_once, max(0.0, budget_s - spent),
                                                            deterministic=deterministic)
                    if slowdown is not None:
                        est = _estimate.estimate(grid, runs, deterministic, probes, at_once=at_once, slowdown=slowdown)
        if probes:
            _cost_model.keep_measured(self.quest_root, probes=probes, est=est, slowdown=slowdown, at_once=at_once,  # type: ignore[attr-defined]
                                      limit_s=limit_s, plan={"scan": scan}, scan_s=est.seconds if est is not None else None,
                                      design=_cost_model.design_key(block, runner.simulate))
        if per_setting is not None and (est is None or per_setting * scan > est.seconds):
            est = _estimate.Estimate(seconds=per_setting * scan, at_least=False, complete=True, cells=scan, trials=scan * runs)
        if est is None:
            return None
        scan_alone = est.seconds  # the coarse scan's own seconds, before the search and the check are added to it
        searched = 0
        sized: Any = None  # FI's size for the search (core/cost_model.py::search_plan), once the study was found too long
        core_est: Any = None  # the scan and the search without the check, once the check was added
        check_s = 0.0
        checked = 0  # the evaluations of FI's check at finer settings, counted when they make the study too long
        whole: Any = None  # the scan with the search's evaluations, when one evaluation was timed
        alternatives_note = ""  # how many alternatives the search's evaluations are for, when the plan compares several
        alternatives = 1
        searching = 1  # how many of them the run searches
        per_alternative = 0
        probe_n = 0
        per_eval: float | None = None  # what one new design costs, once timed
        lower = False
        if not _estimate.known_too_long(est, limit_s):
            # The search shares the scan's time (one deadline for both, core/optimise.py): every evaluation is a design
            # the scan did not compute, so it costs what a setting costs the first time it runs: the search's own
            # baseline timed once here, never less than the scan's first timed setting took the first time (the baseline
            # may run on what the scan's timing computed) nor than the plan's own claim. Each evaluation is a whole
            # setting with its runs.
            search_n = int(_optimisation_plan.evaluations(block)[0] or 0)
            # A study that compares named alternatives searches each one with the whole budget (the alternatives are
            # counted from the plan's own scan, core/optimise_search.py::compared_levels), and asks a few designs of every
            # alternative after the first before it is searched (compare): evaluations outside the budget, counted here.
            alternatives = searching = _optimise_search.level_count(block)
            per_alternative = search_n // alternatives if search_n else 0
            probe_n = (alternatives - 1) * _optimise_search.PROBE_DESIGNS if search_n and alternatives > 1 else 0
            per_eval = max(per_setting or 0.0, first_s) or None
            left = budget_s - (time.monotonic() - timing_started)
            if search_n and left >= 5:
                baseline = _optimise.complete_case(self._protocol_block(state), {}, keep_inside=True)  # type: ignore[attr-defined]
                allowance = max(60.0, 3.0 * limit_s / max(1, (scan + search_n) * runs))
                probe, why = await _estimate._time_one(
                    self.executor, python, self.quest_root, runner.simulate.relative_to(self.quest_root).as_posix(),  # type: ignore[attr-defined]
                    baseline, runs=runs, deterministic=deterministic, timeout_s=int(max(5, min(allowance, left))),
                    env=env, thresholds=thresholds, in_each_run=inside or None)
                if probe is None:
                    # The baseline is the first design the search evaluates: one it fails at is the same repair.
                    self._log.warning("[execute] the simulation failed while FI timed the search's baseline %s", why)  # type: ignore[attr-defined]
                    return self._sizing_failed(why)
                if per_eval is None or probe.seconds > per_eval:
                    per_eval, lower = probe.seconds, probe.at_least
            if search_n and per_eval:
                # The search asks for designs one at a time or several together (its starting points side by side, a
                # population, a simplex's corners), as the search itself shows when it is rehearsed (_search_rounds);
                # several together run ``at_once`` at a time, each as much slower as FI measured. Counted one after
                # another, a search that runs 8 at a time was counted several times too long.
                # One search is rehearsed (all alternatives search alike); what is counted is its rounds once for each
                # alternative the run searches, and the probes of every alternative after the first, asked together.
                alone1, batches1, together1 = await asyncio.to_thread(_search_rounds, block, per_alternative, at_once)

                def priced(searched_n: int) -> tuple[int, int, int, int]:
                    a, b, t = alone1 * searched_n, batches1 * searched_n, together1 * searched_n
                    if probe_n and at_once > 1:
                        b += (alternatives - 1) * math.ceil(_optimise_search.PROBE_DESIGNS / at_once)
                        t += probe_n
                    elif probe_n:
                        a += probe_n
                    return a, b, t, per_alternative * searched_n + probe_n

                # Every alternative is searched, until the simulation itself shows otherwise (below).
                alone, batches, together_n, search_n = priced(searching)
                if batches and slowdown is None and probes and _estimate.known_too_long(
                        _replace(est, seconds=est.seconds + per_eval * search_n), limit_s):
                    spent = time.monotonic() - timing_started
                    slowdown = await self._slowdown_at_once(runner, python, env, probes, runs, thresholds or {},
                                                            inside or None, at_once, max(0.0, budget_s - spent),
                                                            deterministic=deterministic)
                    if slowdown is not None and grid is not None:
                        est = _estimate.estimate(grid, runs, deterministic, probes, at_once=at_once,
                                                 slowdown=slowdown) or est
                        scan_alone = est.seconds

                def whole_study(a: int, b: int, t: int, n: int) -> _estimate.Estimate:
                    search_s = per_eval * (a + (b * slowdown if slowdown is not None else t))
                    return _estimate.Estimate(seconds=est.seconds + search_s, at_least=est.at_least or lower,
                                              complete=est.complete, cells=scan + n, trials=(scan + n) * runs,
                                              once_s=est.once_s, once_axes=list(est.once_axes))

                together = whole_study(alone, batches, together_n, search_n)
                if alternatives > 1 and _estimate.known_too_long(together, limit_s):
                    # Too long when every alternative is searched: the run searches only those that give results
                    # different from the first one's (compare), so FI asks the simulation, as the run will, before it
                    # says the study does not fit.
                    left = budget_s - (time.monotonic() - timing_started)
                    searching = await self._alternatives_that_differ(
                        state, block, runner, python, env, deterministic=deterministic, at_once=at_once,
                        thresholds=thresholds, left_s=left) or alternatives
                    if searching < alternatives:
                        alone, batches, together_n, search_n = priced(searching)
                        together = whole_study(alone, batches, together_n, search_n)
                if alternatives > 1:
                    alternatives_note = (
                        f" ({alternatives} alternatives the plan compares"
                        + (f", {searching} of them searched (the others give the first one's results)"
                           if searching < alternatives else ", each searched")
                        + f" with {per_alternative:,} evaluations, and {probe_n:,} evaluations that tell whether an "
                        "alternative gives the same results as the first)")
                    if searching < alternatives:
                        self._log.info(  # type: ignore[attr-defined]
                            "[execute] FI asked the simulation for %d designs at each of the %d alternatives the plan "
                            "compares; the others give the first one's results, so the search is counted for %d of "
                            "them, not %d", _optimise_search.PROBE_DESIGNS, alternatives, searching, alternatives)
                whole = together
                if _estimate.known_too_long(together, limit_s):
                    est, searched = together, search_n
            # FI's check of the best design at finer settings runs after the search and is held to the same time limit
            # (core/optimise.py): its evaluations are counted at what one design costs here, at the least (a finer setting
            # costs more, so this is a lower bound; a check of 72 runs at ~75 minutes each was not counted and ran far past
            # the limit).
            try:
                check_n = int(_optimisation_plan.budget(block).get("check") or 0)
            except (KeyError, TypeError, ValueError):
                check_n = 0
            unit = per_eval or first_s or None
            finer_s: float | None = None
            finer_basis, finer_low = _cost_model.MEASURED, False
            check_batches: list[int] = []
            check_trials, check_together = 1, at_once
            if check_n and unit:
                # Whenever a check is planned, one design of it at its finest settings is timed: a finer setting costs far
                # more than one design of the search (a check of 72 runs at ~30 minutes each was counted at ~4 minutes a run).
                finer_s, finer_basis, finer_low, finer_mb = await self._time_finer_check(
                    block, state, runner, python, env, deterministic=deterministic, thresholds=thresholds, inside=inside,
                    left_s=budget_s - (time.monotonic() - timing_started), limit_s=limit_s)
                core_est = whole if whole is not None else est
                check_trials = max(1, int(_optimisation_plan.budget(block).get("check_runs") or 1))
                check_batches = _optimisation_plan.check_batches(block)
                # The check runs its designs together like the search (every level, then every nudge, are one request each),
                # as many at a time as the computer's memory lets run, in whole waves, and as much slower beside each
                # other as FI measured (none measured: one after another). See core/cost_model.py::check_wall_s.
                check_together = _execution.at_once_for_memory(at_once, finer_mb)
                if finer_s is None:
                    self._log.info("[execute] the check of the best design is counted at the cost of one search design, at "  # type: ignore[attr-defined]
                                   "the least: one design of it at its finest settings could not be timed")
                per_trial = finer_s if finer_s else unit / runs
                check_s = _cost_model.check_wall_s(per_trial * check_trials, check_batches, at_once=check_together,
                                                   slowdown=slowdown)
                with_check = _replace(core_est, seconds=core_est.seconds + check_s, at_least=True)
                whole = with_check
                if _estimate.known_too_long(with_check, limit_s):
                    est, checked = with_check, check_n
                    if not searched and search_n:
                        searched = search_n
                    elif not searched:
                        searched = 0
            design_key = _cost_model.design_key(block, runner.simulate)
            _cost_model.keep_measured(self.quest_root, per_evaluation_s=per_eval, per_evaluation_at_least=lower, slowdown=slowdown,  # type: ignore[attr-defined]
                                      check_unit_s=unit if check_n else None,
                                      check_unit_basis="search-design lower bound",
                                      finer_unit_s=finer_s, finer_unit_basis=finer_basis, finer_unit_at_least=finer_low,
                                      scan_s=scan_alone,
                                      plan={"evaluations": search_n, "check_runs": check_n, "scan": scan, "runs_per_evaluation": runs,
                                            "check_batches": check_batches, "check_trials": check_trials,
                                            "check_at_once": check_together},
                                      design=design_key)
            sized = self._size_the_search(block, limit_s, check_n, scan_alone, design_key, searched or checked)
            if not searched and not checked:
                # The line a person reads is the whole study: the scan alone said "about 6 minutes" of a search that ran
                # for hours.
                if whole is not None:
                    self._log.info("[execute] the search's coarse scan and its %s evaluations: %s (the coarse scan alone: "  # type: ignore[attr-defined]
                                   "%s)", f"{search_n:,}", _estimate.says(whole, limit_s),
                                   _estimate.plain_duration(est.seconds * _estimate.SAFETY))
                else:
                    self._log.info("[execute] the search's coarse scan: %s", _estimate.says(est, limit_s))  # type: ignore[attr-defined]
                # It fits, and a long study is still sped up before it runs (core/improve_speed.py), on the whole study's
                # estimate (the scan, the search and FI's check at finer settings); a kept speed-up times it all again.
                if not speed_done and grid is not None:
                    sped = await self._speed_up_if_long(
                        state, runner, python, env, grid, runs, thresholds or {}, inside or {},
                        whole if whole is not None else est, limit_s,
                        budget_s=max(0.0, budget_s - (time.monotonic() - timing_started)), probes=probes,
                        deterministic=deterministic, extra_cells=_search_sample(block, grid))
                    if sped is not None and sped.get("kept"):
                        return await self._size_the_scan(state, runner, python, env, speed_done=True)
                return None
        if not speed_done and grid is not None:
            # What FI runs is the same simulation, whatever the study is called: the speed-up a measurement gets is tried
            # here too, on the scan's settings and a few designs the search itself would try, before the study is sized.
            sped = await self._speed_up_if_long(
                state, runner, python, env, grid, runs, thresholds or {}, inside or {}, est, limit_s,
                budget_s=max(0.0, budget_s - (time.monotonic() - timing_started)), probes=probes,
                deterministic=deterministic, extra_cells=_search_sample(block, grid))
            if sped is not None and sped.get("kept"):
                return await self._size_the_scan(state, runner, python, env, speed_done=True)
        text = self._too_long_text(est, limit_s, cut=False)
        if searched or checked:
            text = (f"the coarse scan's {scan:,} settings and the search's {searched:,} evaluations{alternatives_note}"
                    + (f" and FI's check of the best design at finer settings ({checked:,} runs)" if checked else "")
                    + f" would take "
                    f"{'at least ' if est.at_least else 'about '}{_estimate.plain_duration(est.seconds * _estimate.SAFETY)}"
                    f", more than the {_estimate.plain_duration(limit_s)} allowed (one new design takes about "
                    f"{'at least ' if lower else ''}{_estimate.plain_duration(per_eval or 0.0)}); FI does not run a smaller "
                    "study than the plan's")
        # The check's time comes out of the room for settings first: it is not a cost of each setting.
        fits_in = (self._what_fits(core_est, max(1.0, limit_s - check_s * _estimate.SAFETY), search=True)
                   if checked and core_est is not None else self._what_fits(est, limit_s, search=True))
        if fits_in is not None:
            # FI's own check of the best design at finer settings runs after the search and is not in its budget: the plan
            # is told what it costs, so the scan and the search are sized to the time left.
            try:
                fits_in["check"] = int(_optimisation_plan.budget(block).get("check") or 0)
            except (KeyError, TypeError, ValueError):
                pass
            if alternatives > 1:
                # Where the search's evaluations come from, in FI's own numbers: the plan lowers them by comparing fewer
                # alternatives or by giving each a smaller budget.
                first_level = _optimise_search.compared_levels(block)[0]
                fits_in["alternatives"] = alternatives
                fits_in["alternatives_searched"] = searching
                fits_in["alternative_conditions"] = [str(k) for k in first_level]
                fits_in["per_alternative"] = per_alternative
                fits_in["probes"] = probe_n
        # Not stopped while at least one design fits: the scan runs in the plan's order, then the search, then FI's check (the
        # time the check needs is kept for it), up to the time limit, and the paper says what ran and what did not.
        try:
            planned_search = int(_optimisation_plan.evaluations(block)[0] or 0)
            planned_check = int(_optimisation_plan.budget(block).get("check") or 0)
        except (KeyError, TypeError, ValueError):
            planned_search = planned_check = 0
        one_s = per_eval or first_s or _estimate.cheapest_setting_s(probes)
        timed = sized is not None and sized.evaluations is not None  # FI's size was worked out from what was measured
        if timed and (sized.deadline_s or 0.0) < 1.0 <= limit_s - min(check_s * _estimate.SAFETY, limit_s / 2.0):
            # The measured time leaves the search none (a tiny limit), where FI's usual cut would give it some: that stands.
            self._log.info("[execute] the time allowed is too short to keep the measured time for the check and the "  # type: ignore[attr-defined]
                           "analysis and still search; the search keeps FI's usual cut")
            timed = False
        # A resume or a repair sizes the study again from the plan FI already lowered: what it asked for stays in the record.
        asked, kept_to = planned_search, 0
        if (_whole_number(earlier.get("evaluations_lowered_to")) and _whole_number(earlier.get("evaluations"))
                and planned_search == int(earlier["evaluations_lowered_to"])):
            asked, kept_to = int(earlier["evaluations"]), int(earlier["evaluations_lowered_to"])
        if self._run_up_to_the_limit(
                runner, est, limit_s, one_s, kind="search",
                reserve_s=max(check_s * _estimate.SAFETY, sized.check_reserve_s) if timed else check_s * _estimate.SAFETY,
                planned={"scan": scan, "evaluations": asked, "check_runs": planned_check,
                         **({"evaluations_lowered_to": kept_to} if kept_to else {})},
                detail=text, fit=fits_in, deadline_s=sized.deadline_s if timed else None):
            # Only a study that goes on is made smaller in plan.md: one that stops as too long leaves the plan as it was.
            lowered = self._lower_the_evaluations(state, block, sized, planned_search, alternatives, limit_s) if timed else 0
            if lowered:
                record = _estimate.read_record(self.quest_root)  # type: ignore[attr-defined]
                if isinstance(record.get("truncated"), dict):
                    record["truncated"]["evaluations_lowered_to"] = int(lowered)
                    _estimate.write_record(self.quest_root, record)  # type: ignore[attr-defined]
            return None
        self._log.warning("[execute] the search's coarse scan: %s", text)  # type: ignore[attr-defined]
        return self._stop_too_long(text, fit=fits_in)

    async def _alternatives_that_differ(self, state: Any, block: dict[str, Any], runner: Any, python: Any, env: Any, *,
                                         deterministic: bool, at_once: int, thresholds: Any, left_s: float) -> int | None:
        """How many of the alternatives a study compares the run would search, by asking the simulation what the run asks
        (:func:`core.optimise_search.compare`): the same few designs (:func:`core.optimise_search.sizing_designs`) at every
        alternative, one of them once more at the first to see which results change by themselves, and the same rule for
        "gives the first one's results". ``None`` when it could not be asked in the time left or something did not answer
        (every alternative is then counted as searched)."""
        from time import monotonic as _clock  # the clock core/optimise.py judges its deadline by

        from . import optimise as _optimise

        if left_s < 15:
            return None
        levels = _optimise_search.compared_levels(block)
        try:
            designs = _optimise_search.sizing_designs(block)
            names = _optimise_search.compared_names(block, _optimise.watched_results(self._protocol_block(state) or {}, state.get("design")))  # type: ignore[attr-defined]
            cells = [_optimise_search.cell_of(_optimise_search.level_block(block, lv), d) for lv in levels for d in designs]
        except (KeyError, TypeError, ValueError):
            return None
        cells.append(dict(cells[0]))  # the first alternative's first design again
        try:
            got = await _optimise.evaluate_many(
                self.executor, python, self.quest_root, runner.simulate.relative_to(self.quest_root).as_posix(),  # type: ignore[attr-defined]
                "run_cell" if deterministic else "run_trial", cells, at_once=max(1, at_once), runs=1, base_seed=0,
                deadline=_clock() + left_s, env=env, thresholds=thresholds,
                in_each_run=_optimise_search.in_each_run(block) or None)
        except Exception as exc:  # noqa: BLE001 -- the count stays the plan's whole count
            self._log.warning("[execute] FI could not ask the simulation which alternatives give different results (%r); "  # type: ignore[attr-defined]
                              "every alternative is counted as searched", exc)
            return None
        values = [a.get("values") if isinstance(a, dict) and isinstance(a.get("values"), dict) else None
                  for a, _err, _loaded in got]
        n = len(designs)
        replies = [values[i * n:(i + 1) * n] for i in range(len(levels))]
        return _optimise_search.alternatives_searched(block, levels, replies, values[-1], names)

    async def _pilot_gate(self, state: Any, runner: Any, python: Any, env: Any, grid: dict[str, list[Any]],
                          probes: list[_estimate.Probe], thresholds: dict[str, Any], limit_s: float,
                          budget_s: float) -> dict[str, Any] | None:
        """Read what the timed trials returned (:func:`core.run_estimate.pilot`). Nothing wrong: one line in run.log and the
        record in ``.fi/pilot_check.json`` (which QUALITY.md shows). Something wrong: a failed run for the existing repair of the
        simulation, with the exact finding (at most ``engine.exec_reflect_max_iterations`` repairs, counted in the record
        BEFORE the run goes back); what the repairs leave ends in the honest stop. Nobody is asked to debug."""
        from . import degenerate_results as _degenerate
        from . import hidden_check as _hidden

        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        kinds = {str(m.get("id")): str(m.get("kind") or "") for m in protocol.get("metrics") or []
                 if isinstance(m, dict) and str(m.get("id") or "").strip()}
        total = sum(len(p.trials) for p in probes)
        if not total:
            return None  # nothing came back to read (a timed trial ran past its allowance)
        _grid, runs, _t = self._grid_and_runs(state)
        total_trials = _estimate.trial_count(grid, runs, runner.deterministic)
        allowance = max(60.0, 3.0 * limit_s / max(1, total_trials))
        record = _estimate.read_pilot_record(self.quest_root)  # type: ignore[attr-defined]
        repairs = int(record.get("repairs") or 0)
        # Once per version of the code: resizing the run does not change what the simulation does, so a reading already made
        # of this code (and these metrics) is reused, and its extra trials are not run again.
        code_key = _hidden.code_sha(self.quest_root) + "|" + ",".join(f"{k}:{v}" for k, v in sorted(kinds.items()))  # type: ignore[attr-defined]
        if code_key[0] != "|" and record.get("code_key") == code_key and record.get("outcome") == "passed":
            self._log.info("[execute] the pilot has read this version of the simulation already: nothing was wrong")  # type: ignore[attr-defined]
            return None
        if code_key[0] != "|" and record.get("code_key") == code_key and record.get("problem_items"):
            result = _estimate.Pilot(
                trials=int(record.get("trials") or 0), settings=int(record.get("settings") or 0),
                checked=[str(x) for x in record.get("metrics_read") or []],
                problems=[_estimate.PilotProblem(str(d.get("metric")), str(d.get("kind")), str(d.get("says")))
                          for d in record["problem_items"] if isinstance(d, dict)],
                unsure=[str(x) for x in record.get("not_ruled_out") or []])
        else:
            # The extra trials come out of the sizing's own budget (at most half of it in all, across the rounds).
            left = max(0.0, budget_s / 2.0 - self._pilot_spent)
            began = time.monotonic()
            result = await _estimate.pilot(
                self.executor, python, self.quest_root,  # type: ignore[attr-defined]
                runner.simulate.relative_to(self.quest_root).as_posix(),  # type: ignore[attr-defined]
                grid, probes, kinds, deterministic=runner.deterministic, budget_s=left,
                allowance_s=allowance, env=env, thresholds=thresholds, exempt=_degenerate.expected_constants(protocol, state.get("design")),
                in_each_run=_in_run.of_study(protocol))
            self._pilot_spent += time.monotonic() - began
        base = {"trials": result.trials, "settings": result.settings, "metrics_read": result.checked,
                "not_ruled_out": result.unsure, "at": time.time(), "trial_count_read": total, "code_key": code_key,
                "problem_items": [{"metric": p.metric, "kind": p.kind, "says": p.says} for p in result.problems]}
        if result.unsure:
            self._log.info("[execute] the pilot could not try every combination of the small settings, so it says nothing of "  # type: ignore[attr-defined]
                           "%s", ", ".join(f"`{n}`" for n in result.unsure))
        if not result.problems:
            self._log.info("[execute] the pilot read %d trial(s) at %d setting(s): %d metric(s) of the protocol checked, nothing "  # type: ignore[attr-defined]
                           "wrong%s", result.trials, result.settings, len(result.checked),
                           f" (after {repairs} repair(s))" if repairs else "")
            _estimate.write_pilot_record(self.quest_root, {**base, "outcome": "passed", "repairs": repairs, "problems": []})  # type: ignore[attr-defined]
            return None
        says = [p.says for p in result.problems]
        iters = int(state.get("exec_reflect_iter") or 0)
        for line in says:
            self._log.warning("[execute] the pilot found: %s", line)  # type: ignore[attr-defined]
        if iters < self.config.engine.exec_reflect_max_iterations:  # type: ignore[attr-defined]
            # Counted BEFORE the run goes back: a quest killed here never repairs more often than it may.
            _estimate.write_pilot_record(  # type: ignore[attr-defined]
                self.quest_root, {**base, "outcome": "repairing", "repairs": repairs + 1, "problems": says})
            print("[FI] a small trial run before the study shows the simulation does not measure what the protocol names; "
                  "sending it back to be repaired")
            return self._code_review_failed_run(  # type: ignore[attr-defined]
                _estimate.directive(result), script=_split_run.SIMULATE_NAME)
        _estimate.write_pilot_record(  # type: ignore[attr-defined]
            self.quest_root, {**base, "outcome": "stopped", "repairs": repairs, "problems": says})
        return self._stop_before_run(  # type: ignore[attr-defined]
            "a small trial run before the study (" + f"{result.trials} trial(s) at {result.settings} setting(s)) showed: "
            + "; ".join(says),
            "cannot_repair",
            tried=[f"ran {result.trials} trial(s) of the simulation at {result.settings} setting(s) before the full study "
                   "and read what they returned",
                   *[_one(s) for s in says[:3]],
                   f"sent the simulation back to be repaired {iters} time(s) with this named; it still showed it"])

    def _design_after_sizing(self, state: Any) -> dict[str, Any] | None:
        """The state's design with the runs per setting FI lowered, or ``None`` when none were."""
        sized = getattr(self, "_sized_protocol", None)
        if not sized:
            return None
        design = dict(state.get("design") or {})
        design["protocol"] = {**(design.get("protocol") if isinstance(design.get("protocol"), dict) else {}), **sized}
        return design

    def _lower_the_runs(self, state: Any, runs: int, fit: _estimate.Fit, est: _estimate.Estimate, limit_s: float,
                        grid: dict[str, list[Any]], protocol: dict[str, Any], *, topic: bool = False,
                        over_topic: str = "") -> dict[str, Any] | None:
        """FI's own smaller run: the same settings with ``fit.runs`` runs each, written into plan.md as a version of the engine
        (no request to the plan's model). The record of it (the plain note the paper states, the precision those runs give,
        and whether they are fewer than the plan's minimum), or ``None`` when plan.md could not be changed. ``topic``: the
        limit is the one the topic states; ``over_topic``: why that limit cannot be met even so (the run then has the plan's
        minimum and is over it)."""
        fewer = int(fit.runs)
        path = _plan.plan_path(self.quest_root)  # type: ignore[attr-defined]
        try:
            text = path.read_text(encoding="utf-8")
            edited = _plan.edit_design_block(
                text, lambda d: {**d, "protocol": {**(d.get("protocol") or {}), "runs_per_setting": int(fewer)}})
            if edited is None or _plan.parse(edited).design is None:
                return None
            edited = _plan.refresh_model_section(edited)
            _plan.note_edit(self.quest_root, text)  # type: ignore[attr-defined]
            path.write_text(edited, encoding="utf-8")
            _plan.record_version(self.quest_root, edited, by="engine",  # type: ignore[attr-defined]
                                 note="fewer runs per setting, so the experiment fits the time allowed, set by FI")
        except OSError as e:
            self._log.warning("[execute] FI could not lower the runs per setting in plan.md: %r", e)  # type: ignore[attr-defined]
            return None
        whose = "the topic's limit" if topic else "the time allowed"
        cells = _estimate.cell_count(grid)
        if over_topic:
            note = (f"{over_topic} FI lowered the runs per setting from {runs} to the plan's minimum of {fewer} and kept all "
                    f"{cells} setting(s); the experiment is expected to take about "
                    f"{_estimate.plain_duration(fit.seconds * _estimate.SAFETY)}, over the topic's limit; the checks, "
                    "thresholds and tolerances are unchanged.")
        else:
            note = (f"FI made the experiment smaller so that it fits {whose} ({_estimate.plain_duration(limit_s)}): it would "
                    f"have taken {self._padded(est)}. FI lowered the runs per setting from {runs} to {fewer} and kept all "
                    f"{cells} setting(s); the checks, thresholds and tolerances are unchanged.")
        if fit.below_plan:
            note += (f" That is fewer than the {fit.plan_floor} runs per setting the plan allows at the least ("
                     + ("its minimum and its precision target" if fit.precision_known else "its stated minimum")
                     + "): the topic's time limit asked for it, so the plan's minimum number of runs is a part of the study "
                     "not fully done.")
        elif not _estimate.floor_is_stated(protocol) and topic and not over_topic:
            note += (" The plan states no fewest number of runs per setting; for the topic's own limit FI goes down to "
                     f"{_estimate.fewest_for_topic(protocol, grid, runs)} at the least.")
        precision = _estimate.expected_precision(protocol, fewer, est.spreads) if fit.precision_known else None
        if precision is not None:
            note += (f" With {fewer} runs per setting, the 95% interval of `{precision['metric']}` is about "
                     f"±{precision['half_width']:g} ({precision['how']})"
                     + (f"; the plan's target is ±{precision['target']:g}" if precision.get("target") else "") + ".")
        elif not fit.precision_known:
            metric, _kind = _estimate.precision_metric(protocol)
            note += (f" The precision {fewer} runs per setting give for `{metric}` was not known before the run: the timed "
                     "trials did not show its spread, so the 95% interval is the one the run itself gives.")
        self._log.warning("[execute] %s", note)  # type: ignore[attr-defined]
        print(f"[FI] {note}")
        return {"note": note, "from": {"grid": grid, "runs_per_setting": runs},
                "to": {"grid": grid, "runs_per_setting": fewer}, "limit": "topic" if topic else "execution.timeout_s",
                "below_plan_minimum": bool(fit.below_plan), "plan_minimum": fit.plan_floor, "precision": precision,
                "precision_known": bool(fit.precision_known),
                # why the run is over the topic's limit: sized to the plan's minimum (no fewer runs meet it), or, set when the
                # cut is timed again, still over it after the cut (_size_the_run)
                "over_topic_limit": "at_plan_minimum" if over_topic else False}

    async def _time_finer_check(self, block: dict[str, Any], state: Any, runner: Any, python: Any, env: Any, *,
                                deterministic: bool, thresholds: Any, inside: Any, left_s: float,
                                limit_s: float) -> tuple[float | None, str, bool, float | None]:
        """Seconds one RUN of FI's check takes at its finest numerical settings, or ``None`` when the plan names no finer
        setting or it could not be timed in the time left; where the number comes from; whether it is only a lower bound (the
        timing hit its allowance); and the most memory (MB) the timed process held, when known. One design of the check is
        timed as the check runs it (the process start and the loading once, then its ``check_runs`` runs of the simulation:
        :func:`core.run_estimate._time_one` with ``runs=check_runs``) and divided by its ``check_runs``: the check's run count
        (``optimisation_plan.budget(block)["check"]``) is designs times ``check_runs``, so the number of runs times this is
        the designs times one whole design, and the start-up is counted once per design, not once per run. A failure here is
        not the run's: the check is then counted at the search design's cost, as a lower bound, and the log says so."""
        from . import optimise as _optimise
        from . import optimisation_plan as _optimisation_plan
        from . import optimum_check as _optimum_check

        none: tuple[float | None, str, bool, float | None] = (None, _cost_model.MEASURED, False, None)
        if left_s < 5:
            return none
        try:
            level = (_optimum_check.levels_of(block).get("levels") or [{}])[-1]
            if not level:
                return none
            check_runs = max(1, int(_optimisation_plan.budget(block).get("check_runs") or 1))
            cell = _optimise.complete_case(self._protocol_block(state), dict(level), keep_inside=True)  # type: ignore[attr-defined]
            probe, why = await _estimate._time_one(
                self.executor, python, self.quest_root, runner.simulate.relative_to(self.quest_root).as_posix(),  # type: ignore[attr-defined]
                cell, runs=check_runs, deterministic=deterministic, timeout_s=int(max(5, min(limit_s / 4.0, left_s))), env=env,
                thresholds=thresholds, in_each_run=inside or None, name="pilot_finer")
        except Exception as exc:  # noqa: BLE001 -- an optional timing must never stop a run
            self._log.info("[execute] FI could not time one run of its check at finer settings (%r)", exc)  # type: ignore[attr-defined]
            return none
        if probe is None:
            self._log.info("[execute] FI could not time one run of its check at finer settings (%s); the check is counted "  # type: ignore[attr-defined]
                           "at the cost of one search design, at least", str(why)[:200])
            return none
        # Only a design the timing ran whole is a measurement: one cut by its allowance, or stopped early and extrapolated,
        # is what the design costs at the least.
        whole = 1 if deterministic else check_runs
        low = bool(probe.at_least) or 0 < int(probe.timed_trials or 0) < whole  # (0: a probe made by hand)
        if low:
            self._log.info("[execute] one design of the check at finer settings was timed on %d of its %d run(s)%s: %s per "  # type: ignore[attr-defined]
                           "design is a lower bound, and the check is counted at least at that",
                           int(probe.timed_trials or 0), whole, " (the allowance for the timing ran out)" if probe.at_least else "",
                           _estimate.plain_duration(float(probe.seconds)))
        return (float(probe.seconds) / check_runs, "measured, includes the start of the process", low,
                getattr(probe, "peak_mb", None))

    def _size_the_search(self, block: dict[str, Any], limit_s: float, check_runs: int, scan_s: float, design: str,
                         cut: Any) -> Any:
        """How many evaluations the search may use, from FI's own cost model (:func:`core.cost_model.search_plan`) and not from
        the plan's count: what the time allowed leaves after the coarse scan, FI's check of the best design (its runs times
        one measured run at the finest settings, at most half the limit) and the analysis, over what one evaluation was
        measured to cost. ``None`` when the study was not cut. When nothing per evaluation was measured the model gives no
        number and the plan's count and FI's usual reserve stand; that is said in the log and on the console."""
        if not cut:
            return None
        from . import optimisation_plan as _optimisation_plan

        model = _cost_model.read(self.quest_root)
        if model is None:
            return None
        try:
            planned = int(_optimisation_plan.evaluations(block)[0] or 0) or None
        except (KeyError, TypeError, ValueError):
            planned = None
        plan = _cost_model.search_plan(model, limit_s, planned_evaluations=planned, check_runs=check_runs, scan_s=scan_s, design=design,
                                         margin=_estimate.SAFETY)
        if plan.evaluations is None:
            line = (f"FI could not work out how many evaluations the search can use ({plan.why or 'nothing was measured'}), "
                    "so the plan's number stands and the search is stopped at the time limit.")
            self._log.warning("[execute] %s", line)  # type: ignore[attr-defined]
            print(f"[FI] {line}")
        elif plan.why:
            self._log.info("[execute] the search's size: %s", plan.why)  # type: ignore[attr-defined]
        return plan

    def _lower_the_evaluations(self, state: Any, block: dict[str, Any], plan: Any, planned: int, alternatives: int,
                               limit_s: float) -> int:
        """The search's evaluations FI's cost model gives, written into plan.md as a version of the engine (only
        ``evaluation_budget``; no request to the plan's model), when that is fewer than the plan's. The evaluations
        now in plan.md when it was written; ``0`` when it was not (the plan's own count fits, or plan.md cannot be changed: a frozen
        protocol, a redesign, an unreadable plan): the search is then stopped by its deadline alone."""
        count = int(plan.evaluations or 0)
        if not planned or count <= 0 or count >= planned:
            return 0
        if _frozen.load(self.quest_root) is not None or int(state.get("iteration", 0) or 0) > 0:  # type: ignore[attr-defined]
            self._log.info("[execute] the protocol is frozen: the search keeps its %s evaluations and stops at the time "  # type: ignore[attr-defined]
                           "FI worked out for it", f"{planned:,}")
            return 0
        budget = block.get("evaluation_budget") if isinstance(block.get("evaluation_budget"), dict) else None
        if budget is None:
            return 0
        starts, per_start = int(budget["starts"]), int(budget["per_start"])
        factor = count / planned
        new_per = max(1, int(per_start * factor))
        new_starts = starts
        if per_start * factor < 1:
            new_starts = max(1, int(starts * per_start * factor))
        path = _plan.plan_path(self.quest_root)  # type: ignore[attr-defined]
        try:
            text = path.read_text(encoding="utf-8")

            def change(d: dict[str, Any]) -> dict[str, Any]:
                protocol = dict(d.get("protocol") or {})
                opt = dict(protocol.get("optimisation") or {})
                opt["evaluation_budget"] = {**(opt.get("evaluation_budget") or {}), "starts": new_starts, "per_start": new_per}
                return {**d, "protocol": {**protocol, "optimisation": opt}}

            edited = _plan.edit_design_block(text, change)
            if edited is None or _plan.parse(edited).design is None:
                return 0
            edited = _plan.refresh_model_section(edited)
            _plan.note_edit(self.quest_root, text)  # type: ignore[attr-defined]
            path.write_text(edited, encoding="utf-8")
            _plan.record_version(self.quest_root, edited, by="engine",  # type: ignore[attr-defined]
                                 note="fewer search evaluations, so the search, its check and the analysis fit the time allowed, set by FI")
        except OSError as e:
            self._log.warning("[execute] FI could not lower the search's evaluations in plan.md: %r", e)  # type: ignore[attr-defined]
            return 0
        new_total = new_starts * new_per * max(1, alternatives)
        line = (f"FI made the search smaller so that it fits the time allowed ({_estimate.plain_duration(limit_s)}): the plan "
                f"asked for {planned:,} evaluations; one takes {'under a second' if (plan.wall_per_evaluation_s or 0.0) < 1.0 else 'about ' + _estimate.plain_duration(plan.wall_per_evaluation_s or 0.0)}, "
                f"and the coarse scan, FI's check of the best design and the analysis need the rest, so the search is given "
                f"{new_total:,} ({new_starts:,} starting points of {new_per:,}). The checks, thresholds and tolerances are unchanged.")
        self._log.warning("[execute] %s", line)  # type: ignore[attr-defined]
        print(f"[FI] {line}")
        sized_block = {**block, "evaluation_budget": {**budget, "starts": new_starts, "per_start": new_per}}
        self._sized_protocol = {**(getattr(self, "_sized_protocol", None) or {}), "optimisation": sized_block}
        _cost_model.keep_measured(self.quest_root, plan={"evaluations_planned": planned, "evaluations_set": new_total})  # type: ignore[attr-defined]
        return new_total

    def _size_edit_after_read(self) -> bool:
        """Whether FI made the search smaller in plan.md from its cost model (the grid's lowered runs keep their own label):
        a person who read the plan before then did not read the protocol that runs (the record of who approved it says so)."""
        try:
            after = False  # an FI edit of the search's size counts only if no person approved the plan later
            for row in _plan.history(self.quest_root):  # type: ignore[attr-defined]
                if str(row.get("by")) == "user":
                    after = False
                elif str(row.get("by")) == "engine" and str(row.get("note") or "").startswith("fewer search evaluations"):
                    after = True
            return after
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _too_long_text(est: _estimate.Estimate, limit_s: float, cut: bool = True) -> str:
        at_once = f" with {est.at_once} running at the same time" if est.at_once > 1 else ""
        return (f"the experiment's {est.cells:,} settings would take {'at least ' if est.at_least else 'about '}"
                f"{_estimate.plain_duration(est.seconds * _estimate.SAFETY)}{at_once}, more than the "
                f"{_estimate.plain_duration(limit_s)} allowed; FI does not run a smaller study than the plan's settings"
                + (", and they cannot fit even with the fewest runs per setting the plan allows" if cut else ""))

    def _sizing_failed(self, why: str, text: str | None = None) -> dict[str, Any]:
        """A failed run for the existing repair of the simulation: it failed while the experiment was timed, at the setting
        named in ``why``."""
        text = text or f"FI ran a trial of the simulation to time the experiment before the study, and it failed {why}"
        return {
            "exec_result": {
                "returncode": 1, "duration_s": 0.0, "timed_out": False, "stdout_tail": "", "stderr_tail": text[-2000:],
                "packages_note": getattr(self, "_packages_note", ""), "failed_script": _split_run.SIMULATE_NAME,
                "flat_output": "", "numeric_warnings": [],
            },
            "figures": [], "figure_records": {}, "result_json": {}, "exec_patch_pending": False,
            "result_json_replicates": [], "result_json_deterministic": False, "result_json_trials": False,
            "result_json_replicate_seed_ignored": False, "result_json_no_random_source": False,
        }

    async def _small_run(self, state: Any, py: Any, exec_env: Any, *, baseline: bool = True) -> Any:
        """FI's small run of the simulation on the code as it is (core/small_run.py), or ``None`` when the simulation does
        not follow the trial contract (``run_trial`` / ``run_cell`` in simulate.py). A run of the same code and settings is
        not made again: its record (``.fi/small_run.json``) is read back."""
        from . import small_run as _small

        simulate = self.quest_root / "code" / _split_run.SIMULATE_NAME  # type: ignore[attr-defined]
        if not getattr(self, "_trial_mode", False) or not simulate.is_file():
            return None
        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        from . import optimisation_plan as _optim
        from . import optimise as _optimise

        # A search for the best design runs on its coarse scan's settings, with the search's baseline design and every
        # value it compares inside each run, as the search calls the simulation: a fault there would otherwise be met
        # first by the known-answer checks, on their own few repairs.
        block = _optimise.block_of(protocol)
        search = block is not None or _optim.has_block(state.get("design"))
        if search and block is None:
            return None
        self._small_run_ctx = (state, py, exec_env)
        if search:
            try:
                grid = _optimise_search.scan_grid(block)
            except (KeyError, TypeError, ValueError):
                return None  # a block the scan cannot be built from is reported by the run
        else:
            grid = _optimise_search.study_grid(protocol) if isinstance(protocol.get("grid"), dict) else {}  # knows its sub-studies
        thresholds = protocol.get("thresholds") if isinstance(protocol.get("thresholds"), dict) else None
        deterministic = "run_trial" not in set(getattr(self, "_trial_entries", set()) or ())
        inside = (_optimise_search.in_each_run(block) or None if search
                  else protocol.get("in_each_run") if isinstance(protocol.get("in_each_run"), dict) else None)
        key = _small.key_of(simulate, grid, deterministic, thresholds, inside,
                            _optimise_search.cell_of(block, dict((block.get("baseline") or {}).get("values") or {}))
                            if search and isinstance(block.get("baseline"), dict) else None)
        record = _small.read_record(self.quest_root)  # type: ignore[attr-defined]
        module = simulate.relative_to(self.quest_root).as_posix()  # type: ignore[attr-defined]
        if record.get("key") == key:
            self._small_run_note = str(record.get("for_reader") or "")
            if search and baseline and not record.get("crashed") and "baseline" not in record:
                found = await _small.baseline_answer(
                    self.executor, py, self.quest_root, module, block,  # type: ignore[attr-defined]
                    timeout_s=_small.budget(self.config.execution.timeout_s), env=dict(exec_env or {}),  # type: ignore[attr-defined]
                    thresholds=thresholds)
                _small.update_record(self.quest_root, baseline=found)  # type: ignore[attr-defined]
                record = _small.read_record(self.quest_root)  # type: ignore[attr-defined]
            return record
        paired = any(isinstance(m, dict) and m.get("paired") for m in protocol.get("metrics") or [])
        got = await _small.run(
            self.executor, py, self.quest_root, module, grid,  # type: ignore[attr-defined]
            deterministic=deterministic, timeout_s=_small.budget(self.config.execution.timeout_s),  # type: ignore[attr-defined]
            env=dict(exec_env or {}), thresholds=thresholds, paired=paired, in_each_run=inside)
        if search and baseline and not got.crashed:
            # the search's own evaluation of its baseline design, judged as the search judges it (_small_run_checks)
            got.baseline = await _small.baseline_answer(
                self.executor, py, self.quest_root, module, block,  # type: ignore[attr-defined]
                timeout_s=_small.budget(self.config.execution.timeout_s), env=dict(exec_env or {}),  # type: ignore[attr-defined]
                thresholds=thresholds)
        _small.write_record(self.quest_root, key, got)  # type: ignore[attr-defined]
        self._small_run_note = got.for_reader()
        self._log.info("[execute] FI's small run before the code is read: %s", got.says())  # type: ignore[attr-defined]
        return _small.read_record(self.quest_root)  # type: ignore[attr-defined]

    async def _small_run_gate(self, state: Any, py: Any, exec_env: Any) -> dict[str, Any] | None:
        """Before the code is read: FI runs the simulation on a small case (core/small_run.py). A crash (it cannot be loaded,
        or most trials of a setting fail) goes to the repair of a failed run with the place and the traceback, while a
        repair is left; then the code is read with what the run gave. ``None`` to go on."""
        from . import small_run as _small

        record = await self._small_run(state, py, exec_env)
        if record and not record.get("crashed"):
            checked = await self._small_run_checks(state, record)
            if checked is not None:
                return checked
            return self._unread_in_small_run(state, record, self.quest_root / "code" / _split_run.SIMULATE_NAME)  # type: ignore[attr-defined]
        if not record or not record.get("crashed"):
            return None
        if int(state.get("exec_reflect_iter") or 0) >= self.config.engine.exec_reflect_max_iterations:  # type: ignore[attr-defined]
            self._log.warning("[execute] FI's small run still fails (%s) and no repair is left; the study's own run "  # type: ignore[attr-defined]
                              "and its checks decide", record.get("says"))
            return None
        settings = [_small.Setting(str(s.get("key") or ""), int(s.get("planned") or 0), int(s.get("ok") or 0),
                                   int(s.get("failed") or 0), bool(s.get("timed_out")),
                                   [(str(r), int(n)) for r, n in s.get("reasons") or []])
                    for s in record.get("settings") or [] if isinstance(s, dict)]
        tb = ""
        try:
            for path in sorted((self.quest_root / _small.DIR).glob("s*/cell*.out.jsonl")):  # type: ignore[attr-defined]
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line) if line.strip().startswith("{") else {}
                    if isinstance(row, dict) and row.get("traceback"):
                        tb = str(row["traceback"])
                        break
                if tb:
                    break
        except (OSError, ValueError):
            tb = ""
        for s in settings:
            if s.fault and not s.traceback:
                s.traceback = tb
                break
        message = _small.SmallRun(settings, str(record.get("load_error") or "")).directive(self._level_rule(state, self.quest_root / "code" / _split_run.SIMULATE_NAME))  # type: ignore[attr-defined]
        self._log.warning("[execute] FI's small run of the simulation failed before the code is read (%s); sending it "  # type: ignore[attr-defined]
                          "to be repaired", record.get("says"))
        return self._code_review_failed_run(message, script=_split_run.SIMULATE_NAME)  # type: ignore[attr-defined]

    async def _small_run_checks(self, state: Any, record: dict[str, Any]) -> dict[str, Any] | None:
        """The small run against the frozen plan (core/small_run.py): the shape of each number the plan declares one per
        run, for a search the search's own judgement of its baseline design (objective and each constraint), a result the
        plan declares and no trial returned, and FI's analysis on the small run's trials (whatever ``engine.pipeline_gate``
        says). An exact fault is sent back to the code once per quest (its kind and quantity), on a counter of its own in
        ``.fi/small_run.json`` capped by ``engine.exec_reflect_max_iterations``; it never spends a crash repair, and a spent
        crash counter never blocks it. After that, and for everything a small sample can explain, a line in plan.md and
        run.log and a fact for the code's reader. Nothing here stops the quest. ``None`` to go on."""
        from . import optimise as _optimise
        from . import small_run as _small

        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        if not isinstance(protocol, dict):
            return None
        block = _optimise.block_of(protocol)
        faults = list(_small.judge_baseline(block, record)) if block is not None else []
        named = {f.quantity for f in faults if f.send}
        faults += [f for f in _small.shape_faults(protocol, record) if f.quantity not in named]
        named |= {f.quantity for f in faults if f.send}
        faults += [f for f in _small.missing_outputs(protocol, record) if f.quantity not in named]
        faults += await self._small_run_analysis(state, protocol, record, search=block is not None)
        if not faults:
            return None
        ledger = _small.sent(_small.read_record(self.quest_root))  # type: ignore[attr-defined]
        cap = int(self.config.engine.exec_reflect_max_iterations)  # type: ignore[attr-defined]
        new = [f for f in faults if f.send and f.key not in ledger["faults"]]
        # the simulation first: an analysis that cannot read what the simulation does not return is not the analysis' fault
        to_send = [f for f in new if f.script != "derive.py"] or [f for f in new if f.script == "derive.py"][:1]
        if to_send and ledger["count"] < cap:
            _small.update_record(self.quest_root, **{_small.SENT: {  # type: ignore[attr-defined]
                "faults": ledger["faults"] + [f.key for f in to_send], "count": ledger["count"] + 1}})
            for f in faults:
                if f not in to_send and not (f.send and f.key not in ledger["faults"]):
                    self._small_run_fact(f, after="sent" if f.send else "")
            script = to_send[0].script
            self._log.warning("[execute] %s Sending %s back to be repaired (%d of at most %d send-back(s) of the small "  # type: ignore[attr-defined]
                              "run).", " ".join(f.why for f in to_send), script, ledger["count"] + 1, cap)
            print(f"[FI] the small run gives something other than the plan declares; sending {script} back to be repaired")
            message = ("FI ran the simulation on a small case before the study and compared what it gave with the frozen "
                       "plan:\n" + "\n".join(f"- {f.why}" for f in to_send)
                       + "\n\nRepair the code so that it gives what the plan declares, in the form the plan declares. Do "
                       "not change the plan's quantities or the values it gives each run. Keep everything else as it is.")
            failed = self._code_review_failed_run(message, script=script)  # type: ignore[attr-defined]
            if script == "derive.py":
                failed["exec_result"]["failed_body"] = to_send[0].body
            # its own counter (in .fi/small_run.json): ``execute_reflect`` spends no crash repair on it and needs none left
            failed["exec_result"]["gate_repair"] = True
            return failed
        for f in faults:
            self._small_run_fact(f, after="" if not f.send else "sent" if f.key in ledger["faults"] else "spent")
        return None

    def _small_run_fact(self, fault: Any, *, after: str = "") -> None:
        """A fault of the small run that is not sent back (now): one line in plan.md and run.log, and a fact for the code's
        reader. ``after``: ``"sent"`` for an exact fault sent back once already, ``"spent"`` when no send-back is left."""
        line = fault.why + {"sent": " It was sent back to be repaired once and is still there; the study runs with it.",
                            "spent": " No send-back of the small run is left (engine.exec_reflect_max_iterations); the "
                                     "study runs with it."}.get(after, "")
        note = getattr(self, "_small_run_note", "") or ""
        if fault.why not in note:
            self._small_run_note = (note + "\n" if note else "") + line
        self._note_unread_setting(line, why="FI's small run before the code is read")

    async def _small_run_analysis(self, state: Any, protocol: dict[str, Any], record: dict[str, Any], *,
                                  search: bool) -> list[Any]:
        """FI's analysis (derive.py) run on the small run's trials (core/pipeline_gate.py ``analysis_gate`` on
        ``.fi/small_run/raw``), whatever ``engine.pipeline_gate`` says, and split by evidence (core/small_run.py
        ``analysis_split``, ``no_value``): the :class:`core.small_run.Fault` s it gives. Only when FI writes the analysis
        and the small run recorded its trials. The result is kept in the small run's record for the same code."""
        import shutil

        from . import analysis_frame as _analysis
        from . import pipeline_gate as _gate
        from . import small_run as _small

        raw = self.quest_root / _small.TRIALS_PATH  # type: ignore[attr-defined]
        code = self.quest_root / "code"  # type: ignore[attr-defined]
        if state.get("no_simulation_resolved") or not self._analysis_on() or not raw.is_file() \
                or not (code / "derive.py").is_file():
            return []
        docker = self.config.execution.sandbox == "docker"  # type: ignore[attr-defined]
        h = hashlib.sha256(str(record.get("key") or "").encode())
        for path in sorted(code.rglob("*.py")):
            if not ({".git", "__pycache__", "tests"} & set(path.relative_to(code).parts)):
                h.update(path.relative_to(code).as_posix().encode())
                h.update(path.read_bytes().replace(b"\r\n", b"\n"))
        key = h.hexdigest()[:20]
        kept = record.get("analysis") if isinstance(record.get("analysis"), dict) else {}
        if kept.get("key") == key:
            got = _gate.GateResult(**{k: v for k, v in kept.items() if k in _gate.GateResult.__dataclass_fields__})
            missing = [m for m in kept.get("no_value") or [] if isinstance(m, dict)]
        else:
            work = self.quest_root / _small.DIR / "analysis"  # type: ignore[attr-defined]
            shutil.rmtree(work, ignore_errors=True)
            try:
                got = await _gate.analysis_gate(self.quest_root, code, protocol, search=search,  # type: ignore[attr-defined]
                                                sandbox=self.config.execution.sandbox,  # type: ignore[attr-defined]
                                                timeout_s=_small.budget(self.config.execution.timeout_s),  # type: ignore[attr-defined]
                                                raw_dir=raw.parent, work_dir=work)
                # the declared results FI's own record of that run holds no value of (core/analysis_frame.py)
                missing = [m for m in _analysis.not_computed(_analysis.read_record(work)) if m["status"] == "missing"]
            except Exception as exc:  # noqa: BLE001 -- a check that could not run is not a fault of the code
                self._log.warning("[execute] FI's analysis could not be run on the small run's trials: %s", exc)  # type: ignore[attr-defined]
                return []
            finally:
                shutil.rmtree(work, ignore_errors=True)
            _small.update_record(self.quest_root, analysis={"key": key, **got.record(), "no_value": missing})  # type: ignore[attr-defined]
            self._log.info("[execute] FI's analysis on the small run's trials: %s", got.says())  # type: ignore[attr-defined]
        split = _small.analysis_split(got, docker=docker)
        return [*([split] if split is not None else []), *_small.no_value(missing)]

    # --- the analysis gate (core/pipeline_gate.py; ``engine.pipeline_gate``, off by default) -------------------------------
    # The baseline of a change is the gate's result for the code that is KEPT (``_gate_kept``: the key of that code and its
    # result), never for a change that is being judged: it is set where a change is kept (``_gate_commit``) and by
    # ``_gate_first``, and re-derived from the code on disk (``_gate_baseline``) where a repair loop starts and the key of
    # the code on disk is not the kept one (new code, a person's edit, a resume). Nothing is checkpointed.

    def _gate_key(self, code: Path, raw: Path, was: dict[str, str] | None = None, absent: tuple[str, ...] = ()) -> str:
        """What the gate's outcome depends on: the study's inputs and simulate.py (as :func:`trial_runner.code_inputs_sha`),
        every other file of the code folder (the analysis the gate runs) and the recorded raw it runs on. With ``was`` (file
        name -> text) and ``absent`` the key of the code as it was before a change: those files put back, those removed.
        Line endings do not count (a file written on Windows and the same text read back are one code)."""
        files: dict[str, bytes] = {}
        for path in code.rglob("*"):
            if path.is_file() and not ({".git", "__pycache__"} & set(path.relative_to(code).parts)):
                files[path.relative_to(code).as_posix()] = path.read_bytes().replace(b"\r\n", b"\n")
        for rel, text in (was or {}).items():
            files[rel] = text.encode("utf-8").replace(b"\r\n", b"\n")
        for rel in absent:
            files.pop(rel, None)
        h = hashlib.sha256()
        h.update(_trial_runner.code_inputs_sha(  # type: ignore[attr-defined]
            self.quest_root, files.get(_split_run.SIMULATE_NAME, b"").decode("utf-8", errors="replace")).encode())
        for rel in sorted(files):
            h.update(rel.encode())
            h.update(files[rel])
        stat = raw.stat()
        h.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        return h.hexdigest()

    def _gate_applies(self, state: Any) -> bool:
        """The switch is on, the quest simulates, and a raw has been recorded for the gate to run on."""
        if not self.config.engine.pipeline_gate:  # type: ignore[attr-defined]
            return False
        if state.get("no_simulation_resolved"):
            return False
        return (self.quest_root / "raw" / "trials.json").is_file()  # type: ignore[attr-defined]

    async def _analysis_gate(self, state: Any, code_dir: Path | str | None = None) -> Any:
        """The analysis gate's result for the code as it is now (``None`` when the switch is off, the quest is a
        no-simulation quest, no raw has been recorded yet, or the gate could not run: it never turns into a fault). The
        result is kept for the same code and raw, written to ``.fi/pipeline_gate.json`` (with the runs and seconds spent
        in the quest so far) and logged in one line with its seconds."""
        if not self._gate_applies(state):
            return None
        raw = self.quest_root / "raw" / "trials.json"  # type: ignore[attr-defined]
        code = Path(code_dir) if code_dir else self.quest_root / "code"  # type: ignore[attr-defined]
        key = ""
        try:
            from . import pipeline_gate as _gate
            from .small_run import budget as _budget

            key = self._gate_key(code, raw)
            cache = self.__dict__.setdefault("_gate_cache", {})
            if key in cache:
                if cache[key] is not None:  # None: the gate could not run on this code, said once
                    self._gate_seen = (key, cache[key])
                return cache[key]
            protocol = self._protocol_block(state)  # type: ignore[attr-defined]
            began = time.monotonic()
            got = await _gate.analysis_gate(
                self.quest_root, code, protocol if isinstance(protocol, dict) else None,  # type: ignore[attr-defined]
                sandbox=self.config.execution.sandbox,  # type: ignore[attr-defined]
                timeout_s=_budget(self.config.execution.timeout_s))  # type: ignore[attr-defined]
            spent = round(time.monotonic() - began, 1)
        except Exception as exc:  # noqa: BLE001 -- a gate that could not run is not a fault
            self._log.warning("[gate] the analysis could not be checked: %s", exc)  # type: ignore[attr-defined]
            if key:  # said once, and not run again for this code and raw
                self.__dict__.setdefault("_gate_cache", {})[key] = None
                try:
                    (self.fi_dir / "pipeline_gate.json").write_text(  # type: ignore[attr-defined]
                        json.dumps({"error": str(exc)[:300]}, indent=2), encoding="utf-8")
                except OSError:
                    pass
            return None
        cache[key] = got
        self._gate_seen = (key, got)
        path = self.fi_dir / "pipeline_gate.json"  # type: ignore[attr-defined]
        try:
            if "_gate_totals" not in self.__dict__:
                before = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
                self._gate_totals = {"runs": int(before.get("runs") or 0), "seconds": float(before.get("seconds_total") or 0.0)}
            self._gate_totals["runs"] += 1
            self._gate_totals["seconds"] = round(self._gate_totals["seconds"] + spent, 1)
            path.write_text(json.dumps({**got.record(), "runs": self._gate_totals["runs"],
                                        "seconds_total": self._gate_totals["seconds"]}, indent=2), encoding="utf-8")
        except (OSError, ValueError):
            pass
        self._log.info("[gate] %s (%.1f s; %d run(s), %.1f s in this quest)", got.says(), spent,  # type: ignore[attr-defined]
                       self.__dict__.get("_gate_totals", {}).get("runs", 1), self.__dict__.get("_gate_totals", {}).get("seconds", spent))
        return got

    async def _gate_baseline(self, state: Any) -> None:
        """Where a repair loop starts (the code on disk is the kept code): the baseline is the gate's result for the code
        on disk, taken from the cache when it was seen, else from one gate run, whenever the kept key is not the disk's."""
        if not self._gate_applies(state):
            return
        try:
            key = self._gate_key(self.quest_root / "code", self.quest_root / "raw" / "trials.json")  # type: ignore[attr-defined]
        except OSError:
            return
        kept = self.__dict__.get("_gate_kept")
        if kept and kept[0] == key:
            return
        got = await self._analysis_gate(state)
        self._gate_kept = (key, got) if got is not None else None

    async def _gate_baseline_before(self, state: Any, path: Path, old_code: str, package_before: dict[str, str]) -> None:
        """For a caller that has written a change to ``path`` (and maybe the model's package) and not called
        :meth:`_gate_baseline` first: the baseline is the gate's result for the code as it was just before the write
        (``old_code`` and ``package_before`` put back), taken from the kept result when its key is that code's, else from the
        cache or one gate run of that code in a scratch copy; never a result kept for other code."""
        if not self._gate_applies(state):
            return
        code, raw = self.quest_root / "code", self.quest_root / "raw" / "trials.json"  # type: ignore[attr-defined]
        try:
            rel = path.relative_to(code).as_posix()
            from . import code_layout as _layout

            was = {rel: old_code, **package_before}
            absent = tuple(r for r in _layout.package_sources(code) if package_before and r not in package_before)
            key = self._gate_key(code, raw, was, absent)
        except (OSError, ValueError):
            return
        kept = self.__dict__.get("_gate_kept")
        if kept and kept[0] == key:
            return
        cache = self.__dict__.setdefault("_gate_cache", {})
        if key in cache:
            self._gate_kept = (key, cache[key]) if cache[key] is not None else None
            return
        import shutil
        import tempfile

        scratch = Path(tempfile.mkdtemp(prefix="gate_before_"))
        try:
            shutil.copytree(code, scratch / "code", ignore=shutil.ignore_patterns(".git", "__pycache__"))
            for name, text in was.items():
                (scratch / "code" / name).parent.mkdir(parents=True, exist_ok=True)
                (scratch / "code" / name).write_text(text, encoding="utf-8")
            for name in absent:
                (scratch / "code" / name).unlink(missing_ok=True)
            got = await self._analysis_gate(state, code_dir=scratch / "code")
        except OSError:
            got = None
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        self._gate_kept = (key, got) if got is not None else None

    def _gate_commit(self, state: Any) -> None:
        """A change was kept: the code on disk is the new baseline when the gate saw exactly this code (otherwise the next
        loop derives it), and what stands about it is recorded."""
        if not self._gate_applies(state):
            return
        seen = self.__dict__.get("_gate_seen")
        try:
            key = self._gate_key(self.quest_root / "code", self.quest_root / "raw" / "trials.json")  # type: ignore[attr-defined]
        except OSError:
            return
        if seen and seen[0] == key:
            self._gate_kept = seen
            self._gate_standing(seen[1])

    def _gate_standing(self, result: Any) -> None:
        """What the gate found in the code that stays: ok clears the record, anything else (a fault no repair is left for,
        too few trials to tell, the time, a Docker quest) goes to the problems FI could not repair as FI's reading, so
        QUALITY.md and the paper's limitations say it and the quest goes on (core/stop_power.py). A run of the code on the small
        copy that raised an error or drew nothing is FI's own observation (``COPY``); only a gate that could not check is a
        reading."""
        self._clear_cannot_repair("gate")  # type: ignore[attr-defined]
        if result.ok:
            return
        from . import pipeline_gate as _gate

        self._log.warning("[gate] %s", result.says())  # type: ignore[attr-defined]
        self._record_cannot_repair(  # type: ignore[attr-defined]
            result.says(), ["ran the analysis from start to end on a small copy of the recorded results"],
            source=_stop_power.COPY if result.kind in (_gate.FAULT, _gate.SMALL_DATA) else _stop_power.READ, check="gate")

    async def _gate_refusal(self, state: Any) -> str:
        """After a change to the code was written: why it cannot stay (the analysis, which ran before, no longer runs), or
        ``""``. Nothing at all happens with the switch off. The change is compared with the kept code's result."""
        if not self.config.engine.pipeline_gate:  # type: ignore[attr-defined]
            return ""
        from . import pipeline_gate as _gate

        after = await self._analysis_gate(state)
        if after is None:
            return ""
        kept = self.__dict__.get("_gate_kept")
        verdict = _gate.gate_verdict(kept[1] if kept else None, after)
        if verdict.reject:
            return "the analysis no longer runs from start to end on the recorded trials: " + verdict.why
        return ""

    async def _finding_gate(self, state: Any) -> str:
        """The gate for a repair of a reader's finding (``finding_repair.Env.gate``), after the small run."""
        return await self._gate_refusal(state)

    async def _gate_first(self, state: Any) -> dict[str, Any] | None:
        """Before the long run, once the small run has passed: the analysis is run on the recorded results and its result
        is the baseline. A fault goes to the repair of a failed run (while one of the gate's own repairs is left, :data:`GATE_REPAIRS`) with the gate's sentence; anything
        that stands is recorded. ``None`` to go on."""
        after = await self._analysis_gate(state)
        if after is None:
            return None
        self._gate_kept = self.__dict__.get("_gate_seen")
        made = int(state.get("gate_repairs") or 0)
        # ``engine.exec_reflect_max_iterations: 0`` (no repair at all) still allows none
        if after.kind == "fault" and made < min(GATE_REPAIRS, int(self.config.engine.exec_reflect_max_iterations)):  # type: ignore[attr-defined]
            self._log.warning("[gate] the analysis does not run on the recorded results; sending it to be repaired")  # type: ignore[attr-defined]
            if self._analysis_on():
                # FI's analysis: the function of derive.py its record names goes to the repair, never FI's experiment.py
                failed = self._code_review_failed_run(after.says(), script="derive.py")  # type: ignore[attr-defined]
                failed["exec_result"]["failed_body"] = getattr(after, "body", "")
            else:
                failed = self._code_review_failed_run(after.says(), script="experiment.py")  # type: ignore[attr-defined]
            # a repair of the analysis has its own counter and spends no crash repair (``execute_reflect`` reads the mark)
            failed["exec_result"]["gate_repair"] = True
            failed["gate_repairs"] = made + 1
            return failed
        self._gate_standing(after)
        return None

    async def _small_run_again(self) -> str:
        """For a repair of a reader's finding: ``""`` when FI's small run still runs on the code as it is now, else why not.
        ``""`` when there is no small run to make."""
        ctx = getattr(self, "_small_run_ctx", None)
        if not ctx:
            return ""
        record = await self._small_run(*ctx, baseline=False)
        if record and record.get("crashed"):
            return str(record.get("says") or "it failed")
        # The repaired code must also still give the study's results from the function FI calls for the study.
        simulate = self.quest_root / "code" / _split_run.SIMULATE_NAME  # type: ignore[attr-defined]
        try:
            fault = await self._simulation_entry_fault(ctx[0], simulate.read_text(encoding="utf-8")) if simulate.is_file() else ""  # type: ignore[attr-defined]
        except OSError:
            return ""
        return fault  # the analysis gate follows as ``Env.gate`` (one call for each change)

    async def _ask_finding_test_again(self, issue: dict[str, Any], error: Any, record: dict[str, Any],
                                      state: Any = None) -> Any:
        """At the evidence gate, a shown issue's test errors in its own code: the code model is asked once more for it,
        shown the error, as an assertion about values (:func:`core.finding_repair.ask_again`). Its ``TestRun``, or ``None``
        (asked before, no test came back, or the model could not be asked: an error here never stops anything)."""
        from . import code_review as _code_review
        from . import finding_repair as _finding_repair
        from . import review_pairs as _pairs

        if not isinstance(issue, dict) or issue.get("asked_again"):
            return None
        protocol = self._protocol_block(state) if state is not None else None  # type: ignore[attr-defined]
        protocol = protocol if isinstance(protocol, dict) else None
        values = _pairs.plan_values(protocol)

        async def ask(prompt: str, who: str, kind: str) -> str:
            try:
                return await self._chat(prompt, node="implement.finding_test")  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - a request that cannot be made shows nothing and stops nothing
                self._log.warning("[evidence_gate] the code model could not be asked again for the test of %s: %s",  # type: ignore[attr-defined]
                                  issue.get("id"), _one(str(exc)))
                return ""

        async def run_tests(paths: list[str]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for path in paths:
                if (run := await self._rerun_finding_test(path)) is not None:  # type: ignore[attr-defined]
                    out[path] = run
            return out

        async def snapshot() -> dict[str, str]:
            return {}

        code = self.quest_root / "code"  # type: ignore[attr-defined]
        env = _finding_repair.Env(
            root=self.quest_root, code=code, record=record,  # type: ignore[attr-defined]
            save=lambda: _code_review.save(self.quest_root, record),  # type: ignore[attr-defined]
            ask=ask, run_tests=run_tests, snapshot=snapshot, static=dict, keeps=lambda *a: "",
            prompts=self._prompts,  # type: ignore[attr-defined]
            topic=_pairs.scrub(self._topic_block(), values) or "(not stated)",  # type: ignore[attr-defined]
            equations=_code_review.equations_view(protocol, values), scrub=lambda text: _pairs.scrub(text, values),
            log=self._log, package=(self._package_in_use(state) or "") if state is not None else "")  # type: ignore[attr-defined]
        again = await _finding_repair.ask_again(env, issue, _finding_repair.target_file(issue, code), error)
        if again is not None:
            self._log.warning("[evidence_gate] the test of %s errored in its own code; FI asked once more for a test that "  # type: ignore[attr-defined]
                              "asserts on the values, and it %s", issue.get("id"),
                              {"fail": "fails on the code that ran", "pass": "passes on the code that ran"}.get(
                                  again.status, "errors too: it shows nothing"))
        return again

    def _trial_checks_stop(self, state: Any) -> dict[str, Any] | None:
        """When no repair of a fault FI's checks of the trials found is coming (they are spent, or the repair gave up): failed
        trials in some settings only run the study once more with the repairs counted as spent, so those settings are left
        out (FI's trials are reused, the simulation being unchanged); any other fault stops the quest with its plain record
        (``cannot_repair``, no paper). ``None`` when the run was not sent back by those checks (core/trial_checks.py)."""
        result = state.get("exec_result") or {}
        fault = result.get("trial_checks") if isinstance(result, dict) else None
        if not isinstance(fault, dict) or not fault.get("say") or result.get("returncode", 0) == 0:
            return None
        if fault.get("partial"):
            self._log.warning(  # type: ignore[attr-defined]
                "[execute_reflect] no repair of the settings where most trials failed is coming: the study is read once "
                "more and those settings are left out")
            return {"exec_patch_pending": True, "exec_give_up_reason": "",
                    "exec_reflect_iter": max(int(state.get("exec_reflect_iter") or 0),
                                             int(self.config.engine.exec_reflect_max_iterations))}  # type: ignore[attr-defined]
        repairs = _numeric.repairs_made(state)
        return self._stop_before_run(
            str(fault["say"]), "cannot_repair",
            tried=[*[str(t) for t in fault.get("tried") or []],
                   f"sent the simulation back to be repaired with these numbers ({repairs} repair"
                   f"{'s' if repairs != 1 else ''}); the fault stood"])

    async def _group_settings(self, state: Any, runner: Any, python: Any, env: Any, grid: dict[str, list[Any]], runs: int,
                              thresholds: dict[str, Any], inside: dict[str, list[Any]], est: Any, limit_s: float,
                              budget_s: float = 300.0) -> tuple[str, float, Any] | None:
        """The study is over the time allowed: can settings share processes? The settings are spread over the slots the run
        uses at once (``ceil(settings / at_once)`` to a process, so the parallelism stays) and the study is run that way,
        with the runs per setting it will have (at most 3), in the order the run would use and in the opposite order, and
        the numbers are compared on every setting (core/trial_runner.py ``measure_grouping``). The time is the measured
        time of the first run, scaled to the runs per setting and padded by the sizing's margin: no term of the estimate
        decides it. That run stops, and the check with it, when it passes the limit less FI's margin (a longer
        run could not pass the test below) or the study's own estimate without sharing (a grouped run that takes as long saves
        nothing; the check's trials are not reused, so its time is extra). Sets
        ``runner.per_process`` and the groups, and returns ``(the console sentence, the expected seconds, the estimate for
        the run)`` when the study then fits the time allowed; else says why not in the log, leaves one process per setting,
        and returns ``None`` (the cut and the disclosure go on as before)."""
        import dataclasses
        import math

        from . import mc_error as _mc_error
        from . import run_facts as _run_facts

        log = self._log  # type: ignore[attr-defined]
        settings = int(est.cells or 0)
        at_once = max(1, int(est.at_once or 1))
        n = math.ceil(settings / at_once) if settings else 0
        if n < 2:
            return None  # every setting already has a slot of its own: there is no start-up to share
        timeout_s = int(max(30.0, min(budget_s, float(self.config.execution.timeout_s))))  # type: ignore[attr-defined]
        module = runner.simulate.relative_to(self.quest_root).as_posix()  # type: ignore[attr-defined]
        try:
            protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            protocol = {}
        began = time.monotonic()
        try:
            grouping = await _trial_runner.measure_grouping(
                self.executor, python, self.quest_root, module, grid,  # type: ignore[attr-defined]
                runs_per_setting=runs, base_seed=int((env or {}).get("FI_REPLICATE_SEED") or 0),
                deterministic=runner.deterministic, timeout_s=timeout_s, per_process=n, cost=getattr(runner, "cost", None),
                max_pass_s=min(limit_s / _estimate.SAFETY, float(est.seconds or limit_s)), env=env, thresholds=thresholds, paired=_mc_error.paired(protocol, self.quest_root),
                in_each_run=inside, log=log, at_once=at_once)
        except Exception as e:  # noqa: BLE001 -- a check that cannot run leaves the run as it is
            log.warning("[execute] Settings run one per process: the check of sharing processes could not be run (%r).", e)
            return None
        finally:
            self._pilot_spent += time.monotonic() - began  # type: ignore[attr-defined]  # the check's time counts however it ends
        if not grouping.same:
            log.warning("[execute] %s", grouping.says)
            if grouping.reuse is not None:
                grouping.reuse.discard()
            return None
        wanted = 1 if runner.deterministic else max(1, int(runs))
        seconds = grouping.forward_s * wanted / max(1, grouping.runs_checked)
        padded = seconds * _estimate.SAFETY
        if padded > limit_s:
            log.warning("[execute] %s Even with %d settings to a process, timed that way, FI expects about %s, over the limit "
                        "of %s, so the usual cut applies.", grouping.says, n, _estimate.plain_duration(padded),
                        _estimate.plain_duration(limit_s))
            if grouping.reuse is not None:
                grouping.reuse.discard()
            return None
        runner.per_process = n
        runner.group_reuse = grouping.reuse
        runner.shared_groups = grouping.groups
        runner.group_allowance_s = padded
        _run_facts.record_shared_check(self.quest_root, settings=grouping.settings, alone=grouping.alone)  # type: ignore[attr-defined]
        # Whether the first run is kept as the study's own is said when the run starts (trial_runner.run_trials), once it is
        # known that nothing changed after this check.
        log.info("[execute] %s Settings run %d to a process (%d processes); timed that way, FI expects about %s, within the "
                 "limit.", grouping.says, n, len(grouping.groups), _estimate.plain_duration(padded))
        said = ("FI runs several settings together to save start-up time; their numbers matched when they were run in either "
                f"order. Timed that way, it expects about {_estimate.plain_duration(padded)}, within the limit.")
        return said, seconds, dataclasses.replace(est, seconds=seconds)

    async def _speed_up_if_long(self, state: Any, runner: Any, python: Any, env: Any, grid: dict[str, list[Any]], runs: int,
                                thresholds: dict[str, Any], inside: dict[str, list[Any]], est: Any, limit_s: float,
                                fits: bool | None = None, **kwargs: Any) -> dict[str, Any] | None:
        """The speed-up step for a study FI estimates at ``est`` (the whole study: a search's evaluations and FI's check at
        finer settings included; padded by ``run_estimate.SAFETY``): always when it does not fit the time allowed
        (``fits``; default: ``run_estimate.known_too_long`` on ``est``), and when it fits only when it is long
        (``improve_speed.worth_speeding_up``). ``None`` when it is not tried: the study fits and is short (one line says
        so) or ``engine.improve_rounds`` is 0."""
        from . import improve_speed as _speed

        padded_s = float(est.seconds) * _estimate.SAFETY
        if fits is None:
            fits = not _estimate.known_too_long(est, limit_s)
        if not _speed.worth_speeding_up(padded_s, fits):
            self._log.info("[improve] %s", _speed.not_worth_it_line(padded_s))  # type: ignore[attr-defined]
            return None
        if int(self.config.engine.improve_rounds or 0) <= 0:  # type: ignore[attr-defined]
            return None
        return await self._speed_up(state, runner, python, env, grid, runs, thresholds, inside, est, limit_s, **kwargs)

    async def _speed_up(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """:meth:`_speed_up_rounds`, which commits each version it keeps with its own words (code_project.CodeRuns)."""
        with self._hold_code_runs():  # type: ignore[attr-defined]
            return await self._speed_up_rounds(*args, **kwargs)

    async def _speed_up_rounds(self, state: Any, runner: Any, python: Any, env: Any, grid: dict[str, list[Any]], runs: int,
                               thresholds: dict[str, Any], inside: dict[str, list[Any]], est: Any, limit_s: float,
                               budget_s: float = 600.0, probes: list[Any] | None = None,
                               deterministic: bool | None = None,
                               extra_cells: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """The study does not fit, or it fits and is long (core/improve_speed.py ``worth_speeding_up``): rounds
        of one edit each that make the simulation itself faster without changing a number. FI runs a fixed sample of
        trials, the checks' own cases included, twice on the unchanged code and once after each edit; an edit is kept only
        when every number is the same, it passes every guard the correctness loop has, and the sample is faster by at least
        ``improve_speed.MIN_SAVING`` of its time and by more than the two samples of the unchanged code differ
        (``improve_speed.fast_enough``), measured against the faster of those two samples (then against the last kept
        version). A round that is not kept (the edit refused, the reply unreadable, a number changed, or the saving under
        a tenth or within the noise) does not end the step: the model is asked again with the reason. The rounds stop
        after ``engine.improve_rounds`` rounds, at a kept round whose wall time (the model's answer and the sample) is more
        than it saves on the study (its share saved times FI's estimate of the study before it), or before a round when
        the last one took longer than the least saving FI keeps would give back on the study. A setting of the sample the study's timing never ran is timed on its own first, so its time limit comes from
        its own cost. After a kept edit the study is timed twice and the slower timing stands (``probes``), so the noise of
        one timing cannot make it fit. ``{"kept": bool, "rounds": [...], "calls": n, "speedup": x, "probes": [...] | None,
        "stopped": why}``. ``deterministic``: whether a setting is one call with no randomness (default: the runner's own);
        ``extra_cells``: settings the sample holds the change to as well (a search's designs outside its coarse scan)."""
        import asyncio

        from . import changelog as _changelog
        from . import code_project as _code_project
        from . import criteria as _criteria
        from . import improve as _improve
        from . import improve_speed as _speed
        from . import oracle_check as _oracle

        root = self.quest_root  # type: ignore[attr-defined]
        log = self._log  # type: ignore[attr-defined]
        limit = int(self.config.engine.improve_rounds or 0)  # type: ignore[attr-defined]
        module = runner.simulate.relative_to(root).as_posix()
        protocol = self._protocol_block(state) or {}  # type: ignore[attr-defined]
        # The guards of the correctness loop, over every check of the plan (not only the ones a criterion names): a speed-up
        # may not write in a value a check expects, change how a check's number is worked out (so a second implementation
        # stays a second implementation), or recognise the setting a check runs on.
        oracles = [o for o in _oracle.declared(protocol)]
        countable = _criteria.countable(protocol)
        trial_names = {c["trials"] for c in countable if c.get("trials")}
        expected, measures, settings = _improve.guards(protocol, oracles, countable, trial_names)
        det = bool(runner.deterministic if deterministic is None else deterministic)
        cells = _speed.sample_cells(grid, oracles)
        cells += [dict(c) for c in extra_cells or [] if c not in cells]
        for cell in cells:
            for key, value in cell.items():
                if isinstance(value, (int, float, str)) and not isinstance(value, bool):
                    settings.setdefault(str(key), []).append(value)
        timeout_s = max(60, int(self.config.execution.timeout_s))  # type: ignore[attr-defined]
        timeouts: dict[str, int] | None = None
        if probes is not None:
            # A setting of the sample the study's timing never ran (a check's case, a search's design) is timed first, given
            # at most what one sample may take: a setting slower than that could not be sampled anyway. Its time limit in
            # the sample then comes from its own cost (one such setting, finer than the base, was given the base's).
            probes = list(probes)
            timing_started = time.monotonic()
            share = _speed.sample_share_s(budget_s, 2 + limit)
            for cell in _speed.untimed_cells(cells, probes):
                left = budget_s - (time.monotonic() - timing_started)
                if left < 5:
                    break
                probe, why_timed = await _estimate._time_one(
                    self.executor, python, root, module, cell, runs=runs, deterministic=det,  # type: ignore[attr-defined]
                    timeout_s=int(max(5, min(share, left, timeout_s))), env=env, thresholds=thresholds,
                    in_each_run=inside, name="speed_timing")
                if probe is not None:
                    probes.append(probe)  # one that ran out of its time is a lower bound, left out of the sample below
                else:
                    log.info("[improve] A setting of the sample could not be timed (%s); it is left out of the sample.",
                             why_timed)
            budget_s = max(0.0, budget_s - (time.monotonic() - timing_started))
            # The sample is held to what the timing measured: a setting it did not see finish is left out, each gets a
            # few times its own measured cost, and every sample of the step fits in the time left for timing.
            cells, timeouts, why = _speed.bounded_sample(cells, probes, runs=runs, deterministic=det,
                                                         budget_s=budget_s, samples=2 + limit)
            if not cells:
                log.info("[improve] The simulation is not sped up: %s.", why)
                return {"goal": "speed", "limit_s": limit_s, "rounds": [], "kept": False, "calls": 0, "probes": None,
                        "stopped": why}
            if why:
                log.info("[improve] %s; a speed-up is not tried on them.", why[0].upper() + why[1:])

        async def sample(repeats: int = _speed.REPEATS) -> Any:
            return await _speed.run_sample(self.executor, python, root, module, cells,  # type: ignore[attr-defined]
                                           deterministic=det, thresholds=thresholds, in_each_run=inside,
                                           env=env, timeout_s=timeout_s, log=log, timeouts=timeouts, repeats=repeats)

        async def time_study() -> list[Any] | None:
            probes, _why = await _estimate.measure(
                self.executor, python, root, module, grid, runs=runs, deterministic=det,  # type: ignore[attr-defined]
                limit_s=limit_s, budget_s=budget_s, env=env, thresholds=thresholds, cached=None, in_each_run=inside)
            return probes

        padded = est.seconds * _estimate.SAFETY
        record: dict[str, Any] = {"goal": "speed", "limit_s": limit_s, "estimate_s": padded, "rounds": [], "kept": False,
                                  "calls": 0, "probes": None}
        def stop(why: str) -> dict[str, Any]:
            # Every way the step ends without trying a change says why, in run.log and the record.
            record["stopped"] = why
            log.info("[improve] The simulation is not sped up: %s.", why)
            self._save_speed_record(record)
            return record

        first = await sample()
        if first.failed or first.seconds <= 0:
            return stop(f"the sample did not run ({first.failed or 'no time measured'})")
        # The unchanged code is run a second time: a number that is not the same in both runs is not fixed by the code (a
        # measure of its own run time, say), so it cannot show that an edit changed a result and is left out of every comparison.
        again = await sample()
        if again.failed:
            return stop(f"the sample did not run a second time ({again.failed})")
        noisy = _speed.varying(first, again)
        if noisy:
            names = sorted(noisy)
            shown = [f"{k or 'its only setting'} trial {t} {n}" for k, t, n in names[:3]]
            record["varying"] = shown
            which = sorted({n.strip("/") for _k, _t, n in noisy})
            log.info("[improve] %d number(s) of the sample change from one run of the unchanged code to the next (%s); a "
                     "speed-up is judged on the other numbers.", len(noisy),
                     ", ".join(which[:3]) + (f" and {len(which) - 3} more" if len(which) > 3 else ""))
            if len(noisy) >= _speed.count_numbers(first):
                return stop("every number of the sample differs from one run of the unchanged code to the next, so an "
                            "edit cannot be judged")
        needed = padded / max(limit_s, 1e-9)
        # What two samples of the same, unchanged code differ by: an edit must save more than this to count as faster.
        noise_s = abs(first.seconds - again.seconds)
        record["noise_s"] = noise_s
        # The unchanged code's time is the faster of its two samples (noise is slower, never faster): an edit is judged
        # against that, so a slow first sample cannot make an edit that changes nothing look faster.
        unchanged_s = min(first.seconds, again.seconds) if again.seconds > 0 else first.seconds
        original = _improve.snapshot(root)
        best = {"files": original, "sample": _replace(first, seconds=unchanged_s)}
        tried = {_improve.fingerprint(original)}
        # Where the time goes, measured on the costliest setting the timing saw (the base when none was timed).
        timed = {_trial_runner.cell_key(p.cell): float(p.seconds) for p in probes or [] if getattr(p, "cell", None)}
        costliest = max(cells, key=lambda c: timed.get(_trial_runner.cell_key(c), 0.0))
        profile = await _speed.profile_text(self.executor, python, root, module, costliest,  # type: ignore[attr-defined]
                                            deterministic=det, thresholds=thresholds, in_each_run=inside, env=env,
                                            timeout_s=max(60, int(min(timeout_s, budget_s or timeout_s))), log=log)
        record["profiled"] = bool(profile)
        block = _speed.speed_block(padded, limit_s, first, cells, runs, profile=profile)
        over = padded > limit_s
        log.info("[improve] The study would take about %s, %s the %s allowed. Trying to make the simulation itself faster "
                 "without changing any result, one change at a time while each saves at least a tenth (up to %d).",
                 _estimate.plain_duration(padded), "more than" if over else "within", _estimate.plain_duration(limit_s),
                 limit)
        self._progress("[improve] " + ("The study is too slow for the time allowed" if over else "The study takes a while")  # type: ignore[attr-defined]
                       + "; trying to make the simulation faster without changing any result.")
        used = 0
        last_wall_s = 0.0
        while used < limit:
            study_now = padded * best["sample"].seconds / max(unchanged_s, 1e-9)
            if last_wall_s and last_wall_s > _speed.MIN_SAVING * study_now:
                record["stopped"] = (f"the last round took {_estimate.plain_duration(last_wall_s)}, more than the least "
                                     "saving FI keeps would give back on the study")
                break
            used += 1
            round_started = time.monotonic()
            before_s = best["sample"].seconds
            files = _improve.snapshot(root)
            allowed = _improve.editable(root)
            prompt = self._prompts["improve_speed"].substitute(  # type: ignore[attr-defined]
                topic_block=self._topic_block(), speed_block=block,  # type: ignore[attr-defined]
                history_block=_improve.history_block(record["rounds"]), editable=", ".join(allowed) or "(none)",
                code_block=_improve.code_block(files, allowed), rounds_left=f"This is round {used} of {limit}.")
            from . import skeleton as _skeleton

            prompt = _skeleton.repair_note(files.get("simulate.py", "")) + prompt  # what may change, first
            entry: dict[str, Any] = {"round": used}
            record["rounds"].append(entry)
            try:
                reply = await self._chat(prompt, node="improve")  # type: ignore[attr-defined]
                record["calls"] += 1
            except Exception as e:  # noqa: BLE001 -- no answer is no change
                entry.update(outcome="not asked", reason=repr(e)[:200])
                record["stopped"] = "the model could not be asked"
                break
            from .engine import _parse_json_lenient

            edit, why = _improve.parse_edit(_parse_json_lenient(reply, node="improve", want=_improve.EDIT_KEYS))
            new_text = None
            if edit is not None:
                new_text, why = _improve.check_edit(files, edit, allowed, tried=tried, expected=expected,
                                                    measures=measures, settings=settings)
            if edit is None or new_text is None:
                # The edit is kept with its refusal, so whether a guard refused it rightly can be judged afterwards.
                entry.update(outcome="refused", reason=why, why=edit.why if edit else "",
                             **({"file": edit.file, "diff": _improve.diff_text(edit)} if edit is not None
                                else {"reply": str(reply or "")[:8000]}))
                log.info("[improve] Speed-up %d refused before it ran: %s", used, why)
                # Not kept is not the end: the model is asked again with the reason (the history block), up to the cap.
                last_wall_s = time.monotonic() - round_started
                entry["wall_s"] = round(last_wall_s, 3)
                continue
            name = _improve.edited_path(edit)
            candidate = {**files, name: new_text}
            tried.add(_improve.fingerprint(candidate))
            entry.update(why=edit.why, file=name, diff=_improve.diff_text(edit))
            _improve.restore(root, candidate)
            saved = _improve.guard_bytes(root)
            before_guard = _improve.guard_hashes(root)
            after = await sample()
            touched = _improve.changed(before_guard, _improve.guard_hashes(root))
            if touched:
                _improve.put_back(root, saved)
                _improve.restore(root, best["files"])
                entry.update(outcome="aborted", reason="its run changed " + ", ".join(touched[:4]))
                log.warning("[improve] Speed-up %d stopped: its test run changed %s, which FI judges the code by; the "
                            "earlier version is back.", used, ", ".join(touched[:4]))
                record["stopped"] = f"round {used}'s test run changed files FI judges the code by"
                break
            if after.failed:
                entry.update(outcome="not kept", reason=f"the sample did not run ({after.failed})")
            elif (bad := _speed.disagreements(first, after, ignore=noisy)):
                entry.update(outcome="not kept", reason=f"{len(bad)} number(s) of the sample changed, first: {bad[0]}")
            elif not _speed.fast_enough(best["sample"].seconds, after.seconds, noise_s):
                entry.update(outcome="not kept", reason=f"the sample took {after.seconds:.3f} s against "
                                                         f"{best['sample'].seconds:.3f} s: not faster by a tenth and by "
                                                         f"more than the {noise_s:.3f} s two runs of the same code differ")
            elif not await _speed.confirm(root, best["files"], candidate, sample, entry, after,
                                          (before_s - after.seconds) / max(before_s, 1e-9) * study_now, before_s=before_s,
                                          ignore=noisy,
                                          guards_touched=lambda: _improve.changed(before_guard, _improve.guard_hashes(root))):
                if entry.get("outcome") == "aborted":
                    _improve.put_back(root, saved)
                    _improve.restore(root, best["files"])
                    log.warning("[improve] Speed-up %d stopped: its repeated runs changed %s, which FI judges the code by; "
                                "the earlier version is back.", used, ", ".join(entry.get("touched", [])[:4]))
                    record["stopped"] = f"round {used}'s repeated runs changed files FI judges the code by"
                    break
            else:
                entry.update(outcome="kept", reason=f"the sample took {after.seconds:.3f} s against "
                                                    f"{best['sample'].seconds:.3f} s, with the same numbers"
                                                    + (f"; timed again, old and new alternating: old {entry['repeats_old']} "
                                                       f"s, new {entry['repeats_new']} s" if entry.get("repeats_old")
                                                       else "; not timed again (it would cost more than it saves)"))
                best = {"files": candidate, "sample": after}
                record["kept"] = True
                await asyncio.to_thread(
                    _code_project.record_change, root, f"speed-up {used}: {edit.why}", category="changed",
                    results=_changelog.results(
                        "no: every number of a fixed sample of trials (the checks' own cases included) is the same, "
                        f"and the sample runs {unchanged_s / max(best['sample'].seconds, 1e-9):.1f} times faster."),
                    log=log)
            if entry["outcome"] != "kept":
                _improve.restore(root, best["files"])
            log.info("[improve] Speed-up %d %s: %s", used, "kept" if entry["outcome"] == "kept" else "not kept",
                     entry.get("reason"))
            last_wall_s = time.monotonic() - round_started
            entry["wall_s"] = round(last_wall_s, 3)
            if entry["outcome"] != "kept":
                continue  # asked again with the reason, up to the cap
            saved_s = (before_s - best["sample"].seconds) / max(before_s, 1e-9) * study_now
            entry["saves_on_study_s"] = round(saved_s, 1)
            if last_wall_s > saved_s:
                record["stopped"] = (f"round {used} took {_estimate.plain_duration(last_wall_s)}, more than the "
                                     f"{_estimate.plain_duration(saved_s)} it saves on the study")
                break
        else:
            record["stopped"] = f"all {limit} round(s) of engine.improve_rounds were used"
        if record.get("stopped"):
            log.info("[improve] The rounds of speed-up end: %s.", record["stopped"])
        _improve.restore(root, best["files"])
        record["sample_s_before"] = unchanged_s
        record["sample_s_after"] = best["sample"].seconds
        record["speedup"] = unchanged_s / max(best["sample"].seconds, 1e-9)
        record["needed"] = needed
        if record["kept"]:
            # Timed twice; the slower timing stands (one timing of the same study differs by a quarter or more).
            runs_timed = []
            for _ in range(2):
                probes = await time_study()
                if probes is None:
                    break
                e = _estimate.estimate(grid, runs, det, probes)
                if e is not None:
                    runs_timed.append((e.seconds, probes))
            if len(runs_timed) == 2:
                slower = max(runs_timed, key=lambda t: t[0])
                record["probes"] = slower[1]
                record["estimate_after_s"] = [round(t[0] * _estimate.SAFETY, 1) for t in runs_timed]
            # For a search this timing is of its coarse scan only: the whole study, with the search's evaluations, is
            # timed again right after (_size_the_scan), and says its own estimate.
            what = "the search's coarse scan" if extra_cells else "the study"
            log.info("[improve] The simulation is %.1f times faster with the same numbers on the sample%s.", record["speedup"],
                     f"; timed twice, {what} would now take about "
                     f"{_estimate.plain_duration(max(record['estimate_after_s']))}"
                     if record.get("estimate_after_s") else "")
        else:
            log.info("[improve] No change made the simulation clearly faster with the same numbers; it is left as it was.")
        self._save_speed_record(record)
        return record

    def _save_speed_record(self, record: dict[str, Any]) -> None:
        root = self.quest_root  # type: ignore[attr-defined]
        try:
            (root / ".fi").mkdir(parents=True, exist_ok=True)
            (root / ".fi" / "improve_speed.json").write_text(
                json.dumps({k: v for k, v in record.items() if k != "probes"}, indent=1, default=str), encoding="utf-8")
        except OSError:
            pass

    def _search_already_run(self, runner: Any, env: Any) -> bool:
        """Whether the search this run would start already ran, to its end or to the time it was given, on the simulation
        as it is now (FI's own record of it, :meth:`core.optimise.OptimisationRunner.kept_search`: the same simulation
        code and inputs, plan, seed and stops), under the same time limit. Then the run uses that record and only the
        analysis runs again (a repair of the analysis after an eight-hour search ran the whole search again): the scan is
        not timed and the simulation not sped up again (a kept speed-up changes the simulation, so the record would no
        longer be its search), and the time that search was given (the cut FI recorded for it) is given again. ``False``,
        and the study is sized as before, on a first run (no record), after any change to the simulation, for a longer
        time limit, or when the cut on record is not that search's."""
        kept_search = getattr(runner, "kept_search", None)
        best = kept_search(env if isinstance(env, dict) else None) if callable(kept_search) else None
        if not isinstance(best, dict):
            return False
        limit_s = float(self.config.execution.timeout_s)  # type: ignore[attr-defined]
        cut = _estimate.read_record(self.quest_root).get("truncated")  # type: ignore[attr-defined]
        cut = cut if isinstance(cut, dict) and cut.get("kind") == "search" else None
        if str((cut or {}).get("run_id") or "") != str(best.get("run_id") or ""):
            return False  # the cut on record is not the one that search ran under
        if cut is not None and float(cut.get("limit_s") or 0.0) != limit_s:
            return False
        reserve = min(max(0.0, float((cut or {}).get("check_reserve_s") or 0.0)), limit_s / 2.0)
        deadline = (cut or {}).get("search_deadline_s")
        deadline = float(deadline) if isinstance(deadline, (int, float)) and not isinstance(deadline, bool) else None
        stopped = (best.get("evaluations") or {}).get("stopped_because")
        ran_for = (best.get("time") or {}).get("limit_seconds")
        if stopped == "time":
            # A search the clock stopped stands for the time it was given, not a longer one (a second more is the cut's
            # rounding to a tenth of a second); it is given that time again, never more.
            if not isinstance(ran_for, (int, float)) or isinstance(ran_for, bool):
                return False
            now_s = int(limit_s - reserve) if deadline is None else min(int(limit_s - reserve), int(deadline))
            if now_s > ran_for + 1:
                return False
            deadline = float(ran_for) if deadline is None else min(deadline, float(ran_for))
        runner.check_reserve_s = reserve
        runner.search_deadline_s = deadline
        runner.run_id = str((cut or {}).get("run_id") or "")
        if hasattr(runner, "first_cells"):
            runner.first_cells = int(cut["planned_settings"]) if cut and cut.get("planned_settings") else None
        self._log.info("[execute] the search for the best design already ran on this simulation as it is now: FI uses "  # type: ignore[attr-defined]
                       "its record and runs only the analysis again (the run is not timed or sped up again)")
        return True

    def _clear_cut(self, runner: Any) -> None:
        """A search is sized again at every run: the cut an earlier sizing recorded (and the time it kept for FI's check) is
        taken away first, so a study that now fits is never described as cut."""
        runner.check_reserve_s = 0.0
        runner.search_deadline_s = None
        runner.run_id = ""
        if hasattr(runner, "first_cells"):
            runner.first_cells = None
        record = _estimate.read_record(self.quest_root)  # type: ignore[attr-defined]
        if record.pop("truncated", None) is not None:
            _estimate.write_record(self.quest_root, record)  # type: ignore[attr-defined]

    def _run_up_to_the_limit(self, runner: Any, est: _estimate.Estimate, limit_s: float, one_s: float | None, *, kind: str,
                             base: dict[str, Any] | None = None, planned: dict[str, Any] | None = None,
                             reserve_s: float = 0.0, detail: str = "", fit: dict[str, Any] | None = None,
                             first_cells: int | None = None, deadline_s: float | None = None) -> bool:
        """A study FI's own timing says does not fit the time allowed is run anyway, in the plan's order, up to the limit,
        unless not even one setting (or one design) fits: ``True`` to go on. FI does not choose which settings matter, so
        the grid runs in the plan's order (not costliest first), and a search runs its baseline, its coarse scan, its
        search and then FI's check, as it always does; the check keeps the time it needs (``reserve_s``, never more than
        half the limit) so the scan and the search cannot take all of it. The cut is kept in ``.fi/run_estimate.json``
        (``truncated``); the paper and the quality report say what ran from FI's records (:func:`core.disclosure.cut_short_text`)."""
        if not _estimate.one_fits(one_s, limit_s):
            return False
        reserve = min(max(0.0, reserve_s), limit_s / 2.0)
        cut = {"kind": kind, "estimate_s": est.seconds * _estimate.SAFETY, "at_least": bool(est.at_least),
               "limit_s": float(limit_s), "settings": int(est.cells), "one_s": one_s, "order": "plan",
               "run_id": uuid.uuid4().hex[:12], "check_reserve_s": round(reserve, 1), **(planned or {}), "detail": detail.split("; FI does not run")[0],
               **({"planned_settings": int(first_cells)} if first_cells else {}), **({"fit": fit} if fit else {})}
        what = "the search for the best design" if kind == "search" else "the experiment"
        line = (f"The plan has {int(est.cells):,} settings, more than the {_optimise_search.MAX_LISTED_SETTINGS:,} FI lists at most; FI estimates they "
                f"would take {'at least ' if est.at_least else 'about '}{_estimate.plain_duration(est.seconds * _estimate.SAFETY)}"
                f" (the limit is {_estimate.plain_duration(limit_s)}); it will plan the first {first_cells:,} in the plan's order "
                "and run them until the time limit, and the paper will say what was and was not done") if first_cells else (
                f"FI estimates {what} would take {'at least ' if est.at_least else 'about '}"
                f"{_estimate.plain_duration(est.seconds * _estimate.SAFETY)}, more than the "
                f"{_estimate.plain_duration(limit_s)} allowed; it will run the plan's settings in the plan's order until "
                "the time limit, and the paper will say what was and was not done")
        self._log.warning("[execute] %s", line)  # type: ignore[attr-defined]
        print(f"[FI] {line}")
        runner.run_id = cut["run_id"]
        if first_cells:
            runner.first_cells = int(first_cells)
        if kind == "grid":
            runner.cost = None
        else:
            runner.check_reserve_s = reserve
            # The search stops at the time FI's cost model gives it, so the check and the analysis keep theirs.
            runner.search_deadline_s = deadline_s
            if deadline_s is not None:
                cut["search_deadline_s"] = round(float(deadline_s), 1)
        record = {k: v for k, v in (base or {}).items() if k not in ("stopped", "fit")}
        record.setdefault("limit_s", float(limit_s))
        record.update(fits=False, truncated=cut, says=line)
        _estimate.write_record(self.quest_root, record)  # type: ignore[attr-defined]
        return True

    def _stop_too_long(self, text: str, fit: dict[str, Any] | None = None) -> dict[str, Any]:
        """The quest's plain stop: nothing was run for the study, it would not fit, and no paper is written. ``fit``
        (:meth:`_what_fits`): what FI measured, which lets the evidence gate ask the plan once for a study that fits."""
        out = self._stop_before_run(text, "too_long")
        if fit:
            out["exec_result"]["too_long_fit"] = dict(fit)
        return out

    @staticmethod
    def _what_fits(est: _estimate.Estimate, limit_s: float, *, search: bool = False) -> dict[str, Any] | None:
        """What FI measured about a study too long to run: the seconds one setting takes (as the run would take them,
        with the settings that run at the same time), how many settings fit the time allowed with FI's margin, and the
        settings the plan has. ``None`` when nothing was measured."""
        if est.cells <= 0 or est.seconds <= 0:
            return None
        # What is paid once per new value is not a cost of every setting: it is named apart, so the plan can take fewer of
        # those values, and the settings that fit are counted from what is left of the time.
        once = min(float(est.once_s) * max(1.0, float(est.slowdown or 1.0)), est.seconds * 0.999)
        per = (est.seconds - once) / est.cells
        room = max(per, limit_s / _estimate.SAFETY - once)
        out = {"seconds_per_setting": round(per, 3), "at_least": bool(est.at_least), "settings": int(est.cells),
               "settings_that_fit": max(1, int(room / per)), "limit_s": float(limit_s), "search": bool(search)}
        if once > 0:
            out.update(once_s=round(once, 1), once_axes=list(est.once_axes))
        return out

    def _stop_before_run(self, text: str, kind: str, tried: list[str] | None = None,
                         advice: str | None = None, ran: bool = False) -> dict[str, Any]:
        """The quest's plain stop before the study (``too_long``: it would not fit the time allowed; ``not_run``: a text of
        the simulation could not be checked), with no paper. ``ran``: the trials did run and the stop is in the analysis."""
        if ran:
            self._log.warning("[execute] %s; stopping, no paper is written", text)  # type: ignore[attr-defined]
        else:
            self._log.warning("[execute] %s; stopping, the study was not run", text)  # type: ignore[attr-defined]
        return {
            "exec_result": {
                "returncode": 1, "duration_s": 0.0, "timed_out": False, "stdout_tail": "", "stderr_tail": text[-2000:],
                "packages_note": getattr(self, "_packages_note", ""), "failed_script": "", "flat_output": "",
                "numeric_warnings": [], kind: text, **({f"{kind}_tried": tried} if tried else {}),
            **({f"{kind}_advice": advice} if advice else {}),
            },
            "exec_give_up_reason": text,
            "figures": [], "figure_records": {}, "result_json": {}, "exec_patch_pending": False,
            "result_json_replicates": [], "result_json_deterministic": False, "result_json_trials": False,
            "result_json_replicate_seed_ignored": False, "result_json_no_random_source": False,
        }
