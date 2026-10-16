"""Figure text that shows the code's name for a quantity instead of the reader's.

The figures of real quests labelled their axes "Ea_hat (kJ/mol)" and "lnA_hat", and a legend "pi1 = 0.05": the names
the script gave its variables. The plan can say what each quantity is called for a reader (``label`` of a metric,
``protocol.labels`` for a setting; :func:`core.plan.plan_labels`). This module reads each figure's axis labels, panel
titles and legend entries from the figure record the plot-style bootstrap wrote (``core/plot_style.py``) and names each
one that shows a code name, with the plan's label to use instead when the plan gives one. The engine sends these to the
redraw of the figures from the saved numbers (``replot_layout``): the experiment is not run again.

A name counts only when both hold, so that a symbol a scientist writes is never taken for code:

- it is FI's to tie to the study: one of the plan's own names (or one extending it, ``lnA_hat``), or a name the
  experiment's code uses (a variable, an argument, a key it stores a value under);
- it has the shape of code, outside ``$...$`` math: two or more underscores (``t_max_over_k``); one underscore after
  two or more letters (``Ea_hat``, ``lnA_hat``) or before a part longer than a subscript (``p_value``); or two or more
  lower-case letters run into digits (``pi1``).

A letter with a short subscript (``k_B``, ``T_c``, ``E_a``, ``V_max``, ``G_f``), ``log_10``, a symbol with a Greek
subscript (``Re_lambda``), a letter with digits (``c0``) and a short name (``T``, ``N``, ``dt``) are how a scientist writes
a symbol: never flagged. No topic knowledge: the names come from the plan and the code, the shapes are those of code.
Pure, never raises.
"""

from __future__ import annotations

import ast
import re
from typing import Any

_MATH = re.compile(r"\$[^$]*\$")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_RUN_ON = re.compile(r"^[a-z]{2,}[0-9]+$")
_GREEK = frozenset(
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi "
    "chi psi omega".split())
#: The longest part after an underscore that still reads as a subscript (``V_max``, ``t_half``).
_SUBSCRIPT_MAX = 4


def code_shaped(word: str) -> bool:
    """Whether ``word`` is written the way code names a variable rather than the way a scientist writes a symbol."""
    parts = word.split("_")
    if len(parts) > 2:
        return True
    if len(parts) == 2:
        base, sub = parts
        if not base or not sub:
            return False
        if base.lower() in ("log", "ln") and sub.isdigit():
            return False  # log_10
        if sub.lower() in _GREEK and len(base) <= 2:
            return False  # Re_lambda
        return len(base) >= 2 or len(sub) > _SUBSCRIPT_MAX
    return bool(_RUN_ON.match(word))


def code_names(label: str, names: set[str]) -> list[str]:
    """The code names ``label`` shows, in the order it shows them (each once): words of the shape of code that are
    in ``names`` (the plan's names and the code's), or extend one of them (``lnA_hat`` from ``lnA``)."""
    text = _MATH.sub(" ", str(label or ""))
    found: list[str] = []
    for word in _WORD.findall(text):
        if word in found or not code_shaped(word):
            continue
        if word in names or any(word.startswith(name + "_") for name in names if name):
            found.append(word)
    return found


def code_identifiers(sources: list[str]) -> set[str]:
    """The names the experiment's code gives its values: variables, function arguments and names, keyword arguments,
    attributes it sets, and string keys it stores or reads a value under (``results["Ea_hat"]``). A source that does not
    parse gives none. A string that is only drawn (an axis label) is not a key, so a label never vouches for itself."""
    out: set[str] = set()
    for source in sources:
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                out.add(node.id)
            elif isinstance(node, ast.arg):
                out.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(node.name)
            elif isinstance(node, ast.keyword) and node.arg:
                out.add(node.arg)
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                out.add(node.attr)
            elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                if isinstance(node.slice.value, str):
                    out.add(node.slice.value)
            elif isinstance(node, ast.Dict):
                out.update(k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str))
    return {name for name in out if name.isidentifier()}


def _reader_label(name: str, labels: dict[str, str]) -> str | None:
    """The plan's label for ``name``: its own, else that of the longest plan name it extends (``Ea_hat`` from ``Ea``)."""
    if name in labels:
        return labels[name]
    stems = [key for key in labels if name.startswith(key + "_")]
    return labels[max(stems, key=len)] if stems else None


def _texts(ax: dict[str, Any]) -> list[tuple[str, str]]:
    """``(where, text)`` of each piece of an Axes' text a reader reads: its axis labels, its title, its legend entries."""
    out = [("x axis label", str(ax.get("xlabel") or "")), ("y axis label", str(ax.get("ylabel") or "")),
           ("title", str(ax.get("title") or ""))]
    out += [("legend entry", str(s.get("label") or "")) for s in ax.get("series") or [] if isinstance(s, dict)]
    return [(where, text.strip()) for where, text in out if text.strip() and not text.strip().startswith("_")]


def findings(records: dict[str, Any], figures: list[str], labels: dict[str, str], names: set[str],
             code: set[str] | None = None) -> list[dict[str, Any]]:
    """One finding per axis label, panel title or legend entry that shows a code name: ``figure``, ``panel`` (its
    title, or its place), ``where`` (``x axis label``, ``y axis label``, ``title`` or ``legend entry``), ``label`` (the
    text), ``code_names`` and ``rename_to`` (the plan's label, or ``None`` when the plan has none). ``names`` are the
    plan's names, ``code`` those of the experiment's code (:func:`code_identifiers`). A figure without a record has
    nothing to read."""
    out: list[dict[str, Any]] = []
    tied = set(names) | set(labels) | set(code or ())
    for figure in figures:
        record = records.get(figure) if isinstance(records, dict) else None
        if not isinstance(record, dict):
            continue
        axes = [ax for ax in record.get("axes") or [] if isinstance(ax, dict)]
        for index, ax in enumerate(axes):
            seen: set[tuple[str, str]] = set()
            for where, label in _texts(ax):
                if (where, label) in seen:
                    continue
                seen.add((where, label))
                shown = code_names(label, tied)
                if not shown:
                    continue
                rename = next((r for r in (_reader_label(n, labels) for n in shown) if r), None)
                if rename is not None and " ".join(rename.split()) == " ".join(label.split()):
                    continue
                title = str(ax.get("title") or "").strip()
                out.append({
                    "figure": figure,
                    "panel": f'"{title}"' if title and where != "title" else (
                        f"panel {index + 1}" if len(axes) > 1 else ""),
                    "where": where, "label": label, "code_names": shown, "rename_to": rename,
                })
    return out


#: Every sentence about a code name says this, so the redraw can tell them from a person's notes.
MARK = "shows the code name"


def sentence(finding: dict[str, Any]) -> str:
    """The finding as one sentence for the redraw."""
    names = ", ".join(f'"{n}"' for n in finding["code_names"])
    panel = finding.get("panel") or ""
    where = f"in the panel {panel}, " if panel.startswith('"') else (f"in {panel}, " if panel else "")
    what = finding.get("where") or "axis label"
    if what.endswith("axis label"):
        rename = (f'rename it to "{finding["rename_to"]}" (the plan\'s name for it)' if finding.get("rename_to") else
                  "rename it to the reader's name for it: words, its symbol in $...$ and its unit")
    else:
        rename = (f'write {names} as the plan names it, "{finding["rename_to"]}"' if finding.get("rename_to") else
                  f"write {names} as a reader would: in words, or as its symbol in $...$")
    return f'figures/{finding["figure"]}: {where}the {what} "{finding["label"]}" {MARK} {names}; {rename}'
