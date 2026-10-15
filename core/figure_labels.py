"""Axis labels that show the code's name for a quantity instead of the reader's.

The figures of real quests labelled their axes "Ea_hat (kJ/mol)", "lnA_hat" and "fraction of real effects, pi1": the
names the script gave its variables. The plan can say what each quantity is called for a reader (``label`` of a
metric, ``protocol.labels`` for a setting; :func:`core.plan.plan_labels`). This module reads each figure's axis labels
from the figure record the plot-style bootstrap wrote (``core/plot_style.py``) and names each label that shows a code
name, with the plan's label to use instead when the plan gives one. The engine sends these with the figure-layout
findings to the one redraw a run's figures are offered (``_figure_overlap_findings`` in ``core/engine.py``).

What counts as a code name, outside ``$...$`` math (where ``E_a`` is a symbol, not a variable):
- a word joined by underscores (``Ea_hat``, ``n_points``, ``t_max_over_k``);
- a plan name written as letters run into digits (``pi1``, ``c0``) -- only when the plan names it, since a word such as
  ``CO2`` or ``H2O`` is also letters and digits;
- a whole label (less a unit in brackets at its end) that is a plan name (``sigma``, ``m``).

No topic knowledge: the names come from the plan and the shapes are those of code. Pure, never raises.
"""

from __future__ import annotations

import re
from typing import Any

_MATH = re.compile(r"\$[^$]*\$")
_WORD = re.compile(r"[A-Za-z0-9_]+")
_JOINED = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+$")
_RUN_ON = re.compile(r"^[A-Za-z]+[0-9]+$")
_TRAILING_UNIT = re.compile(r"\s*[(\[][^()\[\]]*[)\]]\s*$")


def code_names(label: str, names: set[str]) -> list[str]:
    """The code names ``label`` shows, in the order it shows them (each once)."""
    text = _MATH.sub(" ", str(label or ""))
    found: list[str] = []
    for word in _WORD.findall(text):
        if word in found:
            continue
        if _JOINED.match(word) or (word in names and _RUN_ON.match(word)):
            found.append(word)
    bare = _TRAILING_UNIT.sub("", text).strip(" ,;:")
    if bare in names and bare not in found:
        found.append(bare)
    return found


def _reader_label(name: str, labels: dict[str, str]) -> str | None:
    """The plan's label for ``name``: its own, else that of the longest plan name it extends (``Ea_hat`` from ``Ea``)."""
    if name in labels:
        return labels[name]
    stems = [key for key in labels if name.startswith(key + "_")]
    return labels[max(stems, key=len)] if stems else None


def findings(records: dict[str, Any], figures: list[str], labels: dict[str, str], names: set[str]) -> list[dict[str, Any]]:
    """One finding per axis label that shows a code name: ``figure``, ``panel`` (its title, or its place), ``axis``
    (``x`` or ``y``), ``label``, ``code_names`` and ``rename_to`` (the plan's label, or ``None`` when the plan has
    none). A figure without a record has nothing to read."""
    out: list[dict[str, Any]] = []
    names = set(names) | set(labels)
    for figure in figures:
        record = records.get(figure) if isinstance(records, dict) else None
        if not isinstance(record, dict):
            continue
        axes = [ax for ax in record.get("axes") or [] if isinstance(ax, dict)]
        for index, ax in enumerate(axes):
            for axis in ("x", "y"):
                label = str(ax.get(f"{axis}label") or "").strip()
                if not label:
                    continue
                shown = code_names(label, names)
                if not shown:
                    continue
                rename = next((r for r in (_reader_label(n, labels) for n in shown) if r), None)
                if rename is not None and " ".join(rename.split()) == " ".join(label.split()):
                    continue
                title = str(ax.get("title") or "").strip()
                out.append({
                    "figure": figure,
                    "panel": f'"{title}"' if title else (f"panel {index + 1}" if len(axes) > 1 else ""),
                    "axis": axis, "label": label, "code_names": shown, "rename_to": rename,
                })
    return out


#: Every sentence about an axis label says this, so the redraw request can tell them from the layout findings.
MARK = "shows the code name"


def sentence(finding: dict[str, Any]) -> str:
    """The finding as one sentence for the redraw request."""
    names = ", ".join(f'"{n}"' for n in finding["code_names"])
    where = f'in the panel {finding["panel"]}, ' if finding.get("panel", "").startswith('"') else (
        f'in {finding["panel"]}, ' if finding.get("panel") else "")
    rename = (f'rename it to "{finding["rename_to"]}" (the plan\'s name for it)' if finding.get("rename_to") else
              "rename it to the reader's name for it: words, its symbol in $...$ and its unit")
    return (f'figures/{finding["figure"]}: {where}the {finding["axis"]} axis label "{finding["label"]}" {MARK} '
            f"{names}; {rename}")
