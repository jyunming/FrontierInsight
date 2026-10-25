"""What a repair of a script gives back, as the script it makes.

A repair of ``simulate.py`` (or the analysis) used to be asked for the whole file. A reply that came back cut short lost
the helpers it left out, a stand-in word replaced the script, and a result name changed in passing made a check that had
passed fail again. Now a repair is asked for the **functions** to replace or add; FI puts them in the file with the same
splice a repair of a finding uses (:func:`core.finding_repair.splice`) and leaves every other byte as it was. A reply
that is a whole script (in `code`) is still taken whole. In both cases the one contract check
(:func:`core.trial_runner.repair_breaks_contract`) judges the result, which every repair path asks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import finding_repair as _finding


@dataclass
class Resolved:
    """``code`` is the script the reply makes (``""`` when it makes none), ``kind`` how (``whole`` / ``functions`` /
    ``none``), ``replaced`` the functions put in (``functions`` only) and ``why`` the reason when nothing was made."""
    code: str = ""
    kind: str = "none"
    replaced: list[str] = field(default_factory=list)
    why: str = ""


def resolve(old: str, code: object = "", functions: object = "") -> Resolved:
    """The script that a repair reply makes from ``old``. ``functions`` is the reply's functions to replace or add (their
    whole ``def``, any number); ``code`` is its script. Functions win when both are given. A ``code`` is the whole script
    and is returned as it is, for the contract check to judge (which refuses a stand-in word, a lost entry point and a
    function that is called but no longer defined)."""
    fn_text = functions if isinstance(functions, str) else ""
    code_text = code if isinstance(code, str) else ""
    if fn_text.strip():
        return _splice(old, fn_text)
    if not code_text.strip():
        return Resolved(why="the reply held neither functions nor a script")
    return Resolved(code=code_text, kind="whole")


def _splice(old: str, text: str) -> Resolved:
    new, replaced, why = _finding.splice(old, text, None, None, limit=None)
    if new is None:
        return Resolved(why=why)
    return Resolved(code=new, kind="functions", replaced=replaced)
