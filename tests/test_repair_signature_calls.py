"""A repair that changes a function's parameters is accepted when every call of it binds to the new ones (no call that
worked is broken), and refused, with the same reason as before, when a call that bound to the old ones no longer does."""

from __future__ import annotations

from core import finding_repair as fr
from core import script_reply

# A helper with four parameters that every call gives five arguments: no call of it works.
BROKEN_CALLS = '''import numpy as np


def figure_a(fig, table):
    _plot_pair(fig, table, "speed", "load", 0)


def figure_b(fig, table):
    _plot_pair(fig, table, "speed", "drag", 0)
    _plot_pair(fig, table, "speed", "lift", 1)


def _plot_pair(fig, panel, xid, yid):
    return [], []


def _draw(fig, table, panel, xid, yid):
    fig.points(panel, yid, (xid, [1.0]), [2.0], yid)
'''

# The repair: the helper now takes what every call passes.
REPLY_FITS = '''```python
def _plot_pair(fig, table, xid, yid, panel):
    _draw(fig, table, panel, xid, yid)
```'''

# A file where the old helper is called right: (fig, panel, xid, yid).
WORKING_CALLS = '''def figure_a(fig, table):
    _plot_pair(fig, 0, "speed", "load")
    _plot_pair(fig, 1, "speed", "drag")


def _plot_pair(fig, panel, xid, yid):
    fig.points(panel, yid, (xid, [1.0]), [2.0], yid)
'''

REPLY_BREAKS = '''```python
def _plot_pair(fig, table, xid, yid, panel):
    fig.points(panel, yid, (xid, [1.0]), [2.0], yid)
```'''

REFUSED = "must keep its parameters (fig, panel, xid, yid)"


def test_new_parameters_that_every_call_fits_are_accepted():
    out, replaced, why = fr.splice(BROKEN_CALLS, REPLY_FITS, None, None, limit=None)
    assert why == ""
    assert replaced == ["_plot_pair"]
    assert "def _plot_pair(fig, table, xid, yid, panel):" in out


def test_the_script_repair_path_accepts_it_too():
    made = script_reply.resolve(BROKEN_CALLS, "", REPLY_FITS)
    assert made.why == ""
    assert made.kind == "functions"
    assert "_draw(fig, table, panel, xid, yid)" in made.code


def test_a_call_that_worked_and_no_longer_binds_is_still_refused():
    out, replaced, why = fr.splice(WORKING_CALLS, REPLY_BREAKS, None, None, limit=None)
    assert out is None
    assert REFUSED in why


def test_one_working_call_among_broken_ones_is_still_refused():
    mixed = BROKEN_CALLS.replace('_plot_pair(fig, table, "speed", "lift", 1)', '_plot_pair(fig, 1, "speed", "lift")')
    out, _replaced, why = fr.splice(mixed, REPLY_FITS, None, None, limit=None)
    assert out is None
    assert REFUSED in why


def test_a_call_in_another_file_that_does_not_bind_is_refused():
    other = "from analysis import _plot_pair\n\n\ndef side(fig):\n    _plot_pair(fig, 0, 'a', 'b')\n"
    out, _replaced, why = fr.splice(BROKEN_CALLS, REPLY_FITS, None, None, limit=None, elsewhere=[other])
    assert out is None
    assert REFUSED in why


def test_a_function_that_is_never_called_keeps_the_old_rule():
    alone = "def _plot_pair(fig, panel, xid, yid):\n    return None\n"
    out, _replaced, why = fr.splice(alone, REPLY_FITS.replace("_draw(fig, table, panel, xid, yid)", "return None"),
                                    None, None, limit=None)
    assert out is None
    assert REFUSED in why


def test_a_function_handed_on_keeps_the_old_rule():
    handed = BROKEN_CALLS + "\n\nHANDLERS = [_plot_pair]\n"
    out, _replaced, why = fr.splice(handed, REPLY_FITS, None, None, limit=None)
    assert out is None
    assert REFUSED in why


def test_a_function_fi_calls_itself_keeps_the_old_rule():
    body = ("def figure_a(fig, table):\n    return None\n\n\ndef helper(fig):\n    figure_a(fig, None, 0)\n")
    reply = "```python\ndef figure_a(fig, table, panel):\n    return None\n```"
    out, _replaced, why = fr.splice(body, reply, None, None, limit=None)
    assert out is None
    assert "must keep its parameters (fig, table)" in why


def test_a_call_with_starred_arguments_keeps_the_old_rule():
    starred = BROKEN_CALLS.replace('_plot_pair(fig, table, "speed", "load", 0)', '_plot_pair(*args)')
    out, _replaced, why = fr.splice(starred, REPLY_FITS, None, None, limit=None)
    assert out is None
    assert REFUSED in why
