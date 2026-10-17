"""core/unread_settings.py on the review's probe matrix (tests/unread_settings_cases.py): a setting the code uses, in the
simulation or in the analysis, is never named unread. Saying "unsure" (no setting named, with the reason) is always
allowed; naming a used setting would take it out of the plan."""

from __future__ import annotations

import pytest

from core import unread_settings
from tests.unread_settings_cases import CASES, GRID

PROTOCOL = {"grid": GRID}
USED = sorted(name for name, (used, _files) in CASES.items() if used)


@pytest.mark.parametrize("name", USED)
def test_a_setting_the_code_uses_is_never_named_unread(name):
    _used, files = CASES[name]
    unread, why = unread_settings.unread_axes(files, PROTOCOL)
    assert "T" not in unread, (name, why)


@pytest.mark.parametrize("name", ["control_T_unread", "module_runner_imports_reads_at_import"])
def test_a_setting_nothing_reads_is_named(name):
    _used, files = CASES[name]
    assert unread_settings.unread_axes(files, PROTOCOL) == (["T"], "")


@pytest.mark.parametrize("name, says", [
    ("eval_string", "eval"), ("exec_string", "exec"), ("locals_lookup", "locals"),
    ("decorator_reads_setting", "decorat"), ("module_rebinds_local_function", "helper"),
    ("star_args_before_cell_2", "*"), ("unbound_method_offset_2", "class"),
])
def test_what_cannot_be_followed_is_said(name, says):
    _used, files = CASES[name]
    keys, why = unread_settings.setting_reads(files)
    assert keys is None and says in why, why


def test_the_analysis_is_read_too():
    _used, files = CASES["analysis_only_threshold"]
    keys, why = unread_settings.setting_reads(files)
    assert why == "" and {"a", "T"} <= keys
    flat = {"simulate.py": "def run_trial(cell, t, s):\n    return {'x': cell['a']}\n",
            "experiment.py": "import pandas as pd\ndf = pd.json_normalize(rows)\nprint(df['cell.T'].mean())\n"}
    assert "T" in unread_settings.setting_reads(flat)[0]
    keyed = {"simulate.py": flat["simulate.py"],
             "experiment.py": "for rec in recs:\n    print(rec['key'].split(','))\n"}
    keys, why = unread_settings.setting_reads(keyed)
    assert keys is None and "key" in why
