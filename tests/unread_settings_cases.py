"""Probe cases for "which grid settings does the simulation read". Grid always {a: [1, 2], T: [300, 310]}.
`T_used` says whether T really changes what the study computes (simulation or analysis).

Copied from the review of the plan-fidelity change (its probe matrix); read by tests/test_unread_settings_cases.py."""

GRID = {"a": [1.0, 2.0], "T": [300.0, 310.0]}

CASES = {}


_NAMES = {"pkg____init___py": "pkg/__init__.py", "pkg__model_py": "pkg/model.py", "experiment_py": "experiment.py"}


def case(name, T_used, sim, **extra):
    files = {"simulate.py": sim}
    files.update({_NAMES[k]: v for k, v in extra.items()})
    CASES[name] = (T_used, files)


# --- the user's list -------------------------------------------------------------------------------------------------
case("getattr", True, "def run_trial(cell, t, s):\n    T = getattr(cell, 'get')('T')\n    return {'x': cell['a'] * T}\n")
case("get_in_loop_over_keys", True,
     "def run_trial(cell, t, s):\n    v = {}\n    for k in ('a', 'T'):\n        v[k] = cell.get(k)\n    return {'x': v['a'] * v['T']}\n")
case("spread_to_helper_other_file", True,
     "from pkg.model import solve\ndef run_trial(cell, t, s):\n    return solve(**cell)\n",
     pkg____init___py="", pkg__model_py="def solve(a, T):\n    return {'x': a * T}\n")
case("helper_other_file_param_p", True,
     "from pkg.model import solve\ndef run_trial(cell, t, s):\n    return solve(cell)\n",
     pkg____init___py="", pkg__model_py="def solve(p):\n    return {'x': p['a'] * p['T']}\n")
case("dataclass_from_cell_spread", True,
     "from dataclasses import dataclass\n@dataclass\nclass P:\n    a: float\n    T: float\n"
     "def run_trial(cell, t, s):\n    p = P(**cell)\n    return {'x': p.a * p.T}\n")
case("namedtuple_from_cell", True,
     "from collections import namedtuple\ndef run_trial(cell, t, s):\n    P = namedtuple('P', cell.keys())\n"
     "    p = P(*cell.values())\n    return {'x': p.a * p.T}\n")
case("only_in_run_cell_runner_uses_run_trial", False,
     "def run_trial(cell, t, s):\n    return {'x': cell['a']}\ndef run_cell(cell):\n    return {'x': cell['a'] * cell['T']}\n")
case("seed_from_setting", True,
     "import numpy as np\ndef run_trial(cell, t, s):\n    rng = np.random.default_rng(int(cell['T']) + s)\n"
     "    return {'x': cell['a'] * rng.normal()}\n")
case("fstring", True, "def run_trial(cell, t, s):\n    tag = f'{cell}'\n    return {'x': cell['a'] * len(tag)}\n")
case("json_dumps", True, "import json\ndef run_trial(cell, t, s):\n    return {'x': cell['a'] * len(json.dumps(cell))}\n")
case("dataframe", True,
     "import pandas as pd\ndef run_trial(cell, t, s):\n    df = pd.DataFrame([cell])\n    return {'x': float(df['a'][0] * df['T'][0])}\n")
case("analysis_only_threshold", True,
     "import numpy as np\ndef run_trial(cell, t, s):\n    rng = np.random.default_rng(s)\n    return {'stat': float(cell['a'] * rng.normal())}\n",
     experiment_py=("import json\nrecs = json.load(open('raw/trials.json'))\nfor rec in recs['cells']:\n"
                    "    thr = rec['cell'].get('T', 300.0)\n"
                    "    rate = sum(v['stat'] > thr for v in rec['trials']) / len(rec['trials'])\n"))
case("module_runner_imports_reads_at_import", False,
     "import os, json\nCFG = json.loads(os.environ.get('FI_THRESHOLDS', '{}'))\n"
     "def run_trial(cell, t, s):\n    return {'x': cell['a'] * CFG.get('T', 1)}\n")
# --- holes found by reading the code ---------------------------------------------------------------------------------
case("decorator_reads_setting", True,
     "def scaled(f):\n    def g(c, *rest):\n        return {k: v * c['T'] for k, v in f(c, *rest).items()}\n    return g\n"
     "@scaled\ndef helper(c):\n    return {'x': c['a']}\n"
     "def run_trial(cell, t, s):\n    return helper(cell)\n")
case("unbound_method_offset", True,
     "class Model:\n    def f(self, setting, verbose=False):\n        return {'x': setting['a'] * setting['T']}\n"
     "def run_trial(cell, t, s):\n    m = Model()\n    return Model.f(m, cell, False)\n")
case("star_args_before_cell", True,
     "def helper(n, setting, opts=None):\n    return {'x': n * setting['T']}\n"
     "def run_trial(cell, t, s):\n    xs = [cell['a']]\n    return helper(*xs, cell)\n")
case("eval_string", True, "def run_trial(cell, t, s):\n    T = eval(\"cell['T']\")\n    return {'x': cell['a'] * T}\n")
case("exec_string", True,
     "def run_trial(cell, t, s):\n    ns = {'cell': dict(a=cell['a'])}\n    ns2 = {}\n    exec(\"T = c['T']\", {'c': locals()['cell']}, ns2)\n"
     "    return {'x': cell['a'] * ns2['T']}\n")
case("locals_lookup", True, "def run_trial(cell, t, s):\n    T = locals()['cell']['T']\n    return {'x': cell['a'] * T}\n")
case("self_method_library_base", True,
     "from sklearn.base import BaseEstimator\n"
     "class Other:\n    def predict(self, s):\n        return s['a']\n"
     "class Sim(BaseEstimator):\n    def go(self, cell):\n        return {'x': self.predict(cell)}\n"
     "    def predict(self, s):\n        return s['a'] * self.scale(s)\n    def scale(self, s):\n        return s['T']\n"
     "def run_trial(cell, t, s):\n    return Sim().go(cell)\n")
case("module_rebinds_local_function", True,
     "import functools\ndef helper(c):\n    return c['a']\nhelper = functools.partial(lambda c, f=helper: f(c) * c['T'])\n"
     "def run_trial(cell, t, s):\n    return {'x': helper(cell)}\n")
case("misspelt_axis_with_fallback", True,
     "def run_trial(cell, t, s):\n    return {'x': cell['a'] * cell.get('T_K', 300.0)}\n")
# --- control cases ---------------------------------------------------------------------------------------------------
case("control_T_unread", False, "def run_trial(cell, t, s):\n    return {'x': cell['a'] * 2}\n")
case("control_T_read_lambda", True,
     "def run_trial(cell, t, s):\n    f = lambda: cell['T']\n    return {'x': cell['a'] * f()}\n")
case("control_T_read_closure", True,
     "def run_trial(cell, t, s):\n    def rhs(y):\n        return -cell['T'] * y\n    return {'x': cell['a'] * rhs(1)}\n")
case("control_T_read_in_package_via_module", True,
     "from pkg import model\ndef run_trial(cell, t, s):\n    return model.solve(cell)\n",
     pkg____init___py="", pkg__model_py="def solve(setting):\n    return {'x': setting['a'] * setting['T']}\n")
case("control_T_in_test", False, "def run_trial(cell, t, s):\n    if 'T' in cell:\n        pass\n    return {'x': cell['a']}\n")
# --- variants where the run also reads `a` directly (so the "reads none" guard does not hide the wrong follow) --------
case("unbound_method_offset_2", True,
     "class Model:\n    def f(self, setting, verbose=False):\n        return setting['T']\n"
     "def run_trial(cell, t, s):\n    m = Model()\n    return {'x': cell['a'] * Model.f(m, cell, False)}\n")
case("star_args_before_cell_2", True,
     "def helper(n, m, setting):\n    return n * setting['T']\n"
     "def run_trial(cell, t, s):\n    pair = (1.0, 2.0)\n    return {'x': cell['a'] * helper(*pair, cell)}\n")
