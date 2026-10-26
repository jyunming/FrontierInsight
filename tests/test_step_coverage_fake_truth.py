"""A ``run_trial`` that returns the plan's true value plus noise, with a standard error that matches the noise, and never
calls the model's package, passes FI's failure-rate, calibration, seed and typed-result checks. Step coverage
(core/step_coverage.py) is what tells it from an honest one: it is recorded as a gap in the evidence record and QUALITY.md.
All names here are made up."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core import quality_report as qr
from core import step_coverage as sc
from core.run_checks import RunChecksMixin

PACKAGE_MODEL = '''
def step_response(gain, t):
    return gain * (1.0 - 2.718281828 ** (-t))


def fit_gain(samples):
    return sum(samples) / len(samples)
'''

#: Returns the true value plus noise, and an SE that matches the noise. The package is imported and never called.
FAKE_TRUTH = '''
import random

from gain_model import model


def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    truth = cell["true_gain"]
    noise = cell["sigma"]
    estimate = truth + rng.gauss(0.0, noise)
    return {"gain_hat": estimate, "gain_se": noise, "settle_time": 4.0 + rng.gauss(0.0, 0.1)}
'''

#: The same trial computed through the package: the estimate is fitted from samples the model's own function made.
HONEST = '''
import random

from gain_model import model


def _samples(cell, rng):
    return [model.step_response(cell["true_gain"], 5.0) + rng.gauss(0.0, cell["sigma"]) for _ in range(cell["n"])]


def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    samples = _samples(cell, rng)
    estimate = model.fit_gain(samples)
    return {"gain_hat": estimate, "gain_se": cell["sigma"] / len(samples) ** 0.5, "settle_time": model.step_response(1.0, 4.0)}
'''

METRICS = [{"id": "gain_hat"}, {"id": "gain_se"}, {"id": "settle_time"}]
PACKAGE = ["step_response", "fit_gain"]


def _caught(warnings: list[str]) -> set[str]:
    return {w.split("`")[1] for w in warnings}


def test_truth_plus_noise_that_never_calls_the_package_is_caught() -> None:
    """The old code (nothing read this) passed it; every metric of the fake trial is now named as not computed through
    the model's functions."""
    warnings = sc.check(FAKE_TRUTH, PACKAGE, METRICS)
    assert _caught(warnings) == {"gain_hat", "gain_se", "settle_time"}
    assert all("without calling any function of the model's package" in w for w in warnings)


def test_a_trial_computed_through_the_package_passes() -> None:
    """Known positive: the honest trial calls the package (through a helper of its own for the samples), so nothing is said."""
    assert sc.check(HONEST, PACKAGE, METRICS) == []


def test_package_function_names_are_read_from_the_package_text() -> None:
    assert sc.package_function_names({"gain_model/model.py": PACKAGE_MODEL, "gain_model/bad.py": "def ("}) == [
        "fit_gain", "step_response"]


def test_a_metric_only_the_analysis_computes_is_left_alone_when_only_returned_is_asked() -> None:
    """The analysis script may compute a metric from the trials; the engine holds only what the trial returns."""
    metrics = [*METRICS, {"id": "gain_ratio"}]
    assert "gain_ratio" in _caught(sc.check(HONEST, PACKAGE, metrics))
    assert sc.check(HONEST, PACKAGE, metrics, only_returned=True) == []
    assert _caught(sc.check(FAKE_TRUTH, PACKAGE, metrics, only_returned=True)) == {"gain_hat", "gain_se", "settle_time"}


class _Checks(RunChecksMixin):
    """Only what ``RunChecksMixin._record_step_coverage`` reads of an engine."""

    def __init__(self, root: Path, protocol: dict[str, Any]) -> None:
        self.quest_root = root
        self.fi_dir = root / ".fi"
        self._log = logging.getLogger("step_coverage_test")
        self._protocol = protocol

    def _protocol_block(self, state: Any) -> dict[str, Any]:
        return self._protocol


def _quest(tmp_path: Path, simulate: str) -> Path:
    code = tmp_path / "code"
    (code / "gain_model").mkdir(parents=True)
    (code / "gain_model" / "__init__.py").write_text("", encoding="utf-8")
    (code / "gain_model" / "model.py").write_text(PACKAGE_MODEL, encoding="utf-8")
    (code / "simulate.py").write_text(simulate, encoding="utf-8")
    return tmp_path


def test_the_engine_records_the_fake_truth_as_a_gap_and_the_writer_is_told(tmp_path: Path) -> None:
    """The old code wrote no record at all, so neither the evidence record nor QUALITY.md could name the fake trial."""
    root = _quest(tmp_path, FAKE_TRUTH)
    checks = _Checks(root, {"metrics": METRICS})
    warnings = checks._record_step_coverage({}, root / "code" / "simulate.py")
    assert _caught(warnings) == {"gain_hat", "gain_se", "settle_time"}
    record = json.loads((root / ".fi" / sc.RECORD).read_text(encoding="utf-8"))
    assert record["warnings"] == warnings
    assert "not shown to come from the model's equations" in checks._step_coverage_note()
    quality = qr._step_coverage_summary(SimpleNamespace(step_coverage=record))
    assert quality is not None and len(quality["warnings"]) == 3


def test_the_engine_records_nothing_for_an_honest_trial_and_clears_an_earlier_record(tmp_path: Path) -> None:
    """Known positive: an honest trial leaves no warning, and the record of an earlier (faked) version of the code is not
    left standing against the new one."""
    root = _quest(tmp_path, FAKE_TRUTH)
    checks = _Checks(root, {"metrics": METRICS})
    simulate = root / "code" / "simulate.py"
    assert checks._record_step_coverage({}, simulate)
    simulate.write_text(HONEST, encoding="utf-8")
    assert checks._record_step_coverage({}, simulate) == []
    assert json.loads((root / ".fi" / sc.RECORD).read_text(encoding="utf-8"))["warnings"] == []
    assert checks._step_coverage_note() == ""
    assert qr._step_coverage_summary(SimpleNamespace(step_coverage={"warnings": []})) is None


def test_a_quest_without_a_package_is_not_judged(tmp_path: Path) -> None:
    """Two scripts alone have no package to compute through: nothing is said and no record is left."""
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "simulate.py").write_text(FAKE_TRUTH, encoding="utf-8")
    checks = _Checks(tmp_path, {"metrics": METRICS})
    assert checks._record_step_coverage({}, tmp_path / "code" / "simulate.py") == []
    assert not (tmp_path / ".fi" / sc.RECORD).exists()


def test_the_evidence_record_names_the_gap(tmp_path: Path) -> None:
    """The gap is in the evidence record's list of what stands between the result and publication; an honest quest has none."""
    from core import evidence

    (tmp_path / ".fi").mkdir()
    assert "model's functions" not in json.dumps(evidence.assess(tmp_path, {}))
    (tmp_path / ".fi" / sc.RECORD).write_text(
        json.dumps({"warnings": ["the metric `gain_hat` is computed without calling any function of the model's package"]}),
        encoding="utf-8")
    assert "not shown to be computed through the model's functions" in json.dumps(evidence.assess(tmp_path, {}))


#: The package has `fit_gain`; the simulation defines its own unrelated `fit_gain` and calls that one.
SAME_NAME_LOCAL = '''
import random

from gain_model import model


def fit_gain(samples):
    return 3.0 + samples[0] * 0.0


def run_trial(cell, trial_id, seed):
    rng = random.Random(seed)
    estimate = fit_gain([cell["true_gain"] + rng.gauss(0.0, cell["sigma"])])
    return {"gain_hat": estimate}
'''


def test_a_local_function_with_a_package_function_name_is_not_the_package() -> None:
    """The old code compared bare names, so a local `fit_gain` the trial calls read as the package's `fit_gain` and the fake
    passed. A call counts only when an import binds it to the package."""
    assert sc.check(SAME_NAME_LOCAL, PACKAGE, [{"id": "gain_hat"}]) == []   # the looser name-only reading is fooled
    warnings = sc.check(SAME_NAME_LOCAL, PACKAGE, [{"id": "gain_hat"}], package_modules=["gain_model"])
    assert _caught(warnings) == {"gain_hat"}


def _trial(imports: str, call: str, extra: str = "") -> str:
    return "\n".join([imports, "", extra, "def run_trial(c, t, s):", f"    return {{'a': {call}}}", ""])


def test_calls_bound_to_the_package_by_each_import_form_count() -> None:
    """Known positive: `from pkg.mod import f`, `import pkg.mod as m`, `from pkg import mod` and `import pkg` all bind to the
    package; a local name that shadows the import, or an alias of another library, does not."""
    modules = ["gain_model"]
    bound = [
        _trial("from gain_model.model import fit_gain as fg", "fg([1.0])"),
        _trial("import gain_model.model as m", "m.fit_gain([1.0])"),
        _trial("from gain_model import model", "model.fit_gain([1.0])"),
        _trial("import gain_model", "gain_model.model.fit_gain([1.0])"),
    ]
    for source in bound:
        assert sc.check(source, PACKAGE, ["a"], package_modules=modules) == [], source
    shadowed = _trial("from gain_model.model import fit_gain", "fit_gain([1.0])",
                      "def fit_gain(x):\n    return 1.0\n\n")
    assert _caught(sc.check(shadowed, PACKAGE, ["a"], package_modules=modules)) == {"a"}
    unrelated = _trial("import numpy as gm", "gm.fit_gain([1.0])")
    assert _caught(sc.check(unrelated, PACKAGE, ["a"], package_modules=modules)) == {"a"}


def test_a_name_bound_inside_the_trial_shadows_the_package_there() -> None:
    """The first binding fix read only module-level definitions, so a `fit_gain` defined inside `run_trial` (or a parameter
    of that name) was still taken for the package's and the fake passed."""
    modules = ["gain_model"]
    imports = "from gain_model.model import fit_gain"
    nested_def = (imports + "\n\ndef run_trial(c, t, s):\n    def fit_gain(x):\n        return 3.0\n"
                  "    return {'a': fit_gain([1.0])}\n")
    parameter = (imports + "\n\ndef run_trial(c, t, s, fit_gain=len):\n    return {'a': fit_gain([1.0])}\n")
    assigned = (imports + "\n\ndef run_trial(c, t, s):\n    fit_gain = lambda x: 3.0\n    return {'a': fit_gain([1.0])}\n")
    nested_scope = (imports + "\n\ndef run_trial(c, t, s):\n    def fit_gain(x):\n        return 3.0\n"
                    "    def inner():\n        return fit_gain([1.0])\n    return {'a': inner()}\n")
    for source in (nested_def, parameter, assigned, nested_scope):
        assert _caught(sc.check(source, PACKAGE, ["a"], package_modules=modules)) == {"a"}, source


def test_a_shadow_in_another_function_does_not_hide_the_package_call() -> None:
    """Known positive: a local `fit_gain` in an unrelated helper leaves the trial's own call bound to the package."""
    source = ("from gain_model.model import fit_gain\n\ndef helper():\n    def fit_gain(x):\n        return 0.0\n    return fit_gain\n\n"
              "def run_trial(c, t, s):\n    return {'a': fit_gain([1.0])}\n")
    assert sc.check(source, PACKAGE, ["a"], package_modules=["gain_model"]) == []
