"""A shown fault whose test errors in its own code is asked for once more, as an assertion about values (core/finding_repair.py
``ask_again``, ``RunChecksMixin._ask_finding_test_again``, called by ``Engine._shown_fault_check``).

Replay of r7 (EUV, a dose-to-size typed into the analysis: ``nominal = 60.0 if platform == "CAR" else 40.0`` and an
"interval" of ``[nominal, nominal]``). Its fault F-bed59bff was shown by a test, and on the code that ran its test
(``F-bed59bff_2.py``, kept as the quest wrote it) raises ``AttributeError: 'function' object has no attribute 'co_names'``
in its own code. The old gate counted that error as the fault and stopped; the gate that only takes an assertion lost the
stop. Now FI asks the code model once more, shown the error. The reply below stands in for the model's answer (no model is
called): a test that runs the analysis on the study's own trials and asserts on the interval it reports. It fails by its
assertion, so the quest stops as before, on a test that shows the fault by its values.

Fixture tests/fixtures/replay/finding_v2/r7_final/: ``code/experiment.py`` and the kept test as the quest left them,
``raw/trials.json`` cut to the first 40 trials of each of its 40 settings (``non_finite`` counted again); the code-review
record and the results are tests/fixtures/shown_fault_names/euv_r7/."""

from __future__ import annotations



