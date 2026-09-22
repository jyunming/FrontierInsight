"""PR-5: the red-team campaign — proving, in code, that the checks the re-audit examined actually catch what they
claim to catch, or that they still don't.

This is deliberately NOT a new corpus. The re-audit's four study shapes (SIR / deterministic / paired / clustered)
already have real fault-injection coverage spread across the suite, built as each gate landed:

* frozen protocol tampered mid-quest -- ``tests/test_frozen_protocol.py`` (a full real-graph run: tamper, pause,
  resume without approval pauses again, approve, resume completes on the RESTORED content).
* ``run_manifest`` swept an axis or a value the protocol never listed -- ``tests/test_run_manifest.py`` (27 cases).
* a script rebuilt its RNG stream per setting, or drifted its grid from the frozen protocol -- real fr1/fr2/fr3
  scripts in ``tests/fixtures/protocol_runs/``, exercised by ``tests/test_protocol_check.py`` and
  ``tests/test_precision_rng.py``.
* a paired design fed mismatched trial counts, a cluster design fed no clusters array at all, or a paired-over-
  clusters design (unsupported) -- ``tests/test_metric_spec.py::test_a_cluster_design_uses_the_clusters_and_says_what_it_lacks``.
* a metric spec named an estimator with no ``estimand``/``unit`` -- ``tests/test_metric_spec.py`` (PR #362).

``docs/audits/15_fault_injection_campaign.md`` is the table citing all of the above plus what's below, per shape and
per fault, with what still isn't caught -- that table is the actual PR-5 deliverable; this file is the NEW coverage
it found wasn't there yet:

* a script that NAMES ``FI_REPLICATE_SEED`` (so the source-scan heuristic can't rule it out) but never actually uses
  the value is still called deterministic (below) -- the ``_script_reads_replicate_seed`` docstring already names
  this as an accepted blind spot of a source-scan heuristic; this locks that prose claim to a reproducible fact.
* the design self-critique prompt's report cap (P1-4, still open) -- an ``xfail`` so closing it is forced to show up
  here as an XPASS, not silently.
* a cluster array that IS present but the wrong length -- a different fault than the missing-array case above, with
  its own code path in ``_pooled`` that nothing had exercised.
* ``design_is_stochastic`` trusts the design's own wording, not the script it describes (P1-6, still open) -- a
  characterization test of the trust boundary, not an xfail: nothing here promises a specific fix shape.

P0-5 (a fresh real quest rerun since PR #356/#359/#362 landed) is a separate, unmeasured item -- this file proves the
checks catch synthetic faults, not that a real quest run under them looks any different than before.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine
from core.split_run import design_is_stochastic
from core import metric_spec as ms

AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"


def _er(stdout: str, returncode: int = 0):
    return type("ER", (), {
        "returncode": returncode, "duration_s": 0.1, "timed_out": False,
        "stdout": stdout, "stderr": "",
    })()


def _rj(payload: str) -> str:
    return f"RESULT_JSON: {payload}\n"


def _engine(tmp_path: Path, *, replicates: int, script: str) -> Engine:
    cfg = Config(
        topic="red team", title="rt", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False, clarify_mode="off", execute_replicates=replicates, pilot_run=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60),
        knowledge=KnowledgeConfig(enabled=False), output=OutputConfig(output_dir=tmp_path / "out"),
    )
    eng = Engine(cfg)
    eng.executor.install = AsyncMock(return_value=type("IR", (), {"returncode": 0, "stderr": ""})())  # type: ignore[method-assign]
    eng.quest_root.mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code").mkdir(parents=True, exist_ok=True)
    (eng.quest_root / "code" / "experiment.py").write_text(script, encoding="utf-8")
    return eng


# --- P1-5's realistic shape: the source-scan heuristic's own documented blind spot -------------------------------

# Contains the literal name FI_REPLICATE_SEED (so `_script_reads_replicate_seed` cannot rule it out), reads it,
# and never uses the value: the RNG is seeded from a constant instead. The one difference from TERRA_SHAPED_SCRIPT
# in tests/test_replicate_gating.py is this no-op read -- exactly the case that function's own docstring names as
# what it cannot catch ("a script that NAMES the variable without obeying it... falls through to the runtime
# comparison, which can only call it deterministic").
SEED_NAMED_BUT_DISCARDED_SCRIPT = (
    "import os\n"
    "import numpy as np\n"
    "_unused = os.environ.get('FI_REPLICATE_SEED')  # named, read, never used\n"
    "RNG_SEED = 20260917\n"
    "def main() -> None:\n"
    "    rng = np.random.default_rng(RNG_SEED)\n"
    "    print(rng.random())\n"
)


@pytest.mark.asyncio
async def test_a_script_that_names_but_discards_the_replicate_seed_is_still_called_deterministic(tmp_path: Path) -> None:
    """The blind spot is real, not just prose. A script whose seed is hardcoded is indistinguishable, by a source
    scan, from one that genuinely computes the same answer every time once it merely mentions the variable name
    somewhere -- a comment, an unused read, anything the substring search cannot tell from real use."""
    eng = _engine(tmp_path, replicates=5, script=SEED_NAMED_BUT_DISCARDED_SCRIPT)
    eng.executor.execute = AsyncMock(return_value=_er(_rj('{"rmse": 0.25}')))  # type: ignore[method-assign]
    patch = await eng._node_execute({"deps": []})

    assert eng.executor.execute.await_count == 2, "one extra run settles it, same as the honest case"
    assert patch["result_json_deterministic"] is True, "the documented blind spot: named-but-unused reads as genuine"
    assert patch["result_json_replicate_seed_ignored"] is False
    # Downstream consequence: no error bars are reported, exactly as if this were a legitimately deterministic ODE.
    assert len(patch["result_json_replicates"]) == 2


# --- P1-4: the self-critique checklist's report cap has no code behind it -----------------------------------------


@pytest.mark.xfail(strict=True, reason="P1-4: the prompt caps reported findings below its own mandatory checklist size")
def test_design_self_critique_caps_reported_findings_below_its_own_mandatory_checklist_size() -> None:
    """The prompt's own text asks for at most 5 reported findings against a 12-item MANDATORY checklist ("if any
    apply, you MUST patch them") -- so an applicable finding can go unpatched with nothing in the design, the
    record, or a test noticing. ``strict=True``: the day the cap phrase is gone (raised to match the checklist, or
    removed so every applicable item is reported) this assertion starts passing, which XPASSes and fails the suite
    -- forcing the fix to be noticed here, not just in the prompt diff."""
    text = (AGENTS_DIR / "design_self_critique.md").read_text(encoding="utf-8")
    checklist_items = re.findall(r"^\d+\.\s+\*\*", text, flags=re.MULTILINE)
    assert len(checklist_items) >= 10, "the checklist section moved or was renamed; re-read before trusting this test"
    cap = re.search(r"at most (\d+)", text)
    # `cap is None` (the phrase removed outright) must count as FIXED, same as a cap raised to cover every item --
    # not as "the pattern this test looks for is gone, so nothing to report" (an `assert cap is not None` here would
    # itself raise, which a strict xfail cannot tell apart from the real finding, so the fix would go unnoticed).
    capped = cap is not None and int(cap.group(1)) < len(checklist_items)
    assert not capped, (
        f"prompt caps reported findings at {cap.group(1) if cap else '?'} against a {len(checklist_items)}-item MANDATORY checklist"
    )


# --- a cluster array present but the wrong length: a different fault than a missing one ---------------------------


def test_a_cluster_array_of_the_wrong_length_is_refused_not_silently_reshaped() -> None:
    """tests/test_metric_spec.py's `no_clusters` case tests a `_clusters` array that is entirely ABSENT. Nothing
    exercised the length-mismatch branch of `_pooled` (a `_clusters` array that IS present but doesn't line up with
    `_values`) until now -- the two are different faults with different code paths, and the coverage table
    (docs/audits/15_fault_injection_campaign.md) previously cited the wrong one for this row."""
    reps = [
        {"_seed": 0, "m_values": [1.0, 0.0, 1.0, 0.0], "m_clusters": [0, 0, 1]},  # one id short of four values
        {"_seed": 1, "m_values": [1.0, 0.0, 1.0, 0.0], "m_clusters": [0, 0, 1, 1]},
    ]
    spec = {"id": "m", "kind": "proportion", "cluster": True, "estimand": "P(x)", "unit": "trial"}
    out = ms.statistics(reps, {"metrics": [spec]})
    assert out["estimates"] == {}
    assert "is not the same length as" in out["unsupported"][0]["reason"]


# --- P1-6: the design's own wording, not the script it describes, decides run_manifest applicability --------------


def test_design_is_stochastic_trusts_the_designs_own_wording_not_the_script_it_describes() -> None:
    """A design worth ``not_applicable`` (no run_manifest gap raised, per core/evidence.py) needs only a wording
    that avoids every trigger word in _STOCHASTIC_RE and a `runs_per_setting` under 2 -- nothing here reads the
    script the design will actually produce. Bootstrap resampling, a permutation test and a Poisson-process
    simulation are all genuinely stochastic and all miss the regex (which requires "random walk(s)", not bare
    "random"; "stochastic"; "monte carlo"; "gillespie"; "mcmc"; "agent-based"; "kinetic monte"; "langevin"; or
    "brownian"). This is a characterization of today's trust boundary, not an xfail: nothing here specifies what a
    fix should check instead (the function has no access to the script's own source, only the design's words)."""
    design = {
        "hypothesis": "The estimator's sampling distribution is characterized by resampling the observed data.",
        "method": "Bootstrap resampling with a fixed number of resamples estimates the standard error.",
        "expected_outcome": "The resampled interval brackets the point estimate.",
        "variables": {"resamples": 2000},
        "protocol": {"runs_per_setting": 1},
    }
    assert design_is_stochastic(design) is False, "genuinely stochastic (bootstrap draws random indices), missed by wording alone"
