"""The reference values a plan would declare for the two s1 replay quests (tests/fixtures/reference_values), written the
way core/reference_values.py reads them. Shared by the reference-value and provenance tests; not a test module."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures" / "reference_values"

#: m0 = m - round(pi1 * m), the number of true nulls (pi1 * m is a whole number in every setting of this grid).
M0 = "(m - floor(pi1*m + 0.5))"

FDR_REFERENCES: dict[str, dict[str, Any]] = {
    # BH controls the FDR at pi0 * alpha for independent tests (a limit, not a value).
    "fdr_bh": {"formula": f"{M0}/m*alpha", "kind": "bound", "checked_by": "bh_fdr_upper_bound",
               "note": "the Benjamini-Hochberg bound pi0 x alpha"},
    # Any false discovery among m0 independent nulls each rejected with probability alpha (or alpha/m).
    "any_false_discovery_uncorrected": {"formula": f"1 - (1 - alpha)**{M0}", "checked_by": "uncorrected_fwer_null"},
    "any_false_discovery_bonferroni": {"formula": f"1 - (1 - alpha/m)**{M0}"},
    # With every hypothesis null, BH rejects anything with probability alpha.
    "any_false_discovery_bh": {"formula": "alpha", "when": {"pi1": 0.0}, "checked_by": "bh_fwer_all_null"},
    # Power of a one-sided z test of a shift delta at level alpha/m (Bonferroni) or alpha (uncorrected).
    "power_bonferroni": {"formula": "1 - ndtr(ndtri(1 - alpha/m) - delta)", "when": [{"pi1": 0.05}, {"pi1": 0.2}]},
    "power_uncorrected": {"formula": "1 - ndtr(ndtri(1 - alpha) - delta)", "when": [{"pi1": 0.05}, {"pi1": 0.2}]},
}
#: Uncorrected power is not one of the quest's own metrics; a plan declaring its reference declares the metric too.
POWER_UNCORRECTED_METRIC = {"id": "power_uncorrected", "estimand": "E[S/m1] uncorrected, pi1>0", "kind": "mean",
                            "unit": "a repeat", "cluster": None, "paired": True, "family": "power by method"}

KIN_REFERENCES: dict[str, dict[str, Any]] = {
    "coverage_Ea": {"formula": "reference_coverage", "note": "the nominal coverage of a 95% interval"},
    "coverage_lnA": {"formula": "reference_coverage", "note": "the nominal coverage of a 95% interval"},
    "Ea_mean": {"value": "Ea_hat", "formula": "Ea_true_J_per_mol", "checked_by": "noise_free_Ea",
                "note": "the true activation energy"},
    "lnA_mean": {"value": "lnA_hat", "formula": "lnA_true", "checked_by": "noise_free_lnA", "note": "the true ln A"},
}


def load(name: str, file: str) -> Any:
    return json.loads((FIXTURES / name / file).read_text(encoding="utf-8"))


def with_references(protocol: dict[str, Any], references: dict[str, dict[str, Any]],
                    extra_metrics: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    out = copy.deepcopy(protocol)
    out["metrics"] = list(out["metrics"]) + [dict(m) for m in extra_metrics or []]
    for m in out["metrics"]:
        if m["id"] in references:
            m["reference"] = dict(references[m["id"]])
    return out


def fdr_protocol() -> dict[str, Any]:
    return with_references(load("fdr_s1", "protocol.json"), FDR_REFERENCES, [POWER_UNCORRECTED_METRIC])


def kin_protocol() -> dict[str, Any]:
    return with_references(load("kin_s1", "protocol.json"), KIN_REFERENCES)
