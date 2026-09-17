"""Two guards over the same kind of number, tuned against the same runs.

One makes the checking layer *more* eager: a reference value that is exactly 0
at every setting of a sweep is a root finder that returned the trivial root on
its bracket endpoint, and eight real runs across three model families shipped
one — then drew it as a flat line captioned "convergence to the deterministic
limit".

The other makes it *less* eager: every catch the range gate made came with a
false alarm on a correct zero, each costing a repair round. Below the epidemic
threshold nothing takes off, so at R0 = 0.9 a zero is the answer.

They pull in opposite directions over the same values, so they are pinned
together: the fixtures below are the shapes the real runs actually emitted, and
each guard's tests double as the other's false-positive tests.
"""
from __future__ import annotations

from core import numeric_oracle as no
from core import plausibility as pl

UNIT = {"result_assertions": [{"path": "outbreak_probability", "min": 0, "max": 1},
                              {"path": "deterministic_final_size", "min": 0, "max": 1}]}


def _design(*assertions: dict) -> dict:
    return {"hypothesis": "h", "result_assertions": list(assertions)}


# ---------------------------------------------------------------------------
# T6 — a reference value that is zero at every setting
# ---------------------------------------------------------------------------

# v238's shape: the deterministic final size solved by a root finder bracketed
# on [0, 1], where f(0) == 0, at nine (R0, N) cells.
V238 = {
    "by_R0_N": {
        f"{r0}_{n}": {
            "deterministic_final_size": 0.0,
            "outbreak_probability": p,
        }
        for (r0, n), p in zip(
            [(r, n) for r in ("0.9", "1.5", "3.0") for n in ("100", "1000", "5000")],
            [0.01, 0.05, 0.2, 0.31, 0.33, 0.45, 0.66, 0.67, 0.67],
        )
    }
}


def test_a_reference_zero_at_every_setting_is_a_finding() -> None:
    """The defect itself: nine cells, one deterministic value, always 0."""
    found = no.trivial_reference_findings(V238)
    assert [f.result_path for f in found] == ["deterministic_final_size"]
    assert found[0].kind == "trivial_reference"


def test_the_finding_names_the_bracket_rather_than_the_number() -> None:
    """A writer told 'the number is 0' would write 0 down. It has to say the
    root finder is the suspect."""
    text = no.trivial_reference_findings(V238)[0].describe()
    assert "bracket" in text and "every one of its settings" in text


def test_the_finding_reaches_the_report_through_check() -> None:
    report = no.check("The deterministic limit is approached as N grows.", V238)
    assert not report.ok
    assert report.findings[0].kind == "trivial_reference"
    assert report.to_dict()["findings"][0]["kind"] == "trivial_reference"


def test_a_reference_that_varies_is_left_alone() -> None:
    """g4n3 computed it correctly: 0 below threshold, real values above."""
    ok = {
        "by_R0_N": {
            "R0_0.9_N_100": {"deterministic_final_size": 0.0},
            "R0_1.5_N_100": {"deterministic_final_size": 0.582812},
            "R0_3.0_N_100": {"deterministic_final_size": 0.94048},
            "R0_3.0_N_5000": {"deterministic_final_size": 0.94048},
        }
    }
    assert no.trivial_reference_findings(ok) == []


def test_a_flat_non_zero_reference_across_an_N_sweep_is_not_a_finding() -> None:
    """Deliberately exempt. The deterministic final size does not depend on N,
    so a correct one IS identical across a sweep over N — that flat reference
    line is what a convergence figure draws. Only an identically-*zero*
    reference is evidence of the trivial root."""
    flat = {
        "by_N": {n: {"deterministic_final_size": 0.94048,
                     "mean_final_size": m}
                 for n, m in (("100", 0.90), ("1000", 0.93), ("5000", 0.939))}
    }
    assert no.trivial_reference_findings(flat) == []


def test_a_setting_is_not_a_reference_even_when_it_is_zero() -> None:
    """cx1 reported ode_rtol = 0.0 at all nine cells. A tolerance repeated at
    every point is configuration, not a computation."""
    cfg = {
        "cells": [
            {"ode_rtol": 0.0, "ode_atol": 0.0, "ode_horizon": 500.0,
             "unconditional_final_size_fraction_mean": v}
            for v in (0.0013, 0.0063, 0.0502, 0.1884)
        ]
    }
    assert no.trivial_reference_findings(cfg) == []


def test_a_measured_quantity_is_not_a_reference() -> None:
    """An outbreak probability of 0 everywhere is a different failure, and the
    degenerate-run guard owns it. This check is about reference values."""
    measured = {
        "by_R0": {r: {"outbreak_probability": 0.0, "mean_size": m}
                  for r, m in (("0.9", 0.1), ("1.5", 0.2), ("3.0", 0.3))}
    }
    assert no.trivial_reference_findings(measured) == []


def test_two_settings_are_not_a_sweep() -> None:
    small = {"a": {"deterministic_final_size": 0.0, "p": 0.1},
             "b": {"deterministic_final_size": 0.0, "p": 0.2}}
    assert no.trivial_reference_findings(small) == []


def test_results_where_nothing_varies_are_left_to_the_degenerate_guard() -> None:
    dead = {"by_R0": {r: {"deterministic_final_size": 0.0} for r in "abc"}}
    assert no.trivial_reference_findings(dead) == []


def test_xg3_and_g4h2_spellings_are_recognised() -> None:
    """Two more real runs: a lower-case grouping key, and the same defect under
    the name `theoretical_final_size`."""
    xg3 = {f"by_r0_{r}": {f"by_n_{n}": {"deterministic_final_size": 0.0,
                                        "outbreak_probability": 0.1 * i}
                          for i, n in enumerate(("100", "500", "5000"))}
           for r in ("0.9", "1.5", "3.0")}
    g4h2 = {"by_R0": {r: {n: {"theoretical_final_size": 0.0,
                              "outbreak_probability": 0.1 * i}
                          for i, n in enumerate(("100", "1000", "5000"))}
                      for r in ("0.9", "1.5", "3.0")}}
    assert [f.result_path for f in no.trivial_reference_findings(xg3)] == [
        "deterministic_final_size"]
    assert [f.result_path for f in no.trivial_reference_findings(g4h2)] == [
        "theoretical_final_size"]


# ---------------------------------------------------------------------------
# T8 — a correct zero, and a setting counted once
# ---------------------------------------------------------------------------

def test_the_same_cell_under_two_groupings_is_one_setting() -> None:
    """tokrg5 reported its sweep twice, once by R0 and once by N. One zero
    appeared as two paths, which is exactly the count the rule needs.

    R0 here is 1.5, above the threshold, so the theory rule cannot explain
    these away — counting the cell once is the only thing that clears them.
    """
    rj = {
        "by_R0": {"1.5": {"by_N": {"5000": {"outbreak_probability": 0.0}}},
                  "3.0": {"by_N": {"5000": {"outbreak_probability": 0.44}}}},
        "by_N": {"5000": {"by_R0": {"1.5": {"outbreak_probability": 0.0},
                                    "3.0": {"outbreak_probability": 0.44}}}},
    }
    assert pl.check_design(rj, _design({"path": "outbreak_probability",
                                        "min": 0, "max": 1})) == []


def test_distinct_cells_holding_the_same_segments_are_not_merged() -> None:
    """Two different cells can carry the very same segments: R0 = 0.9 with
    N = 5000, and R0 = 5000 with N = 0.9. A key built by sorting the bare
    segments collapses them into one; pairing each name with the value that
    follows it keeps them apart."""
    a = (("k", "by_R0"), ("k", "0.9"), ("k", "by_N"), ("k", "5000"), ("k", "p"))
    b = (("k", "by_R0"), ("k", "5000"), ("k", "by_N"), ("k", "0.9"), ("k", "p"))
    assert sorted(str(v) for _, v in a[:-1]) == sorted(str(v) for _, v in b[:-1])
    assert pl._setting_key(a) != pl._setting_key(b)


def test_the_same_cell_by_either_route_has_one_key() -> None:
    a = (("k", "by_R0"), ("k", "0.9"), ("k", "by_N"), ("k", "5000"), ("k", "p"))
    b = (("k", "by_N"), ("k", "5000"), ("k", "by_R0"), ("k", "0.9"), ("k", "p"))
    assert pl._setting_key(a) == pl._setting_key(b)


def test_a_zero_below_the_epidemic_threshold_is_expected() -> None:
    """g4n3, the control. Three R0 = 0.9 cells reported 0 and were sent back to
    be 'fixed'; the model then hid the value. 0 is the correct final size below
    threshold."""
    g4n3 = {"by_R0_N": {
        "R0_0.9_N_100": {"deterministic_final_size": 0.0},
        "R0_0.9_N_1000": {"deterministic_final_size": 0.0},
        "R0_0.9_N_5000": {"deterministic_final_size": 0.0},
        "R0_1.5_N_100": {"deterministic_final_size": 0.582812},
        "R0_3.0_N_100": {"deterministic_final_size": 0.94048},
    }}
    assert pl.check_design(g4n3, UNIT) == []


def test_the_reproduction_number_is_read_from_an_enclosing_field() -> None:
    """cx1 keyed its sweep by cell index, so no path names R0 — the cell
    carries it as a field."""
    cx1 = {"cells": [
        {"R0": 0.9, "N": 1000, "major_outbreak_probability": 0.0},
        {"R0": 0.9, "N": 5000, "major_outbreak_probability": 0.0},
        {"R0": 1.5, "N": 100, "major_outbreak_probability": 0.343},
        {"R0": 3.0, "N": 100, "major_outbreak_probability": 0.667},
    ]}
    assert pl.check_design(cx1, _design(
        {"path": "major_outbreak_probability", "min": 0, "max": 1})) == []


def test_the_reproduction_number_is_read_from_a_compound_grouping_key() -> None:
    """v238's `by_R0_N.0.9_100` packs both coordinates into one segment, so the
    name and its value sit two separators apart."""
    tokens = (("k", "by_R0_N"), ("k", "0.9_100"), ("k", "outbreak_probability"))
    assert pl._r0_for(tokens, pl._dotted(tokens), {}) == 0.9


def test_a_value_of_one_below_threshold_is_excused_by_nothing() -> None:
    """tokrg6's first run returned S-infinity/N instead of the final size, so it
    reported 1.0 at R0 = 0.9. Theory excuses a zero below threshold, never a
    one — this must still go back."""
    tokrg6 = {"by_R0": {
        "0.9": {"by_N": {n: {"deterministic_final_size": 1.0}
                         for n in ("100", "1000", "5000")}},
        "1.5": {"by_N": {"100": {"deterministic_final_size": 0.582812}}},
    }}
    v = pl.check_design(tokrg6, UNIT)
    assert [(x.kind, x.value) for x in v] == [("at_bound", 1.0)]


def test_zeros_confined_to_one_level_are_a_regime_not_a_bug() -> None:
    """main8 labelled nothing: its paths read `by_N.100.0.9`, so no name says
    which coordinate is the reproduction number. The shape still does — every
    zero sits at the same level of one coordinate and the quantity is non-zero
    at the others."""
    main8 = {"by_N": {n: {r: {"deterministic_final_size": v}
                          for r, v in (("0.9", 0.0), ("1.5", 0.582812),
                                       ("3.0", 0.94048))}
                      for n in ("100", "1000", "5000")}}
    assert pl.check_design(main8, UNIT) == []


def test_zero_at_every_setting_is_still_the_trivial_answer() -> None:
    """The same run one repair earlier: 0 at all nine cells is not a regime,
    and gets no excuse."""
    main8_broken = {"by_N": {n: {r: {"deterministic_final_size": 0.0}
                                 for r in ("0.9", "1.5", "3.0")}
                             for n in ("100", "1000", "5000")}}
    v = pl.check_design(main8_broken, UNIT)
    assert [x.kind for x in v] == ["at_bound"]
    assert len(v[0].paths) == 9


def test_a_level_explains_the_zeros_only_when_settings_outside_it_differ() -> None:
    """A coordinate level is the explanation only if the quantity behaves
    differently away from it. Here the zeros sit at one level, one setting
    *inside* that level is non-zero, and the settings outside it are non-zero
    too — the level still separates, so this is a regime. If every other
    setting had to differ, a single non-zero inside the level would break it.
    """
    partial = {"by_N": {n: {r: {"deterministic_final_size": v}
                            for r, v in (("0.9", 0.0 if n != "5000" else 0.11),
                                         ("1.5", 0.582812), ("3.0", 0.94048))}
                        for n in ("100", "1000", "5000")}}
    assert pl.check_design(partial, UNIT) == []


def test_half_a_sweep_on_the_bound_is_not_a_minority() -> None:
    """The regime excuse is for a corner of a sweep, not for half of it. Here
    every zero does sit at one level of one coordinate and the quantity is
    non-zero at the other — but it is three cells out of six, which is what a
    broken branch looks like as much as a threshold."""
    half = {"by_R0": {
        "1.5": {"by_N": {n: {"outbreak_probability": 0.0}
                         for n in ("100", "1000", "5000")}},
        "3.0": {"by_N": {n: {"outbreak_probability": 0.44}
                         for n in ("100", "1000", "5000")}},
    }}
    v = pl.check_design(half, _design(
        {"path": "outbreak_probability", "min": 0, "max": 1}))
    assert [(x.kind, x.value, len(x.paths)) for x in v] == [("at_bound", 0.0, 3)]


def test_a_genuine_error_above_threshold_still_fires() -> None:
    """tokrg5's first run reseeded the RNG inside the loop, forcing every
    probability to 0 or 1. The R0 = 1.5 zeros are wrong and must survive the
    filtering that clears the R0 = 0.9 ones."""
    tokrg5 = {"by_R0": {
        "0.9": {"by_N": {n: {"outbreak_probability": 0.0}
                         for n in ("100", "1000", "5000")}},
        "1.5": {"by_N": {n: {"outbreak_probability": 0.0}
                         for n in ("100", "1000", "5000")}},
        "3.0": {"by_N": {n: {"outbreak_probability": 1.0}
                         for n in ("100", "1000", "5000")}},
    }}
    v = pl.check_design(tokrg5, _design(
        {"path": "outbreak_probability", "min": 0, "max": 1}))
    kinds = sorted((x.kind, x.value, len(x.paths)) for x in v)
    assert kinds == [("at_bound", 0.0, 3), ("at_bound", 1.0, 3)]


def test_the_violation_reports_the_zeros_it_recognised() -> None:
    """A repair that is told only 'these three are wrong' may 'fix' the ones
    that were right. What was accounted for travels with the finding."""
    mixed = {"by_R0": {
        "0.9": {"by_N": {n: {"outbreak_probability": 0.0}
                         for n in ("100", "1000", "5000")}},
        "1.5": {"by_N": {n: {"outbreak_probability": 0.0}
                         for n in ("100", "1000", "5000")}},
    }}
    v = pl.check_design(mixed, _design(
        {"path": "outbreak_probability", "min": 0, "max": 1}))
    assert len(v) == 1 and len(v[0].explained) == 3
    text = v[0].describe()
    assert "below the epidemic threshold" in text
    assert "leave those as they are" in text


# ---------------------------------------------------------------------------
# T9 — a bound the design itself derived
# ---------------------------------------------------------------------------

# terra1's shape, at the precision that is the whole point of it. The design
# derived its maximum from the sweep it had chosen — max(0, 1 - 1/R0) over
# R0 = 0.9, 1.5, 3.0 — and declared 2/3, which as a double is
# 0.6666666666666666. The script computed 1.0 - 1.0/3.0, which is
# 0.6666666666666667. Both are two thirds; one is one bit larger.
TERRA1 = {"strata": {
    f"R0={r0},N={n}": {"branching_probability": p}
    for r0, p in (("0.9", 0.0), ("1.5", 0.33333333333333337),
                  ("3.0", 0.6666666666666667))
    for n in ("100", "1000", "5000")
}}
TERRA1_DESIGN = _design({
    "path": "branching_probability", "min": 0.0, "max": 0.6666666666666666,
    "unit": "dimensionless",
    "reason": "For the specified R0 values, max(0,1-1/R0) ranges from 0 to 2/3.",
})
TERRA1_CODE = "branching_probability = max(0.0, 1.0 - 1.0 / r0)\n"


def test_the_legal_maximum_is_not_a_violation() -> None:
    """The run this comes from was told its correct answer was wrong, in a
    message that printed the same number on both sides of the word:
    'branching_probability = 0.666667 violates [0, 0.666667]'. The repair then
    nulled every boundary value — flagging them `exact_theoretical_boundary`
    as it did, so it knew — and the figure lost the reference line it was
    drawn to show."""
    assert pl.check_design(TERRA1, TERRA1_DESIGN, code=TERRA1_CODE) == []


def test_the_maximum_is_still_legal_without_the_zeros_beside_it() -> None:
    """The three values at the maximum on their own: nothing about the rest of
    the sweep is doing the work here."""
    only_max = {"strata": {k: v for k, v in TERRA1["strata"].items()
                           if "R0=3.0" in k or "R0=1.5" in k}}
    assert pl.check_design(only_max, TERRA1_DESIGN, code=TERRA1_CODE) == []


def test_a_zero_below_threshold_is_recognised_in_a_stratum_label() -> None:
    """The same run's other half. `strata.R0=0.9,N=100` spells the coordinate
    with an `=` and packs the cell into one key, so no name matched and no
    numeric segment was left for the positional rule either — three correct
    zeros below the epidemic threshold went back for repair."""
    zeros = {"strata": {k: v for k, v in TERRA1["strata"].items()
                        if "R0=0.9" in k or "R0=1.5" in k}}
    assert pl.check_design(zeros, TERRA1_DESIGN, code=TERRA1_CODE) == []


def test_a_derived_bound_at_every_setting_is_still_the_trivial_answer() -> None:
    """The exemption is for a quantity that varied and arrived at the bound,
    not for one that never left it. A loop that captured the last cell's R0
    for all nine cells reports the maximum everywhere, and that is the shape
    this gate exists to catch — whatever the bound happens to be."""
    captured = {"strata": {k: {"branching_probability": 0.6666666666666667}
                           for k in TERRA1["strata"]}}
    v = pl.check_design(captured, TERRA1_DESIGN, code=TERRA1_CODE)
    assert [(x.kind, len(x.paths)) for x in v] == [("at_bound", 9)]


def test_a_result_past_a_derived_bound_is_still_out_of_range() -> None:
    """The tolerance is for the last bit of a float, not for a wrong answer."""
    over = {"strata": {"R0=3.0,N=100": {"branching_probability": 0.7}}}
    v = pl.check_design(over, TERRA1_DESIGN, code=TERRA1_CODE)
    assert [(x.kind, x.value) for x in v] == [("out_of_range", 0.7)]


def test_a_bound_the_script_writes_down_still_carries_the_signal() -> None:
    """Why the rule is not simply 'any bound but 0 and 1'. A root finder
    bracketed on [0, 100] can return 100 without solving anything, and the
    bracket is in the script — so a design that declares max = 100 is
    declaring a number a fault can reach, and two settings on it still go
    back. The third setting is off the bound deliberately: with every setting
    on it, the carve-out above decides the case before the bracket does."""
    rj = {"by_h": {"0.1": {"crossing": 100.0}, "0.2": {"crossing": 100.0},
                   "0.4": {"crossing": 37.0}}}
    d = _design({"path": "crossing", "min": 0, "max": 100.0})
    code = "root = brentq(f, 0.0, 100.0)\n"
    assert [x.kind for x in pl.check_design(rj, d, code=code)] == ["at_bound"]
    # The same numbers with nothing in the script to explain them are the
    # design's own extreme, and are left alone.
    assert pl.check_design(rj, d, code="root = solve(f)\n") == []


def test_the_walker_agrees_with_flatten_numbers_on_paths() -> None:
    """``_matches`` compares assertion paths against these strings, so the
    token walk must produce exactly the dotted form the oracle produces."""
    sample = {"a": {"b": [1.5, {"c": 2.5}, 0.0]}, "d": 3.5,
              "e": {"0.9": {"f": -1.0}}, "g": True}
    mine = {p: v for _, p, v in pl._walk_leaves(sample)}
    theirs = dict(no.flatten_numbers(sample, keep_zero=True))
    assert mine == theirs


def test_a_scalar_root_keeps_its_name() -> None:
    assert [p for _, p, _ in pl._walk_leaves(4.2)] == ["<root>"]
