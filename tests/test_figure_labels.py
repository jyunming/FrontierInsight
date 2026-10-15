"""Axis labels that show the code's names (core/figure_labels.py), the plan's reader-facing labels (core/plan.py), and
the one redraw that renames them (core/engine.py)."""
from __future__ import annotations

import string
from pathlib import Path

from core import figure_labels
from core.engine import _figure_overlap_directive, _figure_overlap_findings
from core.plan import normalize_protocol, plan_labels, plan_names

# The s1 quests' plans, as far as the figures need them.
KINETICS = {
    "grid": {"sigma": [0.005, 0.01], "n_points": [5, 12], "temperature_K": [300], "t_max_over_k": [3]},
    "metrics": [{"id": "Ea_hat", "kind": "mean", "estimand": "E_a", "unit": "run",
                 "label": "Activation energy $E_a$ (kJ/mol)"}],
    "labels": {"lnA": "Log pre-exponential factor $\\ln A$", "sigma": "Noise SD $\\sigma$ (units of $c_0$)"},
}
FDR = {"grid": {"m": [20, 200], "pi1": [0.0, 0.05]}, "in_each_run": {"method": ["bh", "bonferroni"]}}


def test_the_plan_gives_a_label_per_input_and_output_and_old_plans_give_none() -> None:
    assert plan_labels(KINETICS) == {
        "lnA": "Log pre-exponential factor $\\ln A$", "sigma": "Noise SD $\\sigma$ (units of $c_0$)",
        "Ea_hat": "Activation energy $E_a$ (kJ/mol)",
    }
    assert plan_labels(FDR) == {} and plan_labels(None) == {} and plan_labels({"labels": "x"}) == {}
    assert plan_labels({"labels": {"a": 3, "b": " two  words "}}) == {"b": "two words"}
    assert plan_names(FDR) == {"m", "pi1", "method"}
    assert {"sigma", "n_points", "Ea_hat", "lnA"} <= plan_names(KINETICS)


def test_labels_survive_the_protocol_check() -> None:
    fixed, why = normalize_protocol(KINETICS)
    assert why is None and plan_labels(fixed) == plan_labels(KINETICS)


def test_what_counts_as_a_code_name() -> None:
    names = plan_names(KINETICS) | plan_names(FDR)
    assert figure_labels.code_names("Ea_hat (kJ/mol)", names) == ["Ea_hat"]
    assert figure_labels.code_names("Ea_hat, mean ± SD (kJ/mol)", names) == ["Ea_hat"]
    assert figure_labels.code_names("lnA_hat", names) == ["lnA_hat"]
    assert figure_labels.code_names("fraction of real effects, pi1", names) == ["pi1"]
    assert figure_labels.code_names("sigma", names) == ["sigma"]
    assert figure_labels.code_names("m", names) == ["m"]
    # Words a reader uses, a symbol in math, and letters with digits the plan does not name.
    assert figure_labels.code_names("number of tests, m", names) == []
    assert figure_labels.code_names("noise SD sigma (units of c0)", names) == []
    assert figure_labels.code_names("Activation energy $E_a$ (kJ/mol)", names) == []
    assert figure_labels.code_names("CO2 uptake (mol)", names) == []
    assert figure_labels.code_names("$\\hat{E}_a$ (kJ/mol)", names) == []


def _state(records: dict, protocol: dict | None) -> dict:
    return {"figures": list(records), "figure_records": records, "design": {"protocol": protocol} if protocol else {}}


SCATTER = {"axes": [{"title": "", "xlabel": "Ea_hat (kJ/mol)", "ylabel": "lnA_hat", "series": []}]}
FDR_FIG = {"axes": [{"title": f"m = {m}", "xlabel": "fraction of real effects, pi1",
                     "ylabel": "realised FDR (mean V/max(R,1))", "series": []} for m in (20, 200)]}


def test_each_code_name_is_one_sentence_with_the_plans_label_when_it_has_one() -> None:
    found = _figure_overlap_findings(_state({"fig3.png": SCATTER}, KINETICS))
    assert found == [
        'figures/fig3.png: the x axis label "Ea_hat (kJ/mol)" shows the code name "Ea_hat"; rename it to '
        '"Activation energy $E_a$ (kJ/mol)" (the plan\'s name for it)',
        'figures/fig3.png: the y axis label "lnA_hat" shows the code name "lnA_hat"; rename it to '
        '"Log pre-exponential factor $\\ln A$" (the plan\'s name for it)',
    ]
    found = _figure_overlap_findings(_state({"fdr_vs_pi1.png": FDR_FIG}, FDR))
    assert found == [
        f'figures/fdr_vs_pi1.png: in the panel "m = {m}", the x axis label "fraction of real effects, pi1" shows the '
        "code name \"pi1\"; rename it to the reader's name for it: words, its symbol in $...$ and its unit"
        for m in (20, 200)
    ]


def test_a_label_already_the_plans_or_an_old_plan_without_names_is_left_alone() -> None:
    good = {"axes": [{"xlabel": "Activation energy $E_a$ (kJ/mol)", "ylabel": "count"}]}
    assert _figure_overlap_findings(_state({"a.png": good}, KINETICS)) == []
    # No plan: only the shape of a code name counts.
    assert len(_figure_overlap_findings(_state({"fig3.png": SCATTER}, None))) == 2
    assert _figure_overlap_findings(_state({"fdr_vs_pi1.png": FDR_FIG}, None)) == []
    # After the one redraw nothing more is asked.
    assert _figure_overlap_findings({**_state({"fig3.png": SCATTER}, KINETICS), "figure_overlap_repaired": True}) == []


def test_the_redraw_request_allows_renaming_those_labels_and_nothing_else() -> None:
    labels = _figure_overlap_findings(_state({"fig3.png": SCATTER}, KINETICS))
    text = _figure_overlap_directive(labels)
    assert "label axes with the names the code gives its variables" in text
    assert "Rename each axis label named above" in text
    assert "only the text of the axis labels named above" in text
    assert "For a legend over the data" not in text
    layout = ['figures/a.png: the legend (x) is drawn over the lines or points it labels']
    both = _figure_overlap_directive(layout + labels)
    assert "drawn over what a reader needs" in both and "Rename each axis label" in both
    assert "only positions, sizes and the number of legend columns, the text of the axis labels named above" in both
    alone = _figure_overlap_directive(layout)
    assert "Rename" not in alone and "only positions, sizes and the number of legend columns." in alone


def test_the_code_writer_is_asked_for_the_plans_labels() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("implement.md", "implement_body.md"):
        text = (root / "agents" / name).read_text(encoding="utf-8")
        filled = string.Template(text).safe_substitute()
        assert "never with a variable name from the code" in filled
        assert "`Activation energy $E_a$ (kJ/mol)`" in filled
