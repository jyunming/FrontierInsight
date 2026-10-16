"""Figure text that shows the code's names (core/figure_labels.py), the plan's reader-facing labels (core/plan.py), and
the redraw from the saved numbers that renames them (core/engine.py): the experiment is not run again and no repair
attempt is spent."""
from __future__ import annotations

import asyncio
import string
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core import figure_labels
from core.config import Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig, ProviderConfig
from core.engine import Engine, _figure_label_findings, _figure_overlap_findings
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
    names = plan_names(KINETICS) | plan_names(FDR) | {"lnA_hat", "p_value"}
    assert figure_labels.code_names("Ea_hat (kJ/mol)", names) == ["Ea_hat"]
    assert figure_labels.code_names("Ea_hat, mean ± SD (kJ/mol)", names) == ["Ea_hat"]
    assert figure_labels.code_names("lnA_hat", names) == ["lnA_hat"]
    assert figure_labels.code_names("fraction of real effects, pi1", names) == ["pi1"]
    assert figure_labels.code_names("pi1 = 0.05", names) == ["pi1"]
    assert figure_labels.code_names("p_value", names) == ["p_value"]
    assert figure_labels.code_names("n_points per run", names) == ["n_points"]
    # Words a reader uses, a symbol in math, a short name, and letters with digits the plan does not name.
    assert figure_labels.code_names("number of tests, m", names) == []
    assert figure_labels.code_names("m", names) == []
    assert figure_labels.code_names("sigma", names) == []
    assert figure_labels.code_names("noise SD sigma (units of c0)", names) == []
    assert figure_labels.code_names("Activation energy $E_a$ (kJ/mol)", names) == []
    assert figure_labels.code_names("CO2 uptake (mol)", names) == []
    assert figure_labels.code_names("$\\hat{E}_a$ (kJ/mol)", names) == []
    # The shape alone is not enough: a name neither the plan nor the code uses is left alone.
    assert figure_labels.code_names("Ea_hat (kJ/mol)", set()) == []


# Symbols a scientist writes (the review's false alarms). Each is flagged by nothing even when the plan names the
# symbol AND the experiment's code assigns it.
SYMBOLS = [
    ("k_B T (J)", "k_B"), ("T_c (K)", "T_c"), ("log_10(k)", "log_10"), ("ΔG_f (kJ/mol)", "G_f"),
    ("E_a (kJ/mol)", "E_a"), ("V_max (m/s)", "V_max"), ("T (K)", "T"), ("N (runs)", "N"), ("dt (s)", "dt"),
    ("c0 (mol/L)", "c0"), ("initial concentration c0 (mol/L)", "c0"), ("Re_lambda", "Re_lambda"),
]


@pytest.mark.parametrize(("label", "name"), SYMBOLS)
def test_a_symbol_a_scientist_writes_is_not_taken_for_code(label: str, name: str) -> None:
    code = figure_labels.code_identifiers([f"{name} = 1.0\nresults = {{'{name}': {name}}}\n"])
    assert name in code
    record = {"axes": [{"title": label, "xlabel": label, "ylabel": label, "series": [{"label": label}]}]}
    assert figure_labels.findings({"a.png": record}, ["a.png"], {}, {name}, code) == []


def test_the_code_vouches_for_its_names_but_not_for_a_label_it_only_draws() -> None:
    code = figure_labels.code_identifiers([
        "import json\nEa_hat = fit()\nout = {'lnA_hat': 1}\nax.set_xlabel('t_max_over_k')\n"
        "def run(n_points, *, p_value=0.05):\n    return n_points\n",
        "this is not python (",
    ])
    assert {"Ea_hat", "lnA_hat", "n_points", "p_value", "run"} <= code
    assert "t_max_over_k" not in code  # only drawn as text: a label never vouches for itself


def _state(records: dict, protocol: dict | None) -> dict:
    return {"figures": list(records), "figure_records": records, "design": {"protocol": protocol} if protocol else {}}


SCATTER = {"axes": [{"title": "", "xlabel": "Ea_hat (kJ/mol)", "ylabel": "lnA_hat", "series": []}]}
FDR_FIG = {"axes": [{"title": f"m = {m}", "xlabel": "fraction of real effects, pi1",
                     "ylabel": "realised FDR (mean V/max(R,1))", "series": []} for m in (20, 200)]}
FDR_LEGEND = {"axes": [{"title": "pi1 = 0.05", "xlabel": "number of tests", "ylabel": "realised FDR",
                        "series": [{"label": "pi1 = 0.05"}, {"label": "BH"}, {"label": "_child0"}]}]}


def test_each_code_name_is_one_sentence_with_the_plans_label_when_it_has_one() -> None:
    found = _figure_label_findings(_state({"fig3.png": SCATTER}, KINETICS))
    assert found == [
        'figures/fig3.png: the x axis label "Ea_hat (kJ/mol)" shows the code name "Ea_hat"; rename it to '
        '"Activation energy $E_a$ (kJ/mol)" (the plan\'s name for it)',
        'figures/fig3.png: the y axis label "lnA_hat" shows the code name "lnA_hat"; rename it to '
        '"Log pre-exponential factor $\\ln A$" (the plan\'s name for it)',
    ]
    found = _figure_label_findings(_state({"fdr_vs_pi1.png": FDR_FIG}, FDR))
    assert found == [
        f'figures/fdr_vs_pi1.png: in the panel "m = {m}", the x axis label "fraction of real effects, pi1" shows the '
        "code name \"pi1\"; rename it to the reader's name for it: words, its symbol in $...$ and its unit"
        for m in (20, 200)
    ]


def test_panel_titles_and_legend_entries_are_read_too() -> None:
    found = _figure_label_findings(_state({"fdr.png": FDR_LEGEND}, FDR))
    assert found == [
        'figures/fdr.png: the title "pi1 = 0.05" shows the code name "pi1"; write "pi1" as a reader would: in words, '
        "or as its symbol in $...$",
        'figures/fdr.png: in the panel "pi1 = 0.05", the legend entry "pi1 = 0.05" shows the code name "pi1"; write '
        '"pi1" as a reader would: in words, or as its symbol in $...$',
    ]


def test_a_label_already_the_plans_or_one_no_plan_or_code_names_is_left_alone(tmp_path: Path) -> None:
    good = {"axes": [{"xlabel": "Activation energy $E_a$ (kJ/mol)", "ylabel": "count"}]}
    assert _figure_label_findings(_state({"a.png": good}, KINETICS)) == []
    # Neither a plan nor code: nothing to tie a name to, so nothing is flagged.
    assert _figure_label_findings(_state({"fig3.png": SCATTER}, None)) == []
    assert _figure_label_findings(_state({"fdr_vs_pi1.png": FDR_FIG}, None)) == []
    # The experiment's code names them: flagged, with no plan label to rename to.
    (tmp_path / "experiment.py").write_text("Ea_hat, lnA_hat = fit(data)\n", encoding="utf-8")
    (tmp_path / "replot_layout.py").write_text("pi1 = 0\n", encoding="utf-8")  # FI's own redraw does not count
    found = _figure_label_findings(_state({"fig3.png": SCATTER, "fdr.png": FDR_LEGEND}, None), tmp_path)
    assert len(found) == 2 and all("fig3.png" in f for f in found)


def test_a_code_name_is_not_sent_to_a_repair_of_the_script() -> None:
    """It used to ride the one layout repair: the script rewritten and run again, and a repair attempt spent."""
    assert _figure_overlap_findings(_state({"fig3.png": SCATTER}, KINETICS)) == []
    layout = {"axes": [], "layout": [{"check": "legend_over_data", "legend": ["a"]}]}
    assert len(_figure_overlap_findings(_state({"fig3.png": {**SCATTER, **layout}}, KINETICS))) == 1


def test_the_code_writer_is_asked_for_the_plans_labels() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("implement.md", "implement_body.md"):
        text = (root / "agents" / name).read_text(encoding="utf-8")
        filled = string.Template(text).safe_substitute()
        assert "never with a variable name from the code" in filled
        assert "`Activation energy $E_a$ (kJ/mol)`" in filled


# ---- the route: write -> replot_layout (the saved numbers), once per quest ----------------------------------------


def _engine(tmp_path: Path) -> Engine:
    eng = Engine(Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        engine=EngineConfig(clarify_mode="off", review_loop=True, max_iterations=2),  # type: ignore[arg-type]
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    ))
    root = eng.quest_root
    for sub in ("paper", "code", "figures", "data/results"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "code" / "experiment.py").write_text("print('the experiment')\n", encoding="utf-8")
    (root / "data" / "results" / "fdr.csv").write_text("m,fdr\n20,0.04\n", encoding="utf-8")
    (root / "figures" / "fdr.png").write_bytes(b"old")
    return eng


STATE: dict[str, Any] = {"topic": "t", "title": "t", "iteration": 1, **_state({"fdr.png": FDR_LEGEND}, FDR)}


def test_the_first_draft_sends_code_names_to_the_redraw_from_the_saved_numbers_once(tmp_path: Path) -> None:
    eng = _engine(tmp_path)

    async def chat(prompt: str, **kw: Any) -> str:
        return "# P\n\nBody.\n"

    eng._chat = chat  # type: ignore[method-assign]
    out = asyncio.run(eng._node_write(dict(STATE)))  # type: ignore[arg-type]
    assert out["figure_labels_redrawn"] is True
    assert len(out["refine_layout"]) == 2 and all(figure_labels.MARK in n for n in out["refine_layout"])
    assert eng._route_after_write({**STATE, **out}) == "replot"  # type: ignore[arg-type]
    # A later draft (a revise after the review) does not ask again.
    again = asyncio.run(eng._node_write({**STATE, **out, "refine_layout": []}))  # type: ignore[arg-type]
    assert again["refine_layout"] == [] and "figure_labels_redrawn" not in again
    assert eng._route_after_write({**STATE, **out, **again}) == "check"  # type: ignore[arg-type]


def _replot(eng: Engine, script: str, notes: list[str]) -> tuple[dict, list]:
    ran: list = []

    async def chat(prompt: str, *, node: str = "") -> str:
        return f"```python\n{script}\n```"

    async def execute(cmd: list[str], **kw: Any) -> Any:
        ran.append(cmd)
        proc = subprocess.run([sys.executable, cmd[-1]], cwd=str(kw["cwd"]), capture_output=True, text=True)
        return SimpleNamespace(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr, duration_s=0.1,
                               timed_out=False)

    eng._chat = chat  # type: ignore[method-assign]
    eng.executor = SimpleNamespace(python_path=lambda q: Path(sys.executable), execute=execute)  # type: ignore[assignment]
    return asyncio.run(eng._node_replot_layout({**STATE, "refine_layout": notes})), ran  # type: ignore[arg-type]


def test_the_redraw_renames_them_without_running_the_experiment(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    notes = _figure_label_findings(STATE)  # type: ignore[arg-type]
    out, ran = _replot(eng, "open('figures/fdr.png', 'wb').write(b'renamed')\nprint('REPLOTTED: fdr.png')", notes)
    assert [Path(c[-1]).name for c in ran] == ["replot_layout.py"]  # the experiment did not run
    assert (eng.quest_root / "figures" / "fdr.png").read_bytes() == b"renamed"
    assert out["layout_missed"] == [] and eng._route_after_replot({**STATE, **out}) == "check"  # type: ignore[arg-type]


def test_a_rename_the_redraw_could_not_make_is_not_told_to_the_writer_as_a_readers_request(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    notes = _figure_label_findings(STATE)  # type: ignore[arg-type]
    out, _ = _replot(eng, "raise SystemExit(3)", notes + ["Figure 2 above Figure 1"])
    assert (eng.quest_root / "figures" / "fdr.png").read_bytes() == b"old"
    # The person's own note is still reported; FI's renames are said in run.log only.
    assert out["layout_missed"] == ["Figure 2 above Figure 1"]
    assert out["layout_not_redrawn"] == ["Figure 2 above Figure 1"]
    only_fi, _ = _replot(_engine(tmp_path / "b"), "raise SystemExit(3)", notes)
    assert only_fi["layout_missed"] == [] and only_fi["layout_not_redrawn"] == []
    assert eng._route_after_replot({**STATE, **only_fi}) == "check"  # type: ignore[arg-type]
    # A redraw that answers FI's renames with NOT_DRAWN lines (in its own words) sends nothing back to the writer.
    declined, _ = _replot(_engine(tmp_path / "c"), "print('NOT_DRAWN: the title is fine')\n"
                                                    "print('NOT_DRAWN: the legend is fine')", notes)
    assert declined["layout_missed"] == []
    assert eng._route_after_replot({**STATE, **declined}) == "check"  # type: ignore[arg-type]
