"""A plan can declare ``protocol.sub_studies``: separate parts of the study that each vary only some of the grid's settings.
The study's settings are the union of the sub-studies' settings (their sum, not the product of every setting), counted and
listed by one function; a plan without sub-studies gets exactly the settings it always got. Fake model only."""
from __future__ import annotations

import itertools
import json
import math

import pytest

from core import optimise_search as search
from core import code_project
from core import plan
from core import run_estimate as est
from core import run_manifest as rm
from core import topic_requirements as topic
from core import trial_runner

GRID = {"family": ["conv", "annular", "dipole"], "dipole_radius": [0.1, 0.2, 0.3], "ring": [0.5, 0.7],
        "ambit_type": ["a", "b", "c", "d"], "defocus": [-100, 0, 100]}
SUBS = [
    {"name": "sources", "vary": ["family", "defocus", "ring"], "hold": {}},
    {"name": "dipole", "vary": ["dipole_radius"], "hold": {"family": "dipole"}},
    {"name": "ambit", "vary": ["ambit_type", "defocus"], "hold": {}},
]


def _protocol(**more):
    return {"grid": {k: list(v) for k, v in GRID.items()}, "sub_studies": [dict(s) for s in SUBS], **more}


def test_without_sub_studies_the_settings_are_the_full_product_in_the_same_order() -> None:
    plain = search.study_grid({"grid": GRID})
    assert not isinstance(plain, search.StudyGrid) and plain is not None
    old = [dict(zip(GRID, combo)) for combo in itertools.product(*GRID.values())]
    assert search.grid_cells(plain) == old and search.grid_size(plain) == len(old) == 3 * 3 * 2 * 4 * 3
    assert search.study_grid({}) == {} and search.study_grid(None) == {}
    assert search.study_grid({"grid": GRID, "sub_studies": []}) == GRID


def test_the_settings_are_the_union_of_the_sub_studies_and_counted_exactly() -> None:
    grid = search.study_grid(_protocol())
    assert isinstance(grid, search.StudyGrid) and dict(grid) == GRID
    cells = search.grid_cells(grid)
    keys = [tuple(c.items()) for c in cells]
    assert len(keys) == len(set(keys)) == search.grid_size(grid)  # no setting listed twice, the count is the list's length
    assert all(set(c) == set(GRID) for c in cells)  # every setting carries every axis, a real value each
    # sources 3*3*2=18 (radius, radius, ambit at their first), dipole 3, ambit 4*3=12; shared: family=conv/defocus cells etc.
    sizes = dict(search.sub_study_sizes(grid))
    assert sizes == {"sources": 18, "dipole": 3, "ambit": 12}
    assert search.grid_size(grid) < sum(sizes.values()) and search.grid_size(grid) < math.prod(len(v) for v in GRID.values())
    # a held value is the stated one, an unstated one the first listed
    dipole = [c for c in cells if c["dipole_radius"] != 0.1]
    assert dipole and all(c["family"] == "dipole" and c["defocus"] == -100 and c["ambit_type"] == "a" for c in dipole)
    # every value of every axis is somewhere in the study (no comparison is lost)
    for axis, values in GRID.items():
        assert {c[axis] for c in cells} == set(values)


def test_the_union_count_equals_the_list_for_random_overlaps() -> None:
    import random
    rng = random.Random(7)
    axes = {f"a{i}": list(range(rng.randint(1, 4))) for i in range(6)}
    for _ in range(60):
        subs = []
        for j in range(rng.randint(1, 5)):
            vary = rng.sample(list(axes), rng.randint(1, 4))
            hold = {a: rng.choice(axes[a]) for a in axes if a not in vary and rng.random() < 0.5}
            subs.append({"name": f"s{j}", "vary": vary, "hold": hold})
        grid = search.study_grid({"grid": axes, "sub_studies": subs})
        cells = search.grid_cells(grid)
        assert search.grid_size(grid) == len(cells) == len({tuple(c.items()) for c in cells})


def test_the_count_is_the_length_of_the_list_beyond_the_parts_counted_by_inclusion_exclusion() -> None:
    axes = {f"a{i}": list(range(3)) for i in range(6)}
    subs = [{"vary": [f"a{i % 6}", f"a{(i + 1) % 6}"]} for i in range(14)]  # 14 overlapping parts
    grid = search.study_grid({"grid": axes, "sub_studies": subs})
    assert len(grid.sub_studies) > search.MAX_EXACT_SUB_STUDIES
    assert search.grid_size(grid) == len(search.grid_cells(grid)) < sum(n for _, n in search.sub_study_sizes(grid))
    big = search.study_grid({"grid": {f"b{i}": list(range(60)) for i in range(4)},
                             "sub_studies": [{"vary": [f"b{i % 4}", f"b{(i + 1) % 4}", f"b{(i + 2) % 4}"]} for i in range(14)]})
    assert search.grid_size(big) > search.MAX_LISTED_SETTINGS and search.grid_too_large(big)  # over the limit either way


def test_what_each_run_compares_inside_it_is_crossed_with_every_setting() -> None:
    grid = search.with_everywhere(search.study_grid(_protocol()), {"method": ["x", "y"]})
    cells = search.grid_cells(grid)
    assert len(cells) == 2 * search.grid_size(search.study_grid(_protocol()))
    assert {c["method"] for c in cells} == {"x", "y"}
    assert search.with_everywhere({"a": [1]}, {"b": [1, 2]}) == {"a": [1], "b": [1, 2]}


def test_sub_studies_make_a_grid_that_was_too_large_fit_and_a_still_too_large_one_stops() -> None:
    big = {f"x{i}": list(range(10)) for i in range(8)}  # 10^8 settings as one grid
    assert "100,000,000" in search.grid_too_large(big)
    fits = search.study_grid({"grid": big, "sub_studies": [{"vary": [f"x{i}", f"x{i + 1}"]} for i in range(0, 8, 2)]})
    assert search.grid_size(fits) == 4 * 100 - 3 and search.grid_too_large(fits) == ""
    assert len(search.grid_cells(fits)) == 397
    still = search.study_grid({"grid": big, "sub_studies": [{"name": "everything", "vary": list(big)}]})
    said = search.grid_too_large(still)
    assert "100,000,000" in said and "`everything`" in said and "sub_studies" in said
    with pytest.raises(search.TooManySettings):
        search.grid_cells(still)


def test_a_sub_study_that_does_not_read_is_left_out_not_guessed() -> None:
    for subs in ([{"vary": ["nope"]}], [{"vary": []}], [{"vary": ["family"], "hold": {"defocus": 7}}], ["family"]):
        assert not isinstance(search.study_grid({"grid": GRID, "sub_studies": subs}), search.StudyGrid)


def _normalized(**protocol):
    return plan.normalize_protocol({"grid": {k: list(v) for k, v in GRID.items()}, **protocol})


def test_the_plan_check_reads_a_good_division_and_names_each_planted_fault() -> None:
    fixed, why = _normalized(sub_studies=[dict(s) for s in SUBS])
    assert why is None and [s["name"] for s in fixed["sub_studies"]] == ["sources", "dipole", "ambit"]
    assert fixed["sub_studies"][1]["hold"] == {"family": "dipole"}
    assert isinstance(search.study_grid(fixed), search.StudyGrid)
    faults = {
        "not a setting of": [{"vary": ["nope", "family"]}] + SUBS,
        "no entry of `protocol.sub_studies` varies": SUBS[1:],  # family, ring, defocus... are in no sub-study
        "not one of its values": [dict(SUBS[0], hold={"ambit_type": "zzz"}), SUBS[2]],
        "both varies and holds": [dict(SUBS[0], hold={"family": "conv"}), SUBS[2]],
        "needs `vary`": [{"name": "empty", "vary": []}],
        "lists a setting twice": [{"vary": ["family", "family"]}],
        "must be a list": "family",
    }
    for expect, subs in faults.items():
        got, why = _normalized(sub_studies=subs)
        assert got is None and expect in (why or ""), (expect, why)
    got, why = plan.normalize_protocol({"sub_studies": SUBS})
    assert got is None and "which is missing" in why
    got, why = plan.normalize_protocol({"grid": {"a": [1, 2]}, "in_each_run": {"m": ["x", "y"]},
                                        "sub_studies": [{"vary": ["a", "m"]}]})
    assert got is None and "in_each_run" in why


def test_a_protocol_without_sub_studies_is_normalised_as_before() -> None:
    fixed, why = _normalized()
    assert why is None and "sub_studies" not in fixed
    assert plan.hoist_protocol_parts({"protocol": None, "sub_studies": [{"vary": ["a"]}]})[1] == ["sub_studies"]


# --- every reader of the settings reads the same ones -------------------------------------------------------------------


def _key(cell):
    return ",".join(f"{a}={trial_runner._fmt(v)}" for a, v in cell.items())


def test_the_trial_runner_the_manifest_and_the_generated_script_list_the_same_settings() -> None:
    grid = search.study_grid(_protocol())
    listed = search.grid_cells(grid)
    assert trial_runner.cells(grid) == listed
    assert {frozenset((a, rm._freeze(v)) for a, v in c.items()) for c in listed} == rm._all_cells(grid)
    # the script FI writes into code/ (run.py) holds its own copy of the rule: the same settings in the same order
    source = code_project.RUN_SOURCE
    start = source.index("def cells(")
    ns = {"itertools": itertools, "math": math, "MAX_SETTINGS": 500000}
    exec(source[start:source.index("def cell_key(")], ns)
    assert ns["cells"](dict(grid), [dict(s) for s in grid.sub_studies]) == listed
    assert ns["cells"](dict(GRID), None) == [dict(zip(GRID, c)) for c in itertools.product(*GRID.values())]


def test_run_py_is_given_the_sub_studies_and_a_change_to_them_is_a_difference(tmp_path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    (code / "simulate.py").write_text("def run_cell(cell):" + chr(10) + "    return {'x': 1}" + chr(10), encoding="utf-8")
    protocol = plan.normalize_protocol(_protocol())[0]
    study = code_project.study_of(code, protocol)
    assert [s["name"] for s in study["sub_studies"]] == ["sources", "dipole", "ambit"]
    plain = code_project.study_of(code, {"grid": GRID})
    assert "sub_studies" not in plain
    (code / "study.json").write_text(json.dumps(plain), encoding="utf-8")
    assert any("sub-studies" in d for d in code_project.study_differences(code, protocol))
    (code / "study.json").write_text(json.dumps(study), encoding="utf-8")
    assert code_project.study_differences(code, protocol) == []


def test_the_manifest_holds_a_run_to_the_settings_of_the_sub_studies_not_to_every_combination() -> None:
    protocol = {**_protocol(), "runs_per_setting": 2}
    protocol = plan.normalize_protocol(protocol)[0]
    grid = search.study_grid(protocol)
    cells = search.grid_cells(grid)
    attempted = {_key(c): 2 for c in cells}
    manifest = {"schema": rm.SCHEMA, "realized_grid": {k: list(v) for k, v in GRID.items()}, "attempted_per_cell": attempted,
                "successful_per_cell": dict(attempted), "failed_trials": []}
    assert rm.problems(protocol, manifest) == []  # a run of exactly the union passes
    short = dict(attempted)
    short.pop(_key(cells[-1]))
    assert any("exactly" in f for f in rm.problems(protocol, {**manifest, "attempted_per_cell": short}))  # one missing: caught
    every = {_key(dict(zip(GRID, c))): 2 for c in itertools.product(*GRID.values())}
    found = rm.problems(protocol, {**manifest, "attempted_per_cell": every, "successful_per_cell": dict(every)})
    assert any("exactly" in f for f in found)  # the full product is more than the study asked for: caught


def test_the_time_of_the_study_is_summed_over_its_sub_studies() -> None:
    grid = search.study_grid({"grid": {"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]},
                              "sub_studies": [{"vary": ["a"]}, {"vary": ["b", "c"]}]})
    # base is the middle value of each; b at its first value costs 2x, c at its last value 3x, a does not matter
    probes = [est.Probe(None, None, {"a": 2, "b": 2, "c": 2}, 10.0),
              est.Probe("a", 0, {"a": 1, "b": 2, "c": 2}, 10.0), est.Probe("a", 2, {"a": 3, "b": 2, "c": 2}, 10.0),
              est.Probe("b", 0, {"a": 2, "b": 1, "c": 2}, 20.0), est.Probe("b", 2, {"a": 2, "b": 3, "c": 2}, 10.0),
              est.Probe("c", 0, {"a": 2, "b": 2, "c": 1}, 10.0), est.Probe("c", 2, {"a": 2, "b": 2, "c": 3}, 30.0)]
    e = est.estimate(grid, 1, False, probes)
    cells = search.grid_cells(grid)
    ratio = {"a": {1: 1, 2: 1, 3: 1}, "b": {1: 2, 2: 1, 3: 1}, "c": {1: 1, 2: 1, 3: 3}}
    expected = sum(math.prod(ratio[a][c[a]] for a in c) for c in cells) * 10.0
    assert e is not None and e.cells == len(cells) == 3 + 9 - 1 and abs(e.seconds - expected) < 1e-9
    plain = est.estimate({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]}, 1, False, probes)
    assert plain.cells == 27 and plain.seconds > e.seconds  # the same plan as one grid is dearer


# --- what FI adds to a plan joins a part, it does not multiply the others ----------------------------------------------


def _design(**protocol):
    return {"protocol": {"grid": {"source_family": ["a", "b", "c", "d", "e"]}, **protocol}}


def test_a_setting_added_for_the_topic_joins_the_part_the_answer_names() -> None:
    old = _design()
    new = _design(grid={"source_family": ["a", "b", "c", "d", "e"], "defocus_nm": list(range(9)), "ambit_type": [1, 2, 3, 4]},
                  sub_studies=[{"name": "sources", "vary": ["source_family", "defocus_nm"]},
                               {"name": "ambit", "vary": ["ambit_type"]}])
    merged = topic.merge(old, new, [])
    protocol = plan.normalize_protocol(merged["protocol"])[0]
    assert protocol is not None
    grid = search.study_grid(protocol)
    assert search.grid_size(grid) == 5 * 9 + 4 - 1  # 48, not 5 * 9 * 4 = 180
    # no answer for the parts: the plan has none, so the setting joins the one grid, as before
    plain = topic.merge(old, _design(grid={"source_family": list("abcde"), "defocus_nm": [0, 1]}), [])
    assert "sub_studies" not in plain["protocol"]
    assert search.grid_size(search.study_grid(plain["protocol"])) == 10


def test_a_setting_added_to_a_plan_that_has_sub_studies_without_a_part_is_a_part_of_its_own() -> None:
    old = {"protocol": {"grid": {k: list(v) for k, v in GRID.items()}, "sub_studies": [dict(s) for s in SUBS]}}
    new = {"protocol": {"grid": {**GRID, "noise": [0.0, 0.1, 0.2]}}}
    merged = topic.merge(old, new, [])
    protocol = plan.normalize_protocol(merged["protocol"])[0]
    assert protocol["sub_studies"][-1] == {"name": "noise", "vary": ["noise"], "hold": {}}
    before = search.grid_size(search.study_grid(old["protocol"]))
    assert search.grid_size(search.study_grid(protocol)) == before + 2  # three values at the base: two new settings
    # a part of the plan the answer names takes the setting
    named = {"protocol": {"grid": {**GRID, "noise": [0.0, 0.1]}, "sub_studies": [{"name": "dipole", "vary": ["noise"]}]}}
    joined = topic.merge(old, named, [])["protocol"]["sub_studies"]
    assert [s["vary"] for s in joined if s["name"] == "dipole"] == [["dipole_radius", "noise"]]
