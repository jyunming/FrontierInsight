"""Reading a model's answer part by part (core/answer_reading.py and the plan readers that report through it): one entry
that cannot be read costs that entry, each fault names its own path, a part moved unchanged into an empty place is a
move, and a part written twice with different content is a conflict carrying both versions."""
from __future__ import annotations

import json

import pytest

from core import answer_reading as ar
from core import optimisation_plan as optim
from core import plan


def _block(**extra):
    block = {
        "objective": {"quantity": "drag", "direction": "minimise"},
        "design_variables": [
            {"name": "width_mm", "low": 1.0, "high": 5.0},
            {"name": "speed", "low": 0.5, "high": 2.0},
        ],
        "baseline": {"values": {"width_mm": 2.0, "speed": 1.0}, "source": "the topic"},
        "evaluation_budget": {"starts": 2, "per_start": 10},
        "constraints": [{"quantity": "mass", "limit": "<= 3"}, {"quantity": "heat", "limit": ">= 1"}],
    }
    block.update(extra)
    return block


# --- the shared structure --------------------------------------------------------------------------------------------


def test_paths_round_trip_and_find_the_entry():
    path = ar.join(ar.join(ar.join("protocol", "optimisation"), "constraints"), 2)
    assert path == "protocol.optimisation.constraints[2]"
    assert ar.split(path) == ["protocol", "optimisation", "constraints", 2]
    odd = ar.join("protocol.thresholds", "a name.with dots")
    assert ar.split(odd) == ["protocol", "thresholds", "a name.with dots"]
    protocol = {"optimisation": _block()}
    assert ar.value_at(protocol, path, "protocol") == {"quantity": "heat", "limit": ">= 1"}
    assert ar.value_at(protocol, "protocol.optimisation.constraints[9]", "protocol") is None


def test_a_fault_has_a_known_kind():
    with pytest.raises(ValueError):
        ar.Fault("protocol.grid", "odd", "why")


def test_an_answer_that_is_not_a_mapping_is_the_only_one_refused_whole():
    reading = ar.read_answer(["not", "a", "mapping"], None, lambda a, b: ar.Reading(kept=a))
    assert reading.nothing_read and reading.kept is None
    assert plan.read_protocol("prose").nothing_read
    assert optim.read_block("prose").nothing_read


def test_the_one_ask_names_only_the_faulted_paths():
    faults = [
        ar.Fault("protocol.optimisation.constraints[2]", "unreadable", "`limit` must read like \"<= 120\".",
                 as_written={"quantity": "heat", "limit": "about one"}),
        ar.Fault("protocol.optimisation.search_method", "not_allowed", "the method is not one FI runs.",
                 allowed=("nelder_mead", "random"), as_written="hill"),
        ar.Fault("protocol.grid", "conflict", "the two differ.", as_written={"protocol.grid": {"a": [1]},
                                                                           "protocol.optimisation.grid": {"a": [2]}}),
    ]
    text = ar.ask_text(faults)
    assert "`protocol.optimisation.constraints[2]`" in text and "about one" in text
    assert "Allowed: `nelder_mead`, `random`." in text
    assert "Both versions:" in text
    assert "objective" not in text and "design_variables" not in text  # nothing that read is asked about
    assert ar.ask_text([]) == ""


def test_entries_as_written_are_said_once_each():
    one = ar.Fault("protocol.oracles[2]", "unreadable", "no name", as_written={"check": "x"})
    assert ar.entries_as_written([one, one]) == 'the entry as written: {"check": "x"}'
    assert ar.entries_as_written([one], you=True).startswith("the entry as you wrote it")
    assert ar.entries_as_written([ar.Fault("protocol.grid", "missing", "none")]) == ""


def test_the_record_is_kept_and_a_failed_write_raises(tmp_path):
    reading = ar.Reading(kept={}, faults=[ar.Fault("protocol.grid.a", "unreadable", "why", required=True)],
                         moved=[ar.Moved("protocol.grid", "protocol.optimisation.grid")])
    path = tmp_path / "readings.json"
    ar.write_record(path, ar.record_entry(reading, step="plan", asked=["protocol.grid.a"]))
    ar.write_record(path, ar.record_entry(ar.Reading(kept={}), step="audit"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert [e["step"] for e in saved] == ["plan", "audit"]
    assert saved[0]["faults"] == [{"path": "protocol.grid.a", "kind": "unreadable", "reason": "why", "required": True}]
    assert saved[0]["moved"] == [{"path": "protocol.grid", "to": "protocol.optimisation.grid"}]
    blocked = tmp_path / "a_folder"
    blocked.mkdir()
    with pytest.raises(OSError):  # a record that cannot be written is never swallowed
        ar.write_record(blocked, ar.record_entry(reading, step="plan"))
    not_a_list = tmp_path / "other.json"
    not_a_list.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        ar.write_record(not_a_list, ar.record_entry(reading, step="plan"))


# --- the plan's protocol ---------------------------------------------------------------------------------------------


def test_a_clean_answer_reads_exactly_as_the_strict_reader_reads_it():
    protocol = {"grid": {"width_mm": [1, 2], "tool": ["tool_a", "tool_b"]}, "runs_per_setting": 3,
                "thresholds": {"cut": 0.5}, "oracles": [{"name": "zero_width", "check": "drag is 0 at width 0"}]}
    reading = plan.read_protocol(protocol)
    assert reading.clean
    assert reading.kept == plan.normalize_protocol(protocol)[0]
    assert plan.repair_protocol(protocol) == (plan.normalize_protocol(protocol)[0], [])


def test_a_factor_both_compared_in_a_run_and_swept_costs_that_factor_not_the_grid():
    # The old reader found the part to leave out in the words of its own reason, whose first part named was the grid:
    # the whole grid was left out for a fault of `in_each_run`.
    protocol = {"grid": {"width_mm": [1, 2], "speed": [1, 2]}, "in_each_run": {"speed": [1, 2]}, "runs_per_setting": 2}
    kept, notes = plan.repair_protocol(protocol)
    assert kept["grid"] == {"width_mm": [1, 2], "speed": [1, 2]}
    assert "in_each_run" not in kept
    fault = plan.read_protocol(protocol).faults[0]
    assert fault.path == "protocol.in_each_run.speed" and fault.kind == "conflict"
    assert fault.as_written == {"protocol.grid": [1, 2], "protocol.in_each_run": [1, 2]}


def test_one_unreadable_part_of_the_study_costs_that_part_only():
    protocol = {
        "grid": {"width_mm": [1, 2], "tool": ["tool_a", "tool_b"]},
        "sub_studies": [
            {"name": "widths", "vary": ["width_mm"], "hold": {"tool": "tool_a"}},
            {"name": "tools", "vary": ["tool"], "hold": {"width_mm": 1}},
            {"name": "broken", "vary": ["colour"]},
        ],
    }
    kept, notes = plan.repair_protocol(protocol)
    assert [s["name"] for s in kept["sub_studies"]] == ["widths", "tools"]
    assert kept["grid"] == protocol["grid"]
    reading = plan.read_protocol(protocol)
    assert reading.paths() == ["protocol.sub_studies[3]"]
    assert reading.faults[0].as_written == {"name": "broken", "vary": ["colour"]}
    assert len(notes) == 1 and "entry 3" in notes[0]


def test_parts_that_cannot_stand_together_are_still_refused():
    # Known positive: the entry rule never keeps a set of parts that leaves a value of the grid unrun.
    protocol = {
        "grid": {"width_mm": [1, 2], "tool": ["tool_a", "tool_b"]},
        "sub_studies": [{"name": "widths", "vary": ["width_mm"], "hold": {"tool": "tool_a"}},
                        {"name": "broken", "vary": ["colour"]}],
    }
    kept, notes = plan.repair_protocol(protocol)
    assert "sub_studies" not in kept
    assert any(plan.ASK_MARK in n for n in notes)


def _beside(top_settings, block_settings):
    block = _block(numerical_settings=block_settings)
    return {"optimisation": block, "numerical_settings": top_settings}


def test_the_same_part_beside_the_search_block_is_kept_once_without_a_fault():
    settings = {"mesh_mm": {"search": 0.2, "check": [0.1, 0.05]}}
    kept, notes = plan.repair_protocol(_beside(settings, settings))
    assert kept["optimisation"]["numerical_settings"]["mesh_mm"]["search"] == 0.2
    assert "numerical_settings" not in kept
    assert notes == []  # the old reader said the copy "was left out", and an audit's whole block was refused for it
    reading = plan.read_protocol(_beside(settings, settings))
    assert reading.faults == []
    assert reading.moved == [ar.Moved("protocol.numerical_settings", "protocol.optimisation.numerical_settings",
                                       into_empty=False)]


def test_a_different_part_beside_the_search_block_is_a_conflict_with_both_versions():
    top = {"mesh_mm": {"search": 0.2}}
    inner = {"mesh_mm": {"search": 0.4}}
    reading = plan.read_protocol(_beside(top, inner))
    assert [(f.path, f.kind) for f in reading.faults] == [("protocol.numerical_settings", "conflict")]
    assert reading.faults[0].as_written == {"protocol.numerical_settings": top,
                                            "protocol.optimisation.numerical_settings": inner}
    assert reading.kept["optimisation"]["numerical_settings"]["mesh_mm"]["search"] == 0.4  # the block's own is read
    assert "the two differ" in plan.repair_protocol(_beside(top, inner))[1][0]


def test_a_sweep_moved_into_an_empty_scan_is_a_move_not_a_fault():
    protocol = {"optimisation": _block(), "grid": {"width_mm": [1, 2]}}
    reading = plan.read_protocol(protocol)
    assert reading.faults == []
    assert reading.moved == [ar.Moved("protocol.grid", "protocol.optimisation.grid")]
    assert reading.kept["optimisation"]["grid"] == {"width_mm": [1, 2]}


@pytest.mark.parametrize("bad", [
    {"name": "no_check", "tolerance": [1, 2]},
    {"name": "no_mode", "check": "x", "tolerance_mode": "loose"},
    {"name": "no_text", "check": 12},
    {"check": "a check with no name"},
])
def test_one_bad_check_among_three_is_named_at_its_own_path(bad):
    oracles = [{"name": "first", "check": "a"}, bad, {"name": "third", "check": "c"}]
    reading = plan.read_protocol({"oracles": oracles})
    assert reading.paths() == ["protocol.oracles[2]"]
    assert [o["name"] for o in reading.kept["oracles"]] == ["first", "third"]


def test_a_part_written_twice_beside_and_in_the_protocol_is_a_conflict():
    design = {"hypothesis": "h", "grid": {"width_mm": [1, 2]}, "protocol": {"grid": {"width_mm": [1, 3]}}}
    reading = plan.read_hoist(design)
    assert [(f.path, f.kind) for f in reading.faults] == [("grid", "conflict")]
    assert plan.hoist_protocol_parts(design) == (design, [])  # what the readers see is unchanged
    alike = {"hypothesis": "h", "grid": {"width_mm": [1]}, "protocol": {"grid": {"width_mm": [1]}}}
    assert plan.read_hoist(alike).faults == []
    stray = {"hypothesis": "h", "grid": {"width_mm": [1]}, "protocol": {}}
    assert plan.read_hoist(stray).moved == [ar.Moved("grid", "protocol.grid")]


def test_the_entry_of_a_fault_is_found_by_its_path():
    protocol = {"oracles": [{"name": "a", "check": "x"}, {"check": 3}]}
    fault = plan.read_protocol(protocol).faults[0]
    assert plan.offending_entry(protocol, fault) == '{"check": 3}'


# --- a search's block ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    {"quantity": "heat"},  # no limit at all
    {"quantity": "heat", "limit": "about one"},  # a limit in words
    {"quantity": "heat", "max": 3},  # the bound under another key
    "heat stays below three",  # a sentence, not a mapping
    {"limit": "<= 3"},  # no quantity
])
def test_one_bad_limit_is_named_at_its_path_and_the_rest_of_the_search_is_kept(bad):
    block = _block()
    block["constraints"] = [block["constraints"][0], bad]
    reading = optim.read_block(block)
    assert reading.paths() == ["protocol.optimisation.constraints[2]"]
    fault = reading.faults[0]
    assert fault.kind == "unreadable" and fault.required and fault.as_written == bad
    assert [c["quantity"] for c in reading.kept["constraints"]] == ["mass"]
    assert [v["name"] for v in reading.kept["design_variables"]] == ["width_mm", "speed"]


def test_a_held_condition_that_is_also_a_variable_is_a_conflict_with_both_versions():
    block = _block(fixed={"speed": 1.5, "load": 3})
    reading = optim.read_block(block)
    assert [(f.path, f.kind) for f in reading.faults] == [("protocol.optimisation.fixed.speed", "conflict")]
    assert reading.faults[0].as_written["protocol.optimisation.fixed"] == {"speed": 1.5}
    assert reading.kept["fixed"] == {"load": 3}
    assert [v["name"] for v in reading.kept["design_variables"]] == ["width_mm", "speed"]


def test_an_unreadable_variable_is_held_at_its_baseline_and_said():
    # The old reader left the variable and its baseline value out: the simulation got no value for it at all.
    block = _block()
    block["design_variables"][1] = {"name": "speed", "low": "slow", "high": "fast"}
    fixed, notes = optim.repair(block)
    assert [v["name"] for v in fixed["design_variables"]] == ["width_mm"]
    assert fixed["fixed"] == {"speed": 1.0}
    assert "speed" not in fixed["baseline"]["values"]
    assert "held there" in notes[0] and "baseline" in notes[0]
    reading = optim.read_block(block)
    assert reading.paths() == ["protocol.optimisation.design_variables[2]"] and reading.faults[0].required


def test_an_unreadable_variable_without_a_baseline_value_stops_the_search():
    block = _block()
    block["design_variables"][1] = {"name": "speed", "low": "slow", "high": "fast"}
    block["baseline"] = {"values": {"width_mm": 2.0}, "source": "the topic"}
    assert optim.repair(block)[0] is None
    reading = optim.read_block(block)
    assert not reading.nothing_read and reading.required()
    assert "objective" in reading.kept and "evaluation_budget" in reading.kept


def test_an_unreadable_objective_keeps_every_other_part_in_the_reading():
    block = _block(objective="make the drag small")
    assert optim.repair(block)[0] is None  # a search with no objective cannot start, as before
    reading = optim.read_block(block)
    assert not reading.nothing_read
    assert [f.path for f in reading.required()] == ["protocol.optimisation.objective"]
    assert set(reading.kept) == {"design_variables", "baseline", "evaluation_budget", "constraints"}


def test_a_check_value_that_is_not_finer_is_left_out_alone():
    block = _block(numerical_settings={"mesh_mm": {"search": 0.2, "check": [0.3]}, "step_s": {"search": 1.0}})
    fixed, notes = optim.repair(block)
    assert fixed["numerical_settings"]["mesh_mm"] == {"search": 0.2}
    assert "step_s" in fixed["numerical_settings"]
    assert optim.read_block(block).paths() == ["protocol.optimisation.numerical_settings.mesh_mm.check"]


def test_the_plain_gaps_come_from_the_fault_paths():
    design = {"study_type": "find_best_design", "protocol": {"optimisation": _block(objective="small")}}
    why = optim.normalize(design["protocol"]["optimisation"])[1]
    assert optim.plain_gaps(design, why) == ["what the search should make as low or as high as possible"]
    assert optim.plain_gaps(design, None) == ["what the search should make as low or as high as possible"]


def test_a_clean_block_reads_exactly_as_before():
    block = _block(numerical_settings={"mesh_mm": {"search": 0.2, "check": [0.1]}}, grid={"width_mm": [1, 2]})
    reading = optim.read_block(block)
    assert reading.clean and reading.kept == optim.normalize(block)[0]
    assert optim.repair(block) == (optim.normalize(block)[0], [])


# --- review findings -------------------------------------------------------------------------------------------------


def test_two_bad_limits_are_each_named_at_their_own_place():
    block = _block()
    block["constraints"] = [{"quantity": "mass"}, {"quantity": "heat", "limit": "<= 2"}, {"quantity": "noise"}]
    reading = optim.read_block(block)
    assert reading.paths() == ["protocol.optimisation.constraints[1]", "protocol.optimisation.constraints[3]"]
    assert [f.as_written for f in reading.faults] == [{"quantity": "mass"}, {"quantity": "noise"}]


def test_two_bad_variables_are_each_named_at_their_own_place():
    block = _block()
    block["design_variables"] = [{"name": "width_mm", "low": "a", "high": "b"}, {"name": "speed", "low": 0.5, "high": 2.0},
                                 {"name": "depth_mm", "low": 1.0, "high": 2.0}, {"name": "load", "low": "x", "high": "y"}]
    block["baseline"] = {"values": {"width_mm": 2.0, "speed": 1.0, "depth_mm": 1.5, "load": 3}, "source": "s"}
    reading = optim.read_block(block)
    assert reading.paths() == ["protocol.optimisation.design_variables[1]", "protocol.optimisation.design_variables[4]"]
    assert reading.kept["fixed"] == {"width_mm": 2.0, "load": 3}


def test_a_variable_named_twice_differently_is_a_conflict_and_alike_a_duplicate():
    block = _block()
    block["design_variables"].append({"name": "speed", "low": 0.5, "high": 9.0})
    fault = optim.read_block(block).faults[0]
    assert fault.kind == "conflict" and len(fault.as_written) == 2
    alike = _block()
    alike["design_variables"].append(dict(alike["design_variables"][1]))
    assert optim.read_block(alike).faults[0].kind == "duplicate"


def test_many_bad_entries_never_cost_the_whole_block():
    block = _block()
    block["constraints"] = [{"quantity": f"q{i}"} for i in range(70)] + [{"quantity": "mass", "limit": "<= 3"}]
    fixed, notes = optim.repair(block)
    assert fixed is not None and [c["quantity"] for c in fixed["constraints"]] == ["mass"]
    assert len(optim.read_block(block).faults) == 70


def test_the_same_part_twice_is_recorded_as_kept_once_not_as_a_move_into_an_empty_place():
    settings = {"mesh_mm": {"search": 0.2}}
    moved = plan.read_protocol(_beside(settings, settings)).moved
    assert moved == [ar.Moved("protocol.numerical_settings", "protocol.optimisation.numerical_settings", into_empty=False)]
    assert plan.read_protocol({"optimisation": _block(), "grid": {"width_mm": [1, 2]}}).moved[0].into_empty


def test_a_variable_named_twice_keeps_both_versions_even_when_the_first_cannot_be_read():
    block = _block()
    block["design_variables"] = [{"name": "speed", "low": "slow", "high": 2.0}, {"name": "speed", "low": 0.5, "high": 2.0},
                                 {"name": "width_mm", "low": 1.0, "high": 5.0}]
    # (in the one-pass reading; the repair then holds the unreadable first one at its baseline)
    conflict = next(f for f in optim.block_faults(block)[1] if f.kind == "conflict")
    assert conflict.as_written == [block["design_variables"][0], block["design_variables"][1]]


def test_a_scan_part_named_after_others_left_out_keeps_its_own_number():
    block = _block(fixed={"load": 1})
    block["grid"] = {"colour_mm": [1, 2], "load": [1, 2]}
    block["sub_studies"] = [{"name": "only_colour", "vary": ["colour_mm"]},
                            {"name": "loads", "vary": ["load"], "hold": {"unknown": 1}}]
    paths = optim.read_block(block).paths()
    assert "protocol.optimisation.sub_studies[2]" in paths and "protocol.optimisation.sub_studies[1]" not in paths


def test_many_extra_baseline_values_never_cost_the_whole_block():
    block = _block()
    block["baseline"]["values"].update({f"extra_{i}": i for i in range(100)})
    fixed, _notes = optim.repair(block)
    assert fixed is not None and set(fixed["baseline"]["values"]) == {"width_mm", "speed"}


def test_many_bad_numerical_settings_never_cost_the_whole_protocol():
    settings = {f"step_{i}": {"search": -1} for i in range(80)}
    settings["mesh_mm"] = {"search": 0.2}
    kept, notes = plan.repair_protocol({"grid": {"width_mm": [1, 2]}, "numerical_settings": settings})
    assert kept is not None and set(kept["numerical_settings"]) == {"mesh_mm"} and len(notes) == 80
