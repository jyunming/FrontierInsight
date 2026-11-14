"""The engine's files share one module-level namespace: a name set on ``core.engine`` reaches the code in every part."""
from __future__ import annotations

import importlib

import pytest

import core.engine as engine
import core.engine_base as base
import core.engine_plan as plan_part


def test_a_name_replaced_on_the_engine_is_replaced_in_every_part_that_holds_it():
    assert base._parse_json_lenient is engine._parse_json_lenient is plan_part._parse_json_lenient
    original = engine._parse_json_lenient

    def fake(*args, **kwargs):
        return {"fake": True}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("core.engine._parse_json_lenient", fake)
        assert base._parse_json_lenient is fake
        assert plan_part._parse_json_lenient is fake
    assert engine._parse_json_lenient is original
    assert base._parse_json_lenient is original
    assert plan_part._parse_json_lenient is original


def test_a_name_new_to_the_engine_reaches_every_part_and_leaves_with_the_patch():
    def fake_print(*args, **kwargs):
        return None

    assert "print" not in vars(engine) and "print" not in vars(plan_part)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine, "print", fake_print, raising=False)
        assert vars(base)["print"] is fake_print
        assert vars(plan_part)["print"] is fake_print
    assert "print" not in vars(engine)
    assert "print" not in vars(base)
    assert "print" not in vars(plan_part)


def test_a_modules_own_names_are_not_passed_on():
    spec = plan_part.__spec__
    engine.__doc__ = engine.__doc__
    assert plan_part.__spec__ is spec
    assert plan_part.__name__ == "core.engine_plan"
    assert plan_part.__doc__ != engine.__doc__


def test_a_name_set_on_a_part_changes_that_part_only():
    original = engine._parse_json_lenient
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(plan_part, "_parse_json_lenient", lambda *a, **k: None)
        assert engine._parse_json_lenient is original
        assert base._parse_json_lenient is original
    assert plan_part._parse_json_lenient is original


def test_every_stage_file_is_a_part():
    names = {part.__name__ for part in engine._PARTS}
    for stage in ("ideas", "plan", "data", "code", "improve", "run", "analyze", "write", "review"):
        module = importlib.import_module(f"core.engine_{stage}")
        assert module.__name__ in names
        mixin = next(v for k, v in vars(module).items() if k.endswith("StageMixin"))
        assert issubclass(engine.Engine, mixin)
    assert "core.engine_base" in names
