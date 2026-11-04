"""The registry of checks (core/checks.py) stays true: every row points at code that exists, the words it uses are the
allowed ones, and it cannot drift from the list the engine runs before the code does (core/static_checks.py)."""

from __future__ import annotations

import importlib

import pytest

from core import checks
from core.static_checks import static_checks


def _resolve(row: checks.Check) -> object:
    target: object = importlib.import_module(row.module)
    for part in row.function.split("."):
        target = getattr(target, part)
    return target


def test_the_registry_is_not_empty() -> None:
    assert checks.CHECKS
    assert all(isinstance(row, checks.Check) for row in checks.CHECKS)


@pytest.mark.parametrize("row", checks.CHECKS, ids=lambda row: row.name)
def test_every_row_points_at_code_that_exists(row: checks.Check) -> None:
    assert row.module.startswith("core.")
    assert callable(_resolve(row)), f"{row.module}.{row.function} is not callable"


def test_names_are_unique_and_snake_case() -> None:
    names = [row.name for row in checks.CHECKS]
    assert len(names) == len(set(names)), sorted({n for n in names if names.count(n) > 1})
    assert all(name.replace("_", "").isalnum() and name == name.lower() for name in names)


def test_every_row_uses_the_allowed_words_and_says_what_it_judges() -> None:
    for row in checks.CHECKS:
        assert row.source in checks.SOURCES, row.name
        assert row.power in checks.POWERS, row.name
        assert row.stage in checks.STAGES, row.name
        assert row.note.strip() and "\n" not in row.note, row.name


def test_the_static_checks_the_engine_runs_are_all_listed_with_the_same_power() -> None:
    rows = {row.name: row for row in checks.CHECKS}
    for name, module, on_standing in static_checks():
        assert name in rows, f"static check {name} has no row in core/checks.py"
        assert rows[name].module == module.__name__
        if on_standing == "stop":
            assert rows[name].power == "stop", name
        if on_standing == "note":
            assert rows[name].power in ("note", "repair"), name
