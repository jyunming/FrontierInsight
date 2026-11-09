"""The closed-choice fields of the search block have one list (core/optimisation_plan.py ``CLOSED_CHOICES``): the checks
refuse a wrong value by quoting it, and the ask for the block shows it. Neutral toy topic only (a heat sink)."""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest

from core import optimisation_plan as op
from tests.test_search_no_effect import _block


def _with(field: str, value: Any) -> dict[str, Any]:
    block = copy.deepcopy(_block())
    block["improvement_tolerance"] = {"value": 0.1, "mode": "absolute"}
    node: Any = block
    parts = field.split(".")
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if part.endswith("[]"):
            node = node[part[:-2]][0]
            continue
        if last:
            node[part] = value
        else:
            node = node[part]
    if field.endswith("[].kind") and value == "choice":
        block["design_variables"][0]["values"] = [1.0, 2.0, 3.0]
    return block


def test_every_closed_choice_field_refuses_a_wrong_value_quoting_the_tables_values() -> None:
    for field, values in op.CLOSED_CHOICES:
        fixed, why = op.normalize(_with(field, "no such value"))
        assert fixed is None and why, field
        assert all(v in why for v in values) or all(v in why for v in values if "<" not in v), (field, why)


def test_every_plain_value_in_the_table_is_accepted_by_the_checks() -> None:
    for field, values in op.CLOSED_CHOICES:
        for value in (v for v in values if "<" not in v):
            fixed, why = op.normalize(_with(field, value))
            assert why is None, (field, value, why)


def test_no_check_names_its_own_list_of_values_outside_the_table() -> None:
    source = Path(op.__file__).read_text(encoding="utf-8")
    for line_no, line in enumerate(source.splitlines(), 1):
        if re.search(r"must be (exactly )?one of [^t]|must be `(minimise|absolute)`", line):
            assert "{', '.join(values)}" in line or "{', '.join(KINDS)}" in line, f"line {line_no} writes a list of values of its own: {line.strip()}"
