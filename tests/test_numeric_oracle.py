"""Tests for what stays of ``core.numeric_oracle``: the read of the results themselves.

The comparison of the paper's numbers with the results is gone (a result is a key FI fills in from its own records,
``core/paper_values.py``, and a typed number that is not the plan's is found by ``core/typed_numbers.py``). What stays is
``flatten_numbers`` (used by the plausibility gate, the statistics check and the engine), ``trivial_reference_findings`` and
``latex_to_arithmetic``.
"""
from __future__ import annotations

from core import numeric_oracle as no


def test_flatten_walks_nested_structures_with_paths() -> None:
    got = dict(no.flatten_numbers({"a": {"b": [1.5, {"c": 2.5}]}, "d": 3.5}))
    assert got == {"a.b[0]": 1.5, "a.b[1].c": 2.5, "d": 3.5}


def test_flatten_drops_booleans_and_zeros() -> None:
    got = dict(no.flatten_numbers({"flag": True, "zero": 0.0, "real": 4.2}))
    assert got == {"real": 4.2}


def test_flatten_keeps_zeros_when_asked() -> None:
    """The range checks need them: a quantity computed as exactly 0 can be the
    bug, and the paper check still leaves zeros out."""
    got = dict(no.flatten_numbers({"flag": True, "zero": 0.0, "real": 4.2}, keep_zero=True))
    assert got == {"zero": 0.0, "real": 4.2}
