"""Values read through float32 (``np.float32(v).tolist()``) are the trials' values: a match within a relative 2^-24."""

from __future__ import annotations

import copy
import struct
from collections import Counter

from core import trial_runner as tr
from tests.test_manifest_list_attribution import _at, _fabricated, _found, _honest, _study


def _f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def test_values_read_through_float32_are_the_trials_values() -> None:
    vals, per_cell = _study()
    rj = _honest(vals)
    for by in rj["by_m"].values():
        for p in by["by_method"].values():
            p["fdr_values"] = [_f32(x) for x in p["fdr_values"]]
            p["fdr"] = sum(p["fdr_values"]) / 50
    assert _found(rj, per_cell) == []
    # a value moved by more than float32 rounding is still not a trial's
    bad = copy.deepcopy(rj)
    _at(bad)["uncorrected"]["fdr_values"][0] *= 1.001
    assert _fabricated(_found(bad, per_cell))
    assert tr._not_among([0.0], Counter({tr._value_key(0.0): 1})) == []
