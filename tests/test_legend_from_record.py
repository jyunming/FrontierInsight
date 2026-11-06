"""The legend is FI's, written from the figure record: each entry names what tells it apart from the others in its panel,
and two lines that differ in nothing are one entry. Neutral made-up study; no model is called."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from tests.test_frame_figure import _Host  # noqa: E402

PLACEHOLDERS = {"Reference line", "a curve"}


def _legend(h):
    """Every legend text of the figure (one legend per figure, or one per panel)."""
    texts = [t.get_text() for leg in getattr(h.fig, "legends", []) for t in leg.get_texts()]
    texts += [t.get_text() for ax in h.axes for t in (ax.get_legend().get_texts() if ax.get_legend() else [])]
    return [texts]


def test_identical_reference_lines_are_one_entry():
    h = _Host()
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    h.ref(0, "limit")
    h.ref(0, value=3.0)  # the same plan reference typed as a number: the same line
    refs = h.record()["panels"][0]["refs"]
    assert [r["id"] for r in refs] == ["limit"]
    h.finish(h.labels, {})
    assert _legend(h)[0].count("upper limit") == 1


def test_reference_entry_carries_the_plan_name():
    h = _Host()
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    h.ref(0, "limit")
    h.finish(h.labels, {})
    assert "upper limit" in _legend(h)[0]


def test_entries_never_collide_and_never_empty():
    h = _Host(identity=True)  # the table's mean at size=v is v: the drawn values are the table's
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    h.points(0, "slow", [1, 2], [1.0, 2.0], "growth")
    h.ref(0, "limit")
    h.ref(0, at="growth", key="size=1")
    h.finish(h.labels, {})
    texts = _legend(h)[0]
    assert texts and all(t.strip() for t in texts)
    assert len(texts) == len(set(texts))
    assert not PLACEHOLDERS & set(texts) and not any(t.startswith("computed value") for t in texts)
