"""A drawing FI made and the body changed afterwards is a custom drawing, whatever property the body changed.

FI keeps the whole drawable state of every artist its calls create (data, the transform, visibility, alpha, the clip, path
effects, z-order, ...) and compares it after the body returned. These cases change one property each of a reference line, a
data line, a band, bars and a note; an untouched body (bars grouped side by side, error bars, a band, an image, a note at a bar)
stays verified."""
from __future__ import annotations

import json

import matplotlib
import pytest

matplotlib.use("Agg")

from tests.test_analysis_figures import _calls_run, _drawn, _LimitTable, frame5  # noqa: E402,F401


def _raw(frame5, panel=0):
    return json.loads((frame5 / "g.calls.json").read_text(encoding="utf-8"))["panels"][panel]["raw"]


def _with_ref(tmp_path, frame5, change):
    def body(fig, table):
        _drawn(fig, table, 0)
        _drawn(fig, table, 1)
        line = fig.ref(0, "limit")
        change(fig, line)

    out = _calls_run(tmp_path, body, table=_LimitTable())
    assert out["status"] == "ok", out
    return _raw(frame5)


def _with_series(tmp_path, frame5, change):
    def body(fig, table):
        _drawn(fig, table, 0)
        _drawn(fig, table, 1)
        change(fig, fig.axes[0].lines[0])

    out = _calls_run(tmp_path, body)
    assert out["status"] == "ok", out
    return _raw(frame5)


def _shift(fig, art):
    import matplotlib.transforms as mt

    art.set_transform(art.get_transform() + mt.Affine2D().translate(0, 40))


def _offset_effect(fig, art):
    from matplotlib import patheffects

    art.set_path_effects([patheffects.withStroke(linewidth=1, foreground="w"), patheffects.SimpleLineShadow(offset=(0, 30))])


def _clip(fig, art):
    from matplotlib.transforms import Bbox, TransformedBbox

    art.set_clip_box(TransformedBbox(Bbox([[0, 0], [0.1, 0.1]]), fig.axes[0].transAxes))


CHANGES = {
    "shortened": lambda fig, a: a.set_xdata([0.5, 1.0]),
    "moved by its transform": _shift,
    "hidden": lambda fig, a: a.set_visible(False),
    "made transparent": lambda fig, a: a.set_alpha(0.0),
    "clipped to a corner": _clip,
    "offset by a path effect": _offset_effect,
    "sent behind the axes": lambda fig, a: a.set_zorder(-10),
    "removed": lambda fig, a: a.remove(),
}


def test_the_known_positive_a_reference_bent_into_a_slope_is_caught(tmp_path, frame5):
    assert _with_ref(tmp_path, frame5, lambda fig, line: line.set_ydata([1.5, 9.0])) is True


def test_an_untouched_reference_and_series_stay_verified(tmp_path, frame5):
    assert _with_ref(tmp_path, frame5, lambda fig, line: None) is False
    assert _with_series(tmp_path, frame5, lambda fig, line: None) is False


@pytest.mark.parametrize("name", sorted(CHANGES))
def test_a_reference_line_changed_after_fi_drew_it_is_a_custom_drawing(tmp_path, frame5, name):
    assert _with_ref(tmp_path, frame5, CHANGES[name]) is True


@pytest.mark.parametrize("name", sorted(set(CHANGES) - {"shortened", "removed"}) + ["data moved"])  # a panel left empty fails
def test_a_data_line_changed_after_fi_drew_it_is_a_custom_drawing(tmp_path, frame5, name):
    change = CHANGES.get(name) or (lambda fig, a: a.set_ydata([5.0, 6.0]))
    assert _with_series(tmp_path, frame5, change) is True


def test_the_other_panel_stays_verified(tmp_path, frame5):
    _with_series(tmp_path, frame5, lambda fig, a: a.set_visible(False))
    assert _raw(frame5, 1) is False


def test_a_band_and_error_bars_changed_after_fi_drew_them_are_caught(tmp_path, frame5):
    def band(fig, table):
        _drawn(fig, table, 0)
        _drawn(fig, table, 1)
        fig.band(0, "fast", [1, 2], [0.5, 1.5], [1.5, 2.5], "growth", "other").set_alpha(0.0)

    def bars(fig, table):
        fig.line(0, "fast", [1, 2], [1.0, 2.0], "growth", yerr=[0.1, 0.2])
        _drawn(fig, table, 1)
        for coll in fig.axes[0].collections:
            coll.set_visible(False)  # the error bars' segments

    for body in (band, bars):
        out = _calls_run(tmp_path, body)
        assert out["status"] == "ok", out
        assert _raw(frame5) is True


def test_a_note_moved_by_the_body_is_caught(tmp_path, frame5):
    def body(fig, table):
        _drawn(fig, table, 0)
        _drawn(fig, table, 1)
        fig.note(0, x=1.5, y=1.5, what="mean growth: {growth}", key="size=1").set_position((1.9, 1.9))

    out = _calls_run(tmp_path, body)
    assert out["status"] == "ok", out
    assert _raw(frame5) is True


def test_a_bar_changed_before_a_later_bars_call_stays_caught(tmp_path, frame5):
    """FI moves the earlier bars when a later bars call groups them; the body's change made before that move is not erased."""
    def body(fig, table):
        first = fig.bars(0, "fast", [1, 2], [1.0, 2.0], "growth")
        first[0].set_height(9.0)
        fig.bars(0, "slow", [1, 2], [1.0, 2.0], "growth")
        _drawn(fig, table, 1)

    out = _calls_run(tmp_path, body)
    assert out["status"] == "ok", out
    assert _raw(frame5) is True


def test_an_untouched_body_of_every_call_kind_stays_verified(tmp_path, frame5):
    """Bars grouped side by side (FI moves them twice), error bars, a band, a reference, an image and notes, one at a bar."""
    def body(fig, table):
        _drawn(fig, table, 0)
        fig.line(0, "slow", [1, 2], [2.0, 1.0], "growth", yerr=[0.1, 0.2])
        fig.band(0, "fast", [1, 2], [0.5, 1.5], [1.5, 2.5], "growth", "other")
        fig.ref(0, "limit")
        fig.note(0, what="mean growth: {growth}", key="size=1")
        fig.bars(1, "fast", [1, 2], [1.0, 2.0], "growth", yerr=[0.1, 0.1])
        fig.note(1, x=1.0, y=1.0, what="mean growth: {growth}", key="size=1")
        fig.bars(1, "slow", [1, 2], [1.0, 2.0], "growth")
        fig.image(1, [[1, 2], [3, 4]])

    out = _calls_run(tmp_path, body, table=_LimitTable())
    assert out["status"] == "ok", out
    assert out["declared"]["raw_panels"] == [], out


def test_a_change_inside_a_transform_the_artist_holds_is_seen():
    """The state compares the matrix of every transform an artist holds beyond the axes' own, so a transform changed in place
    (not replaced) is a change too; the axes' own transforms change with the limits and are compared by identity."""
    import matplotlib.pyplot as plt
    import matplotlib.transforms as mt

    from core.frame_figure import _drawn_state

    fig, ax = plt.subplots()
    try:
        step = mt.Affine2D()
        line, = ax.plot([1, 2], [1, 2], transform=step + ax.transData)
        before = _drawn_state(line)
        ax.set_xlim(0, 10)  # the limits move the axes' own transforms: not a change of the artist
        assert _drawn_state(line) == before
        step.translate(0, 40)
        assert _drawn_state(line) != before
    finally:
        plt.close(fig)
