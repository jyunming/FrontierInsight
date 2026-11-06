"""A line FI cannot name is refused while a repair of the figure is left (the body's repair is told what to do), and in the
last run (no repair left, ``FI_FIGURE_FALLBACK=1``) it is left out of the figure, recorded in the figure record's ``removed``
and named in QUALITY.md; the rest of the figure is drawn. Neutral made-up study; no model is called."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from core import frame_figure  # noqa: E402
from core.engine import _caption_from_calls  # noqa: E402
from tests.test_frame_figure import _Host  # noqa: E402


@pytest.fixture
def last_run(monkeypatch):
    monkeypatch.setenv(frame_figure.LAST_RUN_ENV, "1")


def test_refused_constant_message_names_the_line_and_the_registration_step(monkeypatch):
    monkeypatch.delenv(frame_figure.LAST_RUN_ENV, raising=False)
    h = _Host(identity=True)
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    with pytest.raises(ValueError) as e:
        h.ref(0, value=0.05)
    assert "0.05" in str(e.value) and "plan" in str(e.value) and "at=" in str(e.value)


def test_after_repair_round_unmatched_constant_is_not_drawn(last_run):
    h = _Host(identity=True)
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    lines_before = len(h.axes[0].lines)
    assert h.ref(0, value=0.05) == []
    rec = h.record()
    assert rec["panels"][0]["refs"] == [] and len(h.axes[0].lines) == lines_before
    assert rec["removed"] == [{"panel": 1, "what": "a line at 0.05",
                               "why": "FI has no record of where its value comes from"}]


def test_curve_without_registration_removed_not_kept_as_unverified(last_run):
    h = _Host(identity=True)
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    assert h.curve(0, "trend", [1, 2], [1.0, 2.0]) == []
    rec = h.record()
    assert all(s["kind"] != "curve" for s in rec["panels"][0]["series"])
    assert [r["what"] for r in rec["removed"]] == ["a curve the figure computed itself"]


def test_removed_line_absent_from_record_legend_caption_and_named_in_quality(last_run):
    h = _Host(identity=True)
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    h.ref(0, "limit")
    h.ref(0, value=0.05)
    h.finish(h.labels, {})
    rec = h.record()
    texts = [t.get_text() for leg in h.fig.legends for t in leg.get_texts()]
    texts += [t.get_text() for ax in h.axes for t in (ax.get_legend().get_texts() if ax.get_legend() else [])]
    assert not any("0.05" in t for t in texts)
    caption = _caption_from_calls("f.png", rec, None, n_seeds=None, facts=None)
    assert "0.05" not in caption and "upper limit" in caption
    assert rec["removed"][0]["what"] == "a line at 0.05"


def test_panel_left_empty_follows_existing_empty_panel_path(last_run):
    from core import analysis_figures

    h = _Host(identity=True)
    h.ref(1, value=0.05)  # the only thing panel 2 draws
    assert analysis_figures._empty_panels(h.axes) == [1, 2]


def test_quality_md_names_removed_line_in_plain_words(tmp_path, monkeypatch):
    from core import engine as eng

    seen = []
    host = eng.Engine.__new__(eng.Engine)
    monkeypatch.setattr(host, "_clear_cannot_repair", lambda check: None, raising=False)
    monkeypatch.setattr(host, "_record_cannot_repair",
                        lambda text, tried, source, check: seen.append((text, check)), raising=False)

    class _Log:
        def warning(self, *a, **k):
            pass

        info = warning

    host._log = _Log()
    record = {"figures": [{"id": "f2", "file": "f2.png", "status": "ok", "removed": [
        {"panel": 1, "what": "a line at 0.05", "why": "FI has no record of where its value comes from"}]}]}
    host._figure_fallbacks({"f2.png": {}}, record)
    assert seen == [("Figure `f2`, panel 1: a line at 0.05 was left out, because FI has no record of where its value "
                     "comes from.", "figure_line_removed")]
