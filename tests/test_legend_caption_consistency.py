"""FI's check that a figure's legend and caption say what its record draws: true text passes, planted mislabels are
caught. The records are drawn here by FI's own frame (neutral made-up study) and their captions written by FI; no model
is called."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from core.engine import _caption_from_calls  # noqa: E402
from core.figure_text_check import check_figure_text, paper_captions  # noqa: E402
from tests.test_frame_figure import _Host  # noqa: E402


def _drawn(kind: str):
    h = _Host(identity=True)
    if kind == "one_series_one_ref":
        h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
        h.ref(0, "limit")
    elif kind == "two_series_two_refs":
        h.references = {"limit": {"value": 3.0, "label": "upper limit"}, "floor": {"value": 0.5, "label": "lower limit"}}
        h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
        h.points(0, "slow", [1, 2], [1.0, 2.0], "growth")
        h.ref(0, "limit")
        h.ref(0, "floor")
    elif kind == "empty_second_panel":
        h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
        h.ref(0, "limit")
    elif kind == "two_panels":
        h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
        h.line(1, "slow", [1, 2], [1.0, 2.0], "growth")
        h.ref(1, "limit")
    h.finish(h.labels, {})
    rec = h.record()
    legend = rec.get("legend")
    texts = list(legend["entries"]) if legend else None
    caption = _caption_from_calls("f.png", rec, None, n_seeds=None, facts=None)
    return rec, texts, caption


KINDS = ["one_series_one_ref", "two_series_two_refs", "empty_second_panel", "two_panels"]


def _plants(rec, texts, caption):
    """Each planted fault: (name, legend texts, caption)."""
    out = []
    if texts:
        out.append(("drop_entry", texts[:-1], caption))
        out.append(("add_entry", texts + ["Another line"], caption))
        out.append(("rename_to_placeholder", ["Reference line"] + texts[1:], caption))
        out.append(("duplicate_entry", texts + [texts[0]], caption))
    refs = [r for p in rec["panels"] for r in p["refs"]]
    if refs:
        v = f"{refs[0]['value']:.3g}"
        out.append(("change_value", texts, caption.replace(f"at {v}", "at 9.99", 1)))
        out.append(("drop_ref_from_caption", texts, caption.replace("Marked:", "Marked: nothing;", 1)
                    .replace(f"at {v}", "", 1)))
        out.append(("add_ref_to_caption", texts, caption.replace("Marked: ", "Marked: Extra bound at 7; ", 1)))
    empty = [p for p in rec["panels"] if not p["series"] and not p["refs"] and not p["bands"]]
    if empty:
        out.append(("omit_empty_panel", texts, caption.replace(f"Panel {empty[0]['panel']} draws nothing.", "")))
    return out


def test_true_text_passes_on_every_fixture_record() -> None:
    for kind in KINDS:
        rec, texts, caption = _drawn(kind)
        assert check_figure_text(rec, texts, caption) == [], (kind, texts, caption)


def test_planted_mislabel_catch_rate_at_least_95_percent() -> None:
    caught = total = 0
    missed = []
    for kind in KINDS:
        rec, texts, caption = _drawn(kind)
        for name, t, c in _plants(rec, texts, caption):
            total += 1
            if check_figure_text(rec, t, c):
                caught += 1
            else:
                missed.append((kind, name))
    assert total >= 20 and caught / total >= 0.95, (caught, total, missed)


def test_each_plant_kind_is_caught() -> None:
    seen: dict[str, bool] = {}
    for kind in KINDS:
        rec, texts, caption = _drawn(kind)
        for name, t, c in _plants(rec, texts, caption):
            seen[name] = seen.get(name, False) or bool(check_figure_text(rec, t, c))
    assert seen and all(seen.values()), seen


def test_no_false_flag_on_merged_entries_and_registered_curves() -> None:
    from tests.test_registered_curves import _host

    h = _host()
    h._panel(0)[1]["hold"] = {"step": 2}
    h.line(0, "fast", [1, 2, 3], [2.0, 4.0, 6.0], "growth")
    h.ref(0, "law")  # a plan reference drawn as a curve
    h.finish(h.labels, {})
    rec = h.record()
    texts = list(rec["legend"]["entries"]) if rec.get("legend") else None
    caption = _caption_from_calls("f.png", rec, None, n_seeds=None, facts=None)
    assert check_figure_text(rec, texts, caption) == [], (texts, caption)

    h = _Host()
    h.line(0, "fast", [1, 2], [1.0, 2.0], "growth")
    h.ref(0, "limit")
    with pytest.raises(ValueError):
        h.ref(0, value=3.0)  # the same number typed is no reference: refused, one entry
    h.finish(h.labels, {})
    rec = h.record()
    texts = list(rec["legend"]["entries"]) if rec.get("legend") else None
    caption = _caption_from_calls("f.png", rec, None, n_seeds=None, facts=None)
    assert check_figure_text(rec, texts, caption) == [], (texts, caption)


def test_paper_captions_are_read_by_figure_file() -> None:
    md = "Text.\n\n![**Figure 1.** Growth [a] against size. Marked: upper limit at 3.](figures/f1.png)\n"
    assert paper_captions(md) == {"f1.png": "**Figure 1.** Growth [a] against size. Marked: upper limit at 3."}


def test_a_finding_is_disclosed_in_quality_md(tmp_path) -> None:
    import json

    from core import quality_report

    rec, texts, caption = _drawn("one_series_one_ref")
    root = tmp_path / "q"
    (root / ".fi" / "figure_records").mkdir(parents=True)
    (root / "paper").mkdir()
    (root / ".fi" / "figure_records" / "f1.calls.json").write_text(json.dumps(rec), encoding="utf-8")
    wrong = caption.replace("at 3", "at 9")
    (root / "paper" / "paper.md").write_text(f"![{wrong}](figures/f1.png)\n", encoding="utf-8")
    found = quality_report._paper_problems(quality_report._Quest(root))
    assert any("f1.png" in str(r) and "reference line" in str(r) for r in found), found
