"""CJK detection, font choice and XeLaTeX discovery (``generation/_cjk.py``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from generation import _cjk


@pytest.mark.parametrize(
    "text, lang",
    [
        ("陳建明", "zh-tw"),
        ("國立台灣大學", "zh-tw"),
        ("这是简体中文", "zh-cn"),
        ("日本語のテキスト", "ja"),
        ("한국어 논문", "ko"),
        ("Jane Chen, R&D Lab", None),
        ("", None),
    ],
)
def test_cjk_language(text, lang):
    assert _cjk.cjk_language(text) == lang
    assert _cjk.has_cjk(text) is (lang is not None)


def test_full_width_punctuation_stops_pdflatex_too():
    assert _cjk.has_cjk("Results（preliminary）")


def test_find_cjk_font_prefers_noto_then_the_system_font(monkeypatch):
    installed = {
        "zh-tw": {"Microsoft JhengHei", "MingLiU"},
        "zh-cn": {"Microsoft YaHei"},
        "ko": {"Malgun Gothic"},
    }
    monkeypatch.setattr(_cjk, "_installed_families", lambda lang: frozenset(installed.get(lang, ())))
    monkeypatch.setattr(_cjk, "_font_file_present", lambda name: False)
    assert _cjk.find_cjk_font("陳建明") == "Microsoft JhengHei"
    assert _cjk.find_cjk_font("这是简体") == "Microsoft YaHei"
    assert _cjk.find_cjk_font("한국어") == "Malgun Gothic"
    assert _cjk.find_cjk_font("日本語のテキスト") is None
    assert _cjk.find_cjk_font("plain English") is None
    installed["zh-tw"].add("Noto Sans CJK TC")
    assert _cjk.find_cjk_font("陳建明") == "Noto Sans CJK TC"


def test_find_cjk_font_checks_the_windows_font_files_without_fc_list(monkeypatch):
    monkeypatch.setattr(_cjk, "_installed_families", lambda lang: frozenset())
    monkeypatch.setattr(_cjk, "_font_file_present", lambda name: name == "Microsoft JhengHei")
    assert _cjk.find_cjk_font("陳建明") == "Microsoft JhengHei"


def test_installed_families_reads_every_name_fc_list_prints(monkeypatch):
    monkeypatch.setattr(_cjk.shutil, "which", lambda name: "/fake/fc-list" if name == "fc-list" else None)

    def fake_run(cmd, **_kw):
        assert cmd == ["/fake/fc-list", ":lang=zh-tw", "family"]
        return SimpleNamespace(stdout="Microsoft JhengHei,微軟正黑體\nNoto Sans CJK TC\n\n", returncode=0)

    monkeypatch.setattr(_cjk.subprocess, "run", fake_run)
    _cjk._installed_families.cache_clear()
    try:
        assert _cjk._installed_families("zh-tw") == {"Microsoft JhengHei", "微軟正黑體", "Noto Sans CJK TC"}
    finally:
        _cjk._installed_families.cache_clear()


def test_find_xelatex(monkeypatch, tmp_path):
    assert _cjk.find_xelatex(("tectonic", "/t/tectonic")) == ("tectonic", "/t/tectonic")
    monkeypatch.setattr(_cjk.shutil, "which", lambda name: "/bin/xelatex" if name == "xelatex" else None)
    assert _cjk.find_xelatex(("pdflatex", "/bin/pdflatex")) == ("xelatex", "/bin/xelatex")
    monkeypatch.setattr(_cjk.shutil, "which", lambda name: None)
    pdflatex = tmp_path / "pdflatex.exe"
    pdflatex.write_bytes(b"")
    (tmp_path / "xelatex.exe").write_bytes(b"")
    assert _cjk.find_xelatex(("pdflatex", str(pdflatex))) == ("xelatex", str(tmp_path / "xelatex.exe"))
    assert _cjk.find_xelatex(None) is None


def test_xecjk_preamble_sets_all_three_families():
    tex = _cjk.xecjk_preamble("Microsoft JhengHei")
    assert tex.startswith("\\usepackage{xeCJK}\n")
    for kind in ("main", "sans", "mono"):
        assert f"\\setCJK{kind}font{{Microsoft JhengHei}}" in tex
