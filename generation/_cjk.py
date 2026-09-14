"""Chinese, Japanese and Korean text in the LaTeX outputs.

pdflatex with ``inputenc`` stops at the first CJK character ("Unicode
character 陳 (U+9673) not set up for use with LaTeX"), whether it sits in
the title, the body or the author line. XeLaTeX with ``xeCJK`` and an
installed CJK font prints it. The paper and poster generators ask this
module whether their text holds CJK, which font to use, and where XeLaTeX
is; with no font they fall back (paper: the HTML render; poster: a skip
diagnostic that says why).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

_HAN = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f"
_KANA = "\u3040-\u30ff\u31f0-\u31ff"
_HANGUL = "\u1100-\u11ff\u3130-\u318f\uac00-\ud7af"
# CJK punctuation and full-width forms stop pdflatex just the same.
_CJK_RE = re.compile("[" + _HAN + _KANA + _HANGUL + "\u3000-\u303f\uff00-\uffef]")
_HAN_RE = re.compile("[" + _HAN + "]")
_KANA_RE = re.compile("[" + _KANA + "]")
_HANGUL_RE = re.compile("[" + _HANGUL + "]")

# Fonts that cover each language, best first. Noto/Source Han cover all
# four; the system fonts come with Windows and macOS.
_PREFERRED_FONTS: dict[str, tuple[str, ...]] = {
    "zh-tw": (
        "Noto Sans CJK TC", "Noto Sans TC", "Source Han Sans TC",
        "Microsoft JhengHei", "PingFang TC", "WenQuanYi Zen Hei",
    ),
    "zh-cn": (
        "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC",
        "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei",
    ),
    "ja": (
        "Noto Sans CJK JP", "Noto Sans JP", "Source Han Sans JP",
        "Yu Gothic", "Meiryo", "Hiragino Sans",
    ),
    "ko": (
        "Noto Sans CJK KR", "Noto Sans KR", "Source Han Sans KR",
        "Malgun Gothic", "Apple SD Gothic Neo",
    ),
}

# Where Windows keeps its CJK fonts, for a machine without ``fc-list``.
_WINDOWS_FONT_FILES = {
    "Microsoft JhengHei": "msjh.ttc",
    "Microsoft YaHei": "msyh.ttc",
    "Yu Gothic": "YuGothR.ttc",
    "Meiryo": "meiryo.ttc",
    "Malgun Gothic": "malgun.ttf",
}


def has_cjk(text: str) -> bool:
    """Whether ``text`` holds a character pdflatex cannot set."""
    return bool(_CJK_RE.search(text or ""))


def cjk_language(text: str) -> str | None:
    """``ko``, ``ja``, ``zh-tw`` or ``zh-cn`` for text with CJK in it, else
    ``None``. Hangul means Korean and kana means Japanese; Chinese that Big5
    can encode is taken as Traditional, anything else as Simplified."""
    if not has_cjk(text):
        return None
    if _HANGUL_RE.search(text):
        return "ko"
    if _KANA_RE.search(text):
        return "ja"
    try:
        "".join(_HAN_RE.findall(text)).encode("big5")
    except UnicodeEncodeError:
        return "zh-cn"
    return "zh-tw"


@lru_cache(maxsize=None)
def _installed_families(lang: str) -> frozenset[str]:
    """Font families fontconfig says cover ``lang``. XeLaTeX finds fonts
    through fontconfig too, so a family listed here is one it can load."""
    exe = shutil.which("fc-list")
    if exe is None:
        return frozenset()
    try:
        out = subprocess.run(
            [exe, f":lang={lang}", "family"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60,
        ).stdout or ""
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(
        name.strip() for row in out.splitlines() for name in row.split(",") if name.strip()
    )


def _font_file_present(name: str) -> bool:
    if sys.platform != "win32":
        return False
    filename = _WINDOWS_FONT_FILES.get(name)
    fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    return bool(filename) and (fonts / filename).is_file()


def find_cjk_font(text: str) -> str | None:
    """The font family to set ``text``'s CJK characters in, or ``None`` when
    the text has none or no suitable font is installed."""
    lang = cjk_language(text)
    if lang is None:
        return None
    families = _installed_families(lang)
    for name in _PREFERRED_FONTS[lang]:
        if name in families:
            return name
    for name in _PREFERRED_FONTS[lang]:
        if _font_file_present(name):
            return name
    return None


def find_xelatex(engine: tuple[str, str] | None) -> tuple[str, str] | None:
    """A XeTeX engine to use in place of ``engine``. Tectonic is XeTeX
    underneath, so it serves as is. Otherwise ``xelatex`` on PATH, or next
    to the pdflatex that was found (MiKTeX's bin directory is often missing
    from a child process's PATH)."""
    if engine is not None and engine[0] == "tectonic":
        return engine
    found = shutil.which("xelatex")
    if found:
        return ("xelatex", found)
    if engine is not None:
        path = Path(engine[1])
        sibling = path.with_name("xelatex" + path.suffix)
        if sibling.is_file():
            return ("xelatex", str(sibling))
    return None


def xecjk_preamble(font: str) -> str:
    """LaTeX that sets CJK characters in ``font`` under XeLaTeX."""
    return "\\usepackage{xeCJK}\n" + "".join(
        f"\\setCJK{kind}font{{{font}}}\n" for kind in ("main", "sans", "mono")
    )
