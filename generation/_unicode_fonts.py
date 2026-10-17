"""Which font sets each character of a paper typeset with XeLaTeX.

The paper's PDF is typeset with a Unicode TeX engine (XeLaTeX) whenever one is
installed, so a character is set as itself instead of being mapped to something
else. XeLaTeX has no font fallback of its own: a character the text font lacks
is dropped from the page with only a line in the log. This module reads the
character tables (cmap) of the fonts a TeX install and the system ship, so FI
knows before the compile which characters the text fonts set, which a fallback
font sets (the paper source wraps them in that font), which a code font sets
(code is never rewritten, so the code font must hold all of its characters), and
which no font here sets (the paper is then not typeset, never printed with a
character missing).

Everything fails open: with no fontTools, or no font file found, :func:`find_fonts`
returns a plan that knows nothing, and the check after the compile (pandoc's
"Missing character" warnings) is what keeps a dropped character from passing.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Font:
    """One font file and what it sets."""

    name: str
    path: Path
    covers: frozenset[int]
    bold: Path | None = None
    italic: Path | None = None

    def fontspec_file_and_options(self) -> tuple[str, str]:
        """The file name and ``Path=…/[,BoldFont=…][,ItalicFont=…]`` for fontspec, so
        the file is loaded by its path (no name lookup that could find another font of
        the same name)."""
        parts = [f"Path={self.path.parent.as_posix()}/"]
        if self.bold is not None and self.bold.parent == self.path.parent:
            parts.append(f"BoldFont={self.bold.name}")
        if self.italic is not None and self.italic.parent == self.path.parent:
            parts.append(f"ItalicFont={self.italic.name}")
        return self.path.name, ",".join(parts)


@dataclass(frozen=True)
class FontPlan:
    """The fonts found for one engine: the text fonts every template may use (a
    character is set as it is only when all of them hold it), the fallback fonts
    in order of preference, and the code fonts in order of preference."""

    text: tuple[Font, ...] = ()
    fallbacks: tuple[Font, ...] = ()
    monos: tuple[Font, ...] = field(default_factory=tuple)

    @property
    def known(self) -> bool:
        return bool(self.text)

    def in_text_fonts(self, code: int) -> bool:
        if not self.text:
            # Unknown fonts: Latin Modern sets ASCII and the Latin letters at least.
            return code < 0x250
        return all(code in font.covers for font in self.text)

    def fallback_for(self, codes: list[int]) -> Font | None:
        """The first fallback font that sets every one of ``codes``."""
        for font in self.fallbacks:
            if all(code in font.covers for code in codes):
                return font
        return None


# The text fonts of the templates under XeLaTeX: fontspec's Latin Modern Roman and
# Sans, and TeX Gyre Heros, the sans the whitepaper and report templates set.
_TEXT_FILES = ("lmroman10-regular.otf", "lmsans10-regular.otf", "texgyreheros-regular.otf")
# (name, regular, bold, italic); best first. Latin Modern Math matches the look of
# the text for symbols; DejaVu covers Greek, Cyrillic, IPA, modifier letters,
# arrows, box drawing and most symbols; the rest fill what those lack.
_FALLBACK_FILES: tuple[tuple[str, str, str | None, str | None], ...] = (
    ("Latin Modern Math", "latinmodern-math.otf", None, None),
    ("DejaVu Serif", "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSerif-Italic.ttf"),
    ("DejaVu Sans", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf"),
    ("FreeSerif", "FreeSerif.otf", "FreeSerifBold.otf", "FreeSerifItalic.otf"),
    ("Noto Serif", "NotoSerif-Regular.ttf", "NotoSerif-Bold.ttf", "NotoSerif-Italic.ttf"),
    ("Noto Sans", "NotoSans-Regular.ttf", "NotoSans-Bold.ttf", "NotoSans-Italic.ttf"),
    ("Noto Sans Armenian", "NotoSansArmenian-Regular.ttf", "NotoSansArmenian-Bold.ttf", None),
    ("Noto Sans Georgian", "NotoSansGeorgian-Regular.ttf", "NotoSansGeorgian-Bold.ttf", None),
    ("Noto Sans Math", "NotoSansMath-Regular.ttf", None, None),
    ("Noto Sans Symbols", "NotoSansSymbols-Regular.ttf", None, None),
    ("Noto Sans Symbols 2", "NotoSansSymbols2-Regular.ttf", None, None),
    ("Segoe UI Symbol", "seguisym.ttf", None, None),
    ("Symbola", "Symbola.ttf", None, None),
)
# Code fonts; Latin Modern Mono is fontspec's own, so it stays unless the code holds
# a character it lacks.
_MONO_FILES: tuple[tuple[str, str, str | None, str | None], ...] = (
    ("Latin Modern Mono", "lmmono10-regular.otf", "lmmonolt10-bold.otf", "lmmono10-italic.otf"),
    ("DejaVu Sans Mono", "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf", "DejaVuSansMono-Oblique.ttf"),
    ("Noto Sans Mono", "NotoSansMono-Regular.ttf", "NotoSansMono-Bold.ttf", None),
    ("Cascadia Mono", "CascadiaMono.ttf", None, None),
    ("Consolas", "consola.ttf", "consolab.ttf", "consolai.ttf"),
)

# Scripts XeLaTeX would not set correctly without more than a font: written right to
# left (Hebrew, Arabic, Syriac, Thaana, N'Ko) or shaped (the Indic scripts, Thai,
# Lao, Tibetan, Myanmar, Khmer). Such a paper is not typeset, never set wrongly.
_UNSUPPORTED_SCRIPTS: tuple[tuple[int, int], ...] = (
    (0x0590, 0x08FF), (0x0900, 0x0DFF), (0x0E00, 0x0FFF), (0x1000, 0x109F),
    (0x1780, 0x17FF), (0xFB1D, 0xFDFF), (0xFE70, 0xFEFE),
)


def unsupported_script(code: int) -> bool:
    return any(lo <= code <= hi for lo, hi in _UNSUPPORTED_SCRIPTS)


def _font_dirs(engine_path: str | None) -> list[Path]:
    """Where TeX installs and the system keep fonts, the engine's own tree first."""
    dirs: list[Path] = []
    if engine_path:
        for parent in list(Path(engine_path).resolve().parents)[:4]:
            dirs += [parent / "fonts", parent / "texmf-dist" / "fonts"]
    home = Path.home()
    if sys.platform == "win32":
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        dirs.append(windir / "Fonts")
        for env in ("LOCALAPPDATA", "APPDATA"):
            base = os.environ.get(env)
            if base:
                dirs += [Path(base) / "MiKTeX" / "fonts", Path(base) / "Microsoft" / "Windows" / "Fonts"]
    elif sys.platform == "darwin":
        dirs += [Path("/Library/Fonts"), Path("/System/Library/Fonts"), home / "Library" / "Fonts",
                 Path("/Library/TeX/Root/texmf-dist/fonts")]
    dirs += [Path("/usr/share/texlive/texmf-dist/fonts"), Path("/usr/share/texmf/fonts"),
             Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
             home / ".fonts", home / ".local" / "share" / "fonts"]
    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        key = str(d).lower() if sys.platform == "win32" else str(d)
        if key not in seen and d.is_dir():
            seen.add(key)
            out.append(d)
    return out


def _locate(names: set[str], dirs: list[Path]) -> dict[str, Path]:
    """The first file found for each of ``names`` (compared without case)."""
    wanted = {n.lower() for n in names}
    found: dict[str, Path] = {}
    for root in dirs:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fname in filenames:
                low = fname.lower()
                if low in wanted and low not in found:
                    found[low] = Path(dirpath) / fname
            if len(found) == len(wanted):
                return found
    return found


def _cmap(path: Path) -> frozenset[int] | None:
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return None
    try:
        font = TTFont(str(path), lazy=True)
        try:
            return frozenset(font.getBestCmap() or ())
        finally:
            font.close()
    except Exception:  # noqa: BLE001 — an unreadable font is simply not used
        return None


def _fonts(table, located: dict[str, Path]) -> tuple[Font, ...]:
    fonts: list[Font] = []
    for name, regular, bold, italic in table:
        path = located.get(regular.lower())
        if path is None:
            continue
        covers = _cmap(path)
        if covers is None:
            continue
        fonts.append(Font(
            name=name, path=path, covers=covers,
            bold=located.get(bold.lower()) if bold else None,
            italic=located.get(italic.lower()) if italic else None,
        ))
    return tuple(fonts)


@lru_cache(maxsize=8)
def find_fonts(engine_path: str | None) -> FontPlan:
    """The fonts for the engine at ``engine_path`` (cached: a font install does not
    change during a run). Never raises; an empty plan when nothing can be read."""
    try:
        names = set(_TEXT_FILES)
        for table in (_FALLBACK_FILES, _MONO_FILES):
            for _name, regular, bold, italic in table:
                names.update(n for n in (regular, bold, italic) if n)
        located = _locate(names, _font_dirs(engine_path))
        text: list[Font] = []
        for fname in _TEXT_FILES:
            path = located.get(fname.lower())
            covers = _cmap(path) if path is not None else None
            if covers is not None:
                text.append(Font(name=fname, path=path, covers=covers))
        return FontPlan(
            text=tuple(text),
            fallbacks=_fonts(_FALLBACK_FILES, located),
            monos=_fonts(_MONO_FILES, located),
        )
    except Exception:  # noqa: BLE001 — fail open; the compile's own check still holds
        return FontPlan()
