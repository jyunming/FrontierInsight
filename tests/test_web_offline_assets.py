"""The web pages work on a machine with no internet access.

Every page loaded its icon font and text fonts from fonts.googleapis.com and its styles from cdn.tailwindcss.com.
Offline (or on a slow first load) a person saw the icons' names as words ("add", "settings", "terminal") and, with no
Tailwind, an unstyled page. The fonts, the icon font and the Tailwind script are now files FI serves itself
(web/static/vendor/). The first tests read the pages; the browser tests load each page with every other host refused
and check that no request left the machine, the styles applied, and no icon is shown as its name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.web_browser_harness import BASE, Site, browser_page

STATIC = Path(__file__).resolve().parent.parent / "web" / "static"
PAGES = sorted(STATIC.glob("*.html"))
_EXTERNAL = re.compile(r"""<(?:script|link)\b[^>]*\b(?:src|href)\s*=\s*["']\s*(?:https?:)?//""", re.I)
_LOCAL_ASSET = re.compile(r"""\b(?:src|href)\s*=\s*["'](/static/[^"'?#]+)""")


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_no_page_loads_a_script_or_stylesheet_from_another_host(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    assert not _EXTERNAL.findall(text), f"{page.name} loads a script or stylesheet from another host"
    assert "fonts.googleapis.com" not in text and "cdn.tailwindcss.com" not in text


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_local_asset_a_page_names_is_there(page: Path) -> None:
    for ref in _LOCAL_ASSET.findall(page.read_text(encoding="utf-8")):
        assert (STATIC / ref.removeprefix("/static/")).is_file(), f"{page.name} names {ref}, which is not shipped"


def test_every_font_the_stylesheet_names_is_there_and_the_icon_font_is_blank_until_loaded() -> None:
    css = (STATIC / "vendor" / "fonts.css").read_text(encoding="utf-8")
    files = re.findall(r"url\(([^)]+)\)", css)
    assert files and all((STATIC / "vendor" / f).is_file() for f in files)
    assert not re.search(r"url\(\s*https?:", css)
    icon = css[css.index("'Material Symbols Outlined'"):]
    assert "font-display: block" in icon.split("}")[0], "the icon font must not show the icon's name while it loads"
    assert ".material-symbols-outlined" in css


def test_the_vendored_assets_ship_in_the_package() -> None:
    manifest = (STATIC.parent.parent / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include web/static *" in manifest


# --- in a browser, offline --------------------------------------------------------------------------------------------

_ROUTES = ["/", "/interview", "/settings", "/jobs", "/skills", "/tools/digest", "/trash", "/compare", "/quest/no-such-quest"]

# Every visible icon: its name, its width and its font size. A name shown as text is several ems wide; an icon is one.
_ICONS_JS = """() => [...document.querySelectorAll('.material-symbols-outlined')]
  .filter((el) => el.offsetParent !== null && el.textContent.trim())
  .map((el) => ({ name: el.textContent.trim(), width: el.getBoundingClientRect().width,
                  size: parseFloat(getComputedStyle(el).fontSize) }))"""


@pytest.mark.slow
def test_every_page_offline_shows_icons_not_their_names_and_is_styled(tmp_path: Path) -> None:
    site = Site(tmp_path / "outputs")
    checked = 0
    with browser_page(site) as page:
        for route in _ROUTES:
            page.goto(f"{BASE}{route}", wait_until="load")
            page.evaluate("() => document.fonts.ready")
            page.wait_for_timeout(300)
            assert page.evaluate("() => document.fonts.check('24px \"Material Symbols Outlined\"')"), route
            icons = page.evaluate(_ICONS_JS)
            checked += len(icons)
            words = [i["name"] for i in icons if i["width"] > 1.6 * i["size"]]
            assert not words, f"{route} shows icon names as words: {words}"
            # Tailwind ran: the page background is the theme's colour, not the browser's transparent default.
            bg = page.evaluate("() => getComputedStyle(document.body).backgroundColor")
            assert bg not in ("rgba(0, 0, 0, 0)", "transparent"), f"{route} is unstyled (Tailwind did not run)"
    assert checked >= 10, f"only {checked} icons were on the pages, so the check proves little"
    assert site.external == [], f"requests left the machine: {site.external}"
