"""Pins the shape of the VSCode extension icon assets.

Asserts the icon contract so a future asset rework can't silently
drop the file or break the package.json wiring without a test
failure (the marketplace gallery falls back to a default placeholder
square when the icon is missing).
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
EXT_ROOT = REPO_ROOT / "vscode-frontier-insight"
ICON_PNG = EXT_ROOT / "images" / "icon.png"
ICON_SVG = EXT_ROOT / "images" / "icon.svg"


def test_icon_png_and_svg_both_exist() -> None:
    """The renderer must emit both files so future tweaks can edit
    the SVG (the human-editable source) without losing the PNG
    (the file VSCode marketplace actually consumes)."""
    assert ICON_PNG.exists(), f"missing {ICON_PNG}"
    assert ICON_SVG.exists(), f"missing {ICON_SVG}"


def test_icon_png_is_128x128_rgba() -> None:
    """VSCode marketplace consumes 128×128 PNG. RGBA so the dark
    navy bg renders cleanly against light marketplace themes."""
    with Image.open(ICON_PNG) as im:
        size = im.size
        mode = im.mode
    assert size == (128, 128), (
        f"icon.png must be 128×128 for the VSCode marketplace; got {size}"
    )
    assert mode == "RGBA", (
        f"icon.png must be RGBA so the bg alpha is explicit; got {mode}"
    )


def test_package_json_references_icon() -> None:
    """``package.json::icon`` must point at the rendered PNG with
    the marketplace-conventional relative path."""
    pkg = json.loads((EXT_ROOT / "package.json").read_text(encoding="utf-8"))
    assert pkg.get("icon") == "images/icon.png", (
        f"package.json::icon must be 'images/icon.png'; got {pkg.get('icon')!r}"
    )


def test_both_readmes_reference_the_icon() -> None:
    """Both the root README and the extension README must reference
    the icon image. Root uses the extension-rooted path; the
    extension's own README uses the local relative path."""
    root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "vscode-frontier-insight/images/icon.png" in root_readme, (
        "README.md (repo root) must reference the icon via "
        "vscode-frontier-insight/images/icon.png so GitHub renders it"
    )
    ext_readme = (EXT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "images/icon.png" in ext_readme, (
        "vscode-frontier-insight/README.md must reference the icon via "
        "the local relative path images/icon.png"
    )
