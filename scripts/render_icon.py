"""Render the FrontierInsight app icon to both SVG and PNG.

Single source of truth for the icon geometry. Pillow-only — no
``cairo`` system dep, so this runs cleanly on Windows where
``cairosvg`` would fail. Re-run after any geometry tweak; commit
the regenerated SVG + PNG.

Design (concept B — "compass + spike"):
    - Dark navy background, 1px inner border for marketplace contrast.
    - Compass needle: filled triangle pointing up-and-right in bright
      cyan; mirrored downward half in muted slate so the directionality
      reads.
    - Pivot dot at the needle's center in white.
    - Chart spike: 3-point polyline climbing out of the needle's tip
      toward the upper-right corner, same cyan.

Run:
    python scripts/render_icon.py

Writes to ``vscode-frontier-insight/images/``:
    - ``icon.svg`` (vector source, human-editable for color tweaks).
    - ``icon.png`` (128 × 128 RGBA, the file VSCode marketplace
      consumes via ``package.json::icon``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw


SIZE = 128

# Palette. Hex strings for SVG; (R, G, B, A) tuples for PIL.
BG_HEX = "#0B1729"
ACCENT_HEX = "#22D3EE"
MUTED_HEX = "#475569"
PIVOT_HEX = "#F8FAFC"
BORDER_HEX = "#1E293B"


def _hex_to_rgba(h: str, alpha: int = 255) -> tuple[int, int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


# ---- Geometry (canvas coordinates, top-left origin) ---------------------
# Compass needle is a thin lozenge built from two triangles sharing a base
# at the pivot. Upper triangle is the "north" (accent), lower is "south"
# (muted). Needle is tilted ~30° clockwise so the tip emerges at upper-
# right; the spike polyline picks up from that tip and climbs to the
# top-right corner.

PIVOT = (64, 64)
NEEDLE_TIP_NORTH = (95, 33)
NEEDLE_TIP_SOUTH = (33, 95)
NEEDLE_FLANK_LEFT = (58, 70)
NEEDLE_FLANK_RIGHT = (70, 58)

# Chart spike — emerges from the needle's north tip and zig-zags up.
SPIKE: Sequence[tuple[int, int]] = (
    NEEDLE_TIP_NORTH,    # (95, 33)
    (105, 38),           # small dip
    (112, 22),           # peak
    (120, 28),           # back to baseline
)

# Pivot dot radius (white circle at the needle's center).
PIVOT_R = 5


# ---- SVG renderer -------------------------------------------------------

def _emit_svg() -> str:
    """Return the SVG XML for the icon as a string."""
    def pts(seq: Sequence[tuple[int, int]]) -> str:
        return " ".join(f"{x},{y}" for x, y in seq)

    needle_north = (PIVOT, NEEDLE_FLANK_LEFT, NEEDLE_TIP_NORTH, NEEDLE_FLANK_RIGHT)
    needle_south = (PIVOT, NEEDLE_FLANK_LEFT, NEEDLE_TIP_SOUTH, NEEDLE_FLANK_RIGHT)

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" height="{SIZE}">\n'
        f'  <rect width="{SIZE}" height="{SIZE}" fill="{BG_HEX}"/>\n'
        f'  <rect x="0.5" y="0.5" width="{SIZE - 1}" height="{SIZE - 1}" '
        f'fill="none" stroke="{BORDER_HEX}" stroke-width="1"/>\n'
        f'  <polygon points="{pts(needle_south)}" fill="{MUTED_HEX}"/>\n'
        f'  <polygon points="{pts(needle_north)}" fill="{ACCENT_HEX}"/>\n'
        f'  <polyline points="{pts(SPIKE)}" '
        f'stroke="{ACCENT_HEX}" stroke-width="3" '
        f'stroke-linecap="round" stroke-linejoin="round" fill="none"/>\n'
        f'  <circle cx="{PIVOT[0]}" cy="{PIVOT[1]}" r="{PIVOT_R}" '
        f'fill="{PIVOT_HEX}"/>\n'
        f'</svg>\n'
    )


# ---- PNG renderer -------------------------------------------------------

def _emit_png() -> Image.Image:
    """Render the icon to a fresh ``Image`` (RGBA, SIZE × SIZE)."""
    img = Image.new("RGBA", (SIZE, SIZE), _hex_to_rgba(BG_HEX))
    draw = ImageDraw.Draw(img)

    draw.rectangle(
        (0, 0, SIZE - 1, SIZE - 1),
        outline=_hex_to_rgba(BORDER_HEX),
        width=1,
    )

    # South half first so the north half draws on top at the pivot.
    draw.polygon(
        (PIVOT, NEEDLE_FLANK_LEFT, NEEDLE_TIP_SOUTH, NEEDLE_FLANK_RIGHT),
        fill=_hex_to_rgba(MUTED_HEX),
    )
    draw.polygon(
        (PIVOT, NEEDLE_FLANK_LEFT, NEEDLE_TIP_NORTH, NEEDLE_FLANK_RIGHT),
        fill=_hex_to_rgba(ACCENT_HEX),
    )

    # Spike polyline — PIL doesn't have polyline, so emit segments.
    for (x0, y0), (x1, y1) in zip(SPIKE[:-1], SPIKE[1:]):
        draw.line((x0, y0, x1, y1), fill=_hex_to_rgba(ACCENT_HEX), width=3)

    # Pivot dot.
    draw.ellipse(
        (PIVOT[0] - PIVOT_R, PIVOT[1] - PIVOT_R,
         PIVOT[0] + PIVOT_R, PIVOT[1] + PIVOT_R),
        fill=_hex_to_rgba(PIVOT_HEX),
    )
    return img


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / "vscode-frontier-insight" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / "icon.svg"
    svg_path.write_text(_emit_svg(), encoding="utf-8")
    print(f"wrote {svg_path}")

    png_path = out_dir / "icon.png"
    _emit_png().save(png_path, "PNG")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
