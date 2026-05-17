r"""Pin the LaTeX package invariants in templates/paper/*/template.tex.

These templates are consumed by pandoc; every one of them must:

1. Load ``longtable`` + ``array`` — pandoc renders markdown tables as
   ``\begin{longtable}[]{...}``, and without the package the compile
   dies on ``Environment longtable undefined``.

2. Declare ``\newcounter{none}`` — pandoc prefixes caption-less
   tables with ``\def\LTcaptype{none}``; ``longtable`` then calls
   ``\refstepcounter{\LTcaptype}`` internally, so the counter must
   exist or the compile dies on ``No counter 'none' defined``.

3. NOT load ``authblk`` together with ``titling`` — when both patch
   ``\author`` and ``\maketitle``, LaTeX hits ``\author[N]{...}``
   (authblk's optional-affiliation form) and reports ``Missing
   \begin{document}`` because titling's redefinition shadows it.

A real EUV-stochastics quest hit all three failure modes in
succession. This module is the regression guard so a future template
edit (or a new template) doesn't silently re-introduce any of them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO_ROOT / "templates" / "paper"

# Every format we ship a template for. Treat this list as the canonical
# enumeration — if a new format lands, it gets added here AND has to
# pass all three invariants below.
EXPECTED_FORMATS = [
    "essay",
    "generic",
    "iclr",
    "ieee_access",
    "nature_mi",
    "neurips",
    "policy_brief",
    "report",
    "whitepaper",
]


def _all_templates() -> list[Path]:
    return sorted(TEMPLATE_DIR.glob("*/template.tex"))


def test_every_expected_format_has_a_template() -> None:
    """The set of `templates/paper/<fmt>/template.tex` files must match
    EXPECTED_FORMATS. Catches both accidental deletes and accidental
    adds (a new format must be declared in this test before it ships)."""
    on_disk = {p.parent.name for p in _all_templates()}
    expected = set(EXPECTED_FORMATS)
    missing = expected - on_disk
    surprise = on_disk - expected
    assert not missing, f"templates declared but missing on disk: {sorted(missing)}"
    assert not surprise, (
        f"templates on disk but not in EXPECTED_FORMATS — declare them "
        f"in this test before shipping: {sorted(surprise)}"
    )


@pytest.mark.parametrize("fmt", EXPECTED_FORMATS)
def test_template_loads_longtable_and_array(fmt: str) -> None:
    """Pandoc-emitted markdown tables compile to ``\\begin{longtable}``;
    every template must load the package."""
    txt = (TEMPLATE_DIR / fmt / "template.tex").read_text(encoding="utf-8")
    assert r"\usepackage{longtable}" in txt, (
        f"{fmt}/template.tex must load longtable so markdown tables compile"
    )
    assert r"\usepackage{array}" in txt, (
        f"{fmt}/template.tex must load array (paired with longtable for "
        f"cell formatting)"
    )


@pytest.mark.parametrize("fmt", EXPECTED_FORMATS)
def test_template_declares_none_counter(fmt: str) -> None:
    """Pandoc prefixes caption-less tables with ``\\def\\LTcaptype{none}``.
    Without ``\\newcounter{none}`` the compile dies on
    ``No counter 'none' defined`` at the first table."""
    txt = (TEMPLATE_DIR / fmt / "template.tex").read_text(encoding="utf-8")
    assert r"\newcounter{none}" in txt, (
        f"{fmt}/template.tex must \\newcounter{{none}} so longtable can "
        f"resolve pandoc's \\def\\LTcaptype{{none}}"
    )


@pytest.mark.parametrize("fmt", EXPECTED_FORMATS)
def test_template_does_not_pair_authblk_and_titling(fmt: str) -> None:
    """``authblk`` + ``titling`` both patch ``\\author``/``\\maketitle``;
    when titling loads second, ``\\author[N]{...}`` from authblk trips
    LaTeX's ``Missing \\begin{document}``. Use one or the other."""
    txt = (TEMPLATE_DIR / fmt / "template.tex").read_text(encoding="utf-8")
    loads_authblk = r"\usepackage{authblk}" in txt
    loads_titling = r"\usepackage{titling}" in txt
    assert not (loads_authblk and loads_titling), (
        f"{fmt}/template.tex pairs authblk + titling — known LaTeX "
        f"incompatibility, drop one. titling alone is sufficient for "
        f"\\preauthor/\\postauthor hooks; render affiliations as a "
        f"second line of \\author{{}}."
    )
