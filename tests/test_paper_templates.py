r"""Pin the LaTeX package invariants in templates/paper/*/template.tex.

These templates are consumed by pandoc; every one of them must:

1. Load ``longtable`` + ``array`` — pandoc renders markdown tables as
   ``\begin{longtable}[]{...}``, and without the package the compile
   dies on ``Environment longtable undefined``.

2. Load ``booktabs`` — pandoc wraps every pipe table with ``\toprule``
   / ``\midrule`` / ``\bottomrule``; without booktabs the compile dies
   on ``Undefined control sequence. \toprule`` at the first table.

3. Declare ``\newcounter{none}`` — pandoc prefixes caption-less
   tables with ``\def\LTcaptype{none}``; ``longtable`` then calls
   ``\refstepcounter{\LTcaptype}`` internally, so the counter must
   exist or the compile dies on ``No counter 'none' defined``.

4. NOT load ``authblk`` together with ``titling`` — when both patch
   ``\author`` and ``\maketitle``, LaTeX hits ``\author[N]{...}``
   (authblk's optional-affiliation form) and reports ``Missing
   \begin{document}`` because titling's redefinition shadows it.

5. In a ``twocolumn`` template, redefine ``longtable`` — it stops with
   ``longtable not in 1-column mode`` in two-column mode, at the first
   markdown table.

A real EUV-stochastics quest hit several of these failure modes in
succession. This module is the regression guard so a future template
edit (or a new template) doesn't silently re-introduce any of them.
"""
from __future__ import annotations

import re
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
def test_template_loads_booktabs(fmt: str) -> None:
    """Pandoc renders every markdown pipe table with ``\\toprule`` /
    ``\\midrule`` / ``\\bottomrule`` from booktabs. With ``longtable``
    loaded but ``booktabs`` absent, the compile dies on
    ``Undefined control sequence. \\toprule`` at the first table."""
    txt = (TEMPLATE_DIR / fmt / "template.tex").read_text(encoding="utf-8")
    assert r"\usepackage{booktabs}" in txt, (
        f"{fmt}/template.tex must load booktabs so pandoc-rendered "
        f"markdown tables (\\toprule/\\midrule/\\bottomrule) compile"
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
def test_template_defines_the_float_barrier(fmt: str) -> None:
    """``generation/paper.py`` writes ``\\FIfloatbarrier`` into the PDF
    source just above the reference list, so a figure LaTeX is still
    holding back cannot be carried past that list and printed among the
    references. Every template must DEFINE the command — an undefined one
    stops the compile with ``Undefined control sequence``, which would
    turn a layout fix into a paper that does not render at all.

    ``placeins``' ``\\FloatBarrier`` is deliberately NOT used: the package
    is absent from TeX installs FI otherwise supports (a real MiKTeX here
    has no ``placeins.sty``), so the templates test LaTeX's own
    ``\\@deferlist`` instead."""
    txt = (TEMPLATE_DIR / fmt / "template.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\FIfloatbarrier}" in txt, (
        f"{fmt}/template.tex must define \\FIfloatbarrier — "
        f"generation/paper.py emits it above the reference list"
    )
    assert r"\@deferlist" in txt, (
        f"{fmt}/template.tex must test \\@deferlist so the barrier costs "
        f"nothing when no figure is waiting"
    )
    assert r"\usepackage{placeins}" not in txt, (
        f"{fmt}/template.tex must not depend on placeins — it is not "
        f"present in every supported TeX install"
    )


def test_two_column_templates_redefine_longtable() -> None:
    """longtable stops with ``longtable not in 1-column mode`` in a
    ``twocolumn`` document, so every markdown table killed the ieee_access
    compile. A two-column template must set pandoc's longtable in a box."""
    two_column = [
        fmt for fmt in EXPECTED_FORMATS
        if re.search(
            r"\\documentclass\[[^\]]*twocolumn",
            (TEMPLATE_DIR / fmt / "template.tex").read_text(encoding="utf-8"),
        )
    ]
    assert "ieee_access" in two_column
    for fmt in two_column:
        txt = (TEMPLATE_DIR / fmt / "template.tex").read_text(encoding="utf-8")
        assert r"\RenewDocumentEnvironment{longtable}{+b}" in txt, (
            f"{fmt}/template.tex is two-column and must redefine longtable"
        )


@pytest.mark.parametrize("fmt", EXPECTED_FORMATS)
def test_template_can_make_its_text_area_taller(fmt: str) -> None:
    """The visual check repairs a last page holding only a line or two by
    recompiling with ``fi-extra-lines``; the hook must sit in the preamble,
    before the real ``\\begin{document}`` (the first one may be in a comment)."""
    txt = (TEMPLATE_DIR / fmt / "template.tex").read_text(encoding="utf-8")
    hook = txt.find("$if(fi-extra-lines)$")
    assert hook >= 0, f"{fmt}/template.tex has no fi-extra-lines hook"
    assert hook < txt.rindex("\n\\begin{document}")
    assert r"\addtolength{\textheight}{$fi-extra-lines$\baselineskip}" in txt
    # The footer moves up by the same amount, so it stays on the page.
    assert r"\addtolength{\footskip}{-$fi-extra-lines$\baselineskip}" in txt


@pytest.mark.slow
def test_a_two_column_paper_with_a_short_and_a_tall_table_compiles_whole(tmp_path: Path) -> None:
    """A short table becomes a column float; one taller than a column gets
    single-column pages. Nothing runs off the page and the numbering is
    not stepped twice by the measuring pass."""
    from core.config import Config
    from generation._pandoc import find_pandoc
    from generation._pdf_engine import find_pdf_engine
    from generation._pdf_measure import measure_pdf, paper_report
    from generation.paper import PaperGenerator

    if find_pandoc(REPO_ROOT) is None or find_pdf_engine(REPO_ROOT) is None:
        pytest.skip("needs pandoc and a LaTeX engine")
    rows = "\n".join(f"| Step {i} | {i * 0.5:.1f} | value {i} |" for i in range(70))
    md = tmp_path / "paper.md"
    md.write_text(
        "# Table Probe\n\n## Results\n\nText before the tables.\n\n"
        ": Energy drift per integrator.\n\n| Integrator | Drift |\n|---|---|\n| Leapfrog | 3e-7 |\n\n"
        "Text between the tables.\n\n"
        ": Seventy steps.\n\n| Step | Time | Note |\n|---|---|---|\n" + rows + "\n\nText after the tables.\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    out.mkdir()
    config = Config.model_validate({
        "topic": "t",
        "title": "t",
        "output": {
            "kinds": ["paper_md", "paper_pdf"],
            "output_dir": str(tmp_path / "outputs"),
            "paper_format": "ieee_access",
            "html_pdf_fallback": False,
        },
    })
    pdf, skip = PaperGenerator(config)._compile_pdf(md, out)
    assert skip is None, skip.summary if skip else ""
    doc = measure_pdf(pdf)
    text = " ".join(line.text for page in doc.pages for line in page.lines)
    for expected in ("Leapfrog", "Step 0", "Step 69", "Text after the tables", "Table 1:", "Table 2:"):
        assert expected in text
    assert "Table 3:" not in text
    checks = {finding["check"] for finding in paper_report(doc)["findings"]}
    assert not checks & {"overflow", "overwide"}, checks


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


def test_no_template_narrates_the_engine() -> None:
    """A rendered paper/poster must read as a standalone document, never as
    a report on the tool that produced it. Templates have hardcoded engine
    narration in several spots: the venue \\author byline ("Generated by an
    automated research pipeline"), the neurips fallback abstract ("This
    paper was produced end-to-end by Frontier Insight"), and the poster
    header ("Automated research briefing"). Guard EVERY paper template AND
    the poster template against re-introducing it.

    Note: the muted "Frontier Insight" brand wordmark footer / pdfauthor is
    intentional branding and is NOT banned — only phrasing that narrates
    the work as machine-produced."""
    banned = [
        "automated research pipeline",
        "generated by an automated",
        "machine-generated",
        "generated by frontier insight",
        "produced by frontier insight",
        "produced end-to-end",            # "produced end-to-end by Frontier Insight"
        "automated research briefing",
    ]
    # Paper templates + the poster template (the poster has its own author).
    targets = list(_all_templates()) + [REPO_ROOT / "templates" / "poster" / "poster.tex"]
    for tpl in targets:
        low = tpl.read_text(encoding="utf-8").lower()
        for phrase in banned:
            assert phrase not in low, (
                f"{tpl.parent.name}/{tpl.name} contains engine "
                f"self-reference {phrase!r} — a paper/poster must not name "
                f"the tool that produced it"
            )
