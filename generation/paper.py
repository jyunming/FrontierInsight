"""Paper output generator.

For each quest, the engine has already written `paper/paper.md` plus any
`figures/` produced by the experiment. The generator (1) hoists the
markdown to the top of the quest output, (2) optionally compiles it to
PDF via pandoc + LaTeX using a format-specific template, and (3) copies
figures and bundle manifest if they exist.

If pandoc or a LaTeX engine is unavailable, PDF generation is skipped
with a warning — the markdown is still produced. The generator also
writes a ``paper/paper_pdf_skipped.md`` file when a PDF was requested
(via ``output.kinds``) but couldn't be produced, so the user discovers
the failure without grepping run logs. Without this, a host missing
pandoc/pdflatex/tectonic silently produces no PDF.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from core.config import Config
from core.engine import QuestArtifacts
from generation._pdf_engine import find_pdf_engine as _find_pdf_engine_impl


# Unicode math + typographic glyphs LLMs routinely emit that
# ``pdflatex`` + ``\usepackage[utf8]{inputenc}`` can't handle on its
# own. Maps each char to a LaTeX replacement that works in BOTH text
# and math contexts (``\ensuremath`` flips into math mode if
# necessary). This map is the ONLY protection for these glyphs — the
# paper templates load ``inputenc`` but do not currently define
# ``\DeclareUnicodeCharacter`` mappings, so any glyph that escapes
# this dict will reach pdflatex unchanged and fail compile. Extend
# the map (don't carve out skip regions) when new glyphs show up in
# user-reported failures.
_LATEX_UNICODE_REPLACEMENTS: dict[str, str] = {
    # Math operators
    "×": r"\ensuremath{\times}",        # ×
    "÷": r"\ensuremath{\div}",          # ÷
    "±": r"\ensuremath{\pm}",           # ±
    "−": r"\ensuremath{-}",             # − (minus sign, distinct from hyphen)
    "≈": r"\ensuremath{\approx}",       # ≈
    "≠": r"\ensuremath{\neq}",          # ≠
    "≤": r"\ensuremath{\leq}",          # ≤
    "≥": r"\ensuremath{\geq}",          # ≥
    "∝": r"\ensuremath{\propto}",       # ∝
    "∞": r"\ensuremath{\infty}",        # ∞
    "·": r"\ensuremath{\cdot}",         # ·
    "⋅": r"\ensuremath{\cdot}",         # ⋅
    "∑": r"\ensuremath{\sum}",          # ∑
    "∫": r"\ensuremath{\int}",          # ∫
    "∂": r"\ensuremath{\partial}",      # ∂
    "∇": r"\ensuremath{\nabla}",        # ∇
    "√": r"\ensuremath{\sqrt{}}",       # √ (bare; users wanting body need explicit LaTeX)
    # Arrows
    "→": r"\ensuremath{\rightarrow}",   # →
    "←": r"\ensuremath{\leftarrow}",    # ←
    "⇒": r"\ensuremath{\Rightarrow}",   # ⇒
    "⇔": r"\ensuremath{\Leftrightarrow}",   # ⇔
    # Superscripts (common in formulas like 10^4)
    "²": r"\ensuremath{^{2}}",          # ²
    "³": r"\ensuremath{^{3}}",          # ³
    "¹": r"\ensuremath{^{1}}",          # ¹
    "⁰": r"\ensuremath{^{0}}",          # ⁰
    "⁴": r"\ensuremath{^{4}}",          # ⁴
    "⁵": r"\ensuremath{^{5}}",          # ⁵
    "⁶": r"\ensuremath{^{6}}",          # ⁶
    "⁷": r"\ensuremath{^{7}}",          # ⁷
    "⁸": r"\ensuremath{^{8}}",          # ⁸
    "⁹": r"\ensuremath{^{9}}",          # ⁹
    # Degrees + typography
    "°": r"\ensuremath{^{\circ}}",      # °
    "—": "---",                          # — em dash
    "–": "--",                           # – en dash
    "…": r"\ldots{}",                   # …
    "«": "``",                           # «
    "»": "''",                           # »
    "“": "``",                           # "
    "”": "''",                           # "
    "‘": "`",                            # '
    "’": "'",                            # '
    # Common Greek letters (lowercase) — full set; LLMs use these
    # bare in prose like "α = 0.05".
    "α": r"\ensuremath{\alpha}",
    "β": r"\ensuremath{\beta}",
    "γ": r"\ensuremath{\gamma}",
    "δ": r"\ensuremath{\delta}",
    "ε": r"\ensuremath{\epsilon}",
    "ζ": r"\ensuremath{\zeta}",
    "η": r"\ensuremath{\eta}",
    "θ": r"\ensuremath{\theta}",
    "ι": r"\ensuremath{\iota}",
    "κ": r"\ensuremath{\kappa}",
    "λ": r"\ensuremath{\lambda}",
    "μ": r"\ensuremath{\mu}",
    "ν": r"\ensuremath{\nu}",
    "ξ": r"\ensuremath{\xi}",
    "π": r"\ensuremath{\pi}",
    "ρ": r"\ensuremath{\rho}",
    "σ": r"\ensuremath{\sigma}",
    "τ": r"\ensuremath{\tau}",
    "φ": r"\ensuremath{\varphi}",
    "χ": r"\ensuremath{\chi}",
    "ψ": r"\ensuremath{\psi}",
    "ω": r"\ensuremath{\omega}",
    # Common uppercase Greek (only the ones not identical to Latin)
    "Γ": r"\ensuremath{\Gamma}",
    "Δ": r"\ensuremath{\Delta}",
    "Θ": r"\ensuremath{\Theta}",
    "Λ": r"\ensuremath{\Lambda}",
    "Ξ": r"\ensuremath{\Xi}",
    "Π": r"\ensuremath{\Pi}",
    "Σ": r"\ensuremath{\Sigma}",
    "Φ": r"\ensuremath{\Phi}",
    "Ψ": r"\ensuremath{\Psi}",
    "Ω": r"\ensuremath{\Omega}",
}

# Pre-compile a translator for str.translate — that path is faster
# than a chained str.replace and Python's translate handles multi-
# char destination strings transparently.
_LATEX_UNICODE_TRANSLATOR = str.maketrans(_LATEX_UNICODE_REPLACEMENTS)


def _sanitize_unicode_for_latex(markdown: str) -> str:
    """Replace common LLM-emitted Unicode math/typography glyphs with
    LaTeX-safe equivalents so ``pdflatex`` + ``\\usepackage[utf8]{inputenc}``
    doesn't error with ``Unicode character … not set up for use with
    LaTeX``.

    Skipped glyphs (anything outside :data:`_LATEX_UNICODE_REPLACEMENTS`)
    are left as-is. If the template's ``\\DeclareUnicodeCharacter``
    safety net covers them, fine; otherwise pdflatex will still error
    — at which point the right fix is to extend the map. Logged at
    INFO when the sanitizer actually replaced anything so run.log
    surfaces what was rewritten."""
    replaced = markdown.translate(_LATEX_UNICODE_TRANSLATOR)
    return replaced


# Pattern: any code-fence block (used by tests + downstream tooling
# that wants prose-only sanitization for non-PDF flows). The PDF
# pipeline uses the unconditional sanitizer below — preserving raw
# glyphs in code blocks would just push the pdflatex failure
# downstream because the templates don't have a
# ``\DeclareUnicodeCharacter`` safety net to catch them.
_CODE_FENCE_RE = re.compile(r"(```.*?\n.*?```|`[^`\n]+`)", re.DOTALL)


def _sanitize_unicode_outside_code_blocks(markdown: str) -> str:
    """Apply :func:`_sanitize_unicode_for_latex` to prose but leave
    fenced code + inline code untouched. NOTE: this is NOT what the
    PDF pipeline uses — that path sanitizes everything to avoid the
    "raw glyph survives in a code block and pdflatex errors" failure
    mode. This helper is kept as a documented option for any future
    consumer that legitimately needs prose-only rewriting; today no
    caller in this file uses it (tests still pin its behaviour as
    contract)."""
    parts = _CODE_FENCE_RE.split(markdown)
    # Even-indexed parts are prose; odd-indexed are code matches.
    for i in range(0, len(parts), 2):
        parts[i] = _sanitize_unicode_for_latex(parts[i])
    return "".join(parts)


# Pandoc's markdown reader requires inline math to have NO whitespace
# immediately after the opening ``$`` or before the closing ``$``.
# When the LLM emits ``$ \beta $`` (with internal spaces) pandoc treats
# the dollars as literal text (escaping them to ``\$``), and ``\beta``
# then runs OUTSIDE math mode, producing "Undefined control sequence".
#
# The regex REQUIRES the content to start with a LaTeX backslash
# command (``\beta``, ``\dfrac``). This anchor:
#   1. Excludes currency prose like ``$ 100 to $ 200`` — no backslash.
#   2. Prevents cross-matching across two separate math spans like
#      ``$\alpha$ and $\beta$`` — the prose between the two pairs
#      doesn't start with ``\``, so the regex can't bridge them.
# Spaces / tabs (not newlines) are allowed on either side; one of the
# two must be non-empty for the rewrite to be observable.
_INLINE_MATH_TIGHTEN_RE = re.compile(
    r"\$([ \t]*\\[a-zA-Z][^$\n]*?[ \t]*)\$",
)


def _tighten_inline_math(markdown: str) -> str:
    """Normalize ``$  \\command...  $`` to ``$\\command...$`` so pandoc
    parses it as inline math. Only rewrites when the content starts
    with a LaTeX command — currency-like ``$ 100`` and prose-bracketed
    dollars are left untouched.

    Skips fenced code blocks (``` ... ```) and inline code spans
    (`` ` ... ` ``) so a paper that documents LaTeX-via-markdown can
    show a literal ``$ \\beta $`` example without it being silently
    rewritten in the rendered PDF.
    """
    def _strip(m: "re.Match[str]") -> str:
        return "$" + m.group(1).strip() + "$"

    out: list[str] = []
    for segment, is_code in _split_code_segments(markdown):
        if is_code:
            out.append(segment)
        else:
            out.append(_INLINE_MATH_TIGHTEN_RE.sub(_strip, segment))
    return "".join(out)


# Splits markdown into alternating (text, code) segments so a transform
# that should only touch prose can iterate without re-implementing
# fence + backtick tracking. Yields tuples of (segment, is_code).
_FENCED_OR_INLINE_CODE_RE = re.compile(
    # ``` fenced block ``` with optional language tag, OR ` inline ` /
    # `` inline with `backtick` `` (one or two backticks). Non-greedy.
    r"(?P<fence>^```[^\n]*\n.*?^```\s*$)"
    r"|(?P<inline>``[^`\n]*?``|`[^`\n]+?`)",
    re.DOTALL | re.MULTILINE,
)


def _split_code_segments(text: str):
    """Yield (segment, is_code) pairs that, concatenated, reproduce
    ``text`` exactly. ``is_code`` is True for fenced blocks + inline
    spans, False for everything else (prose, headings, math, etc.).
    """
    pos = 0
    for m in _FENCED_OR_INLINE_CODE_RE.finditer(text):
        if m.start() > pos:
            yield text[pos:m.start()], False
        yield m.group(0), True
        pos = m.end()
    if pos < len(text):
        yield text[pos:], False


def _count_sanitized_glyphs(markdown: str) -> int:
    """Count how many source-glyph occurrences the sanitizer rewrote.
    Used for the INFO log so the count is honest. Computing from
    output-length deltas was misleading because the LaTeX
    replacements are typically ~20 chars longer than the source
    glyph — a single ``≈`` rewrite added ~20 chars to length but
    only 1 to the "rewrites" count, making the delta negative."""
    return sum(markdown.count(c) for c in _LATEX_UNICODE_REPLACEMENTS)

# First top-level ATX heading in the markdown body. Used to lift the
# title out of `# Title` and into a YAML metadata block so pandoc
# populates `\title{}` (and `\maketitle` actually renders it) instead
# of letting the H1 land as a numbered `\section{1 Title}` while
# `\title{}` stays empty — that produces a "Frontier Insight /
# 1 A Lightweight..." double-stacked header in the PDF.
_FIRST_H1_RE = re.compile(r"^# +(.+?)\s*$", re.MULTILINE)

# Existing YAML frontmatter at the top of the file. If the upstream
# already supplied one, we leave it alone (assume it knows what it's
# doing) — overlaying our own block would either duplicate keys or
# silently win in pandoc's metadata merge.
_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)

# Captures the body of the first ``## Abstract`` section so it can
# be lifted into YAML metadata and rendered by the template inside a
# proper ``\begin{abstract}...\end{abstract}`` environment instead of
# landing as a numbered ``\section{Abstract}``. The non-greedy
# ``.*?`` plus the lookahead at the next ``##`` (or end of file)
# bounds the capture so it stops at the next H2, not at the end of
# the doc — important because the writer always emits Introduction
# / Methods / Results as further H2 sections.
_ABSTRACT_RE = re.compile(
    r"^##\s+Abstract\s*\n(.*?)(?=^##\s|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _extract_abstract_and_strip(markdown: str) -> tuple[str | None, str]:
    """Find the first ``## Abstract`` block, return ``(abstract, body)``.

    Returns ``(None, markdown)`` when no Abstract heading exists.
    The body returned has the matched heading + body removed so the
    template doesn't render the abstract twice (once via metadata,
    once via the leftover Markdown). Leading blank lines are
    collapsed so the next section doesn't open with a stray gap.
    """
    m = _ABSTRACT_RE.search(markdown)
    if not m:
        return None, markdown
    abstract = m.group(1).strip()
    if not abstract:
        return None, markdown
    start, end = m.span()
    body = markdown[:start] + markdown[end:]
    return abstract, body.lstrip("\n")


# Detects a reference-list line where the title is duplicated:
#   "1. Some Title Of A Paper. Some Title Of A Paper."
# The upstream LLM produces this when the prior-work block it sees
# starts the abstract excerpt with the paper title — it copies the
# "title-then-title" pattern back into the references. Anchored on a
# numbered-list prefix + title-cased sentence to avoid eating
# legitimate doubled phrases in prose. The {15,} floor + Title-case
# start keeps this from misfiring on short repeated tokens like
# "Yes. Yes." in body text.
_DUP_REF_RE = re.compile(
    r"^(\s*\d+\.\s+)([A-Z][^.\n]{15,}\.)\s+\2(?=\s|$)",
    re.MULTILINE,
)


def _extract_title_and_strip(markdown: str) -> tuple[str | None, str]:
    """Find the first ``# H1`` line, return ``(title, body_without_h1)``.

    Returns ``(None, markdown)`` unchanged when:
      - the markdown already begins with a YAML frontmatter block
        (upstream is in charge of metadata; we don't second-guess), or
      - no top-level H1 is found.

    The body returned has the matched H1 line + its trailing newline
    removed, with a single leading-blank-line cleanup so the body
    doesn't start with awkward whitespace after lifting the title.
    """
    if _FRONTMATTER_RE.match(markdown):
        return None, markdown
    m = _FIRST_H1_RE.search(markdown)
    if not m:
        return None, markdown
    title = m.group(1).strip()
    if not title:
        return None, markdown
    start, end = m.span()
    # Consume the line's trailing newline if present so we don't leave
    # a stray blank where the title was.
    if end < len(markdown) and markdown[end] == "\n":
        end += 1
    body = markdown[:start] + markdown[end:]
    return title, body.lstrip("\n")


def _add_metadata_frontmatter(
    body: str,
    title: str,
    abstract: str | None = None,
) -> str:
    """Prepend a YAML metadata block carrying the title (and optional
    abstract).

    Title is JSON-quoted — handles colons, quotes, backslashes, and
    Unicode without bespoke escaping. Abstract uses a YAML literal
    block scalar (``|``) so multi-line content + inline LaTeX math
    pass through to pandoc verbatim; pandoc then parses the indented
    block as markdown so emphasis / math / inline code still render
    inside ``\\begin{abstract}``."""
    lines = [f"title: {json.dumps(title)}"]
    if abstract:
        # YAML literal block scalar: every body line indented by 2
        # spaces. Trailing newlines inside the scalar are preserved,
        # which is fine here — pandoc strips them when rendering
        # ``$abstract$``.
        indented = "\n".join("  " + line for line in abstract.splitlines())
        lines.append(f"abstract: |\n{indented}")
    return "---\n" + "\n".join(lines) + "\n---\n\n" + body


def _add_title_frontmatter(body: str, title: str) -> str:
    """Backwards-compatible alias kept for older tests that imported
    the single-arg helper. New code should call
    :func:`_add_metadata_frontmatter` directly."""
    return _add_metadata_frontmatter(body, title, abstract=None)


def _dedupe_duplicated_references(markdown: str) -> str:
    """Collapse ``"1. Foo. Foo."`` reference lines to ``"1. Foo."``.

    The upstream LLM produces this duplication when the prior-work
    block fed to the writer starts each excerpt with the paper title
    — it carries the pattern into the References section. Fixing it
    in markdown post-processing is cheaper than retraining the prompt
    and doesn't risk breaking legitimate citations. Anchored on a
    numbered-list prefix to keep matches scoped to reference lines."""
    return _DUP_REF_RE.sub(r"\1\2", markdown)


_log = logging.getLogger("frontier_insight.paper")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates" / "paper"
# Frontier Insight brand mark — copied next to the pandoc source at
# compile time so the FI house-style templates can \includegraphics it
# as a subtle bottom-left footer mark (matches the slide/poster
# branding). This is the palette-native TEAL monochrome glyph (no dark
# chip), so it sits quietly on a white page next to the teal wordmark.
# Only the FI house templates (generic, whitepaper, report,
# policy_brief, essay) consume it; the venue-mimicking templates
# (iclr, neurips, ieee_access, nature_mi) stay unbranded so they match
# their real-venue formats.
ICON_PNG = REPO_ROOT / "vscode-frontier-insight" / "images" / "fi_glyph_teal.png"


def _copy_brand_icon(out_dir: Path) -> bool:
    """Copy the FI icon into ``out_dir`` as ``fi_icon.png`` so a template
    can reference it relative to the pandoc working directory. Returns
    ``True`` when the icon is in place, ``False`` (no brand mark) if the
    asset is missing or can't be copied — a paper still compiles either
    way because the templates gate the mark behind ``$if(brandfoot)$``."""
    if not ICON_PNG.is_file():
        return False
    try:
        shutil.copyfile(ICON_PNG, out_dir / "fi_icon.png")
        return True
    except OSError:
        return False


@dataclass
class _PdfSkipReason:
    """Why ``_compile_pdf`` returned ``None``. Surfaced to the user via
    ``paper_pdf_skipped.md`` so they don't have to grep run.log."""

    code: str          # short stable identifier, e.g. "no_pandoc"
    summary: str       # one-line WARNING text already logged
    how_to_fix: str    # 1-3 sentences pointing at the concrete action


class PaperGenerator:
    def __init__(self, config: Config) -> None:
        self.config = config

    def generate(self, art: QuestArtifacts, out_dir: Path) -> dict[str, Path]:
        kinds = set(self.config.output.kinds)
        result: dict[str, Path] = {}
        out_dir.mkdir(parents=True, exist_ok=True)

        if "paper_md" in kinds and art.paper_md is not None:
            dst = out_dir / "paper.md"
            shutil.copy2(art.paper_md, dst)
            result["paper_md"] = dst

        if art.figures_dir is not None and art.figures_dir.exists():
            dst = out_dir / "figures"
            # resolve() so symlinks / alternate spellings of the same directory
            # don't trigger rmtree-then-copytree on the source.
            same = dst.resolve() == art.figures_dir.resolve()
            if not same:
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(art.figures_dir, dst)
            result["figures_dir"] = dst

        if art.bundle_manifest is not None and art.bundle_manifest.exists():
            dst = out_dir / "paper_bundle_manifest.json"
            if dst.resolve() != art.bundle_manifest.resolve():
                shutil.copy2(art.bundle_manifest, dst)
            result["bundle_manifest"] = dst

        if "paper_pdf" in kinds and art.paper_md is not None:
            # If paper_md wasn't requested, the markdown wasn't copied above.
            # Compile from art.paper_md directly so the PDF doesn't depend on
            # paper_md being in output.kinds.
            pdf_src = result.get("paper_md") or art.paper_md
            pdf, skip_reason = self._compile_pdf(pdf_src, out_dir)
            diag_path = out_dir / "paper_pdf_skipped.md"
            sanitized_src = out_dir / "paper_pdf_source.md"
            if pdf is not None:
                result["paper_pdf"] = pdf
                # Success on this run — remove any stale skip
                # diagnostic from a PREVIOUS failed run. Otherwise the
                # quest dir would report both paper.pdf AND
                # paper_pdf_skipped.md, which is confusing.
                if diag_path.is_file():
                    try:
                        diag_path.unlink()
                    except OSError:
                        pass
            elif skip_reason is not None:
                # PDF compile skipped — KEEP ``paper_pdf_source.md`` on
                # disk. ``_compile_pdf`` writes that file FIRST and then
                # invokes pandoc on it, so on failure it's literally
                # "exactly what pandoc consumed and choked on" — the
                # single most useful artifact for diagnosing the
                # compile error. (A previous version deleted it here,
                # citing staleness vs prior successful runs, but a
                # failed-this-run file is NEVER stale: it was written
                # 2 lines earlier in the same call.)
                # User asked for a PDF, none was produced, AND we know why.
                # Write a single-paragraph diagnostic file in the same
                # directory so the failure is discoverable without grepping
                # run.log. Stale files from a prior failed run get
                # overwritten — the LATEST attempt's reason is what
                # matters to the user.
                diag_path.write_text(
                    _render_pdf_skip_md(skip_reason, self.config),
                    encoding="utf-8",
                )
                result["paper_pdf_skipped"] = diag_path
                # Strict mode: ``output.require_pdf=True`` means "don't
                # silently skip the PDF on this quest." The pre-flight
                # check in ``core/engine.py`` catches missing
                # prerequisites BEFORE LLM calls, but it can't predict
                # timeouts, nonzero LaTeX exits, or a missing output
                # file after the pandoc subprocess returns rc=0. Those
                # failures land here, post-LLM. Raise so the caller
                # surfaces a hard error instead of completing the
                # quest with a paper_pdf_skipped.md file — which is
                # the same contract the pre-flight already enforces.
                if self.config.output.require_pdf:
                    raise RuntimeError(
                        f"[paper] output.require_pdf=True but paper.pdf "
                        f"could not be produced. reason={skip_reason.code}: "
                        f"{skip_reason.summary} See {diag_path} for the "
                        f"how-to-fix breakdown."
                    )
        elif "paper_pdf" not in kinds:
            # ``paper_pdf`` was not requested — clean up any stale
            # diagnostic AND the sanitized-source copy from a previous
            # run that did request it. Keeps the quest dir tidy after
            # a config edit so a user who turns off paper_pdf doesn't
            # see leftover artifacts that look like they were produced
            # by the current run.
            for stale_name in ("paper_pdf_skipped.md", "paper_pdf_source.md"):
                stale = out_dir / stale_name
                if stale.is_file():
                    try:
                        stale.unlink()
                    except OSError:
                        pass

        return result

    def _find_pdf_engine(self) -> tuple[str, str] | None:
        """Back-compat shim. The real lookup lives in
        ``generation/_pdf_engine.py`` as ``find_pdf_engine`` so the
        poster generator can call the same code path without
        instantiating a ``PaperGenerator``. The method form is kept
        because existing tests (and ``core/engine.py`` pre-flight)
        monkeypatch ``PaperGenerator._find_pdf_engine`` directly.

        Forwards ``paper.REPO_ROOT`` through so existing tests that
        ``monkeypatch.setattr(paper_mod, "REPO_ROOT", tmp_path)`` keep
        steering the repo-local tectonic probe at the temp dir.
        """
        return _find_pdf_engine_impl(REPO_ROOT)

    def _compile_pdf(
        self, paper_md: Path, out_dir: Path,
    ) -> tuple[Path | None, _PdfSkipReason | None]:
        """Run pandoc + a LaTeX engine over ``paper_md`` to produce
        ``out_dir/paper.pdf``. Returns ``(path, None)`` on success,
        ``(None, reason)`` on any kind of skip — the reason carries
        enough info for ``_render_pdf_skip_md`` to write a useful
        ``paper_pdf_skipped.md`` for the user.
        """
        # Wipe any stale ``paper_pdf_source.md`` from a prior run BEFORE
        # we do anything else. The semantic this guarantees: after
        # ``_compile_pdf`` returns, either ``paper_pdf_source.md`` was
        # written by THIS call (so on failure it's the source pandoc
        # consumed and choked on — preserve it for diagnosis) or it
        # doesn't exist at all. A stale prior-run file masquerading as
        # this-run output is the failure mode we're foreclosing.
        stale_src = out_dir / "paper_pdf_source.md"
        if stale_src.is_file():
            try:
                stale_src.unlink()
            except OSError:
                pass

        # Pandoc itself: `subprocess.run` on Windows auto-appends `.exe`
        # to bare executable names via CreateProcess, so resolving via
        # `shutil.which` is mostly defensive (would matter if pandoc
        # ever shipped as a .cmd shim like marp does). We do it for
        # symmetry + so the absolute path lands in any stderr logs.
        pandoc_exe = shutil.which("pandoc")
        if pandoc_exe is None:
            msg = "pandoc not on PATH; paper.pdf skipped (paper.md only)"
            _log.warning(msg)
            return None, _PdfSkipReason(
                code="no_pandoc",
                summary=msg,
                how_to_fix=(
                    "Install pandoc and ensure it's on your PATH. Windows: "
                    "winget install --id JohnMacFarlane.Pandoc, or download "
                    "from https://pandoc.org/installing.html. macOS: "
                    "`brew install pandoc`. Linux: your package manager."
                ),
            )

        # CRITICAL: pandoc does its OWN PATH lookup for the
        # `--pdf-engine` binary. On corporate Windows boxes the
        # MiKTeX bin dir (`~/AppData/Local/Programs/MiKTeX/miktex/bin/x64/`)
        # often isn't on the Python child's PATH even though MiKTeX
        # is installed. Resolving the engine up-front and passing the
        # full path bypasses pandoc's lookup. Falls back to tectonic
        # (no-admin LaTeX) when pdflatex isn't reachable.
        engine = self._find_pdf_engine()
        if engine is None:
            msg = (
                "no LaTeX engine found (pdflatex or tectonic); paper.pdf "
                "skipped. Run `python launch.py --install-tectonic` for "
                "a no-admin LaTeX install."
            )
            _log.warning(msg)
            return None, _PdfSkipReason(
                code="no_latex_engine",
                summary=msg,
                how_to_fix=(
                    "Easiest no-admin path: run `python launch.py "
                    "--install-tectonic` from the repo root. That drops a "
                    "single self-bootstrapping LaTeX binary (~70 MB) into "
                    "`tools/`; FI auto-detects it on the next quest. "
                    "Standard alternative: install MiKTeX (Windows) or "
                    "TeX Live (macOS/Linux) so `pdflatex` lands on PATH."
                ),
            )
        engine_name, engine_path = engine
        _log.info("paper.pdf: using pdf-engine=%s at %s", engine_name, engine_path)

        fmt = self.config.output.paper_format
        template = TEMPLATES_DIR / fmt / "template.tex"
        out_pdf = out_dir / "paper.pdf"

        # Pre-process: rewrite common Unicode math glyphs (≈, ≥, ×, −,
        # —, lowercase Greek, etc.) into LaTeX commands before pandoc
        # sees the file. ``pdflatex`` + ``inputenc utf8`` can't handle
        # them on its own ("Unicode character ≈ (U+2248) not set up for
        # use with LaTeX"). Sanitize UNCONDITIONALLY (including inside
        # fenced + inline code) because the paper templates don't have
        # a ``\DeclareUnicodeCharacter`` safety net — a raw glyph
        # surviving in a code block would still kill the compile, just
        # one stack-frame later. The user's original paper.md is
        # preserved unchanged; the sanitized copy lands at
        # ``out_dir/paper_pdf_source.md`` so it's visible if the user
        # wants to see exactly what pandoc consumed.
        # Default to "no title lifted" so the OSError fallback below
        # (which skips the title-extraction code path entirely) still
        # has a defined name when the cmd builder consults it for the
        # shift-heading-level decision.
        title: str | None = None
        try:
            md_text = paper_md.read_text(encoding="utf-8")
            glyph_count = _count_sanitized_glyphs(md_text)
            sanitized_md = _sanitize_unicode_for_latex(md_text)
            # Tighten ``$ \beta $`` → ``$\beta$`` so pandoc parses it
            # as inline math instead of treating the dollars as
            # literal text (the latter produced "Missing $" /
            # "Undefined control sequence" errors on quests whose LLM
            # emitted math with whitespace inside the delimiters).
            sanitized_md = _tighten_inline_math(sanitized_md)
            if glyph_count:
                _log.info(
                    "paper.pdf: rewrote %d Unicode glyph occurrence(s) "
                    "in paper.md before pandoc",
                    glyph_count,
                )
            # Lift the first `# H1` to YAML frontmatter so `\title{}`
            # in the template gets populated. Without this, pandoc
            # leaves `\title{}` empty and re-emits the H1 as a
            # numbered `\section{}` — producing a
            # "Frontier Insight" + "1 A Lightweight ..." stacked
            # header in the PDF.
            title, sanitized_md = _extract_title_and_strip(sanitized_md)
            # Lift `## Abstract` body to YAML metadata so the template
            # can wrap it in `\begin{abstract}…\end{abstract}` instead
            # of letting it land as `\section{Abstract}` (numbered "1
            # Abstract", which looks unprofessional next to real venue
            # rendering). Only meaningful when we also have a title —
            # without a title the article-class \maketitle has nothing
            # to attach the abstract to.
            abstract: str | None = None
            if title is not None:
                abstract, sanitized_md = _extract_abstract_and_strip(sanitized_md)
                sanitized_md = _add_metadata_frontmatter(
                    sanitized_md, title, abstract=abstract,
                )
                _log.info(
                    "paper.pdf: lifted H1 title%s into frontmatter",
                    " + abstract" if abstract else "",
                )
            # Collapse "Foo. Foo." reference-line duplications the
            # LLM emits when the prior-work excerpt starts with the
            # paper title.
            sanitized_md = _dedupe_duplicated_references(sanitized_md)
            sanitized_path = out_dir / "paper_pdf_source.md"
            sanitized_path.write_text(sanitized_md, encoding="utf-8")
            pandoc_input = sanitized_path
        except OSError as e:
            _log.warning(
                "paper.pdf: could not pre-process paper.md (%r); "
                "falling back to original — pdflatex may fail on "
                "Unicode glyphs.", e,
            )
            pandoc_input = paper_md

        cmd: list[str] = [
            pandoc_exe,
            str(pandoc_input),
            "-o", str(out_pdf),
            f"--pdf-engine={engine_path}",
            "--standalone",
            # Tolerate the LLM's habit of dropping a bullet/numbered
            # list immediately after the introducing paragraph with no
            # blank line in between. Pandoc's default markdown reader
            # requires the blank line and otherwise jams the entire
            # list into the paragraph as inline text ("Foo - a - b -
            # c.") — the bullets-rendered-as-prose failure mode.
            #
            # +autolink_bare_uris: wrap bare reference URLs in \url{} so
            # the template's xurl can break them at ANY character. Without
            # it, pandoc renders a bare URL as plain text with escaped
            # underscores (\_) that never break — a wiki-style URL then
            # runs straight into the right margin (the reference-URL
            # overflow the FI house templates' xurl + \urlstyle{same}
            # fix once the URL is a real \url{}).
            "--from=markdown+lists_without_preceding_blankline+autolink_bare_uris",
            # The first H1 is lifted into the YAML title above, so the
            # remaining H2/H3/... headings should shift up one level —
            # otherwise an article-class document numbers them as
            # "0.1 Abstract / 0.2 Introduction" (subsections of an
            # implicit section 0) instead of "1 Abstract / 2
            # Introduction". Only applied when we actually lifted a
            # title; without the shift, a paper whose author intended
            # H1=section, H2=subsection would get mangled.
        ]
        if title is not None:
            cmd.append("--shift-heading-level-by=-1")
        # Copy the FI icon next to the source and flip the template's
        # ``brandfoot`` switch so the house-style templates render a
        # subtle footer corner mark + cover icon. Gated on a successful
        # copy: if the asset is missing the variable stays unset and the
        # template's ``$if(brandfoot)$`` blocks no-op, so the paper still
        # compiles unbranded rather than failing on a missing image.
        if _copy_brand_icon(out_dir):
            cmd.extend(["-V", "brandfoot=true"])
        if template.exists():
            cmd.extend(["--template", str(template)])
        else:
            _log.info(
                "no template at %s; using pandoc default (paper_format=%s)",
                template, fmt,
            )

        # Headroom for the first-run package-fetch warm-up. Tectonic's
        # first invocation downloads CTAN packages into
        # `%LOCALAPPDATA%/TectonicProject/Tectonic/` (~30 s); MiKTeX's
        # first compile after a fresh install does the same. 300 s
        # comfortably covers both; bumped to 360 s when tectonic is
        # the picked engine in case the user is on a slow corporate
        # network.
        timeout_s = 360 if engine_name == "tectonic" else 300
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(out_dir),
                timeout=timeout_s,
            )
        except FileNotFoundError:
            msg = "pandoc invocation failed; paper.pdf skipped"
            _log.warning(msg)
            return None, _PdfSkipReason(
                code="pandoc_invocation_failed",
                summary=msg,
                how_to_fix=(
                    "`shutil.which('pandoc')` found pandoc but `subprocess.run` "
                    "could not invoke it. Likely a PATH disagreement between "
                    "the shell that resolved the binary and the Python child "
                    "process that tried to spawn it. Reopen the terminal "
                    "after installing pandoc so the Python session inherits "
                    "the updated PATH. (FI does not consult a separate "
                    "`PANDOC_PATH` env var; only `shutil.which('pandoc')` is "
                    "used to locate the binary.)"
                ),
            )
        except subprocess.TimeoutExpired:
            msg = f"{engine_name} timeout (>{timeout_s}s); paper.pdf skipped"
            _log.warning(msg)
            return None, _PdfSkipReason(
                code=f"{engine_name}_timeout",
                summary=msg,
                how_to_fix=(
                    f"The {engine_name} compile took longer than "
                    f"{timeout_s}s. The first-ever compile on a fresh "
                    f"install can download CTAN packages (~30 s); a slow "
                    f"corporate VPN amplifies this. Try again — the second "
                    f"compile is usually fast. If it consistently times "
                    f"out, raise the timeout in `generation/paper.py:_compile_pdf`."
                ),
            )

        if r.returncode != 0:
            stderr_tail = r.stderr[-500:] if r.stderr else "(empty stderr)"
            msg = (
                f"{engine_name} rc={r.returncode} format={fmt}; "
                f"paper.pdf skipped. stderr_tail={stderr_tail}"
            )
            _log.warning(msg)
            return None, _PdfSkipReason(
                code=f"{engine_name}_rc_{r.returncode}",
                summary=(
                    f"{engine_name} exited with rc={r.returncode} when "
                    f"compiling paper.md with paper_format={fmt!r}."
                ),
                how_to_fix=(
                    f"The LaTeX engine errored. Common causes: a malformed "
                    f"LaTeX template, an unsupported character in paper.md, "
                    f"or a missing CTAN package on the first-ever compile. "
                    f"stderr tail (last 500 chars):\n\n```\n{stderr_tail}\n```\n\n"
                    f"The exact markdown pandoc consumed is preserved at "
                    f"`paper_pdf_source.md` (post-Unicode-sanitization, "
                    f"post-frontmatter-lift). The ``l.N`` line numbers in "
                    f"the stderr above index into pandoc's intermediate "
                    f".tex (which pandoc deletes on exit) — not into "
                    f"`paper_pdf_source.md` directly. To recover the .tex "
                    f"for inspection, run:\n\n"
                    f"```\n"
                    f"pandoc paper_pdf_source.md -o paper.tex --standalone "
                    f"--template templates/paper/{fmt}/template.tex\n"
                    f"```\n\n"
                    f"Then look at the quoted ``l.N`` of `paper.tex`. Once "
                    f"the bad construct is identified, patch either "
                    f"`templates/paper/{fmt}/template.tex` (template bug) "
                    f"or `paper.md` (LLM emitted something pandoc translated "
                    f"into broken LaTeX)."
                ),
            )
        if not out_pdf.exists():
            msg = (
                f"{engine_name} rc=0 but {out_pdf.name} not on disk; "
                f"paper.pdf skipped."
            )
            _log.warning(msg)
            return None, _PdfSkipReason(
                code="output_missing_after_success",
                summary=msg,
                how_to_fix=(
                    f"{engine_name} reported success but didn't write the "
                    f"output file. This is rare — usually a permissions "
                    f"issue on the output directory. Check `{out_dir}` is "
                    f"writable and re-run."
                ),
            )
        return out_pdf, None


def _render_pdf_skip_md(reason: _PdfSkipReason, config: Config) -> str:
    """Render a ``paper_pdf_skipped.md`` diagnostic file. The user sees
    this file next to ``paper.md`` when the PDF couldn't be produced;
    it tells them what failed and how to fix it without having to grep
    ``run.log`` or read the engine source."""
    return (
        f"# paper.pdf was requested but not produced\n\n"
        f"Your `output.kinds` included `paper_pdf` "
        f"(`paper_format: {config.output.paper_format}`), "
        f"but the paper generator couldn't compile a PDF.\n\n"
        f"## What happened\n\n"
        f"**Reason code:** `{reason.code}`\n\n"
        f"{reason.summary}\n\n"
        f"## How to fix it\n\n"
        f"{reason.how_to_fix}\n\n"
        f"## After fixing\n\n"
        f"To compile just the PDF for this quest (without re-running the "
        f"whole LLM loop), run the following from the repo root. It "
        f"mirrors how `launch.py` invokes the generator on a successful "
        f"run (see `launch.py:_run_generators`).\n\n"
        f"```python\n"
        f"from pathlib import Path\n"
        f"from core.config import Config\n"
        f"from core.engine import QuestArtifacts\n"
        f"from generation.paper import PaperGenerator\n"
        f"\n"
        f"# Replace with the actual quest dir path:\n"
        f"quest_root = Path('outputs/<your_quest_id>').resolve()\n"
        f"cfg = Config.from_yaml(quest_root / 'config.yaml')\n"
        f"art = QuestArtifacts(\n"
        f"    quest_id=quest_root.name,\n"
        f"    quest_root=quest_root,\n"
        f"    paper_md=quest_root / 'paper' / 'paper.md',\n"
        f"    figures_dir=quest_root / 'figures',\n"
        f"    bundle_manifest=None,\n"
        f")\n"
        f"# `art.quest_root` is the same target launch.py passes — the\n"
        f"# generator drops `paper/paper.pdf` (and any other output\n"
        f"# kinds you've configured) under this directory.\n"
        f"PaperGenerator(cfg).generate(art, art.quest_root)\n"
        f"```\n\n"
        f"This file is auto-deleted on the next successful PDF compile.\n"
    )
