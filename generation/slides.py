"""Slide deck generator.

Runs only when ``config.output.kinds`` contains ``"slides"`` AND the
quest produced a ``paper.md``. When those gates pass, three render
targets are attempted, each best-effort and independently skipped
if its CLI isn't installed:

1. LLM call: compress ``paper.md`` into Marp markdown (8-12 slides) →
   ``slides.md``. Produced whenever the slides kind is enabled — the
   source of truth for the other two targets.
2. ``marp slides.md -o slides.{html,pdf}`` — produced when the Marp
   CLI is on PATH.
3. ``slides.pptx`` — rendered in-process by ``generation/_pptx_slides.py``
   with python-pptx, applying the same ``fi.css`` design language as the
   HTML/PDF deck. Real PowerPoint the user can open and edit in Office /
   Google Slides / Keynote, satisfying the common "I want an actual
   presentation, not a markdown file" use case.

   Unlike (2) this target has NO external dependency, so it always
   renders. It replaced a ``pandoc slides.md -o slides.pptx`` shell-out,
   which needed pandoc on PATH — exactly what a locked-down host lacks —
   and produced an unthemed plain-white deck that looked nothing like its
   HTML sibling.

The Marp YAML frontmatter at the top of slides.md is harmless to
pandoc (it consumes the leading `---\\nmarp: true\\n...\\n---` block as
document metadata) and slide breaks come from the standalone `---`
horizontal rules between slides — same as Marp uses.

When marp can't render (CLI absent, non-zero rc, timeout, or chromium
dependency missing on a fresh install) the generator writes a
``slides_skipped.md`` diagnostic next to ``slides.md`` so the user
discovers the skip without grepping ``run.log``. Mirrors the
``paper_pdf_skipped.md`` UX from #55 and the
``poster_pdf_skipped.md`` UX from #145. The diagnostic covers BOTH
marp render targets (slides.html + slides.pdf) with a single file
because they share the same engine — splitting into two diagnostics
would just duplicate the same install recipe.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import string
from pathlib import Path

from core.config import Config
from core.engine import (
    _FURTHER_READING_HEADING_RE,
    QuestArtifacts,
    build_further_reading,
    build_references,
    render_further_reading_marp_slide,
    render_references_marp_slide,
)
from core.provider import (
    LLMClient,
    ProxySupervisor,
    PROXY_PROVIDERS,
    model_for_node,
    resolve_endpoint_async,
)
from generation._marp import find_marp
from generation._pptx_slides import render_marp_to_pptx
from generation._skip_md import render_skip_md

_log = logging.getLogger("frontier_insight.slides")

PROMPT_PATH = Path(__file__).resolve().parent.parent / "agents" / "slides.md"
# Custom Marp theme (a polished look layered on the default). Passed to the
# Marp CLI with --theme; the deck's `theme: fi` front-matter selects it.
THEME_PATH = Path(__file__).resolve().parent.parent / "templates" / "slides" / "fi.css"

# Marp's stderr signature when its bundled chromium dependency is
# missing on a fresh install (puppeteer downloads chromium on first
# run; on locked-down corporate hosts the download fails and the
# subsequent PDF render fails too). Surfacing this as a distinct
# reason code lets the user discover the fix (run marp once manually
# to trigger the download, or set ``PUPPETEER_EXECUTABLE_PATH``)
# without having to read the marp stderr blob in run.log.
#
# Phrases are checked case-insensitively as substrings. Keep them
# SPECIFIC enough not to match unrelated node stack traces — a bare
# token like "puppeteer" routinely shows up in module paths and
# would route generic render failures to the wrong how-to-fix text.
_CHROMIUM_SIGNATURES: tuple[str, ...] = (
    "could not find chromium",
    "could not find chrome",
    "chromium revision",
    "failed to launch the browser",
    "no usable sandbox",
    "failed to download chromium",
)


class SlideGenerator:
    def __init__(self, config: Config) -> None:
        self.config = config

    async def generate(
        self,
        art: QuestArtifacts,
        out_dir: Path,
        *,
        supervisor: ProxySupervisor | None = None,
        feedback: str = "",
    ) -> dict[str, Path]:
        # ``feedback`` is what a visual check of the previous deck found; it
        # is appended to the authoring prompt.
        # Cleanup gate: if "slides" is no longer in output.kinds (user
        # removed it from their YAML between runs), remove any stale
        # ``slides_skipped.md`` left over from a prior run. Mirrors
        # PaperGenerator's cleanup when paper_pdf is dropped from
        # kinds. Without this, the stale diagnostic persists forever
        # after the user opts out.
        if "slides" not in self.config.output.kinds:
            stale = out_dir / "slides_skipped.md"
            if stale.is_file():
                try:
                    stale.unlink()
                except OSError:
                    pass
            return {}
        if art.paper_md is None:
            return {}

        slides_md = await self._author_marp(art, out_dir, supervisor=supervisor, feedback=feedback)
        result: dict[str, Path] = {"slides_md": slides_md}
        diag_path = out_dir / "slides_skipped.md"
        # Will be set to a (reason_code, summary, how_to_fix) triple
        # the first time a marp invocation fails. We only write ONE
        # diagnostic per quest covering html + pdf together — they
        # share the marp engine, so duplicate diagnostics would just
        # repeat the same install recipe twice.
        marp_skip: tuple[str, str, str] | None = None

        # Marp branch: HTML + PDF rendering. Use shutil.which's resolved
        # path so Windows `.cmd`/`.bat` shims work — `asyncio.create_
        # subprocess_exec` doesn't apply Windows PATHEXT, so spawning
        # bare "marp" fails on systems where marp lives as marp.CMD.
        marp_exe = find_marp()
        # Security gate for --allow-local-files (see the flag's comment in
        # the render call below). That flag lets Marp read ANY local file
        # the deck references; since slides.md is LLM-authored and the repo
        # now loads .env secrets, refuse to run Marp when the deck points at
        # anything other than our own figures/, a remote URL, or a data URI.
        unsafe_refs = _disallowed_local_image_refs(slides_md)
        if marp_exe is None:
            msg = "marp CLI not found; slides.html/.pdf skipped"
            _log.warning(msg)
            marp_skip = (
                "no_marp",
                msg,
                _MARP_INSTALL_RECIPE,
            )
        elif unsafe_refs:
            preview = ", ".join(unsafe_refs[:5])
            msg = (
                f"slides.html/.pdf skipped: the deck references local file(s) "
                f"outside figures/ that Marp's --allow-local-files would embed "
                f"into the rendered output: {preview}"
            )
            _log.warning(msg)
            marp_skip = (
                "unsafe_local_file_refs",
                msg,
                "Marp runs with --allow-local-files so it can load the quest's "
                "figures/*.png. To avoid exposing arbitrary local files "
                "(including .env secrets) in the rendered deck, FI only permits "
                "image paths under figures/, http(s):// URLs, and data: URIs. "
                "Edit slides.md so every image uses one of those forms, then "
                "re-render with `marp slides.md --allow-local-files "
                "-o slides.pdf`.",
            )
        else:
            # Apply the custom FI theme when its CSS is present; fall back to
            # the deck's built-in theme otherwise.
            theme_args = (
                ["--theme", str(THEME_PATH)] if THEME_PATH.is_file() else []
            )
            for ext in ("html", "pdf"):
                out_path = out_dir / f"slides.{ext}"
                ok, fail_reason = await _run_cli(
                    # --allow-local-files: without it Marp silently refuses to
                    # load the quest's figures/*.png (its default file-access
                    # block), leaving every figure slide blank. The inputs are
                    # our own generated figures, so it's safe to allow.
                    [marp_exe, str(slides_md), "--allow-local-files",
                     *theme_args, "-o", str(out_path)],
                    cwd=out_dir, label=f"marp {ext}",
                )
                if ok:
                    result[f"slides_{ext}"] = out_path
                elif marp_skip is None and fail_reason is not None:
                    # Record only the FIRST failure — subsequent marp
                    # invocations almost always fail the same way (same
                    # CLI, same environment) and a single diagnostic
                    # covers both render targets cleanly.
                    marp_skip = fail_reason

        if marp_skip is not None:
            code, summary, how_to_fix = marp_skip
            # ``slides.md`` typically succeeds even when marp fails
            # (it's authored by an LLM, not the marp CLI), so the
            # H1 here is scoped to the actual failure surface
            # (html / pdf) rather than the umbrella "slides" kind —
            # the user opening their quest dir already sees
            # ``slides.md`` and shouldn't be told slides is missing.
            diag_path.write_text(
                render_skip_md(
                    requested_kind="slides",
                    display_name="slides.html / slides.pdf",
                    reason_code=code,
                    summary=summary,
                    how_to_fix=how_to_fix,
                ),
                encoding="utf-8",
            )
            result["slides_skipped"] = diag_path
        elif diag_path.is_file():
            # Success on every marp render — clean up any stale
            # diagnostic from a prior failed run. Mirrors the poster
            # generator pattern: a successful run must not leave a
            # ``*_skipped.md`` file on disk next to the artifact.
            try:
                diag_path.unlink()
            except OSError:
                pass

        # Pandoc branch: real PowerPoint (.pptx). Same .CMD-on-Windows
        # caveat — resolve via shutil.which. Note: pandoc-pptx failures
        # do NOT trigger ``slides_skipped.md`` — that diagnostic is
        # marp-specific (covers slides.html + slides.pdf which share
        # an engine). A pandoc-pptx skip is logged but doesn't fan out
        # to a third diagnostic; the user still gets ``slides.md`` to
        # open in any tool they like.
        # Rendered in-process with python-pptx, applying the same fi.css
        # design language as the HTML/PDF deck. This replaced a
        # `pandoc slides.md -o slides.pptx` shell-out for two reasons:
        # pandoc's built-in pptx template is unthemed (plain white Calibri),
        # so the one artifact a colleague is most likely to open looked
        # nothing like its siblings; and it needed pandoc on PATH, which is
        # exactly what a locked-down host lacks. python-pptx ships with FI,
        # so slides.pptx now always renders -- and stays editable.
        pptx_path = out_dir / "slides.pptx"
        if render_marp_to_pptx(
            slides_md, pptx_path, figures_dir=out_dir / "figures",
        ):
            result["slides_pptx"] = pptx_path

        return result

    async def _author_marp(
        self,
        art: QuestArtifacts,
        out_dir: Path,
        *,
        supervisor: ProxySupervisor | None,
        feedback: str = "",
    ) -> Path:
        paper_md = art.paper_md.read_text(encoding="utf-8") if art.paper_md else ""
        figures = []
        if art.figures_dir and art.figures_dir.is_dir():
            figures = sorted(p.name for p in art.figures_dir.iterdir() if p.is_file())
        prompt = string.Template(PROMPT_PATH.read_text(encoding="utf-8")).substitute(
            paper_md=paper_md[:8000],
            figure_list="\n".join(f"- figures/{f}" for f in figures) or "(none)",
        )
        if feedback:
            prompt = prompt.rstrip() + "\n\n" + feedback.strip() + "\n"
        own_supervisor = supervisor is None
        sup = supervisor or ProxySupervisor()
        endpoint = await resolve_endpoint_async(self.config.provider, sup)
        client = LLMClient(endpoint)
        try:
            text = await client.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                model=model_for_node(self.config.provider.node_models, "slides"),
                node="slides",
            )
        finally:
            await client.aclose()
            if self.config.provider.name in PROXY_PROVIDERS:
                await sup.release(self.config.provider.name)
            if own_supervisor:
                await sup.shutdown()

        content = _figures_as_own_paragraphs(_fences_as_slide_breaks(_strip_outer_fence(text)))
        # Append a References slide built from the quest's actual sources
        # (web pages + papers), guaranteed rather than left to the LLM —
        # the deck author only saw the first 8000 chars of paper.md and
        # would usually miss the References section at the end. Skip if the
        # LLM already produced one (avoid a duplicate).
        literature = art.raw_state.get("literature") or []
        audience = self.config.output.audience
        ref_slide = render_references_marp_slide(
            build_references(literature, audience=audience))
        if ref_slide and "## References" not in content:
            content = content.rstrip() + "\n\n" + ref_slide + "\n"
        # The web pages get their own slide after it, unless the deck has one.
        further_slide = render_further_reading_marp_slide(
            build_further_reading(literature, audience=audience))
        if further_slide and not _FURTHER_READING_HEADING_RE.search(content):
            content = content.rstrip() + "\n\n" + further_slide + "\n"
        content = _with_author_line(content, self.config.output)

        slides_md = out_dir / "slides.md"
        slides_md.write_text(content, encoding="utf-8")
        _log.info("slides.md written (%d bytes)", slides_md.stat().st_size)
        return slides_md


def _with_author_line(content: str, output) -> str:
    """Add the configured author line to the title slide, as a paragraph
    under its headings. The deck is unchanged when no author field is set."""
    parts = (output.author, output.affiliation, output.contact_email, output.url)
    line = " · ".join(" ".join(str(p).split()) for p in parts if str(p or "").strip())
    if not line:
        return content
    match = _FRONT_MATTER_RE.match(content)
    head = content[: match.end()] if match else ""
    body = content[len(head):]
    start = 0
    for brk in _SLIDE_BREAK_RE.finditer(body):
        if body[start:brk.start()].strip():
            return head + body[:brk.start()].rstrip() + "\n\n" + line + "\n\n" + body[brk.start():]
        start = brk.end()
    return head + body.rstrip() + "\n\n" + line + "\n"


# Markdown / Marp image reference: ``![alt](path ...)`` — captures the
# path token up to the first whitespace or closing paren (so Marp size
# hints like ``![bg fit](figures/x.png)`` and titles are excluded).
_IMG_REF_RE = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+)")


def _disallowed_local_image_refs(slides_md: Path) -> list[str]:
    """Return the image paths in ``slides_md`` that are NOT safe to expose
    to Marp's ``--allow-local-files``. That flag makes Marp read any local
    file the (LLM-authored) deck references — e.g. ``![](../../.env)`` —
    so we permit only our own ``figures/`` outputs, ``http(s)://`` URLs,
    and ``data:`` URIs. Absolute paths, parent-directory traversal, and
    any other local path are flagged so the caller refuses to run Marp."""
    try:
        text = slides_md.read_text(encoding="utf-8")
    except OSError:
        return []
    bad: list[str] = []
    for raw in _IMG_REF_RE.findall(text):
        p = raw.strip().strip("\"'")
        if p.lower().startswith(("http://", "https://", "data:")):
            continue
        # Local path: only figures/<name> with no parent-dir escape is OK.
        norm = p.replace("\\", "/").lstrip("./")
        segs = norm.split("/")
        if norm.startswith("figures/") and ".." not in segs:
            continue
        bad.append(p)
    return bad


_MARP_INSTALL_RECIPE = (
    "Install the Marp CLI. No-admin, no-Node option: "
    "`python launch.py --install-marp` drops the standalone binary "
    "(Node bundled, MIT licensed) into `tools/`, where FI finds it "
    "automatically; airgapped hosts can use `--install-marp-from "
    "<archive>`. Otherwise: `npm install -g @marp-team/marp-cli` "
    "(requires Node.js >=14). "
    "On a first run Marp downloads a Chromium build (~150 MB) for its "
    "PDF renderer; the download happens inside `marp` itself, so kick "
    "it off once manually (`marp --version` is enough) before re-running "
    "the quest. The slides.md source is still on disk next to this "
    "file — once marp is installed you can render it manually with "
    "`marp slides.md -o slides.pdf`."
)


_CHROMIUM_HOW_TO_FIX = (
    "Marp's PDF renderer depends on a bundled Chromium build that "
    "either failed to download or can't be launched. Run `marp "
    "--version` once from a terminal to trigger the puppeteer download "
    "(~150 MB) on a network that allows GitHub release downloads. "
    "Behind a corporate proxy you may need to set "
    "`PUPPETEER_EXECUTABLE_PATH` to point at a system Chrome/Chromium "
    "install; see https://pptr.dev/api/puppeteer.configuration. Re-run "
    "the quest after the download completes; `slides.md` is still on "
    "disk and can be rendered manually."
)


def _classify_marp_failure(
    *,
    label: str,
    code: int | None,
    stderr_tail: str,
    timed_out: bool,
) -> tuple[str, str, str]:
    """Map a failed marp invocation to a ``(reason_code, summary,
    how_to_fix)`` triple suitable for :func:`render_skip_md`.

    Inspecting stderr lets us tell apart "chromium missing" (a known
    first-run failure on locked-down hosts) from a generic non-zero
    exit. The chromium probe is a substring scan against
    :data:`_CHROMIUM_SIGNATURES`; case-insensitive because marp
    versions disagree on capitalization in their error messages.
    """
    if timed_out:
        return (
            "marp_timeout",
            f"{label} exceeded its 120 s timeout; slides render skipped.",
            (
                "The marp invocation took longer than the 120 s "
                "budget. The first-ever marp run downloads Chromium "
                "(~150 MB); retry after the cache is populated. If it "
                "still times out, raise the `timeout_s` in "
                "`generation/slides.py:_run_cli`."
            ),
        )
    stderr_lower = stderr_tail.lower()
    if any(sig in stderr_lower for sig in _CHROMIUM_SIGNATURES):
        return (
            "chromium_missing",
            (
                f"{label} failed because marp could not locate its "
                f"Chromium dependency. stderr tail:\n\n```\n"
                f"{stderr_tail.strip()}\n```"
            ),
            _CHROMIUM_HOW_TO_FIX,
        )
    rc_repr = "killed" if code is None else str(code)
    return (
        f"marp_rc_{rc_repr}",
        (
            f"{label} exited with rc={rc_repr}; slides render skipped. "
            f"stderr tail:\n\n```\n{stderr_tail.strip()}\n```"
        ),
        (
            "Marp reported a render error. Most common cause: a "
            "malformed Marp directive in `slides.md` (theme name "
            "typo, unbalanced frontmatter). Open `slides.md` and try "
            "rendering manually with `marp slides.md -o slides.pdf` "
            "to see the full error context."
        ),
    )


async def _run_cli(
    argv: list[str], *, cwd: Path, label: str, timeout_s: float = 120.0,
) -> tuple[bool, tuple[str, str, str] | None]:
    """Run a render CLI and log+swallow any failure.

    Returns ``(ok, fail_reason)``:
      * ``ok`` is True iff the process exited 0.
      * ``fail_reason`` is the ``(code, summary, how_to_fix)`` triple
        the caller can plug into :func:`render_skip_md`, or ``None``
        for non-marp invocations (the caller decides whether to write
        a diagnostic; pandoc-pptx for example doesn't).

    The slide generator wants all three target formats to be
    independent — a failing `marp pdf` should not stop `pandoc pptx`
    from running, and a spawn-time exception (the path became invalid
    between `shutil.which` and `create_subprocess_exec`, permission
    denied, exec-format error, etc.) must NOT abort the whole
    generator either."""
    is_marp = label.startswith("marp ")
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            # Marp reads stdin until it closes, even when given a file. A
            # quest whose stdin is an open pipe (a background shell; the web
            # launcher passes on the server's stdin) left Marp waiting until
            # the 120 s timeout.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, PermissionError, OSError) as e:
        _log.warning("%s spawn failed: %r; skipped", label, e)
        if is_marp:
            return False, (
                "marp_spawn_failed",
                (
                    f"{label} could not be spawned: {e!r}. shutil.which "
                    "resolved marp but the OS refused the exec — likely "
                    "a stale PATH entry or a `.cmd` shim that vanished."
                ),
                _MARP_INSTALL_RECIPE,
            )
        return False, None
    try:
        _stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        _log.warning("%s timeout (%.0fs); skipped", label, timeout_s)
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        if is_marp:
            return False, _classify_marp_failure(
                label=label, code=None, stderr_tail="", timed_out=True,
            )
        return False, None
    if proc.returncode != 0:
        stderr_tail = stderr.decode("utf-8", errors="replace")[-400:]
        _log.warning(
            "%s rc=%d stderr=%s",
            label, proc.returncode, stderr_tail,
        )
        if is_marp:
            return False, _classify_marp_failure(
                label=label,
                code=proc.returncode,
                stderr_tail=stderr_tail,
                timed_out=False,
            )
        return False, None
    return True, None


def _strip_outer_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```") and s.endswith("```"):
        # remove first fence line and trailing fence
        first_nl = s.find("\n")
        if first_nl > 0:
            s = s[first_nl + 1 : -3].rstrip()
    return s


_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_SLIDE_BREAK_RE = re.compile(r"^---[ \t]*$", re.MULTILINE)
_BARE_FENCE_LINE_RE = re.compile(r"^```[ \t]*$", re.MULTILINE)


_FIGURE_LINE_RE = re.compile(r"^\s*!\[(?P<alt>[^\]]*)\]\([^)]*\)\s*$")


def _figures_as_own_paragraphs(content: str) -> str:
    """Put a blank line around each figure line that has text directly above
    or below it.

    Text on the next line joins the figure's paragraph, and the theme can fit
    a figure into the room left on its slide only when the figure is a
    paragraph of its own. gemma4 wrote its "**Figure 2:** ..." caption on the
    line under the image, and the caption ended up below the slide's bottom
    edge. Background figures (``![bg ...]``) are not in the text flow and are
    left alone, as is anything inside a code fence."""
    lines = content.split("\n")
    out: list[str] = []
    in_code = False
    for index, line in enumerate(lines):
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _FIGURE_LINE_RE.match(line)
        if match is None or "bg" in match.group("alt").split():
            out.append(line)
            continue
        if out and out[-1].strip():
            out.append("")
        out.append(line)
        if index + 1 < len(lines) and lines[index + 1].strip():
            out.append("")
    return "\n".join(out)


def _fences_as_slide_breaks(content: str) -> str:
    """Treat bare ``` lines as slide breaks when the deck has no ``---`` break.

    A model copying the fenced examples in the prompt can separate its slides
    with ``` instead of ``---``. Marp then renders the deck as a few slides
    full of code blocks, and the pptx drops every fenced slide. A deck with at
    least one real ``---`` break is left alone, so its code blocks survive."""
    match = _FRONT_MATTER_RE.match(content)
    head = content[: match.end()] if match else ""
    body = content[len(head):]
    if _SLIDE_BREAK_RE.search(body) or not _BARE_FENCE_LINE_RE.search(body):
        return content
    slides = [s.strip() for s in _BARE_FENCE_LINE_RE.split(body)]
    return head + "\n" + "\n\n---\n\n".join(s for s in slides if s) + "\n"
