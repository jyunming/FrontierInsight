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
3. ``pandoc slides.md -o slides.pptx`` — produced when pandoc is on
   PATH. Real PowerPoint file the user can open and edit in Office /
   Google Slides / Keynote, satisfying the common "I want an
   actual presentation, not a markdown file" use case.

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
import shutil
import string
from pathlib import Path

from core.config import Config
from core.engine import QuestArtifacts
from core.provider import (
    LLMClient,
    ProxySupervisor,
    PROXY_PROVIDERS,
    resolve_endpoint_async,
)
from generation._skip_md import render_skip_md

_log = logging.getLogger("frontier_insight.slides")

PROMPT_PATH = Path(__file__).resolve().parent.parent / "agents" / "slides.md"

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
    ) -> dict[str, Path]:
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

        slides_md = await self._author_marp(art, out_dir, supervisor=supervisor)
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
        marp_exe = shutil.which("marp")
        if marp_exe is None:
            msg = "marp CLI not on PATH; slides.html/.pdf skipped"
            _log.warning(msg)
            marp_skip = (
                "no_marp",
                msg,
                _MARP_INSTALL_RECIPE,
            )
        else:
            for ext in ("html", "pdf"):
                out_path = out_dir / f"slides.{ext}"
                ok, fail_reason = await _run_cli(
                    [marp_exe, str(slides_md), "-o", str(out_path)],
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
        pandoc_exe = shutil.which("pandoc")
        if pandoc_exe is None:
            _log.warning("pandoc not on PATH; slides.pptx skipped")
        else:
            pptx_path = out_dir / "slides.pptx"
            # `--slide-level=2` makes H2 headings drive slide breaks,
            # matching the structure the LLM produces (deck title at H1,
            # one H2 per slide). Marp's `---` separators also become
            # slide breaks under pandoc's default settings.
            ok, _fail = await _run_cli(
                [pandoc_exe, str(slides_md), "--slide-level=2",
                 "-o", str(pptx_path)],
                cwd=out_dir, label="pandoc pptx",
            )
            if ok:
                result["slides_pptx"] = pptx_path

        return result

    async def _author_marp(
        self,
        art: QuestArtifacts,
        out_dir: Path,
        *,
        supervisor: ProxySupervisor | None,
    ) -> Path:
        paper_md = art.paper_md.read_text(encoding="utf-8") if art.paper_md else ""
        figures = []
        if art.figures_dir and art.figures_dir.is_dir():
            figures = sorted(p.name for p in art.figures_dir.iterdir() if p.is_file())
        prompt = string.Template(PROMPT_PATH.read_text(encoding="utf-8")).substitute(
            paper_md=paper_md[:8000],
            figure_list="\n".join(f"- figures/{f}" for f in figures) or "(none)",
        )
        own_supervisor = supervisor is None
        sup = supervisor or ProxySupervisor()
        endpoint = await resolve_endpoint_async(self.config.provider, sup)
        client = LLMClient(endpoint)
        try:
            text = await client.chat([{"role": "user", "content": prompt}], temperature=0.2)
        finally:
            await client.aclose()
            if self.config.provider.name in PROXY_PROVIDERS:
                await sup.release(self.config.provider.name)
            if own_supervisor:
                await sup.shutdown()

        slides_md = out_dir / "slides.md"
        slides_md.write_text(_strip_outer_fence(text), encoding="utf-8")
        _log.info("slides.md written (%d bytes)", slides_md.stat().st_size)
        return slides_md


_MARP_INSTALL_RECIPE = (
    "Install the Marp CLI and ensure it lands on PATH. Recommended: "
    "`npm install -g @marp-team/marp-cli` (requires Node.js >=14). "
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
