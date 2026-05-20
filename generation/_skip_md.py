"""Shared renderer for ``*_skipped.md`` diagnostic files.

When an output generator (paper, poster, slides, speech, …) requests
an artifact that can't be produced — missing CLI, refused/empty LLM
response, render-tool error — it writes a single markdown breadcrumb
named ``<kind>_skipped.md`` next to where the artifact would have
landed. Without this, the user just sees a missing file in the result
dict and has to grep ``run.log`` to find out why.

Three generators share the same shape (paper, slides, speech). Poster
has a near-identical local renderer (``_render_poster_skip_md`` in
``generation/poster.py``) that's intentionally kept separate — it
predates this helper and the refactor risk-vs-value isn't high enough
to pull it through. New skip surfaces should call :func:`render_skip_md`
here.

Shape (mirrors ``paper_pdf_skipped.md`` from #55):
- ``# <kind> was requested but not produced``  — H1, the user-facing tagline
- ``## What happened``                          — reason code + summary
- ``## How to fix it``                          — concrete remediation
- footer: ``This file is auto-deleted on the next successful <kind> run.``

The auto-delete footer is a CONTRACT statement, not a wish: each
caller's success path must ``unlink`` the diagnostic so a recovered
quest doesn't leave both the artifact AND a stale skip file on disk.
The poster generator already does this; the slides + speech callers
in this PR follow suit.
"""

from __future__ import annotations


def render_skip_md(
    *,
    kind: str,
    reason_code: str,
    summary: str,
    how_to_fix: str,
) -> str:
    """Render a ``<kind>_skipped.md`` diagnostic body.

    Parameters
    ----------
    kind:
        Human-readable artifact name as it should appear in the H1 and
        footer — e.g. ``"slides"``, ``"speech (talk.md)"``. NOT a path or
        a config-kind enum value. Free-form because the same diagnostic
        may cover multiple render targets (slides covers html/pdf/pptx
        from a single shared engine).
    reason_code:
        Short stable identifier consumed by tests and log filters —
        e.g. ``"no_marp"``, ``"llm_refused_or_empty"``. Kebab/snake_case
        only; no whitespace.
    summary:
        One-paragraph explanation of what failed. The same string is
        usually emitted as a WARNING log line by the caller.
    how_to_fix:
        1-3 sentences pointing at the concrete remediation (install
        command, config flag, retry instruction). Keep actionable —
        not "consider running" but "run X".
    """
    return (
        f"# {kind} was requested but not produced\n\n"
        f"Your `output.kinds` requested {kind}, but the generator "
        f"couldn't produce it.\n\n"
        f"## What happened\n\n"
        f"**Reason code:** `{reason_code}`\n\n"
        f"{summary}\n\n"
        f"## How to fix it\n\n"
        f"{how_to_fix}\n\n"
        f"This file is auto-deleted on the next successful {kind} run.\n"
    )
