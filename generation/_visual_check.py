"""Screenshot and AI check of a finished output.

A rendered PDF (paper, slides or poster) is measured by ``_pdf_measure`` and
screenshotted. The model gets the screenshots, the measurements and a fixed
checklist of what only eyes can see, and answers in JSON.

Free-form critique invents problems, so every finding must quote text that
is visible where the problem is. A finding whose quote is not in the text of
its page (or the page before or after), whose check is not on the list, or
that names no region, is dropped and kept in the report with the reason.

A transport that cannot send images leaves the measurements as the whole
check, and the report says so. The report goes to ``.fi/visual_check.json``,
one entry per output. Nothing here raises: a quest never stops over a check.

``slides.pptx`` is checked as ``slides_pptx``: LibreOffice exports it to a
PDF, which is measured and checked like the slides.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import string
from pathlib import Path
from typing import Any, Awaitable, Callable

from core.config import Config
from core.provider import (
    PROXY_PROVIDERS,
    ImageInputUnsupported,
    LLMClient,
    ProxySupervisor,
    append_cost_row,
    image_part,
    model_for_node,
    resolve_endpoint_async,
)
from generation._cjk import has_cjk
from generation._office_pdf import pptx_to_pdf
from generation._pdf_measure import (
    measure_pdf,
    paper_report,
    poster_report,
    render_pages,
    slides_report,
)

_log = logging.getLogger("frontier_insight.visual_check")

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = REPO_ROOT / "agents" / "visual_check.md"
REPORT_NAME = "visual_check.json"

# gemma4 read a poster title exactly from a 90 dpi screenshot and invented a
# typo at 50 dpi. The longest edge is then capped so an A1 sheet stays a
# reasonable image.
_DPI = 90
_MAX_EDGE_PX = 2000
_MAX_PAGES = {"paper": 12, "slides": 20, "poster": 1}
_REPLY_KEEP_CHARS = 8000

_COMMON_CHECKS = {
    "cut_off_text": "Text cut off at a page edge, or hidden behind a figure or another block.",
    "overlap": "Text, figures or boxes drawn over each other.",
    "unreadable_figure": "A figure that is blank, broken, cropped, or whose labels are too small to read.",
    "raw_markup": "Raw LaTeX, Markdown or HTML shown as text, for example \\textbf, $x^2$, **bold** or <br>.",
    "broken_math": "An equation or symbol drawn wrongly, or boxes or question marks in place of characters.",
    "garbled_text": "Garbled characters, missing letters, or words run together.",
    "empty_area": "A large empty area inside the content, not counting the page margins.",
}
_KIND_CHECKS = {
    "poster": {
        "reading_order": "A column that starts in the middle of a section, or blocks whose reading order is unclear.",
    },
    "slides": {
        "slide_overflow": "Text running past the bottom of a slide or into its footer.",
        "crowded_slide": "A slide with more text than a speaker could present.",
    },
    "paper": {
        "table_overflow": "A table wider than its text column, or cut off by the page.",
    },
}
_REPORTS = {"paper": paper_report, "slides": slides_report, "poster": poster_report}
_SEVERITIES = ("high", "medium", "low")
# The pptx is a slide deck: the slides' measurements, checklist and page cap
# apply. Its report entry and screenshots keep their own name.
PPTX_KIND = "slides_pptx"
_BASE_KIND = {PPTX_KIND: "slides"}
LABELS = {"paper": "paper", "slides": "slides", "poster": "poster", PPTX_KIND: "slides.pptx"}


def checks_for(kind: str) -> dict[str, str]:
    kind = _BASE_KIND.get(kind, kind)
    return {**_COMMON_CHECKS, **_KIND_CHECKS.get(kind, {})}


async def check_pdf(
    config: Config,
    kind: str,
    pdf: Path,
    quest_root: Path,
    *,
    supervisor: ProxySupervisor | None = None,
) -> dict[str, Any]:
    """Measure, screenshot and check ``pdf``; write and return its report."""
    report: dict[str, Any] = {"kind": kind, "pdf": pdf.name}
    try:
        report.update(await _check(config, kind, pdf, quest_root, supervisor))
    except Exception as exc:  # noqa: BLE001 — a check must never stop the quest
        _log.warning("visual check of %s failed: %r", pdf.name, exc)
        report.update({"transport": "none", "error": f"{type(exc).__name__}: {exc}"[:500]})
    _write_report(quest_root, kind, report)
    return report


async def check_pptx(
    config: Config,
    pptx: Path,
    quest_root: Path,
    *,
    supervisor: ProxySupervisor | None = None,
) -> dict[str, Any]:
    """Export ``pptx`` through LibreOffice and check the PDF as ``slides_pptx``.
    Without LibreOffice the report says the pptx was not checked. Never raises."""
    out_dir = quest_root / ".fi" / "visual_check" / PPTX_KIND
    try:
        pdf, reason = await asyncio.to_thread(pptx_to_pdf, pptx, out_dir)
    except Exception as exc:  # noqa: BLE001 — a check must never stop the quest
        pdf, reason = None, f"{type(exc).__name__}: {exc}"[:500]
    if pdf is None:
        report = {"kind": PPTX_KIND, "pdf": pptx.name, "transport": "none", "reason": reason}
        _write_report(quest_root, PPTX_KIND, report)
        return report
    return await check_pdf(config, PPTX_KIND, pdf, quest_root, supervisor=supervisor)


async def _check(
    config: Config, kind: str, pdf: Path, quest_root: Path, supervisor: ProxySupervisor | None,
) -> dict[str, Any]:
    doc = measure_pdf(pdf)
    if doc is None or not doc.pages:
        return {"transport": "none", "error": "the PDF could not be read"}
    base = _BASE_KIND.get(kind, kind)
    measured = _REPORTS[base](doc)
    result: dict[str, Any] = {
        "pages": len(doc.pages),
        "measured": {"metrics": measured["metrics"], "findings": measured["findings"]},
        "findings": [],
        "dropped": [],
    }
    images = _screenshots(pdf, quest_root, kind)
    if not images:
        return {**result, "transport": "measurements only", "reason": "the pages could not be rendered"}
    result["pages_checked"] = len(images)
    checks = checks_for(kind)
    prompt = string.Template(PROMPT_PATH.read_text(encoding="utf-8")).safe_substitute(
        kind=_KIND_NAMES.get(kind, kind),
        pages=len(images),
        measurements=_measurements_text(measured),
        checklist="\n".join(f"- `{name}`: {text}" for name, text in checks.items()),
    )
    message = {"role": "user", "content": [{"type": "text", "text": prompt}, *(image_part(data) for data in images)]}
    try:
        reply = await _ask(config, [message], supervisor, quest_root)
    except ImageInputUnsupported as exc:
        return {**result, "transport": "measurements only", "reason": str(exc)}
    result["transport"] = "images"
    result["reply"] = reply[:_REPLY_KEEP_CHARS]
    parsed = _json_object(reply)
    raw = parsed.get("findings") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        result["reason"] = "the reply had no findings list"
        return result
    result["findings"], result["dropped"] = grounded_findings(raw, doc, set(checks))
    return result


_KIND_NAMES = {
    "paper": "research paper (PDF)",
    "slides": "slide deck (PDF)",
    "poster": "research poster (PDF)",
    PPTX_KIND: "slide deck (PowerPoint file, exported to PDF by LibreOffice)",
}


async def _ask(
    config: Config, messages: list[dict[str, Any]], supervisor: ProxySupervisor | None, quest_root: Path,
) -> str:
    own_supervisor = supervisor is None
    sup = supervisor or ProxySupervisor()
    endpoint = await resolve_endpoint_async(config.provider, sup)
    client = LLMClient(endpoint)
    try:
        reply = await client.chat(
            messages,
            temperature=0.0,
            model=model_for_node(config.provider.node_models, "visual_check"),
            node="visual_check",
        )
        append_cost_row(quest_root / ".fi", node="visual_check", model=client.last_model, usage=client.last_usage)
        return reply
    finally:
        await client.aclose()
        if config.provider.name in PROXY_PROVIDERS:
            await sup.release(config.provider.name)
        if own_supervisor:
            await sup.shutdown()


def _screenshots(pdf: Path, quest_root: Path, kind: str) -> list[bytes]:
    """Render the pages to ``.fi/visual_check/<kind>/page-N.png``, replacing
    the screenshots of any earlier version, and return them as PNG bytes."""
    shots_dir = quest_root / ".fi" / "visual_check" / kind
    for old in shots_dir.glob("page-*.png"):
        old.unlink(missing_ok=True)
    shots = render_pages(pdf, shots_dir, dpi=_DPI, max_pages=_MAX_PAGES.get(_BASE_KIND.get(kind, kind)))
    return [_scaled_png(path) for path in shots]


def _scaled_png(path: Path) -> bytes:
    """The screenshot as PNG bytes, its longest edge at most _MAX_EDGE_PX.
    The scaled image replaces the file, so the report's screenshots are what
    the model saw."""
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        if max(image.size) > _MAX_EDGE_PX:
            image.thumbnail((_MAX_EDGE_PX, _MAX_EDGE_PX))
            image.save(path)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return buffer.getvalue()


def _measurements_text(measured: dict[str, Any]) -> str:
    metrics = {k: v for k, v in measured["metrics"].items() if k != "per_page"}
    lines = [f"- {key}: {value}" for key, value in metrics.items()]
    if measured["findings"]:
        lines.append("- Already found by the script (do not repeat):")
        lines += [
            f"  - page {f['page']}, {f['region']}: {f['problem']}" for f in measured["findings"][:20]
        ]
    else:
        lines.append("- The script found no problems.")
    return "\n".join(lines)


def _json_object(text: str) -> Any:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s+(\w)")
_SEPARATORS_RE = re.compile(r"[\W_]+")


def _words(text: str) -> list[str]:
    return _SEPARATORS_RE.sub(" ", text.casefold()).split()


def _compact(text: str) -> str:
    """Letters and digits only. A PDF's text layer differs from what is drawn
    in its separators: LaTeX draws an underscore as a rule the text layer
    reads as a space, splits ligatures ("Dif ferential") and hyphenates at
    line ends. With every separator removed, a quote copied from the picture
    still matches the text layer."""
    return "".join(_words(_HYPHEN_BREAK_RE.sub(r"\1\2", text)))


def grounded_findings(raw: list[Any], doc, checks: set[str]) -> tuple[list[dict], list[dict]]:
    """Split the model's findings into those tied to visible text on their
    page and those that are not, each dropped one with its reason."""
    page_text = {
        page.number: _compact(" ".join(line.text for line in page.lines)) for page in doc.pages
    }
    kept: list[dict] = []
    dropped: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            page = int(item.get("page"))
        except (TypeError, ValueError):
            page = 0
        finding = {
            "check": str(item.get("check") or ""),
            "page": page,
            "region": str(item.get("region") or "").strip(),
            "problem": str(item.get("problem") or "").strip(),
            "severity": item.get("severity") if item.get("severity") in _SEVERITIES else "medium",
            "quote": str(item.get("quote") or "").strip(),
            "source": "model",
        }
        quote = _compact(finding["quote"])
        # Three words, or for Chinese, Japanese or Korean, which has no
        # spaces between words, six characters.
        long_enough = len(_words(finding["quote"])) >= 3 or (has_cjk(finding["quote"]) and len(quote) >= 6)
        if finding["check"] not in checks:
            reason = "the check is not on the list"
        elif not finding["region"]:
            reason = "no region"
        elif not long_enough:
            reason = "the quote is too short to place"
        elif not any(quote in page_text.get(n, "") for n in (page - 1, page, page + 1)):
            reason = "the quoted text is not on that page"
        else:
            kept.append(finding)
            continue
        dropped.append({**finding, "dropped_because": reason})
    return kept, dropped


def report_summary(quest_root: Path) -> dict[str, Any] | None:
    """Each output's result in brief, for ``frontier_insight_summary.json``
    and the web quest page; ``None`` when nothing was checked."""
    try:
        checks = json.loads((quest_root / ".fi" / REPORT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(checks, dict):
        return None
    return {
        kind: {
            "label": LABELS.get(kind, kind),
            "transport": report.get("transport"),
            "problems_seen": len(report.get("findings") or []),
            "problems_measured": len((report.get("measured") or {}).get("findings") or []),
            "redos": sum(1 for a in report.get("attempts") or [] if a.get("attempt")),
            "reason": report.get("reason") or report.get("error"),
        }
        for kind, report in checks.items() if isinstance(report, dict)
    }


def _write_report(quest_root: Path, kind: str, report: dict[str, Any]) -> None:
    path = quest_root / ".fi" / REPORT_NAME
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing[kind] = report
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        _log.info("visual check: could not write %s (%r)", path, exc)


# ---------------------------------------------------------------------------
# Redo

# Findings a new version can fix. The slides' model writes the whole deck,
# layout included. The poster's layout is the planner's, so only text the
# model wrote counts there: a new reply would not change empty space, uneven
# columns or overflow. The paper is never rewritten; a script repairs a last
# page that holds only a line or two by making the text area taller.
REDO_CHECKS = {
    "slides": frozenset({
        "overflow", "small_font", "cut_off_text", "overlap", "unreadable_figure",
        "raw_markup", "broken_math", "garbled_text", "slide_overflow", "crowded_slide",
    }),
    "poster": frozenset({"raw_markup", "broken_math", "garbled_text", "captions"}),
    "paper": frozenset({"last_page_nearly_empty"}),
}
_WEIGHTS = {"high": 3, "medium": 2, "low": 1}
# The files one version of an output consists of, copied aside before a redo
# so a version that checks worse can be put back.
_VERSION_FILES = {
    "slides": ("slides.md", "slides.html", "slides.pdf", "slides.pptx"),
    "poster": ("poster.pdf", "poster.tex", ".fi/poster_fit.json", ".fi/poster_reply.txt"),
    "paper": ("paper.pdf",),
}
# A paper is repaired at most once. The repair adds one line to the text area
# and moves the footer up by the same amount so it stays put. Measured on all
# nine templates, a second line would leave the footer only about 9-12 pt
# below the text on the tighter ones (ieee_access, policy_brief).
_REDO_LIMITS = {"paper": 1}


def _open_findings(report: dict[str, Any]) -> list[dict]:
    return list((report.get("measured") or {}).get("findings") or []) + list(report.get("findings") or [])


def redo_findings(kind: str, report: dict[str, Any]) -> list[dict]:
    """The medium and high findings a new version of ``kind`` could fix."""
    checks = REDO_CHECKS.get(kind, frozenset())
    return [
        f for f in _open_findings(report)
        if f.get("check") in checks and f.get("severity") in ("high", "medium")
    ]


def score(report: dict[str, Any]) -> int:
    """Lower is better: every open finding, weighted by its severity."""
    return sum(_WEIGHTS.get(f.get("severity"), 2) for f in _open_findings(report))


def feedback_text(findings: list[dict]) -> str:
    lines = [
        "## Problems a check found in your previous version",
        "",
        "A check of the rendered pages found these. Fix them in this version and keep what was right.",
        "",
    ]
    for f in findings:
        where = f"page {f.get('page')}" + (f", {f['region']}" if f.get("region") else "")
        near = f' Near: "{f["quote"]}".' if f.get("quote") else ""
        lines.append(f"- {where}: {f.get('problem')}{near}")
    return "\n".join(lines)


def _copy_version(source: Path, target: Path, names: tuple[str, ...]) -> None:
    """Make ``target`` hold exactly ``source``'s copy of each named file."""
    import shutil

    for name in names:
        src, dst = source / name, target / name
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            dst.unlink(missing_ok=True)


async def check_and_redo(
    config: Config,
    kind: str,
    pdf: Path,
    quest_root: Path,
    regenerate: Callable[[str], Awaitable[Any]],
    *,
    supervisor: ProxySupervisor | None = None,
) -> dict[str, Any]:
    """Check ``pdf``. While the check finds problems a new version could fix,
    ask ``regenerate(feedback)`` for one, at most
    ``output.visual_check_max_redos`` times, and keep the version that checks
    best; a version that checks no better is put back. Never raises."""
    report = await check_pdf(config, kind, pdf, quest_root, supervisor=supervisor)
    attempts: list[dict[str, Any]] = [{"attempt": 0, "score": score(report), "kept": True}]
    names = _VERSION_FILES.get(kind, ())
    limit = min(int(getattr(config.output, "visual_check_max_redos", 0) or 0), _REDO_LIMITS.get(kind, 2))
    for attempt in range(1, limit + 1):
        fixable = redo_findings(kind, report)
        if not fixable or report.get("transport") == "none":
            break
        saved = quest_root / ".fi" / "visual_check" / kind / f"before-redo-{attempt}"
        _copy_version(quest_root, saved, names)
        entry: dict[str, Any] = {"attempt": attempt, "redo_for": sorted({f["check"] for f in fixable})}
        attempts.append(entry)
        try:
            await regenerate(feedback_text(fixable))
        except Exception as exc:  # noqa: BLE001 — a redo must never stop the quest
            _log.warning("visual check: redo %d of %s failed: %r", attempt, kind, exc)
            entry.update({"error": f"{type(exc).__name__}: {exc}"[:300], "kept": False})
            _copy_version(saved, quest_root, names)
            break
        new = await check_pdf(config, kind, pdf, quest_root, supervisor=supervisor)
        entry["score"] = score(new)
        if new.get("transport") != "none" and entry["score"] < score(report):
            for earlier in attempts[:-1]:
                earlier["kept"] = False
            entry["kept"] = True
            report = new
            continue
        entry["kept"] = False
        _copy_version(saved, quest_root, names)
        _screenshots(pdf, quest_root, kind)
        break
    report["attempts"] = attempts
    _write_report(quest_root, kind, report)
    return report
