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
"""

from __future__ import annotations

import io
import json
import logging
import re
import string
from pathlib import Path
from typing import Any

from core.config import Config
from core.provider import (
    PROXY_PROVIDERS,
    ImageInputUnsupported,
    LLMClient,
    ProxySupervisor,
    image_part,
    model_for_node,
    resolve_endpoint_async,
)
from generation._cjk import has_cjk
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


def checks_for(kind: str) -> dict[str, str]:
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


async def _check(
    config: Config, kind: str, pdf: Path, quest_root: Path, supervisor: ProxySupervisor | None,
) -> dict[str, Any]:
    doc = measure_pdf(pdf)
    if doc is None or not doc.pages:
        return {"transport": "none", "error": "the PDF could not be read"}
    measured = _REPORTS[kind](doc)
    result: dict[str, Any] = {
        "pages": len(doc.pages),
        "measured": {"metrics": measured["metrics"], "findings": measured["findings"]},
        "findings": [],
        "dropped": [],
    }
    shots_dir = quest_root / ".fi" / "visual_check" / kind
    shots = render_pages(pdf, shots_dir, dpi=_DPI, max_pages=_MAX_PAGES.get(kind))
    if not shots:
        return {**result, "transport": "measurements only", "reason": "the pages could not be rendered"}
    result["pages_checked"] = len(shots)
    images = [_scaled_png(path) for path in shots]
    checks = checks_for(kind)
    prompt = string.Template(PROMPT_PATH.read_text(encoding="utf-8")).safe_substitute(
        kind=_KIND_NAMES.get(kind, kind),
        pages=len(images),
        measurements=_measurements_text(measured),
        checklist="\n".join(f"- `{name}`: {text}" for name, text in checks.items()),
    )
    message = {"role": "user", "content": [{"type": "text", "text": prompt}, *(image_part(data) for data in images)]}
    try:
        reply = await _ask(config, [message], supervisor)
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


_KIND_NAMES = {"paper": "research paper (PDF)", "slides": "slide deck (PDF)", "poster": "research poster (PDF)"}


async def _ask(config: Config, messages: list[dict[str, Any]], supervisor: ProxySupervisor | None) -> str:
    own_supervisor = supervisor is None
    sup = supervisor or ProxySupervisor()
    endpoint = await resolve_endpoint_async(config.provider, sup)
    client = LLMClient(endpoint)
    try:
        return await client.chat(
            messages,
            temperature=0.0,
            model=model_for_node(config.provider.node_models, "visual_check"),
            node="visual_check",
        )
    finally:
        await client.aclose()
        if config.provider.name in PROXY_PROVIDERS:
            await sup.release(config.provider.name)
        if own_supervisor:
            await sup.shutdown()


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
