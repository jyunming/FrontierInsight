"""Folder summarizer (post-Phase-P utility).

Given a folder of mixed content (papers, source code, study notes,
experiment logs, prior quest outputs), produce a structured markdown
summary. The LLM picks which sections apply based on what's in the
inventory (literature / code / study / execution / mixed).

CLI: ``python launch.py --summarize <folder>``
VSCode: ``@fi /summarize <folder>``

Always writes input files + the produced summary into Axon as
``fi_summary_input`` / ``fi_summary`` so future quests can retrieve
prior context.
"""

from __future__ import annotations

import logging
import string
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ProviderConfig
from .knowledge import Knowledge, _load_local_paper
from .provider import (
    LLMClient,
    PROXY_PROVIDERS,
    ProxySupervisor,
    resolve_endpoint_async,
)

_log = logging.getLogger("frontier_insight.summarizer")

PROMPT_PATH = Path(__file__).resolve().parent.parent / "agents" / "summarize.md"

# Per-file content cap fed into the prompt. Anything larger gets
# truncated. Sized so a folder with ~20 files fits inside a 100 KB
# prompt budget (~25K tokens).
_PER_FILE_PROMPT_CHARS = 4_000

# Per-file cap for the Axon ingest path. Larger than the prompt
# preview so future retrieval can surface body content beyond the
# first ~4 KB. Bounded so a single huge file doesn't dominate the
# corpus or blow up Axon's chunking.
_PER_FILE_AXON_CHARS = 50_000

# Directories we always skip during recursion. These tend to be noise
# (caches, build outputs, version-control metadata) that drown out the
# user-meaningful files.
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", ".venv", "venv", "__pycache__",
    ".pytest_cache", ".pytest_tmp", ".mypy_cache", ".ruff_cache",
    "dist", "build", ".next", ".turbo",
    ".idea", ".vscode",
    "outputs",         # FI's own quest output dir — don't recurse into it
    ".fi",
})

# Extension → kind classification. Files with extensions outside this
# map are still listed in the manifest but skipped from content blocks
# (we don't try to summarize binary / unknown formats).
_KIND_BY_EXT: dict[str, str] = {
    # Papers / notes.
    ".pdf": "paper", ".md": "paper", ".txt": "paper", ".rst": "paper",
    # Code.
    ".py": "code", ".ts": "code", ".tsx": "code",
    ".js": "code", ".jsx": "code",
    ".rs": "code", ".go": "code", ".java": "code", ".kt": "code",
    ".c": "code", ".h": "code", ".cpp": "code", ".hpp": "code",
    ".cs": "code", ".swift": "code", ".rb": "code", ".php": "code",
    ".sh": "code", ".ps1": "code",
    # Notebooks (parsed as text — good enough for the inventory).
    ".ipynb": "notebook",
    # Configs.
    ".yaml": "config", ".yml": "config", ".json": "config",
    ".toml": "config", ".ini": "config", ".cfg": "config",
    # Logs / outputs.
    ".log": "log", ".jsonl": "log", ".csv": "log", ".tsv": "log",
}

_VALID_KINDS: frozenset[str] = frozenset(
    {"literature", "code", "study", "execution", "mixed", "auto"}
)


@dataclass
class FileEntry:
    """One classified file in the folder. The summarizer prompt cites
    these by their ``ident`` (e.g. ``[3]``) so the reviewer can
    cross-reference back."""

    ident: int               # 1-based index for citation in the summary
    path: Path               # absolute path (kept for ingest; NOT shown in prompt)
    rel_path: str            # forward-slashed relative-to-folder path
    kind: str                # one of: paper, code, notebook, config, log, other
    size_bytes: int
    preview: str = ""        # up to _PER_FILE_PROMPT_CHARS — fed into the LLM prompt
    axon_content: str = ""   # up to _PER_FILE_AXON_CHARS — fed into Axon ingest


@dataclass
class SummaryArtifacts:
    """What the summarizer returns. The summary_path is the headline
    output; the rest is metadata the caller can render or persist."""

    summary_id: str
    summary_root: Path
    summary_path: Path           # the markdown file
    detected_kind: str           # literature / code / study / execution / mixed
    file_count: int
    ingested_to_axon: bool       # True iff Axon was reachable and writes succeeded
    raw_state: dict[str, Any] = field(default_factory=dict)


def _classify_extension(suffix: str) -> str:
    """Map a file extension to a kind tag. Unknown extensions go to
    ``"other"`` so they still appear in the inventory but get no
    content preview."""
    return _KIND_BY_EXT.get(suffix.lower(), "other")


def _detect_folder_kind(entries: list[FileEntry]) -> str:
    """Pick the dominant content kind from the file mix. Heuristic:
    weight by file count; ``mixed`` is the answer when no kind has a
    plurality."""
    counts: dict[str, int] = {}
    for e in entries:
        if e.kind in ("paper", "notebook"):
            counts["paper"] = counts.get("paper", 0) + 1
        elif e.kind == "code":
            counts["code"] = counts.get("code", 0) + 1
        elif e.kind == "log":
            counts["log"] = counts.get("log", 0) + 1
        # config / other don't drive kind detection on their own.

    total = sum(counts.values()) or 1
    if not counts:
        return "mixed"

    paper_share = counts.get("paper", 0) / total
    code_share = counts.get("code", 0) / total
    log_share = counts.get("log", 0) / total

    # Strong single-kind signal.
    if paper_share >= 0.7:
        return "literature"
    if code_share >= 0.7:
        return "code"
    if log_share >= 0.7:
        return "execution"

    # Code + papers blend → study project.
    if paper_share >= 0.2 and code_share >= 0.2:
        return "study"

    # Code + logs blend → execution-flavored study.
    if code_share >= 0.2 and log_share >= 0.2:
        return "execution"

    return "mixed"


def _read_text(path: Path, kind: str) -> str:
    """Best-effort full-text load. Reuses the knowledge layer's PDF /
    md / txt loader where applicable; falls back to plain UTF-8 read
    for code / config / log files. The caller slices this for the
    prompt preview AND for Axon ingestion at different caps."""
    suffix = path.suffix.lower()
    if suffix in (".pdf", ".md", ".txt", ".rst"):
        # The knowledge loader handles encoding + pypdf gracefully.
        doc = _load_local_paper(path)
        if doc is None:
            return ""
        return doc.content
    if kind in ("code", "config", "log", "notebook"):
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            _log.warning("could not read %s: %s", path, e)
            return ""
    return ""


def _walk_folder(folder: Path) -> list[FileEntry]:
    """Walk ``folder`` recursively, skip noise dirs, classify and
    preview each remaining file. The 1-based ``ident`` is the value
    the summary prompt cites — keep it deterministic via sorted walk."""
    if not folder.is_dir():
        raise NotADirectoryError(f"summarize target is not a directory: {folder}")

    collected: list[Path] = []
    for dirpath, dirnames, filenames in _walk_sorted(folder):
        # In-place prune of `dirnames` keeps os.walk from descending.
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if fn.startswith("."):
                continue
            collected.append(dirpath / fn)

    entries: list[FileEntry] = []
    for i, p in enumerate(collected, start=1):
        suffix = p.suffix
        kind = _classify_extension(suffix)
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        rel = p.relative_to(folder).as_posix()
        full = _read_text(p, kind) if kind != "other" else ""
        entries.append(FileEntry(
            ident=i, path=p, rel_path=rel, kind=kind, size_bytes=size,
            preview=full[:_PER_FILE_PROMPT_CHARS],
            axon_content=full[:_PER_FILE_AXON_CHARS],
        ))
    return entries


def _walk_sorted(folder: Path):
    """``os.walk``-compatible iterator that yields children in sorted
    order so the file IDs are deterministic across runs (important for
    the test suite and for resume-style reproducibility)."""
    import os
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames.sort()
        filenames.sort()
        yield Path(dirpath), dirnames, filenames


def _render_file_manifest(entries: list[FileEntry]) -> str:
    """Compact markdown table the prompt can read directly."""
    if not entries:
        return "(empty — no files found)"
    lines = ["| ID | path | kind | size_kb |", "|---|---|---|---|"]
    for e in entries:
        # Ceil-style conversion so e.g. 1536 bytes shows as 2 KB
        # (rounding down to 1 KB would be misleading for files just
        # over the 1 KB boundary). Empty files show as 0.
        kb = (e.size_bytes + 1023) // 1024 if e.size_bytes > 0 else 0
        lines.append(f"| [{e.ident}] | `{e.rel_path}` | {e.kind} | {kb} |")
    return "\n".join(lines)


def _render_content_blocks(entries: list[FileEntry]) -> str:
    """The per-file previews, header-tagged so the LLM can map citations
    back to inventory IDs."""
    blocks: list[str] = []
    for e in entries:
        if not e.preview:
            continue  # binary / unknown — skip the block (still in manifest)
        truncated = e.preview.strip()
        blocks.append(
            f"## [{e.ident}] `{e.rel_path}` ({e.kind})\n\n{truncated}\n"
        )
    return "\n".join(blocks) if blocks else "(no readable content found)"


def _slugify_folder(folder: Path) -> str:
    """Filesystem-safe slug from the folder's basename, for the
    summary-dir name. Falls back to ``summary`` for path-like inputs
    that produce nothing (e.g. a single-dot relative path)."""
    name = folder.resolve().name.lower()
    out = []
    for ch in name:
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        else:
            out.append("-")
    s = "".join(out).strip("-") or "summary"
    return s[:48]


def _new_summary_id(folder: Path) -> str:
    """Timestamp-prefixed unique id, mirroring ``_new_quest_id``."""
    base = _slugify_folder(folder)
    return f"summary-{int(time.time())}-{base}-{uuid.uuid4().hex[:6]}"


async def summarize_folder(
    folder: Path,
    *,
    provider: ProviderConfig,
    output_dir: Path,
    supervisor: ProxySupervisor | None = None,
    kind: str = "auto",
    knowledge: Knowledge | None = None,
) -> SummaryArtifacts:
    """Top-level entry point. Walks ``folder``, builds a prompt,
    invokes one LLM call, writes the produced markdown, and (if a
    ``Knowledge`` handle is supplied) ingests both the input files
    and the final summary into Axon via ``Knowledge.add_text``.

    ``kind`` is the detected-or-supplied content kind. ``"auto"``
    means infer from the file mix; an explicit value (e.g.
    ``"literature"``) overrides detection — useful when the user
    knows the folder shape and wants to skip the heuristic.
    """
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"kind must be one of {sorted(_VALID_KINDS)}; got {kind!r}",
        )
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"summarize target is not a directory: {folder}")

    entries = _walk_folder(folder)
    detected_kind = kind if kind != "auto" else _detect_folder_kind(entries)
    _log.info(
        "summarize: %s files in %s; kind=%s",
        len(entries), folder, detected_kind,
    )

    prompt_template = string.Template(PROMPT_PATH.read_text(encoding="utf-8"))
    prompt = prompt_template.substitute(
        folder_path=str(folder),
        content_kind=detected_kind,
        file_manifest=_render_file_manifest(entries),
        content_blocks=_render_content_blocks(entries),
    )

    own_supervisor = supervisor is None
    sup = supervisor or ProxySupervisor()
    endpoint = await resolve_endpoint_async(provider, sup)
    client = LLMClient(endpoint)
    try:
        markdown = await client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            node="summarize",
        )
    finally:
        await client.aclose()
        if provider.name in PROXY_PROVIDERS:
            await sup.release(provider.name)
        if own_supervisor:
            await sup.shutdown()

    # Land the produced summary alongside other artifacts so it shows
    # up in any future folder listing and (importantly) so VSCode can
    # open it via a relative path.
    summary_id = _new_summary_id(folder)
    summary_root = (output_dir / summary_id).resolve()
    summary_root.mkdir(parents=True, exist_ok=True)
    summary_path = summary_root / "summary.md"
    summary_path.write_text(markdown.strip() + "\n", encoding="utf-8")

    ingested = False
    if knowledge is not None and knowledge.enabled:
        ingested = _ingest_to_axon(
            knowledge, summary_id=summary_id, folder=folder,
            detected_kind=detected_kind, entries=entries,
            summary_markdown=markdown,
        )

    return SummaryArtifacts(
        summary_id=summary_id,
        summary_root=summary_root,
        summary_path=summary_path,
        detected_kind=detected_kind,
        file_count=len(entries),
        ingested_to_axon=ingested,
        raw_state={
            "folder": str(folder),
            "kind": detected_kind,
            "entries": [
                {"ident": e.ident, "rel_path": e.rel_path,
                 "kind": e.kind, "size_bytes": e.size_bytes}
                for e in entries
            ],
        },
    )


def _ingest_to_axon(
    knowledge: Knowledge, *, summary_id: str, folder: Path,
    detected_kind: str, entries: list[FileEntry], summary_markdown: str,
) -> bool:
    """Best-effort ingest of the input files (as ``fi_summary_input``)
    and the final summary (as ``fi_summary``) into Axon. Returns True
    iff every call succeeded; logs and continues on individual
    failures so a partial ingest is preferable to all-or-nothing.

    Goes through ``Knowledge.add_text`` (the public wrapper) so we
    don't duplicate the Axon API-drift fallback chain that the
    knowledge layer already implements (`add_text` vs `ingest_text`).
    Ingests the LARGER ``axon_content`` field (up to
    ``_PER_FILE_AXON_CHARS`` per file) rather than the prompt preview,
    so future retrieval can surface body content beyond the first
    ~4 KB the LLM saw during summarization."""
    ok = True
    # 1. Each input file with readable content goes in as fi_summary_input.
    for e in entries:
        if not e.axon_content:
            continue
        success = knowledge.add_text(
            text=e.axon_content,
            kind="fi_summary_input",
            metadata={
                "summary_id": summary_id,
                "rel_path": e.rel_path,
                "ident": e.ident,
                "file_kind": e.kind,
                "size_bytes": e.size_bytes,
            },
        )
        if not success:
            ok = False

    # 2. The summary itself.
    success = knowledge.add_text(
        text=summary_markdown,
        kind="fi_summary",
        metadata={
            "summary_id": summary_id,
            "folder": str(folder),
            "detected_kind": detected_kind,
            "file_count": len(entries),
            "generated_at": int(time.time()),
        },
    )
    if not success:
        ok = False
    return ok
