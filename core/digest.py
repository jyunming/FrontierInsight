"""Weekly project-manager digest.

Walks ``outputs/`` for quests created or touched inside a date window,
classifies each by terminal-node state (read from LangGraph's
``state.sqlite`` checkpoint store), diffs against the most-recent prior
digest in ``outputs/_digests/`` to surface what *changed* since last
week, then asks the LLM to produce a markdown report.

CLI:    ``python launch.py --digest [--days N]``
VSCode: ``@fi /digest [--days N]``

Design rationale (see also docs/USAGE.md):

* The structured diff (newly-completed, promoted, still-in-progress,
  new-in-progress, dropped, stalled) is computed in code, not by the
  LLM. The model only writes the 1-line synthesis per quest. This is
  the difference between "the model hallucinated I finished X" and
  "FI deterministically saw X move from ``write`` to ``review``
  between two checkpoint snapshots."
* The current digest's prior counterpart is loaded as raw markdown and
  handed to the LLM as context, so themes can carry forward without
  the model having to re-derive them from scratch.
* Output lands in ``outputs/_digests/<digest_id>.md``. The leading
  underscore keeps it sorted at the top of ``ls outputs/`` and
  visually distinct from real quest dirs (mirrors ``outputs/_drafts/``
  from ``/resume``).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import ProviderConfig
from .knowledge import Knowledge
from .provider import (
    LLMClient,
    PROXY_PROVIDERS,
    ProxySupervisor,
    resolve_endpoint_async,
)

_log = logging.getLogger("frontier_insight.digest")

PROMPT_PATH = Path(__file__).resolve().parent.parent / "agents" / "digest.md"

# Default window: rolling 7 days from "now". Overridable via the
# --days CLI flag or by passing explicit since/until to generate_digest.
_DEFAULT_DAYS = 7

# Per-quest abstract cap (chars) fed into the digest prompt. Sized so
# 50+ quests still fit comfortably under any model's context window
# while still giving the LLM enough text to write a useful 1-line
# synthesis. The full paper.md is NOT included.
_ABSTRACT_CHARS = 800

# Hard cap on quests included in the digest prompt itself. If a user
# has run >300 quests in 7 days something's wrong with the heuristic;
# truncate the tail with a note rather than blow the token budget.
# Mirrors the cap pattern in core/summarizer.py.
_MAX_PROMPT_QUESTS = 300

# Stalled threshold: a quest that's appeared as still_in_progress in
# this many consecutive prior digests gets a stronger flag in the
# output. 3 weeks of no progress is the call-to-action point.
_STALLED_DIGEST_COUNT = 3

# Quest_id pattern: <epoch>-<slug>-<6 hex>. Parsing the leading epoch
# is cheaper and more accurate than os.stat (works for quests on
# foreign filesystems where mtime might be lost on copy).
_QUEST_ID_RE = re.compile(r"^(\d{10})-[a-z0-9-]+-[0-9a-f]{6}$")


@dataclass
class QuestSnapshot:
    """One quest as it appears at digest-generation time. Constructed
    by walking the quest directory; nothing here is LLM-derived."""

    quest_id: str
    title: str
    topic: str
    created_at: datetime
    last_modified: datetime
    terminal_node: str          # "review" (completed) | "in_progress" | "unknown"
    paper_abstract: str         # first paragraph(s) of paper.md, truncated
    provider: str | None
    quest_root: Path


@dataclass
class WeekDiff:
    """Structured diff between two consecutive digest windows. All
    lists are stable-sorted by ``quest_id`` for deterministic output."""

    prev_digest_id: str | None
    newly_completed: list[QuestSnapshot] = field(default_factory=list)
    promoted: list[QuestSnapshot] = field(default_factory=list)
    still_in_progress: list[QuestSnapshot] = field(default_factory=list)
    new_in_progress: list[QuestSnapshot] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    stalled: list[str] = field(default_factory=list)


@dataclass
class DigestArtifacts:
    """What :func:`generate_digest` returns. The ``digest_path`` is
    the headline output; everything else is metadata for the caller
    (the VSCode extension or a cron wrapper) to render or persist."""

    digest_id: str
    digest_path: Path
    quest_count: int
    completed_count: int
    in_progress_count: int
    diff: WeekDiff
    ingested_to_axon: bool
    raw_state: dict[str, Any] = field(default_factory=dict)


# ---------- quest-id / timestamp helpers ------------------------------------


def _parse_quest_id_timestamp(quest_id: str) -> datetime | None:
    """Extract the creation timestamp embedded in a quest_id prefix.
    Returns ``None`` for legacy / malformed quest IDs."""
    m = _QUEST_ID_RE.match(quest_id)
    if m is None:
        return None
    try:
        epoch = int(m.group(1))
    except ValueError:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _quest_title_from_id(quest_id: str) -> str:
    """Slug → reading-friendly title fallback when no
    frontier_insight_summary.json is available."""
    m = _QUEST_ID_RE.match(quest_id)
    if m is None:
        return quest_id
    # Trim the leading epoch and trailing -<6 hex>, hyphenize, title-case.
    slug = quest_id[len(m.group(1)) + 1 : -7]
    return slug.replace("-", " ").strip()


# ---------- terminal-node detection -----------------------------------------


def _detect_terminal_node(state_sqlite: Path) -> str:
    """Read the LangGraph checkpoint store to decide whether a quest
    completed or is still in progress.

    Returns one of:
      ``"review"``       — the review channel was written; the LangGraph
                           ENDs after that node, so the quest is complete.
      ``"in_progress"``  — checkpoints exist but no review write yet.
      ``"unknown"``      — the file is missing/corrupt/legacy. Treated as
                           in_progress for diff purposes but flagged.

    Implementation note: we don't deserialize the msgpack/json blobs;
    the existence of the ``review`` channel in the ``writes`` table is
    enough signal and is forward-compatible with any future channel
    additions.
    """
    if not state_sqlite.is_file():
        return "unknown"
    # ``sqlite3.connect`` opens lazily and doesn't fail on a non-DB
    # file until you query — so we wrap the whole read in one try/except.
    try:
        con = sqlite3.connect(state_sqlite)
        try:
            tables = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            if "writes" not in tables:
                return "unknown"
            # The review channel is written exactly once by the review
            # node, regardless of whether the quest then re-routes (the
            # iterate path goes back to design — but the review write
            # itself happens). Any row → reached the review node.
            row = con.execute(
                "SELECT 1 FROM writes WHERE channel='review' LIMIT 1"
            ).fetchone()
            return "review" if row is not None else "in_progress"
        finally:
            con.close()
    except sqlite3.Error:
        return "unknown"


# ---------- snapshot collection ---------------------------------------------


def _read_paper_abstract(paper_md: Path) -> str:
    """Return the first paragraph(s) of paper.md up to ``_ABSTRACT_CHARS``.
    Strips the title header so the abstract isn't preceded by a duplicate
    title in the digest."""
    try:
        text = paper_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    # Drop a leading "# Title" line if present — we already have the
    # title from the summary JSON.
    if text.startswith("#"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    return text.strip()[:_ABSTRACT_CHARS]


def _read_summary_json(quest_root: Path) -> dict[str, Any]:
    """Read ``frontier_insight_summary.json`` if present, else ``{}``.
    Doesn't raise on malformed JSON — returns ``{}`` so one corrupt
    quest doesn't tank the whole digest."""
    path = quest_root / "frontier_insight_summary.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _last_modified_under(root: Path) -> datetime:
    """Max mtime across the quest directory tree. Used to decide
    whether a quest was *touched* during the digest window even if it
    was created earlier."""
    latest = 0.0
    for p in root.rglob("*"):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if mtime > latest:
            latest = mtime
    if latest == 0.0:
        try:
            latest = root.stat().st_mtime
        except OSError:
            latest = time.time()
    return datetime.fromtimestamp(latest, tz=timezone.utc)


def _collect_quest_snapshots(
    outputs_dir: Path, since: datetime, until: datetime,
) -> list[QuestSnapshot]:
    """Walk ``outputs_dir`` and return snapshots for quests whose
    last-modified time is in ``[since, until]``. The window check
    uses *last_modified*, not created_at, so quests that were resumed
    after a long break show up in the digest of the week they actually
    progressed."""
    if not outputs_dir.is_dir():
        return []
    snapshots: list[QuestSnapshot] = []
    for entry in sorted(outputs_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        # Skip leading-underscore dirs (_digests, _drafts, _portfolio).
        if name.startswith("_") or name.startswith("."):
            continue
        created = _parse_quest_id_timestamp(name)
        if created is None:
            # Legacy quest dir; we have no reliable epoch. Skip rather
            # than guess — including it would risk re-attributing very
            # old quests to the current week if their mtime got
            # touched by some unrelated operation.
            continue
        last_mod = _last_modified_under(entry)
        if not (since <= last_mod <= until):
            continue

        summary_json = _read_summary_json(entry)
        title = (
            summary_json.get("title")
            or summary_json.get("quest_title")
            or _quest_title_from_id(name)
        )
        topic = summary_json.get("topic", "")
        provider = summary_json.get("provider")
        paper_md_path = entry / "paper" / "paper.md"
        if not paper_md_path.is_file():
            # The post-Phase-O layout flattened some output paths; try
            # the alternative location too.
            paper_md_path = entry / "paper.md"
        abstract = (
            _read_paper_abstract(paper_md_path)
            if paper_md_path.is_file() else ""
        )

        terminal = _detect_terminal_node(entry / ".fi" / "state.sqlite")

        snapshots.append(QuestSnapshot(
            quest_id=name,
            title=title,
            topic=topic,
            created_at=created,
            last_modified=last_mod,
            terminal_node=terminal,
            paper_abstract=abstract,
            provider=provider,
            quest_root=entry,
        ))
    return snapshots


# ---------- prior-digest discovery + diff -----------------------------------


# A prior digest's filename. We support two shapes:
#   - ISO-week:        YYYY-WNN.md         (the default cadence)
#   - explicit dates:  YYYY-MM-DD-to-YYYY-MM-DD.md  (custom --days runs)
# Sorting lexicographically gives us most-recent-last for ISO-week
# files. Mixed-shape sorting also works because YYYY-W12 < YYYY-W21
# < YYYY-10-01... in lex order (the dash positions differ). We prefer
# ISO-week files as the "canonical" prior digest when both are present.
_DIGEST_FILENAME_RE = re.compile(
    r"^(?P<iso>\d{4}-W\d{2})\.md$|"
    r"^(?P<from>\d{4}-\d{2}-\d{2})-to-(?P<to>\d{4}-\d{2}-\d{2})\.md$"
)


def _find_prior_digest(digests_dir: Path, this_digest_id: str) -> Path | None:
    """Return the path of the most-recent prior digest, or ``None`` if
    none exists. Prefers ISO-week digests (the standard cadence) over
    explicit-date digests, falling back to the latter if no ISO-week
    digest is present."""
    if not digests_dir.is_dir():
        return None
    iso_week_files: list[Path] = []
    dated_files: list[Path] = []
    for p in digests_dir.iterdir():
        if not p.is_file():
            continue
        m = _DIGEST_FILENAME_RE.match(p.name)
        if m is None:
            continue
        if p.stem == this_digest_id:
            # Don't return ourselves if a previous run produced today's
            # file; the caller wants the *prior* one.
            continue
        if m.group("iso") is not None:
            iso_week_files.append(p)
        else:
            dated_files.append(p)
    if iso_week_files:
        return sorted(iso_week_files)[-1]
    if dated_files:
        return sorted(dated_files)[-1]
    return None


def _quest_ids_in_digest(digest_md: str) -> set[str]:
    """Extract every quest_id mentioned in a prior digest, for the
    "dropped" detection. We use a permissive regex against the same
    pattern used by ``_QUEST_ID_RE`` rather than parsing markdown
    sections — robust against the LLM reflowing whitespace."""
    return set(re.findall(
        r"\b\d{10}-[a-z0-9-]+-[0-9a-f]{6}\b", digest_md,
    ))


def _quest_ids_marked_in_progress(digest_md: str) -> set[str]:
    """Find quest IDs that appeared under an "in progress" header in
    the prior digest. Used by the promoted/still-in-progress diff."""
    # We do a coarse pass: split the prior digest into sections at H2
    # headers, then return quest IDs from the "In progress" section.
    # The prompt template enforces this exact header so the lookup is
    # stable. Falls back to "no in-progress info" on layout drift.
    sections = re.split(r"^##\s+", digest_md, flags=re.MULTILINE)
    in_progress = set()
    for sec in sections:
        if sec.lower().startswith("in progress"):
            in_progress = _quest_ids_in_digest(sec)
            break
    return in_progress


def _quest_ids_marked_still_in_progress(digest_md: str) -> set[str]:
    """Pull quest IDs out of the "Still in progress from last digest"
    block under "What changed since…" — these accumulate the stall
    counter.

    The block we're looking for is:

        **⚠️ Still in progress from last digest:**
        - [qid] title
        - [qid] title

    So the strategy is: locate the "What changed" H2 section, find the
    "Still in progress" header line inside it, then collect every
    bullet line until the next bold header (``**``) or blank line. The
    IDs we extract come from those bullets, not the header line itself.
    """
    sections = re.split(r"^##\s+", digest_md, flags=re.MULTILINE)
    for sec in sections:
        if not sec.lower().startswith("what changed"):
            continue
        lines = sec.splitlines()
        collecting = False
        collected: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not collecting:
                # Header is bold + contains "Still in progress".
                if (
                    stripped.startswith("**")
                    and "still in progress" in stripped.lower()
                ):
                    collecting = True
                continue
            # Stop at the next bold header or a blank line.
            if not stripped or stripped.startswith("**"):
                break
            collected.append(line)
        if collected:
            return _quest_ids_in_digest("\n".join(collected))
    return set()


def _compute_diff(
    this: list[QuestSnapshot], prev_md: str | None, prev_path: Path | None,
) -> WeekDiff:
    """Build a :class:`WeekDiff` between this week's snapshots and the
    prior digest (if any). Uses the prior digest's markdown as the
    source-of-truth for what was where last time — that's robust to
    quests being touched again in this window for unrelated reasons."""
    diff = WeekDiff(prev_digest_id=prev_path.stem if prev_path else None)

    this_by_id = {q.quest_id: q for q in this}
    this_completed = {qid for qid, q in this_by_id.items() if q.terminal_node == "review"}
    this_in_progress = {qid for qid, q in this_by_id.items() if q.terminal_node != "review"}

    if prev_md is None:
        # First-ever digest: everything is "new" but we don't pretend
        # we know what changed. The newly_completed list still gets
        # populated so the prompt has something to point at.
        diff.newly_completed = sorted(
            (this_by_id[q] for q in this_completed),
            key=lambda x: x.quest_id,
        )
        diff.new_in_progress = sorted(
            (this_by_id[q] for q in this_in_progress),
            key=lambda x: x.quest_id,
        )
        return diff

    prev_all_ids = _quest_ids_in_digest(prev_md)
    prev_in_progress = _quest_ids_marked_in_progress(prev_md)
    prev_still_in_progress = _quest_ids_marked_still_in_progress(prev_md)

    # Promoted: was in_progress in prev, completed in this.
    promoted = sorted(this_completed & prev_in_progress)
    # Newly completed: completed in this AND wasn't in prev at all (or
    # was only mentioned outside the in_progress section).
    newly_completed = sorted(this_completed - set(promoted))
    # Still in progress: in_progress in both digests.
    still = sorted(this_in_progress & prev_in_progress)
    # New in progress: in this week's in_progress set, didn't appear
    # in prev at all.
    new_in_prog = sorted(this_in_progress - prev_all_ids)
    # Dropped: was in prev digest, hasn't been touched this window.
    dropped = sorted(prev_all_ids - set(this_by_id.keys()))
    # Stalled: still_in_progress AND was already in prev's
    # still-in-progress carry-over → ≥3 consecutive digests stuck.
    stalled = sorted(set(still) & prev_still_in_progress)

    diff.promoted = [this_by_id[q] for q in promoted]
    diff.newly_completed = [this_by_id[q] for q in newly_completed]
    diff.still_in_progress = [this_by_id[q] for q in still]
    diff.new_in_progress = [this_by_id[q] for q in new_in_prog]
    diff.dropped = dropped
    diff.stalled = stalled
    return diff


# ---------- digest-id naming ------------------------------------------------


def _digest_id(since: datetime, until: datetime) -> str:
    """File-stem for a digest. Uses ISO-week format (``YYYY-Www``) for
    the canonical 7-day window aligned on a week boundary, else the
    explicit ``YYYY-MM-DD-to-YYYY-MM-DD`` form."""
    days = (until - since).days
    # Standard rolling-7 window: use the ISO week of `until`.
    if days == 7:
        iso = until.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return f"{since.strftime('%Y-%m-%d')}-to-{until.strftime('%Y-%m-%d')}"


# ---------- prompt rendering ------------------------------------------------


def _format_quest_line(q: QuestSnapshot) -> str:
    """One bullet line for the manifest section of the prompt. Keeps
    each line short — the abstract goes in a separate section so the
    bullet table stays scannable."""
    title = q.title.strip() or _quest_title_from_id(q.quest_id)
    return f"- [{q.quest_id}] **{title}** (provider: {q.provider or 'unknown'})"


def _render_quest_table(snapshots: list[QuestSnapshot]) -> str:
    if not snapshots:
        return "(no quests in window)"
    completed = [q for q in snapshots if q.terminal_node == "review"]
    in_prog = [q for q in snapshots if q.terminal_node != "review"]
    parts: list[str] = []
    if completed:
        parts.append("### Completed")
        parts.extend(_format_quest_line(q) for q in completed)
    if in_prog:
        parts.append("\n### In progress")
        parts.extend(_format_quest_line(q) for q in in_prog)
    return "\n".join(parts)


def _render_abstracts(snapshots: list[QuestSnapshot]) -> str:
    blocks: list[str] = []
    for q in snapshots:
        if not q.paper_abstract:
            continue
        blocks.append(
            f"### [{q.quest_id}] {q.title}\n\n{q.paper_abstract}\n"
        )
    if not blocks:
        return "(no paper abstracts available — quests are still mid-run)"
    return "\n".join(blocks)


def _render_diff_section(diff: WeekDiff) -> str:
    """Pre-render the structured-diff section the LLM will copy into
    its output. The LLM gets a fully-formed bullet list with quest IDs
    + titles + emoji markers — it only has to add the 1-line synthesis.
    """
    if diff.prev_digest_id is None:
        return "(no prior digest — this is the first run)"
    lines = [f"_Comparing against `{diff.prev_digest_id}`._\n"]
    if diff.promoted:
        lines.append("**✅ Promoted to complete this week:**")
        lines.extend(f"- [{q.quest_id}] {q.title}" for q in diff.promoted)
    if diff.new_in_progress:
        lines.append("\n**🆕 Newly started:**")
        lines.extend(f"- [{q.quest_id}] {q.title}" for q in diff.new_in_progress)
    if diff.still_in_progress:
        lines.append("\n**⚠️ Still in progress from last digest:**")
        for q in diff.still_in_progress:
            marker = " 🛑 STALLED (3+ digests)" if q.quest_id in diff.stalled else ""
            lines.append(f"- [{q.quest_id}] {q.title}{marker}")
    if diff.dropped:
        lines.append("\n**❓ Quests in last digest but not touched this window:**")
        lines.extend(f"- [{qid}]" for qid in diff.dropped)
    if (not diff.promoted and not diff.new_in_progress
            and not diff.still_in_progress and not diff.dropped):
        lines.append("(nothing changed since the prior digest)")
    return "\n".join(lines)


def _render_velocity(
    snapshots: list[QuestSnapshot], diff: WeekDiff,
) -> str:
    """Numbers-only velocity block. The LLM will quote these directly."""
    completed_this = sum(1 for q in snapshots if q.terminal_node == "review")
    completed_prev_carry = len(diff.promoted)
    return (
        f"- Quests touched this window: {len(snapshots)}\n"
        f"- Completed this window: {completed_this}\n"
        f"- Of those, carried over from a prior digest: {completed_prev_carry}\n"
        f"- Newly started this window: {len(diff.new_in_progress)}\n"
        f"- Still in progress from prior digest: {len(diff.still_in_progress)}"
        + (f" (of which {len(diff.stalled)} are stalled 3+ digests)"
           if diff.stalled else "")
    )


def _build_digest_prompt(
    *,
    snapshots: list[QuestSnapshot],
    diff: WeekDiff,
    prev_digest_md: str | None,
    since: datetime,
    until: datetime,
) -> str:
    """Substitute the digest prompt template with rendered sections.
    The template (``agents/digest.md``) defines the LLM contract; this
    function is purely mechanical."""
    truncated_for_prompt = 0
    prompt_snapshots = snapshots
    if len(snapshots) > _MAX_PROMPT_QUESTS:
        prompt_snapshots = snapshots[:_MAX_PROMPT_QUESTS]
        truncated_for_prompt = len(snapshots) - _MAX_PROMPT_QUESTS

    quest_table = _render_quest_table(prompt_snapshots)
    if truncated_for_prompt > 0:
        quest_table += (
            f"\n\n_(Note: {truncated_for_prompt} additional quests were "
            f"omitted from this prompt to fit the budget — investigate "
            f"why so many were touched in one window.)_"
        )
    abstracts = _render_abstracts(prompt_snapshots)
    diff_section = _render_diff_section(diff)
    velocity = _render_velocity(prompt_snapshots, diff)
    prev_block = (
        f"## Prior digest ({diff.prev_digest_id}) for reference\n\n"
        f"```markdown\n{prev_digest_md[:8000]}\n```\n"
        if prev_digest_md else ""
    )

    template = string.Template(PROMPT_PATH.read_text(encoding="utf-8"))
    return template.substitute(
        window_from=since.date().isoformat(),
        window_to=until.date().isoformat(),
        quest_table=quest_table,
        abstracts=abstracts,
        diff_section=diff_section,
        velocity_block=velocity,
        prior_digest_block=prev_block,
    )


# ---------- Axon ingest -----------------------------------------------------


def _ingest_digest_to_axon(
    knowledge: Knowledge, *, digest_id: str, markdown: str,
    since: datetime, until: datetime, diff: WeekDiff,
) -> bool:
    """Send the produced digest into Axon as ``kind=fi_digest`` so
    future quests / portfolio runs can retrieve "what we were working
    on in window X." Best-effort — failures are logged but don't fail
    the overall command."""
    try:
        return bool(knowledge.add_text(
            text=markdown,
            kind="fi_digest",
            metadata={
                "digest_id": digest_id,
                "window_from": since.date().isoformat(),
                "window_to": until.date().isoformat(),
                "prev_digest_id": diff.prev_digest_id,
                "promoted_count": len(diff.promoted),
                "stalled_count": len(diff.stalled),
            },
        ))
    except Exception as e:
        _log.warning("digest Axon ingest failed: %r", e)
        return False


# ---------- top-level entry point -------------------------------------------


async def generate_digest(
    outputs_dir: Path,
    *,
    days: int = _DEFAULT_DAYS,
    since: datetime | None = None,
    until: datetime | None = None,
    provider: ProviderConfig,
    supervisor: ProxySupervisor | None = None,
    knowledge: Knowledge | None = None,
    now: datetime | None = None,
) -> DigestArtifacts:
    """Top-level entry. Collects snapshots, computes the diff, calls
    the LLM once, writes ``outputs/_digests/<digest_id>.md``, and
    (optionally) ingests into Axon.

    ``now`` is a test seam — pass a fixed datetime so deterministic
    tests don't depend on wall-clock time.
    """
    now = now or datetime.now(timezone.utc)
    if until is None:
        until = now
    if since is None:
        since = until - timedelta(days=days)
    if since > until:
        raise ValueError(f"--digest window inverted: since={since} > until={until}")

    outputs_dir = Path(outputs_dir).expanduser().resolve()
    snapshots = _collect_quest_snapshots(outputs_dir, since, until)

    digest_id = _digest_id(since, until)
    digests_dir = outputs_dir / "_digests"
    digests_dir.mkdir(parents=True, exist_ok=True)
    digest_path = digests_dir / f"{digest_id}.md"

    prior_path = _find_prior_digest(digests_dir, digest_id)
    prior_md: str | None = None
    if prior_path is not None:
        try:
            prior_md = prior_path.read_text(encoding="utf-8")
        except OSError:
            prior_md = None

    diff = _compute_diff(snapshots, prior_md, prior_path)

    _log.info(
        "digest %s: %d quests touched (%d completed, %d in-progress, "
        "%d promoted, %d still-in-progress, %d stalled, %d dropped)",
        digest_id, len(snapshots),
        sum(1 for q in snapshots if q.terminal_node == "review"),
        sum(1 for q in snapshots if q.terminal_node != "review"),
        len(diff.promoted), len(diff.still_in_progress),
        len(diff.stalled), len(diff.dropped),
    )

    # Empty-window short-circuit: no LLM call, write a brief marker so
    # downstream tooling can still find the digest file.
    if not snapshots:
        marker = (
            f"# Digest {digest_id}\n\n"
            f"Window: {since.date()} → {until.date()}\n\n"
            f"_No quests were touched in this window._\n"
            + (f"\nPrior digest: `{diff.prev_digest_id}`\n"
               if diff.prev_digest_id else "")
        )
        digest_path.write_text(marker, encoding="utf-8")
        return DigestArtifacts(
            digest_id=digest_id, digest_path=digest_path,
            quest_count=0, completed_count=0, in_progress_count=0,
            diff=diff, ingested_to_axon=False,
            raw_state={"empty_window": True},
        )

    prompt = _build_digest_prompt(
        snapshots=snapshots, diff=diff,
        prev_digest_md=prior_md, since=since, until=until,
    )

    own_supervisor = supervisor is None
    sup = supervisor or ProxySupervisor()
    endpoint = await resolve_endpoint_async(provider, sup)
    client = LLMClient(endpoint)
    try:
        markdown = await client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            node="digest",
        )
    finally:
        await client.aclose()
        if provider.name in PROXY_PROVIDERS:
            await sup.release(provider.name)
        if own_supervisor:
            await sup.shutdown()

    digest_path.write_text(markdown.strip() + "\n", encoding="utf-8")

    ingested = False
    if knowledge is not None and knowledge.enabled:
        ingested = _ingest_digest_to_axon(
            knowledge, digest_id=digest_id, markdown=markdown,
            since=since, until=until, diff=diff,
        )

    completed_count = sum(1 for q in snapshots if q.terminal_node == "review")
    return DigestArtifacts(
        digest_id=digest_id,
        digest_path=digest_path,
        quest_count=len(snapshots),
        completed_count=completed_count,
        in_progress_count=len(snapshots) - completed_count,
        diff=diff,
        ingested_to_axon=ingested,
        raw_state={
            "window_from": since.isoformat(),
            "window_to": until.isoformat(),
            "prev_digest_id": diff.prev_digest_id,
        },
    )
