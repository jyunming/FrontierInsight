"""Cross-quest portfolio synthesis.

All-time view across every quest under ``outputs/`` (completed AND
in-progress — the prompt instructs the LLM to weight completed
quests more heavily but in-progress ones still surface ongoing
themes). Companion to ``core.digest``: where the digest is a weekly
snapshot, the portfolio is the unbounded "what have I built up over
months?" question. Surfaces topic clusters, near-duplicate quests
(the same question asked twice with slightly different wording),
themes that span multiple quests, and proposes meta-papers / gap-
filling quests.

CLI:    ``python launch.py --portfolio``
VSCode: ``@fi /portfolio``

v1 design decision: cluster + duplicate detection is delegated to the
LLM in one prompt rather than building an embedding-based graph in
code. Rationale:
* For the typical 10-50 completed quests a hobbyist accumulates, an
  LLM with the titles + abstracts + topics in front of it produces
  thematic clusters that are at least as good as anything we'd get
  out of TF-IDF cosine / sentence-transformers k-means.
* It avoids a hard dependency on an embeddings library at runtime
  (Axon is optional — `/portfolio` should still work on a barebones
  pip install).
* It keeps the v1 module small and reviewable.
A future v2 can add an Axon-similarity graph as a deterministic
pre-pass to seed the clustering, the way the digest's WeekDiff seeds
the digest prompt. That's deferred until the LLM-only approach
demonstrably misses connections.

Reuses ``core.digest`` helpers (snapshot collector, terminal-node
detection, quest_id parsing) — those were written generically to be
shared across PM-style commands.
"""

from __future__ import annotations

import logging
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ProviderConfig
from .digest import (
    _collect_quest_snapshots,
    QuestSnapshot,
)
from .knowledge import Knowledge
from .provider import (
    LLMClient,
    PROXY_PROVIDERS,
    ProxySupervisor,
    resolve_endpoint_async,
)

_log = logging.getLogger("frontier_insight.portfolio")

PROMPT_PATH = Path(__file__).resolve().parent.parent / "agents" / "portfolio.md"

# Hard cap on quests rendered into the LLM prompt. Bigger than the
# digest cap because portfolio is unbounded-in-time: a long-running
# user could easily accumulate hundreds of quests. The prompt-rendering
# code truncates older quests first and appends a note so the model
# knows the tail exists.
_MAX_PROMPT_QUESTS = 150

# Per-quest abstract cap. Smaller than digest's because we're showing
# more quests per prompt — total prompt size is the constraint, and a
# 400-char abstract is usually enough for the LLM to grasp the topic.
_ABSTRACT_CHARS = 400


@dataclass
class PortfolioArtifacts:
    """What :func:`generate_portfolio` returns. ``portfolio_path`` is
    the headline markdown; ``raw_state`` carries enough structured
    metadata (window timestamps, quest counts, and the ordered list
    of quest_ids that went into the prompt) for callers — e.g. the
    Phase-J web GUI or a future ``/portfolio --json`` mode — to
    render their own view without re-walking the outputs tree."""

    portfolio_id: str
    portfolio_path: Path
    completed_count: int
    in_progress_count: int
    ingested_to_axon: bool
    raw_state: dict[str, Any] = field(default_factory=dict)


def _portfolio_id(now: datetime) -> str:
    """File-stem for a portfolio snapshot. Just a date — portfolios
    aren't naturally weekly the way digests are, so we use ISO date.
    Running the command twice in one day overwrites cleanly."""
    return now.date().isoformat()


def _format_quest_block(q: QuestSnapshot) -> str:
    """One bullet block per quest in the prompt. Includes ID, title,
    topic (when present), and a short abstract excerpt — enough for
    the LLM to spot semantic overlap with other quests."""
    title = q.title.strip() or q.quest_id
    topic = q.topic.strip() if q.topic else ""
    abstract = (q.paper_abstract or "")[:_ABSTRACT_CHARS].strip()
    pieces = [f"- **[{q.quest_id}] {title}**"]
    if topic:
        # Trim multi-paragraph topic statements to one line so the
        # prompt stays scannable.
        first_line = topic.splitlines()[0][:200]
        pieces.append(f"  - topic: {first_line}")
    if q.provider:
        pieces.append(f"  - provider: {q.provider}")
    pieces.append(f"  - completed: {q.terminal_node == 'review'}")
    if abstract:
        pieces.append(f"  - abstract: {abstract}")
    return "\n".join(pieces)


def _render_quest_corpus(snapshots: list[QuestSnapshot]) -> tuple[str, int]:
    """Render the prompt's quest-corpus section. Returns
    ``(rendered, n_truncated)`` so the prompt can include an
    "(N additional quests omitted)" note when needed.

    Ordering: most-recent first (by ``last_modified``). The cap drops
    the oldest tail because recent quests are more likely to be the
    relevant context for "what should I do next."
    """
    by_recent = sorted(
        snapshots, key=lambda q: q.last_modified, reverse=True,
    )
    truncated = 0
    if len(by_recent) > _MAX_PROMPT_QUESTS:
        truncated = len(by_recent) - _MAX_PROMPT_QUESTS
        by_recent = by_recent[:_MAX_PROMPT_QUESTS]
    if not by_recent:
        return "(no quests yet)", 0
    blocks = [_format_quest_block(q) for q in by_recent]
    return "\n\n".join(blocks), truncated


def _stats_block(snapshots: list[QuestSnapshot]) -> str:
    """Deterministic numbers the LLM quotes in the velocity / overview
    section. Same pattern as the digest's velocity block — keeps the
    structured facts out of the LLM's hands."""
    total = len(snapshots)
    completed = sum(1 for q in snapshots if q.terminal_node == "review")
    in_prog = total - completed
    providers: dict[str, int] = {}
    for q in snapshots:
        key = q.provider or "unknown"
        providers[key] = providers.get(key, 0) + 1
    # Quest velocity: median days between consecutive ``created_at``
    # values, for completed quests only. Gives a rough cadence number
    # ("user shipped a quest every ~3 days" or "every ~3 weeks").
    completed_dates = sorted(
        q.created_at for q in snapshots if q.terminal_node == "review"
    )
    cadence = "n/a"
    if len(completed_dates) >= 2:
        gaps = [
            (completed_dates[i + 1] - completed_dates[i]).total_seconds() / 86400
            for i in range(len(completed_dates) - 1)
        ]
        gaps.sort()
        mid = gaps[len(gaps) // 2]
        cadence = f"~{mid:.1f} days between completions (median)"
    # Count-descending, then name-ascending. Without the secondary
    # key, two providers tied on count would reshuffle randomly across
    # runs depending on dict insertion order (which depends on
    # filesystem traversal order). The secondary sort by name keeps
    # the stats block stable run-to-run.
    provider_line = ", ".join(
        f"{name}: {n}" for name, n in sorted(
            providers.items(), key=lambda kv: (-kv[1], kv[0]),
        )
    ) or "n/a"
    earliest = (
        min((q.created_at for q in snapshots), default=None)
    )
    latest = (
        max((q.last_modified for q in snapshots), default=None)
    )
    span = "n/a"
    if earliest and latest:
        span = (
            f"{earliest.date()} → {latest.date()} "
            f"({(latest - earliest).days} days)"
        )
    return (
        f"- Total quests on disk: {total}\n"
        f"- Completed: {completed}\n"
        f"- In progress / unknown: {in_prog}\n"
        f"- Time span: {span}\n"
        f"- Completion cadence: {cadence}\n"
        f"- Providers used: {provider_line}"
    )


def _build_portfolio_prompt(
    snapshots: list[QuestSnapshot], *, generated_at: datetime,
) -> str:
    corpus, n_truncated = _render_quest_corpus(snapshots)
    trunc_note = (
        f"\n\n_(Note: {n_truncated} older quest(s) past the prompt cap "
        f"were omitted from this corpus. Recent activity dominates "
        f"the clustering signal.)_"
        if n_truncated > 0 else ""
    )
    stats = _stats_block(snapshots)

    template = string.Template(PROMPT_PATH.read_text(encoding="utf-8"))
    return template.substitute(
        generated_at=generated_at.date().isoformat(),
        stats_block=stats,
        quest_corpus=corpus + trunc_note,
    )


# ---- Axon ingest -----------------------------------------------------------


def _ingest_portfolio_to_axon(
    knowledge: Knowledge, *, portfolio_id: str, markdown: str,
    snapshot_count: int,
) -> bool:
    """Best-effort Axon ingest of the rendered portfolio. ``kind`` is
    ``fi_portfolio`` so it's distinguishable from quest papers and
    digests. Failures log + return False — never raise into the
    caller."""
    try:
        return bool(knowledge.add_text(
            text=markdown,
            kind="fi_portfolio",
            metadata={
                "portfolio_id": portfolio_id,
                "quest_count": snapshot_count,
            },
        ))
    except Exception as e:
        _log.warning("portfolio Axon ingest failed: %r", e)
        return False


# ---- top-level -------------------------------------------------------------


async def generate_portfolio(
    outputs_dir: Path,
    *,
    provider: ProviderConfig,
    supervisor: ProxySupervisor | None = None,
    knowledge: Knowledge | None = None,
    now: datetime | None = None,
) -> PortfolioArtifacts:
    """Generate a portfolio snapshot across every quest under
    ``outputs_dir``. Writes
    ``outputs/_portfolio/<YYYY-MM-DD>.md``; ingests to Axon as
    ``fi_portfolio`` when knowledge is enabled.

    Unlike :func:`core.digest.generate_digest`, there's no time
    window — we scan the entire outputs/ tree. ``now`` is a test seam
    for the portfolio_id stamp.
    """
    now = now or datetime.now(timezone.utc)
    outputs_dir = Path(outputs_dir).expanduser().resolve()

    # Use a very wide window to grab everything. The digest collector
    # filters by ``since <= last_modified <= until``; setting since to
    # the unix epoch and until to a year past ``now`` ensures we don't
    # accidentally drop quests with weird mtime data (clock-skewed
    # source filesystem, files touched by `git checkout` after a
    # large clock change, etc.).
    from datetime import timedelta
    epoch_start = datetime.fromtimestamp(0, tz=timezone.utc)
    until = now + timedelta(days=365)
    snapshots = _collect_quest_snapshots(outputs_dir, epoch_start, until)

    portfolio_id = _portfolio_id(now)
    portfolio_dir = outputs_dir / "_portfolio"
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    portfolio_path = portfolio_dir / f"{portfolio_id}.md"

    completed = sum(1 for q in snapshots if q.terminal_node == "review")
    in_prog = len(snapshots) - completed
    _log.info(
        "portfolio %s: %d quests on disk (%d completed, %d in-progress)",
        portfolio_id, len(snapshots), completed, in_prog,
    )

    if not snapshots:
        marker = (
            f"# Portfolio — {portfolio_id}\n\n"
            f"_No quests on disk under `{outputs_dir}`. Run "
            f"`@fi /new` or `@fi /start <config.yaml>` to populate "
            f"the portfolio._\n"
        )
        portfolio_path.write_text(marker, encoding="utf-8")
        return PortfolioArtifacts(
            portfolio_id=portfolio_id, portfolio_path=portfolio_path,
            completed_count=0, in_progress_count=0,
            ingested_to_axon=False, raw_state={"empty": True},
        )

    prompt = _build_portfolio_prompt(snapshots, generated_at=now)

    own_supervisor = supervisor is None
    sup = supervisor or ProxySupervisor()
    endpoint = await resolve_endpoint_async(provider, sup)
    client = LLMClient(endpoint)
    try:
        markdown = await client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            node="portfolio",
        )
    finally:
        await client.aclose()
        if provider.name in PROXY_PROVIDERS:
            await sup.release(provider.name)
        if own_supervisor:
            await sup.shutdown()

    portfolio_path.write_text(markdown.strip() + "\n", encoding="utf-8")

    ingested = False
    if knowledge is not None and knowledge.enabled:
        ingested = _ingest_portfolio_to_axon(
            knowledge, portfolio_id=portfolio_id, markdown=markdown,
            snapshot_count=len(snapshots),
        )

    return PortfolioArtifacts(
        portfolio_id=portfolio_id,
        portfolio_path=portfolio_path,
        completed_count=completed,
        in_progress_count=in_prog,
        ingested_to_axon=ingested,
        raw_state={
            "generated_at": now.isoformat(),
            "quest_count": len(snapshots),
            # Most-recent-first order, mirroring what
            # _render_quest_corpus puts into the prompt. Callers can
            # use this to re-resolve quest dirs without re-walking
            # outputs/. Kept as plain strings (not Snapshot objects)
            # so raw_state stays JSON-serializable.
            "quest_ids": [
                q.quest_id for q in sorted(
                    snapshots, key=lambda x: x.last_modified, reverse=True,
                )
            ],
        },
    )
