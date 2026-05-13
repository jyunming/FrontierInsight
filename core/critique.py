"""Adversarial second-pass review of a completed quest.

The in-quest review (the ``review`` LangGraph node) is biased: the
same model wrote the paper AND reviewed it. ``/critique`` runs a
**fresh** adversarial pass against a finished quest — ideally with a
different provider so a different model family looks at the work cold
— and writes ``outputs/<quest_id>/critique.md`` next to ``paper.md``.

CLI:    ``python launch.py --critique <quest_id> [--critique-provider <name>]``
VSCode: ``@fi /critique <quest_id>``

Design choices for v1
=====================

* **One LLM call**, not a multi-persona panel. The existing engine's
  ``_run_review_panel`` requires a ``QuestState`` shape and is tightly
  coupled to the LangGraph loop; refactoring it for standalone use is
  out of scope for this PR. Instead the prompt bundles four critique
  angles (methodology, reproducibility, statistics, alternative
  explanations) into one structured output. A future v2 can plug in
  the panel infra once it's refactored.
* **Reuses** the digest/portfolio quest-resolution helpers
  (``_parse_quest_id_timestamp``) but otherwise stands alone — the
  critique is per-quest, not portfolio-wide.
* **Different-provider hint** is informational. We honor a
  ``--critique-provider`` flag so the caller can route through a
  different model family from the one that wrote the paper. The
  prompt also tells the model "you have never seen this paper before"
  as a soft prior — same-provider critiques can still surface issues
  because the temperature + the explicit adversarial framing differ
  from the in-quest review pass.
"""

from __future__ import annotations

import logging
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ProviderConfig
from .digest import _parse_quest_id_timestamp
from .knowledge import Knowledge
from .provider import (
    LLMClient,
    PROXY_PROVIDERS,
    ProxySupervisor,
    resolve_endpoint_async,
)

_log = logging.getLogger("frontier_insight.critique")

PROMPT_PATH = Path(__file__).resolve().parent.parent / "agents" / "critique.md"

# Per-artifact caps fed into the prompt. The paper is usually 4-8 KB
# of markdown; the code can be much larger. Caps prevent a 30 KB
# experiment.py from blowing the prompt budget.
_PAPER_CHARS = 12_000
_CODE_CHARS = 6_000
_PRIOR_REVIEW_CHARS = 3_000


@dataclass
class QuestArtifactsForCritique:
    """The chunks of a finished quest the critique pass needs to see.
    All fields are best-effort: missing files become empty strings so
    we can still critique partial quests (e.g. a quest where the
    paper exists but the in-quest review never ran)."""

    quest_id: str
    quest_root: Path
    paper_md: str
    paper_md_path: Path | None
    code: str
    code_path: Path | None
    prior_review: str
    prior_review_path: Path | None
    quest_provider: str | None


@dataclass
class CritiqueArtifacts:
    """What :func:`generate_critique` returns."""

    quest_id: str
    critique_path: Path
    critique_provider: str
    ingested_to_axon: bool
    raw_state: dict[str, Any] = field(default_factory=dict)


# ---------- quest resolution + artifact loading ----------------------------


def _resolve_quest_dir(outputs_dir: Path, quest_id: str) -> Path:
    """Resolve a quest_id to its on-disk directory. Validates the
    quest_id shape to refuse path-traversal attempts (e.g.
    ``../../etc``) — the caller has already filtered to the outputs
    root but defense in depth is cheap."""
    if _parse_quest_id_timestamp(quest_id) is None:
        raise ValueError(
            f"invalid quest_id (must match <epoch>-<slug>-<6hex>): {quest_id!r}",
        )
    quest_dir = (outputs_dir / quest_id).resolve()
    if not quest_dir.is_dir():
        raise FileNotFoundError(
            f"quest directory not found: {quest_dir}. "
            f"Run `ls {outputs_dir}` to see available quest_ids.",
        )
    # Containment check: even with the regex validation above, a
    # cleverly-crafted quest_id passing both could in theory be a
    # symlink. Resolved-path containment cuts that off.
    if outputs_dir.resolve() not in quest_dir.parents:
        raise ValueError(
            f"quest_id escapes outputs_dir: resolved={quest_dir}",
        )
    return quest_dir


def _read_capped(path: Path, *, limit: int) -> str:
    """Read a text file, capped at ``limit`` chars. Truncation is
    silent — the prompt template embeds these in code-fenced blocks
    where the model can spot truncation by missing closing braces.
    Doesn't raise on missing/unreadable files; returns ``""``."""
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        _log.warning("could not read %s: %r", path, e)
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n…[truncated, original {len(text)} chars]…\n"


def _load_quest_artifacts(quest_dir: Path) -> QuestArtifactsForCritique:
    """Locate the paper.md, code, and prior review under
    ``quest_dir``. We probe the two known layouts (post- and
    pre-Phase-O) so the same critique flow works against historical
    quest dirs and freshly-run ones."""
    quest_id = quest_dir.name

    # Paper: post-Phase-O writes paper/paper.md; older flows wrote
    # the markdown directly at quest_root/paper.md.
    paper_candidates = [
        quest_dir / "paper" / "paper.md",
        quest_dir / "paper.md",
    ]
    paper_path = next((p for p in paper_candidates if p.is_file()), None)
    paper = _read_capped(paper_path, limit=_PAPER_CHARS) if paper_path else ""

    # Code: experiment.py is the canonical name; some older quests
    # used main.py or run.py. Pick the first non-empty match.
    code_candidates = [
        quest_dir / "code" / "experiment.py",
        quest_dir / "code" / "main.py",
        quest_dir / "code" / "run.py",
        quest_dir / "experiment.py",
    ]
    code_path = next((p for p in code_candidates if p.is_file()), None)
    code = _read_capped(code_path, limit=_CODE_CHARS) if code_path else ""

    # Prior review: per-quest review.md (single-reviewer flow) OR the
    # moderator's synthesis from a reviewer-panel run.
    review_candidates = [
        quest_dir / "paper" / "review.md",
        quest_dir / "paper" / "review_moderate.md",
        quest_dir / "review.md",
    ]
    prior_review_path = next(
        (p for p in review_candidates if p.is_file()), None,
    )
    prior_review = (
        _read_capped(prior_review_path, limit=_PRIOR_REVIEW_CHARS)
        if prior_review_path else ""
    )

    # Provider hint: read from frontier_insight_summary.json if
    # present. Used by the prompt to ask the LLM to flag any
    # provider-specific failure mode it knows about.
    provider: str | None = None
    summary_path = quest_dir / "frontier_insight_summary.json"
    if summary_path.is_file():
        try:
            import json
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            provider = summary.get("provider")
        except (OSError, json.JSONDecodeError):
            pass

    return QuestArtifactsForCritique(
        quest_id=quest_id,
        quest_root=quest_dir,
        paper_md=paper, paper_md_path=paper_path,
        code=code, code_path=code_path,
        prior_review=prior_review, prior_review_path=prior_review_path,
        quest_provider=provider,
    )


# ---------- prompt rendering ------------------------------------------------


def _render_artifact_block(label: str, body: str, path: Path | None) -> str:
    """One labeled section for an artifact. Includes the source path
    (so reviewer can cross-reference) and a placeholder marker when
    the artifact wasn't found — gives the LLM a clear signal to skip
    the corresponding critique angle rather than hallucinate."""
    if not body:
        return f"### {label}\n\n_(not found in quest directory)_\n"
    path_suffix = f"  \n_source: `{path}`_" if path else ""
    return f"### {label}{path_suffix}\n\n```\n{body}\n```\n"


def _build_critique_prompt(
    art: QuestArtifactsForCritique,
    *,
    critique_provider: str,
    generated_at: datetime,
) -> str:
    paper_block = _render_artifact_block(
        "Paper (paper.md)", art.paper_md, art.paper_md_path,
    )
    code_block = _render_artifact_block(
        "Experiment code", art.code, art.code_path,
    )
    review_block = _render_artifact_block(
        "Prior in-quest review (for context — your job is to find what they missed)",
        art.prior_review, art.prior_review_path,
    )

    template = string.Template(PROMPT_PATH.read_text(encoding="utf-8"))
    return template.substitute(
        quest_id=art.quest_id,
        quest_provider=art.quest_provider or "unknown",
        critique_provider=critique_provider,
        generated_at=generated_at.date().isoformat(),
        paper_block=paper_block,
        code_block=code_block,
        prior_review_block=review_block,
    )


# ---------- Axon ingest -----------------------------------------------------


def _ingest_critique_to_axon(
    knowledge: Knowledge, *, quest_id: str, markdown: str,
    critique_provider: str, quest_provider: str | None,
) -> bool:
    try:
        return bool(knowledge.add_text(
            text=markdown,
            kind="fi_critique",
            metadata={
                "quest_id": quest_id,
                "critique_provider": critique_provider,
                "quest_provider": quest_provider,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        ))
    except Exception as e:
        _log.warning("critique Axon ingest failed: %r", e)
        return False


# ---------- top-level -------------------------------------------------------


async def generate_critique(
    quest_id: str,
    outputs_dir: Path,
    *,
    provider: ProviderConfig,
    supervisor: ProxySupervisor | None = None,
    knowledge: Knowledge | None = None,
    now: datetime | None = None,
) -> CritiqueArtifacts:
    """Produce an adversarial critique of a previously-run quest.
    Writes ``outputs/<quest_id>/critique.md`` and (optionally) ingests
    it into Axon as ``kind=fi_critique``.

    Raises ``ValueError`` for malformed quest_ids, ``FileNotFoundError``
    when the quest_id doesn't resolve to a real directory, and lets
    LLM/network errors bubble up so the caller can decide how to
    report them.
    """
    now = now or datetime.now(timezone.utc)
    outputs_dir = Path(outputs_dir).expanduser().resolve()
    quest_dir = _resolve_quest_dir(outputs_dir, quest_id)

    art = _load_quest_artifacts(quest_dir)
    if not art.paper_md:
        raise FileNotFoundError(
            f"no paper.md found under {quest_dir}; nothing to critique. "
            f"Make sure the quest has completed at least the `write` node.",
        )

    prompt = _build_critique_prompt(
        art, critique_provider=provider.name, generated_at=now,
    )

    own_supervisor = supervisor is None
    sup = supervisor or ProxySupervisor()
    endpoint = await resolve_endpoint_async(provider, sup)
    client = LLMClient(endpoint)
    try:
        markdown = await client.chat(
            [{"role": "user", "content": prompt}],
            # Slightly higher than digest/portfolio — adversarial
            # critique benefits from a wider exploration of failure
            # modes than 0.2-temperature consensus-seeking.
            temperature=0.4,
            node="critique",
        )
    finally:
        await client.aclose()
        if provider.name in PROXY_PROVIDERS:
            await sup.release(provider.name)
        if own_supervisor:
            await sup.shutdown()

    critique_path = quest_dir / "critique.md"
    critique_path.write_text(markdown.strip() + "\n", encoding="utf-8")

    ingested = False
    if knowledge is not None and knowledge.enabled:
        ingested = _ingest_critique_to_axon(
            knowledge, quest_id=art.quest_id, markdown=markdown,
            critique_provider=provider.name,
            quest_provider=art.quest_provider,
        )

    _log.info(
        "critique %s: written to %s; ingested=%s; "
        "(critique_provider=%s vs quest_provider=%s)",
        art.quest_id, critique_path, ingested,
        provider.name, art.quest_provider,
    )

    return CritiqueArtifacts(
        quest_id=art.quest_id,
        critique_path=critique_path,
        critique_provider=provider.name,
        ingested_to_axon=ingested,
        raw_state={
            "quest_provider": art.quest_provider,
            "had_prior_review": bool(art.prior_review),
            "had_code": bool(art.code),
        },
    )
