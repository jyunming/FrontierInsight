"""``fi --analyze <data_path>`` — start a no-simulation quest with
pre-staged data.

The inverse of ``fi --proposal``: rather than asking the engine to
plan an experiment, the user already has data and wants FI to write
a paper analyzing it. Workflow:

1. User runs ``fi --analyze ./my_data --analyze-topic "Compare ..."``.
2. We mint a new quest_id, create ``<output_root>/<quest_id>/data/``,
   and copy every file from ``my_data/`` (recursive) into it.
3. We build a minimal :class:`Config` with
   ``engine.no_simulation: true`` and ``engine.auto_collect_data:
   false`` — the engine then routes through ``auto_collect_data``
   (which passes through because auto_collect is off) → ``wait_for_data``
   (which passes through because ``data/`` already has files) →
   ``data_load`` → ``analyze → cross_check → write → review``.

The user can still feed ``--config <yaml>`` to override defaults
(provider, reviewer panel, output kinds) — those merge over the
minimal config built here. When ``--config`` is absent we ship a
sensible default tuned for the analyze workflow.

Single LLM cost: ~6 premium requests for a bare analyze run
(no ideate / no literature / no design / no implement / no execute /
no execute_reflect). Closely matches the no-simulation cost row in
``vscode-frontier-insight/README.md``.
"""
from __future__ import annotations

import shutil
import time
import secrets
import string
from pathlib import Path
from typing import Optional

from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)

# Files we skip when copying the user's data dir — they're noise that
# would just inflate ``data_load``'s prompt and likely OOM it. Mirrors
# the summarizer's skip list.
_SKIP_NAMES = frozenset({
    ".DS_Store", "Thumbs.db", "desktop.ini", ".git", ".gitignore",
    "__pycache__", ".venv", "node_modules", ".pytest_cache",
})
_SKIP_SUFFIXES = (".pyc", ".pyo", ".log", ".sqlite", ".sqlite-journal")


def _slug(text: str, *, max_chars: int = 40) -> str:
    """Topic → URL-safe slug for the quest_id middle segment.

    ASCII-only ``[a-z0-9-]+`` — matches the canonical
    ``<epoch>-[a-z0-9-]+-<6hex>`` shape that the digest /
    critique / portfolio tools parse. ``str.isalnum`` would
    otherwise admit non-ASCII letters (CJK, Cyrillic, accented
    Latin), producing a quest_id that those tools would skip or
    reject."""
    out: list[str] = []
    last_dash = False
    for ch in text.strip().lower():
        # ``a-z`` and ``0-9`` only; anything else collapses to a
        # single dash. Matches ``core.engine._slugify``'s contract.
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            out.append(ch)
            last_dash = False
        elif not last_dash and out:
            out.append("-")
            last_dash = True
        if len("".join(out).rstrip("-")) >= max_chars:
            break
    return "".join(out).strip("-") or "analyze"


def _nonce(n: int = 6) -> str:
    """Short random hex suffix matching the rest of the codebase."""
    alphabet = string.hexdigits.lower()
    return "".join(secrets.choice(alphabet) for _ in range(n))


def mint_quest_id(topic: str, *, now_epoch: Optional[int] = None) -> str:
    """Build a quest_id matching the project's convention
    ``<epoch>-<slug>-<nonce>``. Exposed so tests can pin the prefix."""
    epoch = now_epoch if now_epoch is not None else int(time.time())
    return f"{epoch}-{_slug(topic)}-{_nonce()}"


def stage_data(src: Path, dest: Path) -> int:
    """Copy every non-skip file from ``src`` into ``dest`` (recursive).
    Returns the count of files written. ``dest`` is created if absent.

    Symlinks aren't followed (security: a malicious symlink could
    point at ``/etc/passwd``). Hidden directories like ``.git`` are
    skipped wholesale.

    **Containment check:** when ``dest`` sits INSIDE ``src`` — e.g.
    a user running ``fi --analyze .`` with the default
    ``./outputs/<quest>/data`` destination — files under ``dest``
    are skipped during the walk so we don't (a) ingest prior quest
    outputs as new "data" or (b) recurse into our own staged copy
    while the walk is still running. Same check applies to the
    canonical ``outputs/`` parent if it sits inside ``src``."""
    src = src.resolve()
    if not src.exists() or not src.is_dir():
        raise ValueError(
            f"--analyze data path must be an existing directory; got {src!r}"
        )
    dest.mkdir(parents=True, exist_ok=True)
    dest_abs = dest.resolve()
    # Compute a list of "exclude these subtrees from the walk". The
    # primary one is dest itself when dest is inside src. We also
    # exclude any ``outputs`` directory sitting at the top level of
    # src — that's where FI's prior runs live and they'd otherwise
    # snowball into the staged data on a ``fi --analyze .`` run.
    exclude_roots: list[Path] = []
    try:
        dest_abs.relative_to(src)
        exclude_roots.append(dest_abs)
    except ValueError:
        # dest is OUTSIDE src — no containment issue.
        pass
    canonical_outputs = src / "outputs"
    if canonical_outputs.exists() and canonical_outputs.is_dir():
        exclude_roots.append(canonical_outputs.resolve())

    def _is_inside_excluded(entry: Path) -> bool:
        ep = entry.resolve()
        for root in exclude_roots:
            try:
                ep.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    written = 0
    for entry in src.rglob("*"):
        if entry.is_symlink() or not entry.is_file():
            continue
        if entry.name in _SKIP_NAMES:
            continue
        if entry.suffix.lower() in _SKIP_SUFFIXES:
            continue
        # Skip anything whose path component contains a skip-name dir.
        rel = entry.relative_to(src)
        if any(part in _SKIP_NAMES for part in rel.parts):
            continue
        # Containment guard — drop entries that resolve into ``dest``
        # itself OR a canonical ``outputs/`` subtree inside ``src``.
        if exclude_roots and _is_inside_excluded(entry):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry, target)
        written += 1
    return written


def build_analyze_config(
    *, topic: str, output_dir: Path, provider_name: str = "vscode_extension",
    title: Optional[str] = None,
) -> Config:
    """Build a minimal Config for the analyze quest.

    Defaults are tuned for the workflow:
    * ``no_simulation: true`` — skip implement/execute.
    * ``auto_collect_data: false`` — the user already supplied the
      data; no need to ask Axon to find more (the user can pass a
      ``--config`` override to flip this on).
    * ``review_loop: false`` — analyze workflows usually want one
      write+review pass, not a revise cycle.
    * ``clarify_mode: off`` — the topic is the source of truth here;
      the user has already decided what to analyze."""
    return Config(
        topic=topic,
        title=title or _slug(topic, max_chars=60),
        provider=ProviderConfig(name=provider_name),  # type: ignore[arg-type]
        engine=EngineConfig(
            framework="langgraph",
            max_iterations=1,
            review_loop=False,
            clarify_mode="off",
            no_simulation=True,
            auto_collect_data=False,
            ideate_reflect=False,
            cross_check_per_finding_k=0,
            enable_analyze_reroute=False,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=600),
        knowledge=KnowledgeConfig(enabled=False, source_routing="manual"),
        output=OutputConfig(
            kinds=["paper_md", "paper_pdf"],
            output_dir=output_dir,
        ),
    )


def prepare_analyze_quest(
    *, data_path: Path, topic: str, output_root: Path,
    provider_name: str = "vscode_extension",
    title: Optional[str] = None,
) -> tuple[Config, str, int]:
    """Top-level helper: mint a quest_id, stage data, build the
    Config. Returns ``(config, quest_id, files_staged)``.

    Caller (``launch.py``) wires the Config into ``Engine(cfg).run()``
    afterwards. Split this way so it's testable without spinning up
    the LLM."""
    quest_id = mint_quest_id(topic)
    quest_root = output_root.expanduser().resolve() / quest_id
    data_dir = quest_root / "data"
    files_staged = stage_data(data_path.expanduser().resolve(), data_dir)
    cfg = build_analyze_config(
        topic=topic, output_dir=output_root, provider_name=provider_name,
        title=title,
    )
    return cfg, quest_id, files_staged
