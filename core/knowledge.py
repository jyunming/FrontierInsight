"""Research knowledge layer over jyunming/Axon.

Single in-process `AxonBrain` per FI run. Used by the literature node to
retrieve papers and by the post-quest write-back to persist new findings
for cross-quest memory. All routing decisions (hybrid search, reranking,
HyDE, graphRAG) are owned by Axon.

If `axon` isn't installed, the wrapper degrades gracefully: `enabled` is
False, retrievals return empty, and the engine logs a one-line warning.
This keeps Phase A's import-and-skeleton path working on machines that
haven't set up Axon yet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import KnowledgeConfig

_log = logging.getLogger("frontier_insight.knowledge")

# Axon is an optional dependency at import time so FI's skeleton runs
# without it. The flag below tracks availability for the constructor.
try:
    from axon import AxonBrain, AxonConfig  # type: ignore[import-not-found]
    from axon.integrations.langchain import AxonRetriever  # type: ignore[import-not-found]

    _AXON_AVAILABLE = True
except Exception as e:  # ImportError or any init-time failure inside axon
    _AXON_AVAILABLE = False
    _AXON_IMPORT_ERROR: Exception | None = e
    AxonBrain = None  # type: ignore[assignment]
    AxonConfig = None  # type: ignore[assignment]
    AxonRetriever = None  # type: ignore[assignment]


@dataclass
class RetrievedDoc:
    """Loose mirror of LangChain `Document`, decoupled so callers don't
    need to import langchain just to consume retrieval output."""

    content: str
    metadata: dict[str, Any]


class Knowledge:
    def __init__(self, cfg: KnowledgeConfig) -> None:
        self.cfg = cfg
        self.enabled = cfg.enabled and _AXON_AVAILABLE
        self._brain: Any | None = None
        self._retriever: Any | None = None
        if cfg.enabled and not _AXON_AVAILABLE:
            err = globals().get("_AXON_IMPORT_ERROR")
            _log.warning(
                "axon not importable (%s); knowledge layer disabled. "
                "Install with `pip install axon` or via the repo to enable.",
                err,
            )
        if self.enabled:
            self._brain = self._build_brain(cfg)
            self._retriever = AxonRetriever(brain=self._brain, top_k=cfg.top_k)

    @staticmethod
    def _build_brain(cfg: KnowledgeConfig) -> Any:
        if cfg.axon_config is None:
            return AxonBrain(AxonConfig())  # type: ignore[misc]
        if isinstance(cfg.axon_config, Path):
            return AxonBrain(AxonConfig.from_yaml(cfg.axon_config))  # type: ignore[misc]
        if isinstance(cfg.axon_config, dict):
            ac = AxonConfig.model_validate(cfg.axon_config)  # type: ignore[union-attr]
            return AxonBrain(ac)  # type: ignore[misc]
        # Fallback: treat unknown type as YAML-stringifiable.
        ac = AxonConfig.model_validate(yaml.safe_load(yaml.safe_dump(cfg.axon_config)))  # type: ignore[union-attr]
        return AxonBrain(ac)  # type: ignore[misc]

    def search(self, query: str, *, top_k: int | None = None) -> list[RetrievedDoc]:
        if not self.enabled or self._retriever is None:
            return []
        retriever = self._retriever
        if top_k is not None and top_k != self.cfg.top_k:
            retriever = retriever.with_overrides({"top_k": top_k})
        try:
            docs = retriever.invoke(query)
        except Exception as e:
            _log.warning("axon retrieval failed for query=%r: %s", query[:60], e)
            return []
        return [
            RetrievedDoc(
                content=getattr(d, "page_content", str(d)),
                metadata=getattr(d, "metadata", {}) or {},
            )
            for d in docs
        ]

    def ingest(self, source: Path | str) -> bool:
        """Ingest a file or URL. Returns True on success."""
        if not self.enabled or self._brain is None:
            return False
        try:
            # Axon's CLI/API expose `add_document` / `ingest`; try both.
            ingest_fn = getattr(self._brain, "ingest", None) or getattr(
                self._brain, "add_document", None
            )
            if ingest_fn is None:
                _log.warning("axon brain exposes no ingest/add_document; skipping")
                return False
            ingest_fn(str(source))
            return True
        except Exception as e:
            _log.warning("axon ingest failed for %s: %s", source, e)
            return False

    def add_quest_artifacts(
        self,
        *,
        quest_id: str,
        paper_md_path: Path,
        summary: str,
    ) -> bool:
        """Write a finished quest into Axon for cross-quest memory.

        Tagged `fi-quest:<quest_id>`; the ideate node's future retrievals
        will surface past quests alongside literature.
        """
        if not self.enabled or self._brain is None:
            return False
        if not self.cfg.write_back_quests:
            return False
        try:
            # Try the most idiomatic path; tolerate missing API surface.
            add_text = getattr(self._brain, "add_text", None) or getattr(
                self._brain, "ingest_text", None
            )
            metadata = {"tag": f"fi-quest:{quest_id}", "kind": "fi_quest"}
            if add_text is not None:
                paper_md = (
                    paper_md_path.read_text(encoding="utf-8")
                    if paper_md_path.exists()
                    else ""
                )
                add_text(paper_md, metadata=metadata)
                if summary:
                    add_text(summary, metadata={**metadata, "kind": "fi_quest_summary"})
                return True
            # Fallback: ingest the paper file by path.
            return self.ingest(paper_md_path)
        except Exception as e:
            _log.warning("axon write-back failed for quest %s: %s", quest_id, e)
            return False
