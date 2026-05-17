"""Pluggable dataset adapters for ``_node_auto_collect_data``.

Each adapter implements a small protocol — ``search(query, top_k)
-> list[DatasetRow]`` — and gets registered by name in
``ADAPTER_REGISTRY``. The engine reads ``EngineConfig.dataset_adapters``
(a list of adapter names) and invokes each one in addition to the
existing Axon retrieval, writing all hits into the same
``<quest_root>/data/auto_collected/`` directory so the downstream
``wait_for_data`` / ``data_load`` nodes pick them up uniformly.

Adapters are intentionally OPT-IN (default empty list) because each
one hits an external API and we don't want a default-enabled
WorldBank lookup firing on every no-simulation quest the user runs.

Not currently shipped:
  - Hofstede + Pew. Neither has a free programmatic API; both would
    require bundling licensed CSVs and a separate ingest path.
  - OECD + Eurostat. Both use SDMX-JSON which has a steeper learning
    curve than the WorldBank REST API.
"""
from __future__ import annotations

from .base import DatasetAdapter, DatasetRow
from .wikipedia import WikipediaAdapter
from .worldbank import WorldBankAdapter

ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "worldbank": WorldBankAdapter,
    "wikipedia": WikipediaAdapter,
}

__all__ = [
    "DatasetAdapter",
    "DatasetRow",
    "ADAPTER_REGISTRY",
    "WorldBankAdapter",
    "WikipediaAdapter",
]
