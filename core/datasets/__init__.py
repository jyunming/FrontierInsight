"""Phase D2 — pluggable dataset adapters for ``_node_auto_collect_data``.

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

Out of scope for D2:
  - Hofstede + Pew. Neither has a free programmatic API; both would
    require bundling licensed CSVs and a separate ingest path. Push
    to a follow-up PR.
  - OECD + Eurostat. Both use SDMX-JSON which has a steeper learning
    curve than the WorldBank REST API. Once the abstraction is
    validated by WorldBank in production, add these as δ.2 / δ.3.
"""
from __future__ import annotations

from .base import DatasetAdapter, DatasetRow
from .worldbank import WorldBankAdapter

ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {
    "worldbank": WorldBankAdapter,
}

__all__ = ["DatasetAdapter", "DatasetRow", "ADAPTER_REGISTRY", "WorldBankAdapter"]
