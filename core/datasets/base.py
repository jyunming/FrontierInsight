"""Abstract base for the auto-collect dataset adapters.

A ``DatasetAdapter`` is a thin protocol over a structured-data API
(World Bank, OECD SDMX, Eurostat, Pew CSV, etc.). The engine asks
each enabled adapter for relevant rows on a topic; the adapter
translates the topic into its native query language, fetches, and
returns a list of ``DatasetRow``. The engine then renders each row
as a Markdown file under ``<quest_root>/data/auto_collected/`` so
the existing ``wait_for_data`` / ``data_load`` pipeline picks it up
without changes.

Why a thin abstraction and not a richer one (filters by year /
geography / metric ID): every adapter does these differently, and
the engine doesn't need to push structured filters down. The
LLM-generated query string is the only universal handle. Each
adapter is free to use heuristics on the query (year-mention
detection, country-code lookup) without imposing those concerns on
the contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetRow:
    """One result from a dataset adapter.

    ``content`` is the rendered body that lands in the
    ``auto_collected/<rank>_<slug>.md`` file — a short prose summary
    plus the raw data points (a CSV-style table works well, the LLM
    parses it fine). ``metadata`` is a dict written into the YAML
    front matter; standard keys are ``source`` (adapter name),
    ``url`` (canonical URL for the data point), ``title`` (one-line
    description for the manifest), and optionally ``year``,
    ``country``, ``indicator`` if the adapter knows them.

    Why a dataclass and not just a tuple: the field names show up
    in adapter code and tests; positional ordering would invite
    swap-args bugs in adapters with 5+ metadata fields.
    """
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Adapter contract. Implementations get one chance per quest to
    return up to ``top_k`` rows for the given query string. The
    engine is responsible for slug-naming, ranking, and file writes;
    the adapter is responsible only for the HTTP/parse layer.

    Each adapter is registered in ``ADAPTER_REGISTRY`` by a short
    name (``"worldbank"``, ``"oecd"``, ...); users opt in by listing
    those names in ``engine.dataset_adapters`` in their YAML.

    Adapters should NOT raise on transient failures — catch the
    error, log it, return ``[]``. The engine treats an empty return
    as "this adapter had nothing useful" and moves on; raising would
    abort the whole auto_collect_data node.
    """

    #: Short name used in ``EngineConfig.dataset_adapters`` and in
    #: the registry. Subclasses override.
    name: str = "base"

    @abstractmethod
    async def search(self, query: str, *, top_k: int) -> list[DatasetRow]:
        """Translate ``query`` into adapter-native form, fetch up to
        ``top_k`` rows, and return them. Implementations must be
        defensive: never raise on a transient API error; return an
        empty list and let the engine fall through.
        """
        raise NotImplementedError
