"""Typed configuration for Frontier Insight runs (post-DS redesign).

The schema is intentionally narrow — anything that varies per run goes in
YAML, anything that's a code-level concern stays in Python. Path fields
expand `~` on load via mode='before' validators.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

ProviderName = Literal[
    "codex",
    "openai",
    "gemini",
    "ollama",
    "vllm",
    "claude_code",
    "github_copilot_cli",
    "github_copilot_vscode",
    "codex_cli",      # local Codex CLI (uses ChatGPT Plus/Pro OAuth via `codex login`)
    "claude_cli",     # local Claude Code CLI (uses Claude Pro/Max OAuth via `claude login`)
    "copilot_cli",    # local GitHub Copilot CLI (uses `gh auth login` Copilot Pro/Business)
    "gemini_cli",     # local @google/gemini-cli (uses `gemini` OAuth / Google AI Studio key)
]
EngineFramework = Literal["langgraph"]
SandboxKind = Literal["venv", "docker"]
PaperFormat = Literal["generic", "neurips", "iclr", "ieee_access", "nature_mi"]
OutputKind = Literal["paper_md", "paper_pdf", "slides", "poster", "speech"]


def _expand(v: object) -> object:
    if isinstance(v, str):
        return Path(v).expanduser()
    if isinstance(v, Path):
        return v.expanduser()
    return v


class ProviderConfig(BaseModel):
    name: ProviderName = "codex"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EngineConfig(BaseModel):
    framework: EngineFramework = "langgraph"
    max_iterations: int = 2
    review_loop: bool = True


class ExecutionConfig(BaseModel):
    sandbox: SandboxKind = "venv"
    timeout_s: int = 60 * 30
    python_version: str = "3.11"
    docker_image: str = "python:3.11-slim"


class KnowledgeConfig(BaseModel):
    """Wraps the Axon knowledge layer.

    `axon_config` may be either an inline dict (passed straight to
    `AxonConfig.model_validate(...)`) or a path to an existing Axon YAML
    (`AxonConfig.from_yaml(path)`).
    """

    enabled: bool = True
    axon_config: Path | dict[str, Any] | None = None
    top_k: int = 5
    write_back_quests: bool = True
    # Ordered list of external literature sources used by
    # `Knowledge.search()` when Axon is disabled OR returns zero results.
    # Results from all listed sources run in parallel, are merged and
    # de-duplicated (by DOI / arXiv-id / PMID / normalized title), then
    # truncated to top_k. Source-list order is the dedup priority.
    # Override per-quest in YAML; set to `[]` or "none" to disable.
    #
    # Supported source names:
    #   openalex         — broadest single open index (~200M works, all fields, free).
    #   arxiv            — physics / CS / math / quant-ph / q-bio / stats (free).
    #   crossref         — DOI metadata across paywalled publishers (free).
    #   semantic_scholar — broad coverage + citation graph (free, rate-limited).
    #   pubmed           — biomedical via NCBI E-utilities (free).
    #   core             — 240M open-access papers (free; needs CORE_API_KEY env).
    #   google_scholar   — EXPERIMENTAL via `scholarly` package; no official API;
    #                      rate-limited / blocked by Google. Prefer openalex / s2.
    external_fallback: list[str] | str = Field(
        default_factory=lambda: ["openalex", "arxiv", "crossref"]
    )
    # How the literature node picks which external sources to query:
    #   "auto"   — agent asks the LLM to choose sources from the
    #              catalog (built-in + Axon-ingested supplements) given
    #              the current topic. Falls back to `external_fallback`
    #              if the LLM call fails or returns no valid sources.
    #   "manual" — use `external_fallback` verbatim, no LLM routing.
    source_routing: Literal["auto", "manual"] = "auto"
    # When True, seed the built-in source catalog into Axon at engine
    # construction so it's queryable as `kind="fi_source_catalog"`
    # knowledge later (lets users add their own venue/journal entries
    # for paywalled sources the agent should know about but FI can't
    # programmatically search, e.g. SPIE, IEEE Xplore).
    seed_source_catalog: bool = True
    # Files (PDF / Markdown / plain text) the user has manually placed
    # for this quest — e.g. paywalled PDFs they downloaded from SPIE /
    # IEEE / their institutional library. These are loaded at engine
    # construction and PINNED to the head of every literature-retrieval
    # result for this quest. If Axon is enabled, they are ALSO ingested
    # permanently as `kind="fi_local_paper"` so future quests find them.
    # Paths may be `~`-prefixed; globs are NOT expanded (pass each file).
    local_papers: list[Path] = Field(default_factory=list)
    # Phase 2 paywall-access support: when True, after external retrieval
    # returns Crossref/OpenAlex/etc. hits, FI opportunistically tries to
    # GET the publisher PDF using whatever network access the host has
    # (institutional VPN / Shibboleth / EZproxy already authenticated at
    # the OS level). Login-wall HTML pages are rejected by a
    # Content-Type + %PDF-magic check, so the quest never hangs on a
    # paywalled venue. Off by default — opt-in per quest.
    try_fetch_full_text: bool = False
    # Per-doc HTTP timeout (landing-page GET, PDF GET). Short is good.
    full_text_fetch_timeout_s: float = 15.0
    # Total budget across all docs in one literature batch — caps wall
    # time so a slow VPN can't stall the literature node indefinitely.
    full_text_fetch_total_s: float = 90.0
    # Hard cap on extracted text per doc so a 200-page review doesn't
    # blow up downstream prompts. Truncates the middle.
    full_text_max_kb: int = 64

    @field_validator("local_papers", mode="before")
    @classmethod
    def _expand_local_papers(cls, v: object) -> object:
        if v is None:
            return []
        if isinstance(v, (str, Path)):
            v = [v]
        return [_expand(p) for p in v]

    @field_validator("external_fallback", mode="before")
    @classmethod
    def _normalize_fallback(cls, v: object) -> object:
        # Accept "none" / "" / a single string / a list. Always
        # normalize to list[str]. Empty list = disabled.
        if v in (None, "", "none"):
            return []
        if isinstance(v, str):
            return [v]
        return v
    # When True (default), only quests whose final review verdict is
    # "accept" are written back into Axon's long-term store. When False
    # (and `write_back_quests` is True), every finished quest is
    # ingested regardless of verdict — useful while building up an
    # initial corpus or for debugging.
    write_back_only_on_accept: bool = True

    @field_validator("axon_config", mode="before")
    @classmethod
    def _expand_axon_path(cls, v: object) -> object:
        # Path-shaped input (str or Path) gets `~` expansion; dicts pass
        # through untouched so they reach AxonConfig.model_validate(...).
        if isinstance(v, (str, Path)):
            return _expand(v)
        return v


class OutputConfig(BaseModel):
    kinds: list[OutputKind] = Field(default_factory=lambda: ["paper_md", "paper_pdf"])
    paper_format: PaperFormat = "generic"
    output_dir: Path = Path("./outputs")

    @field_validator("output_dir", mode="before")
    @classmethod
    def _expand_output(cls, v: object) -> object:
        return _expand(v)


class Config(BaseModel):
    topic: str
    title: str | None = None
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    extra_directives: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
