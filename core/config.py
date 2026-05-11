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
