"""The providers, models and multi-model profiles a person can pick from: the doctor's provider list, the web
Settings and Tools pages (``GET /api/provider-choices``) and the ``--*-ensemble`` flags of the one-shot tools all read
them from here."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Choice:
    """One option in a single- or multi-select question."""

    value: Any
    label: str
    description: str = ""


ENSEMBLE_PROFILES: tuple[Choice, ...] = (
    Choice("off", "Single model (default, cheapest)",
           "One LLM call per node. Cost baseline = 1×."),
    Choice("cross_check_only", "Fan out cross_check only (~1.3× cost)",
           "3 models vote per finding; catches issues the writer might miss. Minimal extra spend."),
    Choice("ideate_and_check", "Fan out ideate + cross_check (~2.0× cost)",
           "3 models brainstorm + moderator picks the strongest; 3 models vote on each finding."),
    Choice("full", "Fan out ideate + analyze + cross_check (~2.5× cost)",
           "Multi-model on the three highest-leverage nodes. Best signal; highest cost."),
)


# Cost multipliers vs a single-model baseline. Frontends display these
# next to the profile choice so the user sees the trade-off before
# launch. The numbers are conservative estimates: ideate goes from 1
# call to 4 (3 fan-out + 1 moderator); analyze same; cross_check goes
# from K calls to 3K (no moderator). Across a typical quest of 7-18
# LLM calls, that's the rough multiplier each profile lands at.
_ENSEMBLE_COST_MULTIPLIERS: dict[str, float] = {
    "off": 1.0,
    "cross_check_only": 1.3,
    "ideate_and_check": 2.0,
    "full": 2.5,
}


def estimate_ensemble_cost_multiplier(profile: str) -> float:
    """Return the rough LLM-cost multiplier vs the no-ensemble baseline.
    Frontends render ``~Nx more LLM calls`` next to the profile picker
    so the user sees the cost discipline implication at decision time."""
    return _ENSEMBLE_COST_MULTIPLIERS.get(profile, 1.0)


# An ensemble fans one node out over several models and merges what they say, so it
# needs at least two. FI never picks them: which models are worth the money, and which
# the user has access to, is the user's decision. A profile without models configures
# nothing.
ENSEMBLE_MIN_MODELS = 2


def parse_ensemble_models(raw: Any) -> list[str]:
    """The model ids the user named for an ensemble: a comma / semicolon / newline
    separated string, or a list. Order kept, blanks and repeats dropped."""
    if raw is None:
        return []
    parts = re.split(r"[,;\n]", raw) if isinstance(raw, str) else list(raw)
    out: list[str] = []
    for part in parts:
        model = str(part).strip()
        if model and model not in out:
            out.append(model)
    return out


# Providers. The ``model_options`` table below carries one curated
# model list per provider; ``provider_model`` reuses these at runtime.
# vscode_extension is intentionally absent from the user-facing list —
# the VSCode extension pins it silently and the CLI shouldn't ask.
PROVIDER_CHOICES: tuple[Choice, ...] = (
    Choice("openai", "openai — HTTP direct (OpenAI API)",
           "Needs OPENAI_API_KEY. Fastest path, billed against your OpenAI account."),
    Choice("codex", "codex — HTTP direct (ChatGPT Plus/Pro OAuth backend)",
           "Same transport as openai, points at ChatGPT's backend. Needs OPENAI_API_KEY."),
    Choice("claude_cli", "claude_cli — local Claude Code CLI",
           "Needs `claude` binary on PATH + `claude auth login`. Uses your Claude Pro/Max OAuth."),
    Choice("codex_cli", "codex_cli — local Codex CLI",
           "Needs `codex` binary on PATH + `codex login`. Uses your ChatGPT Plus/Pro OAuth."),
    Choice("copilot_cli", "copilot_cli — local GitHub Copilot CLI",
           "Needs `gh` + Copilot Pro/Business. Uses `gh auth login`."),
    Choice("gemini_cli", "gemini_cli — local Gemini CLI",
           "Needs `gemini` binary on PATH. Uses Google AI Studio OAuth."),
    Choice("ollama", "ollama — local server",
           "Needs Ollama running. Free, fully local; smaller models."),
)


# Per-provider curated model list. Keep these aligned with the actual
# models each provider supports today; users can pick "Other" to type
# a brand-new model name not yet on the list.
PROVIDER_MODEL_OPTIONS: dict[str, tuple[Choice, ...]] = {
    "openai": (
        Choice("gpt-5", "gpt-5", "Latest flagship. Best overall quality."),
        Choice("gpt-5-mini", "gpt-5-mini", "Cheap; good for high-volume nodes."),
        Choice("gpt-4o", "gpt-4o", "Previous-generation default; still strong."),
        Choice("o3", "o3", "Reasoning model; slower + pricier."),
        Choice("o3-mini", "o3-mini", "Cheaper reasoning model."),
    ),
    "codex": (
        Choice("gpt-5", "gpt-5", "ChatGPT backend; matches Plus/Pro web UI."),
        Choice("gpt-5-mini", "gpt-5-mini", "Cheaper Plus/Pro backend variant."),
        Choice("gpt-4o", "gpt-4o", "Previous-generation default."),
    ),
    "claude_cli": (
        Choice("claude-opus-4-7", "claude-opus-4-7", "Latest Opus; strongest model."),
        Choice("claude-sonnet-4-6", "claude-sonnet-4-6", "Latest Sonnet; balanced."),
        Choice("claude-haiku-4-5-20251001", "claude-haiku-4-5", "Latest Haiku; cheapest."),
    ),
    "codex_cli": (
        Choice("gpt-5", "gpt-5", "ChatGPT backend default."),
        Choice("gpt-5-mini", "gpt-5-mini", "Cheaper ChatGPT backend variant."),
        Choice("gpt-4o", "gpt-4o", "Previous-generation default."),
    ),
    "copilot_cli": (
        Choice("claude-sonnet-4.6", "claude-sonnet-4.6",
               "Balanced. The models offered depend on your Copilot subscription; `copilot help config` lists them."),
        Choice("claude-haiku-4.5", "claude-haiku-4.5", "Cheapest and fastest Claude."),
        Choice("claude-opus-4.8", "claude-opus-4.8", "Strongest Claude."),
        Choice("gpt-5.3-codex", "gpt-5.3-codex", "OpenAI Codex model."),
        Choice("mai-code-1.1-flash", "mai-code-1.1-flash", "Fast coding model."),
    ),
    "gemini_cli": (
        Choice("gemini-2.5-pro", "gemini-2.5-pro", "Latest Pro; long context."),
        Choice("gemini-2.5-flash", "gemini-2.5-flash", "Cheaper / faster."),
        Choice("gemini-2.0-flash", "gemini-2.0-flash", "Previous-generation flash."),
    ),
    "ollama": (
        Choice("llama3.3:70b", "llama3.3:70b", "Latest large Llama; needs ~48GB VRAM."),
        Choice("llama3.2:3b", "llama3.2:3b", "Tiny; fits a laptop GPU."),
        Choice("qwen2.5:32b", "qwen2.5:32b", "Strong open-source mid-size."),
        Choice("qwen2.5:7b", "qwen2.5:7b", "Lighter Qwen for laptops."),
    ),
}


def schema() -> dict[str, Any]:
    """The lists above as JSON, for the web pages' pickers."""
    def _choice(c: Choice) -> dict[str, Any]:
        return {"value": c.value, "label": c.label, "description": c.description}

    return {
        "providers": [_choice(c) for c in PROVIDER_CHOICES],
        "provider_models": {name: [_choice(c) for c in opts] for name, opts in PROVIDER_MODEL_OPTIONS.items()},
        "ensemble_profiles": [_choice(c) for c in ENSEMBLE_PROFILES],
        "ensemble_cost_multipliers": dict(_ENSEMBLE_COST_MULTIPLIERS),
        "ensemble_min_models": ENSEMBLE_MIN_MODELS,
    }
