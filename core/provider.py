"""Unified async LLM client + proxy supervisor.

Every provider presents the same `LLMClient.chat(messages) -> str` surface
but resolves to one of three transports:

* **HTTP** (default) — direct OpenAI-compatible endpoint. Used by
  `codex`, `openai`, `gemini`, `ollama`, `vllm`.
* **HTTP via local proxy** — `ProxySupervisor` spawns a child process
  that exposes an OpenAI-compatible REST API on a free localhost port,
  then the http path targets that port. Used by `claude_code` (wrapper
  via `RichardAtCT/claude-code-openai-wrapper`) and `github_copilot_*`
  (via `npx copilot-api@latest start`). Ref-counted across quests.
* **CLI exec** — `LLMClient` spawns a local CLI binary per chat call,
  pipes the prompt in, and reads the response back. No proxy process,
  no HTTP. Used by:
  - `codex_cli` — `codex exec --output-last-message <tmp>` reusing the
    user's `codex login` ChatGPT Plus/Pro OAuth. Prompt is piped on
    stdin (not argv) so it does not appear in local process listings.
  - `claude_cli` — `claude --print --output-format text` reusing the
    user's `claude auth login` Claude Pro/Max OAuth (no `ANTHROPIC_API_KEY`
    needed; OAuth from the CLI's keychain is honored). Prompt on stdin.
  - `copilot_cli` — `copilot -s --allow-all-tools -p <prompt>` reusing
    the user's `gh auth login` Copilot Pro/Business credentials.
    `-s/--silent` strips the trailing stats block; `--allow-all-tools`
    is required for non-interactive mode. **Prompt is on argv** because
    the CLI doesn't document a stdin path — prefer `claude_cli`,
    `codex_cli`, or `gemini_cli` for sensitive prompts.
  - `gemini_cli` — `gemini --yolo -o json -p ""` (from
    `@google/gemini-cli`) with the prompt on stdin. `--yolo` auto-
    approves tool calls (else stdin deadlocks on confirmation prompts).
    `-o json` emits a structured envelope after some CLI warnings; the
    `output_extractor` pulls out the `response` field.

Proxy spawn details:
* `claude_code` — `poetry run python main.py <port>` from
  `FI_CLAUDE_CODE_WRAPPER_DIR` (defaults to `~/claude-code-openai-wrapper`).
* `github_copilot_cli` / `github_copilot_vscode` —
  `npx copilot-api@latest start --port <N> --rate-limit 60 --wait`.
  Both names route to the same proxy.

`ProxySupervisor` is reference-counted so a fleet of N quests using the
same proxy provider shares one proxy process.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .config import ProviderConfig
from .proc_tree import _DESCENDANT_WAIT_S as _TREE_KILL_WAIT_S
from .proc_tree import AsyncProcessTree, ProcessTree
from .thinking_capture import (
    LOOP_CHECK_EVERY, LOOP_TAIL_KEPT, ThinkingLoop, add_thinking, as_text, has_holder, note_declined, note_empty_parts,
    note_extension_older, note_loop, note_sampling, note_thinking, note_timing, repeating_cycle,
    wanted as thinking_wanted,
)

_log = logging.getLogger("frontier_insight.provider")

# Known direct providers and their default endpoints. Users override via
# `provider.base_url` / `provider.model` / `provider.api_key_env`.
_DIRECT_DEFAULTS: dict[str, dict[str, str]] = {
    "codex": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5",
        "api_key_env": "OPENAI_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-pro",
        "api_key_env": "GEMINI_API_KEY",
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5-coder:32b",
        "api_key_env": "",  # Ollama ignores auth; pass empty.
    },
    "vllm": {
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "api_key_env": "",
    },
}

PROXY_PROVIDERS: frozenset[str] = frozenset(
    {"claude_code", "github_copilot_cli", "github_copilot_vscode"}
)
# Back-compat alias for code that already imports the underscore-prefixed name.
# New callers should prefer `PROXY_PROVIDERS`.
_PROXY_PROVIDERS = PROXY_PROVIDERS

# `github_copilot_cli` and `github_copilot_vscode` are config aliases
# that BOTH spawn the same `npx copilot-api@latest start` proxy
# (see ``ProxySupervisor._spawn`` at L313). Without canonicalization
# the supervisor's ``_handles`` dict would key by raw provider name,
# so a fleet with one quest on each alias would spawn TWO redundant
# proxies (and only release one back). Normalize at the supervisor
# boundary so both aliases share a single handle with a single
# refcount.
_PROXY_ALIASES: dict[str, str] = {
    "github_copilot_vscode": "github_copilot_cli",
}


def _canonical_proxy_name(provider_name: str) -> str:
    """Return the canonical name for proxy-provider aliases that
    spawn the same upstream proxy. Used by ``ProxySupervisor`` to
    key its handle dict so two aliases share one proxy process."""
    return _PROXY_ALIASES.get(provider_name, provider_name)


def model_for_node(node_models: dict[str, str] | None, node: str) -> str | None:
    """Resolve ``provider.node_models[node]`` — exact match wins, then a
    dot-prefix match (``"review_panel"`` catches ``"review_panel.foo"``
    when no persona-specific entry exists). ``None`` on any miss, which
    is always a safe no-op: ``LLMClient.chat(model=None)`` falls through
    to the endpoint's already-valid default, so callers never need to
    know whether a given node key maps to anything, or guess a model
    name that might not exist on whichever provider is actually active."""
    if not node_models:
        return None
    if node in node_models:
        return node_models[node]
    if "." in node:
        base = node.split(".", 1)[0]
        if base in node_models:
            return node_models[base]
    return None


def node_output_limit(limits: dict[str, int] | None, node: str) -> int | None:
    """``provider.node_max_tokens`` for ``node``: an exact entry, then the part before the first dot (as
    :func:`model_for_node`), else ``None`` (no limit sent)."""
    if not limits or not node:
        return None
    if node in limits:
        return limits[node]
    if "." in node and node.split(".", 1)[0] in limits:
        return limits[node.split(".", 1)[0]]
    return None


def node_budget(budgets: dict[str, float], node: str, default: float) -> float:
    """The per-node time budget for ``node``: an exact entry wins, then the entry
    for the part before the first dot (``write.patch`` gets ``write``'s budget,
    as it gets ``write``'s model in :func:`model_for_node`), then ``default``."""
    if node in budgets:
        return budgets[node]
    if "." in node and node.split(".", 1)[0] in budgets:
        return budgets[node.split(".", 1)[0]]
    return default


@dataclass(frozen=True)
class _CliSpec:
    """How to invoke a local CLI as a chat endpoint."""

    argv: tuple[str, ...]            # base command; prompt may be appended
    pass_prompt_via: str             # "stdin" | "arg"
    output_via: str                  # "stdout" | "last_message_file" | "stream_json"
    # Optional post-process step on the raw collected content. Used when
    # the CLI emits warnings/info before the real response or wraps the
    # response in a JSON envelope (see gemini_cli). `None` means no
    # extraction — `_run_cli` returns the raw collected content as-is.
    # Ignored for ``output_via="stream_json"`` (the stream parser is
    # already format-aware).
    output_extractor: Callable[[str], str] | None = None
    # If set, and the user passed `provider.model` in their YAML config,
    # FI inserts `[model_flag, <provider.model>]` after argv[0] so the
    # CLI uses the user-specified model instead of its default. Leave
    # `provider.model` empty in YAML to keep the CLI's own default (the
    # most-recent `/model` selection for claude; codex's built-in default,
    # since FI's codex calls do not read `~/.codex/config.toml`).
    model_flag: str | None = None
    # Hard ceiling (characters) on the prompt this CLI will accept on a
    # single turn. ``None`` = no known limit. When set and a prompt exceeds
    # it, ``_run_cli`` trims the MIDDLE of the prompt (keeping the task head
    # + the output-format tail) so the call still goes through instead of the
    # CLI rejecting it with a hard "input_too_large" error. codex_cli caps a
    # turn at 1,048,576 chars; we sit under it to leave room for codex's own
    # system prompt + tool schema wrapped around our instructions.
    max_input_chars: int | None = None
    # Optional transform applied to the prompt just before it is written to
    # stdin. Needed by CLIs whose stdin is a structured stream rather than
    # raw text: antigravity reads one NDJSON envelope per turn and rejects a
    # bare string. ``None`` writes the prompt unchanged.
    stdin_encoder: Callable[[str], str] | None = None
    # How this CLI takes images when a message carries image parts:
    # ``"stream_json"`` sends them inline in a ``--input-format stream-json``
    # user turn (claude); ``"file_flag"`` writes each to a temp file passed
    # with ``image_flag``; ``"file_ref"`` writes them to a temp folder the CLI
    # is given access to (``add_dir_flag``) and names each file in the prompt,
    # for an agent CLI that opens images with its own file viewer but takes
    # only text in its input (agy). ``None`` means the CLI cannot take images,
    # and a call that carries them raises ``ImageInputUnsupported``.
    image_input: str | None = None
    image_flag: str | None = None
    add_dir_flag: str | None = None
    # Settings a CLI reads only from a file in the user's home (agy): written to a fresh home folder of the call's
    # own, which the CLI is pointed at through HOME / USERPROFILE, so the user's own settings are never used or changed.
    home_settings: dict[str, Any] | None = None
    # Pull real token usage out of the CLI's own output. Several CLIs report
    # what they actually consumed — including the system prompt and tool
    # schema they wrap around ours, which the char-count estimator cannot
    # see and which dominates the total. Returning None means "this CLI does
    # not report usage", and the estimator stays in charge.
    usage_extractor: Callable[[str], dict[str, Any] | None] | None = None
    # Pull the model's reasoning summary (if the CLI prints one) out of its stdout, for ``.fi/thinking.jsonl``
    # (core/thinking_capture.py). ``None``: this CLI's reasoning is not read. Read beside ``usage_extractor``, from
    # the same stdout, on the path that collects it (``last_message_file``); never the answer.
    reasoning_extractor: Callable[[str], str] | None = None
    # What this CLI must be told for its output to carry the reasoning at all (``output.save_thinking``): added to a
    # call only when the engine will keep the reasoning (``thinking_capture.wanted()``), never otherwise. claude and
    # codex leave the reasoning text out of a non-interactive call unless asked; ``()``: nothing to ask.
    thinking_args: tuple[str, ...] = ()
    # Environment variables to clear for this CLI's subprocess, and the one
    # variable whose presence means the user chose the env path deliberately
    # and we must not touch anything.
    #
    # Copilot resolves credentials COPILOT_GITHUB_TOKEN > GH_TOKEN >
    # GITHUB_TOKEN > its own stored login, so any ambient GITHUB_TOKEN
    # silently outranks `copilot /login`. That variable is normally present
    # for git and gh, where a repo-scoped fine-grained PAT is the usual
    # thing to have -- and such a PAT has no "Copilot Requests" permission,
    # so the call comes back 401 with a message about the token being
    # invalid or expired. The login it displaced was working the whole time.
    #
    # Diagnosed the hard way: identical calls succeeded from a shell and
    # failed from the quest process, through four rounds of ruling out auth,
    # model, prompt size, prompt content, concurrency, cwd and binary path.
    # Only a shim recording the child's environment showed the token.
    env_unset: tuple[str, ...] = ()
    env_unset_override: str | None = None
    # How this CLI takes ``provider.reasoning_effort``: ``effort_args`` turns
    # a level into the argv to add, and ``effort_levels`` lists the levels the
    # CLI accepts (``None`` = every level FI accepts). ``effort_args=None``
    # means the CLI has no such setting — the level is then not sent, and a
    # warning says so once per process (see ``_cli_effort_args``).
    effort_args: Callable[[str], list[str]] | None = None
    effort_levels: frozenset[str] | None = None
    # Every CLI call runs with its own new, empty temporary directory as the
    # working directory, removed when the call ends. ``cwd_flag`` is for a
    # CLI that also takes that directory as an argument (codex ``-C``);
    # ``None`` passes it only as the process's working directory.
    cwd_flag: str | None = None


def _child_env(spec: _CliSpec) -> dict[str, str] | None:
    """The environment for a CLI subprocess, or None to inherit unchanged.

    Only ``spec.env_unset`` is removed, and only when the spec's override
    variable is absent -- setting that variable is how a user says "I mean
    to authenticate through the environment", and then nothing is touched.
    """
    if not spec.env_unset:
        return None
    if spec.env_unset_override and os.environ.get(spec.env_unset_override):
        return None
    present = [k for k in spec.env_unset if k in os.environ]
    if not present:
        return None
    env = dict(os.environ)
    for key in present:
        env.pop(key, None)
    _log.info(
        "cleared %s for the %s subprocess so its own stored login is used; "
        "set %s to authenticate through the environment instead",
        ", ".join(present), spec.argv[0], spec.env_unset_override,
    )
    return env


#: agy answer-only. It has no flag that turns its tools off (``--mode plan`` and ``--sandbox`` do not; checked), and
#: in print mode it follows the user's ``~/.gemini/antigravity-cli/settings.json``, which is often
#: ``"toolPermission": "always-proceed"``: one figure reading ran 66 shell commands, fetched a paper from arXiv and read
#: FI's own source. These settings, given to every call in a home of its own: a tool that needs approval is refused
#: (print mode cannot ask), and the deny rules refuse shell commands, file writes, URL fetches and MCP tools outright;
#: files outside the call's folder and the image folder FI adds cannot be read. Web search cannot be turned off this
#: way (no setting or rule reaches it; checked), so an agy call can still search the web.
_ANTIGRAVITY_SETTINGS: dict[str, Any] = {
    "toolPermission": "request-review",
    "allowNonWorkspaceAccess": False,
    "artifactReviewPolicy": "asks-for-review",
    "permissions": {"deny": ["command(*)", "write_file(*)", "read_url(*)", "mcp(*)"]},
}


def _encode_antigravity_stdin(prompt: str) -> str:
    """Wrap a prompt as one antigravity stream-json turn.

    ``agy`` will not take a prompt on stdin as plain text — ``-p ""`` is
    rejected with *empty prompt*. Its only stdin path is
    ``--input-format stream-json``, which reads one NDJSON envelope per line
    and requires an ``event`` field; a bare string in ``message`` fails to
    unmarshal.

    This matters more than it looks. The alternative is passing the prompt in
    argv like ``copilot_cli`` does, and on Windows that puts the whole prompt
    under the ~8 KB command-line ceiling — which is why ``copilot_cli`` has to
    trim heavily and is documented as a poor fit for FI's long nodes. Going
    through stdin removes the ceiling entirely.
    """
    return json.dumps({
        "event": "user",
        "message": {"role": "user", "content": _ANTIGRAVITY_PREFACE + prompt},
    }, ensure_ascii=False) + "\n"


#: Said to agy before every request: its shell, file writes and URL fetches are refused (``_ANTIGRAVITY_SETTINGS``), and
#: a model that tries one and is refused may end its turn with no answer, which FI would take for a failed call.
_ANTIGRAVITY_PREFACE = (
    "Answer this request directly in your reply, from what it contains. Do not run commands, write files, fetch web "
    "pages or search the web: those tools are turned off for this request. The only files you may open are image "
    "files the request names.\n\n"
)


def _encode_claude_stream_json(prompt: str, images: list[tuple[str, bytes]]) -> str:
    """One ``--input-format stream-json`` user turn for claude: the images as
    base64 blocks, then the prompt. Checked against the real CLI: it named
    the shapes and colours in a test image."""
    content: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": base64.b64encode(data).decode("ascii")},
        }
        for mime, data in images
    ]
    content.append({"type": "text", "text": prompt})
    return json.dumps(
        {"type": "user", "message": {"role": "user", "content": content}}, ensure_ascii=False,
    ) + "\n"


def _extract_claude_usage(raw: str) -> dict[str, Any] | None:
    """Real token counts from claude's ``result`` event.

    Claude splits its input three ways — ``input_tokens`` is only what was
    neither cached nor being cached, with ``cache_creation_input_tokens`` and
    ``cache_read_input_tokens`` carrying the rest. On a one-word prompt those
    were 2, 19807 and 15749: reporting the 2 as "the prompt" would understate
    the call by four orders of magnitude, so the prompt total is the sum and
    the cache split is kept alongside it.

    That differs from codex and antigravity, where ``input_tokens`` is the
    whole input and the cache figure is a subset of it. The shapes are not
    interchangeable, which is why each CLI needs its own reader rather than
    one generic "find a usage object" pass.

    ``total_cost_usd`` is reported by the CLI itself and is carried through —
    it is the only provider that supplies it, and a real figure beats FI's
    per-token estimate.
    """
    facts = _claude_stream_facts(raw)
    for line in reversed(raw.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except ValueError:
            continue
        if not isinstance(evt, dict) or evt.get("type") != "result":
            continue
        # Which model actually answered, as the CLI reports it: the model of the message that carries the answer (the
        # last assistant message), else the model the CLI says it switched to, else the one that wrote the most in
        # ``modelUsage`` (a helper model the CLI used on the side writes little). Read even when the token counts are
        # missing.
        served = facts.pop("answer_model", None)
        by_model = evt.get("modelUsage")
        if not served and isinstance(facts.get("switched"), dict):
            served = facts["switched"]["to"]
        if not served and isinstance(by_model, dict) and by_model:
            best = max(by_model, key=lambda m: int((by_model[m] or {}).get("outputTokens") or 0)
                       if isinstance(by_model[m], dict) else 0)
            if isinstance(best, str) and best.strip():
                served = best.strip()
        u = evt.get("usage") or {}
        if not u:
            return {**({"served_model": served} if served else {}), **facts} or None
        fresh = int(u.get("input_tokens") or 0)
        cache_write = int(u.get("cache_creation_input_tokens") or 0)
        cache_read = int(u.get("cache_read_input_tokens") or 0)
        out = int(u.get("output_tokens") or 0)
        prompt = fresh + cache_write + cache_read
        measured: dict[str, Any] = {
            "prompt_tokens": prompt,
            "completion_tokens": out,
            "total_tokens": prompt + out,
            "cached_input_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "thinking_tokens": int(
                (u.get("output_tokens_details") or {}).get("thinking_tokens") or 0
            ),
            "estimated": False,
        }
        cost = evt.get("total_cost_usd")
        if isinstance(cost, (int, float)):
            measured["cost_usd_reported"] = float(cost)
        if served:
            measured["served_model"] = served
        measured.update(facts)
        return measured
    facts.pop("answer_model", None)
    return facts or None


#: claude's stream-json ``system`` events that say the CLI answered with a model other than the one asked for: after
#: the asked-for model declined the request (``model_refusal_fallback``: the CLI's own safety-refusal fallback, read
#: from Claude Code 2.1.287; it then takes the answer from another model, a more expensive one in the case that showed
#: it), when that model was unavailable (``model_fallback``), or by the account's model policy
#: (``model_consent_fallback``). Each names ``original_model`` and ``fallback_model``.
_CLAUDE_SWITCH_EVENTS = frozenset({"model_refusal_fallback", "model_fallback", "model_consent_fallback"})

#: The environment variable that turns the claude CLI's refusal fallback off for one call (Claude Code 2.1.x): a
#: request the asked-for model declines then comes back declined instead of being answered by another model.
CLAUDE_NO_REFUSAL_FALLBACK_ENV = "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK"


def _claude_stream_facts(raw: str) -> dict[str, Any]:
    """What claude's stream says about who answered, beside the token counts: ``answer_model`` (the model of the last
    assistant message, the one carrying the answer; ``<synthetic>`` messages the CLI makes itself are skipped),
    ``switched`` (``{"from", "to", "why"}`` when the CLI said it answered with another model than the one asked for) and
    ``refused`` (the answer is a refusal: the CLI said the request was declined with no other model to answer, or the
    last model turn ended with ``stop_reason: refusal``)."""
    facts: dict[str, Any] = {}
    last_stop = None
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except ValueError:
            continue
        if not isinstance(evt, dict):
            continue
        kind = evt.get("type")
        if kind == "system":
            sub = evt.get("subtype")
            to = evt.get("fallback_model")
            if sub in _CLAUDE_SWITCH_EVENTS and isinstance(to, str) and to.strip():
                facts["switched"] = {"from": str(evt.get("original_model") or ""), "to": to.strip(), "kind": sub,
                                     "why": str(evt.get("trigger") or "")}
                # What the model switched away from said no longer ends the call: the other model answers next.
                last_stop = None
                facts.pop("refused", None)
            elif sub == "model_refusal_no_fallback":
                facts["refused"] = True
        elif kind == "assistant":
            model = (evt.get("message") or {}).get("model") if isinstance(evt.get("message"), dict) else None
            if isinstance(model, str) and model.strip() and not model.startswith("<"):
                facts["answer_model"] = model.strip()
        elif kind == "stream_event":
            event = evt.get("event") if isinstance(evt.get("event"), dict) else {}
            if event.get("type") == "message_delta":
                stop = (event.get("delta") or {}).get("stop_reason")
                last_stop = str(stop) if stop else last_stop
    if last_stop == "refusal":
        facts["refused"] = True
    return facts


def _claude_stream_fact_line(raw: bytes) -> str | None:
    """The part of one claude stream-json line :func:`_claude_stream_facts` reads, kept small (an assistant line can
    be over 100 KB of thinking): a model-switch or refusal ``system`` event whole, an assistant message as its model
    only, a turn's end as its stop reason only; ``None`` for any other line."""
    head = raw[:200]
    if b'"system"' in head and b"fallback" in raw:
        return raw.decode("utf-8", errors="replace").strip()
    if b'"assistant"' in head and b'"model"' in raw:
        try:
            msg = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            return None
        if not isinstance(msg, dict):
            return None
        model = (msg.get("message") or {}).get("model") if isinstance(msg.get("message"), dict) else None
        if msg.get("type") == "assistant" and isinstance(model, str):
            return json.dumps({"type": "assistant", "message": {"model": model}})
        return None
    if b'"message_delta"' in raw and b'"stop_reason"' in raw:
        try:
            msg = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            return None
        if not isinstance(msg, dict):
            return None
        event = msg.get("event") if isinstance(msg.get("event"), dict) else {}
        stop = (event.get("delta") or {}).get("stop_reason") if event.get("type") == "message_delta" else None
        if stop:
            return json.dumps({"type": "stream_event", "event": {"type": "message_delta",
                                                                 "delta": {"stop_reason": stop}}})
    return None


def claude_model_family_differs(asked: str, served: str) -> bool:
    """Whether ``served`` (a full claude model id) is plainly another model than ``asked`` (an alias such as
    ``haiku`` or ``opus[1m]``, or a full id): their family words (haiku / sonnet / opus / fable) differ. ``False``
    when either names no family it knows, so an alias it cannot read is never called a switch."""
    families = ("haiku", "sonnet", "opus", "fable")

    def family(name: str) -> str | None:
        words = re.split(r"[-_.\[\]\s]+", (name or "").lower())
        found = [f for f in families if f in words]
        return found[0] if len(found) == 1 else None

    a, s = family(asked), family(served)
    return a is not None and s is not None and a != s


def _extract_codex_reasoning(raw: str) -> str:
    """The reasoning summaries in ``codex exec --json`` output: the text of each completed item of type ``reasoning``
    (once per item id), joined by blank lines; ``""`` when there are none. The answer is never taken from here."""
    parts: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except ValueError:
            continue
        if not isinstance(evt, dict) or evt.get("type") != "item.completed":
            continue
        item = evt.get("item")
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        text = as_text(item.get("text")) or ""
        key = str(item.get("id") or len(parts))
        if text.strip() and key not in seen:
            seen.add(key)
            parts.append(text.strip())
    return "\n\n".join(parts)


def _extract_codex_usage(raw: str) -> dict[str, Any] | None:
    """Real token counts from codex's ``turn.completed`` event.

    codex emits this on stdout under ``--json`` while the assistant's answer
    goes to the ``--output-last-message`` file, so the two are read from
    different places.

    The gap this closes is not a rounding difference. On a one-word prompt
    the char-count estimator reported 7 tokens and codex reported 20,953 —
    the rest being the system prompt and tool schema codex wraps around every
    call, which the estimator cannot see. A cost or budget figure built on
    the estimate is wrong by orders of magnitude.

    ``cached_input_tokens`` is a subset of ``input_tokens``, not an addition,
    so it is carried alongside rather than summed.
    """
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except ValueError:
            continue
        if evt.get("type") != "turn.completed":
            continue
        u = evt.get("usage") or {}
        if not u:
            return None
        inp = int(u.get("input_tokens") or 0)
        out = int(u.get("output_tokens") or 0)
        return {
            "prompt_tokens": inp,
            "completion_tokens": out,
            "total_tokens": inp + out,
            "cached_input_tokens": int(u.get("cached_input_tokens") or 0),
            "cache_write_tokens": int(u.get("cache_write_input_tokens") or 0),
            "reasoning_tokens": int(u.get("reasoning_output_tokens") or 0),
            "estimated": False,
        }
    return None


def _extract_antigravity_usage(raw: str) -> dict[str, Any] | None:
    """Real token counts from antigravity's ``result`` event.

    Worth having because the char-count estimator is not slightly wrong here,
    it is wrong by orders of magnitude: on a one-word prompt the estimator
    reported 7 tokens while the CLI reported ~10,900, the difference being
    the system prompt and tool schema it wraps around ours on every call.
    Cost and budget conclusions drawn from the estimate are meaningless.

    ``cache_read_tokens`` is reported separately and is *part of* the input
    rather than additional to it, so it is carried alongside rather than
    summed — adding it would double-count the cheapest tokens in the call.
    """
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except ValueError:
            continue
        if evt.get("event") != "result":
            continue
        u = (evt.get("result") or {}).get("usage") or {}
        if not u:
            return None
        inp = int(u.get("input_tokens") or 0)
        out = int(u.get("output_tokens") or 0)
        return {
            "prompt_tokens": inp,
            "completion_tokens": out,
            "total_tokens": int(u.get("total_tokens") or (inp + out)),
            "cached_input_tokens": int(u.get("cache_read_tokens") or 0),
            "thinking_tokens": int(u.get("thinking_tokens") or 0),
            "estimated": False,
        }
    return None


def _extract_antigravity_response(raw: str) -> str:
    """Pull the assistant text out of antigravity's event stream.

    The stream is NDJSON: an ``init`` event carrying the tool catalogue,
    optional progress events, then a ``result`` event holding the answer. The
    init event is large and irrelevant, so scanning for ``result`` is both
    cheaper and more robust than assuming a position.

    A non-SUCCESS result is raised rather than returned: the CLI reports
    failures *inside* a 0-exit-status envelope, so silently returning the
    empty ``response`` field would surface a model failure as an empty
    completion, which the pipeline would then treat as a valid answer.

    A capacity failure ("UNAVAILABLE (code 503): No capacity available for
    model ... on the server") is the server's, not the request's, and clears
    on its own; it raises ``_CliCapacityError`` so the call is retried. As a
    plain RuntimeError it was never retried: on 2026-09-15 it ended two quests
    in ``implement`` and skipped a third quest's claim check.
    """
    err = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except ValueError:
            continue
        if evt.get("event") != "result":
            continue
        res = evt.get("result") or {}
        if str(res.get("status", "")).upper() == "SUCCESS":
            return str(res.get("response") or "")
        err = str(res.get("error") or "") or "antigravity reported a non-SUCCESS result"
    if err:
        if _is_capacity_error(err):
            raise _CliCapacityError(f"antigravity: {err}")
        raise RuntimeError(f"antigravity: {err}")
    return raw.strip()


# How long one line of a CLI's stdout may be. asyncio's default is 64 KiB, and a
# claude stream-json line holds a whole message: a real call printed a 110,525-byte
# `assistant` line carrying nothing but its thinking block. Past 64 KiB the reader
# stopped as if at EOF while the CLI, its pipe full, could not exit, so three of
# four sonnet quests died at `plan` or `implement` as a "wedge".
_CLI_STREAM_LIMIT = 256 * 1024 * 1024


_CLI_SPECS: dict[str, _CliSpec] = {
    "claude_cli": _CliSpec(
        # `claude --print` prints the response to stdout and exits.
        # We use ``stream-json`` + ``--include-partial-messages`` so the
        # CLI emits SSE-style events (one JSON-per-line) instead of one
        # silent flush at the end. This is load-bearing on Sonnet 4.6:
        # complex prompts go into extended-thinking mode (tens of
        # thousands of ``thinking_delta`` tokens before any answer text)
        # which under ``--output-format text`` looks identical to a
        # hung process. With stream-json the reader sees a steady
        # stream of events and the inactivity-timer watchdog correctly
        # distinguishes "model is thinking" from "process is stuck".
        #
        # Answer-only: FI asks for text, never for agentic work. Run as an
        # agent, the CLI reads the user's own setup and acts on the machine,
        # so these turn that off (checked against Claude Code 2.1.x: the init
        # event then lists no tools, MCP servers or skills and no memory
        # path, and the usage reports no web search or fetch):
        #   --tools ""                 no built-in tools (Bash, Read, WebSearch, ...);
        #                              the empty string is its own argument
        #   --strict-mcp-config        no MCP server from any config file
        #   --disable-slash-commands   no skills
        #   --safe-mode                no CLAUDE.md, auto-memory, plugins, hooks,
        #                              agents or other customisations (a Haiku
        #                              call went from 8,171 to 5,204 input tokens)
        #   --no-session-persistence   nothing saved to resume later
        # A model may still emit a tool call; with no tools each one comes
        # back "No such tool available" and nothing runs. ``--bare`` would do
        # much of this in one flag but accepts API-key auth only, not the
        # subscription login FI relies on.
        argv=(
            "claude", "--print",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--verbose",  # required by claude_cli for stream-json + --print
            "--tools", "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--safe-mode",
            "--no-session-persistence",
        ),
        pass_prompt_via="stdin",
        output_via="stream_json",
        usage_extractor=lambda raw: _extract_claude_usage(raw),
        # A non-interactive claude call omits the thinking text (each thinking block arrives with an empty text and a
        # signature; checked on Claude Code 2.1.287 with Haiku). The user setting `showThinkingSummaries`, given
        # for this call only, brings the text back (an older CLI ignores an unknown settings key).
        thinking_args=("--settings", '{"showThinkingSummaries":true}'),
        model_flag="--model",   # provider.model = "opus" / "sonnet" / "claude-opus-4-7"
        image_input="stream_json",
        # `claude --effort <level>` (Claude Code 2.1.x). Checked against the
        # CLI: it takes low/medium/high/xhigh/max and IGNORES anything else
        # with a warning on stderr, so an unsupported level would silently
        # run at the default — hence skipped here with our own warning.
        effort_args=lambda level: ["--effort", level],
        effort_levels=frozenset({"low", "medium", "high", "xhigh", "max"}),
    ),
    "codex_cli": _CliSpec(
        # `codex exec` runs Codex non-interactively. We pipe the prompt
        # on stdin (codex reads "instructions from stdin" when no
        # positional PROMPT is given) rather than placing it on the
        # command line — argv would otherwise be visible in `ps`/Task
        # Manager and any other local process listing. stdout is the
        # agent log (token counts, tool calls); the final assistant
        # message is written to the file passed via --output-last-message.
        # ``--json`` turns stdout into JSONL carrying a ``turn.completed``
        # event with the real token counts. The answer still comes from the
        # --output-last-message file, so the two coexist: verified that both
        # flags together return the answer AND the usage envelope.
        #
        # Answer-only: FI asks for text, never for agentic work. With only
        # `exec --json`, one real call inside an FI node read the user's
        # personal skills, ran shell commands and did 8 web searches, 1.35M
        # tokens in that one call. Checked against codex-cli 0.149 with a
        # prompt asking it to list files, run `echo hello` and search the
        # web: before, 2 command_execution + 2 web_search items; with these
        # flags, no tool item and ~13k input tokens.
        #   --ignore-user-config   ~/.codex/config.toml is not read: no MCP
        #                          servers, profiles or custom model
        #                          providers from it (auth still works).
        #                          Effort comes from provider.reasoning_effort.
        #   --ephemeral            no session files written
        #   --skip-git-repo-check  the call's directory is not a git repo
        #   -s read-only           sandbox, in case a command runs anyway
        #   --disable <feature>    every tool surface: shell, unified exec,
        #                          skills, apps, plugins, browser and computer
        #                          use, memories, subagents, image generation,
        #                          code mode (which otherwise ran `echo hello`
        #                          through its JS REPL, or spawned a subagent),
        #                          image viewing and tool suggestions
        #   -c web_search=disabled web search (`-c tools.web_search=false`
        #                          does nothing)
        # plus `-C <the call's empty directory>` (``cwd_flag``).
        argv=(
            "codex", "exec", "--json",
            "--ignore-user-config",
            "--ephemeral",
            "--skip-git-repo-check",
            "-s", "read-only",
            "--disable", "shell_tool",
            "--disable", "unified_exec",
            "--disable", "skill_search",
            "--disable", "apps",
            "--disable", "plugins",
            "--disable", "browser_use",
            "--disable", "computer_use",
            "--disable", "memories",
            "--disable", "multi_agent",
            "--disable", "image_generation",
            "--disable", "code_mode_host",
            "--disable", "view_image",
            "--disable", "tool_suggest",
            "-c", "web_search=disabled",
        ),
        cwd_flag="-C",
        pass_prompt_via="stdin",
        output_via="last_message_file",
        usage_extractor=lambda raw: _extract_codex_usage(raw),
        # `--json` prints each reasoning summary Codex produced as an `item.completed` event whose item is of type
        # `reasoning` (read from codex-cli 0.159's own list of item types: agent_message, reasoning, command_execution,
        # ...; not seen in a real call). Whether a model returns one depends on the model and Codex's own summary
        # setting.
        reasoning_extractor=lambda raw: _extract_codex_reasoning(raw),
        # Without a summary setting codex returns no reasoning item (checked with gpt-6-luna on codex-cli 0.159: the same
        # question gave none, and with the setting a `reasoning` item). It is a short summary, not the full chain.
        thinking_args=("-c", 'model_reasoning_summary="detailed"'),
        # provider.model = "gpt-5.5". Left blank, codex uses its own default
        # model: config.toml's `model` is not read (--ignore-user-config).
        model_flag="-m",
        # `codex exec -i <file>` attaches an image to the prompt; checked with
        # a test image, which it named correctly.
        image_input="file_flag",
        image_flag="-i",
        # codex `turn/start` rejects input over 1,048,576 chars with
        # ``input_too_large``. Sit ~150K under to leave room for codex's
        # own system prompt + tools schema. Hit by analyze/write on quests
        # whose literature + result_json grow large (e.g. after a broaden).
        max_input_chars=900_000,
        # `-c key=value` sets a config value for this call only (config.toml
        # itself is not read, see --ignore-user-config above, so this is the
        # only effort codex gets); the value is parsed as TOML
        # (`codex exec --help`). The codex-cli
        # 0.149 binary's string table lists none/minimal/low/medium/high/
        # xhigh/max/ultra as effort values, so every level FI accepts is
        # passed; whether the model honours a level is the model's business.
        effort_args=lambda level: ["-c", f'model_reasoning_effort="{level}"'],
    ),
    "copilot_cli": _CliSpec(
        # GitHub Copilot CLI (`copilot --prompt`). WARNING — this is an
        # AGENTIC CLI: it interprets prompts as user coding tasks and
        # may reply conversationally instead of running stateless LLM
        # inference. Empirically broken as a chat backend for FI's
        # pipeline (paper.md fills with "Are you trying to X?", code
        # node returns the empty stub). engine._warn_if_unsanctioned_provider
        # prints a loud warning when this provider is selected. Kept
        # in _CLI_SPECS so the configuration shape remains stable for
        # users who set it via the interview before reading docs.
        #
        # `-s/--silent` strips the trailing stats block. `--allow-all-tools`
        # is needed to avoid interactive permission prompts; removing it
        # would make the CLI hang on confirmation. The fundamental issue
        # is the agent loop, not this flag — switch to vscode_extension /
        # claude_cli / codex_cli / gemini_cli / openai for FI use.
        argv=("copilot", "-s", "--allow-all-tools", "-p"),
        pass_prompt_via="arg",
        output_via="stdout",
        model_flag="--model",   # provider.model = "gpt-5.2"
        # See _CliSpec.env_unset: an ambient GITHUB_TOKEN outranks the
        # copilot login and 401s if it lacks "Copilot Requests".
        env_unset=("GITHUB_TOKEN", "GH_TOKEN"),
        env_unset_override="COPILOT_GITHUB_TOKEN",
        # Prompt is passed as a command-line ARG, so on Windows the whole
        # command line is subject to the cmd.exe limit (~8191 chars) — copilot
        # ships as `copilot.BAT` and a long design/write prompt (with
        # accumulated feedback_history) fails with `rc=1: The command line is
        # too long`. Cap under that so the shared prompt-trimmer (see
        # `_truncate_prompt_to_fit`) kicks in first. NOTE: 7 KB is small for a
        # rich node prompt, so copilot_cli trims heavily on long-context nodes
        # — a stdin/temp-file transport (like codex/gemini) is the real fix;
        # prefer codex_cli / openai for long prompts on Windows.
        max_input_chars=7000,
    ),
    "gemini_cli": _CliSpec(
        # `@google/gemini-cli` non-interactive. `--yolo` auto-approves
        # tool calls (otherwise stdin would deadlock waiting for user
        # confirmation). `-o json` emits a structured envelope whose
        # `response` field holds the agent answer; the envelope is
        # preceded by a few lines of CLI-level warnings (true-color,
        # MCP issues, etc.) that we strip via the output extractor.
        # Prompt is piped on stdin (the CLI documents stdin support and
        # appends -p text after it; we pass an empty -p so stdin alone
        # is the prompt content). Avoids argv leakage.
        argv=("gemini", "--yolo", "-o", "json", "-p", ""),
        pass_prompt_via="stdin",
        output_via="stdout",
        output_extractor=lambda raw: _extract_gemini_response(raw),
        model_flag="-m",        # provider.model = "gemini-3-pro"
    ),
    "antigravity_cli": _CliSpec(
        # Google Antigravity (`agy`). Non-interactive via ``--print``, but
        # unlike codex/claude it will NOT read a bare prompt from stdin —
        # ``-p ""`` is rejected outright. Its stdin path is the stream-json
        # input format, which takes one NDJSON envelope per turn; see
        # ``_encode_antigravity_stdin``. That is worth the extra encoding
        # step because the alternative (prompt in argv, as ``copilot_cli``
        # does) caps the prompt at the Windows ~8 KB command-line limit.
        #
        # ``--dangerously-skip-permissions`` is deliberately NOT passed. FI
        # asks this CLI for text completion, not for agentic work. Not passing
        # it is not enough on its own: agy follows the user's settings.json,
        # often "always-proceed", so its tools are refused through settings
        # in a home of the call's own (``home_settings``,
        # ``_ANTIGRAVITY_SETTINGS``).
        argv=(
            "agy", "--print", "",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            # agy's own print-mode timeout defaults to 5 minutes, and when it
            # fires the process exits rc=1 with an EMPTY stderr — there is no
            # message saying what happened. FI's inactivity watchdog is longer,
            # so it sits waiting while the child has already given up, and the
            # quest dies with an unexplained transient error.
            #
            # Observed: an `execute_reflect` turn (failing code + traceback +
            # repair instructions) ran 311s and was killed by agy at its 5m
            # mark. Raised well past FI's own ceilings so FI's timeouts stay
            # the ones that decide, and a slow node fails with a diagnosis
            # instead of a bare rc=1.
            "--print-timeout", "30m",
        ),
        pass_prompt_via="stdin",
        stdin_encoder=lambda p: _encode_antigravity_stdin(p),
        output_via="stdout",
        output_extractor=lambda raw: _extract_antigravity_response(raw),
        usage_extractor=lambda raw: _extract_antigravity_usage(raw),
        model_flag="--model",   # provider.model = e.g. "gemini-3-pro"
        # `agy --help`: "--effort  Reasoning effort for the current CLI
        # session (low|medium|high)".
        effort_args=lambda level: ["--effort", level],
        effort_levels=frozenset({"low", "medium", "high"}),
        # agy's stream-json input takes text only ("content block type
        # \"image\" is not supported"), but it opens an image file with its
        # own viewer when the file's folder is added to its workspace: checked
        # against the real CLI, it named the shapes and colours in a test image.
        image_input="file_ref",
        add_dir_flag="--add-dir",
        home_settings=_ANTIGRAVITY_SETTINGS,
    ),
}
CLI_PROVIDERS: frozenset[str] = frozenset(_CLI_SPECS)
_CLI_PROVIDERS = CLI_PROVIDERS  # back-compat alias

# (provider, level) pairs already warned about, so a quest making ~50 calls
# logs "this level is not applied" once rather than on every call.
_REASONING_EFFORT_WARNED: set[tuple[str, str]] = set()

# HTTP providers whose server accepts only some levels. Ollama 0.17.7's
# OpenAI-compatible endpoint answers anything else with a 400 ("invalid
# reasoning value: 'max' (must be "high", "medium", "low", or "none")"),
# which would fail every call of the quest.
_HTTP_EFFORT_LEVELS: dict[str, frozenset[str]] = {
    "ollama": frozenset({"low", "medium", "high"}),
}


#: Whether the 'extension is older' line was said by a call outside an engine step (one list cell, not a global flag).
_EXTENSION_OLDER_SAID: list[bool] = []


def _warn_reasoning_effort_once(provider: str, level: str, reason: str) -> None:
    key = (provider, level)
    if key in _REASONING_EFFORT_WARNED:
        return
    _REASONING_EFFORT_WARNED.add(key)
    _log.warning(
        "[provider] reasoning_effort=%s is not applied on %s: %s",
        level, provider, reason,
    )


def _ordered_levels(levels: frozenset[str]) -> str:
    from .config import REASONING_EFFORT_LEVELS
    return ", ".join(lv for lv in REASONING_EFFORT_LEVELS if lv in levels)


def _cli_effort_args(spec: _CliSpec, level: str) -> list[str]:
    """The argv that sets ``level`` on this CLI, or ``[]`` (with a one-time
    warning) when the CLI has no such setting or does not accept the level."""
    if not level:
        return []
    name = next((k for k, v in _CLI_SPECS.items() if v is spec), spec.argv[0])
    if spec.effort_args is None:
        _warn_reasoning_effort_once(
            name, level,
            "this CLI has no reasoning-effort setting FI can pass, so it runs "
            "at its own default",
        )
        return []
    if spec.effort_levels is not None and level not in spec.effort_levels:
        _warn_reasoning_effort_once(
            name, level,
            f"it accepts {_ordered_levels(spec.effort_levels)}, so the level is "
            "not passed and the CLI runs at its own default",
        )
        return []
    return spec.effort_args(level)


#: Hosts whose chat calls are streamed. Long non-streamed requests to Moonshot's Kimi API stalled on the way (zero bytes
#: until the read timeout, 27k-token prompts, 2026-09-27) while the same request streamed completed; other providers
#: were not seen to stall, so they keep the plain request.
_STREAMED_HOSTS = frozenset({"api.moonshot.ai", "api.moonshot.cn"})


def _http_streams(endpoint: "ResolvedEndpoint") -> bool:
    """Whether this HTTP endpoint's chat calls are streamed (see :data:`_STREAMED_HOSTS`)."""
    from urllib.parse import urlparse

    try:
        host = (urlparse(endpoint.base_url or "").hostname or "").lower()
    except ValueError:
        return False
    return host in _STREAMED_HOSTS


#: An error sent inside a stream that will clear by itself (read from its type, code and message): tried again, like a
#: 429 or a 5xx. Checked first, so "rate limit ... quota exceeded, retry in 20s" is a rate limit, not an empty account.
_STREAM_TRANSIENT = re.compile(
    r"\b(rate[ _-]?limit\w*|overload\w*|capacity|try again|retry (in|after)|temporar\w*|timed? ?out|"
    r"unavailable|busy|server[ _]error|internal[ _]error|bad gateway)\b"
)
#: Classes that trying again cannot fix (the same request fails the same way): read from the error's ``type`` and
#: ``code`` only, never its free-text message, as whole words. Moonshot's ``content_filter`` and
#: ``exceeded_current_quota_error`` are among them.
_STREAM_PERMANENT = re.compile(
    r"\b(content filter|moderation|invalid request|invalid api key|invalid authentication|authentication error|"
    r"unauthori[sz]ed|permission denied|permission error|forbidden|not found|quota|insufficient balance|"
    r"insufficient quota|billing|payment required|account suspended|context length exceeded)\b"
)


def _stream_error(error: Any, request: Any) -> BaseException:
    """The exception an error sent inside a stream becomes: transient (``httpx.RemoteProtocolError``, tried again) when
    it says it will clear -- a rate limit, an overload, a 429 or 5xx code (:data:`_STREAM_TRANSIENT`) -- or when nothing
    names a permanent class; a 4xx-equivalent ``httpx.HTTPStatusError`` (not tried again) only when its ``type`` or
    ``code`` names one (:data:`_STREAM_PERMANENT`)."""
    if isinstance(error, dict):
        kind = " ".join(str(error.get(k) or "") for k in ("type", "code"))
        message = str(error.get("message") or error.get("type") or error.get("code") or error)
        status = next((error.get(k) for k in ("code", "status", "status_code")
                       if isinstance(error.get(k), int) or str(error.get(k) or "").isdigit()), None)
    else:
        kind, message, status = "", str(error), None
    text = f"the model's stream reported an error: {message[:300]}"

    def norm(s: str) -> str:
        return re.sub(r"[_\-]+", " ", s.lower())

    if status is not None and (int(status) == 429 or int(status) >= 500):
        return httpx.RemoteProtocolError(text)
    # A permanent type or code wins over a hopeful message ("invalid request ... try again with a shorter prompt"),
    # unless the type or code itself says it will clear.
    if _STREAM_PERMANENT.search(norm(kind)) and not _STREAM_TRANSIENT.search(norm(kind)):
        return httpx.HTTPStatusError(text, request=request, response=httpx.Response(400, request=request, text=text))
    if _STREAM_TRANSIENT.search(norm(f"{kind} {message}")):
        return httpx.RemoteProtocolError(text)
    if status is not None and 400 <= int(status) < 500:
        return httpx.HTTPStatusError(text, request=request, response=httpx.Response(400, request=request, text=text))
    return httpx.RemoteProtocolError(text)


def _reasoning_of(data: dict[str, Any]) -> str:
    """The reasoning text an OpenAI-compatible reply carries next to its answer (``reasoning_content`` or
    ``reasoning``), or ``""``."""
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    if not isinstance(message, dict):
        return ""
    for key in ("reasoning_content", "reasoning", "reasoning_text"):
        value = as_text(message.get(key))  # a list of summary strings too
        if value:
            return value
    return ""


def _ollama_native(base_url: str) -> str | None:
    """Ollama's own chat URL for an OpenAI-compatible ``base_url`` that ends in ``/v1`` (the default), else ``None``."""
    base = base_url.rstrip("/")
    return base[: -len("/v1")] + "/api/chat" if base.endswith("/v1") else None


def _ollama_native_body(body: dict[str, Any], *, keep_temperature: bool = True) -> dict[str, Any] | None:
    """The same request in Ollama's native ``/api/chat`` form, asking the model to think (``think``): Ollama's
    OpenAI-compatible endpoint never returns a model's reasoning, its own API does (``message.thinking``). The user's
    ``reasoning_effort`` (``body["reasoning_effort"]``) is passed on as ``think``'s named level; ``None`` when the
    request has something the native form would change (a message with an image, or a field of ``extra_body``).
    ``keep_temperature`` False leaves the temperature out, so Ollama uses the model's own recommended sampling (a model
    that thinks at length can loop at FI's temperature 0)."""
    known = {"model", "messages", "temperature", "max_tokens", "reasoning_effort"}
    messages = body.get("messages")
    if set(body) - known or not isinstance(messages, list):
        return None
    if any(not isinstance(m, dict) or not isinstance(m.get("content"), str) for m in messages):
        return None
    options: dict[str, Any] = {}
    if keep_temperature and body.get("temperature") is not None:
        options["temperature"] = body["temperature"]
    if isinstance(body.get("max_tokens"), int) and not isinstance(body.get("max_tokens"), bool):
        options["num_predict"] = body["max_tokens"]
    return {"model": body["model"], "messages": messages, "stream": True, "think": body.get("reasoning_effort") or True,
            **({"options": options} if options else {})}


def _ollama_as_openai(data: dict[str, Any]) -> dict[str, Any]:
    """Ollama's native reply in the shape of an OpenAI-compatible one (answer, ``reasoning_content``, finish reason,
    usage), so everything after the call reads it as it reads any other."""
    message = data.get("message") if isinstance(data.get("message"), dict) else {}
    thought = message.get("thinking")
    done = data.get("done_reason")
    prompt, out = int(data.get("prompt_eval_count") or 0), int(data.get("eval_count") or 0)
    return {
        "model": data.get("model") if isinstance(data.get("model"), str) else None,
        "choices": [{"index": 0, "finish_reason": "length" if done == "length" else "stop",
                     "message": {"role": "assistant", "content": message.get("content") or "",
                                 **({"reasoning_content": thought} if isinstance(thought, str) and thought else {})}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out},
    }


#: A streamed call's whole-call budget as a multiple of the step's HTTP timeout (which stays the limit on silence between
#: two chunks): a model that keeps streaming its thinking can finish a long step, a call that goes on for ever is still
#: ended.
_STREAM_TOTAL_FACTOR = 4


async def _post_ollama_streamed(http: Any, url: str, body: dict[str, Any], headers: dict[str, str],
                                timeout: float, inactivity: float | None = None) -> dict[str, Any]:
    """Ollama's native chat call as a stream (newline-delimited JSON), put back together into the shape
    :func:`_ollama_as_openai` returns (a model that thinks can be silent to a plain request for longer than its read
    timeout; a stream sends its thinking as it goes).

    Time: ``inactivity`` (default ``timeout``, the step's HTTP timeout) bounds the silence between two lines, so a model
    whose thinking keeps arriving is not cut off by it and a stalled stream is; ``timeout`` x
    :data:`_STREAM_TOTAL_FACTOR` bounds the whole call. Both end as ``httpx.ReadTimeout`` and are tried again under the usual rules. An error
    status raises as ``raise_for_status`` would (the caller reads a refusal of ``think`` from it); an ``error`` sent
    inside the stream is classified by :func:`_stream_error`; a stream that ends without ``done: true`` was cut off
    and is tried again."""
    wait = timeout if inactivity is None else inactivity
    total = timeout * _STREAM_TOTAL_FACTOR
    content: list[str] = []
    thinking: list[str] = []
    last: dict[str, Any] = {}
    done = False
    request = httpx.Request("POST", url)
    # When the first line, the first thinking and the first answer text arrived (seconds after the call started), for
    # one run.log line: it tells a model that was thinking from one that was waiting (queued) before saying anything.
    started = time.monotonic()
    first_any: float | None = None
    first_thinking: float | None = None
    first_content: float | None = None
    tail, unchecked = "", 0  # the end of the reasoning so far, and how much of it has arrived since the last look
    try:
        async with asyncio.timeout(total):
            async with http.stream("POST", url, json=body, headers=headers, timeout=timeout) as r:
                request = r.request
                if r.status_code >= 400:
                    await r.aread()
                    r.raise_for_status()
                lines = r.aiter_lines().__aiter__()
                while True:
                    try:
                        raw = await asyncio.wait_for(lines.__anext__(), wait)
                    except StopAsyncIteration:
                        break
                    except TimeoutError:
                        raise httpx.ReadTimeout(
                            f"the model's stream sent nothing for {wait:g} s (its read timeout)") from None
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError:
                        raise httpx.RemoteProtocolError(
                            f"the model's stream sent a line that is not JSON: {line[:200]!r}") from None
                    if not isinstance(chunk, dict):
                        raise httpx.RemoteProtocolError(
                            f"the model's stream sent a line that is not an object: {line[:200]!r}")
                    if chunk.get("error"):
                        raise _stream_error(chunk["error"], request)
                    message = chunk.get("message") if isinstance(chunk.get("message"), dict) else {}
                    now = time.monotonic() - started
                    first_any = now if first_any is None else first_any
                    if isinstance(message.get("content"), str) and message["content"]:
                        content.append(message["content"])
                        first_content = now if first_content is None else first_content
                    if isinstance(message.get("thinking"), str) and message["thinking"]:
                        thinking.append(message["thinking"])
                        first_thinking = now if first_thinking is None else first_thinking
                        tail = (tail + message["thinking"])[-LOOP_TAIL_KEPT:]
                        unchecked += len(message["thinking"])
                        if unchecked >= LOOP_CHECK_EVERY and not content:
                            unchecked = 0
                            cycle = repeating_cycle(tail)
                            if cycle is not None:
                                # Stop reading at once (leaving the `async with` closes the stream): a loop only ends
                                # when the model's context is full.
                                raise ThinkingLoop(kind=cycle[0], block=cycle[1], repeats=cycle[2],
                                                   chars=sum(map(len, thinking)))
                    if chunk.get("done"):
                        last, done = chunk, True
                        break
    except TimeoutError:
        raise httpx.ReadTimeout(
            f"the model's stream did not finish its answer within {total:g} s ({_STREAM_TOTAL_FACTOR} times the step's {timeout:g} s limit)") from None
    if not done:
        raise httpx.RemoteProtocolError("the model's stream ended before its answer did (no done message)")
    end = time.monotonic() - started

    def _s(v: float | None) -> str:
        return "none" if v is None else f"{v:.0f} s"

    summary = (f"{last.get('model') or body.get('model') or 'the model'}: first output after {_s(first_any)}, thinking "
               f"from {_s(first_thinking)}, answer from {_s(first_content)}, done after {_s(end)} "
               f"({sum(map(len, thinking))} thinking chars, {sum(map(len, content))} answer chars)")
    _log.info("[ollama] %s", summary)
    note_timing(summary)  # the engine writes it in run.log (this module's log does not reach it)
    return {"model": last.get("model"), "done_reason": last.get("done_reason"),
            "prompt_eval_count": last.get("prompt_eval_count"), "eval_count": last.get("eval_count"),
            "message": {"content": "".join(content), **({"thinking": "".join(thinking)} if thinking else {})}}


async def _post_streamed(http: Any, url: str, body: dict[str, Any], headers: dict[str, str],
                         timeout: float) -> dict[str, Any]:
    """The chat call as a stream, put back together into the shape a plain call returns (``choices[0].message.content``,
    ``finish_reason``, ``usage``, ``model``).

    Time: ``timeout`` (the step's own HTTP timeout) is the limit on silence between two reads (httpx's read timeout);
    ``timeout`` x :data:`_STREAM_TOTAL_FACTOR` bounds the whole call. A stream kept open by keep-alive comments, or one
    that trickles for longer than that, fails with ``httpx.ReadTimeout`` and is tried again under the usual rules. Events follow the SSE format
    (several ``data:`` lines of one event are joined; an event that does not parse is a protocol error, never skipped).
    An error status raises as ``raise_for_status`` would; an error sent inside the stream is a 4xx-equivalent when
    retrying cannot fix it (:func:`_stream_error`) and transient otherwise. A stream that ends without a
    ``finish_reason`` was cut off and is tried again; one that finished without ``[DONE]`` or without usage is kept,
    with a warning."""
    options = body.get("stream_options") if isinstance(body.get("stream_options"), dict) else {}
    stream_body = {**body, "stream": True, "stream_options": {**options, "include_usage": True}}
    parts: list[str] = []
    reasoning: list[str] = []
    usage: dict[str, Any] = {}
    # Only a model the stream names: the one asked for is not a report of who answered.
    model: str | None = None
    finish = None
    done = False
    request = httpx.Request("POST", url)

    def take(event_name: str, data_lines: list[str]) -> bool:
        """Fold one event in; True when it was ``[DONE]``."""
        nonlocal usage, model, finish
        if not data_lines:
            return False
        data = "\n".join(data_lines)
        if data.strip() == "[DONE]":
            return True
        try:
            chunk = json.loads(data)
        except ValueError:
            if event_name == "error":
                raise _stream_error(data, request) from None
            raise httpx.RemoteProtocolError(f"the model's stream sent an event that is not JSON: {data[:200]!r}") from None
        if event_name == "error" or (isinstance(chunk, dict) and chunk.get("error")):
            raise _stream_error(chunk.get("error", chunk) if isinstance(chunk, dict) else chunk, request)
        if not isinstance(chunk, dict):
            raise httpx.RemoteProtocolError(f"the model's stream sent an event that is not an object: {data[:200]!r}")
        if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
            usage = chunk["usage"]
        if isinstance(chunk.get("model"), str) and chunk["model"]:
            model = chunk["model"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                parts.append(piece)
            thought = as_text(delta.get("reasoning_content") or delta.get("reasoning"))
            if thought:
                reasoning.append(thought)
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
                # Some servers send the usage on the choice that finishes.
                if isinstance(choice.get("usage"), dict) and choice["usage"]:
                    usage = choice["usage"]
        return False

    try:
        async with asyncio.timeout(timeout * _STREAM_TOTAL_FACTOR):
            async with http.stream("POST", url, json=stream_body, headers=headers, timeout=timeout) as r:
                request = r.request
                if r.status_code >= 400:
                    await r.aread()
                    r.raise_for_status()
                event_name, data_lines = "", []
                async for raw in r.aiter_lines():
                    line = raw.rstrip("\r")
                    if line == "":
                        done = take(event_name, data_lines)
                        event_name, data_lines = "", []
                        if done:
                            break
                        continue
                    if line.startswith(":"):
                        continue  # a comment (keep-alive): not progress
                    field, _, value = line.partition(":")
                    value = value[1:] if value.startswith(" ") else value
                    if field == "data":
                        data_lines.append(value)
                    elif field == "event":
                        event_name = value.strip()
                if not done:
                    done = take(event_name, data_lines)
    except TimeoutError:
        raise httpx.ReadTimeout(
            f"the model's stream did not finish its answer within {timeout * _STREAM_TOTAL_FACTOR:g} s "
            f"({_STREAM_TOTAL_FACTOR} times the step's {timeout:g} s limit)"
        ) from None
    if finish is None:
        raise httpx.RemoteProtocolError("the model's stream ended before its answer did (no finish reason)")
    if not done or not usage:
        _log.warning("[provider] a streamed answer finished (%s) but the stream sent %s; kept as it is",
                     finish, " and ".join(x for x, ok in (("no [DONE]", done), ("no usage", usage)) if not ok))
    return {
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts) if parts else None,
                                             **({"reasoning_content": "".join(reasoning)} if reasoning else {})},
                     "finish_reason": finish}],
        "usage": usage,
    }


def _http_reasoning_effort(endpoint: "ResolvedEndpoint") -> str:
    """The ``reasoning_effort`` value to put in an OpenAI-compatible request
    body, or ``""`` (with a one-time warning) when it must not be sent."""
    level = endpoint.reasoning_effort
    if not level:
        return ""
    name = endpoint.provider_name
    if name in _PROXY_PROVIDERS:
        _warn_reasoning_effort_once(
            name, level,
            "FI does not send it through the proxy providers, so the proxy's "
            "model runs at its own default",
        )
        return ""
    levels = _HTTP_EFFORT_LEVELS.get(name)
    if levels is not None and level not in levels:
        _warn_reasoning_effort_once(
            name, level,
            f"its server accepts {_ordered_levels(levels)} and rejects other "
            "levels with a 400, so the level is not sent",
        )
        return ""
    return level

# Sentinel returned by `resolve_endpoint` when no API key env var was set
# (or the provider is configured as keyless, e.g. ollama/vllm). The OpenAI
# SDK requires a non-empty key string; downstream callers special-case this
# value to skip the `Authorization` header entirely.
_NO_KEY_SENTINEL = "not-needed"


@dataclass
class ResolvedEndpoint:
    base_url: str
    model: str
    api_key: str
    transport: str = "http"          # "http" | "cli" | "vscode_bridge"
    # The YAML `provider.name` (e.g. "openai", "claude_cli",
    # "vscode_extension") that this endpoint was resolved from. Carried
    # through to ``LLMClient._error_note`` so a failed LLM call's
    # traceback says ``provider=openai`` rather than just
    # ``transport=http`` — same transport class can come from openai,
    # gemini, ollama, vllm, … and the user wants to know which one.
    # Empty when an endpoint was hand-constructed in a test.
    provider_name: str = ""
    cli_spec: _CliSpec | None = None  # set when transport == "cli"
    # Only set (non-empty) when the user explicitly chose a CLI model via
    # YAML `provider.model`. `_run_cli` injects `[spec.model_flag, value]`
    # into argv only when this is non-empty. Keeps `model` free to carry
    # a human-readable display string for the Engine's startup log line
    # (e.g. "claude_cli (CLI default)") rather than going blank.
    cli_model_override: str = ""
    # VSCode-extension bridge. When transport == "vscode_bridge",
    # the client picks ONE of two transports based on what the caller
    # populated:
    #   * ``vscode_bridge_socket`` (preferred for --serve / --tools) →
    #     Unix-domain socket (POSIX) or named pipe (Windows) at the
    #     extension's session-long PersistentBridge address.
    #   * ``vscode_bridge_port`` (per-chat-command spawns) → loopback
    #     TCP port the extension's per-command Bridge picked.
    vscode_bridge_port: int = 0
    vscode_bridge_socket: str = ""
    # As with `cli_model_override`: the user's explicit YAML
    # `provider.model` (empty when unset). Sent as the wire-level
    # `model_hint` to the extension; an empty string is the documented
    # signal for "use the model selected in the Chat picker." We keep
    # this separate from `model` because the latter carries a
    # human-readable display string (e.g. "(VSCode chat default)")
    # for the Engine's startup log, and that string is NOT a valid
    # selectChatModels family filter.
    vscode_model_override: str = ""
    # ``provider.reasoning_effort`` (empty when unset). Carried on the
    # endpoint rather than passed to ``LLMClient`` so every client built
    # from a resolved endpoint — the engine's, the fallback chain's, and the
    # slides/poster/speech/visual-check generators' — applies it.
    reasoning_effort: str = ""
    # ``provider.fixed_temperature`` and ``provider.extra_body`` (HTTP providers only): a model that accepts one
    # temperature, and request fields a model needs (see ProviderConfig).
    fixed_temperature: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    # ``provider.node_max_tokens``: the output limit sent for a step (HTTP providers only), looked up by the call's node.
    node_max_tokens: dict[str, int] = field(default_factory=dict)


@dataclass
class _ProxyHandle:
    name: str
    port: int
    proc: subprocess.Popen[bytes]
    refcount: int = 0
    # The proxy's process tree (``npx`` starts ``node``; ``poetry run`` starts ``python``): stopping it stops them all.
    tree: ProcessTree | None = None


@dataclass
class ProxySupervisor:
    """Reference-counted lifecycle for proxy subprocesses.

    The spawn paths call out to `claude-code-openai-wrapper` /
    `copilot-api` to start a localhost proxy and reuse it across
    quests that share the same provider name.
    """

    _handles: dict[str, _ProxyHandle] = field(default_factory=dict)
    # Fast lock: guards ONLY the _handles / _key_locks bookkeeping dicts. It is
    # NEVER held across a blocking spawn or terminate.
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Per-canonical-key locks: serialize spawn/teardown for the SAME provider
    # (so a proxy is never double-spawned) while letting DIFFERENT providers
    # spawn concurrently. Previously a single global lock was held across the
    # up-to-60s ``_spawn`` warmup, so in a --fleet one provider's proxy startup
    # blocked EVERY other provider's acquire — serializing all proxy warmups.
    _key_locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    async def _key_lock(self, key: str) -> asyncio.Lock:
        """Get-or-create the per-key lock under the fast bookkeeping lock."""
        async with self._lock:
            lk = self._key_locks.get(key)
            if lk is None:
                lk = asyncio.Lock()
                self._key_locks[key] = lk
            return lk

    async def acquire(self, provider_name: str) -> _ProxyHandle:
        if provider_name not in _PROXY_PROVIDERS:
            raise ValueError(f"{provider_name!r} is not a proxy provider")
        # Canonicalize aliases (e.g. github_copilot_vscode → github_copilot_cli)
        # so two aliases that spawn the same proxy share a single handle.
        key = _canonical_proxy_name(provider_name)
        key_lock = await self._key_lock(key)
        # Per-key lock: a slow spawn for THIS provider must not block acquires
        # of OTHER providers, but two acquires of the same provider must not
        # both spawn.
        async with key_lock:
            async with self._lock:
                handle = self._handles.get(key)
            if handle is None:
                # `_spawn` ends with a blocking poll of `/v1/models` (up to
                # 60s). Run it in a worker thread AND outside the global lock,
                # so only same-key acquires wait for it.
                handle = await asyncio.to_thread(self._spawn, key)
                async with self._lock:
                    self._handles[key] = handle
            async with self._lock:
                handle.refcount += 1
            return handle

    async def release(self, provider_name: str) -> None:
        key = _canonical_proxy_name(provider_name)
        key_lock = await self._key_lock(key)
        async with key_lock:
            async with self._lock:
                handle = self._handles.get(key)
                if handle is None:
                    return
                handle.refcount -= 1
                teardown = handle.refcount <= 0
                if teardown:
                    # Remove under the global lock BEFORE the (blocking)
                    # terminate, using the canonical key so releasing via a
                    # non-canonical alias can't strand a dead handle. A same-key
                    # acquire waits on this key lock and respawns cleanly.
                    self._handles.pop(key, None)
            if teardown:
                await asyncio.to_thread(self._terminate, handle)

    async def shutdown(self) -> None:
        async with self._lock:
            handles = list(self._handles.values())
            self._handles.clear()
        # Terminate off the event loop so the up-to-5s waits never stall it.
        for h in handles:
            await asyncio.to_thread(self._terminate, h)

    @staticmethod
    def _terminate(handle: _ProxyHandle) -> None:
        """Blocking best-effort teardown of one proxy process. Run via
        ``asyncio.to_thread`` so the up-to-5s wait never stalls the loop.

        A proxy started as a process tree is stopped with everything it
        started, at once: a polite stop of the launcher alone (``npx``, a
        ``.cmd`` shim on Windows) left the real server running and holding its
        port."""
        if handle.tree is not None and getattr(handle.tree, "real", False):
            handle.tree.close()
            return
        handle.proc.terminate()
        try:
            handle.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            handle.proc.kill()

    def _spawn(self, provider_name: str) -> _ProxyHandle:
        port = _free_port()
        env = os.environ.copy()
        if provider_name == "claude_code":
            # Repo path is configurable via FI_CLAUDE_CODE_WRAPPER_DIR; the
            # wrapper has no PyPI release. Default assumes a sibling clone.
            wrapper_dir = env.get(
                "FI_CLAUDE_CODE_WRAPPER_DIR",
                str(Path.home() / "claude-code-openai-wrapper"),
            )
            cmd = ["poetry", "run", "python", "main.py", str(port)]
            cwd: str | None = wrapper_dir
            env["PORT"] = str(port)
        elif provider_name in ("github_copilot_cli", "github_copilot_vscode"):
            # `npx` will resolve and run copilot-api; `--rate-limit` and
            # `--wait` are recommended defaults given the abuse-detection
            # caveat in the upstream README.
            cmd = [
                "npx", "copilot-api@latest", "start",
                "--port", str(port),
                "--rate-limit", "60",
                "--wait",
            ]
            cwd = None
        else:
            raise NotImplementedError(provider_name)

        # cwd validation gives a clearer error than the generic
        # "proxy CLI not found" when the wrapper checkout is missing.
        if cwd is not None and not Path(cwd).is_dir():
            raise RuntimeError(
                f"proxy {provider_name!r}: working directory {cwd!r} does not exist. "
                f"Set FI_CLAUDE_CODE_WRAPPER_DIR to a clone of "
                f"RichardAtCT/claude-code-openai-wrapper with `poetry install` run."
            )
        # On Windows, subprocess.Popen does NOT honor PATHEXT, so an
        # unqualified name like "npx" raises FileNotFoundError even when
        # npx.CMD is sitting in a PATH directory (same gap as the CLI-exec
        # transports below — shutil.which does honor PATHEXT). "poetry" hits
        # the same resolution path; only substitute when a match is found so
        # a genuinely-missing binary still surfaces the RuntimeError below
        # with its real name rather than "None".
        resolved_cmd0 = shutil.which(cmd[0])
        if resolved_cmd0:
            cmd = [resolved_cmd0, *cmd[1:]]
        try:
            # stdout/stderr -> DEVNULL: the proxies are long-lived and
            # write enough log volume to fill an OS pipe buffer if we
            # left them as PIPE without draining. Drop them entirely.
            tree = ProcessTree(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=cwd,
                env=env,
            )
            proc = tree.proc
        except FileNotFoundError as e:
            raise RuntimeError(
                f"proxy CLI {cmd[0]!r} not found on PATH. "
                f"For claude_code: clone RichardAtCT/claude-code-openai-wrapper, "
                f"`poetry install`, set FI_CLAUDE_CODE_WRAPPER_DIR. "
                f"For github_copilot_*: ensure Node and `npx` are on PATH; "
                f"run `npx copilot-api@latest auth` once."
            ) from e
        # If readiness times out, kill the orphan to avoid leaking proxies.
        try:
            _wait_for_openai_endpoint(port, timeout_s=60)
        except BaseException:
            tree.close()  # the proxy and everything it started
            raise
        return _ProxyHandle(name=provider_name, port=port, proc=proc, tree=tree)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_TRANSIENT_BRIDGE_MARKERS = (
    "net::err_http2",
    "net::err_connection",
    "net::err_network",
    "err_http2_protocol_error",
    "econnreset",
    "etimedout",
    "socket hang up",
    "503",
    "504",
    "502",
    "network connection",
    "firewall rules and network",
    "temporarily unavailable",
    "rate limit",
    "request failed",
    "bridge connection dropped",
    "bridge write failed",
    # The extension asked the chat model for its reasoning (Copilot's `_enableThinking`) and the answer failed before
    # its first part: it will not ask that model again, so the same call made again goes through without the option
    # (vscode-frontier-insight/src/lm-messages.ts THINKING_DECLINED_MARKER).
    "the model did not accept the request for its reasoning",
    # The TS-side bridge fires this when it sees no streaming chunks
    # for 180 s; treat as transient so Python's 6-attempt budget
    # retries the request. Wall-time math: each Python attempt invokes
    # the TS bridge (which itself does up to 4 retries with 2/4/8 s
    # backoff = ~14 s) + the 180 s stall budget = up to ~3 min per
    # attempt. Python then waits its own exponential backoff (4/8/16/
    # 32/60 s, capped at 60 s) between attempts. Worst-case wall time
    # before a clean "upstream Copilot unavailable" error: ~18-20 min,
    # not the indefinite hang the old code allowed. Most quests
    # converge on a successful retry well before that.
    "bridge stalled",
)


def _is_bridge_error_transient(msg: str) -> bool:
    """Classify whether a BridgeError message looks worth retrying.

    Pattern-matches against well-known transient markers Copilot's
    backend emits via `vscode.lm.sendRequest` failures (HTTP/2
    protocol errors, connection resets, 5xx, rate limits) AND against
    bridge-side connection failures. Auth errors and "no model
    available for hint" are NOT considered transient.
    """
    m = (msg or "").lower()
    return any(marker in m for marker in _TRANSIENT_BRIDGE_MARKERS)


def _extract_gemini_response(raw: str) -> str:
    """`gemini -o json` emits a structured envelope after a few lines of
    CLI-level warnings (true-color hint, MCP issues, etc.). Scan stdout
    for the first valid JSON object (incrementally decoding from each
    `{`) and return its `response` field. Falls back to the raw text if
    no envelope is parseable — most failures still yield usable content
    for `_parse_json_lenient` downstream.

    Incremental decode avoids two failure modes of a naive
    `find('{')` + `rfind('}')` slice: (a) trailing non-JSON output after
    the envelope confuses `json.loads`, and (b) an earlier `{` inside a
    warning line shifts the start past the real envelope's opening
    brace, again breaking the parse."""
    decoder = json.JSONDecoder()
    i = 0
    n = len(raw)
    while i < n:
        if raw[i] != "{":
            i += 1
            continue
        try:
            envelope, _ = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(envelope, dict):
            response = envelope.get("response")
            if isinstance(response, str):
                return response
        # Parsed an unrelated object; keep scanning past its opening brace.
        i += 1
    return raw


# Rough token cost of one image part, for the usage estimate when a transport
# reports no usage. Vision models bill a page screenshot at around a thousand
# tokens; the estimate only has to be the right order.
_IMAGE_TOKENS_ESTIMATE = 1000
_DATA_URL_RE = re.compile(r"^data:(image/[\w.+-]+);base64,(.+)$", re.DOTALL)
_IMAGE_SUFFIXES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}


class ImageInputUnsupported(RuntimeError):
    """This transport cannot send images to its model. A caller that attaches
    screenshots catches it and checks from the text measurements alone."""


def image_part(data: bytes, mime: str = "image/png") -> dict[str, Any]:
    """An OpenAI ``image_url`` content part carrying ``data`` inline."""
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def _content_text(content: Any) -> str:
    """A message's text: the string itself, or the text parts of an OpenAI
    content-part list joined by blank lines."""
    if isinstance(content, list):
        return "\n\n".join(
            str(part["text"]) for part in content
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
        )
    return "" if content is None else str(content)


def _content_images(content: Any) -> list[tuple[str, bytes]]:
    """``(mime type, bytes)`` for each base64 ``data:`` image part."""
    images: list[tuple[str, bytes]] = []
    if not isinstance(content, list):
        return images
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        ref = part.get("image_url")
        url = ref.get("url", "") if isinstance(ref, dict) else str(ref or "")
        match = _DATA_URL_RE.match(url)
        if match:
            images.append((match.group(1), base64.b64decode(match.group(2))))
    return images


def _message_images(messages: list[dict[str, Any]]) -> list[tuple[str, bytes]]:
    return [image for m in messages for image in _content_images(m.get("content"))]


def _with_text(content: Any, text: str) -> Any:
    """``content`` with its text replaced and its image parts kept."""
    if isinstance(content, list):
        images = [p for p in content if isinstance(p, dict) and p.get("type") == "image_url"]
        return [{"type": "text", "text": text}, *images]
    return text


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    """Flatten OpenAI Chat-Completions messages into one text block.

    FI's engine usually sends a single `user` message; we still handle
    system/assistant prior-turn messages defensively for callers that
    build multi-message conversations.
    """
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = _content_text(m.get("content", ""))
        if role == "system":
            parts.append(f"[system]\n{content}")
        elif role == "assistant":
            parts.append(f"[assistant prior turn]\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(p for p in parts if p)


class _CliTransientError(RuntimeError):
    """Raised when a CLI invocation fails in a way worth retrying
    (non-zero exit with no parseable output). Distinct from `RuntimeError`
    so the retry predicate can target it precisely."""


class _OtherModelAnswer(RuntimeError):
    """A claude CLI answer FI did not use because of which model gave it (``_ask_the_asked_model_again``): recorded as
    a failed attempt with the token counts it cost, never raised."""

    def __init__(self, message: str, *, usage: dict[str, Any] | None, outcome: str) -> None:
        super().__init__(message)
        self.usage = dict(usage) if isinstance(usage, dict) and usage else None
        self.fi_outcome = outcome


class _CliWedgeError(_CliTransientError):
    """The specific failure where a CLI streams (often minutes of
    extended-thinking) then closes stdout WITHOUT producing an answer and
    won't exit even to SIGKILL — an unrecoverable hang, not a transient blip.
    Retrying it just wedges again the same way (observed 4×/4 on a heavy
    claude_cli code-gen prompt). A subclass so the retry loop can cap it
    faster than a normal transient and the message can carry switch-provider
    guidance."""


class _CliCapacityError(_CliTransientError):
    """The model's server has no capacity for the request right now (HTTP 503,
    UNAVAILABLE, RESOURCE_EXHAUSTED). Not the account's quota: a call made
    minutes later with the same account and model succeeds. Retried like any
    transient, after a longer wait, since capacity takes longer than a
    dropped connection to come back."""


# Matched case-insensitively against the CLI's error text.
_CAPACITY_MARKERS: tuple[str, ...] = (
    "no capacity",
    "unavailable",
    "code 503",
    "resource_exhausted",
    "overloaded",
    "at capacity",  # codex: "Selected model is at capacity. Please try a different model."
)


def _is_capacity_error(message: str) -> bool:
    text = message.lower()
    return any(m in text for m in _CAPACITY_MARKERS)


def _cli_stdout_errors(stdout_b: bytes | None) -> list[str]:
    """The failure messages a CLI reported as JSON events on its stdout.

    ``codex exec --json`` says why a turn failed only on stdout, as
    ``{"type":"error","message":...}`` and
    ``{"type":"turn.failed","error":{"message":...}}``; its stderr holds just
    "Reading prompt from stdin...". Without these, a failed call read
    "codex exited rc=1: Reading prompt from stdin..." whatever the cause.

    Only top-level events count. codex also emits an ``error`` *item* on
    every call made with code mode disabled ("Code Mode is unavailable ..."),
    which is a notice rather than the failure, and its "unavailable" would
    read as a capacity error.
    """
    if not stdout_b:
        return []
    messages: list[str] = []
    for line in stdout_b.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            evt = json.loads(line)
        except ValueError:
            continue
        if not isinstance(evt, dict):
            continue
        if evt.get("type") == "error":
            msg = evt.get("message")
        elif evt.get("type") == "turn.failed":
            err = evt.get("error")
            msg = err.get("message") if isinstance(err, dict) else err
        else:
            continue
        if isinstance(msg, str):
            msg = msg.strip()[:500]
            if msg and msg not in messages:
                messages.append(msg)
    return messages


def _cli_retry_wait(retry_state: "Any") -> float:
    """Seconds before the next CLI attempt: 30 to 90 after a capacity error,
    otherwise the jittered exponential wait every other transient gets. The
    short wait (at most 20 s) spent all four attempts on a capacity outage in
    about a minute."""
    outcome = getattr(retry_state, "outcome", None)
    exc = outcome.exception() if outcome is not None else None
    if isinstance(exc, _CliCapacityError):
        import random

        return random.uniform(30.0, 90.0)
    return wait_random_exponential(multiplier=1, max=20)(retry_state)


def _stop_on_repeated_wedge(retry_state: "Any") -> bool:
    """Tenacity stop: bail after the 2nd wedge instead of burning the full
    4-attempt budget on an unrecoverable hang. A wedge means the CLI streamed
    then died without an answer and won't reap — retrying reproduces it, so
    the remaining ~2 attempts (each a multi-minute call) are pure waste. The
    `_CliWedgeError` then reraises with switch-provider guidance."""
    outcome = getattr(retry_state, "outcome", None)
    exc = outcome.exception() if outcome is not None else None
    return isinstance(exc, _CliWedgeError) and retry_state.attempt_number >= 2


# Retrying a call that ran out of wall-clock with the SAME budget is close to
# guaranteed to fail the same way: a codex ``implement_outline`` was killed at
# 600 s three times in a row and cost ~40 minutes and the outline stage. So a
# retry that follows a timeout kill gets a longer budget. Only a timeout widens
# it — a capacity error or a wedge is not a "needed more time" failure, and
# widening those would just lengthen a hang. Capped so a pathological node
# cannot stretch the budget without bound.
_TIMEOUT_RETRY_GROWTH = 1.5
_TIMEOUT_RETRY_MAX_GROWTH = 2.25  # 1.5², reached from the third attempt on


def _is_timeout_kill(exc: BaseException | None) -> bool:
    """True when the child was killed for exceeding its wall-clock budget, as
    opposed to failing for capacity, wedging, or exiting non-zero."""
    if not isinstance(exc, _CliTransientError):
        return False
    if isinstance(exc, (_CliCapacityError, _CliWedgeError)):
        return False
    return "wall-clock" in str(exc).lower()


def _timeout_for_attempt(retry_state: "Any", base_timeout_s: float) -> float:
    """Wall-clock budget for the attempt about to run: the base budget, grown
    while the previous attempt was killed for exceeding it."""
    outcome = getattr(retry_state, "outcome", None)
    exc = outcome.exception() if outcome is not None else None
    if not _is_timeout_kill(exc):
        return base_timeout_s
    attempt = getattr(retry_state, "attempt_number", 1) or 1
    growth = min(_TIMEOUT_RETRY_GROWTH ** (attempt - 1), _TIMEOUT_RETRY_MAX_GROWTH)
    return base_timeout_s * growth


# Output gates. CLI vendors sometimes deliver upstream-state messages
# as plain ``text_delta`` events instead of structured error envelopes
# — the message then gets aggregated into the response and the engine
# happily writes it to ``paper.md`` / ``experiment.py`` / etc. The
# OPC quest produced a 64-byte paper.md containing literally
# ``"You've hit your session limit · resets 2:30am (Europe/Brussels)"``.
# Each gate names itself, defines a short list of patterns it watches
# for, and a length cap so a legitimate long paper that happens to
# mention "rate limit" in body text doesn't false-positive. The
# registry is iterated in declaration order; the first gate that
# fires names itself in the raised ``_CliTransientError`` so retry
# logs show *which* gate triggered (and a future maintainer can tune
# the right pattern list without grepping the whole file).
#
# All gates produce ``_CliTransientError`` → tenacity retries (with the
# user's per-node fallback model, if one is configured). We do NOT
# silently swallow these into the artifact — that's the bug they exist
# to catch.

_CLI_RATE_LIMIT_MARKERS: tuple[str, ...] = (
    "you've hit your session limit",
    "you have hit your session limit",
    "you've hit your weekly limit",
    "you have hit your weekly limit",
    "you've hit your opus limit",
    "you have hit your opus limit",
    "rate limit exceeded · resets",
    "rate limit exceeded - resets",
    "out of credits - upgrade your plan",
    "claude usage limit reached",
    # The CLI names the limit differently per plan and window (session,
    # weekly, Opus...); every one ends "<name> limit · resets <when>", so a
    # wording not listed above is still caught here.
    "limit · resets",
)

# Long-duration / non-recoverable CLI failures: an hours-away session or usage
# limit, exhausted credits, or a dead / mis-scoped auth token. These CANNOT
# clear within the ~4-attempt retry window, so retrying just burns 3-4
# multi-minute calls before crashing anyway (the failure mode that truncated an
# earlier run here). Matched case-insensitively against the CLI error message
# (which carries the CLI's stderr). Deliberately separate from the momentary
# "rate limit exceeded · resets <soon>" case, which IS worth a retry.
_CLI_FATAL_MARKERS: tuple[str, ...] = (
    "session limit",
    "weekly limit",
    "usage limit reached",
    "out of credits",
    "token may be invalid",
    "copilot requests",          # gh token lacking the Copilot scope
    "run '/login'",
    "re-authenticate",
)


def _retry_http_error(exc: BaseException) -> bool:
    """HTTP retry predicate: retry genuine transients only. A 4xx (bad/expired
    key, content policy, malformed or oversized body) is deterministic —
    retrying wastes the backoff budget and crashes anyway — so only 5xx and 429
    are retried alongside transport / read-timeout errors. (The prior
    ``retry_if_exception_type(HTTPStatusError)`` retried every 4xx despite the
    inline comment claiming it didn't.)"""
    if isinstance(exc, (httpx.TransportError, httpx.ReadTimeout)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        resp = getattr(exc, "response", None)
        sc = getattr(resp, "status_code", None)
        if isinstance(sc, int):
            if sc == 429 and _is_exhausted_quota(resp):
                # An empty account or used-up quota does not come back in minutes: fail now, not after a long wait.
                return False
            return sc >= 500 or sc == 429
        # A real HTTP error always carries an int status; a malformed/absent
        # one is surfaced rather than looped.
        return False
    return False


#: A 429 error ``type`` / ``code`` that names a used-up quota, credit or billing problem (OpenAI's
#: ``insufficient_quota``, Moonshot's ``exceeded_current_quota_error``), after ``_``/``-`` become spaces.
_QUOTA_USED_UP_KIND = re.compile(
    r"\b(insufficient (quota|balance|credits?|funds)|exceeded current quota|quota exceeded|billing|payment required|"
    r"account suspended|out of credits?|credit balance)\b"
)
#: A type / code that says the limit clears by itself (a per-minute limit): wins over a quota word in the same field
#: (a ``too_many_requests_error`` with code ``token_quota_exceeded`` is a per-minute limit).
_RATE_LIMIT_KIND = re.compile(r"\b(rate[ _-]?limit\w*|too many requests|per (second|minute)|overload\w*)\b")
#: A 429 message that says a whole billing period's allowance is gone (Ollama's "you have reached your monthly usage
#: limit" comes with the generic type ``api_error``): only these narrow phrases, never the message in general.
_QUOTA_USED_UP_MESSAGE = re.compile(r"\b((monthly|weekly|daily) (usage )?limit|usage limit reached|out of credits?)\b")


def _is_exhausted_quota(resp: Any) -> bool:
    """A 429 that says the account's quota, credit or allowance is used up (waiting minutes will not bring it back)
    rather than a rate limit. Read from the error's ``type`` and ``code`` (:data:`_QUOTA_USED_UP_KIND`, unless they
    also name a rate limit), or from a message naming a monthly / weekly / daily limit. A sane ``Retry-After`` means the
    server expects the call to succeed soon, so it is a rate limit whatever the body says; so is a body that cannot be
    read or names neither."""
    try:
        err = resp.json()
    except Exception:  # noqa: BLE001 -- no JSON body: nothing names a quota
        return False
    if isinstance(err, dict) and isinstance(err.get("error"), (dict, str)):
        err = err["error"]
    if isinstance(err, str):  # Ollama's native shape: {"error": "<message>"}
        err = {"message": err}
    if not isinstance(err, dict):
        return False
    if _retry_after_s(resp) is not None:
        return False

    def norm(s: str) -> str:
        return re.sub(r"[_\-]+", " ", s.lower())

    kind = norm(" ".join(str(err.get(k) or "") for k in ("type", "code")))
    if _RATE_LIMIT_KIND.search(kind):
        return False
    if _QUOTA_USED_UP_KIND.search(kind):
        return True
    message = norm(str(err.get("message") or ""))
    # A message that names a per-minute limit as well ("requests per minute exceeded; daily limit 1000") is a rate limit.
    return bool(_QUOTA_USED_UP_MESSAGE.search(message)) and not _RATE_LIMIT_KIND.search(message)


#: Waits (seconds, before jitter) after each failed attempt while the provider's server is down or busy -- an HTTP 5xx
#: (a Cloudflare 520-529 in front of the provider among them) or a 429 rate limit. A provider outage of a few minutes
#: used to end the quest: four attempts about a second apart (a Moonshot 521 on 2026-09-30, after ~134k tokens spent).
#: Five waits of 10-90 s, each jittered by +/-20 % so the quests of a --fleet do not retry in step, ride out
#: about 3-4.5 minutes (176-264 s) before the call fails.
_HTTP_OUTAGE_WAITS_S: tuple[float, ...] = (10.0, 20.0, 40.0, 60.0, 90.0)
#: Attempts in all while the server is down or busy (one more than the waits); other transient failures keep four.
_HTTP_OUTAGE_ATTEMPTS = len(_HTTP_OUTAGE_WAITS_S) + 1
_HTTP_ATTEMPTS = 4
#: The longest ``Retry-After`` FI waits for. A longer one (or an unreadable one) is ignored in favour of the schedule
#: above, so a server asking for an hour cannot hold a quest that long.
_RETRY_AFTER_MAX_S = 120.0
#: The most one call waits in all during an outage, whatever ``Retry-After`` asks: the last wait is cut to fit.
_HTTP_OUTAGE_MAX_WAIT_S = 300.0
#: 5xx statuses that are not an outage: the server cannot do this (501, 505), or the request itself took longer than
#: the Cloudflare front allows (524) -- the same request fails the same way. They keep the short four-attempt budget.
_NOT_AN_OUTAGE_5XX = frozenset({501, 505, 524})


def _http_outage_status(exc: BaseException | None) -> int | None:
    """The status when ``exc`` is a server outage or rate limit worth a long wait (5xx, or a 429 that is not a used-up
    quota), else ``None``."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    sc = getattr(getattr(exc, "response", None), "status_code", None)
    if not isinstance(sc, int):
        return None
    if (sc >= 500 and sc not in _NOT_AN_OUTAGE_5XX) or (sc == 429 and not _is_exhausted_quota(exc.response)):
        return sc
    return None


def _http_short_retry() -> bool:
    """True when this call should not wait out an outage: a healthy provider later in ``provider.fallback`` can take
    the call (see :class:`FallbackLLMClient`)."""
    slot = CALL_SLOT.get() or {}
    return bool(slot.get("short_retry"))


def _in_outage(retry_state: "Any") -> bool:
    """Whether this call has met a server outage or rate limit on any attempt so far (sticky: a read timeout between
    two 503s is the same outage, and keeps its budget). Never under :func:`_http_short_retry`."""
    if _http_short_retry():
        return False
    if getattr(retry_state, "_fi_outage", False):
        return True
    outcome = getattr(retry_state, "outcome", None)
    exc = outcome.exception() if outcome is not None else None
    if _http_outage_status(exc) is None:
        return False
    try:
        retry_state._fi_outage = True
    except Exception:  # noqa: BLE001 -- a state that takes no attribute only loses stickiness
        pass
    return True


def _retry_after_s(resp: Any) -> float | None:
    """The server's ``Retry-After`` on ``resp`` (seconds or an HTTP date) when it is present and sane: above zero and
    at most :data:`_RETRY_AFTER_MAX_S`. ``None`` otherwise."""
    headers = getattr(resp, "headers", None)
    raw = headers.get("retry-after") if headers is not None else None
    if not raw:
        return None
    raw = str(raw).strip()
    try:
        seconds = float(raw)
    except ValueError:
        from email.utils import parsedate_to_datetime

        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError, IndexError):
            return None
        if when is None:
            return None
        import datetime as _dt

        if when.tzinfo is None:
            when = when.replace(tzinfo=_dt.timezone.utc)
        seconds = (when - _dt.datetime.now(_dt.timezone.utc)).total_seconds()
    if not (seconds == seconds) or seconds <= 0 or seconds > _RETRY_AFTER_MAX_S:  # NaN, past, or too long
        return None
    return seconds


def _http_outage_wait_s(attempt_number: int, exc: BaseException | None) -> float:
    """Seconds to wait after failed attempt ``attempt_number`` while the server is down or busy: the jittered schedule
    (:data:`_HTTP_OUTAGE_WAITS_S`, +/-20 %), or the server's own sane ``Retry-After`` (plus up to 10 % so a fleet does
    not come back in step). On a 429 the ``Retry-After`` is taken as given (the server knows when its limit resets); on a
    5xx it only lengthens the wait, since a tiny one would spend the whole budget in seconds."""
    import random

    base = _HTTP_OUTAGE_WAITS_S[min(max(attempt_number, 1), len(_HTTP_OUTAGE_WAITS_S)) - 1]
    scheduled = base * random.uniform(0.8, 1.2)
    ra = _retry_after_s(getattr(exc, "response", None)) if isinstance(exc, httpx.HTTPStatusError) else None
    if ra is None:
        return scheduled
    asked = ra * random.uniform(1.0, 1.1)
    return asked if exc.response.status_code == 429 else max(asked, scheduled)


def _http_retry_wait(retry_state: "Any") -> float:
    """tenacity ``wait`` for the HTTP transport: a long, jittered wait while the provider's server is down or busy
    (:func:`_http_outage_wait_s`, cut so one call waits at most :data:`_HTTP_OUTAGE_MAX_WAIT_S` in all); the short
    jittered exponential wait (at most 20 s) for anything else transient -- a dropped connection or a read timeout --
    and for any failure when another provider can take the call (:func:`_http_short_retry`)."""
    if _in_outage(retry_state):
        outcome = getattr(retry_state, "outcome", None)
        exc = outcome.exception() if outcome is not None else None
        left = max(0.0, _HTTP_OUTAGE_MAX_WAIT_S - float(getattr(retry_state, "idle_for", 0.0) or 0.0))
        return min(_http_outage_wait_s(retry_state.attempt_number, exc), left)
    return wait_random_exponential(multiplier=1, max=20)(retry_state)


def _http_attempts(retry_state: "Any") -> int:
    """How many attempts the HTTP transport makes in all for this call, given its failures so far."""
    return _HTTP_OUTAGE_ATTEMPTS if _in_outage(retry_state) else _HTTP_ATTEMPTS


def _http_retry_stop(retry_state: "Any") -> bool:
    """tenacity ``stop`` for the HTTP transport: six attempts once the server has been down or busy in this call (or
    until :data:`_HTTP_OUTAGE_MAX_WAIT_S` has been waited), four otherwise."""
    if retry_state.attempt_number >= _http_attempts(retry_state):
        return True
    return _in_outage(retry_state) and float(getattr(retry_state, "idle_for", 0.0) or 0.0) >= _HTTP_OUTAGE_MAX_WAIT_S


async def _http_retry_sleep(seconds: float) -> None:
    """tenacity ``sleep`` for the HTTP transport: gives this call's ``FI_MAX_CONCURRENT_LLM_CALLS`` slot back for the
    wait, so a provider outage of minutes does not hold slots other calls are queued for. The slot is taken again
    before the next attempt. A cancel during the wait or the re-take leaves it given back (``held`` false), so the
    dispatch does not release it a second time and the cancel is not held up waiting for a free slot."""
    box = _HELD_CALL_SLOT.get()
    if box is None or not box.get("held"):
        await asyncio.sleep(seconds)
        return
    box["held"] = False
    box["sem"].release()
    await asyncio.sleep(seconds)
    await box["sem"].acquire()  # cancellation-safe: a cancelled acquire takes nothing
    box["held"] = True


def _retry_cli_error(exc: BaseException) -> bool:
    """CLI retry predicate: retry OSErrors and transient CLI failures, EXCEPT a
    long-duration session/usage/credit limit or an auth failure (see
    ``_CLI_FATAL_MARKERS``) — those can't clear within the retry window, so
    abort fast with the original message rather than burning the budget."""
    if isinstance(exc, OSError):
        return True
    if isinstance(exc, _CliTransientError):
        return not any(m in str(exc).lower() for m in _CLI_FATAL_MARKERS)
    return False


# --- Process-wide LLM-call backpressure -------------------------------------
#
# A single quest fans out to a handful of concurrent LLM calls (a review panel,
# an ensemble). A ``--fleet`` run multiplies that by the number of live quests,
# so an unbounded process can burst into dozens/hundreds of simultaneous
# provider calls and trip provider rate limits (audit: "no global backpressure").
# This semaphore caps concurrent in-flight ``chat`` dispatches across EVERY
# LLMClient in the process. It is created lazily inside the running event loop
# (an ``asyncio.Semaphore`` binds to the loop it is first awaited on), sized from
# ``FI_MAX_CONCURRENT_LLM_CALLS``. Default 0 == unlimited (a no-op nullcontext,
# zero behaviour change) so single quests are never throttled unless an operator
# opts in; set e.g. ``FI_MAX_CONCURRENT_LLM_CALLS=8`` for a large fleet.
#
# Deadlock note: the slot is held only around a single ``_chat_impl`` dispatch,
# which never re-enters ``chat``. Fallback/retry loops acquire a fresh slot per
# attempt (release between), so a saturated cap queues — it cannot deadlock.
# In-provider retries run inside that one dispatch; the HTTP path gives its
# slot back for the length of each wait (``_http_retry_sleep``), so a provider
# outage of minutes does not hold slots other quests' calls are queued for.
_LLM_CALL_SEM: "asyncio.Semaphore | None" = None
#: The current ``chat`` dispatch's slot, ``{"sem": <Semaphore>, "held": bool}`` (``None`` when the cap is off):
#: ``held`` says whether the dispatch holds it right now, so it is released exactly once.
_HELD_CALL_SLOT: "contextvars.ContextVar[dict[str, Any] | None]" = contextvars.ContextVar(
    "fi_held_call_slot", default=None)
_LLM_CALL_SEM_LIMIT: int = -1  # -1 == "not yet resolved from the environment"
_LLM_CALL_SEM_LOOP: "asyncio.AbstractEventLoop | None" = None


def _llm_call_limit() -> int:
    """Resolve the concurrency cap from the environment (0/negative == off)."""
    try:
        return int(os.environ.get("FI_MAX_CONCURRENT_LLM_CALLS", "0") or "0")
    except ValueError:
        return 0


def _llm_call_slot():
    """Return an async context manager gating one LLM dispatch on the
    process-wide cap. ``nullcontext`` (no-op) when the cap is disabled.

    A single shared semaphore enforces the cap across all concurrent callers on
    one event loop; it is recreated when the limit changes (env tweaked) or the
    running loop changes (an ``asyncio.Semaphore`` is bound to the loop it was
    created on — reusing one across loops raises, which pytest-asyncio would
    hit since each test gets a fresh loop)."""
    global _LLM_CALL_SEM, _LLM_CALL_SEM_LIMIT, _LLM_CALL_SEM_LOOP
    limit = _llm_call_limit()
    if limit <= 0:
        return contextlib.nullcontext()
    loop = asyncio.get_running_loop()
    if (
        _LLM_CALL_SEM is None
        or _LLM_CALL_SEM_LIMIT != limit
        or _LLM_CALL_SEM_LOOP is not loop
    ):
        _LLM_CALL_SEM = asyncio.Semaphore(limit)
        _LLM_CALL_SEM_LIMIT = limit
        _LLM_CALL_SEM_LOOP = loop
    return _LLM_CALL_SEM


# Placeholder / "I gave up" patterns that some CLIs emit instead of a
# real response. Distinct from refusal — these signal the LLM lost the
# plot mid-call rather than declining to help.
_CLI_PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "[placeholder]",
    "please provide more details",
    "i need more information to continue",
    "i'm unable to generate a response right now",
)

# Refusal patterns. False-positive risk is higher here — a real paper
# might quote a refusal. So we keep the list tight to phrases that
# typically come from policy-classifier interception (not from the
# topic itself) and only flag when the whole output is short.
_CLI_REFUSAL_MARKERS: tuple[str, ...] = (
    "i cannot assist with that",
    "i can't help with that request",
    "i'm sorry, but i cannot",
    "as an ai language model, i cannot",
    "i'm not able to provide",
)


_CLI_OUTPUT_GATE_MAX_LEN = 500


class _ContentGate:
    """One row in the output-gate registry. ``name`` shows up in retry
    logs / exception messages so you can see which gate fired without
    grepping the marker tables. ``check`` returns the matched marker
    when the gate triggers (otherwise None).
    """

    __slots__ = ("name", "description", "_check")

    def __init__(
        self,
        name: str,
        description: str,
        check: Callable[[str], str | None],
    ) -> None:
        self.name = name
        self.description = description
        self._check = check

    def check(self, text: str) -> str | None:
        return self._check(text)


def _short_marker_check(
    markers: tuple[str, ...], max_len: int = _CLI_OUTPUT_GATE_MAX_LEN,
) -> Callable[[str], str | None]:
    """Build a gate-check that fires when ``text`` is short AND contains
    one of the given case-insensitive markers. The length guard is
    what keeps a real paper-length output from tripping these gates
    just by mentioning the marker substring in body text."""

    def _check(text: str) -> str | None:
        if not text or len(text) > max_len:
            return None
        lowered = text.lower()
        for marker in markers:
            if marker in lowered:
                return marker
        return None

    return _check


# Note: there is intentionally no ``empty_response`` gate here even
# though an empty CLI response is degenerate. Some legitimate paths
# return empty (mock-CLI unit tests assert argv shape against an
# empty-stdout fake; the engine itself handles empty-result-json via
# the execute_reflect loop). Adding a transient retry on empty would
# burn 4× retry budget on those paths for no upstream benefit.
_OUTPUT_GATES: tuple[_ContentGate, ...] = (
    _ContentGate(
        name="rate_limit_message",
        description="upstream rate-limit / session-limit text delivered as content",
        check=_short_marker_check(_CLI_RATE_LIMIT_MARKERS),
    ),
    _ContentGate(
        name="placeholder_response",
        description="model emitted a placeholder / give-up pattern",
        check=_short_marker_check(_CLI_PLACEHOLDER_MARKERS),
    ),
    _ContentGate(
        name="refusal",
        description="policy-classifier refusal short-circuited the response",
        check=_short_marker_check(_CLI_REFUSAL_MARKERS),
    ),
)


def _check_output_gates(text: str) -> tuple[str, str] | None:
    """Iterate the gate registry in declaration order. Return
    ``(gate_name, matched_marker)`` for the first gate that fires, or
    None if every gate cleared the output."""
    for gate in _OUTPUT_GATES:
        marker = gate.check(text)
        if marker is not None:
            return gate.name, marker
    return None


def _looks_like_rate_limit_message(text: str) -> str | None:
    """Back-compat shim for callers that only care about the
    rate-limit gate. New callers should use ``_check_output_gates``
    so every gate gets its chance."""
    return _short_marker_check(_CLI_RATE_LIMIT_MARKERS)(text)


def _truncate_prompt_to_fit(prompt: str, max_chars: int) -> str:
    """Trim ``prompt`` to at most ``max_chars`` by cutting the MIDDLE, so the
    task framing (head) and the output-format instructions (tail) — both of
    which FI puts at the ends of every node prompt — survive. The bulky
    context (literature excerpts, large result_json) lives in the middle and
    is what gets dropped. Lossy, but it lets a capped CLI finish instead of
    hard-failing with ``input_too_large``."""
    if len(prompt) <= max_chars:
        return prompt
    dropped = len(prompt) - max_chars
    marker = (
        f"\n\n[... {dropped} characters of context were trimmed here to fit "
        f"this provider's input limit; the task above and the output-format "
        f"instructions below are intact ...]\n\n"
    )
    budget = max_chars - len(marker)
    if budget <= 0:  # absurdly small cap — just hard-cut the head
        return prompt[:max_chars]
    head = int(budget * 0.78)   # keep more of the head (task + early data)
    tail = budget - head        # keep the output-format instructions
    return prompt[:head] + marker + prompt[-tail:]


async def _run_cli(
    spec: _CliSpec,
    prompt: str,
    *,
    images: list[tuple[str, bytes]] | None = None,
    model: str = "",
    timeout_s: float = 300.0,
    inactivity_timeout_s: float = 180.0,
    post_eof_reap_timeout_s: float = 60.0,
    heartbeat_cb: Callable[[dict[str, Any]], None] | None = None,
    node: str = "",
    usage_out: dict[str, Any] | None = None,
    reasoning_effort: str = "",
    extra_env: dict[str, str] | None = None,
) -> str:
    """Spawn the CLI and collect its response. Three output modes:

    * ``output_via="stdout"`` — collect raw stdout until EOF.
    * ``output_via="last_message_file"`` — final answer lands in a temp
      file; stdout is treated as an opaque agent log.
    * ``output_via="stream_json"`` — line-buffered JSON events (Claude
      Code CLI's ``--output-format stream-json``); the final ``result``
      envelope's text is the return value (the aggregated text deltas
      only when no envelope arrives), thinking deltas are counted but
      discarded.

    Every call runs in a new, empty temporary directory -- the child's
    working directory, also passed as ``spec.cwd_flag`` when set -- that is
    removed when the call ends.

    Two timeouts protect against stuck children:

    * ``timeout_s`` — hard ceiling on TOTAL wall-clock. Default 300 s
      preserves historical behaviour; per-node overrides in
      ``ProviderConfig.node_cli_timeout_s`` can raise this for
      reasoning-heavy nodes (``implement``, ``execute_reflect``).
    * ``inactivity_timeout_s`` — soft watchdog reset on every stdout
      line or stream event. Default 180 s. This is what tells "model is
      thinking and emitting thinking_delta events" apart from "process
      hung silently". The hard ceiling still applies on top — even a
      well-streaming child must finish within ``timeout_s``.

    ``last_message_file`` mode can't watch stdout for activity (it's
    DEVNULL'd to save memory on long-running CLI logs), so the
    inactivity timer is silently disabled there and only the hard
    ceiling applies. Same for any future spec that points stdout
    elsewhere.

    ``heartbeat_cb`` is called periodically (best-effort) with a dict
    describing progress: ``{kind, elapsed_s, idle_s, text_chars,
    thinking_tokens}``. The Engine wires this into its run.log so the
    user sees ``[implement] still thinking — 2400 thinking deltas,
    elapsed 4m22s`` instead of silent dead air. Errors raised by the
    callback are swallowed; never block the LLM call.
    """
    # Resolve the binary up front. On Windows, `asyncio.create_subprocess_exec`
    # does NOT honor PATHEXT, so an unqualified name like "codex" raises
    # FileNotFoundError even when `codex.CMD` is sitting in a PATH directory.
    # `shutil.which` does honor PATHEXT and returns the qualified path, so
    # we substitute argv[0] with whatever it resolves to.
    binary_name = spec.argv[0]
    resolved = shutil.which(binary_name)
    if resolved is None:
        raise RuntimeError(
            f"CLI provider binary {binary_name!r} not found on PATH. "
            f"Install and log in (`claude auth login`, `codex login`, or "
            f"`copilot` via GitHub Copilot CLI) before using this provider."
        )
    # Inject explicit model selection right after the resolved binary,
    # if both the spec supports it and the caller supplied a model.
    # Empty model => CLI keeps its own default.
    argv = [resolved]
    if spec.model_flag and model:
        argv.extend([spec.model_flag, model])
    argv.extend(spec.argv[1:])
    # ``provider.reasoning_effort``: empty unless the level is set AND this
    # CLI can express it. Kept ahead of a trailing prompt flag, like images.
    effort_args = _cli_effort_args(spec, reasoning_effort)
    # Ask for the reasoning only when the engine will keep it (``output.save_thinking``).
    effort_args = [*effort_args, *(spec.thinking_args if thinking_wanted() else ())]
    if spec.pass_prompt_via == "arg":
        argv[-1:-1] = effort_args
    else:
        argv.extend(effort_args)
    tmp_out_path: Path | None = None
    if spec.output_via == "last_message_file":
        tmp = tempfile.NamedTemporaryFile(
            prefix="fi_cli_out_", suffix=".txt", delete=False
        )
        tmp.close()
        tmp_out_path = Path(tmp.name)
        argv.extend(["--output-last-message", str(tmp_out_path)])

    # Images: claude reads them inline from a stream-json turn; a CLI with an
    # image flag gets one temp file per image, removed in the finally below
    # with the answer file. The flags go before a trailing prompt flag.
    image_paths: list[Path] = []
    image_folder: Path | None = None  # agy's images: a folder of their own, removed with them
    home_dir: str | None = None  # agy's settings home (``home_settings``), removed with the call folder
    # The call's one try/finally starts here, before the images are written, so a failure anywhere after it (a full
    # disk while writing them, the prompt's encoding) still removes them.
    call_dir: str | None = None
    tree: AsyncProcessTree | None = None
    aborted = False
    try:
        if images:
            if spec.image_input == "stream_json":
                argv.extend(["--input-format", "stream-json"])
            elif spec.image_input == "file_flag" and spec.image_flag:
                flags: list[str] = []
                for mime, data in images:
                    tmp_image = tempfile.NamedTemporaryFile(
                        prefix="fi_cli_image_", suffix=_IMAGE_SUFFIXES.get(mime, ".png"), delete=False,
                    )
                    image_paths.append(Path(tmp_image.name))
                    with tmp_image:
                        tmp_image.write(data)
                    flags.extend([spec.image_flag, str(image_paths[-1])])
                if spec.pass_prompt_via == "arg":
                    argv[-1:-1] = flags
                else:
                    argv.extend(flags)
            elif spec.image_input == "file_ref" and spec.add_dir_flag:
                image_dir = Path(tempfile.mkdtemp(prefix="fi_cli_images_"))
                image_folder = image_dir
                for n, (mime, data) in enumerate(images, 1):
                    image_paths.append(image_dir / f"image_{n}{_IMAGE_SUFFIXES.get(mime, '.png')}")
                    image_paths[-1].write_bytes(data)
                argv.extend([spec.add_dir_flag, str(image_dir)])
                # The images, in the order the request refers to them: the first file is the first image it mentions.
                prompt = (
                    f"This request comes with {len(image_paths)} image(s), saved as files. Open each one with your file "
                    "viewer and look at it before answering; they are, in the order the request refers to them:\n"
                    + "\n".join(f"{n}. {path}" for n, path in enumerate(image_paths, 1))
                    + "\n\n" + prompt
                )
            else:
                if tmp_out_path is not None:
                    tmp_out_path.unlink(missing_ok=True)
                raise ImageInputUnsupported(f"{spec.argv[0]} cannot send images to its model")

        # Cap the prompt for CLIs with a hard per-turn input limit (codex_cli),
        # trimming the middle context so the call goes through instead of the
        # CLI rejecting it with ``input_too_large`` and failing the node.
        if spec.max_input_chars is not None and len(prompt) > spec.max_input_chars:
            orig_len = len(prompt)
            prompt = _truncate_prompt_to_fit(prompt, spec.max_input_chars)
            _log.warning(
                "[provider] %s%s prompt was %d chars, over the %d-char input cap; "
                "trimmed middle context to fit. The model may miss some "
                "literature/result detail — consider a leaner knowledge footprint "
                "(top_k / literature_excerpt_chars) or evidence_gate_max_broaden: 0.",
                spec.argv[0], f" [{node}]" if node else "", orig_len,
                spec.max_input_chars,
            )

        if spec.pass_prompt_via == "arg":
            argv.append(prompt)
            stdin_bytes: bytes | None = None
        else:  # stdin
            if images and spec.image_input == "stream_json":
                payload = _encode_claude_stream_json(prompt, images)
            else:
                payload = spec.stdin_encoder(prompt) if spec.stdin_encoder else prompt
            stdin_bytes = payload.encode("utf-8")

        # When the real answer lands in `tmp_out_path`, the CLI's stdout is just
        # an agent log; capturing it into a PIPE for a long prompt wastes memory,
        # so it is dropped. stderr stays piped so we can include its tail in
        # error messages.
        #
        # Unless the spec reads usage out of that log. codex reports what it
        # actually consumed in a `turn.completed` event on stdout, and that is
        # the only place the real number exists: the char-count estimator sees
        # only the prompt FI sent, not the system prompt and tool schema codex
        # wraps around it, which is most of the input. Measured on a one-word
        # prompt — estimator 7 tokens, codex 20,953.
        stdout_target = (
            asyncio.subprocess.DEVNULL
            if spec.output_via == "last_message_file" and spec.usage_extractor is None
            else asyncio.subprocess.PIPE
        )

        # Single try/finally so the tmpfile is unlinked on every exit path —
        # spawn failure, transient error, exception during communicate(), or
        # success. Previously a `FileNotFoundError` from spawn leaked the file.
        #
        # The CLI runs in a new, empty directory of its own, removed in the same
        # finally. Without one it inherited FI's working directory (a repository
        # checkout in trend runs) and codex read the files there. FI's own files
        # for the call (the answer file, images) live elsewhere, by absolute path,
        # so the directory is still empty when the CLI starts.
        call_dir = tempfile.mkdtemp(prefix="fi_cli_call_")
        child_env = _child_env(spec)
        if extra_env:
            # This one call's own settings (the claude CLI kept on the asked-for model: _chat_cli).
            child_env = {**(child_env if child_env is not None else os.environ), **extra_env}
        if spec.home_settings is not None:
            home_dir = tempfile.mkdtemp(prefix="fi_cli_home_")
            _write_cli_home(Path(home_dir), spec.home_settings)
            child_env = {**(child_env if child_env is not None else os.environ), "HOME": home_dir,
                         "USERPROFILE": home_dir}
            # A grant inherited from an agy or Antigravity terminal FI was started from would reach the call.
            child_env.pop("ANTIGRAVITY_PERM_GRANTS", None)
        if spec.cwd_flag:
            # Ahead of a trailing prompt flag and its prompt, like the flags above.
            at = len(argv) - 2 if spec.pass_prompt_via == "arg" else len(argv)
            argv[at:at] = [spec.cwd_flag, call_dir]
        try:
            # A process tree (core/proc_tree.py): a timeout, a cancelled call or a failure stops the CLI together
            # with every helper it started (codex and node-based CLIs start their own), not the CLI alone.
            tree = await AsyncProcessTree.start(
                *argv,
                stdin=(
                    asyncio.subprocess.PIPE
                    if stdin_bytes is not None
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=stdout_target,
                stderr=asyncio.subprocess.PIPE,
                env=child_env,
                cwd=call_dir,
                limit=_CLI_STREAM_LIMIT,
            )
            proc = tree.proc
        except FileNotFoundError as e:
            raise RuntimeError(
                f"CLI provider binary {argv[0]!r} not found on PATH. "
                f"Install and log in (`claude auth login` or `codex login`) "
                f"before using this provider."
            ) from e

        try:
            if spec.output_via == "stream_json":
                # Streaming path: only used by claude_cli today. Reads
                # stdout line-by-line, parses ``--output-format stream-
                # json`` events, resets the inactivity watchdog on each
                # event, surfaces text_deltas as the aggregated answer.
                # Required for Sonnet 4.6 extended-thinking spans.
                return await _collect_via_streaming(
                    proc, argv, spec, stdin_bytes,
                    timeout_s=timeout_s,
                    inactivity_timeout_s=inactivity_timeout_s,
                    post_eof_reap_timeout_s=post_eof_reap_timeout_s,
                    heartbeat_cb=heartbeat_cb,
                    node=node,
                    usage_out=usage_out,
                    tree=tree,
                )
            # Legacy ``communicate()`` path for everything else
            # (codex_cli's ``last_message_file``, plus gemini_cli /
            # copilot_cli plain ``stdout`` mode). Only the total
            # wall-clock budget applies — these CLIs don't emit
            # stream-style progress, so an inactivity timer would either
            # fire false-positives (silent agent log) or do nothing useful
            # (single stdout flush at end). Preserved here so existing
            # tests that mock ``proc.communicate()`` keep working.
            return await _collect_via_communicate(
                proc, argv, spec, stdin_bytes, tmp_out_path, timeout_s,
                heartbeat_cb=heartbeat_cb, node=node, usage_out=usage_out,
                tree=tree,
            )
        except (RuntimeError, asyncio.TimeoutError) as exc:
            # The CLI's own log is in the call's home and goes with it: its errors go into the message first.
            note = _cli_home_errors(home_dir) if home_dir is not None else ""
            if note and exc.args and isinstance(exc.args[0], str):
                exc.args = (f"{exc.args[0]}\n{spec.argv[0]}'s own log said: {note}", *exc.args[1:])
            raise
        except asyncio.CancelledError:
            aborted = True  # the CLI may have exited while a helper it started still holds its output pipe
            raise
    finally:
        try:
            # First: a CLI still running (a cancelled call, an error) is stopped with its helpers and waited for,
            # and on Windows anything it left running goes with the job. Its folders below are then not in use.
            if tree is not None:
                await tree.aclose(aborted=aborted)
        finally:
            if tmp_out_path is not None:
                tmp_out_path.unlink(missing_ok=True)
            for image_path in image_paths:
                image_path.unlink(missing_ok=True)
            if image_folder is not None:
                shutil.rmtree(image_folder, ignore_errors=True)
            if home_dir is not None:
                _remove_call_dir(home_dir)
            if call_dir is not None:
                _remove_call_dir(call_dir)


def _write_cli_home(home: Path, settings: dict[str, Any]) -> None:
    """Lay out a CLI call's own home (agy's): FI's settings, with the model the user chose in their own agy settings
    kept (FI never picks a model for them), and an update check marked as just done, so a call does not start agy's
    background updater (a fresh home otherwise starts one on every call)."""
    folder = home / ".gemini" / "antigravity-cli"
    folder.mkdir(parents=True)
    chosen = dict(settings)
    try:
        own = json.loads((Path.home() / ".gemini" / "antigravity-cli" / "settings.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        own = {}
    if isinstance(own, dict) and isinstance(own.get("model"), str) and own["model"].strip():
        chosen["model"] = own["model"]
    (folder / "settings.json").write_text(json.dumps(chosen), encoding="utf-8")
    (folder / "last_check.timestamp").write_bytes(b"")


#: The lines agy writes when a print-mode run fails (its own messages). Only these are shown: agy writes other error
#: lines on every call, successful ones included (a sign-in check before its silent sign-in, a file watcher, a missing
#: conversations folder in a fresh home, ...), and one of those shown with a failure would read as its cause.
_AGY_FAILURE_LINES = (
    "Print mode: run ended with error",
    "Print mode: turn ended with error after partial response",
    "Print mode: stream failed before the cascade started",
    "Print mode: timed out waiting for cascade to start running",
    "Print mode: print timeout after",
    "Print mode: SendUserMessage failed",
    "Print mode: WaitForConversationFullyIdle failed",
    "Print mode: conversation update stream failed",
    "Print mode: not logged in and no controlling terminal",
    "Print mode: silent auth failed",
    "Print mode: auth error",
    "Print mode: auth timed out",
    "Print mode: auth cancelled or interrupted",
    "Print mode: eligibility check failed",
)


def _cli_home_errors(home: str) -> str:
    """The line where agy's own log in a call's home (``cli.log``) says its run failed, or ''."""
    try:
        text = (Path(home) / ".gemini" / "antigravity-cli" / "cli.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    failed = [line.strip() for line in text.splitlines() if any(mark in line for mark in _AGY_FAILURE_LINES)]
    return failed[-1][:600] if failed else ""


def _remove_call_dir(path: str) -> None:
    """Remove a CLI call's working directory, whatever the CLI left in it.

    Never raises: the call's answer or error matters more than a leftover
    directory. On Windows a directory cannot be removed while a process still
    has it as its working directory (a grandchild the kill did not reach), and
    that case is logged.
    """
    shutil.rmtree(path, ignore_errors=True)
    if os.path.exists(path):
        _log.warning(
            "could not remove the CLI call directory %s; a process may still "
            "be using it", path,
        )


# Cadence of the run.log heartbeat for the non-streaming CLIs (codex/copilot/
# gemini). Kept well under the Engine's 30 s heartbeat throttle so a fresh
# "still waiting" line reliably lands every ~30 s. Module-level so tests can
# shrink it.
_COMMUNICATE_HEARTBEAT_INTERVAL_S = 10.0


async def _collect_via_communicate(
    proc: asyncio.subprocess.Process,
    argv: list[str],
    spec: _CliSpec,
    stdin_bytes: bytes | None,
    tmp_out_path: Path | None,
    timeout_s: float,
    *,
    heartbeat_cb: Callable[[dict[str, Any]], None] | None = None,
    node: str = "",
    usage_out: dict[str, Any] | None = None,
    tree: AsyncProcessTree | None = None,
) -> str:
    """Legacy ``communicate()`` path. Two sub-cases:

    * ``output_via="last_message_file"`` (codex_cli) — stdout is
      DEVNULL because the real answer lives in a temp file. The temp
      file gets read after the child exits.
    * ``output_via="stdout"`` (gemini_cli, copilot_cli) — stdout is
      captured as the raw response; ``output_extractor`` may then
      strip per-CLI envelope/banner lines.

    Only the total wall-clock budget applies. The stream-json
    inactivity-timer path lives in ``_collect_via_streaming`` and is
    used for ``output_via="stream_json"`` (claude_cli) — that one
    distinguishes "model is thinking" from "process is hung" by
    resetting on every stream event.

    These CLIs emit no stream events, so without help they'd be totally
    silent in run.log for the whole call — and the dashboard, which infers
    a quest's status from log recency, shows them as "pending"/idle. A
    background heartbeat ticks ``heartbeat_cb`` every ~10 s with zero
    progress counters (there's genuinely no token-level signal here), which
    the Engine renders as ``[node] still waiting on LLM — no events yet,
    elapsed=Ns`` at its usual 30 s throttle. Purely cosmetic — it can't
    affect the result and is torn down in the finally."""
    _beat_stop = asyncio.Event()
    _beat_task: asyncio.Task[None] | None = None
    if heartbeat_cb is not None:
        _beat_start = time.monotonic()

        async def _beat() -> None:
            while not _beat_stop.is_set():
                try:
                    await asyncio.wait_for(
                        _beat_stop.wait(), timeout=_COMMUNICATE_HEARTBEAT_INTERVAL_S,
                    )
                except asyncio.TimeoutError:
                    el = time.monotonic() - _beat_start
                    try:
                        heartbeat_cb({
                            "kind": "cli_progress", "node": node,
                            "elapsed_s": el, "idle_s": el,
                            "text_chars": 0, "thinking_tokens": 0,
                        })
                    except Exception:
                        pass

        _beat_task = asyncio.ensure_future(_beat())
    try:
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(stdin_bytes), timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            elapsed = f"{timeout_s:g}s" if timeout_s >= 1 else f"{timeout_s * 1000:g}ms"
            kill_clean = await _kill_and_reap(proc, spec.argv[0], tree)
            raise _CliTransientError(
                f"{spec.argv[0]} exceeded {elapsed} wall-clock and was killed"
                + ("" if kill_clean else " (post-kill wait timed out)")
            )
        if proc.returncode != 0:
            stderr_tail = stderr_b.decode("utf-8", "replace")[-500:]
            # codex says why on stdout (see ``_cli_stdout_errors``); its
            # stderr alone read "Reading prompt from stdin..." for every cause.
            reported = _cli_stdout_errors(stdout_b)
            if reported:
                message = (
                    f"{argv[0]} exited rc={proc.returncode}: {' | '.join(reported)} "
                    f"(stderr: {stderr_tail.strip()})"
                )
            else:
                message = f"{argv[0]} exited rc={proc.returncode}: {stderr_tail}"
            # "Selected model is at capacity" clears on its own after a while,
            # so it gets the capacity wait, not the short one.
            if _is_capacity_error(" ".join(reported)):
                raise _CliCapacityError(message)
            raise _CliTransientError(message)
        raw_stdout = (stdout_b or b"").decode("utf-8", errors="replace")
        if spec.output_via == "last_message_file":
            assert tmp_out_path is not None
            content = tmp_out_path.read_text(encoding="utf-8", errors="replace")
        else:
            content = raw_stdout
        # Usage is read from STDOUT, not from `content`.
        #
        # For most CLIs those are the same thing. For codex they are not: the
        # answer arrives in the --output-last-message file while the token
        # counts arrive as a `turn.completed` event on stdout. Reading usage
        # from `content` would parse the assistant's prose looking for a
        # usage envelope, find none, and fall back to the estimate — silently,
        # since a missing reading is indistinguishable from a CLI that does
        # not report one.
        if usage_out is not None and spec.usage_extractor is not None:
            try:
                _measured = spec.usage_extractor(raw_stdout)
            except Exception:  # noqa: BLE001 - accounting never fails a call
                _measured = None
            if _measured:
                usage_out.update(_measured)
        if spec.reasoning_extractor is not None:
            try:
                note_thinking(spec.reasoning_extractor(raw_stdout))
            except Exception:  # noqa: BLE001 - a keepsake never fails a call
                pass
        if spec.output_extractor is not None:
            content = spec.output_extractor(content)
        final = content.strip()
        gate_hit = _check_output_gates(final)
        if gate_hit is not None:
            gate_name, marker = gate_hit
            raise _CliTransientError(
                f"{spec.argv[0]} tripped output gate {gate_name!r} (matched: "
                f"{marker!r}). This would have been written to disk as the "
                f"artifact; treating as transient so tenacity retries (which "
                f"may also escalate the model)."
            )
        return final
    finally:
        _beat_stop.set()
        if _beat_task is not None:
            try:
                await _beat_task
            except asyncio.CancelledError:
                pass


async def _collect_via_streaming(
    proc: asyncio.subprocess.Process,
    argv: list[str],
    spec: _CliSpec,
    stdin_bytes: bytes | None,
    *,
    timeout_s: float,
    inactivity_timeout_s: float,
    post_eof_reap_timeout_s: float,
    heartbeat_cb: Callable[[dict[str, Any]], None] | None,
    node: str,
    usage_out: dict[str, Any] | None = None,
    tree: AsyncProcessTree | None = None,
) -> str:
    """Read the child's stdout line-by-line with two independent
    timeouts (total + inactivity) and emit periodic heartbeats.

    Distinguishes "model is thinking — events flowing, just no text
    yet" from "process is hung — no events at all" by resetting the
    inactivity timer on every line, regardless of whether it parsed
    as a useful event.
    """
    assert proc.stdout is not None
    start = time.monotonic()
    last_activity = start
    stop = asyncio.Event()
    aggregated: list[str] = []
    raw_result_lines: list[str] = []
    # Running counter so the heartbeat doesn't recompute
    # ``sum(len(s) for s in aggregated)`` on every 1-s tick (that
    # would be O(N²) in stream length). Always updated in lock-step
    # with ``aggregated.append`` so the two never drift.
    text_chars_total = 0
    thinking_token_count = 0
    thinking_streamed = False  # a thinking_delta was seen: the final assistant message would repeat it
    note_thinking("")  # this attempt's own reasoning only
    error_message: str | None = None  # populated from stream_json error events
    result_envelope_seen = False  # see _parse_stream_json_line caller below
    unreadable_line: str | None = None  # a line past even _CLI_STREAM_LIMIT: see read_stdout
    turns: list[dict[str, Any]] = []  # per model turn: its streamed text and why it stopped (see the envelope below)

    async def write_stdin() -> None:
        if stdin_bytes is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(stdin_bytes)
            await proc.stdin.drain()
            proc.stdin.close()
        except (BrokenPipeError, ConnectionResetError):
            # Child exited before consuming stdin — the stdout reader
            # will surface the real error (rc != 0 with stderr tail).
            pass

    async def read_stdout() -> None:
        nonlocal last_activity, thinking_token_count, error_message, thinking_streamed
        nonlocal text_chars_total, result_envelope_seen, unreadable_line
        while True:
            try:
                line = await proc.stdout.readline()
            except ValueError as e:
                # A line longer than the stream's limit. Stopping here used to look exactly like EOF while the CLI,
                # its pipe full, could never exit: every long-thinking claude call then died as a "wedge" after the
                # reap timeout, or returned only the text streamed before that line. The limit is now far above any
                # real line (_CLI_STREAM_LIMIT); past it, drain to EOF so the child can exit, and say why.
                unreadable_line = f"a stream line longer than {_CLI_STREAM_LIMIT} bytes ({e})"
                while await proc.stdout.read(1 << 16):
                    last_activity = time.monotonic()
                return
            except Exception:
                break
            if not line:
                return  # EOF
            last_activity = time.monotonic()
            if spec.output_via == "stream_json":
                # Keep the raw ``result`` envelope: the usage counts live
                # there, and `aggregated` holds only assembled text by the
                # time the extractor runs. Without this the streaming CLIs
                # silently stay on the char-count estimate.
                if usage_out is not None and spec.usage_extractor is not None:
                    raw_line = line.decode("utf-8", errors="replace")
                    if '"type"' in raw_line and '"result"' in raw_line:
                        raw_result_lines.append(raw_line)
                    else:
                        # Who answered, beside the counts (claude's stream): a model switch, the answer's model, a
                        # refused turn. Kept small; every other line is dropped as before.
                        try:
                            fact = _claude_stream_fact_line(line)
                        except Exception:  # noqa: BLE001 -- a record of who answered never stops the call
                            fact = None
                        if fact is not None:
                            raw_result_lines.append(fact)
                text_delta, thinking_inc, err, is_result = (
                    _parse_stream_json_line(line)
                )
                for kind, thought in _stream_thinking(line):
                    if kind == "delta":
                        thinking_streamed = True
                        add_thinking(thought)
                    elif kind == "start":
                        if thinking_streamed:  # a new thinking block: keep it apart from the one before
                            add_thinking("\n\n")
                    elif not thinking_streamed:
                        add_thinking(thought + "\n")
                turn_mark = _stream_turn_mark(line)
                if turn_mark == "start":
                    turns.append({"text": [], "stop": None})
                elif turn_mark is not None and turns:
                    turns[-1]["stop"] = turn_mark
                if text_delta and not is_result and turns:
                    turns[-1]["text"].append(text_delta)
                if is_result:
                    # An answer longer than the output limit arrives as several
                    # turns: the CLI ends one at `max_tokens`, asks the model to
                    # resume mid-thought, and the envelope carries only the last.
                    # A real call returned alpha8116..alpha9000 of alpha1..alpha9000.
                    # The turns cut off that way come before the envelope's text.
                    cut = []
                    for turn in reversed(turns[:-1]):
                        if turn["stop"] != "max_tokens":
                            break
                        cut.insert(0, "".join(turn["text"]))
                    # Always joined: every guard tried against "a CLI that joins the
                    # turns itself" (never observed) could skip a join the real CLI
                    # needs, and a skipped join truncates silently where a doubled
                    # one would show. If the CLI ever changes, the doubling is seen.
                    joined = "".join(cut)
                    if joined:
                        last = "".join(turns[-1]["text"]) if turns else ""
                        text_delta = joined + (text_delta or last)
                    # The ``result`` envelope holds the answer: the final
                    # turn's text. The streamed deltas span every turn, so
                    # when the model narrates before tool calls they carry
                    # that too -- a real run returned "I'll run the echo
                    # command using the bash tool since it's available in
                    # this environment.Let me try other tool names:CANNOT
                    # RUN" where the envelope said "CANNOT RUN". Its text
                    # therefore replaces the deltas; the deltas are the
                    # answer only when no envelope (or an empty one) arrives.
                    result_envelope_seen = True
                    if text_delta:
                        aggregated[:] = [text_delta]
                        text_chars_total = len(text_delta)
                elif text_delta and not result_envelope_seen:
                    aggregated.append(text_delta)
                    text_chars_total += len(text_delta)
                thinking_token_count += thinking_inc
                if err is not None and error_message is None:
                    error_message = err
            else:
                decoded = line.decode("utf-8", errors="replace")
                aggregated.append(decoded)
                text_chars_total += len(decoded)

    async def watchdog() -> None:
        while not stop.is_set():
            await asyncio.sleep(1.0)
            now = time.monotonic()
            elapsed = now - start
            idle = now - last_activity
            if elapsed > timeout_s:
                raise _CliTransientError(
                    f"{spec.argv[0]} exceeded {timeout_s:g}s total wall-clock "
                    f"and was killed (last activity {idle:.0f}s ago, "
                    f"thinking_tokens={thinking_token_count})"
                )
            if idle > inactivity_timeout_s:
                raise _CliTransientError(
                    f"{spec.argv[0]} silent for {idle:.0f}s "
                    f"(inactivity threshold {inactivity_timeout_s:g}s); "
                    f"thinking_tokens={thinking_token_count}, "
                    f"text_chars={text_chars_total}"
                )
            # Best-effort heartbeat. Cadence: only when the caller asked
            # for one. Errors swallowed; never block the LLM call.
            if heartbeat_cb is not None:
                try:
                    heartbeat_cb({
                        "kind": "cli_progress",
                        "node": node,
                        "elapsed_s": elapsed,
                        "idle_s": idle,
                        "text_chars": text_chars_total,
                        "thinking_tokens": thinking_token_count,
                    })
                except Exception:
                    pass

    writer_task = asyncio.create_task(write_stdin())
    reader_task = asyncio.create_task(read_stdout())
    watchdog_task = asyncio.create_task(watchdog())
    watchdog_raised: BaseException | None = None
    try:
        # Race: either the reader finishes (EOF) or the watchdog
        # raises. ``return_when=FIRST_COMPLETED`` lets us catch either.
        done, _pending = await asyncio.wait(
            {reader_task, watchdog_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in done:
            if t.exception() is not None:
                # Save the watchdog's _CliTransientError so we can
                # raise it AFTER we've reaped the child — otherwise
                # the watchdog timeout escapes ``_collect_via_streaming``
                # with the child still running.
                watchdog_raised = t.exception()
    finally:
        stop.set()
        for t in (writer_task, reader_task, watchdog_task):
            if not t.done():
                t.cancel()
        # Reap any cancellations cleanly.
        for t in (writer_task, reader_task, watchdog_task):
            try:
                await t
            except (asyncio.CancelledError, _CliTransientError, Exception):
                pass
        # If the watchdog tripped a timeout, the child is still alive
        # (the watchdog's raise pre-empted any reap path). Kill it
        # before we propagate the error to the caller so it doesn't
        # keep burning CPU + LLM tokens in the background. Done
        # inside ``finally`` so even if the caller cancels us mid-
        # watchdog the kill still runs.
        if watchdog_raised is not None:
            await _kill_and_reap(proc, spec.argv[0], tree)
    if watchdog_raised is not None:
        raise watchdog_raised
    # Reader finished (EOF). Wait for the child to exit and check rc.
    # Important: if we already collected text_deltas before EOF, the
    # LLM call SUCCEEDED — claude.exe streamed its response and then
    # closed stdout. The child handle lingering past a few seconds on
    # Windows (proactor proc cleanup latency) is NOT a reason to throw
    # away ~8 minutes of generation. Two-stage policy:
    #
    #   1. Wait up to 60 s for the OS to fully reap (generous because
    #      this is post-EOF, the heavy work is done; we're just paying
    #      cleanup latency).
    #   2. On reap-timeout: if we got text, RETURN IT (with a warning).
    #      Only kill+raise when there's literally nothing to return.
    #
    # This fixes a regression where a successful 8-minute body call
    # got discarded because Windows took 11 s to mark claude.exe
    # exited — tenacity then re-ran the whole call from scratch.
    if unreadable_line is not None:
        # The answer is not all here, and returning part of it would pass a truncated script or plan off as whole.
        try:
            await asyncio.wait_for(proc.wait(), timeout=post_eof_reap_timeout_s)
        except asyncio.TimeoutError:
            await _kill_and_reap(proc, spec.argv[0], tree)
        raise _CliWedgeError(f"{spec.argv[0]} printed {unreadable_line}; the answer could not be read whole")
    have_output = text_chars_total > 0 or bool(aggregated)
    try:
        rc = await asyncio.wait_for(proc.wait(), timeout=post_eof_reap_timeout_s)
    except asyncio.TimeoutError:
        if have_output:
            # Don't waste the LLM result. Kill the lingering process
            # (best-effort) and continue with what we collected.
            _log.warning(
                "%s stdout closed and reap timed out, but the call "
                "produced %d text chars — returning result anyway "
                "(killing lingering child best-effort)",
                spec.argv[0], text_chars_total,
            )
            await _kill_and_reap(proc, spec.argv[0], tree)
            # Skip the rc-based error check below — we never got rc.
            # An empty error_message and have_output=True means good.
            if error_message is not None:
                raise _CliTransientError(
                    f"{spec.argv[0]} stream error: {error_message[:500]}"
                )
            _raise_if_cut_off(turns, spec)
            return _finalise_stream_content(aggregated, spec, usage_out, raw_result_lines)
        # No output AND no exit — genuinely stuck.
        await _kill_and_reap(proc, spec.argv[0], tree)
        stderr_b = b""
        if proc.stderr is not None:
            try:
                stderr_b = await asyncio.wait_for(proc.stderr.read(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
        raise _CliWedgeError(
            f"{spec.argv[0]} stdout closed but child didn't exit within "
            f"{post_eof_reap_timeout_s:g}s (no output collected): the CLI "
            f"closed its output without an answer and did not exit. If a resume "
            f"stops the same way, use a different provider for this node (e.g. "
            f"resume with `--config <codex_or_openai>.yaml`). stderr tail: "
            f"{stderr_b.decode('utf-8', 'replace')[-500:]}"
        )
    if rc != 0:
        # Even on non-zero exit, prefer the collected text over a
        # retry IF we got substantial output. Some CLIs emit a clean
        # stream then exit non-zero on a benign cleanup error
        # (atexit hooks, signal handler, etc.) — throwing the
        # response away there is strictly worse than logging the rc
        # and returning.
        if have_output:
            stderr_b = b""
            if proc.stderr is not None:
                try:
                    stderr_b = await asyncio.wait_for(proc.stderr.read(), timeout=2)
                except asyncio.TimeoutError:
                    pass
            _log.warning(
                "%s exited rc=%d but emitted %d text chars — returning "
                "result. stderr tail: %s",
                spec.argv[0], rc, text_chars_total,
                stderr_b.decode("utf-8", "replace")[-300:],
            )
            if error_message is not None:
                raise _CliTransientError(
                    f"{spec.argv[0]} stream error: {error_message[:500]}"
                )
            _raise_if_cut_off(turns, spec)
            return _finalise_stream_content(aggregated, spec, usage_out, raw_result_lines)
        stderr_b = b""
        if proc.stderr is not None:
            try:
                stderr_b = await asyncio.wait_for(proc.stderr.read(), timeout=2)
            except asyncio.TimeoutError:
                pass
        # The claude CLI writes nothing to stderr when the API is unreachable; the reason is in its stream.
        reason = f" (stream error: {error_message[:500]})" if error_message is not None else ""
        raise _CliTransientError(
            f"{argv[0]} exited rc={rc}{reason}: "
            f"{stderr_b.decode('utf-8', 'replace')[-500:]}"
        )
    if error_message is not None:
        # stream-json carried a fatal error event but the process exited
        # 0 anyway (claude_cli does this for some error classes). Treat
        # as transient so tenacity retries.
        raise _CliTransientError(
            f"{spec.argv[0]} stream error: {error_message[:500]}"
        )
    _raise_if_cut_off(turns, spec)
    return _finalise_stream_content(aggregated, spec, usage_out, raw_result_lines)


def _raise_if_cut_off(turns: list[dict[str, Any]], spec: Any) -> None:
    """A streamed CLI answer whose last turn stopped at the output limit (``stop_reason: max_tokens``) is cut off: the
    CLI resumes a turn cut that way, so a LAST one cut means it gave up. Never handed on as whole."""
    if turns and turns[-1].get("stop") == "max_tokens":
        raise ModelAnswerTruncated(
            f"the model's answer was cut off at its output limit ({spec.argv[0]}: stop_reason max_tokens): an "
            "incomplete answer is not used"
        )


def _finalise_stream_content(
    aggregated: list[str],
    spec: _CliSpec,
    usage_out: dict[str, Any] | None = None,
    raw_lines: list[str] | None = None,
) -> str:
    """Concatenate streamed text, apply per-spec extractor, and
    apply the rate-limit-pattern guard. Centralised so all three
    streaming return paths get the same protection — without this,
    the OPC quest's ``paper.md`` became literally the string
    ``"You've hit your session limit · resets 2:30am ..."`` because
    claude_cli delivered the rate-limit message as ``text_delta``
    events instead of an error envelope, and the engine wrote it to
    disk as if it were the assistant's answer."""
    content = "".join(aggregated)
    # Read usage from the RAW output, before the extractor narrows it to the
    # assistant text — the usage lives in the envelope the extractor discards.
    if usage_out is not None and spec.usage_extractor is not None:
        # The envelope, not the assembled prose — see the capture above.
        source = "\n".join(raw_lines) if raw_lines else content
        try:
            measured = spec.usage_extractor(source)
        except Exception:  # noqa: BLE001 - accounting must never fail a call
            measured = None
        if measured:
            usage_out.update(measured)
    if spec.output_extractor is not None and spec.output_via != "stream_json":
        content = spec.output_extractor(content)
    final = content.strip()
    gate_hit = _check_output_gates(final)
    if gate_hit is not None:
        gate_name, marker = gate_hit
        raise _CliTransientError(
            f"{spec.argv[0]} tripped output gate {gate_name!r} (matched: "
            f"{marker!r}; output: {final[:200]!r}). This would have been written to disk as the "
            f"artifact; treating as transient so tenacity retries (and "
            f"may escalate the model)."
        )
    return final


async def _kill_and_reap(
    proc: asyncio.subprocess.Process, name: str, tree: AsyncProcessTree | None = None,
) -> bool:
    """Kill the child and wait (5 s; 10 s with ``tree``) for the OS to reap it.
    Returns True iff the wait completed cleanly. Logs a warning otherwise — a
    leaked child becomes a zombie on POSIX or holds an OS handle on
    Windows until the parent exits.

    With ``tree`` (how ``_run_cli`` starts every CLI) the whole process tree
    goes: the CLI and every helper it started, waited for up to 10 s."""
    if tree is not None:
        if await tree.kill():
            return True
        _log.warning(
            "CLI %s did not reap within %.0fs after SIGKILL; "
            "process may be wedged in uninterruptible state",
            name, _TREE_KILL_WAIT_S,
        )
        return False
    try:
        proc.kill()
    except ProcessLookupError:
        return True  # already exited
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
        return True
    except asyncio.TimeoutError:
        _log.warning(
            "CLI %s did not reap within 5s after SIGKILL; "
            "process may be wedged in uninterruptible state",
            name,
        )
        return False


def _stream_turn_mark(raw: bytes) -> str | None:
    """``"start"`` for a stream-json line that opens a model turn, the stop
    reason for one that ends it (``"max_tokens"``, ``"end_turn"``, ``"tool_use"``),
    else None. Cheap: only lines naming those events are parsed."""
    if b'"message_start"' not in raw and b'"message_delta"' not in raw:
        return None
    try:
        event = (json.loads(raw.decode("utf-8", errors="replace")) or {}).get("event") or {}
    except (ValueError, AttributeError):
        return None
    if event.get("type") == "message_start":
        return "start"
    if event.get("type") == "message_delta":
        reason = (event.get("delta") or {}).get("stop_reason")
        return str(reason) if reason else None
    return None


def _stream_thinking(raw: bytes) -> list[tuple[str, str]]:
    """The reasoning text one stream-json line carries: ``("delta", text)`` for a streamed ``thinking_delta``, or
    ``("block", text)`` for a thinking block of a whole assistant message (used only when nothing was streamed), and
    ``("start", "")`` where a new streamed thinking block begins."""
    if b"thinking" not in raw:
        return []
    try:
        msg = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return []
    if not isinstance(msg, dict):
        return []
    if msg.get("type") == "stream_event":
        delta = (msg.get("event") or {}).get("delta") if isinstance(msg.get("event"), dict) else None
        event = msg.get("event") if isinstance(msg.get("event"), dict) else {}
        if isinstance(delta, dict) and delta.get("type") == "thinking_delta" and isinstance(delta.get("thinking"), str):
            return [("delta", delta["thinking"])]
        if event.get("type") == "content_block_start" and (event.get("content_block") or {}).get("type") == "thinking":
            return [("start", "")]
        return []
    if msg.get("type") == "assistant" and isinstance(msg.get("message"), dict):
        content = msg["message"].get("content")
        if isinstance(content, list):
            return [("block", b["thinking"]) for b in content
                    if isinstance(b, dict) and b.get("type") == "thinking" and isinstance(b.get("thinking"), str)]
    return []


def _parse_stream_json_line(raw: bytes) -> tuple[str, int, str | None, bool]:
    """Parse one line of Claude CLI's ``--output-format stream-json``
    output. Returns
    ``(text_delta, thinking_delta_token_count, error, is_result_envelope)``.

    - ``text_delta`` is the new assistant-visible text (empty for
      thinking-only events).
    - ``thinking_delta_token_count`` is a count, NOT the thinking text
      itself; we don't keep the thinking-text body because it's both
      large and not useful to downstream parsers.
    - ``error`` is a fatal-error message extracted from
      ``{"type":"error",...}`` events or from a ``result`` envelope
      marked ``is_error`` (or with an ``error*`` subtype), else None.
    - ``is_result_envelope`` is True when this line came from a
      ``{"type":"result", "result":"<full text>"}`` envelope. The CLI
      emits a stream of ``text_delta`` events AND a final result envelope;
      the envelope holds only the final turn's text, while the deltas span
      every turn, so the caller takes the envelope's text as the answer
      instead of the deltas (appending both would also double it).

    Tolerates non-JSON lines (the CLI sometimes emits status lines
    before stream-json events fully start) by returning all-empties.
    """
    try:
        msg = json.loads(raw.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError):
        return "", 0, None, False
    if not isinstance(msg, dict):
        return "", 0, None, False
    mtype = msg.get("type")
    # Stream event wrapper — actual content is nested under .event.
    if mtype == "stream_event":
        event = msg.get("event") or {}
        if not isinstance(event, dict):
            return "", 0, None, False
        if event.get("type") == "content_block_delta":
            delta = event.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                return str(delta.get("text", "")), 0, None, False
            if dtype == "thinking_delta":
                thinking = str(delta.get("thinking", ""))
                # Rough token estimate: 4 chars/token. Used only for
                # heartbeat progress, not billing.
                return "", max(1, len(thinking) // 4), None, False
        return "", 0, None, False
    # Some events carry the final assembled text in a different shape
    # (e.g. ``"type":"result"`` with ``"result":"<text>"``). Capture
    # it as a fallback so we don't return empty on quests that emit
    # the result event instead of streaming deltas. The ``is_result``
    # flag lets the caller skip appending when streamed deltas already
    # supplied the same content.
    if mtype == "result":
        result = msg.get("result")
        # A failed call still ends with a result envelope, its text the error: with the network down the claude CLI
        # ended `{"subtype":"success","is_error":true,"result":"API Error: Can't reach the API server ... (ENOTFOUND)"}`
        # and exited 1, and that sentence was taken as the model's answer by a dozen nodes of one quest.
        subtype = str(msg.get("subtype") or "")
        if msg.get("is_error") is True or subtype.startswith("error"):
            return "", 0, str(result or subtype or "the CLI reported an error"), False
        if isinstance(result, str):
            return result, 0, None, True
    if mtype == "error":
        err = msg.get("error") or msg.get("message") or "unknown stream error"
        return "", 0, str(err), False
    return "", 0, None, False


def _wait_for_openai_endpoint(port: int, *, timeout_s: int) -> None:
    """Both proxies expose `/v1/models`. Poll it for actual readiness
    rather than just a TCP bind — the FastAPI/Bun startup window between
    bind and serve has burned us before."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=2.0) as c:
                r = c.get(url)
                if r.status_code < 500:
                    return
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise TimeoutError(
        f"proxy /v1/models on 127.0.0.1:{port} did not respond within {timeout_s}s "
        f"(last error: {last_err!r})"
    )


def resolve_endpoint(
    provider: ProviderConfig,
    supervisor: ProxySupervisor | None = None,
) -> ResolvedEndpoint:
    """Synchronous resolution for direct providers; raises for proxy ones.

    Proxy providers must go through `resolve_endpoint_async` so the
    supervisor can spawn the child process and assign a port.
    """
    name = provider.name
    if name in _PROXY_PROVIDERS:
        raise RuntimeError(
            f"provider {name!r} requires async resolution via resolve_endpoint_async"
        )
    if name == "vscode_extension":
        # The FI VSCode extension is the parent process for chat-spawned
        # quests; it passes ``--vscode-bridge-port N`` and the port
        # lives in ``provider.extra["bridge_port"]``.
        #
        # For --serve / --tools subprocesses we don't have the
        # extension as a parent — instead they connect to the
        # extension's session-long PersistentBridge over a per-user
        # OS-managed IPC channel (Unix socket on POSIX, named pipe on
        # Windows). The path arrives via ``provider.extra["bridge_socket"]``
        # (``launch.py`` populates it from ``--vscode-bridge-socket``
        # or the canonical per-user default).
        socket_path = provider.extra.get("bridge_socket") or ""
        port = int(provider.extra.get("bridge_port", 0))
        if not socket_path and port <= 0:
            raise RuntimeError(
                "vscode_extension provider requires either "
                "extra['bridge_socket'] (--vscode-bridge-socket; what "
                "--serve / --tools and the extension's `@fi /update` / "
                "`@fi /generate` terminals use) or extra['bridge_port'] "
                "(--vscode-bridge-port; what the VSCode chat-spawn "
                "path uses). Are you launching FI from outside the "
                "extension's reach? Use copilot_cli for headless "
                "Copilot runs."
            )
        return ResolvedEndpoint(
            base_url="",
            model=provider.model or "(VSCode chat default)",
            api_key=_NO_KEY_SENTINEL,
            transport="vscode_bridge",
            provider_name=name,
            vscode_bridge_port=port,
            vscode_bridge_socket=socket_path,
            # The display string above is for logs only — it would be
            # an invalid family filter for selectChatModels. The real
            # override is empty unless the YAML pinned a model.
            vscode_model_override=provider.model or "",
            reasoning_effort=provider.reasoning_effort or "",
        )
    if name in _CLI_PROVIDERS:
        # CLI providers exec a local binary per chat call. No URL, no key
        # (the CLI uses its own OAuth keychain). `cli_model_override` is
        # the user's explicit YAML choice (empty → CLI keeps its own
        # default). `model` carries a human-readable display string for
        # the Engine's startup log so the line never goes blank.
        return ResolvedEndpoint(
            base_url="",
            model=provider.model or f"{name} (CLI default)",
            api_key=_NO_KEY_SENTINEL,
            transport="cli",
            provider_name=name,
            cli_spec=_CLI_SPECS[name],
            cli_model_override=provider.model or "",
            reasoning_effort=provider.reasoning_effort or "",
        )
    defaults = _DIRECT_DEFAULTS.get(name)
    if defaults is None:
        raise ValueError(f"unknown provider: {name!r}")
    api_key_env = provider.api_key_env or defaults["api_key_env"]
    api_key = os.environ.get(api_key_env, "") if api_key_env else _NO_KEY_SENTINEL
    return ResolvedEndpoint(
        base_url=provider.base_url or defaults["base_url"],
        model=provider.model or defaults["model"],
        api_key=api_key or _NO_KEY_SENTINEL,
        provider_name=name,
        reasoning_effort=provider.reasoning_effort or "",
        fixed_temperature=provider.fixed_temperature,
        extra_body=dict(provider.extra_body or {}),
        node_max_tokens=dict(provider.node_max_tokens or {}),
    )


def missing_api_key(provider: ProviderConfig) -> str | None:
    """What to tell a person whose provider needs an API key that is not in the environment, ``None`` when nothing is missing.

    Without this the request goes out with a placeholder key and the first sign of the problem is a ``401`` traceback from
    inside the first model call. A gateway of your own (a ``base_url`` that is not the provider's) that never named a key is
    left alone: it may not need one."""
    defaults = _DIRECT_DEFAULTS.get(provider.name)
    if defaults is None or provider.name in _PROXY_PROVIDERS or provider.name in _CLI_PROVIDERS:
        return None
    env = provider.api_key_env or defaults["api_key_env"]
    if not env or os.environ.get(env):
        return None
    own_gateway = bool(provider.base_url) and provider.base_url != defaults["base_url"]
    if own_gateway and not provider.api_key_env:
        return None
    return (
        f"provider {provider.name!r} needs an API key in the environment variable {env}, and it is not set. "
        f"Set it (PowerShell: $env:{env}=\"<your key>\"; bash: export {env}=<your key>) or put {env}=<your key> in a .env "
        f"file in this folder, or choose another provider (docs/PROVIDERS.md). Nothing was started."
    )


async def resolve_endpoint_async(
    provider: ProviderConfig,
    supervisor: ProxySupervisor,
) -> ResolvedEndpoint:
    if provider.name not in _PROXY_PROVIDERS:
        return resolve_endpoint(provider)
    handle = await supervisor.acquire(provider.name)
    api_key = os.environ.get(provider.api_key_env or "", "") or _NO_KEY_SENTINEL
    return ResolvedEndpoint(
        base_url=f"http://127.0.0.1:{handle.port}/v1",
        model=provider.model or "default",
        api_key=api_key,
        provider_name=provider.name,
        reasoning_effort=provider.reasoning_effort or "",
        fixed_temperature=provider.fixed_temperature,
        extra_body=dict(provider.extra_body or {}),
        node_max_tokens=dict(provider.node_max_tokens or {}),
    )


# Per-1k-token USD pricing for the model variants FI talks to. Used by
# the cost-instrumentation hook in `Engine` to convert each chat
# response's ``usage`` block into a $ figure for ``.fi/cost.jsonl``.
# Numbers come from each provider's published rate card; update
# opportunistically when rates change. Missing entries fall through
# to cost=None (we don't fabricate). All values are USD per 1000
# tokens; downstream multiplies by token count and divides by 1000.
#
# Entries are matched by SUBSTRING on the model name (case-insensitive)
# — handles dated variants like "claude-opus-4-7-20251201" matching
# "claude-opus-4-7". A key followed by "." or a digit is a different model
# ("gpt-5" is not "gpt-5.6-terra"), so it does not match. Order matters:
# more-specific (longer) keys are checked first.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI — gpt-5 family
    "gpt-5-mini": {"prompt_per_1k": 0.00025, "completion_per_1k": 0.002},
    "gpt-5": {"prompt_per_1k": 0.005, "completion_per_1k": 0.015},
    # OpenAI — gpt-4o family
    "gpt-4o-mini": {"prompt_per_1k": 0.00015, "completion_per_1k": 0.0006},
    "gpt-4o": {"prompt_per_1k": 0.0025, "completion_per_1k": 0.01},
    "gpt-4-turbo": {"prompt_per_1k": 0.01, "completion_per_1k": 0.03},
    # OpenAI — o1 reasoning family
    "o1-preview": {"prompt_per_1k": 0.015, "completion_per_1k": 0.06},
    "o1-mini": {"prompt_per_1k": 0.003, "completion_per_1k": 0.012},
    # Anthropic — Claude 4.x
    "claude-opus-4-7": {"prompt_per_1k": 0.015, "completion_per_1k": 0.075},
    "claude-sonnet-4-6": {"prompt_per_1k": 0.003, "completion_per_1k": 0.015},
    "claude-haiku-4-5": {"prompt_per_1k": 0.0008, "completion_per_1k": 0.004},
    # Anthropic — Claude 3.x (still in active use via copilot_cli etc.)
    "claude-3-5-sonnet": {"prompt_per_1k": 0.003, "completion_per_1k": 0.015},
    # Gemini
    "gemini-2.5-pro": {"prompt_per_1k": 0.00125, "completion_per_1k": 0.005},
    "gemini-2.5-flash": {"prompt_per_1k": 0.00015, "completion_per_1k": 0.0006},
    # Local — free
    "llama3.1:70b": {"prompt_per_1k": 0.0, "completion_per_1k": 0.0},
    "llama3.1:8b": {"prompt_per_1k": 0.0, "completion_per_1k": 0.0},
    "qwen2.5:32b": {"prompt_per_1k": 0.0, "completion_per_1k": 0.0},
}


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Multiply token counts by per-model rates from
    :data:`MODEL_PRICING`. Returns ``None`` when no pricing row matches
    — callers log usage but skip the cost field so partial data is
    obvious in the chart.

    A key names one model. The same characters followed by more version
    (``gpt-5`` inside ``gpt-5.6-terra``, ``gpt-5.5``) name a different model
    with its own price, so that row is not borrowed: the answer is ``None``,
    not another model's rate presented as this one's."""
    if not model:
        return None
    needle = model.lower()
    # Longest key first so "gpt-4o-mini" matches before "gpt-4o".
    for key in sorted(MODEL_PRICING.keys(), key=len, reverse=True):
        if re.search(re.escape(key.lower()) + r"(?![.\d])", needle):
            rates = MODEL_PRICING[key]
            return (
                prompt_tokens * rates["prompt_per_1k"] / 1000.0
                + completion_tokens * rates["completion_per_1k"] / 1000.0
            )
    return None


IO_DIRNAME = "io"
IO_ON_MARKER = "ON"


def set_model_call_archive(fi_dir: Path, on: bool) -> None:
    """Turn ``output.save_model_calls`` on or off for a quest: the marker in ``.fi/io/`` is what every writer below
    checks, so the engine's calls and the output generators' calls are kept alike. Files already kept stay."""
    marker = Path(fi_dir) / IO_DIRNAME / IO_ON_MARKER
    try:
        if on:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("output.save_model_calls: true\n", encoding="utf-8")
        else:
            marker.unlink(missing_ok=True)
    except OSError as e:
        _log.debug("[io] could not set the model-call archive: %r", e)


def _archive_model_call(fi_dir: Path, record: dict[str, Any], messages: Any, response: Any) -> None:
    """One call's whole prompt and answer, when the quest keeps them (``output.save_model_calls``)."""
    folder = Path(fi_dir) / IO_DIRNAME
    if messages is None or not (folder / IO_ON_MARKER).is_file():
        return
    from .audit_log import redact  # an API key pasted into a topic is not kept in the clear

    def _no_images(value: Any) -> Any:
        # A page image sent to the visual check is megabytes of base64: kept as its size, not its bytes.
        if isinstance(value, str) and value.startswith("data:image") and len(value) > 200:
            return f"[image, {len(value):,} characters]"
        if isinstance(value, list):
            return [_no_images(v) for v in value]
        if isinstance(value, dict):
            return {k: _no_images(v) for k, v in value.items()}
        return value

    messages = _no_images(messages)
    import uuid

    # A random part too: calls made at once (an ensemble) can share a clock tick.
    name = f"{time.time_ns()}-{uuid.uuid4().hex[:6]}-{re.sub(r'[^A-Za-z0-9_.-]+', '_', record['node'] or 'call')[:60]}.json"
    try:
        (folder / name).write_text(json.dumps({
            **record, "messages": redact(messages, whole=True), "response": redact(response, whole=True),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as e:
        _log.debug("[io] failed to keep a model call: %r", e)


def append_cost_row(
    fi_dir: Path, *, node: str, model: str | None, usage: dict[str, Any] | None,
    messages: Any = None, response: Any = None, ledger: bool = True, client: Any = None,
) -> None:
    """Append one model call to ``<fi_dir>/cost.jsonl`` as ``{ts, node,
    model, usage, cost_usd}``. The engine's nodes and the output generators
    (slides, poster, talk script, visual check) all write through here, so a
    quest's log counts every call it made. A call without ``usage`` still gets
    its row and is counted. Best-effort: a failed write is logged and never
    stops the caller.

    With ``messages`` and ``response``, a quest that keeps its model calls (``output.save_model_calls``) also gets the
    call's whole prompt and answer in ``.fi/io/``."""
    model = model or ""
    cost = None
    if usage:
        cost = estimate_cost_usd(
            model,
            int(usage.get("prompt_tokens", 0) or 0),
            int(usage.get("completion_tokens", 0) or 0),
        )
    record = {"ts": time.time(), "node": node, "model": model, "usage": usage, "cost_usd": cost}
    try:
        fi_dir.mkdir(parents=True, exist_ok=True)
        with (fi_dir / "cost.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        _log.debug("[cost] failed to write cost.jsonl: %r", e)
    _archive_model_call(fi_dir, record, messages, response)
    if ledger:
        # A call made outside the engine (an output generator) also goes in the quest's record of its model calls,
        # with who answered it as this task last recorded (LAST_CALL). The engine records its own calls itself.
        from core import attempt_records as _attempts

        try:
            row = _attempts.model_call_row(
                node=node, attempt=_attempts.next_attempt(fi_dir, node), served=LAST_CALL.get() or {"model": model},
                requested_model=None, reports_model=reports_model(client) if client is not None else False,
                messages=messages, response=response, usage=usage,
            )
            _attempts.append_model_call(fi_dir, fi_dir.parent.name, row)
        except Exception as e:  # noqa: BLE001 -- accounting never fails a call
            _log.debug("[cost] failed to record the model call: %r", e)


#: The provider and model that answered the most recent chat call made in the current asyncio task. A client's
#: ``last_provider``/``last_model`` attributes are shared by every call on it, so calls made at the same time (a review
#: panel's reviewers) overwrite each other's; a context variable is the task's own. The engine clears it before a call
#: and reads it after (``Engine._chat``).
LAST_CALL: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("fi_last_call", default=None)
#: The attempts of the current call that failed before it answered or gave up (a retry, a provider the fallback chain
#: passed over): ``{"provider", "model", "error", "fallback", "exc"}`` each. The engine sets a fresh list before a call
#: and writes one line per entry in the quest's record of its model calls; ``None`` (nobody listening) records nothing.
CALL_ATTEMPTS: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "fi_call_attempts", default=None)
#: The provider of the fallback chain being tried now (``{"provider", "fallback"}``), so a retry inside it is noted
#: under that provider, not whichever the client last named.
CALL_SLOT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("fi_call_slot", default=None)


class _ModelAnswerProblem(RuntimeError):
    """An answer the model gave that is not used: what the quest's pause says (``node``: the step, ``limit``: the
    output limit sent, ``usage``: what the call cost, ``provider`` / ``model``: who answered)."""

    fi_outcome = "error"

    def __init__(self, message: str, *, node: str = "", limit: int | None = None, usage: dict[str, Any] | None = None,
                 provider: str | None = None, model: str | None = None, finish_reason: str | None = None,
                 refused: bool = False) -> None:
        super().__init__(message)
        self.refused = refused  # the model turned down ``limit`` as an output size (HTTP 4xx), rather than cutting off at it
        self.node, self.limit, self.usage = node, limit, usage
        self.provider, self.model, self.finish_reason = provider, model, finish_reason


class ModelAnswerTruncated(_ModelAnswerProblem):
    """The model's answer was cut off at its output limit (``finish_reason: length``): an incomplete answer is never
    handed on as if it were whole. The quest stops for a person (a larger limit or another model for the step)."""

    fi_outcome = "truncated"


class ModelAnswerFiltered(_ModelAnswerProblem):
    """The provider withheld the model's answer with its content filter (``finish_reason: content_filter``). Asking the
    same provider again gives the same result, so it is not retried; the quest stops for a person."""

    fi_outcome = "content_filtered"


def outcome_of(exc: BaseException | None) -> str:
    """How a failed call is named in the record of calls: a truncated or filtered answer by what happened to it,
    anything else by its error's class."""
    if exc is None:
        return "error"
    return str(getattr(exc, "fi_outcome", "") or type(exc).__name__)


#: Finish reasons that mean the answer ended as the model meant it to.
_FINISHED_NORMALLY = frozenset({"stop", "end_turn", "eos", "stop_sequence", "tool_calls", "function_call"})
#: Finish reasons that mean the answer was cut off at the output limit (``max_tokens``: some OpenAI-compatible relays).
_CUT_OFF = frozenset({"length", "max_tokens"})
_FILTERED = frozenset({"content_filter"})
#: A call that set ``max_tokens`` and was cut off at it is asked once more with this many times the limit, capped.
_TRUNCATION_RETRY_GROWTH = 2
_TRUNCATION_RETRY_CAP = 65536


def _finish_reason(data: Any) -> str | None:
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError):
        return None
    reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    return str(reason) if reason else None


def _note_failed_attempt(provider: Any, model: Any, exc: BaseException | None, *, fallback: bool = False) -> None:
    attempts = CALL_ATTEMPTS.get()
    if attempts is None:
        return
    slot = CALL_SLOT.get() or {}
    attempts.append({"provider": slot.get("provider") or provider, "model": model,
                     "error": outcome_of(exc),
                     # What a failed attempt that did answer cost, and why its answer ended (a cut-off answer is paid).
                     **({"usage": exc.usage} if isinstance(getattr(exc, "usage", None), dict) else {}),
                     **({"finish_reason": exc.finish_reason} if getattr(exc, "finish_reason", None) else {}),
                     "fallback": bool(fallback or slot.get("fallback")), "exc": exc})


def reports_model(client: Any) -> bool:
    """Whether ``client``'s connection names the model that answered each call (an HTTP API, the claude CLI), so a call
    on it whose model went unnamed is a gap."""
    endpoint = getattr(client, "endpoint", None) or getattr(getattr(client, "_primary", None), "endpoint", None)
    if endpoint is None:
        return False
    return getattr(endpoint, "transport", "") == "http" or getattr(endpoint, "provider_name", "") == "claude_cli"

class LLMClient:
    """Thin async wrapper that speaks OpenAI Chat Completions.

    Built on `httpx.AsyncClient` rather than the openai SDK directly so
    multiple Engines in one process can share a single connection pool
    cleanly. The request shape is OpenAI-standard.

    Each call updates ``self.last_usage`` (or sets it to ``None``) with
    the response's token-usage info when the upstream returned one.
    The Engine reads this attribute after each chat call to write a
    line to ``<quest_root>/.fi/cost.jsonl``. CLI transports
    (claude_cli / codex_cli / copilot_cli / gemini_cli) and the
    vscode_bridge don't return structured usage today; their calls
    leave ``last_usage = None`` and the chart shows runtime-only.
    """

    def __init__(
        self,
        endpoint: ResolvedEndpoint,
        *,
        http: httpx.AsyncClient | None = None,
        timeout_s: float = 120.0,
        cli_timeout_s: float = 300.0,
        cli_inactivity_timeout_s: float | None = 180.0,
        node_cli_timeout_s: dict[str, float] | None = None,
        node_http_timeout_s: dict[str, float] | None = None,
        node_model_fallbacks: dict[str, str] | None = None,
        max_prompt_chars: int = 0,
        heartbeat_cb: Callable[[dict[str, Any]], None] | None = None,
        run_log: "logging.Logger | None" = None,
    ) -> None:
        self.endpoint = endpoint
        # The quest's own log (run.log): a failed call and the retry after it are written there too.
        self._run_log = run_log
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(timeout=timeout_s)
        # Base HTTP read-timeout (seconds) and per-node overrides for the
        # OpenAI-compatible transport. Reasoning-heavy nodes (implement/write)
        # routinely outrun a flat 120 s on slow/local models; the per-node map
        # gives them headroom while cheap nodes keep the tight base that
        # catches a hung server fast. Lookup misses fall back to the base.
        self._http_timeout_s = timeout_s
        self._node_http_timeout_s = node_http_timeout_s or {}
        # Last-resort prompt-size guard (total characters across messages).
        # 0 disables. Prevents a runaway prompt from blowing the context
        # window into a hard 400 / stall. See ``_trim_messages``.
        self._max_prompt_chars = max(0, int(max_prompt_chars))
        # CLI providers (claude_cli/codex_cli/copilot_cli/gemini_cli)
        # use this wall-clock cap per chat call; a stuck child gets
        # killed and tenacity retries. Defaults to 5 minutes — longer
        # than the typical 10–90 s per call but bounded so concurrent
        # fleet contention can't hang the whole quest indefinitely.
        self._cli_timeout_s = cli_timeout_s
        # Inactivity-watchdog soft timeout. None disables and lets only
        # the total ceiling apply (compat shim for callers that haven't
        # opted into streaming yet). See ProviderConfig docstring.
        self._cli_inactivity_timeout_s = cli_inactivity_timeout_s
        # Per-node ``cli_timeout_s`` overrides; map node-name → seconds.
        # Lookup misses fall back to ``self._cli_timeout_s``. Used by
        # ``_chat_cli`` when ``node`` is passed in.
        self._node_cli_timeout_s = node_cli_timeout_s or {}
        # Per-node model escalation map — empty unless the user set
        # ``provider.node_model_fallbacks``; FI never picks a model on
        # the user's behalf. Used by ``_chat_cli`` on retry 2+ when the
        # primary model failed transiently. It exists for the
        # paralysis-thinking a smaller Claude model can fall into on
        # long code-gen prompts (extended thinking spins forever
        # without producing any text): retrying on a stronger model
        # escapes it. The string is sent as-is to whichever CLI is
        # active, so the user must name a model that CLI accepts.
        # ``node_models`` (above) sets the FIRST-ATTEMPT model; this
        # dict sets the SECOND-ATTEMPT-AND-LATER model.
        self._node_model_fallbacks = node_model_fallbacks or {}
        # Optional periodic progress callback (Engine wires this to its
        # run.log heartbeat logger). Receives a dict each ~1 s during
        # CLI streaming; see ``_run_cli`` for the payload shape.
        self._heartbeat_cb = heartbeat_cb
        # Lazily-built VSCode-extension bridge client. The bridge
        # connection is shared across every chat call from this
        # LLMClient instance.
        self._bridge: Any | None = None
        # Last chat call's usage info (prompt_tokens /
        # completion_tokens / total_tokens) parsed from the response
        # body, or ``None`` when the transport doesn't return one.
        # Engine reads this after each chat() call to write a
        # cost-tracking row. Reset to None at the top of every
        # chat() so a previous call's usage can't leak into a
        # subsequent one if the new transport doesn't populate it.
        self.last_usage: dict[str, int] | None = None
        # Same idea for the model that ACTUALLY ran — for proxy
        # transports (copilot_cli etc.) the requested model and the
        # routed model can differ. Engine uses last_model to look up
        # the right pricing row when writing cost.jsonl.
        self.last_model: str | None = None
        # Constant for a plain LLMClient (one endpoint, one provider for its whole life) — present so the Engine's
        # audit trace can read ``last_provider`` uniformly regardless of whether ``self._client`` is this class or
        # FallbackLLMClient (whose ``last_provider`` genuinely varies call to call).
        self.last_provider: str | None = endpoint.provider_name or endpoint.transport

    async def aclose(self) -> None:
        if self._bridge is not None:
            try:
                await self._bridge.aclose()
            except Exception:
                pass
            self._bridge = None
        if self._owns_http:
            await self._http.aclose()

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
        model: str | None = None,
        node: str = "",
    ) -> str:
        """Run one chat completion. ``model`` is an optional per-call
        override for per-node model routing. When provided and
        non-empty, it replaces the endpoint's default model for THIS
        call only — useful for sending different nodes through different
        models on the same provider (most relevant on Copilot, where
        all the model variants share one CLI and one premium-request
        budget). Falls back to ``self.endpoint.model`` when omitted.

        ``node`` is the FI engine node name (``"ideate"``, ``"implement"``,
        …); only used by the vscode_bridge transport so the extension
        can tag chat-panel progress messages with the right node name.
        Other transports ignore it for routing but DO attach it to any
        exception via ``Exception.add_note`` so error tracebacks show
        which node + which provider failed instead of just a raw
        httpx / CLI stderr stack.
        """
        try:
            # Gate the dispatch on the process-wide concurrency cap
            # (``FI_MAX_CONCURRENT_LLM_CALLS``; no-op when unset) so a fleet of
            # quests can't burst past provider rate limits. The slot is held
            # only for this one dispatch and released on return, so retries /
            # fallbacks re-queue rather than deadlock.
            gate = _llm_call_slot()
            # The HTTP path gives the slot back while it waits out a provider outage (_http_retry_sleep), so whether
            # this dispatch still holds it at the end is tracked, not assumed.
            box: dict[str, Any] | None = None
            if isinstance(gate, asyncio.Semaphore):
                await gate.acquire()
                box = {"sem": gate, "held": True}
            held = _HELD_CALL_SLOT.set(box)
            try:
                text = await self._chat_impl(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra=extra,
                    model=model,
                    node=node,
                )
            finally:
                _HELD_CALL_SLOT.reset(held)
                if box is not None and box["held"]:
                    box["held"] = False
                    gate.release()
            # This call's own token counts, in this task's record of it: the client's ``last_usage`` is shared by
            # calls running at the same time.
            LAST_CALL.set({**(LAST_CALL.get() or {}), "usage": self.last_usage})
            return text
        except Exception as e:
            # ``except Exception`` excludes ``asyncio.CancelledError``
            # (a BaseException subclass since Python 3.8) — see the
            # cancellation-contract comment in _chat_impl below for why
            # that matters. CancelledError must propagate clean and
            # un-noted; everything else gets a "[FI] provider=…, node=…,
            # model=…" note so the user knows what was running when it
            # blew up. ``add_note`` is non-invasive (no exception
            # wrapping, no type change), keeps the original traceback
            # intact, and appears in standard `traceback.print_exc`
            # output on Python >=3.11 (our minimum).
            try:
                e.add_note(self._error_note(node=node, model=model))
                auth_note = self._auth_error_note(e)
                if auth_note:
                    e.add_note(auth_note)
            except AttributeError:  # pragma: no cover — needs Py<3.11
                pass
            raise

    def _error_note(self, *, node: str, model: str | None) -> str:
        """Format the FI-context note attached to LLM call exceptions.

        Includes the provider name, transport class, effective model,
        and (when set) the engine node name. Single line, prefixed
        ``[FI]`` so it's grep-able in error reports."""
        provider = self.endpoint.provider_name or self.endpoint.transport
        effective_model = model or self.endpoint.model
        parts = [
            f"provider={provider}",
            f"transport={self.endpoint.transport}",
            f"model={effective_model}",
        ]
        if node:
            parts.append(f"node={node}")
        return "[FI] " + ", ".join(parts)

    def _auth_error_note(self, exc: BaseException) -> str | None:
        """A plain-language guess at an HTTP 401/403/402, or ``None`` for anything else.

        ``missing_api_key`` catches the key being unset entirely, before a single request goes
        out. This is the other half: the key IS set, but the provider rejected it (revoked,
        wrong account, no access to this model, quota/credit exhausted) — the exact case that
        used to surface as a bare, unexplained ``HTTPStatusError``. Same status-code set
        ``_is_fatal_provider_error`` treats as fatal (opens the circuit breaker immediately, no
        retry budget wasted); this is the message half of that same classification."""
        if not isinstance(exc, httpx.HTTPStatusError):
            return None
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status not in (401, 403, 402):
            return None
        provider = self.endpoint.provider_name or self.endpoint.transport
        if status == 402:
            return (
                f"[FI] HTTP {status}: {provider} reports payment/quota exhausted for this key — "
                f"check billing/credits on the provider's dashboard, or set a different key."
            )
        return (
            f"[FI] HTTP {status}: {provider} did not accept the API key FI sent. Check the "
            f"environment variable it read the key from (your config's provider.api_key_env, or "
            f"that provider's own default — see docs/PROVIDERS.md) for a revoked, wrong-account, "
            f"or model-unauthorized key."
        )

    def _retry_note(self, node: str | None, where: str, total: "int | Callable[[Any], int]",
                    model: Callable[[Any], str | None] | str | None = None) -> Callable[[Any], None]:
        """tenacity ``before_sleep``: the failed call and the coming retry, in FI's log and the quest's run.log."""
        def note(rs: Any) -> None:
            try:
                exc = rs.outcome.exception() if getattr(rs, "outcome", None) else None
                _note_failed_attempt(self.last_provider, (model(rs) if callable(model) else model) or self.last_model,
                                     exc)
            except Exception:  # noqa: BLE001 -- a record never stops a retry
                pass
            try:
                line = _retry_line(node, where, rs, total(rs) if callable(total) else total,
                                   model=model(rs) if callable(model) else model)
                _log.warning("%s", line)
                if self._run_log is not None:
                    self._run_log.warning("%s", line)
            except Exception:  # noqa: BLE001 -- a log line never stops a retry
                pass
        return note

    def _trim_messages(
        self, messages: list[dict[str, str]], *, node: str = "",
    ) -> list[dict[str, str]]:
        """Last-resort guard against a runaway prompt. When ``max_prompt_chars``
        is set and the total message size exceeds it, shrink the prompt so the
        TOTAL is guaranteed to land at or under the cap. No-op when the guard is
        disabled (0) or the prompt already fits — so normal calls pay only one
        ``sum`` and are otherwise untouched. Returns a new list; never mutates
        the caller's messages.

        Common case (one dominant message — a giant RESULT_JSON / literature
        dump): the largest message is trimmed in the MIDDLE, keeping head + tail
        (where the task framing and the trailing instruction live). The kept
        length is derived from the largest message's *budget* (cap minus the
        other messages), so the result total is exactly the cap — never over,
        even when the cap is smaller than the trim marker (then the marker is
        dropped and the content is hard-clamped to the budget).

        Pathological case (the OTHER messages alone already exceed the cap):
        trimming one message can't help, so every message is hard-clamped to its
        proportional share of the cap. Loses the head+tail nicety, but still
        guarantees the bound.

        Only text counts toward the cap and only text is trimmed; the image
        parts of a content-part list ride along untouched."""
        cap = self._max_prompt_chars
        if cap <= 0:
            return messages
        lengths = [len(_content_text(m.get("content", ""))) for m in messages]
        total = sum(lengths)
        if total <= cap:
            return messages
        trimmed = [dict(m) for m in messages]
        idx = max(range(len(trimmed)), key=lambda i: lengths[i])
        others = total - lengths[idx]
        budget = cap - others  # chars the largest message may keep
        marker = f"\n\n... [FI: prompt trimmed to fit the {cap}-char cap] ...\n\n"
        if budget > 0:
            content = _content_text(trimmed[idx].get("content", ""))
            if budget > len(marker) + 2:
                keep = budget - len(marker)
                head = keep // 2
                tail = keep - head
                new_content = content[:head] + marker + (
                    content[-tail:] if tail else ""
                )
            else:
                # Budget too tight to fit the marker — hard-clamp to the budget.
                new_content = content[:budget]
            trimmed[idx]["content"] = _with_text(trimmed[idx].get("content", ""), new_content)
        else:
            # Even excluding the largest message the prompt is over the cap;
            # clamp every message to its proportional share (floors sum <= cap).
            for i, m in enumerate(trimmed):
                share = (cap * lengths[i]) // total if total else 0
                m["content"] = _with_text(m.get("content", ""), _content_text(m.get("content", ""))[:share])
        _log.warning(
            "[prompt-trim] node=%s prompt %d chars > cap %d — trimmed to fit "
            "(%d chars now)",
            node or "?", total, cap,
            sum(len(_content_text(m.get("content", ""))) for m in trimmed),
        )
        return trimmed

    async def _chat_impl(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int | None,
        extra: dict[str, Any] | None,
        model: str | None,
        node: str,
    ) -> str:
        """The actual chat dispatch, split out from ``chat`` so the
        error-context note in ``chat`` wraps a single call site."""
        # Reset cost-tracking state at the top of every
        # call so a transport that DOESN'T return usage (CLI / bridge)
        # leaves last_usage = None until the post-call estimator fills
        # it in below. The Engine cost-logger reads last_usage after
        # the call returns; today every call carries at least an
        # estimated row so the cost chart shows real numbers instead
        # of empty bars for CLI / bridge transports.
        self.last_usage = None
        self.last_model = (model or self.endpoint.model)
        LAST_CALL.set({"provider": self.last_provider, "model": self.last_model, "reported": False})
        note_thinking("")  # an earlier attempt's or provider's reasoning is not this attempt's
        # Last-resort prompt-size guard (all transports) — a runaway prompt
        # otherwise blows the context window into a hard 400 / stall.
        messages = self._trim_messages(messages, node=node)
        if self.endpoint.transport == "cli":
            text = await self._chat_cli(
                messages, model_override=model, node=node,
            )
            self._fill_usage_estimate_if_missing(messages, text)
            return text
        if self.endpoint.transport == "vscode_bridge":
            if self.endpoint.reasoning_effort:
                _warn_reasoning_effort_once(
                    self.endpoint.provider_name or "vscode_extension",
                    self.endpoint.reasoning_effort,
                    "the vscode.lm bridge has no reasoning-effort setting FI "
                    "can pass, so the chat model runs at its own default",
                )
            text = await self._chat_vscode_bridge(
                messages, model_override=model, temperature=temperature,
                node=node,
            )
            from .vscode_bridge import (
                EXTENSION_OLDER_NOTE, LAST_BRIDGE_EMPTY_PARTS, LAST_BRIDGE_PROTOCOL, LAST_BRIDGE_THINKING,
                LAST_BRIDGE_THINKING_DECLINED, LAST_BRIDGE_USAGE, LAST_SERVED, extension_is_older, is_router_alias,
            )

            note_thinking(LAST_BRIDGE_THINKING.get() or "")
            note_declined(LAST_BRIDGE_THINKING_DECLINED.get() or "")
            note_empty_parts(LAST_BRIDGE_EMPTY_PARTS.get())
            if extension_is_older(LAST_BRIDGE_PROTOCOL.get()):
                # The engine says it once in run.log and on the console; a call made outside an engine step says it here.
                note_extension_older()
                if not has_holder() and not _EXTENSION_OLDER_SAID:
                    _EXTENSION_OLDER_SAID.append(True)
                    _log.warning("%s", EXTENSION_OLDER_NOTE)
            served = LAST_SERVED.get()
            if served and served.get("id") and not is_router_alias(served):
                # The extension named the chat model it selected and sent this very call to (as VS Code reports it):
                # known as reported, not assumed from the hint. The provider stays FI's own name; the model's vendor
                # rides beside it. A router alias ("auto") names no model, so it is left unreported.
                self.last_model = served["id"]
                LAST_CALL.set({"provider": self.last_provider, "model": served["id"], "reported": True,
                               **({"vendor": served["vendor"]} if served.get("vendor") else {}),
                               **({"family": served["family"]} if served.get("family") else {})})
            # The extension counts with the model's own tokenizer when it
            # can, which beats the char/4 estimate — take it, and let the
            # estimator fill in only when it could not. This call's own counts (per request, in this task).
            measured = LAST_BRIDGE_USAGE.get()
            if measured:
                self.last_usage = measured
            self._fill_usage_estimate_if_missing(messages, text)
            return text
        body: dict[str, Any] = {
            "model": (model or self.endpoint.model),
            "messages": messages,
            # A model that accepts one temperature gets it, whatever the node would have asked for.
            "temperature": (
                self.endpoint.fixed_temperature if self.endpoint.fixed_temperature is not None else temperature
            ),
        }
        if max_tokens is None:
            max_tokens = node_output_limit(self.endpoint.node_max_tokens, node)
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        # ``provider.reasoning_effort`` — absent from the body when unset, so
        # an unset config sends exactly what it always did. A per-call
        # ``extra`` still wins.
        reasoning_effort = _http_reasoning_effort(self.endpoint)
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        if self.endpoint.extra_body:
            body.update(self.endpoint.extra_body)
        if extra:
            body.update(extra)
        streams = _http_streams(self.endpoint)
        if "stream" in body:
            # FI decides whether a call streams (only Moonshot's, see _STREAMED_HOSTS) and reads the answer to match; a
            # `stream` of your own would leave the answer unreadable (a streamed reply parsed as one JSON document).
            raise ValueError(
                "provider.extra_body (or a per-call extra) sets `stream`: FI decides that itself -- it streams calls to "
                "Moonshot's API and sends every other request whole. Remove `stream` from extra_body."
            )
        url = self.endpoint.base_url.rstrip("/") + "/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        # Ollama (and vLLM with auth disabled) treat the OpenAI-compat
        # endpoint as keyless. Sending `Authorization: Bearer not-needed`
        # is harmless against most servers but Ollama's strict-mode reverse
        # proxies have rejected it. Only attach when we actually have a key.
        if self.endpoint.api_key and self.endpoint.api_key != _NO_KEY_SENTINEL:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"
        # Retry on transient upstream failures (5xx from cloud-routed Ollama
        # models, brief network blips). 4xx errors are not retried.
        #
        # Cancellation contract: ``asyncio.CancelledError`` is a
        # BaseException subclass (Python 3.8+), NOT Exception, so
        # tenacity's ``retry_if_exception_type(...)`` predicate cannot
        # match it — cancellation propagates straight through the retry
        # wrapper. httpx >= 0.27 then propagates that cancellation
        # through its anyio-based transport so the in-flight POST
        # unwinds promptly. ``test_chat_propagates_cancellation_promptly``
        # in tests/ verifies the asyncio await-chain unwinds in <1 s on
        # cancel (the test uses ``httpx.MockTransport``, which is
        # in-memory and never opens a real socket — socket-level
        # cancellation is a downstream httpx/anyio contract we trust
        # the upstream test suites to enforce).
        #
        # The two dangerous patterns that would silently break this:
        #   1. Wrapping the retry block in ``except BaseException``
        #      (``except Exception`` is fine — that's the rule above;
        #      it's also what the outer ``chat`` wrapper above uses
        #      for its error-context note attach).
        #   2. Adding ``asyncio.CancelledError`` to the
        #      ``retry_if_exception_type`` tuple.
        # Either would let tenacity catch the cancel and retry the
        # POST instead of letting it propagate. Re-run the two
        # cancellation tests in tests/test_provider.py if you touch
        # this retry config.
        think_off = {"on": False}  # a loop (or a thinking that used up the answer) once: this call asks no more for reasoning

        async def send(request_body: dict[str, Any]) -> dict[str, Any]:
            # Ollama only hands a model's reasoning back through its own API, so when the reasoning will be kept
            # (``output.save_thinking``) the call goes there; a model that cannot think (or a server that does not
            # know the call) is remembered and sent to the OpenAI-compatible endpoint as before.
            native_url = _ollama_native(self.endpoint.base_url) if self.endpoint.provider_name == "ollama" else None
            refused = self.__dict__.setdefault("_ollama_not_thinking", set())
            # FI's own temperature is not sent with a request to think (the model's recommended sampling is used
            # then), unless the person set one: ``provider.fixed_temperature`` or a ``temperature`` of their own.
            explicit = (self.endpoint.fixed_temperature is not None or "temperature" in (self.endpoint.extra_body or {})
                        or "temperature" in (extra or {}))
            native = (_ollama_native_body(request_body, keep_temperature=explicit)
                      if native_url and thinking_wanted() and not think_off["on"]
                      and request_body.get("model") not in refused else None)
            if native is None or native_url is None:
                return await send_to(url, request_body)
            try:
                note_sampling(await self._ollama_sampling_note(native_url, request_body, explicit))
            except Exception:  # noqa: BLE001 -- a note about sampling never stops the call
                pass

            async def without_reasoning() -> dict[str, Any]:
                think_off["on"] = True
                return await send_to(url, request_body)

            try:
                reply = _ollama_as_openai(await send_to(native_url, native, native=True))
            except ThinkingLoop as loop:
                note_loop(loop.kind, loop.block, loop.repeats, loop.chars)
                _log.info("[%s] %s; asking again without reasoning", node or "chat", loop)
                return await without_reasoning()
            except httpx.HTTPStatusError as e:
                sc = getattr(getattr(e, "response", None), "status_code", None)
                if sc not in (400, 404, 405, 501):
                    raise
                try:
                    said = str(getattr(getattr(e, "response", None), "text", "") or "")
                except Exception:  # noqa: BLE001 -- a body that cannot be read only loses the detail
                    said = ""
                if "think" in said.lower() or sc in (404, 405, 501):
                    refused.add(request_body.get("model"))  # not asked again; any other 400 is asked afresh next time
                _log.info("[%s] Ollama's own chat call (which returns the model's reasoning) was not accepted for "
                          "%s (HTTP %s); using the OpenAI-compatible call, whose answers carry no reasoning",
                          node or "chat", request_body.get("model"), sc)
                return await send_to(url, request_body)
            message = reply["choices"][0]["message"]
            if _finish_reason(reply) == "length" and message.get("reasoning_content") and not str(
                    message.get("content") or "").strip():
                # The reasoning used up all the room there was and no answer came: paid for, noted, asked once more
                # without reasoning (asking again with it would spend the same).
                spent = len(message["reasoning_content"])
                _note_failed_attempt(self.last_provider, reply.get("model") or request_body.get("model"),
                                     cut_off(reply, request_body.get("max_tokens")))
                note_loop("length", 0, 0, spent)
                _log.info("[%s] the model's reasoning used up the whole answer space (%d characters) without an answer; "
                          "asking again without reasoning", node or "chat", spent)
                return await without_reasoning()
            return reply

        async def send_to(call_url: str, request_body: dict[str, Any], *, native: bool = False) -> dict[str, Any]:
            data: dict[str, Any] = {}
            tries = timeouts = 0  # attempts made, and how many of them ended in a timeout
            async for attempt in AsyncRetrying(
                # Six attempts over about 3-4.5 minutes once the provider's
                # server is down or busy (5xx, 429 rate limit; a sane
                # Retry-After is honoured), four with short waits for a
                # dropped connection or read timeout, or when a fallback
                # provider can take the call; a 4xx and a used-up quota are
                # not retried (see _http_retry_wait / _stop). Every wait is
                # jittered: a deterministic schedule synchronises N
                # concurrent quests in a --fleet that all hit the same
                # upstream blip, so they would all retry at the same instant
                # and re-clog the upstream. The concurrency slot is given
                # back while waiting (_http_retry_sleep).
                stop=_http_retry_stop,
                wait=_http_retry_wait,
                sleep=_http_retry_sleep,
                retry=retry_if_exception(_retry_http_error),
                before_sleep=self._retry_note(
                    node, f"{self.endpoint.provider_name or 'provider'} over HTTP", _http_attempts,
                    model or self.endpoint.model),
                reraise=True,
            ):
                with attempt:
                    # Per-node HTTP read-timeout: heavy nodes (implement/write) get
                    # headroom, cheap nodes keep the tight base that catches a hung
                    # server fast. Miss → base client timeout.
                    http_timeout = node_budget(
                        self._node_http_timeout_s, node, self._http_timeout_s,
                    )
                    tries += 1
                    try:
                        if native:
                            data = await _post_ollama_streamed(self._http, call_url, request_body, headers, http_timeout)
                        elif streams:
                            data = await _post_streamed(self._http, call_url, request_body, headers, http_timeout)
                        else:
                            r = await self._http.post(
                                call_url, json=request_body, headers=headers, timeout=http_timeout,
                            )
                            # Raise for any error status; the retry predicate
                            # (_retry_http_error) retries only 5xx / 429, letting a 4xx
                            # (bad key, quota, content policy, oversized body) surface
                            # immediately instead of burning the backoff budget.
                            r.raise_for_status()
                            data = r.json()
                    except httpx.TimeoutException as e:
                        # Said on the error for whoever sorts the failure (core/crash_kind.py): when every try ended
                        # like this, retrying later will not help; the step needs more time (or a faster model).
                        timeouts += 1
                        e.fi_timeouts = {"tries": tries, "all": timeouts == tries, "limit_s": http_timeout}  # type: ignore[attr-defined]
                        raise
            return data

        def usage_of(reply: dict[str, Any]) -> dict[str, int] | None:
            u = reply.get("usage") or {}
            if not (u and isinstance(u, dict)):
                return None
            return {
                "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(u.get("completion_tokens", 0) or 0),
                "total_tokens": int(u.get("total_tokens", 0) or (u.get("prompt_tokens", 0) or 0)
                                    + (u.get("completion_tokens", 0) or 0)),
            }

        def problem(kind: type, reply: dict[str, Any], why: str, limit: int | None,
                    refused: bool = False) -> _ModelAnswerProblem:
            return kind(why, node=node, limit=limit, usage=usage_of(reply), provider=self.last_provider,
                        model=reply.get("model") if isinstance(reply.get("model"), str) else (model or self.endpoint.model),
                        finish_reason=_finish_reason(reply), refused=refused)

        def refused_limit(e: httpx.HTTPStatusError, asked: int, prefix: str) -> _ModelAnswerProblem | None:
            """A 4xx that is about the output size asked for (not a key, quota or rate problem): named for a person."""
            resp = getattr(e, "response", None)
            sc = getattr(resp, "status_code", None)
            if not (isinstance(sc, int) and sc in (400, 413, 422)):
                return None
            try:
                said = str(getattr(resp, "text", "") or "")[:600]
            except Exception:  # noqa: BLE001 -- a body that cannot be read only loses the detail
                said = ""
            if not re.search(r"max[_ ]?(completion[_ ])?tokens|output|context|too large|exceed|limit", said, re.I):
                return None
            return problem(ModelAnswerTruncated, {}, (
                f"{prefix}the model refused an output limit of {asked} tokens (HTTP {sc}: {' '.join(said.split())[:200]})"),
                asked, refused=True)

        def cut_off(reply: dict[str, Any], limit: int | None) -> _ModelAnswerProblem:
            return problem(ModelAnswerTruncated, reply,
                           "the model's answer was cut off at its output limit"
                           + (f" ({limit} tokens)" if limit else "") + ": an incomplete answer is not used", limit)

        asked = body.get("max_tokens")
        try:
            data = await send(body)
        except httpx.HTTPStatusError as e:
            refused = refused_limit(e, asked, "") if isinstance(asked, int) and not isinstance(asked, bool) else None
            if refused is not None:
                raise refused from e
            raise
        finish = _finish_reason(data)
        limit = body.get("max_tokens")
        if finish in _CUT_OFF and isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            # Cut off at the limit this call set: asked once more with room to finish (bounded), never handed on cut.
            bigger = min(limit * _TRUNCATION_RETRY_GROWTH, _TRUNCATION_RETRY_CAP)
            if bigger > limit:
                _log.warning("[%s] the model's answer was cut off at max_tokens=%d; asking once more with %d",
                             node or "chat", limit, bigger)
                first = cut_off(data, limit)
                _note_failed_attempt(self.last_provider, first.model, first)
                body = {**body, "max_tokens": bigger}
                try:
                    data = await send(body)
                except httpx.HTTPStatusError as e:
                    sc = getattr(getattr(e, "response", None), "status_code", None)
                    if isinstance(sc, int) and 400 <= sc < 500 and sc != 429:
                        # The larger limit is more than this model allows: the answer stays cut off at the first one.
                        raise problem(ModelAnswerTruncated, {}, (
                            f"the model's answer was cut off at its output limit ({limit} tokens), and the model "
                            f"refused a larger one ({bigger} tokens: HTTP {sc})"), bigger, refused=True) from e
                    raise
                finish = _finish_reason(data)
        # Capture token usage when the upstream returned
        # one. OpenAI-compatible servers (openai / codex / gemini /
        # ollama recent versions) include ``usage`` at the response
        # top level; older Ollama omits it. Missing → leave
        # last_usage = None so the cost-logger writes "unknown".
        measured = usage_of(data)
        if measured is not None:
            self.last_usage = measured
        # Some providers (notably some Copilot proxies) echo the
        # actual-routed model in the response — record it so the
        # cost chart attributes the call to the model that ran,
        # not the one we asked for.
        if isinstance(data.get("model"), str):
            self.last_model = data["model"]
            LAST_CALL.set({"provider": self.last_provider, "model": data["model"], "reported": True})
        # Why the answer ended, and what it cost, in this call's own record (and so in the quest's record of its calls).
        LAST_CALL.set({**(LAST_CALL.get() or {}), "finish_reason": finish, "usage": self.last_usage})
        note_thinking(_reasoning_of(data))
        if finish in _CUT_OFF:
            raise cut_off(data, body.get("max_tokens"))
        if finish in _FILTERED:
            raise problem(ModelAnswerFiltered, data,
                          "the provider withheld the model's answer with its content filter; asking it again would "
                          "give the same result", body.get("max_tokens"))
        if finish is not None and finish not in _FINISHED_NORMALLY:
            _log.info("[%s] the model's answer ended with finish_reason %r (taken as finished)", node or "chat", finish)
        text = data["choices"][0]["message"]["content"]
        # Older Ollama versions omit ``usage`` from the response. Fall
        # through to char-based estimation so the cost.jsonl row still
        # carries real numbers instead of nulls.
        self._fill_usage_estimate_if_missing(messages, text)
        return text

    async def _ollama_sampling_note(self, native_url: str, request_body: dict[str, Any], explicit: bool) -> str:
        """One plain phrase for which sampling a request to think uses: the person's own temperature, or the model's
        recommended one (its ``parameters`` as Ollama's ``/api/show`` lists them, asked once per model; "the model's own
        defaults" when they cannot be read)."""
        model = str(request_body.get("model") or "")
        if explicit:
            return f"{model}: FI sends temperature {request_body.get('temperature')} (set in the provider settings)"
        cache = self.__dict__.setdefault("_ollama_defaults", {})
        if model not in cache:
            cache[model] = ""
            try:
                r = await self._http.post(native_url[: -len("/chat")] + "/show", json={"model": model},
                                          headers={"Content-Type": "application/json"}, timeout=10)
                r.raise_for_status()
                params = r.json().get("parameters")
                if isinstance(params, str):
                    cache[model] = ", ".join(" ".join(x.split()) for x in params.splitlines() if x.strip())[:200]
            except Exception:  # noqa: BLE001 -- the model's defaults not being readable only changes the wording
                pass
        got = cache[model]
        return (f"{model}: reasoning is on, so FI sent no temperature of its own and the model's own defaults are used"
                + (f" ({got})" if got else ""))

    # ~4 chars per token holds reasonably well across English-text
    # tokenizers (BPE / tiktoken / SentencePiece). It's not exact —
    # code-heavy prompts run ~3 chars/token, math-heavy ~5 — but for
    # a cost chart that previously showed empty bars on CLI/bridge
    # transports, "roughly correct" beats "null". Estimated rows are
    # flagged with ``estimated: True`` so the chart can render them
    # differently if it ever wants to.
    _CHARS_PER_TOKEN = 4

    def _fill_usage_estimate_if_missing(
        self, messages: list[dict[str, str]], completion: str,
    ) -> None:
        """Populate ``last_usage`` with a char-based token estimate when
        the transport didn't return structured usage. No-op when usage
        was already captured upstream (HTTP responses with ``usage``)."""
        if self.last_usage is not None:
            return
        prompt_chars = sum(len(_content_text(m.get("content", ""))) for m in messages)
        completion_chars = len(completion or "")
        prompt_tokens = (
            max(1, prompt_chars // self._CHARS_PER_TOKEN)
            + _IMAGE_TOKENS_ESTIMATE * len(_message_images(messages))
        )
        completion_tokens = max(0, completion_chars // self._CHARS_PER_TOKEN)
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated": True,
        }

    async def _chat_vscode_bridge(
        self,
        messages: list[dict[str, str]],
        *,
        model_override: str | None = None,
        temperature: float = 0.2,
        node: str = "",
    ) -> str:
        """Route the chat call through the FI VSCode extension
        via the localhost TCP bridge. The extension makes the actual
        ``vscode.lm`` call on the authenticated user's behalf so we
        never touch the Copilot HTTP API directly — that's the whole
        point of the sanctioned path. ``model_override`` becomes the
        ``model_hint`` the extension passes to ``selectChatModels``."""
        # Lazy-import so non-VSCode runs don't pay for the module load.
        from .vscode_bridge import VSCodeBridgeClient
        if self._bridge is None:
            # Prefer the IPC socket when available (--serve / --tools
            # path), fall back to the TCP port (chat-spawned per-command
            # bridges still use TCP). The client picks the transport
            # based on which kwarg is non-empty.
            self._bridge = VSCodeBridgeClient(
                host="127.0.0.1",
                port=self.endpoint.vscode_bridge_port,
                socket_path=(self.endpoint.vscode_bridge_socket or None),
                # Wall-clock cap per chat call so a TS-side silent stall
                # (Copilot HTTP/2 hang the inactivity timer missed, or
                # an older .vsix without the timer at all) can't wedge
                # the engine forever on ``await fut``. The bridge
                # raises ``BridgeError("bridge stalled ...")`` on
                # timeout; ``_TRANSIENT_BRIDGE_MARKERS`` recognises it
                # and tenacity below retries the call.
                cli_timeout_s=self._cli_timeout_s,
            )
            await self._bridge.connect()
        # Per-call override wins; otherwise use the user's
        # YAML-pinned model (`vscode_model_override`). DO NOT fall
        # through to `self.endpoint.model` — that field carries a
        # human-readable display string ("(VSCode chat default)")
        # for logging, and selectChatModels would reject it as an
        # invalid family filter. Empty hint = "use whatever the user
        # picked in the Chat model picker", which the extension
        # handles by calling selectChatModels({vendor: "copilot"}).
        if model_override is not None:
            hint = model_override
        else:
            hint = self.endpoint.vscode_model_override
        # Retry transient bridge errors with exponential backoff. The
        # extension's own sendRequest retries inside the TS bridge,
        # but a user on an older .vsix won't have that — and even on
        # the latest, the bridge surfaces `lm_error` after its own
        # retry exhausts. This is the second-chance layer.
        #
        # The retry predicate is a callable, NOT `retry_if_exception_type(BridgeError)`
        # — the latter would retry every BridgeError (including auth /
        # no-model-available / user-cancelled), which is wrong. We
        # specifically only want to retry transient-looking ones.
        from .vscode_bridge import BridgeError

        def _retry_transient_bridge(exc: BaseException) -> bool:
            return (
                isinstance(exc, BridgeError)
                and _is_bridge_error_transient(str(exc))
            )

        # Budget: 6 attempts. Each inter-attempt wait is drawn
        # uniformly from ``[0, 2 · 2^attempt]`` seconds, capped at 60
        # — typical expected total ~2 minutes of cumulative backoff,
        # spread randomly so concurrent fleet clients don't synchronise
        # on a shared upstream blip. Sustained Copilot HTTP/2 outages
        # have been observed lasting 30–90 s; the prior 3-attempt /
        # ~14 s budget was too tight and crashed quests on transient
        # upstream issues. The TS-side bridge also retries 4×, so
        # total wall time before a real failure exceeds 2 minutes.
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(6),
                # Jittered backoff (see HTTP path comment): a
                # deterministic schedule synchronises concurrent
                # bridge clients in a --fleet on a shared upstream
                # blip. Random spreads them over the window.
                wait=wait_random_exponential(multiplier=2, max=60),
                retry=retry_if_exception(_retry_transient_bridge),
                before_sleep=self._retry_note(node, "the VS Code connection", 6, hint or None),
                reraise=True,
            ):
                with attempt:
                    return await self._bridge.chat(
                        messages, model_hint=hint or "", temperature=temperature,
                        node=node,
                        # Ask the chat model for its reasoning only when the engine will keep it
                        # (``output.save_thinking``, a step's call): a generator's call keeps none, so asks for none.
                        ask_thinking=thinking_wanted(),
                    )
        except BridgeError as exc:
            # The extension could not hand the screenshots to the model (an
            # older VS Code, or a model without image input).
            if _message_images(messages) and "image" in str(exc).lower():
                raise ImageInputUnsupported(str(exc)) from exc
            if _is_bridge_error_transient(str(exc)):
                raise BridgeError(
                    "Copilot backend was unavailable across 6 retry "
                    "attempts (~2 min of cumulative backoff). This is "
                    "an upstream Copilot/HTTP issue, not a problem "
                    "with your config or network — please retry the "
                    f"quest in a few minutes. Last error: {exc}"
                ) from exc
            raise
        # Unreachable — tenacity reraise=True always raises on exhaustion.
        raise RuntimeError("vscode-bridge retry exhausted without raising")

    async def _chat_cli(
        self,
        messages: list[dict[str, str]],
        *,
        model_override: str | None = None,
        node: str = "",
    ) -> str:
        """Exec a local CLI binary with the prompt and return its output.

        Flattens the OpenAI-style `messages` list into a single text prompt
        (most LLM CLIs don't have a separate system/user channel — they
        accept one block of text). Retries with exponential backoff on
        `OSError` (transport-level OS errors during spawn) and
        `_CliTransientError` (any non-zero CLI exit). A missing CLI binary
        on PATH raises `RuntimeError` from `_run_cli` and is NOT retried —
        the user must install the CLI before this provider can succeed.
        Persistent auth/quota failures also surface here when the CLI's
        OAuth refresh has run out of options, since they are reported as
        non-zero exits and so will be retried up to 4 times before raising.

        ``node`` is the FI engine node name; when set, looks up
        ``node_cli_timeout_s[node]`` for the per-call wall-clock budget
        so reasoning-heavy nodes (implement, execute_reflect) can get
        longer ceilings than fast ones (clarify, ideate).

        Model escalation on retry (opt-in): when the user has set
        ``provider.node_model_fallbacks``, retry attempt 2+ looks up
        ``node_model_fallbacks[node]`` and switches to that model. With
        no mapping (the default) every attempt uses the primary model.
        The motivation is a smaller Claude model paralysing on long
        code-gen prompts (extended-thinking spins forever without
        producing text); escaping to a stronger model on retry works.
        The escalation is per-call and per-node so the user's primary
        model preference is honoured on first try.
        """
        spec = self.endpoint.cli_spec
        if spec is None:  # pragma: no cover — guarded by transport check
            raise RuntimeError("transport=cli but no cli_spec set")
        prompt = _messages_to_text(messages)
        # A CLI that cannot take images raises ImageInputUnsupported from
        # _run_cli before anything is spawned; it is not retried.
        images = _message_images(messages)

        # Pick the per-call total-timeout: node-specific override wins,
        # else the client-level default.
        effective_total_timeout = node_budget(
            self._node_cli_timeout_s, node, self._cli_timeout_s,
        )
        # Effective inactivity timeout. None means "disabled, only the
        # total ceiling applies" — preserved for compat / tests.
        effective_inactivity = (
            self._cli_inactivity_timeout_s
            if self._cli_inactivity_timeout_s is not None
            else effective_total_timeout
        )

        # First-attempt model: per-call override > endpoint default.
        primary_model = (
            model_override if model_override is not None
            else self.endpoint.cli_model_override
        )
        # On retry, escalate to a different model when configured.
        fallback_model = self._node_model_fallbacks.get(node, "") if node else ""

        # Diagnostic visibility: when tenacity catches an exception
        # and decides to retry, the exception text used to be swallowed
        # (the caller saw nothing in run.log until all 4 attempts
        # exhausted). This callback logs each caught exception + which
        # model the next attempt will use, so silent retries become
        # debuggable. The actual log_warning is closed over in the
        # ``_log`` reference below.
        # The model the NEXT try is asked on (a node's fallback model from the second try on).
        _retry_log = self._retry_note(
            node or spec.argv[0], f"{spec.argv[0]} CLI", 4,
            lambda rs: ((fallback_model if (fallback_model and rs.attempt_number + 1 >= 2) else primary_model)
                        or "the CLI's default"),
        )

        async for attempt in AsyncRetrying(
            # Normal transients get 4 attempts; a repeated WEDGE bails after 2
            # (an unrecoverable hang — retrying just wedges again, ~2 wasted
            # multi-minute calls) and reraises with switch-provider guidance.
            stop=stop_after_attempt(4) | _stop_on_repeated_wedge,
            # Jittered backoff so concurrent --fleet quests don't all
            # retry the same upstream at the same instant — see
            # the HTTP path's note. wait_random_exponential picks a
            # uniform random value from [0, multiplier * 2^attempt],
            # capped at ``max``, which spreads retries over a window. A
            # capacity error waits longer (see ``_cli_retry_wait``).
            wait=_cli_retry_wait,
            retry=retry_if_exception(_retry_cli_error),
            reraise=True,
            before_sleep=_retry_log,
        ):
            with attempt:
                # Escalate to fallback model on retry 2+ if configured.
                # The first attempt always uses primary_model so the
                # user's stated preference is tried first.
                attempt_no = attempt.retry_state.attempt_number
                effective_model = (
                    fallback_model
                    if (attempt_no >= 2 and fallback_model)
                    else primary_model
                )
                # A retry after a timeout kill gets a longer budget; every
                # other failure keeps the configured one.
                attempt_timeout = _timeout_for_attempt(
                    attempt.retry_state, effective_total_timeout,
                )
                if attempt_timeout > effective_total_timeout:
                    _log.info(
                        "[%s] previous attempt was killed at %gs; attempt %d "
                        "gets %gs",
                        node or spec.argv[0], effective_total_timeout,
                        attempt_no, attempt_timeout,
                    )
                measured: dict[str, Any] = {}
                text = await _run_cli(
                    spec, prompt,
                    images=images,
                    model=effective_model,
                    timeout_s=attempt_timeout,
                    inactivity_timeout_s=(
                        effective_inactivity
                        if self._cli_inactivity_timeout_s is not None
                        else attempt_timeout
                    ),
                    heartbeat_cb=self._heartbeat_cb,
                    node=node,
                    usage_out=measured,
                    reasoning_effort=self.endpoint.reasoning_effort,
                )
                # A real reading from the CLI beats the char-count estimate,
                # and by a wide margin: these CLIs wrap our prompt in their
                # own system prompt and tool schema, which is most of the
                # input and which the estimator cannot see at all.
                served = measured.pop("served_model", None)
                switched = measured.pop("switched", None)
                if measured.pop("refused", False):
                    # The model declined and no other model answered: what came back is a refusal, never an answer.
                    # Like a content filter's withheld answer it is not retried (the same model declines again): the
                    # quest stops for a person, and the call's cost is recorded with it.
                    raise ModelAnswerFiltered(
                        f"{spec.argv[0]}: {effective_model or 'the model'} declined this request (a refusal, not an "
                        "answer)",
                        node=node, usage=dict(measured) or None, provider=self.last_provider,
                        model=served if isinstance(served, str) else (effective_model or None),
                        finish_reason="refusal",
                    )
                kept_switch = None
                if isinstance(switched, dict):
                    text, served, measured, kept_switch = await self._ask_the_asked_model_again(
                        spec, prompt, images=images, model=effective_model, switched=switched, first_text=text,
                        first_served=served, first_measured=measured, timeout_s=attempt_timeout,
                        inactivity_timeout_s=(effective_inactivity if self._cli_inactivity_timeout_s is not None
                                              else attempt_timeout),
                        node=node,
                    )
                if isinstance(served, str) and served:
                    # The CLI said which model answered this very call (claude_cli does). Set from this call's own
                    # reading, in this task's own record: calls running at the same time never see each other's.
                    self.last_model = served
                    LAST_CALL.set({"provider": self.last_provider, "model": served, "reported": True,
                                   # The model the CLI switched away from, when the answer kept is another model's.
                                   **({"switched_from": kept_switch.get("from") or effective_model}
                                      if kept_switch else {})})
                if measured:
                    self.last_usage = measured
                return text
        raise RuntimeError("unreachable: tenacity reraise=True must raise on exhaustion")

    def _say_model(self, line: str) -> None:
        """One plain warning about which model answered, in FI's log and the quest's run.log (every time, not once)."""
        _log.warning("%s", line)
        if self._run_log is not None:
            try:
                self._run_log.warning("%s", line)
            except Exception:  # noqa: BLE001 -- a log line never stops a call
                pass

    async def _ask_the_asked_model_again(
        self, spec: _CliSpec, prompt: str, *, images: list[tuple[str, bytes]], model: str, switched: dict[str, Any],
        first_text: str, first_served: Any, first_measured: dict[str, Any], timeout_s: float,
        inactivity_timeout_s: float, node: str,
    ) -> tuple[str, Any, dict[str, Any], dict[str, Any] | None]:
        """The claude CLI answered with another model than the one asked for (``switched``: it does so on its own when
        the asked-for model declines a request, and that model can cost far more: a Haiku quest's ``ideate`` call was
        answered by a Fable model at about twelve times Haiku's price). A quest pays for the model the person chose, so
        this is never accepted silently: it is said in run.log every time, and the asked-for model is asked once more
        with the CLI's switching turned off (``CLAUDE_NO_REFUSAL_FALLBACK_ENV``). One direct call, not another round
        of retries: a model that declines twice would decline again.

        Returns ``(text, served model, usage, switch kept)``: the asked-for model's answer when it gave one (the first
        answer is then recorded as a paid attempt that was not used), else the first answer with its switch, said
        plainly again."""
        to = str(switched.get("to") or "another model")
        asked = model or str(switched.get("from") or "") or "the asked-for model"
        reason = {
            "model_refusal_fallback": "declined this request",
            "model_fallback": "was not available",
            "model_consent_fallback": "is not allowed by the account's model settings",
        }.get(str(switched.get("kind") or ""), "could not be used")
        where = f"[model] {node}: " if node else "[model] "
        if not model:
            self._say_model(f"{where}the claude tool's default model {reason}, so it answered with {to} instead; "
                            f"that answer is kept (no model is named in the config to ask again)")
            return first_text, first_served or to, first_measured, switched
        if switched.get("kind") != "model_refusal_fallback":
            # Only the switch after a refusal can be turned off for one call; asking again after any other switch
            # would only switch again, at the other model's price.
            self._say_model(f"{where}{asked} {reason}, so the claude tool answered with {to} instead; that answer is "
                            f"kept, and this call cost {to}'s price, not {asked}'s")
            return first_text, first_served or to, first_measured, switched
        self._say_model(f"{where}{asked} {reason}, so the claude tool answered with {to} instead; asking {asked} once "
                        "more with switching turned off")
        second: dict[str, Any] = {}
        problem = ""
        text2 = ""
        served2: Any = None
        try:
            text2 = await _run_cli(
                spec, prompt, images=images, model=model, timeout_s=timeout_s,
                inactivity_timeout_s=inactivity_timeout_s, heartbeat_cb=self._heartbeat_cb, node=node,
                usage_out=second, reasoning_effort=self.endpoint.reasoning_effort,
                extra_env={CLAUDE_NO_REFUSAL_FALLBACK_ENV: "1"},
            )
            served2 = second.pop("served_model", None)
            again = second.pop("switched", None)
            label = ""
            if second.pop("refused", False):
                problem, label = "it declined again", "refused"
            elif isinstance(again, dict):
                problem, label = f"the tool switched again, to {again.get('to')}", "switched_again"
            elif not (text2 or "").strip():
                problem, label = "it gave no answer", "no_answer"
            elif isinstance(served2, str) and claude_model_family_differs(model, served2):
                problem, label = f"{served2} answered, not {asked}", "answered_by_other_model"
            if problem:
                _note_failed_attempt(self.last_provider, served2 or model,
                                     _OtherModelAnswer(problem, usage=second, outcome=label))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- the first answer is kept; the failure is said and recorded
            problem = f"the call failed: {type(e).__name__}"
            _note_failed_attempt(self.last_provider, model, e)
        if not problem:
            # The first answer is not used, but it was paid for: recorded as an attempt with its token counts.
            _note_failed_attempt(self.last_provider, first_served or to,
                                 _OtherModelAnswer(f"answered by {to}, not {asked}", usage=first_measured,
                                                   outcome="answered_by_other_model"))
            self._say_model(f"{where}{asked} answered when asked again; its answer is kept (the {to} answer it "
                            "replaces was still paid for)")
            return text2, served2 or model, second, None
        self._say_model(f"{where}{asked} could not answer ({problem}); the {to} answer is kept, and this call cost "
                        f"{to}'s price, not {asked}'s")
        return first_text, first_served or to, first_measured, switched


# ---------------------------------------------------------------------------
# Provider fallback chain + per-provider circuit breaker
# ---------------------------------------------------------------------------


def _is_fatal_provider_error(exc: BaseException) -> bool:
    """A provider failure that won't clear within the run — auth/quota/credit
    exhaustion — so the breaker should open immediately rather than after the
    usual failure threshold. Transient blips (5xx, timeouts) trip only after
    repeated failures."""
    if isinstance(exc, _CliTransientError):
        return any(m in str(exc).lower() for m in _CLI_FATAL_MARKERS)
    if isinstance(exc, httpx.HTTPStatusError):
        sc = getattr(getattr(exc, "response", None), "status_code", None)
        return isinstance(sc, int) and (sc in (401, 403, 402) or (sc == 429 and _is_exhausted_quota(exc.response)))
    return False


class _AppendToFile(logging.Handler):
    """Appends each record to a file and closes it again: nothing stays open (a Windows lock) and nothing is shared."""

    def __init__(self, path: Path) -> None:
        super().__init__(logging.WARNING)
        self.path = Path(path)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(self.format(record) + "\n")
        except Exception:  # noqa: BLE001 -- a log line never stops a call
            pass


def quest_run_log(quest_root: Path | str | None) -> "logging.Logger | None":
    """A logger that adds warnings to ``<quest_root>/.fi/run.log`` (for a model call made outside the engine, after it
    closed the quest's own log: the output generators, the critique). Not registered anywhere; ``None`` without a
    quest folder."""
    if not quest_root:
        return None
    fi_dir = Path(quest_root) / ".fi"
    if not fi_dir.is_dir():
        return None
    logger = logging.Logger(f"frontier_insight.run_log.{Path(quest_root).name}", logging.WARNING)
    logger.addHandler(_AppendToFile(fi_dir / "run.log"))
    logger.propagate = False
    return logger


def _retry_line(node: str | None, where: str, rs: Any, total: int, *, model: str | None = None) -> str:
    """One line for a failed model call that is about to be tried again: which step, which provider and model, which
    attempt of how many, what went wrong and how long until the next try. Nothing that looks like a key."""
    exc = rs.outcome.exception() if getattr(rs, "outcome", None) else None
    wait = getattr(getattr(rs, "next_action", None), "sleep", None)
    from core.audit_log import redact_text

    # The same credential removal as the audit trace (key-shaped tokens and the values of secret environment
    # variables), on the whole message first, then cut: a cut first could leave half a key the patterns miss.
    what = redact_text(f"{type(exc).__name__}: {exc}", whole=True) if exc else "no error captured"
    what = " ".join(what.split())
    what = what if len(what) <= 240 else what[:240] + "..."
    status = _http_outage_status(exc)
    if status == 429:
        what = f"the provider is limiting how often FI may call it (HTTP 429): {what}"
    elif status is not None:
        what = f"the provider's server is down or busy (HTTP {status}): {what}"
    return (f"[{node or 'model'}] the model call failed ({where}{', model ' + model if model else ''}), "
            f"attempt {rs.attempt_number} of {total}: {what}; trying again"
            + (f" in {wait:.0f}s" if isinstance(wait, (int, float)) else ""))


@dataclass
class _FallbackSlot:
    """One rung of the fallback chain: a provider plus its breaker state."""

    label: str
    # ``None`` for the primary (already constructed); an async factory for
    # lazily-built fallbacks (their proxy, if any, spins up only on first use).
    factory: "Callable[[], Any] | None"
    client: "LLMClient | None" = None
    failures: int = 0
    tripped: bool = False  # circuit open
    tripped_at: float | None = None  # monotonic time the circuit opened

    def record_failure(self, threshold: int, *, fatal: bool, now: float) -> None:
        self.failures += 1
        if fatal or self.failures >= threshold:
            self.tripped = True
            self.tripped_at = now  # (re)start the cooldown clock

    def record_success(self) -> None:
        # A closed circuit (or a successful half-open probe) resets everything.
        self.failures = 0
        self.tripped = False
        self.tripped_at = None

    def is_available(self, now: float, cooldown_s: float) -> bool:
        """Usable if the circuit is closed, or open-but-cooled-down (half-open:
        allow ONE probe). ``cooldown_s <= 0`` keeps the legacy behaviour —
        a tripped circuit stays open for the rest of the run."""
        if not self.tripped:
            return True
        if cooldown_s <= 0 or self.tripped_at is None:
            return False
        return (now - self.tripped_at) >= cooldown_s


class FallbackLLMClient:
    """Try an ordered chain of providers so a single provider's outage (rate
    limit, auth failure, proxy crash) doesn't forfeit the whole quest.

    ``chat`` calls the primary; if it terminally fails (after the primary's own
    in-provider tenacity retries), the call moves to the next provider, and so
    on. Each provider has a **circuit breaker**: after ``breaker_threshold``
    failures — or a single auth/quota error — its circuit opens and it is
    skipped for the rest of the run, so subsequent calls jump straight to a
    live provider instead of re-burning the dead one's retry budget on every
    call. Fallbacks are built lazily (nothing is resolved or spawned until the
    primary actually fails), and the provider names of any fallbacks that were
    materialised are recorded in :attr:`built_fallback_providers` so the caller
    can release their proxies on shutdown.

    Presents the read surface the Engine uses on a plain ``LLMClient``
    (``chat`` / ``last_model`` / ``last_usage`` / ``last_provider`` / ``aclose``);
    ``last_model``, ``last_usage`` and ``last_provider`` reflect whichever
    provider served the most recent call, not the primary that was asked first —
    the Engine's audit trace records this one, since a fallback slot's label is
    the truthful answer to "which model said this."
    """

    def __init__(
        self,
        primary: "LLMClient",
        factories: "list[tuple[str, Callable[[], Any]]]" = (),
        *,
        breaker_threshold: int = 2,
        breaker_cooldown_s: float = 90.0,
        log: "logging.Logger | None" = None,
    ) -> None:
        self._log = log or logging.getLogger("fi.provider.fallback")
        primary_label = getattr(
            getattr(primary, "endpoint", None), "provider_name", None,
        ) or "primary"
        self._slots: list[_FallbackSlot] = [
            _FallbackSlot(label=primary_label, factory=None, client=primary)
        ]
        for name, factory in factories:
            self._slots.append(_FallbackSlot(label=name, factory=factory))
        self._threshold = max(1, int(breaker_threshold))
        # Half-open recovery: a tripped provider is re-probed once after this
        # many seconds (a probe with a healthy provider after it gets the short
        # retry budget, see _http_short_retry) so a transient outage doesn't
        # permanently drop it for the whole quest. 0 disables (open for the run).
        self._cooldown_s = float(breaker_cooldown_s)
        # Read by the Engine cost logger immediately after each chat().
        self.last_usage: Any = None
        self.last_model: str | None = None
        self.last_provider: str | None = None
        # Fallback providers actually materialised (for proxy release on close).
        self.built_fallback_providers: list[str] = []

    async def _client_for(self, slot: _FallbackSlot) -> "LLMClient":
        if slot.client is None:
            assert slot.factory is not None
            slot.client = await slot.factory()
            self.built_fallback_providers.append(slot.label)
            self._log.info("[fallback] initialised provider %s", slot.label)
        return slot.client

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        errors: list[tuple[str, BaseException]] = []
        now = time.monotonic()
        for idx, slot in enumerate(self._slots):
            if not slot.is_available(now, self._cooldown_s):
                continue
            probing = slot.tripped  # a cooled-down open circuit -> half-open probe
            if probing:
                self._log.info(
                    "[fallback] half-open probe of %s after %.0fs cooldown",
                    slot.label, now - (slot.tripped_at or now),
                )
            try:
                client = await self._client_for(slot)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # building/resolving the fallback failed
                # Timestamp the trip at failure time, not chat() entry — a slow
                # call that then fails must not backdate the cooldown clock.
                slot.record_failure(self._threshold, fatal=True, now=time.monotonic())
                self._log.warning(
                    "[fallback] could not initialise %s: %r", slot.label, e,
                )
                errors.append((slot.label, e))
                continue
            # This provider's own record of the call: what an earlier one named is not this one's answer.
            LAST_CALL.set(None)
            note_thinking("")
            # No long outage wait (see _http_short_retry) while a healthy provider (circuit closed) comes after this
            # one: minutes on a dead provider are better spent on a live one. When every later provider is tripped
            # too, this one waits the outage out, as it would with no fallback chain.
            healthy_next = any(not s.tripped for s in self._slots[idx + 1:])
            slot_token = CALL_SLOT.set({"provider": slot.label, "fallback": idx > 0,
                                        "short_retry": healthy_next})
            try:
                text = await client.chat(messages, **kwargs)
            except asyncio.CancelledError:
                # Cancellation is not a provider failure — never fall back on it.
                raise
            except Exception as e:
                failed = LAST_CALL.get() or {}
                _note_failed_attempt(slot.label, failed.get("model") or getattr(client, "last_model", None), e,
                                     fallback=idx > 0)
                if not isinstance(e, _ModelAnswerProblem):
                    # A cut-off or withheld answer is about this request, not the provider's health: no circuit trip.
                    fatal = _is_fatal_provider_error(e)
                    slot.record_failure(self._threshold, fatal=fatal, now=time.monotonic())
                more = idx < len(self._slots) - 1
                self._log.warning(
                    "[fallback] provider %s failed (%s)%s; %s",
                    slot.label, type(e).__name__,
                    " [circuit opened]" if slot.tripped else "",
                    "trying next provider" if more else "no more providers",
                )
                errors.append((slot.label, e))
                continue
            finally:
                CALL_SLOT.reset(slot_token)
            # Success — snapshot the serving provider's cost fields, and close
            # the circuit (a successful half-open probe re-admits the provider).
            if probing:
                self._log.info("[fallback] provider %s recovered — circuit closed", slot.label)
            slot.record_success()
            self.last_usage = getattr(client, "last_usage", None)
            served = LAST_CALL.get() or {}
            self.last_model = served.get("model") or getattr(client, "last_model", None)
            self.last_provider = slot.label
            LAST_CALL.set({**served, "provider": slot.label, "model": self.last_model, "fallback": idx > 0})
            if idx > 0:
                self._log.info(
                    "[fallback] request served by %s (primary unavailable)",
                    slot.label,
                )
            return text
        # Every provider failed on this call, or all circuits are already open.
        if errors:
            _, last_err = errors[-1]
            if len(errors) > 1 and not all(getattr(e, "fi_timeouts", {}).get("all") for _, e in errors):
                # The last provider timing out on every try is not "the model is too slow" when another failed another
                # way: crash_kind reads it as a passing problem.
                try:
                    last_err.fi_timeouts["all"] = False  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001 -- no mark on the error
                    pass
            try:
                chain = ", ".join(lbl for lbl, _ in errors)
                last_err.add_note(f"[FI] all providers exhausted: {chain}")
            except Exception:  # pragma: no cover — needs Py<3.11
                pass
            raise last_err
        raise RuntimeError(
            "no LLM providers available: every circuit is open "
            f"({', '.join(s.label for s in self._slots)})"
        )

    async def aclose(self) -> None:
        for slot in self._slots:
            if slot.client is not None:
                try:
                    await slot.client.aclose()
                except Exception:  # pragma: no cover — best-effort cleanup
                    pass
