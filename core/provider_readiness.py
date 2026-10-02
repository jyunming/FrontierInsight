"""How far a model provider is ready on this computer, in one plain sentence.

The old check was one yes/no ("READY") that only meant "a key is set" or "the command is on PATH": it said nothing
about being signed in, the server running or the model being there, so a person found out after answering the whole
interview, or after the quest had started. This module answers the same question as a ladder, and stops at the
first step that is not met::

    not installed -> installed -> signed in (or: cannot tell) -> reachable -> model available

An API key stops at "key present" unless the service is asked (``call_service``): FI never calls a model to find
out, only the provider's free model list (OpenAI-compatible ``GET /models``; Ollama ``GET /api/tags``). A CLI is
asked with its own sign-in status command (``claude auth status``, ``codex login status``), with a short time limit
and never a generation; a CLI without one says "cannot tell".

Three levels of checking, so each caller pays only for what it needs:

* :func:`check_local` -- no subprocess, no network: the command on PATH, the key in the environment. For the quick
  ``fi --doctor`` and the interview's provider list.
* :func:`check` with ``sign_in=True`` (the default) -- also the CLIs' status commands and a local Ollama server.
  For the Settings page. No call leaves this computer except what a CLI's own status command does.
* :func:`preflight` -- also the provider's model list over the network. Before a quest is launched, by ``fi demo``
  and by ``fi --doctor --deep``.

Every surface (CLI, web, VS Code through the CLI) prints :attr:`Readiness.sentence` and :attr:`Readiness.fix`
as they are, so the wording is the same everywhere.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Callable

__all__ = [
    "Readiness", "STATE_LABELS", "check_local", "check", "check_all", "preflight", "label", "one_line",
    "ollama_models", "CLI_BINARIES", "KEY_ENVS",
]

#: The command each CLI provider runs (``core.provider._CLI_SPECS[...].argv[0]``; a test keeps them equal).
CLI_BINARIES: dict[str, str] = {
    "claude_cli": "claude",
    "codex_cli": "codex",
    "copilot_cli": "copilot",
    "gemini_cli": "gemini",
    "antigravity_cli": "agy",
}

#: The environment variable an HTTP provider reads its key from (``core.provider._DIRECT_DEFAULTS``).
KEY_ENVS: dict[str, str] = {"openai": "OPENAI_API_KEY", "codex": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}

#: Where an HTTP provider is served by default (``core.provider._DIRECT_DEFAULTS``).
_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "codex": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://127.0.0.1:11434/v1",
    "vllm": "http://127.0.0.1:8000/v1",
}

#: How a person signs in to each CLI; ``None``: no command FI knows of.
_SIGN_IN: dict[str, str | None] = {
    "claude_cli": "claude auth login",
    "codex_cli": "codex login",
    "copilot_cli": "copilot login",
    "gemini_cli": None,
    "antigravity_cli": None,
}

#: Where each CLI comes from, for "not installed".
_INSTALL: dict[str, str] = {
    "claude_cli": "Install Claude Code (https://docs.anthropic.com/en/docs/claude-code), then run `claude auth login`.",
    "codex_cli": "Install the Codex CLI (`npm install -g @openai/codex`), then run `codex login`.",
    "copilot_cli": "Install the GitHub Copilot CLI (https://github.com/github/copilot-cli), then run `copilot login`.",
    "gemini_cli": "Install the Gemini CLI (`npm install -g @google/gemini-cli`), then run `gemini` once to sign in.",
    "antigravity_cli": "Install Google Antigravity's `agy` command and sign in once.",
}

#: A short badge for each state (the web Settings page, the CLI list). The sentence says the rest.
STATE_LABELS: dict[str, str] = {
    "not_installed": "Not installed",
    "installed": "Installed",
    "not_signed_in": "Not signed in",
    "signed_in": "Signed in",
    "sign_in_unknown": "Signed in: unknown",
    "no_key": "No key",
    "key_present": "Key present",
    "key_rejected": "Key rejected",
    "not_running": "Not running",
    "unreachable": "Not reachable",
    "reachable": "Reachable",
    "model_available": "Model available",
    "model_missing": "Model missing",
    "model_not_listed": "Model not listed",
    "unknown": "Checked on first call",
}

#: States a launch would fail on. Everything else either works or cannot be told before the first call.
_BLOCKING = frozenset({"not_installed", "not_signed_in", "no_key", "key_rejected", "not_running", "unreachable",
                       "model_missing"})


@dataclass(frozen=True)
class Readiness:
    """One provider's state. ``sentence`` and ``fix`` are shown to a person exactly as they are."""

    provider: str
    state: str
    sentence: str
    fix: str = ""
    model: str = ""

    @property
    def blocked(self) -> bool:
        """A quest launched now would fail on this, before or at its first call."""
        return self.state in _BLOCKING

    @property
    def label(self) -> str:
        return label(self.state)

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "label": self.label, "blocked": self.blocked}


def label(state: str) -> str:
    return STATE_LABELS.get(state, state)


def one_line(r: Readiness) -> str:
    """``Signed in — claude auth status says you are signed in; ...`` plus `` To fix: ...`` when there is one."""
    text = f"{r.label} — {r.sentence}"
    return f"{text} To fix: {r.fix}" if r.fix else text


# ---------------------------------------------------------------------------------------------------------------------
# Running a CLI's status command
# ---------------------------------------------------------------------------------------------------------------------

def _run_command(argv: list[str], timeout_s: float) -> tuple[int | None, str]:
    """Run ``argv`` with no input; ``(returncode, stdout+stderr)``, or ``(None, "")`` when it did not finish within
    ``timeout_s`` (it is then stopped with everything it started) or could not be started. Tests replace this."""
    from core.proc_tree import ProcessTree

    try:
        with ProcessTree(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as tree:
            try:
                out, _ = tree.proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                tree.kill()
                try:
                    tree.proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    pass  # something it started still holds the pipe: the answer is "did not answer" anyway
                return None, ""
            return tree.proc.returncode, (out or b"").decode("utf-8", "replace")
    except OSError:
        return None, ""


def _claude_signed_in(rc: int | None, out: str) -> bool | None:
    """``claude auth status`` prints JSON with ``loggedIn``. Only that field is read: the rest (an e-mail address,
    an organisation id) is never kept or shown."""
    if rc is None:
        return None
    start, end = out.find("{"), out.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(out[start:end + 1])
    except ValueError:
        return None
    value = data.get("loggedIn") if isinstance(data, dict) else None
    return value if isinstance(value, bool) else None


def _codex_signed_in(rc: int | None, out: str) -> bool | None:
    """``codex login status``: ``Logged in using ...`` and exit 0, or ``Not logged in`` and a non-zero exit."""
    if rc is None:
        return None
    low = out.lower()
    if "not logged in" in low:
        return False
    if rc == 0 and "logged in" in low:
        return True
    return None


#: provider -> (status command after the binary, parser). A CLI not listed here has no status command.
_STATUS: dict[str, tuple[tuple[str, ...], Callable[[int | None, str], bool | None]]] = {
    "claude_cli": (("auth", "status"), _claude_signed_in),
    "codex_cli": (("login", "status"), _codex_signed_in),
}


# ---------------------------------------------------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------------------------------------------------

def _key_env(provider: str, api_key_env: str) -> str:
    return (api_key_env or "").strip() or KEY_ENVS.get(provider, "")


def _base_url(provider: str, base_url: str) -> str:
    """The address the engine will call: ``provider.base_url``, else the provider's default
    (``core.provider._DIRECT_DEFAULTS``). ``OLLAMA_HOST`` is not read: the engine does not read it either."""
    return ((base_url or "").strip() or _BASE_URLS.get(provider, "")).rstrip("/")


def _own_gateway(provider: str, base_url: str, api_key_env: str) -> bool:
    """A ``base_url`` of the person's own (not the provider's) that names no key: it may need none, and the engine
    starts it without one (``core.provider.missing_api_key``), so it is not "no key" here either."""
    url = (base_url or "").strip().rstrip("/")
    return bool(url) and url != _BASE_URLS.get(provider, "") and not (api_key_env or "").strip()


#: When FI checks what a local check leaves open; said in those sentences.
_WHEN = "when you launch from `fi demo`, `fi --new` or the web form, and by `fi --doctor --deep`"


def check_local(provider: str, *, model: str | None = None, base_url: str = "", api_key_env: str = "") -> Readiness:
    """No subprocess and no network: the command on PATH, the key in the environment."""
    model = (model or "").strip()
    if provider in CLI_BINARIES:
        exe = CLI_BINARIES[provider]
        if shutil.which(exe) is None:
            return Readiness(provider, "not_installed", f"No `{exe}` command on this computer.",
                             _INSTALL[provider], model)
        if provider not in _STATUS:
            return Readiness(provider, "installed",
                             f"The `{exe}` command is here; FI cannot check the sign-in ahead of time (this command "
                             "has no status check), so the first call will tell.", "", model)
        return Readiness(provider, "installed",
                         f"The `{exe}` command is here; whether you are signed in is checked {_WHEN}.", "", model)
    if provider == "ollama":
        if shutil.which("ollama") is None:
            return Readiness(provider, "not_installed",
                             "No `ollama` command on this computer (an Ollama server elsewhere still works: set "
                             f"provider.base_url to it; it is checked {_WHEN}).",
                             "Install Ollama from https://ollama.com, or set provider.base_url to your server.", model)
        return Readiness(provider, "installed",
                         f"The `ollama` command is here; whether its server is running is checked {_WHEN}.", "", model)
    if provider == "vscode_extension":
        return Readiness(provider, "unknown",
                         "Used from VS Code: the model picked in the Copilot Chat picker answers; VS Code checks it.",
                         "", model)
    if provider in KEY_ENVS and _own_gateway(provider, base_url, api_key_env):
        return Readiness(provider, "unknown",
                         f"Your own server at {_base_url(provider, base_url)}, with no key named; it is asked {_WHEN}.",
                         "", model)
    if provider in KEY_ENVS or (api_key_env or "").strip():
        env = _key_env(provider, api_key_env)
        if not os.environ.get(env, "").strip():
            return Readiness(provider, "no_key", f"{env} is not set.",
                             f"Set {env} in your terminal or in a .env file in the folder you run FI from.", model)
        return Readiness(provider, "key_present", f"{env} is set; the service has not been asked yet.", "",
                         model)
    if provider == "vllm":
        return Readiness(provider, "unknown", f"A server you run: it is asked {_WHEN}.", "", model)
    return Readiness(provider, "unknown", "FI cannot check this provider ahead of time; the first call will.", "",
                     model)


async def _sign_in(provider: str, local: Readiness, timeout_s: float) -> Readiness:
    exe = shutil.which(CLI_BINARIES[provider]) or CLI_BINARIES[provider]
    how = _SIGN_IN.get(provider)
    status = _STATUS.get(provider)
    if status is None:
        return Readiness(provider, "sign_in_unknown",
                         "FI cannot tell whether you are signed in (this command has no status check); the first "
                         "call will.",
                         "", local.model)
    args, parse = status
    rc, out = await asyncio.to_thread(_run_command, [exe, *args], timeout_s)
    signed_in = parse(rc, out)
    if signed_in is True:
        return Readiness(provider, "signed_in",
                         f"`{CLI_BINARIES[provider]} {' '.join(args)}` says you are signed in; the model itself "
                         "answers on the first call.", "", local.model)
    if signed_in is False:
        return Readiness(provider, "not_signed_in",
                         f"`{CLI_BINARIES[provider]} {' '.join(args)}` says you are not signed in.",
                         f"Run `{how}` in a terminal." if how else "", local.model)
    why = "did not answer in time" if rc is None else "gave an answer FI could not read"
    return Readiness(provider, "sign_in_unknown",
                     f"FI cannot tell whether you are signed in (`{CLI_BINARIES[provider]} "
                     f"{' '.join(args)}` {why}); the first call will.", "", local.model)


def _model_matches(wanted: str, ids: list[str]) -> bool:
    """``llama3.2`` is ``llama3.2:latest``; ``models/gemini-2.5-pro`` is ``gemini-2.5-pro``."""
    def norm(s: str) -> str:
        s = s.strip().lower().rsplit("/", 1)[-1]
        return s[: -len(":latest")] if s.endswith(":latest") else s
    w = norm(wanted)
    return any(norm(i) == w for i in ids)


async def _get_json(url: str, headers: dict[str, str], timeout_s: float) -> tuple[int | None, object]:
    """``(status, json)``; ``(None, None)`` when nothing answered."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            res = await client.get(url, headers=headers)
    except (httpx.HTTPError, OSError, ValueError):
        return None, None
    try:
        return res.status_code, res.json()
    except ValueError:
        return res.status_code, None


async def _ollama(provider: str, model: str, base_url: str, timeout_s: float, *, default_model: bool = False
                  ) -> Readiness:
    url = _base_url(provider, base_url)
    root = url[: -len("/v1")] if url.endswith("/v1") else url
    status, data = await _get_json(f"{root}/api/tags", {}, timeout_s)
    if status is None or status >= 400 or not isinstance(data, dict):
        default = _base_url(provider, "") == url
        if shutil.which("ollama") is not None:
            return Readiness(provider, "not_running",
                             f"The `ollama` command is here, but no Ollama server answers at {root}.",
                             "Start Ollama (open the Ollama app, or run `ollama serve`)."
                             + ("" if default else f" Or check provider.base_url ({root})."), model)
        if default:
            return Readiness(provider, "not_installed",
                             f"No `ollama` command here, and no Ollama server answers at {root}.",
                             "Install Ollama from https://ollama.com, or set provider.base_url to your server.", model)
        return Readiness(provider, "unreachable", f"No Ollama server answers at {root}.",
                         "Start the server, or check provider.base_url.", model)
    names = [str(m.get("name") or "") for m in (data.get("models") or []) if isinstance(m, dict)]
    if not model and default_model:
        # Before a launch: no model named means the engine calls the provider's default (core.provider._DIRECT_DEFAULTS).
        model = _default_model(provider)
    if not model:
        return Readiness(provider, "reachable", f"Running at {root} ({len(names)} model(s) downloaded).",
                         "", model)
    if _model_matches(model, names):
        return Readiness(provider, "model_available", f"Running, and {model} is downloaded.", "", model)
    return Readiness(provider, "model_missing", f"Running, but {model} is not downloaded.",
                     f"Run `ollama pull {model}`, or pick a model you have"
                     + (f" ({', '.join(chat[:5])})." if (chat := [n for n in names if "embed" not in n.lower()])
                        else "."), model)


def _default_model(provider: str) -> str:
    try:
        from core.provider import _DIRECT_DEFAULTS
        return _DIRECT_DEFAULTS.get(provider, {}).get("model", "")
    except Exception:  # noqa: BLE001 -- without it the server alone is checked
        return ""


async def ollama_models(base_url: str = "", timeout_s: float = 3.0) -> list[str]:
    """The models an Ollama server has downloaded (``GET /api/tags``); ``[]`` when it does not answer."""
    url = _base_url("ollama", base_url)
    root = url[: -len("/v1")] if url.endswith("/v1") else url
    status, data = await _get_json(f"{root}/api/tags", {}, timeout_s)
    if status is None or status >= 400 or not isinstance(data, dict):
        return []
    return [str(m.get("name") or "") for m in (data.get("models") or []) if isinstance(m, dict) and m.get("name")]


async def _http_service(provider: str, local: Readiness, base_url: str, api_key_env: str,
                        timeout_s: float) -> Readiness:
    """The provider's free model list (OpenAI-compatible ``GET /models``): the key accepted, the model listed."""
    env = _key_env(provider, api_key_env)
    key = os.environ.get(env, "").strip() if env else ""
    url = _base_url(provider, base_url)
    if not url:
        return local
    status, data = await _get_json(f"{url}/models", {"Authorization": f"Bearer {key}"} if key else {}, timeout_s)
    model = local.model
    if status is None:
        return Readiness(provider, "unreachable", f"Could not reach {url}.",
                         "Check the internet connection (or a proxy setting), or provider.base_url.", model)
    if status == 403:
        # A key may be allowed to call a model but not to list them (a restricted project key): not a refusal.
        return Readiness(provider, "reachable",
                         f"The service at {url} answers, but this key may not list its models; the first call will "
                         "tell whether the model can be used.", "", model)
    if status == 401 and not key:
        return Readiness(provider, "key_rejected", f"The server at {url} refused the request: it needs a key.",
                         "Name the environment variable that holds it in provider.api_key_env.", model)
    if status == 401:
        return Readiness(provider, "key_rejected", f"The service at {url} refused the key in {env or 'the request'}.",
                         f"Check {env}: a key that was revoked, mistyped or has no access." if env else
                         "Check the key this server expects.", model)
    if status >= 400 or not isinstance(data, dict):
        # A server with no model list (or another error) is reachable; the first call says the rest.
        return Readiness(provider, "reachable", f"The service at {url} answers; its model list was not available.",
                         "", model)
    ids = [str(m.get("id") or "") for m in (data.get("data") or []) if isinstance(m, dict)]
    if not model and not (base_url or "").strip():
        # No model named: the provider's own default is what the quest will call (core.provider._DIRECT_DEFAULTS).
        model = _default_model(provider)
    accepts = "The service accepts the key" if key else f"The server at {url} answers"
    if not model:
        return Readiness(provider, "reachable", f"{accepts}.", "", model)
    if not ids or _model_matches(model, ids):
        return Readiness(provider, "model_available", f"{accepts} and lists {model}." if ids else f"{accepts}.", "",
                         model)
    # Not blocking: a list can be incomplete (paged, or not every model a key can call); the first call tells.
    return Readiness(provider, "model_not_listed",
                     f"{accepts} but does not list {model}; the first call will tell whether it can be used.",
                     "If that is a typing mistake, check the model name.", model)


async def check(provider: str, *, model: str | None = None, base_url: str = "", api_key_env: str = "",
                sign_in: bool = True, call_service: bool = False, timeout_s: float = 5.0,
                default_model: bool = False) -> Readiness:
    """The provider's state, as far as the asked-for checks go (see the module docstring). Never raises."""
    local = check_local(provider, model=model, base_url=base_url, api_key_env=api_key_env)
    try:
        if provider in CLI_BINARIES:
            if local.state == "not_installed" or not sign_in:
                return local
            return await _sign_in(provider, local, timeout_s)
        if provider == "ollama":
            if not sign_in:
                return local
            return await _ollama(provider, local.model, base_url, timeout_s, default_model=default_model)
        if provider == "vllm" and sign_in:
            # A server on this computer (or one named by base_url): its model list is local and free.
            return await _http_service(provider, local, base_url, api_key_env, timeout_s)
        if call_service and (local.state == "key_present"
                             or (provider in KEY_ENVS and _own_gateway(provider, base_url, api_key_env))):
            return await _http_service(provider, local, base_url, api_key_env, timeout_s)
        return local
    except Exception:  # noqa: BLE001 -- a check that breaks says "cannot tell", never takes the page down
        return Readiness(provider, "unknown", "FI could not check this provider; the first call will.", "",
                         local.model)


async def check_all(providers: list[str], **kwargs: object) -> list[Readiness]:
    """:func:`check` for each provider at the same time (so the slowest status command sets the wait)."""
    return list(await asyncio.gather(*(check(p, **kwargs) for p in providers)))  # type: ignore[arg-type]


async def preflight(provider: str, *, model: str | None = None, base_url: str = "", api_key_env: str = "",
                    timeout_s: float = 5.0) -> Readiness:
    """Everything that can be checked without paying for a call, for the provider and model a person chose."""
    # A launch with no model named calls the provider's default, so that is the model looked for (Ollama).
    return await check(provider, model=model, base_url=base_url, api_key_env=api_key_env, sign_in=True,
                       call_service=True, timeout_s=timeout_s, default_model=True)
