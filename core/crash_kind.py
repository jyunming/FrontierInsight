"""What kind of failure stopped a quest, in words a scientist can act on (``quest_failed.md``, the to-do card).

A quest that stops on an error used to hand the person an exception and the last 80 lines of ``run.log`` and ask them
to "fix the cause". The person running FI decides the research question, its inputs and what result they want, not how
a bug is fixed, so the failure is sorted first and only an action on their own machine or account is ever asked of them:

* ``transient`` -- the model service or the network had a passing problem (a timeout, a dropped connection, a 429 rate
  limit or a 5xx the provider's own retries did not outlast). The engine tries the step again by itself
  (:data:`RETRY_WAITS_S`) before it stops; nothing is asked of the person but to continue later.
* ``setup`` -- one action on this machine or account: sign in again, a used-up allowance, a model the service does not
  know, a program or package that is not installed, a local model service that is not running, a file another program
  holds open, a full disk. The card names that one action, in one plain sentence.
* ``fi`` -- a programming error raised inside FI's own code. Said plainly to be FI's problem, not the person's; nothing
  for them to fix.
* ``unknown`` -- none of the above. Nothing for them to debug either: continue later, the details are kept.

The technical detail (exception, provider, the log tail) stays in ``quest_failed.md``'s details section and ``run.log``.
Pure functions; no I/O except :func:`write` / :func:`read` / :func:`clear` of ``.fi/failure.json``.
"""
from __future__ import annotations

import asyncio
import errno
import json
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: How long the engine waits before each automatic retry of a step that failed on a passing problem. The provider has
#: already retried the call itself (seconds to ~2 minutes); these are the longer waits a service outage needs.
RETRY_WAITS_S: tuple[float, ...] = (60.0, 300.0)

#: A step that had been running longer than this when it failed is not run again by itself (a long experiment is never
#: repeated twice over for a problem that may not have passed): the quest stops and the person continues it.
RETRY_ONLY_UNDER_S = 600.0

#: The quest's own record of the last failure (read by the to-do card, the web quest page, the terminal and VS Code).
FAILURE_FILE = "failure.json"

# FI's own code: the package folders and the launcher, wherever FI is installed (a checkout or site-packages).
_CORE = Path(__file__).resolve().parent
_FI_PLACES = (_CORE, _CORE.parent / "generation", _CORE.parent / "web")
_FI_FILES = (_CORE.parent / "launch.py",)

#: Exception types that mean a mistake in the code that raised them, never a passing condition or a setting.
_PROGRAMMING_ERRORS = (AttributeError, KeyError, IndexError, TypeError, NameError, UnboundLocalError, AssertionError,
                       ZeroDivisionError, RecursionError, NotImplementedError)

_AUTH_WORDS = ("invalid api key", "invalid_api_key", "incorrect api key", "did not accept the api key", "http 401",
               "http 403", "unauthorized", "unauthorised", "authentication", "token may be invalid", "run '/login'",
               "re-authenticate", "not logged in", "please log in", "login required", "copilot requests",
               "permission denied for model")
#: Read only from an error that carries no HTTP answer (a CLI's or the bridge's text): an HTTP 429 is judged by the
#: provider's own rule, which reads the error's type and code and its Retry-After (a per-minute limit is not a quota).
_QUOTA_WORDS = ("monthly usage limit", "weekly limit", "session limit", "usage limit reached", "out of credits",
                "insufficient_quota", "insufficient quota", "exceeded your current quota", "exceeded_current_quota",
                "payment required", "credit balance", "add usage credits")
_MODEL_MISSING_WORDS = ("model not found", "model_not_found", "no such model", "unknown model", "not a valid model",
                        "pull model manifest", "is not available for your account")
#: Too general alone ("file ... does not exist" in a CLI's stderr): read as a missing model only from an HTTP answer or
#: when the text names the model.
_MODEL_MISSING_LOOSE = ("does not exist",)
_TRANSIENT_WORDS = ("timed out", "timeout", "temporarily", "temporary failure", "try again", "rate limit",
                    "ratelimit", "overloaded", "server error", "bad gateway", "service unavailable",
                    "connection reset", "connection aborted", "remote end closed", "remoteprotocolerror")
_BRIDGE_GONE = ("bridge connection dropped", "bridge write failed", "bridge closed", "bridge is not connected")


@dataclass(frozen=True)
class Failure:
    """One failure as the person sees it: ``kind`` (see the module doc), ``say`` (what happened, one sentence),
    ``do`` (what happens next or the one thing to do), ``detail`` (the technical one-liner for a bug report)."""

    kind: str
    say: str
    do: str
    detail: str
    node: str = ""
    retries: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _chain(exc: BaseException) -> list[BaseException]:
    """The exception and what caused it, outermost first, at most 6 deep: an explicit cause (``raise ... from``) always;
    the exception being handled when it was raised only when that was not suppressed and the outer one is not a
    programming error (a bug raised inside an ``except`` is the bug, not the error it was handling)."""
    out: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and cur not in out and len(out) < 6:
        out.append(cur)
        if cur.__cause__ is not None:
            cur = cur.__cause__
        elif cur.__context__ is not None and not cur.__suppress_context__ and not isinstance(cur, _PROGRAMMING_ERRORS):
            cur = cur.__context__
        else:
            cur = None
    return out


def _response(exc: BaseException) -> Any:
    return getattr(exc, "response", None)


def _status(exc: BaseException) -> int | None:
    sc = getattr(_response(exc), "status_code", None)
    return sc if isinstance(sc, int) else None


def _text(exc: BaseException) -> str:
    """The exception's message, any notes attached to it (provider/model context) and, for an HTTP error, the start of
    the service's answer, lower-cased."""
    try:
        notes = " ".join(str(n) for n in getattr(exc, "__notes__", ()) or ())
        body = ""
        resp = _response(exc)
        if resp is not None:
            try:
                body = str(resp.text)[:2000]
            except Exception:  # noqa: BLE001 -- a streamed or closed response: no body to read
                body = ""
        return f"{type(exc).__name__} {exc} {notes} {body}".lower()
    except Exception:  # noqa: BLE001 -- an exception whose str() fails: its type name is all there is
        return type(exc).__name__.lower()


def _is_model_call(exc: BaseException) -> bool:
    """An error from a call to a model: an HTTP error or transport error, a CLI or bridge error the provider raised, or
    any error the provider marked with its ``[FI] provider=`` note. Only these can mean "the service does not know the
    model" (a missing input file or a docker 404 cannot)."""
    if type(exc).__module__.startswith("core.provider") or type(exc).__name__ in ("BridgeError", "_CliTransientError"):
        return True
    return any(str(n).startswith("[FI] provider=") for n in getattr(exc, "__notes__", ()) or ())


def _fi_runs_it(provider: str) -> bool:
    """Whether FI itself starts the local service for this provider (a proxy ``ProxySupervisor`` runs)."""
    try:
        from core.provider import PROXY_PROVIDERS

        return provider in PROXY_PROVIDERS
    except Exception:  # noqa: BLE001
        return False


def _host(exc: BaseException) -> str:
    """The host an HTTP error was for ('' when it carries no request: httpx raises when ``.request`` is unset)."""
    try:
        return str(exc.request.url.host or "")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return ""


def _inputs_problem(exc: BaseException, text: str) -> bool:
    """A file the person named in the quest's settings (``execution.inputs``, ``knowledge.local_papers``) that is not
    there or cannot be used: theirs to put right, so it is a setup problem, not FI's."""
    if not isinstance(exc, (FileNotFoundError, ValueError)):
        return False
    try:
        frames = traceback.extract_tb(exc.__traceback__)
        raised_in = Path(frames[-1].filename).name if frames else ""
    except Exception:  # noqa: BLE001
        raised_in = ""
    return raised_in == "example_inputs.py" or text.split(" ", 1)[-1].startswith("execution.inputs")


#: How FI's own refusals of a setting the person chose begin (the message's first words, lower-cased): the setting is
#: theirs, so changing it or setting up what it needs is their one action. Only these exact openings count, read from
#: the start of the message alone (never its notes, a service's answer, or a file name further on).
_SETTING_REFUSALS = (
    "engine.skills_required names",
    "[preflight] paper_pdf requested with output.require_pdf",
    "execution.sandbox=docker requires",
    "docker daemon not reachable",
    "venv creation failed via",
)


def _settings_refusal(exc: BaseException) -> tuple[str, str] | None:
    """(say, do) for FI's own refusal of a setting the person chose, else None. Only a ``RuntimeError`` or
    ``ValueError`` FI raised with one of :data:`_SETTING_REFUSALS` as its opening; never a programming error, an OS
    error or a provider's error (a CLI's output can say anything)."""
    if (isinstance(exc, (_PROGRAMMING_ERRORS, OSError)) or not isinstance(exc, (RuntimeError, ValueError))
            or _is_model_call(exc)):
        return None
    head = " ".join(_one_line(exc, 400).split()).lower()
    if not head.startswith(_SETTING_REFUSALS):
        return None
    if head.startswith("execution.sandbox=docker requires"):
        return ("The quest's settings run the experiment in Docker, but the Python package FI uses to reach Docker is "
                "not installed.", "Install it (`pip install docker`), then continue the quest.")
    if head.startswith("docker daemon not reachable"):
        return ("The quest's settings run the experiment in Docker, which is not installed or not running here.",
                "Start Docker (Docker Desktop on Windows or macOS), or install it, then continue the quest.")
    if head.startswith("[preflight] paper_pdf"):
        return ("The quest's settings require a PDF of the paper, but the programs that make one are not installed "
                "here.", "Install what the setup check (`python launch.py --doctor`) names for PDFs, then continue the "
                         "quest.")
    if head.startswith("venv creation failed via"):
        version = re.search(r"python_version=([^)\s]+)", head)
        named = f" ({version.group(1)})" if version else ""
        return (f"The Python version the quest's settings ask for{named} could not be set up on this machine.",
                "Install that Python version or change `execution.python_version` in the settings, then continue the "
                "quest.")
    return (f"A setting of the quest cannot be used here: {_one_line(exc, 160)}",
            "Change that setting in the quest's settings, then continue the quest.")


def _quota_used_up(exc: BaseException) -> bool:
    """A 429 the provider's own rule reads as a used-up quota or credit (``core.provider._is_exhausted_quota``)."""
    if _status(exc) != 429:
        return False
    try:
        from core import provider as _provider

        return bool(_provider._is_exhausted_quota(_response(exc)))
    except Exception:  # noqa: BLE001
        return False


def _program_name(exc: FileNotFoundError) -> str | None:
    """The program a failed start named ('' when the start named none: Windows reports a missing program as
    ``[WinError 2]`` with no file name), or None when this is not a failed start of a program (a missing data file,
    or FI's own environment, whose loss is not something to install)."""
    try:
        frames = traceback.extract_tb(exc.__traceback__)
    except Exception:  # noqa: BLE001
        frames = []
    starting = bool(frames) and Path(frames[-1].filename).name in ("subprocess.py", "base_subprocess.py",
                                                                    "windows_utils.py", "unix_events.py")
    name = str(getattr(exc, "filename", "") or "")
    if not starting or ".venv" in name.replace("\\", "/").split("/"):
        return None
    return Path(name).name if name else ""


def _setup(exc: BaseException, provider: str, model: str) -> tuple[str, str] | None:
    """(say, do) when the failure needs one action on this machine or account, else None."""
    text = _text(exc)
    status = _status(exc)
    service = provider or "the model service"
    model_call = _is_model_call(exc)
    has_http_answer = status is not None
    if _inputs_problem(exc, text):
        return (f"A file named in the quest's settings cannot be used: {_one_line(exc, 160)}",
                "Put the file in place or change the settings to name the right one, then continue the quest.")
    refused = _settings_refusal(exc)
    if refused:
        return refused
    if model_call and (_quota_used_up(exc) or (not has_http_answer and any(w in text for w in _QUOTA_WORDS))):
        return (f"The account FI uses for {service} has used up its allowance (a usage limit or credit balance).",
                "Add credits or wait until the limit resets, then continue the quest.")
    if model_call and (status in (401, 403) or (not has_http_answer and any(w in text for w in _AUTH_WORDS))):
        return (f"FI could not sign in to {service}.",
                "Sign in to it again (the setup check, `python launch.py --doctor`, shows how), then continue the quest.")
    names_model = bool(model) and model.lower() in text
    if model_call and (status == 404 or any(w in text for w in _MODEL_MISSING_WORDS)
                       or ((has_http_answer or names_model) and any(w in text for w in _MODEL_MISSING_LOOSE))):
        named = f"`{model}`" if model else "named in the settings"
        return (f"{service} does not know the model {named}.",
                "Choose a model this machine can use (the setup check, `python launch.py --doctor`, lists them), then continue the quest.")
    try:
        import httpx

        if isinstance(exc, (httpx.UnsupportedProtocol, httpx.LocalProtocolError)):
            return (f"The address set for {service} is not a valid web address.",
                    "Correct the address in the quest's settings (the setup check, `python launch.py --doctor`, checks it), then continue the quest.")
        host = _host(exc)
        if isinstance(exc, httpx.ConnectError) and host.strip("[]") in ("127.0.0.1", "localhost", "::1"):
            if model_call and _fi_runs_it(provider):
                return None  # FI's own helper for this provider: FI starts it again on resume, nothing to start
            if model_call:
                return (f"The model service on this machine ({service}) is not running.",
                        "Start it, then continue the quest.")
            port = ""
            try:
                port = f":{exc.request.url.port}" if exc.request.url.port else ""  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                port = ""
            where = f"[{host.strip('[]')}]" if ":" in host else host
            return (f"A service on this machine that FI uses ({where}{port}) is not running.",
                    "Start it, then continue the quest.")
    except ImportError:
        pass
    if any(w in text for w in _BRIDGE_GONE):
        return ("VS Code's connection to the model closed (VS Code was closed or reloaded).",
                "Continue the quest from VS Code (`@fi /resume`).")
    if isinstance(exc, PermissionError) and getattr(exc, "winerror", None) == 32:
        held = Path(str(getattr(exc, "filename", "") or "")).name
        return (f"Another program has a file of the quest open{f' ({held})' if held else ''}.",
                "Close that file, then continue the quest.")
    if isinstance(exc, ModuleNotFoundError):
        name = getattr(exc, "name", "") or ""
        return (f"A Python package FI needs is not installed{f' ({name})' if name else ''}.",
                "Install what the setup check (`python launch.py --doctor`) names, then continue the quest.")
    if isinstance(exc, FileNotFoundError):
        program = _program_name(exc)
        if program is not None:
            return (f"A program FI needs was not found on this machine{f' ({program})' if program else ''}.",
                    "Install what the setup check (`python launch.py --doctor`) names, then continue the quest.")
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return ("The disk is full.", "Free some space on the disk, then continue the quest.")
    if type(exc).__name__ == "AxonUnavailable":
        return ("The knowledge store (Axon) could not be reached.",
                "Start it (the setup check, `python launch.py --doctor`, says how), then continue the quest.")
    return None


def _transient(exc: BaseException) -> bool:
    """A passing problem: trying again later can fix it. Reuses the provider's own retry rules where they apply."""
    if isinstance(exc, _PROGRAMMING_ERRORS):
        return False
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    try:
        from core import provider as _provider  # lazy: heavy, and optional in some tests
    except Exception:  # noqa: BLE001
        _provider = None  # type: ignore[assignment]
    if _provider is not None:
        try:
            import httpx

            if isinstance(exc, (httpx.TransportError, httpx.HTTPStatusError)):
                return bool(_provider._retry_http_error(exc))
        except ImportError:
            pass
        if type(exc).__name__ == "_CliWedgeError":  # hangs again the same way (the provider says so)
            return False
        if isinstance(exc, getattr(_provider, "_CliTransientError", ())):
            return bool(_provider._retry_cli_error(exc))
        if type(exc).__name__ == "BridgeError" and _provider._is_bridge_error_transient(str(exc)):
            return True
    status = _status(exc)
    if status is not None:
        return status == 429 or status >= 500
    # The words alone ("timeout", "try again") are read only from a call to a model: in another error (one of FI's own
    # messages, say) they would cost two retries and a card that blames the network.
    return _is_model_call(exc) and any(w in _text(exc) for w in _TRANSIENT_WORDS)


def _fi_frame(exc: BaseException) -> bool:
    """Whether the innermost frame (where the error was raised) is in FI's own code. A frame further out is always
    FI's (the engine runs every step), so only the innermost one says whose code went wrong."""
    try:
        frames = traceback.extract_tb(exc.__traceback__)
    except Exception:  # noqa: BLE001
        return False
    if not frames:
        return False
    try:
        path = Path(frames[-1].filename).resolve()
    except (OSError, ValueError):
        return False
    return path in _FI_FILES or any(place == path.parent or place in path.parents for place in _FI_PLACES)


def _one_line(exc: BaseException, limit: int = 200) -> str:
    try:
        msg = " ".join(str(exc).split())
    except Exception:  # noqa: BLE001
        msg = ""
    return (msg[: limit - 1] + "…") if len(msg) > limit else msg


def _detail(exc: BaseException) -> str:
    """``Type: message`` on one line (without the notes Python 3.11 prints after it)."""
    return f"{type(exc).__module__}.{type(exc).__qualname__}: {_one_line(exc, 280)}".removeprefix("builtins.")


def classify(exc: BaseException, *, node: str = "", provider: str = "", model: str = "",
             retries: int = 0) -> Failure:
    """Sort one failure (see the module doc). Setup problems are checked first (a 429 that says the quota is used up
    is not a passing problem), then passing problems, then FI's own programming errors. Never raises: a failure to sort
    is an unexpected error."""
    try:
        return _classify(exc, node=node, provider=provider, model=model, retries=retries)
    except Exception:  # noqa: BLE001 -- sorting must never hide the error it sorts
        return Failure("unknown", "FI stopped on an error it did not expect.", _NOTHING_TO_FIX,
                       type(exc).__name__, node, retries)


_NOTHING_TO_FIX = ("Nothing for you to fix: continue the quest later and it picks up at the step that stopped. The "
                   "details are kept for FI's maintainers in quest_failed.md.")


def _classify(exc: BaseException, *, node: str, provider: str, model: str, retries: int) -> Failure:
    chain = _chain(exc)
    detail = _detail(exc)
    where = f" at the step `{node}`" if node and not node.startswith("(") else ""
    for e in chain:
        found = _setup(e, provider, model)
        if found:
            return Failure("setup", found[0], found[1], detail, node, retries)
    if any(_transient(e) for e in chain):
        tried = f"FI already tried the step again {retries} time(s) by itself. " if retries else ""
        return Failure("transient",
                       f"The model service or the network had a passing problem{where}.",
                       f"{tried}Nothing for you to fix: continue the quest in a few minutes and it picks up at the "
                       f"step that stopped.",
                       detail, node, retries)
    if isinstance(exc, _PROGRAMMING_ERRORS) and _fi_frame(exc):
        return Failure("fi", f"This is a problem in FI itself{where}, not in your settings or your research.",
                       _NOTHING_TO_FIX, detail, node, retries)
    said = _one_line(exc)
    return Failure("unknown", f"FI stopped{where} on an error it did not expect" + (f": {said}" if said else "."),
                   _NOTHING_TO_FIX, detail, node, retries)


TITLES = {
    "transient": "The quest stopped on a passing problem",
    "setup": "The quest stopped: one thing to set up",
    "fi": "The quest stopped on a problem in FI",
    "unknown": "The quest stopped on an unexpected error",
}


def write(fi_dir: Path, failure: Failure) -> None:
    """``.fi/failure.json``: the failure as the person sees it, with the time it was written (``at``, so a reader can
    tell this run's failure from an earlier one's)."""
    fi_dir = Path(fi_dir)
    fi_dir.mkdir(parents=True, exist_ok=True)
    record = {**failure.as_dict(), "title": TITLES.get(failure.kind, ""), "at": time.time()}
    (fi_dir / FAILURE_FILE).write_text(json.dumps(record, indent=1), encoding="utf-8")


def read(fi_dir: Path, *, since: float | None = None) -> dict[str, Any] | None:
    """The record, or None when there is none, it cannot be read, or (with ``since``) it was written before then."""
    try:
        data = json.loads((Path(fi_dir) / FAILURE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("say"):
        return None
    if since is not None and float(data.get("at") or 0) < since:
        return None
    return data


def clear(fi_dir: Path) -> None:
    try:
        (Path(fi_dir) / FAILURE_FILE).unlink(missing_ok=True)
    except OSError:
        pass
