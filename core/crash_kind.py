"""What kind of failure stopped a quest, in words a scientist can act on (``quest_failed.md``, the to-do card).

A quest that stops on an error used to hand the person an exception and the last 80 lines of ``run.log`` and ask them
to "fix the cause". A domain scientist cannot debug FI, so the failure is sorted first:

* ``transient`` -- the model service or the network had a passing problem (a timeout, a dropped connection, a 429 rate
  limit or a 5xx, a file another program held for a moment). The engine tries the step again a few times by itself
  (:data:`RETRY_WAITS_S`) before it stops.
* ``setup`` -- something on this machine or account needs one action: sign in again, a used-up quota, a model name the
  service does not know, a program or package that is not installed, a full disk. The card names that one action.
* ``fi`` -- a programming error inside FI's own code (an ``AttributeError``, ``KeyError``, ``TypeError`` ... raised from
  FI's files). The card says plainly that this is FI's problem, not the person's settings or research.
* ``unknown`` -- none of the above: the card says FI did not expect it and that ``--resume`` continues from the step
  that stopped.

The technical detail (exception, provider, the log tail) stays in ``quest_failed.md``'s details section and ``run.log``.
Pure functions; no I/O except :func:`write`/:func:`read` of ``.fi/failure.json``.
"""
from __future__ import annotations

import asyncio
import errno
import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: How long the engine waits before each automatic retry of a step that failed on a passing problem. The provider has
#: already retried the call itself (seconds to ~2 minutes); these are the longer waits a service outage needs.
RETRY_WAITS_S: tuple[float, ...] = (60.0, 300.0)

#: The quest's own record of the last failure (read by the to-do card, the web quest page and VS Code).
FAILURE_FILE = "failure.json"

# FI's own source tree: a frame from one of these files (and not from an installed package under it) is FI's code.
_FI_ROOT = Path(__file__).resolve().parent.parent
_NOT_FI = (".venv", "site-packages", "dist-packages", "node_modules", ".pytest_tmp", "outputs")

#: Exception types that mean a mistake in the code that raised them, never a passing condition or a setting.
_PROGRAMMING_ERRORS = (AttributeError, KeyError, IndexError, TypeError, NameError, UnboundLocalError, AssertionError,
                       ZeroDivisionError, RecursionError, NotImplementedError)

_AUTH_WORDS = ("invalid api key", "invalid_api_key", "incorrect api key", "did not accept the api key", "http 401",
               "http 403", "unauthorized", "unauthorised",
               "authentication", "token may be invalid", "run '/login'", "re-authenticate", "not logged in",
               "please log in", "login required", "copilot requests", "permission denied for model")
_QUOTA_WORDS = ("monthly usage limit", "weekly limit", "session limit", "usage limit reached", "out of credits",
                "insufficient_quota", "insufficient quota", "exceeded your current quota", "exceeded_current_quota",
                "quota exceeded", "billing", "payment required", "credit balance", "add usage credits")
_MODEL_MISSING_WORDS = ("model not found", "model_not_found", "no such model", "unknown model", "does not exist",
                        "not a valid model", "pull model manifest", "is not available for your account")
_TRANSIENT_WORDS = ("timed out", "timeout", "temporarily", "temporary failure", "try again", "rate limit",
                    "ratelimit", "overloaded", "server error", "bad gateway", "service unavailable",
                    "connection reset", "connection aborted", "connection refused", "connection error",
                    "remote end closed", "remoteprotocolerror", "being used by another process")


@dataclass(frozen=True)
class Failure:
    """One failure as the person sees it: ``kind`` (see the module doc), ``say`` (what happened, one sentence),
    ``do`` (the one thing to do), ``detail`` (the technical one-liner for a bug report)."""

    kind: str
    say: str
    do: str
    detail: str
    node: str = ""
    retries: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _chain(exc: BaseException) -> list[BaseException]:
    """The exception and what caused it (``__cause__`` / ``__context__``), outermost first, at most 6 deep."""
    out: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and cur not in out and len(out) < 6:
        out.append(cur)
        cur = cur.__cause__ or cur.__context__
    return out


def _text(exc: BaseException) -> str:
    """The exception's message, any notes attached to it (provider/model context) and, for an HTTP error, the start of
    the service's answer, lower-cased."""
    notes = " ".join(str(n) for n in getattr(exc, "__notes__", ()) or ())
    body = ""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body = str(resp.text)[:2000]
        except Exception:  # noqa: BLE001 -- a streamed or closed response: no body to read
            body = ""
    return f"{type(exc).__name__} {exc} {notes} {body}".lower()


def _quota_used_up(exc: BaseException) -> bool:
    """A 429 the provider's own rule reads as a used-up quota or credit (``core.provider._is_exhausted_quota``)."""
    if _status(exc) != 429:
        return False
    try:
        from core import provider as _provider

        return bool(_provider._is_exhausted_quota(getattr(exc, "response", None)))
    except Exception:  # noqa: BLE001
        return False


def _status(exc: BaseException) -> int | None:
    resp = getattr(exc, "response", None)
    sc = getattr(resp, "status_code", None)
    return sc if isinstance(sc, int) else None


def _fi_frame(exc: BaseException) -> bool:
    """Whether the traceback runs through FI's own files (not an installed package and not the quest's code)."""
    try:
        frames = traceback.extract_tb(exc.__traceback__)
    except Exception:  # noqa: BLE001 -- an odd traceback object: not FI's, as far as we can tell
        return False
    for fr in frames:
        try:
            path = Path(fr.filename).resolve()
        except (OSError, ValueError):
            continue
        if any(part in _NOT_FI for part in path.parts):
            continue
        if path == _FI_ROOT or _FI_ROOT in path.parents:
            return True
    return False


def _one_line(exc: BaseException, limit: int = 200) -> str:
    msg = " ".join(str(exc).split())
    return (msg[: limit - 1] + "…") if len(msg) > limit else msg


def _program_name(exc: FileNotFoundError) -> str:
    """The program a failed start named: a bare command (``pandoc``, ``claude``) or an ``.exe``; a missing data file
    (a path with folders, or a file FI reads) is not a missing program."""
    name = str(getattr(exc, "filename", "") or "")
    if not name:
        return ""
    p = Path(name)
    if p.suffix.lower() in (".exe", ".cmd", ".bat") or (len(p.parts) == 1 and not p.suffix):
        return p.name
    return ""


def _setup(exc: BaseException, provider: str, model: str) -> tuple[str, str] | None:
    """(say, do) when the failure needs one action on this machine or account, else None."""
    text = _text(exc)
    status = _status(exc)
    service = provider or "the model service"
    if _quota_used_up(exc) or any(w in text for w in _QUOTA_WORDS):
        return (f"The account FI uses for {service} has used up its allowance (a usage limit or credit balance).",
                "Add credits or wait until the limit resets, then continue the quest.")
    if status in (401, 403) or any(w in text for w in _AUTH_WORDS):
        return (f"FI could not sign in to {service}.",
                "Sign in to it again (`fi --doctor` shows how for each service), then continue the quest.")
    if status == 404 or any(w in text for w in _MODEL_MISSING_WORDS):
        named = f"`{model}`" if model else "the model in the settings"
        return (f"{service} does not know the model {named}.",
                "Run `fi --doctor` to see the models this machine can use, set one in the quest's settings, then "
                "continue the quest.")
    if isinstance(exc, ModuleNotFoundError):
        name = getattr(exc, "name", "") or _one_line(exc)
        return (f"A Python package FI needs is not installed ({name}).",
                "Run `fi --doctor` to see what to install, install it, then continue the quest.")
    if isinstance(exc, FileNotFoundError) and _program_name(exc):
        return (f"A program FI needs was not found on this machine ({_program_name(exc)}).",
                "Run `fi --doctor` to see what to install, install it, then continue the quest.")
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC:
        return ("The disk is full.", "Free some space on the disk, then continue the quest.")
    if type(exc).__name__ == "AxonUnavailable":
        return ("The knowledge store (Axon) could not be reached.",
                "Start it (`fi --doctor` says how), then continue the quest.")
    if type(exc).__name__ == "_CliWedgeError":
        return (f"The command-line tool for {service} hung without answering.",
                "Try again later; if it keeps happening, choose another model in the quest's settings.")
    return None


def _transient(exc: BaseException) -> bool:
    """A passing problem: trying again later can fix it. Reuses the provider's own retry rules where they apply."""
    try:
        from core import provider as _provider  # lazy: the provider module is heavy and optional in some tests
    except Exception:  # noqa: BLE001
        _provider = None  # type: ignore[assignment]
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, PermissionError) and getattr(exc, "winerror", None) == 32:  # a file another program holds
        return True
    if _provider is not None:
        try:
            import httpx

            if isinstance(exc, (httpx.TransportError, httpx.HTTPStatusError)):
                return bool(_provider._retry_http_error(exc))
        except ImportError:
            pass
        if isinstance(exc, getattr(_provider, "_CliTransientError", ())):
            return bool(_provider._retry_cli_error(exc))
        if type(exc).__name__ == "BridgeError" and _provider._is_bridge_error_transient(str(exc)):
            return True
    status = _status(exc)
    if status is not None:
        return status == 429 or status >= 500
    return any(w in _text(exc) for w in _TRANSIENT_WORDS) and not isinstance(exc, _PROGRAMMING_ERRORS)


def classify(exc: BaseException, *, node: str = "", provider: str = "", model: str = "",
             retries: int = 0) -> Failure:
    """Sort one failure (see the module doc). Setup problems are checked first (a 429 that says the quota is used up
    is not a passing problem), then passing problems, then FI's own programming errors."""
    chain = _chain(exc)
    detail = "".join(traceback.format_exception_only(type(exc), exc)).strip().splitlines()[-1][:300]
    where = f" at the step `{node}`" if node and not node.startswith("(") else ""
    for e in chain:
        found = _setup(e, provider, model)
        if found:
            return Failure("setup", found[0], found[1], detail, node, retries)
    if any(_transient(e) for e in chain):
        tried = (f"FI tried the step again {retries} time(s) by itself and it still failed. " if retries else "")
        return Failure("transient",
                       f"The model service or the network had a passing problem{where} ({_one_line(exc, 120)}).",
                       f"{tried}Wait a few minutes, then continue the quest: it picks up at the step that stopped.",
                       detail, node, retries)
    if isinstance(exc, _PROGRAMMING_ERRORS) and _fi_frame(exc):
        return Failure("fi",
                       f"This is a problem in FI itself{where}, not in your settings or your research.",
                       "Continue the quest later: it picks up at the step that stopped. If it stops again at the same "
                       "step, report it to FI's maintainers with the file quest_failed.md.",
                       detail, node, retries)
    said = _one_line(exc)
    return Failure("unknown",
                   f"FI stopped{where} on an error it did not expect" + (f": {said}" if said else "."),
                   "Continue the quest: it picks up at the step that stopped. If it stops again at the same step, "
                   "report it to FI's maintainers with the file quest_failed.md.",
                   detail, node, retries)


TITLES = {
    "transient": "The quest stopped on a passing problem",
    "setup": "The quest stopped: one thing to set up",
    "fi": "The quest stopped on a problem in FI",
    "unknown": "The quest stopped on an unexpected error",
}


def write(fi_dir: Path, failure: Failure) -> None:
    """``.fi/failure.json``: the failure as the person sees it, for the to-do card, the web page and VS Code."""
    fi_dir = Path(fi_dir)
    fi_dir.mkdir(parents=True, exist_ok=True)
    (fi_dir / FAILURE_FILE).write_text(json.dumps({**failure.as_dict(), "title": TITLES.get(failure.kind, "")},
                                                  indent=1), encoding="utf-8")


def read(fi_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((Path(fi_dir) / FAILURE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("say") else None


def clear(fi_dir: Path) -> None:
    try:
        (Path(fi_dir) / FAILURE_FILE).unlink(missing_ok=True)
    except OSError:
        pass
