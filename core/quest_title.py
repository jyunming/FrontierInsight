"""A quest's title, and changing it after the quest has run.

The title the model picked (or the one suggested at the start) can be a poor one. :func:`rename` sets a finished or
paused quest's title in every place it lives: the paper's title line (``paper/paper.md``, and an older top-level
``paper.md``), ``config.yaml``'s ``title``, ``frontier_insight_summary.json`` and the latest saved state, which a resume or
``--rerun`` reads (``.fi/state.sqlite``; a ``--from <step>`` rerun starts from an earlier one, which keeps the old
title). The change is recorded in the quest's audit trace with the paper's hash before and after, which
``core/evidence.py`` follows from a finished quest's seal. Results, data and code are
never touched. Already-rendered outputs (paper.pdf, slides, poster, talk) keep the old title until they are made again;
:attr:`RenameResult.outputs_to_redo` names them, and :meth:`RenameResult.lines` says how.

The CLI (``fi tools rename``), the web quest page (``POST /api/quests/<id>/title``) and VS Code (``@fi /rename``) all
call this one function.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_LENGTH = 200

# The same first-``# ``-heading rule the paper generator uses to lift the title (generation/paper.py:_FIRST_H1_RE).
_FIRST_H1_RE = re.compile(r"^# +(.+?)[ \t]*(?=\r?$)", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?\r?\n)---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_FM_TITLE_RE = re.compile(r"^title:[ \t]*([^\r\n]*)", re.MULTILINE)
_YAML_TITLE_RE = re.compile(r"^title:[^\r\n]*", re.MULTILINE)

# Each output a person may already have, the kind that makes it again (``--emit <kind>``), and the files that show it.
_OUTPUTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("paper_pdf", ("paper.pdf",)),
    ("slides", ("slides.pdf", "slides.html", "slides.pptx")),
    ("poster", ("poster.pdf",)),
    ("speech", ("talk.md",)),
)

_RUNNING_WINDOW_S = 300.0

# The paper a finished quest's seal names (core/engine.py:_seal_trace).
_SEALED_PAPER = "paper/paper.md"


class RenameRefused(ValueError):
    """The title cannot be changed: the reason is a sentence to show the person as it is."""


class QuestRunning(RenameRefused):
    """The quest is still running: its files are still being written."""


@dataclass
class RenameResult:
    quest_id: str
    old: str | None
    new: str
    changed: list[str] = field(default_factory=list)   # files changed, relative to the quest folder
    saved_state: bool = False                          # the saved state a resume reads now has the new title
    state_problem: str = ""                            # why a saved state that exists was not updated
    trace_problem: str = ""                            # why the change is not in the quest's trace
    outputs_to_redo: list[str] = field(default_factory=list)
    quest_root: str = ""

    def lines(self) -> list[str]:
        """What happened, in a few plain lines, for the CLI and VS Code."""
        out = [f"Title of {self.quest_id} is now: {self.new}"]
        if self.old:
            out.append(f"  (was: {self.old})")
        if self.changed:
            out.append(f"  Updated: {', '.join(self.changed)}.")
        if self.state_problem:
            out.append(f"  The saved state was not updated ({self.state_problem}): resuming the quest would use the old "
                       "title; rename it again once nothing else has the quest open.")
        if self.trace_problem:
            out.append(f"  Note: {self.trace_problem}.")
        if self.outputs_to_redo:
            out.append("  Already-made outputs still show the old title. To make them again with the new one:")
            # The quest's own folder and the folder that holds it, so the command works from any directory.
            where = (f'"{self.quest_root}" --output "{Path(self.quest_root).parent}"' if self.quest_root
                     else self.quest_id)
            for kind in self.outputs_to_redo:
                out.append(f"    python launch.py --resume {where} --emit {kind}")
        return out

    def as_dict(self) -> dict[str, Any]:
        return {"quest_id": self.quest_id, "old_title": self.old, "title": self.new, "changed": self.changed,
                "saved_state": self.saved_state, "state_problem": self.state_problem, "trace_problem": self.trace_problem,
                "outputs_to_redo": self.outputs_to_redo}


def clean(title: Any) -> str:
    """The title as it will be saved, or :class:`RenameRefused` saying what is wrong with it."""
    if title is not None and not isinstance(title, str):
        raise RenameRefused("The title must be text.")
    text = title or ""
    if "\n" in text or "\r" in text:
        raise RenameRefused("The title must be one line.")
    text = " ".join(text.split())
    if not text:
        raise RenameRefused("The title is empty.")
    if len(text) > MAX_LENGTH:
        raise RenameRefused(f"The title is {len(text)} characters; keep it to {MAX_LENGTH} or fewer.")
    import unicodedata

    if any(unicodedata.category(c) in ("Cc", "Cf", "Cs") for c in text):
        raise RenameRefused("The title has an invisible control character in it.")
    return text


def _paper_title(text: str) -> str | None:
    fm = _FRONTMATTER_RE.match(text)
    if fm:
        m = _FM_TITLE_RE.search(fm.group(1))
        if m:
            raw = m.group(1).strip()
            try:
                import yaml

                value = yaml.safe_load(raw)
                return (str(value).strip() or None) if value is not None else None
            except Exception:  # noqa: BLE001 -- an odd front matter still has a readable title
                return raw.strip("'\"") or None
    m = _FIRST_H1_RE.search(text, fm.end() if fm else 0)
    return (m.group(1).strip() or None) if m else None


def _config_title(quest_root: Path) -> str | None:
    try:
        import yaml

        data = yaml.safe_load((quest_root / "config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 -- no or unreadable config: no title from it
        return None
    value = data.get("title") if isinstance(data, dict) else None
    return (str(value).strip() or None) if value else None


def current_title(quest_root: Path) -> str | None:
    """The title a reader sees: the paper's, else the one in ``config.yaml``, else ``None``."""
    for rel in ("paper/paper.md", "paper.md"):
        try:
            found = _paper_title((quest_root / rel).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
        if found:
            return found
    return _config_title(quest_root)


def looks_running(quest_root: Path, *, now: float | None = None) -> bool:
    """Whether the quest seems to be running now: it has not stopped for a person, and its ``run.log`` changed in the
    last five minutes while it has no final summary (or the log is newer than the summary: it was resumed)."""
    fi = quest_root / ".fi"
    if (fi / "pause.json").is_file() or (quest_root / "NEXT_STEP.md").is_file():
        return False
    try:
        log_mtime = (fi / "run.log").stat().st_mtime
    except OSError:
        return False
    now = time.time() if now is None else now
    if now - log_mtime >= _RUNNING_WINDOW_S:
        return False
    try:
        summary_mtime = (quest_root / "frontier_insight_summary.json").stat().st_mtime
    except OSError:
        return True
    if log_mtime <= summary_mtime + 5:
        return False
    # The log is newer than the summary: a resumed run, or only an output made again afterwards (`--emit`), which
    # also writes to run.log but runs no step. A resumed run always starts a step in the trace after the summary.
    from core import audit_log

    from datetime import datetime

    for e in reversed(audit_log.read(fi / "audit.jsonl")):
        try:
            ts = datetime.fromisoformat(str(e.get("ts"))).timestamp()
        except ValueError:
            continue
        if ts <= summary_mtime:
            break
        if e.get("kind") == "node_started":
            return True
    return False


def _retitle_paper(text: str, title: str) -> str:
    """The paper with its title changed and nothing else: the front matter's ``title`` when it has one, else the first
    ``# `` heading, else a heading added at the top."""
    fm = _FRONTMATTER_RE.match(text)
    if fm and _FM_TITLE_RE.search(fm.group(1)):
        import yaml

        body = fm.group(1)
        line = f"title: {json.dumps(title, ensure_ascii=False)}"
        new_body = _FM_TITLE_RE.sub(lambda _m: line, body, count=1)
        try:
            ok = (yaml.safe_load(new_body) or {}).get("title") == title
        except (yaml.YAMLError, AttributeError):
            ok = False
        if not ok:
            raise RenameRefused("The paper's front matter writes its title over several lines; change it by hand.")
        out = text[:fm.start(1)] + new_body + text[fm.end(1):]
        # A heading under the front matter that repeats the old title is the title too (the poster reads the heading).
        old = _paper_title(text)
        body_start = fm.start(1) + len(new_body) + (fm.end() - fm.end(1))
        h1 = _FIRST_H1_RE.search(out, body_start)
        if old and h1 and h1.group(1).strip() == old:
            out = out[:h1.start()] + f"# {title}" + out[h1.end():]
        return out
    start = fm.end() if fm else 0
    m = _FIRST_H1_RE.search(text, start)
    if m:
        return text[:m.start()] + f"# {title}" + text[m.end():]
    return text[:start] + f"# {title}\n\n" + text[start:]


def _retitle_yaml(text: str, title: str) -> str:
    """``config.yaml`` with its ``title`` changed and nothing else: a one-line edit keeps its comments and order. A title
    written some other way (over several lines, or a quoted key) falls back to writing the whole mapping again."""
    import yaml

    try:
        before = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        raise RenameRefused(f"config.yaml cannot be read ({str(e).splitlines()[0]}); fix it by hand first.") from e
    if not isinstance(before, dict):
        raise RenameRefused("config.yaml is not a YAML mapping; fix it by hand first.")
    expected = {**before, "title": title}
    line = f"title: {json.dumps(title, ensure_ascii=False)}"
    if _YAML_TITLE_RE.search(text):
        edited: str | None = _YAML_TITLE_RE.sub(lambda _m: line, text, count=1)
    elif "title" in before:
        edited = None  # written in a way the one-line edit cannot see: adding a line would make a second title
    else:
        m = re.search(r"^topic:", text, re.MULTILINE)
        edited = (text[:m.start()] + line + "\n" + text[m.start():]) if m else line + "\n" + text
    if edited is not None:
        try:
            after = yaml.safe_load(edited) or {}
        except yaml.YAMLError:
            after = None
        if isinstance(after, dict) and after == expected and len(_YAML_TITLE_RE.findall(edited)) == 1:
            return edited
    return yaml.safe_dump(expected, sort_keys=False, indent=2, allow_unicode=True)


async def _update_saved_state(quest_root: Path, title: str) -> bool:
    """Put the new title into the latest saved state, in place, so a resume (or ``--rerun``) reads it. The checkpoint
    keeps its id (and so its pending writes); only the two title values change. A ``--from <step>`` rerun starts from an
    earlier saved state, which keeps the old title."""
    sqlite_path = quest_root / ".fi" / "state.sqlite"
    if not sqlite_path.is_file():
        return False
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    config = {"configurable": {"thread_id": quest_root.name, "checkpoint_ns": ""}}
    async with AsyncSqliteSaver.from_conn_string(str(sqlite_path)) as saver:
        current = await saver.aget_tuple(config)
        if current is None or current.checkpoint is None:
            return False
        checkpoint = dict(current.checkpoint)
        values = dict(checkpoint.get("channel_values") or {})
        values["title"] = title
        values["title_confirmed"] = True
        checkpoint["channel_values"] = values
        parent = (current.parent_config or {}).get("configurable", {}).get("checkpoint_id")
        put_config = {"configurable": {"thread_id": quest_root.name, "checkpoint_ns": "",
                                       **({"checkpoint_id": parent} if parent else {})}}
        await saver.aput(put_config, checkpoint, current.metadata or {}, {})
    return True


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _read(path: Path) -> str:
    """A file's text exactly as it is on disk (its line endings kept), so writing it back changes nothing else."""
    return path.read_bytes().decode("utf-8")


def _write_all(edits: list[tuple[str, Path, bytes, bytes]]) -> None:
    """Write every file, or none: each goes to a temporary file first, and one that cannot be written puts the files
    already written back byte for byte."""
    done: list[tuple[Path, bytes]] = []
    for _rel, path, old_bytes, new_bytes in edits:
        tmp = path.with_name(path.name + ".renaming")
        try:
            tmp.write_bytes(new_bytes)
            os.replace(tmp, path)
            done.append((path, old_bytes))
        except OSError as e:
            not_restored = []
            for written, before in reversed(done):
                try:
                    written.write_bytes(before)
                except OSError:
                    not_restored.append(written.name)
            for _r, p, _o, _n in edits:
                try:
                    p.with_name(p.name + ".renaming").unlink(missing_ok=True)
                except OSError:
                    pass
            tail = (f" These could not be put back as they were: {', '.join(not_restored)}." if not_restored
                    else " Nothing was changed.")
            raise RenameRefused(f"{path.name} could not be written ({e.strerror or e}); close it where it is open and "
                                f"try again.{tail}") from e


def rename(quest_root: Path, new_title: Any) -> RenameResult:
    """Set the title of the quest in ``quest_root``. Raises :class:`RenameRefused` (nothing changed) when the title is
    not usable, the folder is not a quest, a file cannot be read or written, or the quest is still running."""
    quest_root = Path(quest_root).resolve()
    title = clean(new_title)
    if not (quest_root / ".fi").is_dir():
        raise RenameRefused(f"No quest at {quest_root}.")
    if looks_running(quest_root):
        raise QuestRunning(f"Quest {quest_root.name} is still running; rename it once it has stopped or finished.")
    trace = quest_root / ".fi" / "audit.jsonl"
    # The change is recorded in the trace, and a finished quest's seal reads the paper through that record: a trace
    # that cannot be written now would leave a changed paper with nothing to account for it.
    try:
        with trace.open("ab"):
            pass
    except OSError as e:
        raise RenameRefused(f"The quest's trace ({trace.name}) cannot be written ({e.strerror or e}); close it where it "
                            "is open and try again. Nothing was changed.") from e
    old = current_title(quest_root)
    result = RenameResult(quest_id=quest_root.name, old=old, new=title, quest_root=str(quest_root))
    paper_before = _sha256(quest_root / _SEALED_PAPER)

    # Every new text is worked out (and encoded) before anything is written, and then all are written or none.
    edits: list[tuple[str, Path, bytes, bytes]] = []

    def _add(rel: str, path: Path, old_text: str, new_text: str) -> None:
        if new_text != old_text:
            edits.append((rel, path, old_text.encode("utf-8"), new_text.encode("utf-8")))

    try:
        for rel in ("paper/paper.md", "paper.md"):
            path = quest_root / rel
            if path.is_file():
                text = _read(path)
                _add(rel, path, text, _retitle_paper(text, title))
        cfg_path = quest_root / "config.yaml"
        if cfg_path.is_file():
            text = _read(cfg_path)
            _add("config.yaml", cfg_path, text, _retitle_yaml(text, title))
        summary_path = quest_root / "frontier_insight_summary.json"
        if summary_path.is_file():
            text = _read(summary_path)
            try:
                summary = json.loads(text)
            except ValueError:
                summary = None  # a summary that cannot be read is left alone; the next finish writes it again
            if isinstance(summary, dict) and summary.get("title") != title:
                _add("frontier_insight_summary.json", summary_path, text,
                     json.dumps({**summary, "title": title}, indent=2, ensure_ascii=False))
    except (OSError, UnicodeError) as e:
        raise RenameRefused(f"A file of the quest cannot be read ({e}); nothing was changed.") from e
    _write_all(edits)
    result.changed.extend(rel for rel, *_ in edits)

    if (quest_root / ".fi" / "state.sqlite").is_file():
        try:
            result.saved_state = _run(_update_saved_state(quest_root, title))
            if not result.saved_state:
                result.state_problem = "it holds no saved step for this quest"
        except Exception as e:  # noqa: BLE001 -- the files are renamed; the saved state is reported, not fatal
            result.state_problem = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
    if result.saved_state:
        result.changed.append(".fi/state.sqlite")

    result.outputs_to_redo = [kind for kind, names in _OUTPUTS if any((quest_root / n).is_file() for n in names)]
    if not edits:
        return result  # the title was already this everywhere: nothing to record

    from core import audit_log

    # The paper's hash before and after: a finished quest's seal names the paper's hash, and core/evidence.py follows
    # these from it, so a changed title is not read as a paper edited after the quest was sealed.
    recorded = audit_log.AuditLog(trace, quest_root.name).append(
        "title_changed", node="rename", old=old, new=title, changed=list(result.changed),
        paper_path=_SEALED_PAPER, paper_sha256_before=paper_before,
        paper_sha256_after=_sha256(quest_root / _SEALED_PAPER),
    )
    if recorded is None:
        result.trace_problem = ("the change could not be recorded in the quest's trace, so a finished quest's evidence "
                                "check will read the paper as edited after it finished")
    return result


def _run(coro: Any) -> Any:
    """Run ``coro`` to the end from sync code, also when called inside a running event loop (the web server)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
