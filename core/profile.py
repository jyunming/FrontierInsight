"""The person's own details, asked once and kept for every later quest: name, affiliation, email, web page.

The new-quest interview does not ask for the byline: it is a folded "Paper byline (optional)" line of the review screen,
and the first time a paper is written from a terminal the name is asked once (launch._ask_byline_once). What is given is
kept in ``~/.frontier-insight/profile.json`` (``FI_PROFILE_PATH`` overrides it), which the CLI, the web page and the VS
Code extension all read and write. A later interview fills the byline from it and shows it on the review screen, where it
can be changed; a change there is kept for later quests too. The file existing means the person has been asked, even
when they left every field blank (the interviews do not write a blank byline while none was ever given).
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

#: The fields kept, in the order the interview asks them (the interview's question ids).
FIELDS = ("author", "affiliation", "contact_email", "url")


def path() -> Path:
    override = os.environ.get("FI_PROFILE_PATH")
    return Path(override) if override else Path.home() / ".frontier-insight" / "profile.json"


def load() -> dict[str, str] | None:
    """The saved details, every field present (blank when not given), or ``None`` when the person has not been asked
    yet (no file). An unreadable file is ``None`` too, so the person is asked again rather than given someone's
    half-written details."""
    try:
        raw = json.loads(path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    return {k: " ".join(str(raw.get(k) or "").split())[:300] for k in FIELDS}


def save(values: dict[str, Any]) -> dict[str, str]:
    """Keep the details (every field; a missing one is blank), written whole then renamed. Returns what was kept."""
    kept = {k: " ".join(str(values.get(k) or "").split())[:300] for k in FIELDS}
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(kept, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return kept
