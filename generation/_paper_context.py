"""The whole final paper and its checked claims, for the outputs the model writes freely (slides, poster, talk).

Those three used to see ``paper_md[:8000]``: the cut dropped the later sections, and a scope or a number the model
could not see it filled in from its own guess. They now read the whole paper plus the claim ledger
(``paper/claims.json``, written by the claim check), and the prompt tells them to state as findings only the ledger's
claims. The wording of that rule is in ``agents/claims_rule.md`` and ``agents/claims_list.md``.

A paper without a ledger (the claim check off, or no ``claims.json``) still gets the whole paper; the rule and the
list are then left out of the prompt, because there is no list to hold the model to.

Nothing is cut here. A prompt longer than ``provider.max_prompt_chars`` (when that cap is set) is trimmed by the
provider in the middle of its largest message; :func:`note_prompt_fit` says so in ``run.log`` and in
``.fi/prompt_fit.json`` before the call.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from core import paper_facts as _paper_facts
from core.provider import quest_run_log

_AGENTS = Path(__file__).resolve().parent.parent / "agents"
_log = logging.getLogger("frontier_insight.generation.paper_context")
FIT_RECORD = Path(".fi") / "prompt_fit.json"


@dataclass(frozen=True)
class PaperContext:
    paper: str  # the whole paper
    claims_rule: str  # the instruction that goes first in the prompt ("" without a ledger)
    claims: str  # the claim list section that follows the paper ("" without a ledger)
    n_claims: int


def checked_claims(quest_root: Path | str | None) -> list[str]:
    """The claims of ``paper/claims.json`` that rest on the run or a source, each with its scope when the ledger
    records one. A claim the ledger marks unsupported is not one the model may state. Empty without a readable
    ledger."""
    if not quest_root:
        return []
    try:
        data = json.loads((Path(quest_root) / "paper" / "claims.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = data.get("claims") if isinstance(data, dict) else None
    out: list[str] = []
    for c in rows if isinstance(rows, list) else []:
        if not isinstance(c, dict) or str(c.get("basis") or "").strip().lower() == "unsupported":
            continue
        text = " ".join(str(c.get("claim") or "").split())
        if not text:
            continue
        scope = " ".join(str(c.get("scope") or "").split())
        out.append(f"- {text}" + (f" (holds for: {scope})" if scope else ""))
    return out


def paper_and_claims(paper_md: str, quest_root: Path | str | None) -> PaperContext:
    """``paper_md`` whole, plus the instruction and the claim list when the quest has checked claims."""
    lines = [_paper_facts.without_marks(x) for x in checked_claims(quest_root)]
    if not lines:
        return PaperContext(paper_md, "", "", 0)
    rule = (_AGENTS / "claims_rule.md").read_text(encoding="utf-8")
    section = (_AGENTS / "claims_list.md").read_text(encoding="utf-8").replace("$claims", "\n".join(lines))
    return PaperContext(paper_md, rule, section, len(lines))


def note_prompt_fit(prompt: str, max_prompt_chars: int, kind: str, quest_root: Path | str | None) -> dict | None:
    """When the prompt is longer than the provider's cap, say so (run.log and ``.fi/prompt_fit.json``) and return
    the record; ``None`` when it fits or no cap is set. The call is still made: the provider trims the middle of the
    paper, loudly."""
    cap = int(max_prompt_chars or 0)
    if cap <= 0 or len(prompt) <= cap:
        return None
    record = {"prompt_chars": len(prompt), "cap": cap, "paper_trimmed_by_provider": True}
    msg = (f"{kind}: the prompt is {len(prompt):,} characters and the provider's cap is {cap:,}; the middle of the "
           f"paper is cut before the call")
    _log.warning(msg)
    run_log = quest_run_log(quest_root)
    if run_log is not None:
        run_log.warning(msg)
    if quest_root:
        path = Path(quest_root) / FIT_RECORD
        try:
            have = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            have = have if isinstance(have, dict) else {}
            have[kind] = record
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(have, indent=2) + "\n", encoding="utf-8")
        except (OSError, ValueError):
            pass
    return record
