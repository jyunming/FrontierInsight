"""The whole final paper and its checked claims, for the outputs the model writes freely (slides, poster, talk).

Those three used to see ``paper_md[:8000]``: the cut dropped the later sections, and a number the model could not see
it filled in from its own guess. They now read the whole paper plus the claim ledger (``paper/claims.json``, written by
the claim check), and the prompt tells them to state as findings only the ledger's claims. The wording is in
``agents/claims_rule.md``, ``claims_none.md`` and ``claims_list.md``; the rule stands at the very top of the prompt and
the claim list before the paper.

The ledger is used only when it describes this paper: the claim check's receipt (``needs/receipts/claim_check.json``)
must have judged these bytes of ``paper.md`` (a page-limit fit that only takes sentences out carries the receipt over).
A ledger left by an earlier draft, or by a check that failed, is not used: the model gets the whole paper with no claim
list and no rule, and ``run.log`` and ``.fi/prompt_fit.json`` say why. A ledger in which every claim is unsupported
gets the rule in its ``claims_none.md`` form.

Nothing is cut here. These generators build their model client without ``provider.max_prompt_chars``, so no cap applies
to their prompts; :func:`note_prompt_fit` records each prompt's size and which ledger it used in ``.fi/prompt_fit.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from core import paper_facts as _paper_facts
from core import receipts as _receipts
from core.provider import quest_run_log

_AGENTS = Path(__file__).resolve().parent.parent / "agents"
_log = logging.getLogger("frontier_insight.generation.paper_context")
FIT_RECORD = Path(".fi") / "prompt_fit.json"


@dataclass(frozen=True)
class PaperContext:
    paper: str  # the whole paper
    claims_rule: str  # the instruction that goes first in the prompt ("" without a usable ledger)
    claims: str  # the claim list section ("" without a usable ledger or without a usable claim)
    n_claims: int
    ledger: str  # "used" | "no_confirmed_claim" | "none" (no ledger) | "stale" (it is not this paper's)


def _ledger(quest_root: Path | str | None, paper_bytes: bytes | None) -> tuple[str, list[str]]:
    """``(state, claim lines)``: ``none`` without a ledger, ``stale`` when the claim check's receipt did not judge
    these bytes of the paper, else ``used`` / ``no_confirmed_claim``."""
    if not quest_root:
        return "none", []
    root = Path(quest_root)
    try:
        data = json.loads((root / "paper" / "claims.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "none", []
    rows = data.get("claims") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return "none", []
    status, record, _problem = _receipts.read(root, "claim_check")
    if (record is None or status not in ("pass", "fail") or (data.get("status") or "ok") != "ok" or paper_bytes is None
            or (record.get("input_hashes") or {}).get("paper") != _receipts.sha256(paper_bytes)):
        return "stale", []
    out: list[str] = []
    for c in rows:
        if not isinstance(c, dict) or str(c.get("basis") or "").strip().lower() == "unsupported":
            continue
        text = " ".join(str(c.get("claim") or "").split())
        if text:
            out.append(f"- {_paper_facts.without_marks(text)}")
    return ("used" if out else "no_confirmed_claim"), out


def paper_and_claims(paper_md: str, quest_root: Path | str | None, paper_file: Path | str | None = None) -> PaperContext:
    """``paper_md`` whole, plus the instruction and the claim list when the quest has a ledger of this paper
    (``paper_file`` is the paper.md the ledger must describe)."""
    try:
        paper_bytes = Path(paper_file).read_bytes() if paper_file else None
    except OSError:
        paper_bytes = None
    state, lines = _ledger(quest_root, paper_bytes)
    if state == "used":
        section = (_AGENTS / "claims_list.md").read_text(encoding="utf-8").replace("$claims", "\n".join(lines))
        return PaperContext(paper_md, (_AGENTS / "claims_rule.md").read_text(encoding="utf-8"), section, len(lines), state)
    if state == "no_confirmed_claim":
        return PaperContext(paper_md, (_AGENTS / "claims_none.md").read_text(encoding="utf-8"), "", 0, state)
    return PaperContext(paper_md, "", "", 0, state)


def note_prompt_fit(prompt: str, kind: str, quest_root: Path | str | None, ctx: PaperContext) -> dict:
    """Record the prompt's size and the ledger it used in ``.fi/prompt_fit.json`` (and a stale ledger in ``run.log``).
    No cap applies to these prompts, so nothing is said about a cut."""
    record = {"prompt_chars": len(prompt), "paper_chars": len(ctx.paper), "claims_listed": ctx.n_claims,
              "ledger": ctx.ledger}
    if ctx.ledger == "stale":
        msg = (f"{kind}: paper/claims.json is not the claim check's result for this paper.md; the whole paper is "
               f"given with no claim list")
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
