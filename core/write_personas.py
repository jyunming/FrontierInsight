"""The headings a write persona forbids, in one place: the persona's prompt (``agents/write_persona_<name>.md``, where
``@FORBIDDEN_HEADINGS@`` stands for this list, filled by ``Engine._load_persona_prefix``) and FI's own placing of the
Limitations (``core/paper_facts.py``: a format that forbids the heading gets a ``**Limitations.**`` paragraph instead)
read the same list."""

from __future__ import annotations

TOKEN = "@FORBIDDEN_HEADINGS@"

#: Persona (the paper format) -> the headings its writer must not use.
FORBIDDEN_HEADINGS: dict[str, tuple[str, ...]] = {
    "essay": ("Methods", "Results", "Discussion", "Limitations", "Conclusion and Future Work", "Abstract"),
    "policy_brief": ("Methods", "Results", "Discussion", "Limitations", "Conclusion", "Future Work", "Abstract",
                     "Executive Summary", "Findings", "Introduction"),
}


def forbidden_text(name: str) -> str:
    """The list as the persona's prompt shows it: ``## Methods``, ``## Results``, ... in code spans."""
    return ", ".join(f"`## {h}`" for h in FORBIDDEN_HEADINGS.get(name, ()))


def forbids_limitations(name: str) -> bool:
    return "limitations" in {h.lower() for h in FORBIDDEN_HEADINGS.get(name, ())}


def fill(name: str, prompt: str) -> str:
    """``prompt`` of persona ``name`` with its forbidden-headings list filled in."""
    return prompt.replace(TOKEN, forbidden_text(name))
