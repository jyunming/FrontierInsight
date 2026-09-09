"""Which part of the skill library a quest's catalogue is built from.

Removing the domain filter made the catalogue the cost centre it had been
designed to remove. Measured on real published skills, one catalogue entry
runs ~1,073 characters, so a library covering every scientific domain — 105
importable skills from one published collection alone — costs roughly 28,000
tokens **per quest**. That is far better than the ~93,000 **per pass** blanket
injection cost, but it grows with the library, and the library is meant to
grow.

So the catalogue is layered:

    general layer   every skill with no domain tag — always a candidate
    domain layers   tagged skills, admitted only when the topic looks related

A skill is general **because nobody tagged it**, not because of a list kept
here. `uncertainty-and-units` and `statistical-power` apply whatever the field
is; `scanpy` and `pysam` do not. The tag is assigned at import, lives in
``provenance.json``, and grants nothing — it only routes, which is why
unhashed storage is safe for it (contrast the blocking-validator grant, which
must live in hashed content).

Relevance reuses ``core.passages._hybrid_scores`` — the same all-MiniLM cosine
blended with lexical overlap that already ranks passages, and that the engine
already reuses elsewhere. Two reasons beyond avoiding a new dependency:

* published descriptions are **keyword-dense on purpose** (the Agent Skills
  discovery stage matches on them), so the lexical half does real work here
  rather than merely propping up the semantic half;
* it already degrades to pure lexical when the model is missing or
  ``FI_OFFLINE`` is set, so an air-gapped install keeps working.

**Degradation is deliberately toward more candidates, never fewer.** If
scoring cannot run at all, every skill becomes a candidate and a line says so.
Selection can only narrow, so a too-large catalogue costs tokens and is
visible in the selection report; a silently too-small one changes results with
nothing to show for it.
"""

from __future__ import annotations

import logging
from typing import Any

from core.skills.base import Skill, SkillState

_log = logging.getLogger("fi.skills")

#: How many domain-tagged skills may join the general layer. A cap rather than
#: a score threshold: thresholds need tuning per corpus and fail silently when
#: a topic's vocabulary happens to sit far from every description, while a cap
#: has a predictable worst-case token cost, which is the whole point.
MAX_DOMAIN_SKILLS = 15


def domains_of(skill: Skill) -> list[str]:
    """Domain tags for a skill; empty means general.

    Untagged is general on purpose. An import that forgets its tags makes a
    skill *more* available rather than less, which is the safe direction: it
    shows up as a wrong candidate a person can see in the selection report,
    not as a capability that silently stopped being offered.
    """
    raw = skill.provenance().get("domains")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(x).strip().lower() for x in raw if str(x).strip()]


def _relevance_text(st: SkillState) -> str:
    """The text a skill is matched on — what the selector would read anyway.

    Deliberately the same description and scope limit that reach the
    catalogue, so a skill cannot be ranked on information the model never
    sees, and so improving one improves the other.
    """
    from core.skills.selection import describe, scope_limit

    parts = [st.skill.name.replace("-", " "), " ".join(domains_of(st.skill))]
    parts.append(describe(st.skill))
    parts.append(scope_limit(st.skill))
    return "\n".join(p for p in parts if p)


def split_layers(states: list[SkillState]) -> tuple[list[SkillState], list[SkillState]]:
    """(general, domain-tagged)."""
    general = [st for st in states if not domains_of(st.skill)]
    tagged = [st for st in states if domains_of(st.skill)]
    return general, tagged


def rank_domain_skills(
    tagged: list[SkillState], topic: str, *, limit: int = MAX_DOMAIN_SKILLS,
) -> tuple[list[SkillState], bool]:
    """Most topic-relevant tagged skills, plus whether ranking actually ran.

    The flag matters: "no domain skills were relevant" and "relevance could
    not be computed" must not look the same to the caller, because the right
    response differs — narrow in the first case, widen in the second.
    """
    if not tagged:
        return [], True
    if not (topic or "").strip():
        return tagged[:limit], False

    try:
        from core.passages import _hybrid_scores

        texts = [_relevance_text(st) for st in tagged]
        scores = _hybrid_scores(texts, topic)
    except Exception as e:  # noqa: BLE001 - ranking is an optimisation, never a gate
        _log.warning(
            "[skills] domain ranking unavailable (%s); every skill is a candidate", e
        )
        return tagged, False

    if not scores or len(scores) != len(tagged):
        return tagged, False

    order = sorted(range(len(tagged)), key=lambda i: scores[i], reverse=True)
    return [tagged[i] for i in order[:limit]], True


def select_layers(
    states: list[SkillState], topic: str, *, limit: int = MAX_DOMAIN_SKILLS,
) -> tuple[list[SkillState], dict[str, Any]]:
    """The states a catalogue should be built from, plus an account of why.

    The report travels into the quest so "why was `scanpy` never considered?"
    is answerable afterwards, rather than living in a log line that scrolls
    away.
    """
    general, tagged = split_layers(states)
    picked, ranked = rank_domain_skills(tagged, topic, limit=limit)

    report: dict[str, Any] = {
        "general": [st.skill.name for st in general],
        "domain_available": [st.skill.name for st in tagged],
        "domain_admitted": [st.skill.name for st in picked],
        "ranked": ranked,
        "limit": limit,
    }
    if tagged and not ranked:
        report["note"] = (
            "domain relevance could not be computed, so every domain skill "
            "was admitted — a larger catalogue, never a silently smaller one"
        )
    return general + picked, report


def render_layer_report(report: dict[str, Any]) -> str:
    """Human-readable account for the quest's selection report."""
    gen = report.get("general") or []
    avail = report.get("domain_available") or []
    admitted = report.get("domain_admitted") or []
    lines = [f"General layer: {len(gen)} skill(s), always candidates."]
    if not avail:
        lines.append("No domain-tagged skills installed.")
        return "\n".join(lines)

    if report.get("ranked"):
        lines.append(
            f"Domain layer: {len(admitted)} of {len(avail)} admitted by topic "
            f"relevance (cap {report.get('limit')})."
        )
    else:
        lines.append(
            f"Domain layer: all {len(avail)} admitted — {report.get('note', 'ranking did not run')}."
        )
    held = [n for n in avail if n not in set(admitted)]
    if held:
        shown = ", ".join(held[:12])
        more = f" (+{len(held) - 12} more)" if len(held) > 12 else ""
        lines.append(f"Not admitted for this topic: {shown}{more}")
    return "\n".join(lines)
