"""Choose which skills a quest should carry.

The problem this solves is measured: injecting every trusted skill into
``design`` and the three ``implement`` stages costs ~465 tokens per skill per
node, so a 50-skill library adds ~93,000 tokens to every quest pass. The
library is meant to grow, which made accumulation the largest single token
cost — against the efficiency it was supposed to serve.

It also solves a contradiction. The prompt tells the model "a skill used
outside its stated scope is worse than none, read its *when NOT to use*
section first", and then hands it every skill regardless of topic.

So: build a **catalogue** — name, kind, a description, one line of scope
limit, and the usage record — and let one small call pick from it. A 50-skill
catalogue costs roughly 13k tokens once per quest, against ~93,000 per pass
for blanket injection. See ``MAX_DESCRIPTION`` for why the description is not
trimmed harder than it is.

This module is deliberately free of engine and LLM imports. It assembles the
catalogue and parses a response; the call itself belongs to the engine, which
owns providers, retries and cost accounting.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from core.skills.base import Kind, Skill, SkillState

#: How much of a description reaches the catalogue.
#:
#: This was 110 — one line, sized for FI's own hand-written skills, whose
#: descriptions are prose. Measured against a real published skill it turned
#: out to be actively harmful: ``uncertainty-and-units`` carries an
#: 845-character description, and at 110 the catalogue rendered
#:
#:     Track physical units and propagate measurement uncertainty in
#:     scientific calculations using pint and uncertain
#:
#: cut mid-word, with every phrase that would make it selectable — the
#: plausibility checks, the dimensionless groups, the trigger questions —
#: past the cut. A quest asking exactly what that skill answers would not
#: have picked it.
#:
#: The mismatch is one of intent rather than a bug on either side. The Agent
#: Skills standard matches on name + description at its discovery stage, so
#: authors write descriptions *to be matched on*: keyword-dense and long
#: deliberately.
#:
#: 900 is sized from the real case rather than guessed. That description's
#: three sentences run 115, 568 and 160 characters: the keyword-dense one is
#: the second, and the trigger questions are the third, so anything under
#: ~850 drops the part that makes the skill findable. Two earlier attempts
#: missed this — raising the cap to 600 alone still yielded one sentence,
#: because the sentence-filling below stops before it overruns.
#:
#: The cost is bounded and worth stating plainly: measured on the real entry
#: (1,073 characters including its scope line), 50 such skills come to about
#: 13k tokens, **once** per quest, against the ~93,000 **per pass**
#: that blanket injection cost. Still an order of magnitude better, but the
#: original "under a thousand tokens" claim no longer holds and has been
#: corrected in ``docs/capabilities.md`` and ``dev/skill-selection-spec.md``.
MAX_DESCRIPTION = 900
MAX_SCOPE_LIMIT = 200

_FRONTMATTER_DESC = re.compile(r"^description:\s*(.+)$", re.M)
_NOT_SECTION = re.compile(
    r"^##+\s*When\s+NOT\s+to\s+use.*?$\n+(.*?)(?=^##|\Z)", re.M | re.S | re.I
)

#: Prose statements of scope, for skills that carry no *when NOT to use*
#: heading. Imported skills mostly do not: the published ones state their
#: limits inside a Scope section instead ("It does not cover statistical
#: inference, model selection, or study design"). Without this the catalogue
#: shows no ``NOT for:`` line at all for imported skills, while the selection
#: prompt tells the model to read that line hardest.
_NOT_SENTENCE = re.compile(
    r"[^.!?\n]*\b(?:does\s+not\s+cover|do\s+not\s+cover|not\s+covered|"
    r"does\s+not\s+handle|is\s+not\s+for|are\s+not\s+for|out\s+of\s+scope|"
    r"not\s+suitable\s+for|never\s+use\s+(?:this\s+)?for)\b[^.!?]*[.!?]",
    re.I,
)


def _first_sentence(text: str, cap: int) -> str:
    flat = " ".join(text.split())
    if not flat:
        return ""
    # Cut at the first sentence end, else hard-trim.
    m = re.search(r"(?<=[.!?])\s", flat)
    out = flat[: m.start()] if m and m.start() < cap else flat[:cap]
    return out.strip().rstrip(",;:")


def _fill_sentences(text: str, cap: int) -> str:
    """As many whole sentences as fit in ``cap``.

    Raising the cap alone did not fix the truncation this was written for.
    ``_first_sentence`` stops at the first sentence boundary whenever it
    falls under the cap, and the real description's first sentence is 114
    characters — so a 600-character budget still yielded one sentence, with
    every trigger phrase in sentences two and three thrown away.

    Filling to the budget keeps FI's own one-sentence descriptions exactly as
    they were while letting a keyword-dense imported one actually arrive.
    """
    flat = " ".join(text.split())
    if not flat:
        return ""
    if len(flat) <= cap:
        return flat.strip().rstrip(",;:")

    out = ""
    for part in re.split(r"(?<=[.!?])\s+", flat):
        candidate = f"{out} {part}".strip() if out else part
        if len(candidate) > cap:
            break
        out = candidate
    # A first sentence longer than the whole budget still has to be cut.
    return (out or flat[:cap]).strip().rstrip(",;:")


def describe(skill: Skill) -> str:
    """One line saying what the skill is for.

    Prefers the front-matter ``description``: that field exists because
    someone wrote it to make the skill findable, and it survives import from
    other agents for exactly this step. Falls back to the first prose line of
    ``SKILL.md``.
    """
    text = skill.instructions()
    m = _FRONTMATTER_DESC.search(text)
    if m:
        return _fill_sentences(m.group(1), MAX_DESCRIPTION)
    body = re.sub(r"^---.*?^---", "", text, flags=re.S | re.M)
    for line in body.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return _fill_sentences(s, MAX_DESCRIPTION)
    return ""


def scope_limit(skill: Skill) -> str:
    """One line saying where the skill does not apply.

    Costs ~40 characters per entry and is worth it: the section exists to stop
    a skill being reached for outside its domain, and selection is the first
    place that decision is made.

    Two sources, in order. A ``## When NOT to use`` heading is FI's own
    convention and wins when present. Failing that, a sentence stating a
    limit anywhere in the instructions — because imported skills largely do
    not use that heading, and an empty ``NOT for:`` line is worse than no
    convention at all when the prompt tells the model to weight it heavily.
    """
    text = skill.instructions()
    m = _NOT_SECTION.search(text)
    if m:
        body = re.sub(r"^\s*[-*]\s*", "", m.group(1).strip(), flags=re.M)
        return _first_sentence(body, MAX_SCOPE_LIMIT)

    m = _NOT_SENTENCE.search(text)
    if m:
        return _first_sentence(m.group(0).strip(), MAX_SCOPE_LIMIT)
    return ""


def usage_record(skill: Skill) -> tuple[int, int]:
    """(times used, times the quest was accepted)."""
    raw = skill.provenance().get("taught_by_projects")
    if not isinstance(raw, list):
        return 0, 0
    used = accepted = 0
    for item in raw:
        used += 1
        if isinstance(item, dict) and str(item.get("outcome", "")).lower() == "accept":
            accepted += 1
    return used, accepted


@dataclass
class CatalogueEntry:
    name: str
    kind: Kind
    description: str
    scope_limit: str = ""
    used: int = 0
    accepted: int = 0

    def render(self) -> str:
        bits = [f"- **{self.name}** ({self.kind.value})"]
        if self.description:
            bits.append(f": {self.description}")
        line = "".join(bits)
        if self.scope_limit:
            line += f"\n  NOT for: {self.scope_limit}"
        if self.used:
            line += f"\n  track record: used by {self.used} quest(s), {self.accepted} accepted"
        else:
            line += "\n  track record: not yet used"
        return line


@dataclass
class Catalogue:
    entries: list[CatalogueEntry] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def render(self) -> str:
        return "\n".join(e.render() for e in self.entries)

    @property
    def names(self) -> set[str]:
        return {e.name for e in self.entries}


def build_catalogue(
    states: list[SkillState],
    *,
    exclude: list[str] | None = None,
    survey_mode: bool = False,
) -> Catalogue:
    """Assemble the candidate list from trusted skills.

    ``survey_mode`` drops library skills: with no experiment there is nothing
    for them to compute. That is decided here rather than asked of the model,
    because it is a certainty, and routing a certainty through a probabilistic
    call only adds ways to be wrong.

    Tool skills survive, for a narrower reason than "a synthesis might read a
    document" — reading documents is the engine's own job, in ``data_load``.
    What a literature quest can genuinely need is to **query a source FI does
    not know how to reach**: a specialist database, an internal API, a
    subscription service. That is external software being driven, and it is
    exactly what a tool skill is for.
    """
    skip = {n.strip() for n in (exclude or []) if n.strip()}
    out: list[CatalogueEntry] = []
    for st in states:
        if not st.loadable or st.skill.name in skip:
            continue
        skill = st.skill
        if survey_mode and skill.kind is Kind.LIBRARY:
            continue
        used, accepted = usage_record(skill)
        out.append(
            CatalogueEntry(
                name=skill.name,
                kind=skill.kind,
                description=describe(skill),
                scope_limit=scope_limit(skill),
                used=used,
                accepted=accepted,
            )
        )
    # Stable order so an unchanged library renders an unchanged prompt — the
    # selection call runs at temperature 0 and a shuffled catalogue would
    # break the reproducibility that buys.
    out.sort(key=lambda e: e.name)
    return Catalogue(entries=out)


@dataclass
class Selection:
    chosen: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)
    #: Why each rejected candidate was rejected, when the model said.
    #:
    #: Measured, not assumed: the earlier prompt asked only for selections,
    #: which made declining free — a reason cost a sentence, abstaining cost
    #: nothing. On one topic and catalogue it returned an empty list from
    #: *both* codex and antigravity, for a topic where the skill plainly
    #: applied and for one where it plainly did not. Requiring a reason for
    #: every candidate made both models select on the applicable topic and
    #: decline, correctly and with a stated reason, on the other.
    declined: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen": self.chosen,
            "reasons": self.reasons,
            "declined": self.declined,
            "ignored_unknown": self.unknown,
        }


def parse_selection(text: str, catalogue: Catalogue) -> Selection:
    """Read the model's answer, keeping only names that really exist.

    Selection can narrow the candidate set and nothing else. A name that is
    not in the catalogue — hallucinated, or a skill that is real but not
    trusted — is dropped and reported, never promoted: a model must not be
    able to talk an unapproved skill into a quest.
    """
    sel = Selection()
    if not text or not text.strip():
        return sel

    data: Any = None
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except ValueError:
                data = None
    if not isinstance(data, dict):
        return sel

    raw = data.get("skills")
    if not isinstance(raw, list):
        return sel

    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            name, why = item.strip(), ""
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            why = str(item.get("reason") or "").strip()
        else:
            continue
        if not name or name in seen:
            continue
        seen.add(name)
        if name not in catalogue.names:
            sel.unknown.append(name)
            continue
        sel.chosen.append(name)
        if why:
            sel.reasons[name] = why

    # Declines, when the model gave them. Kept because they are the only
    # evidence that a candidate was considered at all: without them an empty
    # selection and a catalogue the model never read produce identical
    # output, and the difference is exactly what needs diagnosing when a
    # skill is silently never used.
    for item in data.get("declined") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        why = str(item.get("reason") or "").strip()
        # A name in both lists is contradictory; the selection wins, since
        # that is the side that carries a consequence.
        if name and name in catalogue.names and name not in sel.chosen:
            sel.declined[name] = why
    return sel


def render_selection_report(
    catalogue: Catalogue,
    sel: "Selection",
    near: list[SkillState],
) -> str:
    """Human-readable account of what selection decided and why.

    Written to the quest so "why did this quest not use ambit?" is answerable
    afterwards. Without it the reasoning exists only in a log line that
    scrolls away, and a skill can sit unapproved indefinitely while every
    quest quietly does without it.
    """
    lines = [f"Considered {len(catalogue.entries)} candidate skill(s)."]
    if sel.chosen:
        lines.append("")
        lines.append("Selected:")
        for name in sel.chosen:
            why = sel.reasons.get(name, "(no reason given)")
            lines.append(f"  - {name}: {why}")
    else:
        lines.append("")
        lines.append("Selected: none — the quest generated its own code.")

    skipped = [e.name for e in catalogue.entries if e.name not in set(sel.chosen)]
    if skipped:
        lines.append("")
        lines.append("Considered but not selected:")
        for name in skipped:
            why = sel.declined.get(name)
            # A decline without a reason means the model did not account for
            # that entry — worth showing as such rather than smoothing over,
            # because it is the signature of a selection call that skimmed.
            lines.append(f"  - {name}: {why or '(no reason given)'}")
    if sel.unknown:
        lines.append("")
        lines.append(
            "Named by the model but not candidates (ignored): "
            + ", ".join(sel.unknown)
        )
    if near:
        lines.append("")
        lines.append("Not available — these could not be candidates at all:")
        for st in near:
            lines.append(f"  - {st.skill.name} ({st.status.value}): {st.reason}")
        lines.append(
            "  Approve with: python launch.py --approve-skill <name> "
            "--approve-as <you>"
        )
    return "\n".join(lines) + "\n"


def near_misses(
    all_states: list[SkillState], chosen: list[str], catalogue: Catalogue,
) -> list[SkillState]:
    """Skills that were never candidates because they are not trusted.

    Reported so an unapproved skill cannot sit unnoticed forever while every
    quest quietly does without it. This is the half of the learning loop that
    tells a person there is something to approve.
    """
    return [
        st for st in all_states
        if not st.loadable and st.skill.name not in catalogue.names
    ]
