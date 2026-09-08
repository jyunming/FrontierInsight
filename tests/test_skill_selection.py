"""Choosing which skills a quest carries.

The flaw this replaces was measured: injecting every trusted skill into four
prompts cost ~465 tokens per skill per node, so a 50-skill library added
~93,000 tokens to every quest pass — the accumulation mechanism becoming its
own largest cost.

Two properties matter most here and neither is about picking well:

**Selection can only narrow.** It must never be able to talk an unapproved
skill into a quest, however confidently it names one.

**Nothing is fatal.** A quest must not fail because an additive step could not
answer — every failure path has to end in "no skills", which is the behaviour
that existed before skills.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.skills import Skill, SkillState, Status
from core.skills.base import Kind
from core.skills import selection as sel


def make_state(
    tmp: Path,
    name: str,
    *,
    description: str = "",
    not_for: str = "",
    kind: str | None = None,
    used: list[dict] | None = None,
    loadable: bool = True,
) -> SkillState:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    body = ["---", f"name: {name}"]
    if description:
        body.append(f"description: {description}")
    body += ["---", "", f"# {name}", ""]
    if not_for:
        body += ["## When NOT to use this", "", not_for, ""]
    (d / "SKILL.md").write_text("\n".join(body), encoding="utf-8")
    prov: dict = {}
    if kind:
        prov["kind"] = kind
    if used is not None:
        prov["taught_by_projects"] = used
    if prov:
        (d / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    return SkillState(
        skill=Skill(name=name, path=d),
        status=Status.TRUSTED if loadable else Status.PROPOSED,
    )


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


def test_catalogue_is_one_line_per_skill_not_the_whole_body(tmp_path: Path) -> None:
    """The entire point: candidates cost a line, not their full instructions."""
    st = make_state(
        tmp_path, "ambit",
        description="Aerial image formation and NILS at 193 nm",
    )
    # Pad the body so a full-body render would be obviously larger.
    (st.skill.path / "SKILL.md").write_text(
        (st.skill.path / "SKILL.md").read_text(encoding="utf-8") + "\n" + "x " * 2000,
        encoding="utf-8",
    )
    rendered = sel.build_catalogue([st]).render()
    assert "Aerial image formation" in rendered
    assert len(rendered) < 300, "catalogue entry must stay a line, not a body"


def test_description_comes_from_front_matter(tmp_path: Path) -> None:
    """That field exists to make a skill findable, and survives import from
    other agents for exactly this step."""
    st = make_state(tmp_path, "a", description="Convert documents between formats")
    assert sel.describe(st.skill) == "Convert documents between formats"


def test_description_falls_back_to_first_prose_line(tmp_path: Path) -> None:
    d = tmp_path / "b"
    d.mkdir()
    (d / "SKILL.md").write_text("# b\n\nDrives the widget press.\n", encoding="utf-8")
    # The period stays: the whole line is one sentence, so there is nothing
    # to cut at, and trimming punctuation would only make it read oddly.
    assert sel.describe(Skill(name="b", path=d)) == "Drives the widget press."


def test_scope_limit_is_carried_into_the_catalogue(tmp_path: Path) -> None:
    """The 'when NOT to use' line is the one selection must read hardest, so
    it is worth ~40 characters per entry."""
    st = make_state(
        tmp_path, "a",
        description="Aerial imaging",
        not_for="Not for resist chemistry or stochastic LER.",
    )
    rendered = sel.build_catalogue([st]).render()
    assert "NOT for: Not for resist chemistry" in rendered


def test_usage_record_appears_and_counts_accepts(tmp_path: Path) -> None:
    st = make_state(
        tmp_path, "a", description="x",
        used=[{"outcome": "accept"}, {"outcome": "reject"}, {"outcome": "accept"}],
    )
    assert sel.usage_record(st.skill) == (3, 2)
    assert "used by 3 quest(s), 2 accepted" in sel.build_catalogue([st]).render()


def test_untried_skill_says_so_rather_than_showing_nothing(tmp_path: Path) -> None:
    st = make_state(tmp_path, "a", description="x")
    assert "not yet used" in sel.build_catalogue([st]).render()


def test_only_loadable_skills_become_candidates(tmp_path: Path) -> None:
    """Selection must not be able to reach an unapproved skill at all."""
    ok = make_state(tmp_path, "good", description="x")
    no = make_state(tmp_path, "bad", description="x", loadable=False)
    assert sel.build_catalogue([ok, no]).names == {"good"}


def test_excluded_skills_are_not_candidates(tmp_path: Path) -> None:
    a = make_state(tmp_path, "a", description="x")
    b = make_state(tmp_path, "b", description="x")
    assert sel.build_catalogue([a, b], exclude=["a"]).names == {"b"}


def test_survey_mode_drops_libraries_and_keeps_tools(tmp_path: Path) -> None:
    """No experiment means nothing for a library to compute — but a literature
    synthesis may still need to read a PDF."""
    lib = make_state(tmp_path, "lib", description="x", kind="library")
    tool = make_state(tmp_path, "tool", description="x", kind="tool")
    assert sel.build_catalogue([lib, tool], survey_mode=True).names == {"tool"}
    assert sel.build_catalogue([lib, tool]).names == {"lib", "tool"}


def test_catalogue_order_is_stable(tmp_path: Path) -> None:
    """Selection runs at temperature 0; a shuffled catalogue would throw away
    the reproducibility that buys."""
    states = [make_state(tmp_path, n, description="x") for n in ("c", "a", "b")]
    first = sel.build_catalogue(states).render()
    second = sel.build_catalogue(list(reversed(states))).render()
    assert first == second


# ---------------------------------------------------------------------------
# Parsing — selection may only narrow
# ---------------------------------------------------------------------------


@pytest.fixture
def cat(tmp_path: Path) -> sel.Catalogue:
    return sel.build_catalogue([
        make_state(tmp_path, "ambit", description="x"),
        make_state(tmp_path, "pandoc", description="y"),
    ])


def test_parses_names_and_reasons(cat) -> None:
    got = sel.parse_selection(
        json.dumps({"skills": [{"name": "ambit", "reason": "computes the image"}]}),
        cat,
    )
    assert got.chosen == ["ambit"]
    assert got.reasons["ambit"] == "computes the image"


def test_a_name_outside_the_catalogue_is_dropped_and_reported(cat) -> None:
    """The security property: a model must not be able to introduce a skill,
    whether hallucinated or real-but-unapproved."""
    got = sel.parse_selection(
        json.dumps({"skills": [{"name": "ambit"}, {"name": "semulator"}]}), cat,
    )
    assert got.chosen == ["ambit"]
    assert got.unknown == ["semulator"]


def test_empty_selection_is_a_valid_answer(cat) -> None:
    assert sel.parse_selection(json.dumps({"skills": []}), cat).chosen == []


def test_bare_strings_are_accepted(cat) -> None:
    assert sel.parse_selection(json.dumps({"skills": ["pandoc"]}), cat).chosen == ["pandoc"]


def test_json_wrapped_in_prose_is_recovered(cat) -> None:
    text = 'Here is my answer:\n{"skills": [{"name": "ambit"}]}\nHope that helps.'
    assert sel.parse_selection(text, cat).chosen == ["ambit"]


@pytest.mark.parametrize("text", ["", "   ", "not json", "[]", '{"wrong": 1}'])
def test_unparseable_output_selects_nothing_rather_than_raising(text, cat) -> None:
    got = sel.parse_selection(text, cat)
    assert got.chosen == [] and got.unknown == []


def test_duplicates_are_collapsed(cat) -> None:
    got = sel.parse_selection(
        json.dumps({"skills": ["ambit", {"name": "ambit", "reason": "r"}]}), cat,
    )
    assert got.chosen == ["ambit"]


def test_order_is_preserved(cat) -> None:
    """The model is asked to rank; the ranking must survive parsing."""
    got = sel.parse_selection(json.dumps({"skills": ["pandoc", "ambit"]}), cat)
    assert got.chosen == ["pandoc", "ambit"]


# ---------------------------------------------------------------------------
# Near misses — the other half of the learning loop
# ---------------------------------------------------------------------------


def test_untrusted_skills_are_reported_as_near_misses(tmp_path: Path) -> None:
    """Without this, a skill can sit unapproved forever while every quest
    quietly does without it."""
    ok = make_state(tmp_path, "good", description="x")
    no = make_state(tmp_path, "bad", description="x", loadable=False)
    catalogue = sel.build_catalogue([ok])
    missed = sel.near_misses([no], ["good"], catalogue)
    assert [m.skill.name for m in missed] == ["bad"]


def _catalogue(tmp_path: Path, names: list[str]):
    """A catalogue holding one trusted entry per name."""
    return sel.build_catalogue([make_state(tmp_path, n) for n in names])


# ---------------------------------------------------------------------------
# Declines
#
# Measured, not theorised. The first prompt asked only for selections, so a
# reason cost a sentence and abstaining cost nothing — and on one topic and
# catalogue it returned `{"skills": []}` from BOTH codex and antigravity, for
# a topic the skill plainly fitted. Requiring a reason per candidate made both
# models select it, and still decline correctly on an unrelated topic.
# ---------------------------------------------------------------------------


def test_declines_are_captured_with_their_reasons(tmp_path: Path) -> None:
    cat = _catalogue(tmp_path, ["ambit", "pandoc"])
    out = sel.parse_selection(
        json.dumps({
            "skills": [{"name": "ambit", "reason": "computes the aerial image"}],
            "declined": [{"name": "pandoc", "reason": "no document is produced here"}],
        }),
        cat,
    )
    assert out.chosen == ["ambit"]
    assert out.declined == {"pandoc": "no document is produced here"}


def test_a_decline_for_an_unknown_name_is_dropped(tmp_path: Path) -> None:
    """Selection may only narrow. A name outside the catalogue carries no
    meaning in either direction, so it cannot enter through `declined` any
    more than it can through `skills`."""
    cat = _catalogue(tmp_path, ["ambit"])
    out = sel.parse_selection(
        json.dumps({"skills": [], "declined": [{"name": "invented", "reason": "x"}]}),
        cat,
    )
    assert out.declined == {}


def test_a_name_in_both_lists_counts_as_selected(tmp_path: Path) -> None:
    """Contradictory output has to resolve one way; selection wins because it
    is the side that carries a consequence into the quest."""
    cat = _catalogue(tmp_path, ["ambit"])
    out = sel.parse_selection(
        json.dumps({
            "skills": [{"name": "ambit", "reason": "needed"}],
            "declined": [{"name": "ambit", "reason": "not needed"}],
        }),
        cat,
    )
    assert out.chosen == ["ambit"]
    assert "ambit" not in out.declined


def test_a_response_without_declines_still_parses(tmp_path: Path) -> None:
    """Older or terser models may omit the field entirely; that must degrade
    to 'no reasons recorded', never to an error."""
    cat = _catalogue(tmp_path, ["ambit"])
    out = sel.parse_selection(json.dumps({"skills": []}), cat)
    assert out.chosen == [] and out.declined == {}


def test_declines_reach_the_quest_record(tmp_path: Path) -> None:
    """The record is what makes 'why was this never used?' answerable after
    the run, so the reasons have to survive serialisation."""
    cat = _catalogue(tmp_path, ["ambit"])
    out = sel.parse_selection(
        json.dumps({"skills": [], "declined": [{"name": "ambit", "reason": "off topic"}]}),
        cat,
    )
    assert out.to_dict()["declined"] == {"ambit": "off topic"}
