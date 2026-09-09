"""Layering the catalogue, so breadth does not cost what it used to.

FI is domain-general, so there is no domain filter on what may be installed.
That makes the catalogue the cost centre selection was built to remove: at the
measured ~1,073 characters per entry, a hundred-odd skills is ~28,000 tokens
per quest, growing with a library that is meant to grow.

The split is by tag, not by a list kept in code: **untagged means general**,
and general skills are always candidates. Tagged ones must earn their place
against the topic.

The property most of these defend is the direction of failure. Selection can
only narrow, so admitting too many skills costs tokens and shows up in the
selection report; admitting too few changes the result with nothing to show
for it. Every degradation here therefore widens.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from core.skills import Skill, SkillState, Status
from core.skills import layers


def make(root: Path, name: str, desc: str, domains: list[str] | None = None) -> SkillState:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        textwrap.dedent(f"""\
            ---
            name: {name}
            description: {desc}
            ---

            # {name}
            """),
        encoding="utf-8",
    )
    prov = {"domains": domains} if domains is not None else {}
    (d / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    return SkillState(skill=Skill(name=name, path=d), status=Status.TRUSTED)


@pytest.fixture
def library(tmp_path: Path) -> list[SkillState]:
    """A small mixed library: three general, four tagged across two fields."""
    return [
        make(tmp_path, "uncertainty-and-units",
             "Track physical units and propagate measurement uncertainty."),
        make(tmp_path, "statistical-power",
             "Sample size and statistical power calculations for planning studies."),
        make(tmp_path, "sympy", "Exact symbolic mathematics in Python."),
        make(tmp_path, "scanpy",
             "Single-cell RNA-seq analysis: clustering, marker genes, UMAP.",
             ["genomics", "biomedical"]),
        make(tmp_path, "pysam",
             "Read and write SAM, BAM, CRAM and VCF genomic alignment files.",
             ["genomics"]),
        make(tmp_path, "openpiv",
             "Particle image velocimetry: velocity fields, vorticity, turbulence.",
             ["fluids", "physics"]),
        make(tmp_path, "pymatgen",
             "Crystal structure analysis, phase diagrams, materials science.",
             ["materials", "physics"]),
    ]


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def test_untagged_skills_are_general(library) -> None:
    general, tagged = layers.split_layers(library)
    assert {s.skill.name for s in general} == {
        "uncertainty-and-units", "statistical-power", "sympy"
    }
    assert len(tagged) == 4


def test_a_skill_with_no_provenance_is_general(tmp_path: Path) -> None:
    """An import that forgets its tags must over-offer, not withdraw.

    The wrong candidate is visible in the selection report and costs tokens;
    a capability that silently stopped being offered is neither.
    """
    d = tmp_path / "bare"
    d.mkdir()
    (d / "SKILL.md").write_text("# bare\n\nDoes a thing.\n", encoding="utf-8")
    st = SkillState(skill=Skill(name="bare", path=d), status=Status.TRUSTED)
    assert layers.domains_of(st.skill) == []
    general, tagged = layers.split_layers([st])
    assert general and not tagged


def test_domains_are_normalised(tmp_path: Path) -> None:
    st = make(tmp_path, "x", "desc", ["  Genomics ", "BIOMEDICAL"])
    assert layers.domains_of(st.skill) == ["genomics", "biomedical"]


def test_a_single_domain_string_is_accepted(tmp_path: Path) -> None:
    d = tmp_path / "y"
    d.mkdir()
    (d / "SKILL.md").write_text("# y\n", encoding="utf-8")
    (d / "provenance.json").write_text(json.dumps({"domains": "physics"}), "utf-8")
    assert layers.domains_of(Skill(name="y", path=d)) == ["physics"]


# ---------------------------------------------------------------------------
# Relevance
# ---------------------------------------------------------------------------


def test_general_skills_are_always_candidates(library) -> None:
    """Whatever the topic — this is what 'general' means."""
    for topic in ("single-cell RNA sequencing of tumour samples",
                  "turbulent flow in a microchannel",
                  "something with no obvious field at all"):
        chosen, _ = layers.select_layers(library, topic)
        names = {s.skill.name for s in chosen}
        assert {"uncertainty-and-units", "statistical-power", "sympy"} <= names, topic


def test_topic_admits_the_related_domain(library) -> None:
    """The point of the layer. Scoring blends embeddings with lexical overlap
    and degrades to lexical alone, so this must hold either way — which is
    why the topic here shares vocabulary with the target descriptions."""
    chosen, report = layers.select_layers(
        library, "single-cell RNA-seq clustering and marker genes", limit=2,
    )
    names = [s.skill.name for s in chosen]
    assert "scanpy" in names
    assert report["ranked"] is True


def test_the_cap_is_honoured(library) -> None:
    chosen, report = layers.select_layers(library, "genomics", limit=1)
    admitted = report["domain_admitted"]
    assert len(admitted) == 1
    # General skills are not counted against the domain cap.
    assert len(chosen) == 3 + 1


def test_an_empty_topic_admits_everything_rather_than_guessing(library) -> None:
    chosen, report = layers.select_layers(library, "")
    assert len(chosen) == len(library)
    assert report["ranked"] is False


# ---------------------------------------------------------------------------
# Degradation widens, never narrows
# ---------------------------------------------------------------------------


def test_a_broken_ranker_admits_every_domain_skill(library, monkeypatch) -> None:
    """The direction-of-failure property.

    Selection can only narrow, so a too-large catalogue costs tokens and is
    visible in the report. A silently too-small one changes the result with
    nothing to show for it — so when relevance cannot be computed, everything
    is admitted and the report says why.
    """
    import core.passages as passages

    def boom(chunks, query):
        raise RuntimeError("no model, no lexical, nothing")

    monkeypatch.setattr(passages, "_hybrid_scores", boom)

    chosen, report = layers.select_layers(library, "anything")
    assert len(chosen) == len(library), "degradation narrowed the catalogue"
    assert report["ranked"] is False
    assert "note" in report and "never a silently smaller one" in report["note"]


def test_mismatched_score_count_is_treated_as_failure(library, monkeypatch) -> None:
    """A ranker returning the wrong number of scores must not be zipped
    against the skills — that would rank each one on another's text."""
    import core.passages as passages

    monkeypatch.setattr(passages, "_hybrid_scores", lambda chunks, query: [1.0])
    chosen, report = layers.select_layers(library, "genomics")
    assert len(chosen) == len(library)
    assert report["ranked"] is False


def test_no_domain_skills_installed_is_not_a_failure(tmp_path: Path) -> None:
    """Today's state: everything is general, and nothing about the pipeline
    changes. Layering must be inert until someone tags something."""
    lib = [make(tmp_path, "a", "one"), make(tmp_path, "b", "two")]
    chosen, report = layers.select_layers(lib, "any topic")
    assert len(chosen) == 2
    assert report["ranked"] is True
    assert report["domain_available"] == []


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_report_names_what_was_held_back(library) -> None:
    """"Why was scanpy never considered?" has to be answerable afterwards."""
    _, report = layers.select_layers(library, "turbulent flow velocimetry", limit=1)
    text = layers.render_layer_report(report)
    assert "General layer: 3 skill(s)" in text
    assert "Not admitted for this topic" in text


def test_report_is_explicit_when_ranking_did_not_run(library, monkeypatch) -> None:
    import core.passages as passages

    monkeypatch.setattr(
        passages, "_hybrid_scores",
        lambda c, q: (_ for _ in ()).throw(RuntimeError("x")),
    )
    _, report = layers.select_layers(library, "topic")
    text = layers.render_layer_report(report)
    assert "all 4 admitted" in text


def test_report_handles_an_empty_library() -> None:
    _, report = layers.select_layers([], "topic")
    assert "No domain-tagged skills installed." in layers.render_layer_report(report)
