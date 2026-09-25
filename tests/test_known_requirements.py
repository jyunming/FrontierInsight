"""The preset skills' packages, and the versions they must not get (core/skills/known_requirements.py)."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from core.skills import known_requirements as kr

ROOT = Path(__file__).resolve().parent.parent


def _script_skills() -> dict:
    tree = ast.parse((ROOT / "scripts" / "import_scientist_skills.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "SKILLS":
            return ast.literal_eval(node.value)
    raise AssertionError("no SKILLS table in the import script")


def test_every_preset_skill_has_a_package_list_and_no_other() -> None:
    """The import script reads its package lists from the table, so every skill it imports is in it."""
    assert set(_script_skills()) == set(kr.PRESET_PIP_REQUIRES)


def test_the_node_chart_tool_is_gone_and_python_plotting_skills_are_in() -> None:
    skills = _script_skills()
    assert "lieflat-charts" not in skills
    assert not (ROOT / "scripts" / "skill-assets").exists()
    for name in ("matplotlib", "seaborn", "scientific-visualization"):
        assert skills[name][0] == "kdense"
        assert kr.PRESET_PIP_REQUIRES[name]


PRESET = {"imported_from": "C:/dev/FrontierInsight/.skill-sources/geoscience/landlab/SKILL.md"}


def test_landlab_is_pinned_wherever_it_comes_from_on_the_python_that_needs_it() -> None:
    """landlab 2.11.0 declares Python >= 3.11 but cannot be imported under 3.11 (3.12-only syntax)."""
    on311 = (3, 11)
    assert kr.pinned("landlab", python=on311) == "landlab==2.10.1"
    assert kr.pinned("Landlab", python=on311) == "Landlab==2.10.1"
    assert kr.pinned("landlab[all]", python=on311) == "landlab[all]==2.10.1", "extras kept"
    assert kr.pinned('landlab; python_version < "3.13"', python=on311) == 'landlab==2.10.1; python_version < "3.13"'
    assert kr.pinned("landlab>=2.9", python=on311) == "landlab>=2.9", "a requirement that names a version is the person's choice"
    assert kr.pinned("landlab", python=(3, 12)) == "landlab", "2.11.0 imports fine on 3.12"
    assert kr.pinned("numpy", python=on311) == "numpy"


def test_pip_requires_prefers_what_the_skill_recorded_and_falls_back_to_the_table_for_a_preset() -> None:
    recorded = SimpleNamespace(name="scipy", provenance=lambda: {"pip_requires": ["scipy", "numpy"]})
    assert kr.pip_requires(recorded) == ["scipy", "numpy"]
    old_preset = SimpleNamespace(name="scipy", provenance=lambda: dict(PRESET))  # imported before pip_requires existed
    assert kr.pip_requires(old_preset) == kr.PRESET_PIP_REQUIRES["scipy"]
    declared_none = SimpleNamespace(name="scipy", provenance=lambda: {"pip_requires": [], **PRESET})
    assert kr.pip_requires(declared_none) == [], "an empty list is a declaration, not a missing record"
    own_named_like_a_preset = SimpleNamespace(name="pymc", provenance=lambda: {"imported_from": "C:/me/skills/pymc/SKILL.md"})
    assert kr.pip_requires(own_named_like_a_preset) == [], "a skill of one's own gets nothing it did not declare"


def test_the_cobra_module_maps_to_a_package_that_exists() -> None:
    import launch

    assert launch._MODULE_TO_PIP.get("cobra", "cobra") == "cobra"


def test_a_quest_installs_a_preset_skills_packages_even_when_its_record_has_none() -> None:
    """Most preset skills on a user's machine were imported before pip_requires was recorded: a quest in its own clean
    environment still gets their packages, pinned (landlab 2.10.1)."""
    from core import experiment_deps

    import sys

    old_landlab = SimpleNamespace(name="landlab", provenance=lambda: dict(PRESET))
    old_scipy = SimpleNamespace(name="scipy", provenance=lambda: dict(PRESET))
    landlab = "landlab==2.10.1" if sys.version_info[:2] < (3, 12) else "landlab"
    assert experiment_deps.skill_requirements([old_landlab, old_scipy]) == [landlab, "scipy"]
