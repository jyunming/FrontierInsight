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


def test_landlab_is_pinned_wherever_it_comes_from() -> None:
    """landlab 2.11.0 declares Python >= 3.11 but cannot be imported under 3.11 (3.12-only syntax)."""
    assert kr.PRESET_PIP_REQUIRES["landlab"] == ["landlab==2.10.1"]
    assert kr.pinned("landlab") == "landlab==2.10.1"
    assert kr.pinned("Landlab") == "landlab==2.10.1"
    assert kr.pinned("landlab>=2.9") == "landlab>=2.9", "a requirement that names a version is the person's choice"
    assert kr.pinned("numpy") == "numpy"


def test_pip_requires_prefers_what_the_skill_recorded_and_falls_back_to_the_table() -> None:
    recorded = SimpleNamespace(name="landlab", provenance=lambda: {"pip_requires": ["landlab", "numpy"]})
    assert kr.pip_requires(recorded) == ["landlab==2.10.1", "numpy"]
    old_import = SimpleNamespace(name="scipy", provenance=lambda: {})          # imported before pip_requires existed
    assert kr.pip_requires(old_import) == kr.PRESET_PIP_REQUIRES["scipy"]
    declared_none = SimpleNamespace(name="scipy", provenance=lambda: {"pip_requires": []})
    assert kr.pip_requires(declared_none) == [], "an empty list is a declaration, not a missing record"
    unknown = SimpleNamespace(name="my-own-skill", provenance=lambda: {})
    assert kr.pip_requires(unknown) == []


def test_the_cobra_module_maps_to_a_package_that_exists() -> None:
    import launch

    assert launch._MODULE_TO_PIP.get("cobra", "cobra") == "cobra"
