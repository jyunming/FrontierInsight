"""Scaffolding a skill from software that already exists.

Teaching FI a skill by hand means writing four files, two of them tedious to
get right. This drafts them — but the value depends on drawing the line in the
right place:

**Introspection, not generation.** A model asked to write down a library's
signatures will occasionally invent one, and a confidently wrong signature is
worse than none: it teaches every future quest to call something that does not
exist. These tests pin that the surface comes from ``inspect``.

**Judgement stays with the person.** When the skill applies, when it does not,
and what its outputs may legally be are left as TODOs. A scaffold that filled
those in with plausible text would manufacture exactly the false confidence the
promotion gate exists to catch.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from core.skills import Skill, Status, evaluate
from core.skills import scaffold


@pytest.fixture
def fake_pkg(tmp_path: Path, monkeypatch):
    """A tiny installable package to introspect."""
    pkg = tmp_path / "pkg_demo"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        textwrap.dedent('''
            """Demo package."""

            class Widget:
                """A widget."""

            def spin(rpm: float = 10.0, *, reverse: bool = False) -> float:
                """Spin the widget."""
                return rpm
        '''),
        encoding="utf-8",
    )
    (pkg / "extra.py").write_text(
        textwrap.dedent('''
            """Extra bits."""
            import json  # imported, must NOT be advertised as ours

            def measure(x: int) -> int:
                """Measure something."""
                return x
        '''),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    # The self-test runs in a subprocess, deliberately: it must verify the
    # module is importable in a fresh interpreter, which is what catches a
    # broken or half-upgraded install. So the child needs the path too —
    # sys.path surgery in this process does not reach it.
    import os as _os

    monkeypatch.setenv(
        "PYTHONPATH",
        _os.pathsep.join([str(tmp_path), _os.environ.get("PYTHONPATH", "")]).strip(
            _os.pathsep
        ),
    )
    for mod in [m for m in sys.modules if m.startswith("pkg_demo")]:
        del sys.modules[mod]
    return "pkg_demo"


# ---------------------------------------------------------------------------
# The API surface is read, not written
# ---------------------------------------------------------------------------


def test_signatures_come_from_introspection(fake_pkg) -> None:
    surface, modules, count = scaffold.render_api_surface(fake_pkg)
    # The real defaults and annotations, not a paraphrase of them.
    assert "def spin(rpm: float = 10.0, *, reverse: bool = False) -> float" in surface
    assert "class Widget" in surface
    assert count >= 2


def test_submodules_are_included(fake_pkg) -> None:
    surface, modules, _ = scaffold.render_api_surface(fake_pkg)
    assert "pkg_demo.extra" in modules
    assert "def measure(x: int) -> int" in surface


def test_imported_names_are_not_advertised_as_ours(fake_pkg) -> None:
    """A module that imports json must not offer json's API as its own."""
    surface, _, _ = scaffold.render_api_surface(fake_pkg)
    assert "json" not in surface.replace("pkg_demo", "")


def test_docstring_summaries_ride_along(fake_pkg) -> None:
    surface, _, _ = scaffold.render_api_surface(fake_pkg)
    assert "# Spin the widget." in surface


def test_c_extension_signatures_come_from_the_docstring() -> None:
    """Scientific libraries are very often C extensions, and ``inspect``
    cannot read their signatures — it raises, and a naive scaffold emits a
    useless ``(...)``.

    Those libraries conventionally put the real signature on the first
    docstring line, so it is read from there. Still not generation: the string
    comes from the package either way.

    Found by scaffolding a real library (gdstk) rather than a fixture.
    """
    gdstk = pytest.importorskip("gdstk")

    surface, _, _ = scaffold.render_api_surface("gdstk")
    assert "class Polygon(points, layer=0, datatype=0)" in surface, surface[:400]
    assert "(...)" not in surface, "a C-extension signature was lost"


def test_docstring_signature_is_not_repeated_as_a_comment() -> None:
    """When the docstring's first line IS the signature, echoing it after a
    ``#`` is pure noise in a prompt that pays for every character."""
    pytest.importorskip("gdstk")

    surface, _, _ = scaffold.render_api_surface("gdstk")
    assert "class Cell(name)    # Cell(name)" not in surface


def test_generated_markdown_is_well_formed(fake_pkg) -> None:
    surface, _, _ = scaffold.render_api_surface(fake_pkg)
    assert surface.count("```") % 2 == 0


# ---------------------------------------------------------------------------
# What the scaffold refuses to decide
# ---------------------------------------------------------------------------


def test_draft_writes_the_full_envelope(fake_pkg, tmp_path: Path) -> None:
    d = scaffold.draft("demo", fake_pkg, tmp_path / "root")
    for f in ("SKILL.md", "api_surface.md", "selftest.py", "provenance.json"):
        assert (d.path / f).is_file(), f
    assert d.entries >= 2


def test_judgement_is_left_as_todos(fake_pkg, tmp_path: Path) -> None:
    """The scaffold must not manufacture coverage claims it cannot justify."""
    d = scaffold.draft("demo", fake_pkg, tmp_path / "root")
    skill_md = (d.path / "SKILL.md").read_text(encoding="utf-8")
    assert "When NOT to use this" in skill_md
    assert skill_md.count("TODO") >= 4
    prov = json.loads((d.path / "provenance.json").read_text(encoding="utf-8"))
    assert prov["result_assertions"] == [], "bounds must not be invented"


def test_scaffolded_skill_is_proposed_not_trusted(fake_pkg, tmp_path: Path) -> None:
    """Drafting is not approving. A fresh scaffold must sit at the gate."""
    d = scaffold.draft("demo", fake_pkg, tmp_path / "root")
    state = evaluate(Skill(name="demo", path=d.path), ledger=tmp_path / "l.json")
    assert state.status is Status.PROPOSED
    assert not state.loadable


def test_scaffolded_selftest_runs_and_passes(fake_pkg, tmp_path: Path) -> None:
    """The stub must be green out of the box — a scaffold that ships a failing
    test would train people to ignore the quarantine state."""
    d = scaffold.draft("demo", fake_pkg, tmp_path / "root")
    from core.skills.registry import run_selftest

    passed, output = run_selftest(Skill(name="demo", path=d.path))
    assert passed, output


def test_selftest_says_it_is_not_yet_sufficient(fake_pkg, tmp_path: Path) -> None:
    """Importability is not correctness, and the file must say so — otherwise
    a green stub reads as certification."""
    d = scaffold.draft("demo", fake_pkg, tmp_path / "root")
    text = (d.path / "selftest.py").read_text(encoding="utf-8")
    assert "not enough to certify" in text
    assert "TODO" in text


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_existing_skill_is_not_silently_overwritten(fake_pkg, tmp_path: Path) -> None:
    """Rewriting would discard someone's judgement and lapse approval without
    saying so."""
    root = tmp_path / "root"
    scaffold.draft("demo", fake_pkg, root)
    with pytest.raises(FileExistsError, match="lapses its approval"):
        scaffold.draft("demo", fake_pkg, root)


def test_overwrite_is_possible_when_asked(fake_pkg, tmp_path: Path) -> None:
    root = tmp_path / "root"
    scaffold.draft("demo", fake_pkg, root)
    assert scaffold.draft("demo", fake_pkg, root, overwrite=True).entries >= 2


def test_unimportable_module_raises_rather_than_guessing(tmp_path: Path) -> None:
    with pytest.raises(ModuleNotFoundError):
        scaffold.draft("demo", "no_such_module_xyz", tmp_path / "root")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_requires_a_source_module(capsys, tmp_path, monkeypatch) -> None:
    import launch

    monkeypatch.setenv("FI_SKILLS_DIR", str(tmp_path / "skills"))
    assert launch._teach_skill("demo", "") == 2
    assert "--from" in capsys.readouterr().out


def test_cli_reports_an_unimportable_module_actionably(
    capsys, tmp_path, monkeypatch
) -> None:
    import launch

    monkeypatch.setenv("FI_SKILLS_DIR", str(tmp_path / "skills"))
    assert launch._teach_skill("demo", "no_such_module_xyz") == 1
    out = capsys.readouterr().out
    assert "Install it" in out and "rather than guessing" in out


def test_cli_drafts_and_points_at_the_next_step(
    fake_pkg, capsys, tmp_path, monkeypatch
) -> None:
    import launch

    monkeypatch.setenv("FI_SKILLS_DIR", str(tmp_path / "skills"))
    assert launch._teach_skill("demo", fake_pkg) == 0
    out = capsys.readouterr().out
    assert "introspection" in out
    assert "When NOT to use" in out          # the section it flags as load-bearing
    assert "--approve-skill demo" in out     # the gate it does not cross


# ---------------------------------------------------------------------------
# Generated self-tests
#
# The user chose generated self-tests over hand-written ones. That is a
# legitimate trade — writing 39 by hand is the real cost of a skill library —
# but it weakens what TRUSTED means, because a generated test proves the
# tooling runs and not that it behaves as described.
#
# These pin that the weakening is *visible*. A green tick that quietly covers
# less than it used to is the failure this whole subsystem exists to prevent.
# ---------------------------------------------------------------------------


def test_generated_selftest_marks_itself(tmp_path: Path) -> None:
    d = tmp_path / "imported"
    (d / "scripts").mkdir(parents=True)
    (d / "SKILL.md").write_text("# imported\n", encoding="utf-8")
    scaffold.generate_selftest(d, "imported")

    text = (d / "selftest.py").read_text(encoding="utf-8")
    assert scaffold.GENERATED_MARKER in text
    assert "not written by a person" in text
    assert "does not prove" in text
    assert scaffold.selftest_is_generated(d)


def test_the_marker_lives_in_the_file_not_in_provenance(tmp_path: Path) -> None:
    """Rewriting the test is all it takes to clear the marker.

    A provenance flag would go stale at exactly the moment it mattered — the
    moment a person replaces the generated test with a real one — and would
    then describe hand-written work as generated.
    """
    d = tmp_path / "s"
    d.mkdir()
    (d / "SKILL.md").write_text("# s\n", encoding="utf-8")
    scaffold.generate_selftest(d, "s")
    assert scaffold.selftest_is_generated(d)

    (d / "selftest.py").write_text(
        "# a real check someone wrote\nimport sys; sys.exit(0)\n", encoding="utf-8"
    )
    assert not scaffold.selftest_is_generated(d)


def test_generated_selftest_probes_bundled_scripts(tmp_path: Path) -> None:
    """Every bundled script must answer --help with exit 0 — the published
    libraries' own contributor contract already requires it, so this is a real
    check of 'the tooling is present and runnable' needing no per-skill
    knowledge."""
    d = tmp_path / "withscripts"
    (d / "scripts").mkdir(parents=True)
    (d / "SKILL.md").write_text("# withscripts\n", encoding="utf-8")
    (d / "scripts" / "tool.py").write_text(
        "import argparse\n"
        "argparse.ArgumentParser().parse_args()\n",
        encoding="utf-8",
    )
    scaffold.generate_selftest(d, "withscripts")

    from core.skills.registry import run_selftest

    passed, out = run_selftest(Skill(name="withscripts", path=d))
    assert passed, out
    # The probe asserts the script LOADS, not that it follows a --help
    # convention: plenty of legitimate scripts take a positional argument or
    # are helper modules, and failing those quarantined working skills.
    assert "present and loadable" in out


def test_generated_selftest_fails_on_a_broken_script(tmp_path: Path) -> None:
    """The check has teeth: a script that cannot run must quarantine the
    skill, which is the one thing a generated test genuinely can catch."""
    d = tmp_path / "broken"
    (d / "scripts").mkdir(parents=True)
    (d / "SKILL.md").write_text("# broken\n", encoding="utf-8")
    (d / "scripts" / "tool.py").write_text(
        "import nonexistent_module_xyz\n", encoding="utf-8"
    )
    scaffold.generate_selftest(d, "broken")

    from core.skills.registry import run_selftest

    passed, out = run_selftest(Skill(name="broken", path=d))
    assert not passed
    assert "tool.py" in out


def test_generated_selftest_resolves_nested_sibling_imports(tmp_path: Path) -> None:
    """A script two directories deep that imports a sibling package a level
    up (``from helpers import x`` when ``helpers/`` sits next to ``scripts/``,
    not next to the script itself) must not fail -- Python only puts the
    invoked script's own directory on sys.path by default, so this failed
    with ModuleNotFoundError before the probe started adding ancestor
    directories to PYTHONPATH. Reproduces the real xlsx skill's layout
    (scripts/office/validators/docx.py importing scripts/office/helpers/)."""
    d = tmp_path / "nested"
    (d / "scripts" / "helpers").mkdir(parents=True)
    (d / "scripts" / "validators").mkdir(parents=True)
    (d / "SKILL.md").write_text("# nested\n", encoding="utf-8")
    (d / "scripts" / "helpers" / "__init__.py").write_text(
        "def safe_extract():\n    pass\n", encoding="utf-8"
    )
    (d / "scripts" / "validators" / "docx.py").write_text(
        "import argparse\n"
        "from helpers import safe_extract\n"
        "if __name__ == '__main__':\n"
        "    argparse.ArgumentParser().parse_args()\n",
        encoding="utf-8",
    )
    scaffold.generate_selftest(d, "nested")

    from core.skills.registry import run_selftest

    passed, out = run_selftest(Skill(name="nested", path=d))
    assert passed, out
    assert "present and loadable" in out


def test_generated_selftest_tolerates_a_relative_import_submodule(tmp_path: Path) -> None:
    """A file using `from . import x` cannot be run standalone by Python's
    own import semantics, no matter what's on PYTHONPATH -- it raises
    "attempted relative import with no known parent package" for a
    correctly-installed package just as readily as a broken one. That
    message must land in the lenient "unprobed" bucket, not "failures"."""
    d = tmp_path / "relimport"
    (d / "scripts").mkdir(parents=True)
    (d / "SKILL.md").write_text("# relimport\n", encoding="utf-8")
    (d / "scripts" / "_sibling.py").write_text("VALUE = 1\n", encoding="utf-8")
    (d / "scripts" / "submodule.py").write_text(
        "from . import _sibling\n", encoding="utf-8"
    )
    scaffold.generate_selftest(d, "relimport")

    from core.skills.registry import run_selftest

    passed, out = run_selftest(Skill(name="relimport", path=d))
    assert passed, out
    assert "submodule.py" in out  # listed under "did not answer --help"


def test_generated_selftest_treats_a_help_timeout_as_unprobed_not_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A script that runs long enough to hit the --help probe's timeout has
    already imported everything it needs -- a genuinely missing dependency
    fails within milliseconds, not after the full timeout window. Verified
    by calling the generated module's main() directly with subprocess.run
    patched to raise TimeoutExpired, rather than actually waiting out a real
    60s timeout in the test suite.

    Patches via the `monkeypatch` fixture (auto-restored at test end) rather
    than a raw `mod.subprocess.run = ...` assignment -- `import subprocess`
    inside the generated module binds to the SAME process-wide module
    object every other caller uses, so an unrestored raw assignment leaks
    into every later test's `subprocess.run` calls in this session (caught
    live: it broke run_selftest()'s OWN subprocess.run in two unrelated
    tests below, each instantly "timing out" instead of actually running)."""
    d = tmp_path / "slowdemo"
    (d / "scripts").mkdir(parents=True)
    (d / "SKILL.md").write_text("# slowdemo\n", encoding="utf-8")
    (d / "scripts" / "demo.py").write_text(
        "import time\ntime.sleep(9999)\n", encoding="utf-8"
    )
    selftest_path = scaffold.generate_selftest(d, "slowdemo")

    import importlib.util
    import subprocess as real_subprocess

    spec = importlib.util.spec_from_file_location("generated_selftest", selftest_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    def fake_run(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise real_subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 60))

    monkeypatch.setattr(real_subprocess, "run", fake_run)
    rc = mod.main()
    assert rc == 0


def test_a_scriptless_skill_says_it_proved_nothing(tmp_path: Path) -> None:
    """The honest edge of the user's choice.

    For a prose-only skill a generated test cannot probe anything at all. It
    still passes — so the skill can be approved — but it must say loudly that
    a green tick here means only that the files are on disk.
    """
    d = tmp_path / "prose"
    d.mkdir()
    (d / "SKILL.md").write_text("# prose\n\nHow to think about X.\n", encoding="utf-8")
    scaffold.generate_selftest(d, "prose")

    from core.skills.registry import run_selftest

    passed, out = run_selftest(Skill(name="prose", path=d))
    assert passed, out
    assert "cannot probe" in out
    assert "no more" in out


def test_a_missing_skill_md_fails_the_generated_test(tmp_path: Path) -> None:
    d = tmp_path / "gone"
    d.mkdir()
    (d / "SKILL.md").write_text("# gone\n", encoding="utf-8")
    scaffold.generate_selftest(d, "gone")
    (d / "SKILL.md").unlink()

    from core.skills.registry import run_selftest

    passed, out = run_selftest(Skill(name="gone", path=d))
    assert not passed and "SKILL.md" in out


def test_generation_refuses_to_replace_a_written_test(tmp_path: Path) -> None:
    """Silently downgrading a hand-written test to a generated one would
    weaken the gate without anyone deciding to."""
    d = tmp_path / "hand"
    d.mkdir()
    (d / "SKILL.md").write_text("# hand\n", encoding="utf-8")
    (d / "selftest.py").write_text("import sys; sys.exit(0)\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must not silently replace"):
        scaffold.generate_selftest(d, "hand")
    assert scaffold.generate_selftest(d, "hand", overwrite=True).is_file()


def test_generation_ignores_the_compatibility_prose(tmp_path: Path) -> None:
    """Built from the file listing, never from documentation English.

    Parsing that prose to decide dependencies failed three times while this
    library was being surveyed — "accountability" read as a credential, an
    optional GPU read as required. A self-test built on such a misreading
    would quarantine working skills or pass broken ones.
    """
    d = tmp_path / "prosey"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: prosey\n"
        "compatibility: Requires Python 3.99, CUDA 13, an API key, and conda.\n"
        "---\n# prosey\n",
        encoding="utf-8",
    )
    scaffold.generate_selftest(d, "prosey")
    text = (d / "selftest.py").read_text(encoding="utf-8")
    for leaked in ("3.99", "CUDA", "API key", "conda"):
        assert leaked not in text, f"{leaked!r} leaked in from the prose"
