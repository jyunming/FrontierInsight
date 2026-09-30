"""What a quest environment is given: packages, the selected skills' packages, and library skills on the path.

The live failure this pins (2026-09-25, a user's quest on another machine): the code-writing step asked pip for
``lieflat_charts`` (a Node.js tool skill, not a Python package) together with numpy; pip installed nothing, and the
oracle run died on ``import numpy``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core import experiment_deps as deps_mod
from core.execution import ExecutionResult
from tests.test_engine_execute_retry import _mk_engine, _mock_execute_router


def _skill(tmp_path: Path, name: str, kind: str, pip_requires: list[str] | None = None) -> SimpleNamespace:
    folder = tmp_path / "skills" / name
    folder.mkdir(parents=True, exist_ok=True)
    prov = {"pip_requires": pip_requires or []}
    return SimpleNamespace(
        name=name, path=folder, kind=SimpleNamespace(value=kind),
        provenance=lambda prov=prov: prov,
    )


def test_only_the_quests_own_files_are_kept_from_pip(tmp_path: Path) -> None:
    """Skill names still go to pip: many are named after the package they teach (matplotlib, scipy), whatever kind."""
    install, dropped = deps_mod.split_deps(
        ["numpy>=1.26", "lieflat_charts", "matplotlib", "helpers", "numpy"], local_modules=["experiment", "helpers"],
    )
    assert install == ["numpy>=1.26", "lieflat_charts", "matplotlib"]
    assert dict(dropped) == {"helpers": "is the quest's own file code/helpers.py, not a package"}


def test_a_skill_name_pip_cannot_install_is_explained_as_the_skill(tmp_path: Path) -> None:
    tool = _skill(tmp_path, "lieflat-charts", "tool")
    lib = _skill(tmp_path, "sir-kernels", "library")
    as_skill, others = deps_mod.explain_failures(
        [("lieflat_charts", "no such package on PyPI"), ("Sir.Kernels", "no such package on PyPI"),
         ("nopkg", "no such package on PyPI")], [tool, lib],
    )
    reasons = dict(as_skill)
    assert "a tool, not a Python package" in reasons["lieflat_charts"]
    assert "puts on the experiment's path" in reasons["Sir.Kernels"]
    assert others == [("nopkg", "no such package on PyPI")]


def test_the_note_names_a_selected_skill_that_cannot_be_used() -> None:
    note = deps_mod.repair_note([], [], ["broken-skill"])
    assert note.startswith("FI NOTE (skills)") and "broken-skill" in note and "do not import" in note


def test_skill_requirements_and_library_paths(tmp_path: Path) -> None:
    lib = _skill(tmp_path, "sir-kernels", "library", ["scipy>=1.11", "numpy"])
    (lib.path / "scripts").mkdir()
    tool = _skill(tmp_path, "lieflat-charts", "tool", ["numpy"])
    assert deps_mod.skill_requirements([lib, tool]) == ["scipy>=1.11", "numpy"]
    assert deps_mod.library_paths([lib, tool]) == [lib.path, lib.path / "scripts"]


def test_failure_reasons_are_plain() -> None:
    assert "no such package on PyPI" in deps_mod.failure_reason(
        "ERROR: Could not find a version that satisfies the requirement lieflat_charts\n"
        "ERROR: No matching distribution found for lieflat_charts")
    assert deps_mod.repair_note([], []) == ""
    note = deps_mod.repair_note([("lieflat_charts", "is the skill lieflat-charts, a tool")], [("nopkg", "no such package on PyPI")])
    assert note.startswith("FI NOTE (packages)") and "lieflat_charts" in note and "remove the import" in note


@pytest.mark.asyncio
async def test_one_missing_package_no_longer_leaves_numpy_uninstalled(tmp_path: Path) -> None:
    """The batch line fails on one name; FI installs the rest one at a time and tells the repair which failed."""
    eng = _mk_engine(tmp_path)
    ok = ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)
    missing = ExecutionResult(returncode=1, stdout="", stderr="ERROR: No matching distribution found for nopkg", duration_s=0.1)
    installed: list[list[str]] = []

    async def install(pkgs, *, quest_root):  # noqa: ANN001
        installed.append(list(pkgs))
        return missing if "nopkg" in pkgs else ok

    eng.executor.install = install  # type: ignore[method-assign]
    crash = ExecutionResult(returncode=1, stdout="", stderr="ModuleNotFoundError: No module named 'nopkg'", duration_s=1.0)
    eng.executor.execute = _mock_execute_router(crash)  # type: ignore[method-assign]

    update = await eng._node_execute({"deps": ["numpy", "nopkg", "matplotlib"]})

    assert installed[0] == ["numpy", "nopkg", "matplotlib"]   # one line first
    assert ["numpy"] in installed and ["matplotlib"] in installed  # then each on its own
    note = update["exec_result"]["packages_note"]
    assert note.startswith("FI NOTE (packages)") and "nopkg: no such package on PyPI" in note
    # Kept apart from the run's own stderr: the analysis reads that, and no slice of it cuts the traceback's end.
    assert update["exec_result"]["stderr_tail"] == "ModuleNotFoundError: No module named 'nopkg'"


@pytest.mark.asyncio
async def test_selected_skills_packages_are_installed_and_libraries_are_on_the_path(tmp_path: Path, monkeypatch) -> None:
    eng = _mk_engine(tmp_path)
    lib = _skill(tmp_path, "sir-kernels", "library", ["scipy"])
    tool = _skill(tmp_path, "lieflat-charts", "tool")
    monkeypatch.setattr(eng, "_experiment_skill_states", lambda _state: [SimpleNamespace(skill=lib), SimpleNamespace(skill=tool)])
    # A quest's own clean environment (the research profile's): the skills' packages go into it.
    monkeypatch.setattr(eng.config.execution, "shared_interpreter", False)
    installed: list[list[str]] = []

    async def install(pkgs, *, quest_root):  # noqa: ANN001
        installed.append(list(pkgs))
        return ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)

    eng.executor.install = install  # type: ignore[method-assign]
    envs: list[dict] = []
    happy = ExecutionResult(returncode=0, stdout='RESULT_JSON: {"v": 1}', stderr="", duration_s=1.0)

    async def execute(cmd, *_, env=None, **__):  # noqa: ANN001
        envs.append(env or {})
        return happy

    eng.executor.execute = execute  # type: ignore[method-assign]
    await eng._node_execute({"deps": ["numpy"]})

    assert installed == [["numpy", "scipy"]]      # the library's own package added
    run_env = envs[-1]
    assert str(lib.path) in run_env.get("PYTHONPATH", "").split(__import__("os").pathsep)
    assert str(tool.path) not in run_env.get("PYTHONPATH", "")


@pytest.mark.asyncio
async def test_on_the_shared_interpreter_a_quest_does_not_install_the_skills_packages(tmp_path: Path, monkeypatch) -> None:
    """They are in FI's own interpreter from the skill's approval; a quest does not change it."""
    eng = _mk_engine(tmp_path)
    lib = _skill(tmp_path, "sir-kernels", "library", ["scipy"])
    monkeypatch.setattr(eng, "_experiment_skill_states", lambda _state: [SimpleNamespace(skill=lib)])
    monkeypatch.setattr(eng.config.execution, "shared_interpreter", True)
    installed: list[list[str]] = []

    async def install(pkgs, *, quest_root):  # noqa: ANN001
        installed.append(list(pkgs))
        return ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)

    eng.executor.install = install  # type: ignore[method-assign]
    eng.executor.execute = _mock_execute_router(  # type: ignore[method-assign]
        ExecutionResult(returncode=0, stdout='RESULT_JSON: {"v": 1}', stderr="", duration_s=1.0))
    await eng._node_execute({"deps": ["numpy"]})
    assert installed == [["numpy"]]


@pytest.mark.asyncio
async def test_a_library_skill_named_unlike_its_module_is_explained_not_called_missing(tmp_path: Path, monkeypatch) -> None:
    """Every skill name goes to pip; one that is not a package fails there, and the note says how the skill
    is used, not to remove an import that works from the path."""
    eng = _mk_engine(tmp_path)
    lib = _skill(tmp_path, "sir-kernels", "library")
    monkeypatch.setattr(eng, "_experiment_skill_states", lambda _state: [SimpleNamespace(skill=lib)])

    async def install(pkgs, *, quest_root):  # noqa: ANN001
        bad = "sir-kernels" in pkgs
        return ExecutionResult(returncode=1 if bad else 0, stdout="",
                               stderr="ERROR: No matching distribution found for sir-kernels" if bad else "", duration_s=0.1)

    eng.executor.install = install  # type: ignore[method-assign]
    eng.executor.execute = _mock_execute_router(  # type: ignore[method-assign]
        ExecutionResult(returncode=0, stdout='RESULT_JSON: {"v": 1}', stderr="", duration_s=1.0))
    update = await eng._node_execute({"deps": ["numpy", "sir-kernels"]})
    note = update["exec_result"]["packages_note"]
    assert "sir-kernels: is the skill sir-kernels" in note and "remove the import" not in note


def _code(tmp_path: Path, **files: str) -> list[Path]:
    code = tmp_path / "code"
    code.mkdir(exist_ok=True)
    for name, text in files.items():
        (code / f"{name}.py").write_text(text, encoding="utf-8")
    return deps_mod.code_sources(code)


def test_a_listed_package_no_script_imports_is_not_installed(tmp_path: Path) -> None:
    """The live case (a user's quest on another machine): the model listed scikit-image, no script imported it, and
    FI installed it anyway (then the test import looked for a module ``scikit_image``)."""
    sources = _code(tmp_path, experiment="import numpy as np\nimport matplotlib.pyplot as plt\n")
    plan = deps_mod.plan_installs(
        ["matplotlib", "numpy", "scikit-image", "scipy"], sources, local=["experiment"], python=None)
    # numpy / matplotlib: imported. scipy: other libraries load it by themselves, so a listed one is kept.
    assert plan.install == ["matplotlib", "numpy", "scipy"]
    assert plan.unused == ["scikit-image"]


def test_a_listed_package_the_code_uses_is_installed(tmp_path: Path) -> None:
    sources = _code(
        tmp_path,
        experiment="from skimage.filters import sobel\ntry:\n    import cv2\nexcept ImportError:\n    cv2 = None\n",
        helpers="import importlib\nfitz = importlib.import_module('fitz')\n",
    )
    plan = deps_mod.plan_installs(["scikit-image", "opencv-python-headless", "PyMuPDF", "Pillow"], sources,
                                  local=["experiment", "helpers"], python=None)
    # skimage imported; cv2 an optional import (still a use); fitz named in a string. Pillow: nothing uses PIL.
    assert plan.install == ["scikit-image", "opencv-python-headless", "PyMuPDF"]
    assert plan.unused == ["Pillow"]


def test_what_fi_cannot_judge_is_installed_as_before(tmp_path: Path) -> None:
    sources = _code(tmp_path, experiment="import serial_thing\n")
    # A package whose import name FI does not know (and is not installed to ask): kept, never dropped on a guess --
    # dropping it would fail the run, the repair would list it again, and it would be dropped again.
    plan = deps_mod.plan_installs(["some-unknown-pkg", "sir-kernels"], sources, local=["experiment"], python=None)
    assert plan.install == ["some-unknown-pkg", "sir-kernels"] and plan.unused == []
    # A selected skill's name is kept too, whatever the code imports.
    plan = deps_mod.plan_installs(["scikit-image"], sources, local=["experiment"], python=None, keep=["scikit-image"])
    assert plan.install == ["scikit-image"]
    # A script that does not parse: what it imports cannot be told, so everything listed is installed.
    broken = _code(tmp_path, analysis="def (:\n")
    plan = deps_mod.plan_installs(["scikit-image"], broken, local=[], python=None)
    assert plan.install == ["scikit-image"] and plan.unused == []
    assert "cannot parse" in plan.all_because  # run.log says why
    # No scripts at all: the same.
    plan = deps_mod.plan_installs(["scikit-image"], [], local=[], python=None)
    assert plan.install == ["scikit-image"] and "no script" in plan.all_because


def test_what_a_library_skill_on_the_path_imports_counts_as_used(tmp_path: Path) -> None:
    sources = _code(tmp_path, experiment="import sir_kernels\n")
    lib = tmp_path / "skills" / "sir-kernels"
    lib.mkdir(parents=True)
    (lib / "sir_kernels.py").write_text("from skimage import measure\n", encoding="utf-8")
    (lib / "legacy.py").write_text("print 'python 2'\n", encoding="utf-8")  # cannot be imported: skipped
    tool = tmp_path / "skills" / "img-tool"
    (tool / "scripts").mkdir(parents=True)
    (tool / "scripts" / "run_it.py").write_text("import cv2\n", encoding="utf-8")
    plan = deps_mod.plan_installs(
        ["scikit-image", "Pillow", "opencv-python"], sources, local=["experiment"], python=None,
        skill_files=deps_mod.skill_sources([SimpleNamespace(path=lib), SimpleNamespace(path=tool)]))
    # skimage: the library skill imports it. cv2: a tool skill's script the experiment runs imports it.
    assert plan.install == ["scikit-image", "opencv-python"] and plan.unused == ["Pillow"]


def test_an_installed_package_is_kept_and_only_a_missing_one_can_be_left_out(tmp_path: Path, monkeypatch) -> None:
    """Leaving out an installed package saves nothing and loses it from requirements.txt and the lock file (torch
    under transformers). The environment's record never makes a package count as unused."""
    sources = _code(tmp_path, experiment="import numpy\n")
    monkeypatch.setattr(deps_mod, "env_packages", lambda _py, _d, _m=(): deps_mod.EnvInfo(
        dists={"numpy": ["numpy"], "seaborn": ["seaborn"], "brand-new-pkg": None, "scikit-image": None,
               "pillow": ["PIL"], "pyyaml": []}, present=set()))
    plan = deps_mod.plan_installs(["numpy", "seaborn", "brand-new-pkg", "scikit-image", "Pillow", "PyYAML"], sources,
                                  local=["experiment"], python="py")
    # PyYAML: installed, though its record names no module (an editable install): still installed.
    assert plan.install == ["numpy", "seaborn", "brand-new-pkg", "Pillow", "PyYAML"]
    assert plan.unused == ["scikit-image"]


def test_after_a_run_failed_on_a_missing_module_everything_listed_is_installed(tmp_path: Path) -> None:
    """A package another library imports by itself can look unused; once the run fails on an import and the repair
    lists it again, it is installed rather than left out again until the repair budget is spent."""
    sources = _code(tmp_path, experiment="import numpy\n")
    plan = deps_mod.plan_installs(["numpy", "sklearn", "scikit-image"], sources, local=["experiment"], python=None,
                                  keep_listed=True)
    assert plan.install == ["numpy", "scikit-learn", "scikit-image"] and plan.unused == []


def test_a_direct_reference_keeps_its_name_and_blocks_a_second_copy(tmp_path: Path, monkeypatch) -> None:
    assert deps_mod.requirement_name("scikit-image @ git+https://github.com/x/y") == "scikit-image"
    assert deps_mod.requirement_name("git+https://github.com/x/y@main") == ""
    assert deps_mod.requirement_name("git@github.com:o/r.git") == ""
    assert deps_mod.requirement_name("foo @ file:///x/foo.whl") == "foo"
    sources = _code(tmp_path, experiment="import skimage\nimport cv2\n")
    monkeypatch.setattr(deps_mod, "env_packages", lambda _py, _d, _m=(): deps_mod.EnvInfo(dists={}, present=set()))
    plan = deps_mod.plan_installs(["scikit-image @ git+https://github.com/x/y"], sources, local=[], python="py")
    assert plan.install == ["scikit-image @ git+https://github.com/x/y", "opencv-python"]
    # An alternative build FI does not know may be the cv2 already: no second cv2 next to it (skimage is still added).
    plan = deps_mod.plan_installs(["opencv-contrib-python-rolling"], sources, local=[], python="py")
    assert plan.added == [("skimage", "scikit-image")]
    # A request with no name at all could be anything: nothing is added next to it.
    plan = deps_mod.plan_installs(["git+https://github.com/x/y"], sources, local=[], python="py")
    assert plan.added == []


def test_an_unknown_new_package_does_not_stop_a_well_known_import_being_added(tmp_path: Path, monkeypatch) -> None:
    sources = _code(tmp_path, experiment="import cv2\nimport numpy\nfrom PIL import Image\n")
    monkeypatch.setattr(deps_mod, "env_packages", lambda _py, _d, _m=(): deps_mod.EnvInfo(
        dists={"numpy": ["numpy"], "optuna": None, "pillow-simd": None}, present=set()))
    plan = deps_mod.plan_installs(["numpy", "optuna"], sources, local=["experiment"], python="py")
    assert plan.added == [("PIL", "Pillow"), ("cv2", "opencv-python")]
    plan = deps_mod.plan_installs(["numpy", "pillow-simd"], sources, local=["experiment"], python="py")
    assert plan.added == [("cv2", "opencv-python")]  # pillow-simd may be the PIL already
    # The module's name as a whole word only: yamllint is not PyYAML, biotite is not biopython...
    assert not deps_mod._may_provide("yamllint", "yaml", "PyYAML")
    assert not deps_mod._may_provide("biotite", "Bio", "biopython")
    assert not deps_mod._may_provide("cryptography", "Crypto", "pycryptodome")
    assert not deps_mod._may_provide("scikit-network", "skimage", "scikit-image")
    # ...but a fork or another build of the package may be it.
    assert deps_mod._may_provide("scikit-image-nightly", "skimage", "scikit-image")
    assert deps_mod._may_provide("python-dateutil-fork", "dateutil", "python-dateutil")
    assert deps_mod._may_provide("opencv-contrib-python-rolling", "cv2", "opencv-python")


def test_when_the_environment_cannot_be_asked_everything_listed_is_installed(tmp_path: Path, monkeypatch) -> None:
    sources = _code(tmp_path, experiment="import numpy\n")
    monkeypatch.setattr(deps_mod, "env_packages", lambda _py, _d, _m=(): None)
    plan = deps_mod.plan_installs(["numpy", "scikit-image"], sources, local=["experiment"], python="py")
    assert plan.install == ["numpy", "scikit-image"] and plan.unused == []
    assert "could not be asked" in plan.all_because  # run.log says why


def test_code_sources_and_the_quests_own_module_names(tmp_path: Path) -> None:
    code = tmp_path / "code"
    for rel in ("experiment.py", "run.py", "src/community.py", "pkg/__init__.py", "pkg/run.py", ".git/hook.py",
                "run_output/copy.py", "pkg/optuna.py", "utils/numba.py", "src/mylib/core.py"):
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        (code / rel).write_text("x = 1\n", encoding="utf-8")
    got = {p.relative_to(code).as_posix() for p in deps_mod.code_sources(code)}
    # FI's run.py is left out only at the top.
    assert got == {"experiment.py", "src/community.py", "pkg/__init__.py", "pkg/run.py", "pkg/optuna.py",
                   "utils/numba.py", "src/mylib/core.py"}
    local = deps_mod.local_module_names(code)
    # src/ is a source folder a project puts on the path: its modules and sub-folders are imported by name.
    assert {"experiment", "community", "pkg", "src", "utils", "mylib"} <= local
    # A module inside any other folder does not hide the library it is named after.
    assert "optuna" not in local and "numba" not in local
    assert len(deps_mod.code_sources(code, limit=2)) == 2


def test_a_well_known_import_the_list_lacks_is_installed_under_its_pip_name(tmp_path: Path, monkeypatch) -> None:
    sources = _code(tmp_path, experiment="import skimage\nimport sklearn.linear_model\nimport seaborn\nimport numpy\n")
    asked: dict = {}

    def env(_py, dists, modules=()):  # noqa: ANN001
        asked["modules"] = list(modules)
        return deps_mod.EnvInfo(dists={"numpy": ["numpy"]}, present={"sklearn"})

    monkeypatch.setattr(deps_mod, "env_packages", env)
    plan = deps_mod.plan_installs(["numpy"], sources, local=["experiment"], python="py")
    # sklearn is already importable; seaborn is not a name whose package FI knows differs, so the run decides.
    assert asked["modules"] == ["skimage", "sklearn"]
    assert plan.added == [("skimage", "scikit-image")]
    assert plan.install == ["numpy", "scikit-image"]
    # In a container (not asked), nothing is added.
    assert deps_mod.plan_installs(["numpy"], sources, local=["experiment"], python=None).added == []


def test_an_import_name_listed_as_a_package_is_asked_of_pip_by_its_package_name(tmp_path: Path) -> None:
    sources = _code(tmp_path, experiment="import sklearn\nimport cv2\n")
    plan = deps_mod.plan_installs(["sklearn>=1.3", "cv2", "scikit-learn"], sources, local=["experiment"], python=None)
    assert plan.install == ["scikit-learn>=1.3", "opencv-python"]  # one request per package: the first
    assert plan.renamed == [("sklearn>=1.3", "scikit-learn>=1.3"), ("cv2", "opencv-python")]
    # A selected skill that happens to share an import name keeps its own name (explain_failures finds it by it).
    plan = deps_mod.plan_installs(["umap"], sources, local=["experiment"], python=None, keep=["umap"])
    assert plan.install == ["umap"] and plan.renamed == []


def test_env_packages_asks_the_real_interpreter() -> None:
    import sys

    info = deps_mod.env_packages(sys.executable, ["pytest", "no-such-dist-fi-test"], ["json", "no_such_mod_fi_test"])
    assert info is not None
    assert "pytest" in (info.dists["pytest"] or [])
    assert info.dists["no-such-dist-fi-test"] is None
    assert info.present == {"json"}
    assert deps_mod.env_packages(Path("no") / "such" / "python", ["pytest"]) is None


def _env_with(*installed: str):  # noqa: ANN202
    def env(_py, dists, modules=()):  # noqa: ANN001
        names = [deps_mod.normalize(deps_mod.requirement_name(d)) for d in dists]
        return deps_mod.EnvInfo(dists={n: ([n] if n in installed else None) for n in names}, present=set())
    return env


@pytest.mark.asyncio
async def test_execute_installs_what_the_code_uses_and_says_what_it_left_out(tmp_path: Path, monkeypatch) -> None:
    eng = _mk_engine(tmp_path)
    (eng.quest_root / "code" / "experiment.py").write_text("import matplotlib\nimport numpy\n", encoding="utf-8")
    monkeypatch.setattr(deps_mod, "env_packages", _env_with("matplotlib", "numpy"))
    installed: list[list[str]] = []

    async def install(pkgs, *, quest_root):  # noqa: ANN001
        installed.append(list(pkgs))
        return ExecutionResult(returncode=0, stdout="", stderr="", duration_s=0.1)

    eng.executor.install = install  # type: ignore[method-assign]
    eng.executor.execute = _mock_execute_router(  # type: ignore[method-assign]
        ExecutionResult(returncode=0, stdout='RESULT_JSON: {"v": 1}', stderr="", duration_s=1.0))
    update = await eng._node_execute({"deps": ["matplotlib", "numpy", "scikit-image"]})

    assert installed[0] == ["matplotlib", "numpy"]
    log = (eng.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "the model listed scikit-image but the code does not import it; not installed" in log
    assert "scikit-image: the code does not import it" in update["exec_result"]["packages_note"]
    assert "scikit_image" not in log

    # The run then fails on a module scikit-image's code would have imported, and the repair lists it again (the
    # script still does not import it): this time it is installed, not left out until the repair budget is spent.
    installed.clear()
    failed = {"stderr_tail": "ModuleNotFoundError: No module named 'skimage'", "stdout_tail": ""}
    await eng._node_execute({"deps": ["matplotlib", "numpy", "scikit-image"], "exec_result": failed})
    assert installed[0] == ["matplotlib", "numpy", "scikit-image"]
    log = (eng.quest_root / ".fi" / "run.log").read_text(encoding="utf-8")
    assert "installing every package the model listed: the last run failed on an import" in log


def test_unusable_selected_skills_are_recorded(tmp_path: Path, monkeypatch) -> None:
    import core.engine as engine_mod

    eng = _mk_engine(tmp_path)
    ok = SimpleNamespace(skill=_skill(tmp_path, "good", "library"))
    monkeypatch.setattr(engine_mod, "_resolve_selected_skills", lambda *a, **k: ([ok], {}))
    state = {"selected_skills": ["good", "broken", "a-writing-skill"],
             "skill_selection": {"uses": {"a-writing-skill": "writing"}}}
    assert eng._experiment_skill_states(state) == [ok]
    assert eng._unusable_skills == ["broken"]
