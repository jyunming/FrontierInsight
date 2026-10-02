"""The first step after ``pip install`` works from the wheel alone.

The install guide used to say ``fi --config examples/integrator_bakeoff/config.yaml`` right after ``pip install``,
but ``examples/`` is not in the wheel (only the source checkout and the sdist carry it), so the first command a
pip user typed named a file that was not there. The first step is now ``fi demo``, which writes its example from
the package itself. This builds the wheel, installs it into a fresh virtual environment (``--system-site-packages``
for the dependencies, so nothing is downloaded), and runs ``fi demo`` the way the guide says, with no terminal to
answer in, no key and no network.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.test_doctor_quick import _BLOCK_NETWORK

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.slow

# What the wheel is built from: the packages pyproject.toml ships, plus examples/ (so its exclusion is what is tested).
_SOURCES = ("core", "generation", "web", "agents", "templates", "examples")
_FILES = ("pyproject.toml", "README.md", "LICENSE", "MANIFEST.in", "launch.py")


def _bin(venv: Path, name: str) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin") / (name + (".exe" if os.name == "nt" else ""))


def _run(argv: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600,
                          **kw)  # type: ignore[call-overload]


def test_fi_demo_from_the_wheel_names_no_missing_file(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    for name in _FILES:
        shutil.copy2(ROOT / name, src / name)
    for name in _SOURCES:
        shutil.copytree(ROOT / name, src / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "FI_"))
           and k.upper() not in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")}  # a proxy would hide a connection
    env.update({"PIP_CACHE_DIR": str(tmp_path / "pip_cache"), "PIP_NO_INPUT": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
    dist = tmp_path / "dist"
    built = _run([sys.executable, "-m", "pip", "wheel", str(src), "--no-deps", "--no-build-isolation",
                  "--no-index", "-w", str(dist)], cwd=tmp_path, env=env)
    assert built.returncode == 0, built.stdout[-3000:] + built.stderr[-3000:]
    (wheel,) = list(dist.glob("frontier_insight-*.whl"))
    names = zipfile.ZipFile(wheel).namelist()
    assert not [n for n in names if n.startswith("examples/")]  # the audit's finding, reproduced
    assert "launch.py" in names and "core/demo.py" in names

    venv = tmp_path / "venv"
    made = _run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], env=env)
    assert made.returncode == 0, made.stderr[-3000:]
    py = _bin(venv, "python")
    installed = _run([str(py), "-m", "pip", "install", "--no-deps", "--no-index", str(wheel)], env=env)
    assert installed.returncode == 0, installed.stdout[-3000:] + installed.stderr[-3000:]

    # A first-time user: an empty folder, no key, no model provider on PATH, no network.
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_BLOCK_NETWORK, encoding="utf-8")
    home = tmp_path / "user"
    home.mkdir()
    keys = ("OPENAI_API_KEY", "GEMINI_API_KEY", "OLLAMA_HOST", "PYTHONPATH")
    run_env = {k: v for k, v in env.items() if k not in keys}
    path = [str(_bin(venv, "python").parent)]
    if os.name == "nt":
        path.append(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32"))
    else:
        path.append("/usr/bin:/bin")
    run_env.update({"PATH": os.pathsep.join(path), "PYTHONPATH": str(site), "FI_TEST_NET_LOG": str(tmp_path / "net.log"),
                    "FI_HOME": str(tmp_path / "fi_home"), "HF_HOME": str(tmp_path / "hf"),
                    "PYTHONIOENCODING": "utf-8"})

    where = _run([str(py), "-c", "import launch; print(launch.__file__)"], cwd=home, env=run_env)
    assert where.returncode == 0, where.stderr[-3000:]
    assert Path(where.stdout.strip()).resolve().is_relative_to(venv.resolve()), where.stdout  # the wheel, not a checkout

    demo = _run([str(_bin(venv, "fi")), "demo"], cwd=home, env=run_env, stdin=subprocess.DEVNULL)
    out = demo.stdout + demo.stderr
    # No model can be ready here (no key, no provider command on PATH, no network): it says what to fix, exit 1.
    assert demo.returncode == 1 and "Fix the model first" in out, out[-4000:]
    assert "examples/" not in out
    written = home / "fi-demo.yaml"
    assert written.is_file(), out[-4000:]
    assert "fi --config fi-demo.yaml" in out
    assert "Ready?" in out  # the model check ran (with no key and no network it says what stops it)

    loads = _run([str(py), "-c", "from core.config import Config; print(Config.from_yaml('fi-demo.yaml').provider.name)"],
                 cwd=home, env=run_env)
    assert loads.returncode == 0 and loads.stdout.strip(), loads.stderr[-3000:]

    # The quick doctor from the wheel: no connection at all (the demo above may ask the model's service).
    run_env["FI_TEST_NET_LOG"] = str(tmp_path / "doctor_net.log")
    doctor = _run([str(_bin(venv, "fi")), "--doctor"], cwd=home, env=run_env, stdin=subprocess.DEVNULL)
    assert doctor.returncode == 0, (doctor.stdout + doctor.stderr)[-4000:]
    assert not (tmp_path / "doctor_net.log").exists(), (tmp_path / "doctor_net.log").read_text(encoding="utf-8")
