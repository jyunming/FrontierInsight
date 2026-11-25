"""TypeScript compile-check for the FI VSCode extension.

This test was added in response to a real bug: the TypeScript source
was committed without ever running ``npm run compile``, and two type
errors went unnoticed until a developer tried to install the extension
locally. The Python-side bridge tests cover the wire protocol but tell
you nothing about whether the ``.ts`` files even parse.

Skipped when ``node`` / ``npm`` are not on PATH (CI without Node).
Skipped when ``vscode-frontier-insight/node_modules`` is not present
(pre-`npm install`). Otherwise runs ``npm run compile`` and asserts
exit code 0 + that ``out/extension.js`` and ``out/bridge.js`` exist.

This is a structural / typing test only — it does NOT exercise the
actual ``vscode.lm`` runtime (impossible outside VSCode). For that,
manually install the extension via "Developer: Install Extension
from Location..." and run ``@fi /start fi-demo.yaml`` (written by ``fi demo``).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
EXT_DIR = REPO_ROOT / "vscode-frontier-insight"


def _resolve_npm() -> str:
    """Find the absolute path of `npm`. On Windows, `npm` is a `.cmd`
    shim — `shutil.which` returns the full path including the `.cmd`
    extension, which `subprocess.run` can invoke directly without
    `shell=True`. That keeps argv-as-list semantics correct on POSIX
    (where `shell=True` + list silently drops everything after argv[0])."""
    return shutil.which("npm") or "npm"


def _resolve_node() -> str:
    return shutil.which("node") or "node"


def _have_npm() -> bool:
    return shutil.which("npm") is not None and shutil.which("node") is not None


def _have_node_modules() -> bool:
    return (EXT_DIR / "node_modules").is_dir()


@pytest.mark.skipif(not _have_npm(), reason="node/npm not on PATH")
@pytest.mark.skipif(
    not _have_node_modules(),
    reason="run `npm install` in vscode-frontier-insight/ first",
)
def test_vscode_extension_typescript_compiles() -> None:
    """Runs the extension's `npm run compile` (= `tsc -p ./`) and
    verifies it exits 0 with the expected JS output files present.

    Catches the exact class of regression that bit us once: a `.ts`
    edit that passes flake-style inspection but fails `tsc --strict`.
    """
    proc = subprocess.run(
        [_resolve_npm(), "run", "compile"],
        cwd=str(EXT_DIR),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"`npm run compile` failed (rc={proc.returncode}).\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    # Strict-mode tsc emits these two when bridge.ts + extension.ts
    # compile successfully. Their absence means an emit suppression
    # config landed accidentally (e.g., `"noEmit": true`).
    assert (EXT_DIR / "out" / "extension.js").is_file(), (
        "extension.js was not emitted; check tsconfig.json `noEmit` / `outDir`"
    )
    assert (EXT_DIR / "out" / "bridge.js").is_file(), (
        "bridge.js was not emitted; check tsconfig.json `noEmit` / `outDir`"
    )


def test_vscode_extension_manifest_exposes_chat_participant() -> None:
    """Sanity-check the extension manifest. Wrong fields here are the
    second-most-common silent breakage (after tsc errors): the
    extension installs fine but `@fi` simply doesn't appear in the
    Chat picker because the activationEvents / chatParticipants
    contribution is malformed."""
    import json
    pkg = json.loads((EXT_DIR / "package.json").read_text(encoding="utf-8"))
    contribs = pkg.get("contributes", {})
    participants = contribs.get("chatParticipants", [])
    assert participants, "package.json missing chatParticipants contribution"
    fi = next((p for p in participants if p.get("name") == "fi"), None)
    assert fi is not None, "chat participant `fi` not found in package.json"
    assert fi["id"] == "frontier-insight.fi"
    cmd_names = {c["name"] for c in fi.get("commands", []) if isinstance(c, dict)}
    assert {"start", "fleet", "resume"}.issubset(cmd_names), (
        f"expected /start, /fleet and /resume commands; got {cmd_names}"
    )
    # A study starts from a commission (`fi new` in a terminal): the chat's setup interview and its /update are gone.
    assert not {"new", "update"} & cmd_names, cmd_names
    # The activationEvents string must match the participant id, or
    # VSCode loads the extension but never wakes it up on `@fi`.
    activation = pkg.get("activationEvents", [])
    assert "onChatParticipant:frontier-insight.fi" in activation


def test_vscode_extension_manifest_exposes_axon_settings() -> None:
    """Both Axon knobs must be declared, or they're invisible in the
    Settings UI and the "Don't show again" button writes to a setting
    VSCode doesn't know about — which silently fails to persist."""
    import json
    pkg = json.loads((EXT_DIR / "package.json").read_text(encoding="utf-8"))
    props = pkg["contributes"]["configuration"]["properties"]

    assert "frontierInsight.axonUrl" in props
    assert props["frontierInsight.axonUrl"]["default"] == "", (
        "an empty default is what selects automatic discovery"
    )

    wait = props.get("frontierInsight.axonStartupWaitSec")
    assert wait is not None, "activation wait is not configurable"
    assert wait["type"] == "number"
    # Axon loads an embedding model and opens vector indexes before it
    # serves, and users start it well after opening the editor. A short
    # default is what produced "not detected" notices for a sidecar that
    # was merely still booting.
    assert wait["default"] >= 300, f"wait default too short: {wait['default']}s"
    assert wait["minimum"] == 0, "0 must be accepted — it's the opt-out"


def test_vscode_extension_manifest_has_package_script() -> None:
    """The `npm run package` script must exist + invoke vsce. Without
    it, users hit "where's my .vsix?" after `npm run compile` (which
    only produces .js, not a packaged extension). Background:
    `npm run compile` ran `tsc -p ./` and emitted `out/*.js`, leaving
    a developer wondering where the installable file was. This test
    pins the explicit packaging step in the manifest."""
    import json
    pkg = json.loads((EXT_DIR / "package.json").read_text(encoding="utf-8"))
    scripts = pkg.get("scripts", {})
    assert "package" in scripts, (
        "package.json `scripts.package` is missing — users won't be able "
        "to run `npm run package` to produce the .vsix"
    )
    pkg_script = scripts["package"]
    assert "vsce" in pkg_script, (
        f"`scripts.package` should call vsce; got {pkg_script!r}"
    )
    # vsce needs these top-level fields or it warns / errors.
    assert pkg.get("license"), "package.json missing `license` (vsce requires it)"
    assert pkg.get("repository"), "package.json missing `repository` (vsce wants it for marketplace)"
    # @vscode/vsce must be in devDependencies so `npm install` brings it in.
    devs = pkg.get("devDependencies", {})
    assert "@vscode/vsce" in devs, (
        "devDependencies must include @vscode/vsce so `npm run package` works "
        "after a fresh `npm install`"
    )


@pytest.mark.skipif(not _have_npm(), reason="node/npm not on PATH")
@pytest.mark.skipif(
    not _have_node_modules(),
    reason="run `npm install` in vscode-frontier-insight/ first",
)
def test_vscode_extension_package_produces_vsix() -> None:
    """Runs `npm run package` and verifies a `.vsix` is produced.
    This is the gate against the "where's my .vsix?" failure mode —
    `npm run compile` alone produces only `out/*.js`. End users want
    a single installable file (vsce wraps everything into a .vsix
    that VSCode's "Install from VSIX..." accepts directly)."""
    vsix_path = EXT_DIR / "vscode-frontier-insight.vsix"
    if vsix_path.exists():
        # Pre-existing artifact from a manual run; remove so the test
        # asserts on what THIS run actually produces.
        vsix_path.unlink()
    proc = subprocess.run(
        [_resolve_npm(), "run", "package"],
        cwd=str(EXT_DIR),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"`npm run package` failed (rc={proc.returncode}).\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    assert vsix_path.is_file(), (
        f"expected {vsix_path.name} to exist after `npm run package`; "
        f"check the `--out` flag in package.json scripts.package"
    )
    # vsix should be reasonably small (<300 KB; ~200 KB of compiled code, README and icons today) — if it suddenly
    # balloons, something is leaking node_modules / sources / .git into the bundle (that adds megabytes).
    size_kb = vsix_path.stat().st_size / 1024
    assert size_kb < 300, (
        f"vsix is {size_kb:.0f} KB — check .vscodeignore is excluding "
        f"node_modules and src/*.ts"
    )
