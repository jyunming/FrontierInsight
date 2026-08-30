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
from Location..." and run ``@fi /start examples/integrator_bakeoff/config.yaml``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
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
    # `/new` is the interview-driven flow added so first-time users
    # don't have to write YAML by hand. It's not optional — its
    # absence regresses a load-bearing first-run UX.
    assert {"new", "start", "fleet"}.issubset(cmd_names), (
        f"expected /new, /start and /fleet commands; got {cmd_names}"
    )
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
    not (EXT_DIR / "out" / "extension.js").exists()
    and not _have_node_modules(),
    reason="run `npm install && npm run compile` in vscode-frontier-insight/ first",
)
def test_interview_generated_yaml_parses_with_python_config() -> None:
    """The interview's TypeScript YAML generator must produce a config
    that FI's Python ``Config.from_yaml`` accepts. Catches drift
    between the two sides — e.g., the TS picking field names that
    don't exist in the pydantic schema, or omitting required ones.

    Runs a tiny Node script that imports the compiled ``interview.js``,
    feeds in a representative answers object, writes the resulting
    YAML to a tmpfile, then loads it from Python.
    """
    import json
    import tempfile

    # Compile first (idempotent if already built). Bypass if `out/`
    # already has the file — gates earlier in this suite cover the
    # compile path proper.
    # We require the *core* (vscode-free) module — interview.ts itself
    # imports vscode and won't run outside the editor runtime.
    compiled = EXT_DIR / "out" / "interview-core.js"
    if not compiled.exists():
        sub = subprocess.run(
            [_resolve_npm(), "run", "compile"], cwd=str(EXT_DIR),
            capture_output=True, text=True, timeout=120,
        )
        assert sub.returncode == 0, sub.stderr

    # Drive the compiled module via a tiny inline Node script. Path
    # is built into the script via a JSON-escaped literal so we
    # don't have to worry about quoting / Windows backslashes.
    answers = {
        "topic": "Compare two numerical integrators on a damped harmonic oscillator.\nReport energy drift.",
        "title": "rk4-vs-euler",
        "output_kinds": ["paper_md", "slides"],
        "paper_format": "generic",
        "clarify_mode": "auto",
        "review_panel": ["methodologist", "statistician"],
        "knowledge_enabled": False,
        "no_simulation": False,
        "max_iterations": 2,
    }
    interview_path = str(compiled).replace("\\", "/")
    node_script = (
        f"const {{ answersToYaml }} = require({json.dumps(interview_path)});\n"
        f"const answers = {json.dumps(answers)};\n"
        "process.stdout.write(answersToYaml(answers));\n"
    )
    # Write the driver script to a tempfile rather than passing via
    # `node -e "..."` — the shell-quoting of a JSON-string-inside-a-
    # node-string-inside-a-cmd-arg is unreliable on Windows (loses the
    # contents silently).
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8",
    ) as f:
        f.write(node_script)
        driver_path = f.name
    try:
        proc = subprocess.run(
            [_resolve_node(), driver_path],
            capture_output=True, text=True, timeout=30,
        )
    finally:
        Path(driver_path).unlink(missing_ok=True)
    assert proc.returncode == 0, (
        f"node script failed: {proc.stderr}\n{proc.stdout}"
    )
    yaml_blob = proc.stdout
    assert "topic:" in yaml_blob
    assert "rk4-vs-euler" in yaml_blob
    # clarify is emitted under the unified `pauses:` namespace, double-quoted
    # to defeat YAML 1.1's Norway problem (unquoted `off`/`no` coerce to bool).
    assert "pauses:" in yaml_blob
    assert 'clarify: "auto"' in yaml_blob
    # Reviewer panel rendered as a YAML list of double-quoted strings.
    assert '- "methodologist"' in yaml_blob
    assert '- "statistician"' in yaml_blob

    # Drive it through Python's Config.from_yaml — this is the gate
    # that catches "TS made up a field name the schema doesn't know".
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    ) as f:
        f.write(yaml_blob)
        tmp_path_str = f.name
    try:
        from core.config import Config
        cfg = Config.from_yaml(Path(tmp_path_str))
        # The fields the interview definitely sets must round-trip.
        assert cfg.title == "rk4-vs-euler"
        assert "Compare two numerical integrators" in cfg.topic
        assert cfg.pauses.clarify == "auto"
        assert cfg.engine.review_panel == ["methodologist", "statistician"]
        assert cfg.output.kinds == ["paper_md", "slides"]
        assert cfg.knowledge.enabled is False
        assert cfg.provider.name == "vscode_extension"
        # COST DISCIPLINE — default-ON engine features (ideate_reflect,
        # cross_check, analyze_reroute) are explicitly turned OFF by
        # the interview. With them ON, a single quest can burn 25%+ of
        # a 1500-req/month Copilot allotment. The interview must
        # generate the cost-bounded values.
        assert cfg.engine.ideate_reflect is False, (
            "ideate_reflect on by default fires one extra LLM call per quest"
        )
        assert cfg.engine.cross_check_per_finding_k == 0, (
            "cross_check on by default fires ~2 calls × N findings × iterations "
            "— the single biggest Copilot-budget burner observed in real runs"
        )
        assert cfg.engine.enable_analyze_reroute is False, (
            "analyze re-route on by default lets each quest run multiple "
            "design iterations, multiplying every inner-loop call"
        )
        # When knowledge is disabled, source_routing must be 'manual' —
        # otherwise an LLM source-picker fires per asearch (3–5 wasted
        # calls per quest with no Axon corpus to route into).
        from core.config import KnowledgeConfig
        assert cfg.knowledge.source_routing == "manual", (
            "source_routing must be 'manual' when knowledge is disabled "
            "— the LLM source picker is pointless without an Axon corpus "
            "and burns 3–5 premium requests per quest"
        )
    finally:
        Path(tmp_path_str).unlink(missing_ok=True)


@pytest.mark.skipif(not _have_npm(), reason="node/npm not on PATH")
@pytest.mark.skipif(
    not (EXT_DIR / "out" / "interview-core.js").exists() and not _have_node_modules(),
    reason="run `npm install && npm run compile` in vscode-frontier-insight/ first",
)
@pytest.mark.parametrize("clarify_mode", ["off", "auto", "interactive"])
def test_interview_yaml_norway_problem_clarify_mode_off_parses_as_string(
    clarify_mode: str,
) -> None:
    """YAML 1.1's "Norway problem": unquoted barewords like ``off`` /
    ``no`` / ``false`` / ``y`` are coerced to Python booleans before
    the pydantic schema sees them. The interview's TS YAML generator
    must quote these so the Literal validator on
    ``EngineConfig.clarify_mode`` accepts them as strings.

    Regression: a user picked "Just run it" (which maps to
    ``clarify_mode: off``), the unquoted YAML parsed as
    ``clarify_mode: False`` (boolean), and pydantic raised
    ``literal_error``. Quoting every string-valued scalar in the
    generated YAML is the fix; this test pins it for all three
    clarify_mode values.
    """
    import json
    import tempfile

    compiled = EXT_DIR / "out" / "interview-core.js"
    if not compiled.exists():
        subprocess.run(
            [_resolve_npm(), "run", "compile"], cwd=str(EXT_DIR),
            capture_output=True, text=True, timeout=120, check=True,
        )

    answers = {
        "topic": "norway problem regression",
        # Include extra YAML-booleanish strings in the title to make
        # sure they survive too.
        "title": "no-yes-off-test",
        "output_kinds": ["paper_md"],
        "paper_format": "generic",
        "clarify_mode": clarify_mode,
        "review_panel": [],
        "knowledge_enabled": False,
        "no_simulation": False,
        "max_iterations": 2,
    }
    interview_path = str(compiled).replace("\\", "/")
    node_script = (
        f"const {{ answersToYaml }} = require({json.dumps(interview_path)});\n"
        f"const answers = {json.dumps(answers)};\n"
        "process.stdout.write(answersToYaml(answers));\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8",
    ) as f:
        f.write(node_script)
        driver_path = f.name
    try:
        proc = subprocess.run(
            [_resolve_node(), driver_path],
            capture_output=True, text=True, timeout=30,
        )
    finally:
        Path(driver_path).unlink(missing_ok=True)
    assert proc.returncode == 0, proc.stderr
    yaml_blob = proc.stdout

    # The Norway-problem fix: the clarify pause value must be quoted so YAML
    # parses it as a string, not a boolean. clarify is now under `pauses:`
    # with the canonical vocabulary (interactive → ask).
    expected = "ask" if clarify_mode == "interactive" else clarify_mode
    assert f'clarify: "{expected}"' in yaml_blob, (
        f"pauses.clarify is unquoted; YAML will coerce {expected!r} "
        f"to a boolean and pydantic will reject it.\nGenerated YAML:\n{yaml_blob}"
    )

    # Round-trip through Python's Config loader — this is the gate
    # that the original bug actually failed against.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    ) as f:
        f.write(yaml_blob)
        tmp_path_str = f.name
    try:
        from core.config import Config
        cfg = Config.from_yaml(Path(tmp_path_str))
        assert cfg.pauses.clarify == expected
        # Title with "no" / "yes" / "off" tokens survives as a string,
        # not a coerced boolean.
        assert cfg.title == "no-yes-off-test"
    finally:
        Path(tmp_path_str).unlink(missing_ok=True)


@pytest.mark.skipif(not _have_npm(), reason="node/npm not on PATH")
@pytest.mark.skipif(
    not (EXT_DIR / "out" / "interview-core.js").exists() and not _have_node_modules(),
    reason="run `npm install && npm run compile` in vscode-frontier-insight/ first",
)
@pytest.mark.parametrize(
    "paper_format",
    [
        # All nine venues PaperFormat in core/config.py declares. The
        # interview must produce a YAML that Pydantic accepts for each.
        "generic", "neurips", "iclr", "ieee_access", "nature_mi",
        "essay", "report", "policy_brief", "whitepaper",
    ],
)
def test_interview_paper_format_round_trips_through_pydantic(
    paper_format: str,
) -> None:
    """User-visible parity gap (task #39): the VSCode interview used
    to hardcode ``output.paper_format: "generic"`` regardless of the
    user's choice — the 8 other venues the clarify agent picks were
    unreachable from the UI. This test pins the fix: each format the
    interview offers must survive the TS → YAML → Pydantic round-trip.
    """
    import json
    import tempfile

    compiled = EXT_DIR / "out" / "interview-core.js"
    if not compiled.exists():
        subprocess.run(
            [_resolve_npm(), "run", "compile"], cwd=str(EXT_DIR),
            capture_output=True, text=True, timeout=120, check=True,
        )

    answers = {
        "topic": "paper-format parity regression",
        "title": "paper-format-roundtrip",
        "output_kinds": ["paper_md"],
        "paper_format": paper_format,
        "clarify_mode": "off",
        "review_panel": [],
        "knowledge_enabled": False,
        "no_simulation": False,
        "max_iterations": 2,
    }
    interview_path = str(compiled).replace("\\", "/")
    node_script = (
        f"const {{ answersToYaml }} = require({json.dumps(interview_path)});\n"
        f"const answers = {json.dumps(answers)};\n"
        "process.stdout.write(answersToYaml(answers));\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8",
    ) as f:
        f.write(node_script)
        driver_path = f.name
    try:
        proc = subprocess.run(
            [_resolve_node(), driver_path],
            capture_output=True, text=True, timeout=30,
        )
    finally:
        Path(driver_path).unlink(missing_ok=True)
    assert proc.returncode == 0, proc.stderr
    yaml_blob = proc.stdout
    assert f'paper_format: "{paper_format}"' in yaml_blob, (
        f"interview YAML must surface the chosen paper_format verbatim; "
        f"got:\n{yaml_blob}"
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    ) as f:
        f.write(yaml_blob)
        tmp_path_str = f.name
    try:
        from core.config import Config
        cfg = Config.from_yaml(Path(tmp_path_str))
        assert cfg.output.paper_format == paper_format, (
            f"Pydantic must accept paper_format={paper_format!r} from the "
            f"interview-generated YAML; got {cfg.output.paper_format!r}"
        )
    finally:
        Path(tmp_path_str).unlink(missing_ok=True)


@pytest.mark.skipif(not _have_npm(), reason="node/npm not on PATH")
@pytest.mark.skipif(
    not (EXT_DIR / "out" / "interview-core.js").exists() and not _have_node_modules(),
    reason="run `npm install && npm run compile` in vscode-frontier-insight/ first",
)
@pytest.mark.parametrize("no_simulation", [True, False])
def test_interview_no_simulation_round_trips_through_pydantic(
    no_simulation: bool,
) -> None:
    """Parity gap (task #39): the interview used to never emit
    ``engine.no_simulation`` — the gate that routes observational
    quests away from implement → execute. The clarify agent had to
    auto-detect it from the topic. Now the user can pre-set it.
    This test pins both YAML branches reach Pydantic intact."""
    import json
    import tempfile

    compiled = EXT_DIR / "out" / "interview-core.js"
    # Mirror the compile-on-miss path the sibling tests use — the
    # ``skipif`` above admits a fresh ``node_modules/`` checkout where
    # nothing has been compiled yet, and ``require(interview_path)``
    # would otherwise fail with MODULE_NOT_FOUND.
    if not compiled.exists():
        subprocess.run(
            [_resolve_npm(), "run", "compile"], cwd=str(EXT_DIR),
            capture_output=True, text=True, timeout=120, check=True,
        )
    answers = {
        "topic": "observational quest routing regression",
        "title": "no-sim-roundtrip",
        "output_kinds": ["paper_md"],
        "paper_format": "essay" if no_simulation else "generic",
        "clarify_mode": "off",
        "review_panel": [],
        "knowledge_enabled": False,
        "no_simulation": no_simulation,
        "max_iterations": 2,
    }
    interview_path = str(compiled).replace("\\", "/")
    node_script = (
        f"const {{ answersToYaml }} = require({json.dumps(interview_path)});\n"
        f"const answers = {json.dumps(answers)};\n"
        "process.stdout.write(answersToYaml(answers));\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8",
    ) as f:
        f.write(node_script)
        driver_path = f.name
    try:
        proc = subprocess.run(
            [_resolve_node(), driver_path],
            capture_output=True, text=True, timeout=30,
        )
    finally:
        Path(driver_path).unlink(missing_ok=True)
    assert proc.returncode == 0, proc.stderr
    yaml_blob = proc.stdout
    expected = "true" if no_simulation else "false"
    assert f"no_simulation: {expected}" in yaml_blob, (
        f"interview YAML must emit engine.no_simulation={expected}; "
        f"got:\n{yaml_blob}"
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8",
    ) as f:
        f.write(yaml_blob)
        tmp_path_str = f.name
    try:
        from core.config import Config
        cfg = Config.from_yaml(Path(tmp_path_str))
        assert cfg.engine.no_simulation is no_simulation
    finally:
        Path(tmp_path_str).unlink(missing_ok=True)


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
    # vsix should be reasonably small (<200 KB) — if it suddenly balloons,
    # something is leaking node_modules / sources / .git into the bundle.
    size_kb = vsix_path.stat().st_size / 1024
    assert size_kb < 200, (
        f"vsix is {size_kb:.0f} KB — check .vscodeignore is excluding "
        f"node_modules and src/*.ts"
    )
