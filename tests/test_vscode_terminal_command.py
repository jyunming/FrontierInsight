"""The command lines `@fi /update` and `@fi /generate` type into a terminal.

Those two commands do not spawn Python as a child of the extension, so
they cannot inherit the per-command TCP bridge `/start` uses. They reach
`vscode.lm.*` only if the line they send carries the session-long
PersistentBridge address. Before this was wired, every `@fi /update` and
every `@fi /generate slides|poster|speech` died at endpoint resolution
with "vscode_extension provider requires either extra['bridge_socket']
... or extra['bridge_port']", because the `@fi /new` interview pins
`provider.name: vscode_extension` into every YAML it writes.

The line also has to *parse*, and the three shells disagree about how one
may begin. Measured directly, not assumed:

    PowerShell   `"python" --version`    -> ParserError
    PowerShell   `& "python" --version`  -> runs
    cmd.exe      `"python" --version`    -> runs (spaces in the path too)
    cmd.exe      `& "python" --version`  -> `& was unexpected at this time.`
    bash         `'python' --version`    -> runs

So a single spelling cannot work everywhere: PowerShell needs the call
operator, cmd.exe rejects a leading one. The builder picks per shell.

Driving the compiled JS under plain node tests the real artifact rather
than grepping the TypeScript for a substring. The last test covers the
only glue node cannot see — that extension.ts actually calls these
builders instead of interpolating its own line.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

EXT = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"
TC_JS = EXT / "out" / "terminal-command.js"
TC_TS = EXT / "src" / "terminal-command.ts"
EXTENSION_TS = EXT / "src" / "extension.ts"

# A Windows interpreter path with a space in it — the case the README
# points users at ("use a venv path") and the one the old unquoted
# `${pythonPath} launch.py ...` split into two tokens.
WIN_PY = r"C:\Program Files\Python311\python.exe"
WIN_PIPE = r"\\.\pipe\fi-bridge-me"
WIN_YAML = r"C:\Users\me\My Outputs\q-1\config.yaml"
POSIX_PY = "/usr/bin/python3"
POSIX_SOCK = "/run/user/1000/fi-bridge.sock"

NODE_SCRIPT = """
const tc = require(process.argv[1]);
const spec = JSON.parse(process.argv[2]);
const out = {commands: {}, detect: {}};
for (const [name, c] of Object.entries(spec.commands)) {
    out.commands[name] = tc[c.fn](c.opts);
}
for (const [name, d] of Object.entries(spec.detect)) {
    out.detect[name] = tc.detectShell(d[0], d[1]);
}
process.stdout.write(JSON.stringify(out));
"""

SPEC = {
    "commands": {
        "ps_update": {
            "fn": "updateTerminalCommand",
            "opts": {
                "pythonPath": WIN_PY, "questId": "q-1",
                "bridgeSocket": WIN_PIPE, "shell": "powershell",
            },
        },
        "cmd_update": {
            "fn": "updateTerminalCommand",
            "opts": {
                "pythonPath": WIN_PY, "questId": "q-1",
                "bridgeSocket": WIN_PIPE, "shell": "cmd",
            },
        },
        "posix_update": {
            "fn": "updateTerminalCommand",
            "opts": {
                "pythonPath": POSIX_PY, "questId": "q-1",
                "bridgeSocket": POSIX_SOCK, "shell": "posix",
            },
        },
        "ps_generate": {
            "fn": "generateTerminalCommand",
            "opts": {
                "pythonPath": WIN_PY, "yamlPath": WIN_YAML, "questId": "q-1",
                "kind": "slides", "bridgeSocket": WIN_PIPE,
                "shell": "powershell",
            },
        },
        "posix_generate": {
            "fn": "generateTerminalCommand",
            "opts": {
                "pythonPath": POSIX_PY, "yamlPath": "/out/q-1/config.yaml",
                "questId": "q-1", "kind": "poster",
                "bridgeSocket": POSIX_SOCK, "shell": "posix",
            },
        },
        # /ingest and /install-tectonic make no model call, so they get
        # no bridge address — but they still need the shell-correct head.
        "no_socket": {
            "fn": "buildLaunchCommand",
            "opts": {
                "pythonPath": "python",
                "args": ["--ingest", "/a b/c.pdf"],
                "shell": "posix",
            },
        },
        "ps_no_socket": {
            "fn": "buildLaunchCommand",
            "opts": {
                "pythonPath": WIN_PY, "args": ["--install-tectonic"],
                "shell": "powershell",
            },
        },
    },
    "detect": {
        "pwsh": [r"C:\Program Files\PowerShell\7\pwsh.exe", "win32"],
        "powershell": [
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            "win32",
        ],
        "cmd": [r"C:\Windows\System32\cmd.exe", "win32"],
        "gitbash": [r"C:\Program Files\Git\bin\bash.exe", "win32"],
        "bash": ["/bin/bash", "linux"],
        "zsh": ["/bin/zsh", "darwin"],
        "empty_win": ["", "win32"],
        "empty_posix": ["", "linux"],
    },
}


def _build() -> dict:
    node = shutil.which("node")
    if node is None or not TC_JS.is_file():
        pytest.skip(
            "needs node and a compiled "
            "vscode-frontier-insight/out/terminal-command.js"
        )
    if TC_JS.stat().st_mtime < TC_TS.stat().st_mtime:
        pytest.skip("out/terminal-command.js is stale; run npm run compile")
    done = subprocess.run(
        [node, "-e", NODE_SCRIPT, str(TC_JS), json.dumps(SPEC)],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_update_and_generate_carry_the_bridge_address() -> None:
    """The bug this branch exists for: without `--vscode-bridge-socket`
    on the line, the spawned Python cannot reach `vscode.lm` at all."""
    cmds = _build()["commands"]
    for name in (
        "ps_update", "cmd_update", "posix_update",
        "ps_generate", "posix_generate",
    ):
        assert "--vscode-bridge-socket" in cmds[name], (
            f"{name} carries no bridge address: {cmds[name]!r}"
        )
    assert WIN_PIPE in cmds["ps_update"]
    assert WIN_PIPE in cmds["ps_generate"]
    assert POSIX_SOCK in cmds["posix_update"]


def test_exact_lines_for_windows_update_and_generate() -> None:
    """Pin the whole string: a path with spaces stays ONE token, the
    quest id and flags stay bare and readable, and PowerShell gets the
    call operator that cmd.exe must not have."""
    cmds = _build()["commands"]
    assert cmds["ps_update"] == (
        r'& "C:\Program Files\Python311\python.exe" launch.py '
        r'--update q-1 --vscode-bridge-socket "\\.\pipe\fi-bridge-me"'
    )
    assert cmds["cmd_update"] == (
        r'"C:\Program Files\Python311\python.exe" launch.py '
        r'--update q-1 --vscode-bridge-socket "\\.\pipe\fi-bridge-me"'
    )
    assert cmds["ps_generate"] == (
        r'& "C:\Program Files\Python311\python.exe" launch.py '
        r'--config "C:\Users\me\My Outputs\q-1\config.yaml" '
        r'--resume q-1 --emit slides '
        r'--vscode-bridge-socket "\\.\pipe\fi-bridge-me"'
    )


def test_spaced_paths_are_quoted_as_single_tokens() -> None:
    """`C:\\Program Files\\...` is the path the extension's own comment
    names as typical, and the old `/update` / `/generate` lines
    interpolated it raw."""
    cmds = _build()["commands"]
    assert '"C:\\Program Files\\Python311\\python.exe"' in cmds["ps_update"]
    assert '"C:\\Program Files\\Python311\\python.exe"' in cmds["cmd_update"]
    # The yaml path carries a spaced directory too ("My Outputs").
    assert '"C:\\Users\\me\\My Outputs\\q-1\\config.yaml"' in cmds["ps_generate"]
    # POSIX quoting is single-quote, and a spaced arg survives it.
    assert "'/a b/c.pdf'" in cmds["no_socket"]


def test_powershell_gets_the_call_operator_and_cmd_does_not() -> None:
    """Measured: PowerShell raises ParserError on a line starting with a
    quoted string; cmd.exe fails with `& was unexpected at this time.`
    on a leading `&`. So this is not a style preference."""
    cmds = _build()["commands"]
    assert cmds["ps_update"].startswith('& "')
    assert cmds["ps_generate"].startswith('& "')
    assert cmds["ps_no_socket"].startswith('& "')
    assert cmds["cmd_update"].startswith('"')
    assert not cmds["cmd_update"].startswith("&")
    assert cmds["posix_update"].startswith("'")
    assert not cmds["posix_update"].startswith("&")


def test_commands_without_a_model_call_get_no_bridge_address() -> None:
    """`--ingest` embeds through Axon and `--install-tectonic` downloads a
    binary; neither resolves an LLM endpoint, so neither needs the
    bridge. Passing one anyway would be dead weight on the line."""
    cmds = _build()["commands"]
    assert "--vscode-bridge-socket" not in cmds["no_socket"]
    assert "--vscode-bridge-socket" not in cmds["ps_no_socket"]


def test_detect_shell_classifies_the_profiles_vscode_reports() -> None:
    """`vscode.env.shell` is an absolute path to the shell binary, or ""
    where the host has no shell. Windows defaults to PowerShell, which is
    the one shell that errors on an unprefixed quoted head, so the empty
    fallback must guess it rather than cmd."""
    detect = _build()["detect"]
    assert detect["pwsh"] == "powershell"
    assert detect["powershell"] == "powershell"
    assert detect["cmd"] == "cmd"
    assert detect["gitbash"] == "posix"
    assert detect["bash"] == "posix"
    assert detect["zsh"] == "posix"
    assert detect["empty_win"] == "powershell"
    assert detect["empty_posix"] == "posix"


def test_extension_uses_the_builders_instead_of_its_own_string() -> None:
    """The only glue node cannot check: that the two call sites actually
    send what the builders produce. Guards the regression directly — the
    old code interpolated `${pythonPath} launch.py ...` by hand, with no
    bridge address and no quoting."""
    src = EXTENSION_TS.read_text(encoding="utf-8")
    assert "${pythonPath} launch.py" not in src, (
        "extension.ts still builds a launch.py line by hand; that form "
        "carries no bridge address and does not quote the interpreter."
    )
    assert "updateTerminalCommand(" in src
    assert "generateTerminalCommand(" in src
    assert "persistentBridgePath()" in src
    # `paths.map(shellQuote)` passed the array INDEX as the second
    # argument once shellQuote took a shell; the call site now hands the
    # raw paths to the builder, which quotes them itself.
    assert "paths.map(shellQuote)" not in src
