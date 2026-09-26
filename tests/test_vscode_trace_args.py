"""``--node <name>`` / ``--detail <level>`` beside ``@fi /trace <quest_id>`` in the VSCode chat panel.

Runs the compiled parser (``vscode-frontier-insight/out/trace-args.js``) with Node, and checks that
`extension.ts` wires `/trace` through to `runTrace`, which shells out to `launch.py --trace` — the
same trace `core/audit_log.py` renders for the CLI and the web quest page.

Skips when Node or a current compiled extension is missing: the Python CI job does not build the
extension (its own job compiles it).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

EXT = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"
ARGS_JS = EXT / "out" / "trace-args.js"
ARGS_TS = EXT / "src" / "trace-args.ts"
TRACE_TS = EXT / "src" / "trace.ts"


def _run(expression: str):
    node = shutil.which("node")
    if node is None or not ARGS_JS.is_file():
        pytest.skip("needs node and a compiled vscode-frontier-insight/out/trace-args.js")
    if ARGS_JS.stat().st_mtime < ARGS_TS.stat().st_mtime:
        pytest.skip("out/trace-args.js is older than the source; run npm run compile")
    script = f"const m = require(process.argv[1]); process.stdout.write(JSON.stringify({expression}));"
    done = subprocess.run(
        [node, "-e", script, str(ARGS_JS)], capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.parametrize(
    "text,quest_id,node,detail",
    [
        ("1790003131-my-quest", "1790003131-my-quest", "", "checks"),
        ("", "", "", "checks"),
        ("q --node design", "q", "design", "checks"),
        ("--node design q", "q", "design", "checks"),
        ("q --detail summary", "q", "", "summary"),
        ("q --node design --detail debug", "q", "design", "debug"),
        ('q --node "the design step"', "q", "the design step", "checks"),
        ("q --node 'the design step'", "q", "the design step", "checks"),
        ("q --node=design", "q", "design", "checks"),
    ],
)
def test_the_quest_id_node_and_detail_are_taken_out_of_the_text(
    text: str, quest_id: str, node: str, detail: str,
) -> None:
    got = _run(f"m.parseTraceArgs({json.dumps(text)})")
    assert got["questId"] == quest_id and got["node"] == node and got["detail"] == detail
    assert "error" not in got


@pytest.mark.parametrize("text", ["q --node", "q --node  ", "q --node="])
def test_a_node_without_a_name_is_reported(text: str) -> None:
    got = _run(f"m.parseTraceArgs({json.dumps(text)})")
    assert "needs a node name" in got["error"]


@pytest.mark.parametrize(
    "text,which",
    [
        ("q --node --detail checks", "node"),  # a following flag must not be swallowed as this one's value
        ("q --detail --node design", "detail"),
    ],
)
def test_a_flag_immediately_before_another_flag_is_reported_missing_not_swallowed(text: str, which: str) -> None:
    got = _run(f"m.parseTraceArgs({json.dumps(text)})")
    assert "error" in got and f"--{which}" in got["error"]
    # the other flag still parses normally out of what's left
    if which == "node":
        assert got["detail"] == "checks"
    else:
        assert got["node"] == "design"


@pytest.mark.parametrize("text", ["q --detail", "q --detail="])
def test_a_detail_without_a_value_is_reported(text: str) -> None:
    got = _run(f"m.parseTraceArgs({json.dumps(text)})")
    assert "needs one of" in got["error"]


def test_an_unknown_detail_level_is_reported() -> None:
    got = _run(f'm.parseTraceArgs({json.dumps("q --detail everything")})')
    assert "must be one of" in got["error"] and got["detail"] == "checks"


def test_the_details_list_matches_the_engines() -> None:
    assert _run("m.DETAILS") == ["summary", "checks", "debug"]


def test_trace_ts_decodes_the_stream_as_utf8_not_per_chunk() -> None:
    body = TRACE_TS.read_text(encoding="utf-8")
    # setEncoding("utf8") on the stream buffers a split multibyte character across chunks;
    # d.toString() per chunk (the earlier version) can mangle one that lands on a chunk boundary.
    assert 'setEncoding("utf8")' in body
    assert 'stdout += d.toString()' not in body and 'stderr += d.toString()' not in body


def test_a_nonzero_exit_shows_stderr_too_not_only_whichever_stream_was_non_empty() -> None:
    body = TRACE_TS.read_text(encoding="utf-8")
    assert "res.code === 0 ? res.stdout" in body and "res.stderr" in body


def test_trace_ts_shells_out_to_launch_py_trace_and_never_reparses_it() -> None:
    body = TRACE_TS.read_text(encoding="utf-8")
    assert '"--trace"' in body and '"--trace-node"' in body and '"--trace-detail"' in body
    assert "parseTraceArgs(" in body
    # The whole point: render launch.py's own stdout text, don't re-implement audit_log.py's
    # event selection/rendering here — no import of, or call into, an audit-log module.
    assert "import" not in "\n".join(line for line in body.splitlines() if "audit" in line.lower())


def test_the_chain_broken_warning_only_fires_on_an_actual_chain_break() -> None:
    body = TRACE_TS.read_text(encoding="utf-8")
    # A missing quest or missing trace file also exits 1 (launch.py's own text explains which);
    # the extra "hash chain" line must not fire for those too.
    assert 'text.includes("CHAIN BROKEN")' in body


def test_the_trace_command_is_wired_into_the_chat_dispatcher() -> None:
    extension = (EXT / "src" / "extension.ts").read_text(encoding="utf-8")
    assert 'cmd === "trace"' in extension
    assert "runTrace(prompt, stream, token)" in extension
    assert 'import { runFollow, runTrace, runWhy } from "./trace";' in extension
    # /why and /follow go through the same module, and each reaches launch.py's own flags.
    assert 'cmd === "why"' in extension and "runWhy(prompt, stream, token)" in extension
    assert 'cmd === "follow"' in extension and "runFollow(prompt, stream, token)" in extension
    trace = (EXT / "src" / "trace.ts").read_text(encoding="utf-8")
    assert '"--why"' in trace and '"--follow"' in trace


def test_the_trace_command_is_declared_in_package_json() -> None:
    package = json.loads((EXT / "package.json").read_text(encoding="utf-8"))
    commands = package["contributes"]["chatParticipants"][0]["commands"]
    names = [c["name"] for c in commands]
    assert "trace" in names and "why" in names and "follow" in names
