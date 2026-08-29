"""Node-level tests for the extension's Axon discovery module.

``src/axon-endpoint.ts`` is the VSCode half of the same contract
``core/axon_endpoint.py`` implements for CLI + web: find a sidecar whose
port is no longer fixed. It deliberately imports no ``vscode`` API, which
means plain Node can exercise it — so the behaviour that used to be
untestable (does the extension actually find Axon?) is pinned here.

Skipped when ``node`` / ``npm`` are unavailable, matching
``test_vscode_extension_typescript.py``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
EXT_DIR = REPO_ROOT / "vscode-frontier-insight"


def _have_node() -> bool:
    return shutil.which("node") is not None and shutil.which("npm") is not None


def _have_node_modules() -> bool:
    return (EXT_DIR / "node_modules").is_dir()


pytestmark = [
    pytest.mark.skipif(not _have_node(), reason="node/npm not on PATH"),
    pytest.mark.skipif(
        not _have_node_modules(),
        reason="run `npm install` in vscode-frontier-insight/ first",
    ),
]


@pytest.fixture(scope="module")
def compiled() -> Path:
    """Compile the extension once and hand back the emitted module."""
    proc = subprocess.run(
        [shutil.which("npm") or "npm", "run", "compile"],
        cwd=str(EXT_DIR), capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"tsc failed:\n{proc.stdout}\n{proc.stderr}"
    out = EXT_DIR / "out" / "axon-endpoint.js"
    assert out.is_file(), "compile did not emit out/axon-endpoint.js"
    return out


def _run_node(script: str, tmp_path: Path) -> dict:
    """Run a snippet against the compiled module; return its JSON result."""
    runner = tmp_path / "runner.js"
    runner.write_text(script, encoding="utf-8")
    proc = subprocess.run(
        [shutil.which("node") or "node", str(runner)],
        cwd=str(EXT_DIR), capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"node failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _preamble(compiled: Path) -> str:
    return f"const A = require({json.dumps(str(compiled))});\n"


# ---------------------------------------------------------------------------
# Reading Axon's config.yaml without a YAML dependency
# ---------------------------------------------------------------------------


def test_config_parse_extracts_the_three_fields_we_need(
    compiled: Path, tmp_path: Path,
) -> None:
    """The extension has no runtime deps, so it reads Axon's flat,
    machine-written config directly. Windows paths matter here: an
    unquoted ``C:\\Users\\...`` is the normal value of ``store.base``."""
    script = _preamble(compiled) + r"""
const text = [
  'api:',
  '  key: some-secret',
  '  host: 127.0.0.1',
  '  port: 8420',
  '  allow_origins: []',
  'security:',
  '  mount_refresh_mode: switch',
  'store:',
  '  base: C:\\Users\\jyunm',
  'llm:',
  '  vllm_base_url: http://localhost:8000/v1',
].join('\n');
console.log(JSON.stringify(A.parseAxonConfig(text)));
"""
    got = _run_node(script, tmp_path)
    assert got == {"apiHost": "127.0.0.1", "apiPort": 8420, "storeBase": "C:\\Users\\jyunm"}


def test_config_parse_ignores_a_port_from_the_wrong_section(
    compiled: Path, tmp_path: Path,
) -> None:
    """``llm.vllm_base_url`` points at ``:8000`` in Axon's own defaults.
    Only ``api.port`` may set the API port."""
    script = _preamble(compiled) + r"""
const text = ['llm:', '  port: 8000', 'api:', '  port: 8420'].join('\n');
console.log(JSON.stringify(A.parseAxonConfig(text)));
"""
    assert _run_node(script, tmp_path).get("apiPort") == 8420


def test_config_parse_survives_garbage(compiled: Path, tmp_path: Path) -> None:
    script = _preamble(compiled) + r"""
console.log(JSON.stringify(A.parseAxonConfig('!!! not yaml @@@\n\n   ')));
"""
    assert _run_node(script, tmp_path) == {}


# ---------------------------------------------------------------------------
# Bind address vs. connect address
# ---------------------------------------------------------------------------


def test_wildcard_bind_addresses_become_loopback(
    compiled: Path, tmp_path: Path,
) -> None:
    """The lock file records the bound address, routinely ``0.0.0.0``,
    which you cannot connect to."""
    script = _preamble(compiled) + r"""
const out = {};
for (const h of ['0.0.0.0', '::', '[::]', '', '192.168.1.50']) out[h] = A.probeHost(h);
console.log(JSON.stringify(out));
"""
    got = _run_node(script, tmp_path)
    assert got["0.0.0.0"] == "127.0.0.1"
    assert got["::"] == "127.0.0.1"
    assert got["[::]"] == "127.0.0.1"
    assert got[""] == "127.0.0.1"
    assert got["192.168.1.50"] == "192.168.1.50"


# ---------------------------------------------------------------------------
# Candidate assembly against a synthetic store
# ---------------------------------------------------------------------------


def _store_with_lock(tmp_path: Path, port: int, host: str = "0.0.0.0") -> Path:
    """Build a ``<base>/AxonStore/<user>/.axon-api.lock`` tree."""
    user_dir = tmp_path / "AxonStore" / "someuser"
    user_dir.mkdir(parents=True)
    (user_dir / ".axon-api.lock").write_text(
        json.dumps({"host": host, "port": port, "pid": 4242}), encoding="utf-8",
    )
    return tmp_path


def test_lock_file_is_found_by_scanning_when_usernames_disagree(
    compiled: Path, tmp_path: Path,
) -> None:
    """Python's ``getpass.getuser()`` and Node's ``os.userInfo()`` can
    disagree (domain accounts, a ``USERNAME`` override). The store scan
    is what keeps discovery working when they do."""
    base = _store_with_lock(tmp_path / "store", 9137)
    script = _preamble(compiled) + f"""
const env = {{ AXON_STORE_BASE: {json.dumps(str(base))} }};
console.log(JSON.stringify(A.lockFileEndpoints({{}}, env)));
"""
    assert _run_node(script, tmp_path) == [{"host": "127.0.0.1", "port": 9137}]


def test_stale_axon_port_does_not_hide_a_live_sidecar(
    compiled: Path, tmp_path: Path,
) -> None:
    """The bug this module exists to fix: an environment still pointing
    at the old 8000 must cost one dead probe, not the whole discovery."""
    base = _store_with_lock(tmp_path / "store", 8420)
    script = _preamble(compiled) + f"""
const env = {{ AXON_STORE_BASE: {json.dumps(str(base))}, AXON_PORT: '8000',
               AXON_CONFIG_PATH: 'C:/nonexistent.yaml' }};
console.log(JSON.stringify(A.axonCandidates({{ env }})));
"""
    cands = _run_node(script, tmp_path)
    urls = [f"http://{c['host']}:{c['port']}" for c in cands]
    assert urls[0] == "http://127.0.0.1:8000", "the stale env var is tried first"
    assert "http://127.0.0.1:8420" in urls, "the live sidecar is still reachable"


def test_current_default_is_probed_before_the_legacy_one(
    compiled: Path, tmp_path: Path,
) -> None:
    script = _preamble(compiled) + f"""
const env = {{ AXON_STORE_BASE: {json.dumps(str(tmp_path / 'empty'))},
               AXON_CONFIG_PATH: 'C:/nonexistent.yaml' }};
console.log(JSON.stringify(A.axonCandidates({{ env }}).map(c => c.port)));
"""
    assert _run_node(script, tmp_path) == [8420, 8000]


def test_duplicate_endpoints_are_probed_once(
    compiled: Path, tmp_path: Path,
) -> None:
    """Lock file and config normally agree; probing twice would double
    the cost of the common case."""
    base = _store_with_lock(tmp_path / "store", 8420, host="127.0.0.1")
    script = _preamble(compiled) + f"""
const env = {{ AXON_STORE_BASE: {json.dumps(str(base))},
               AXON_CONFIG_PATH: 'C:/nonexistent.yaml' }};
console.log(JSON.stringify(A.axonCandidates({{ env }}).map(c => c.port)));
"""
    assert _run_node(script, tmp_path) == [8420, 8000]


def test_explicit_url_replaces_the_candidate_list(
    compiled: Path, tmp_path: Path,
) -> None:
    """``frontierInsight.axonUrl`` names one specific Axon. Two instances
    hold different corpora, so silently using a local one when the named
    one is down would change what a quest reads."""
    base = _store_with_lock(tmp_path / "store", 8420)
    script = _preamble(compiled) + f"""
const env = {{ AXON_STORE_BASE: {json.dumps(str(base))} }};
console.log(JSON.stringify(
  A.axonCandidates({{ env, overrideUrl: 'http://remote.box:9000' }})));
"""
    cands = _run_node(script, tmp_path)
    assert cands == [
        {"host": "remote.box", "port": 9000, "source": "frontierInsight.axonUrl setting"},
    ]


def test_preferred_endpoint_never_suggests_the_legacy_port(
    compiled: Path, tmp_path: Path,
) -> None:
    """Suggesting 8000 would send the user to start a sidecar on a port
    current Axon has moved off."""
    base = _store_with_lock(tmp_path / "store", 9137)
    script = _preamble(compiled) + f"""
const env = {{ AXON_STORE_BASE: {json.dumps(str(base))},
               AXON_CONFIG_PATH: 'C:/nonexistent.yaml' }};
console.log(JSON.stringify(A.preferredEndpoint({{ env }})));
"""
    got = _run_node(script, tmp_path)
    assert got["port"] == 8420, "the lock file describes a server that already exists"


# ---------------------------------------------------------------------------
# Telling Axon apart from whatever else is on the port
# ---------------------------------------------------------------------------


def test_another_service_answering_200_is_rejected(
    compiled: Path, tmp_path: Path,
) -> None:
    """Axon's own config defaults ``vllm_base_url`` to
    ``localhost:8000/v1`` — the very port FI used to probe."""
    script = _preamble(compiled) + r"""
const out = {
  axonLive: A.looksLikeAxon('{"status":"alive"}'),
  axonReady: A.looksLikeAxon('{"status":"ok","project":"default"}'),
  vllmModels: A.looksLikeAxon('{"object":"list","data":[{"id":"gemma4-26b"}]}'),
  empty: A.looksLikeAxon(''),
  notJson: A.looksLikeAxon('<html>hi</html>'),
};
console.log(JSON.stringify(out));
"""
    got = _run_node(script, tmp_path)
    assert got["axonLive"] is True
    assert got["axonReady"] is True
    assert got["vllmModels"] is False, "a model list is not an Axon health payload"
    # Inconclusive bodies stay accepted — the status code did the real
    # filtering, and refusing here would break an older Axon.
    assert got["empty"] is True
    assert got["notJson"] is True


# ---------------------------------------------------------------------------
# End-to-end against a real socket
# ---------------------------------------------------------------------------


def test_discovery_finds_a_real_server_and_reports_readiness(
    compiled: Path, tmp_path: Path,
) -> None:
    """Stand up a stub Axon on an ephemeral port, write a lock file
    pointing at it, and confirm discovery finds it end to end."""
    base = tmp_path / "store"
    (base / "AxonStore" / "someuser").mkdir(parents=True)
    lock = base / "AxonStore" / "someuser" / ".axon-api.lock"

    script = _preamble(compiled) + f"""
const http = require('http');
const fs = require('fs');
const server = http.createServer((req, res) => {{
  if (req.url === '/health/live') {{
    res.writeHead(200, {{'Content-Type': 'application/json'}});
    res.end(JSON.stringify({{status: 'alive'}}));
  }} else if (req.url === '/health/ready') {{
    res.writeHead(200, {{'Content-Type': 'application/json'}});
    res.end(JSON.stringify({{status: 'ok', project: 'default'}}));
  }} else {{
    res.writeHead(404); res.end();
  }}
}});
server.listen(0, '127.0.0.1', async () => {{
  const port = server.address().port;
  // Record the wildcard bind address the way Axon actually does.
  fs.writeFileSync({json.dumps(str(lock))},
    JSON.stringify({{host: '0.0.0.0', port, pid: process.pid}}));
  const env = {{ AXON_STORE_BASE: {json.dumps(str(base))},
                 AXON_CONFIG_PATH: 'C:/nonexistent.yaml' }};
  const found = await A.discoverAxon({{ env, timeoutMs: 2000 }});
  server.close();
  console.log(JSON.stringify({{
    live: found.live, ready: found.ready, source: found.source,
    matched: found.port === port,
  }}));
}});
"""
    got = _run_node(script, tmp_path)
    assert got == {
        "live": True, "ready": True, "source": "store lock file", "matched": True,
    }


def test_discovery_reports_what_it_tried_and_where_to_start(
    compiled: Path, tmp_path: Path,
) -> None:
    """"Not detected" has to be debuggable — a wrong port is the common
    cause, and the user can only fix what they can see.

    Driven through an explicit override at a closed port so the outcome
    doesn't depend on whether a real Axon happens to be running on the
    machine executing the tests."""
    script = _preamble(compiled) + f"""
const env = {{ AXON_STORE_BASE: {json.dumps(str(tmp_path / 'empty'))},
               AXON_CONFIG_PATH: 'C:/nonexistent.yaml' }};
A.discoverAxon({{ env, overrideUrl: 'http://127.0.0.1:9', timeoutMs: 500 }}).then(r => {{
  console.log(JSON.stringify({{
    live: r.live, suggested: r.port, attempts: r.attempts,
  }}));
}});
"""
    got = _run_node(script, tmp_path)
    assert got["live"] is False
    assert len(got["attempts"]) == 1
    # The message names both the endpoint and where it came from.
    assert "http://127.0.0.1:9" in got["attempts"][0]
    assert "frontierInsight.axonUrl setting" in got["attempts"][0]
    # With an override configured, the endpoint we point the user at is
    # the one they configured — not a default they never asked for.
    assert got["suggested"] == 9
