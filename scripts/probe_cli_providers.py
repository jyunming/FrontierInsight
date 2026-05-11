"""Live smoke-test for the three CLI-exec providers.

Invokes each provider through `LLMClient.chat()` with a tiny canonical
prompt (`"Reply with the JSON object {\"a\": 1} and nothing else."`) and
prints (provider, wall_time, success, content_excerpt) for each.

Skips a provider when its binary is not on PATH. Does NOT require any
API key — every supported CLI uses its own local OAuth keychain.

Usage:
    python scripts/probe_cli_providers.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import ProviderConfig
from core.provider import _CLI_SPECS, LLMClient, resolve_endpoint


PROMPT = 'Reply with the JSON object {"a": 1} and nothing else.'


async def probe(name: str) -> dict[str, object]:
    binary = _CLI_SPECS[name].argv[0]
    if shutil.which(binary) is None:
        return {"provider": name, "ok": False, "reason": f"{binary!r} not on PATH"}
    ep = resolve_endpoint(ProviderConfig(name=name))
    client = LLMClient(ep, timeout_s=180.0)
    t0 = time.monotonic()
    try:
        content = await client.chat([{"role": "user", "content": PROMPT}])
        elapsed = time.monotonic() - t0
        return {
            "provider": name,
            "ok": True,
            "elapsed_s": round(elapsed, 1),
            "content_excerpt": content.strip()[:200],
        }
    except Exception as e:
        return {
            "provider": name,
            "ok": False,
            "elapsed_s": round(time.monotonic() - t0, 1),
            "reason": f"{type(e).__name__}: {e}"[:300],
        }
    finally:
        await client.aclose()


async def main() -> int:
    results = []
    for name in ("claude_cli", "codex_cli", "copilot_cli"):
        print(f"[probe] {name}...", flush=True)
        r = await probe(name)
        results.append(r)
        if r.get("ok"):
            print(f"  OK in {r['elapsed_s']}s: {r['content_excerpt']!r}")
        else:
            print(f"  SKIP/FAIL: {r.get('reason', '?')}")
    failed = sum(1 for r in results if not r.get("ok"))
    print(f"\n[probe] {len(results) - failed}/{len(results)} providers responded")
    return 1 if failed and any("not on PATH" not in r.get("reason", "") for r in results if not r.get("ok")) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
