"""Live probe for the `github_copilot_vscode` provider.

Companion to `scripts/probe_cli_providers.py`. This one is *separately*
runnable because it requires a real environment:

  - The VSCode "GitHub Copilot Chat" extension installed and signed in.
  - `copilot-api` available via `npx` (no install needed; `npx -y`).
  - Network connectivity for OAuth + Copilot API calls.

What it does:
  1. Spawn `copilot-api` with the vscode-auth flag on a free port.
  2. Wait for `GET /v1/models` to respond — that's our readiness probe.
  3. List the available models the user's VSCode Copilot has access to.
  4. Run one tiny chat completion with each of N selected models so we
     KNOW the per-call `"model"` field override (Phase O) actually
     hits different backends.

Cost: each model exercised burns ONE premium request. By default we
run 2 (a known-cheap and a known-expensive model). The premium-request
budget is shared with VSCode Copilot Chat in the same subscription —
running this probe twice a day costs ~4 of your monthly allotment.

Usage:
    python scripts/probe_copilot_vscode.py
    python scripts/probe_copilot_vscode.py --models gpt-5,claude-sonnet-4-6

When `--no-chat` is set we only probe readiness + list models (zero
premium requests consumed).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import ProviderConfig
from core.provider import (
    ProxySupervisor, LLMClient, resolve_endpoint_async,
)


async def _probe_models_endpoint(base_url: str, timeout_s: float = 8.0) -> list[str]:
    """Pull the list of models the proxy exposes. Empty list on failure."""
    import httpx
    url = base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as c:
            r = await c.get(url)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        print(f"  -- /models probe failed: {e}", flush=True)
        return []
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("id"), str):
            out.append(it["id"])
    return out


async def _one_chat(client: LLMClient, model: str, prompt: str) -> tuple[bool, str, float]:
    t0 = time.monotonic()
    try:
        text = await client.chat(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
        )
        return True, text.strip()[:120], time.monotonic() - t0
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:200], time.monotonic() - t0


async def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="probe_copilot_vscode",
        description="Live-verify the github_copilot_vscode provider.",
    )
    p.add_argument(
        "--models", type=str, default="",
        help="Comma-separated model IDs to exercise (default: probe 2 from /v1/models).",
    )
    p.add_argument(
        "--no-chat", action="store_true",
        help="Probe readiness + list models only — does NOT consume any premium requests.",
    )
    args = p.parse_args(argv)

    print("=" * 72)
    print("PROBE: github_copilot_vscode")
    print("=" * 72)

    supervisor = ProxySupervisor()
    try:
        ep = await resolve_endpoint_async(
            ProviderConfig(name="github_copilot_vscode"), supervisor,
        )
        print(f"  proxy ready at {ep.base_url}  (default model: {ep.model!r})")
    except Exception as e:
        print(f"  !! readiness FAILED: {type(e).__name__}: {e}")
        print()
        print("  Likely causes:")
        print("    - VSCode 'GitHub Copilot Chat' extension not signed in.")
        print("    - `copilot-api` not reachable via `npx -y copilot-api@latest start`.")
        print("    - No network to GitHub Copilot API.")
        await supervisor.shutdown()
        return 1

    available = await _probe_models_endpoint(ep.base_url)
    if available:
        print(f"  /v1/models lists {len(available)} model(s): {', '.join(available[:10])}"
              + (" ..." if len(available) > 10 else ""))
    else:
        print("  /v1/models returned no model list (proxy may not expose it)")

    if args.no_chat:
        print()
        print("  --no-chat set: skipping live chat completions (zero premium requests).")
        await supervisor.release("github_copilot_vscode")
        await supervisor.shutdown()
        return 0

    if args.models:
        models_to_try = [m.strip() for m in args.models.split(",") if m.strip()]
    elif available:
        # Pick two from the list to keep premium-request burn low.
        models_to_try = available[:2]
    else:
        # Fall back to two well-known VSCode Copilot model IDs.
        models_to_try = ["gpt-5", "claude-sonnet-4-6"]

    print()
    print(f"  exercising {len(models_to_try)} model(s) — each consumes 1 premium request:")
    print(f"    {models_to_try}")
    print()

    client = LLMClient(ep)
    try:
        for m in models_to_try:
            ok, text, dt = await _one_chat(
                client, m,
                prompt='Respond ONLY with the JSON: {"a": 1}. No prose.',
            )
            mark = "OK" if ok else "--"
            print(f"  {mark} model={m}  wall={dt:.1f}s  reply={text!r}")
    finally:
        await client.aclose()
        await supervisor.release("github_copilot_vscode")
        await supervisor.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
