"""Integration verification for non-claude_cli providers via the
engine's ``LLMClient._chat_cli`` path.

Standalone CLI invocations only prove the binary itself accepts the
prompt. This script proves the *engine's* code path — ``resolve_
endpoint`` → ``LLMClient.chat`` → ``_chat_cli`` → ``_run_cli`` —
returns valid text for each of: ``codex_cli``, ``gemini_cli``,
``copilot_cli`` (gpt-5-mini). Runs them sequentially.

Each test reports PASS/FAIL with a one-line summary. Exit code is
0 only if all three pass; otherwise the count of failures. Costs
real LLM quota.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.config import ProviderConfig  # noqa: E402
from core.provider import LLMClient, resolve_endpoint  # noqa: E402


PROMPT_PATH = Path(r"C:\Users\jyunm\AppData\Local\Temp\body_prompt_utf8.txt")


async def run_one(label: str, provider_cfg: ProviderConfig,
                  prompt: str, *, expect_min_chars: int = 1000) -> dict:
    ep = resolve_endpoint(provider_cfg)
    client = LLMClient(
        ep,
        cli_timeout_s=1200.0,
        cli_inactivity_timeout_s=300.0,
        node_cli_timeout_s={"implement": 1800.0},
        node_model_fallbacks={},  # no escalation — test the primary's behaviour
    )
    started = time.monotonic()
    try:
        text = await client.chat(
            [{"role": "user", "content": prompt}],
            node="implement",
        )
        elapsed = time.monotonic() - started
        ok = len(text) >= expect_min_chars
        return {
            "label": label,
            "ok": ok,
            "elapsed_s": elapsed,
            "chars": len(text),
            "head": text[:160].replace("\n", " "),
            "err": None,
        }
    except Exception as e:
        elapsed = time.monotonic() - started
        return {
            "label": label,
            "ok": False,
            "elapsed_s": elapsed,
            "chars": 0,
            "head": "",
            "err": f"{type(e).__name__}: {e}",
        }
    finally:
        await client.aclose()


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    if not PROMPT_PATH.is_file():
        print(f"ERROR: missing {PROMPT_PATH}")
        return 1
    prompt = PROMPT_PATH.read_text(encoding="utf-8", errors="replace")
    print(f"loaded prompt: {len(prompt)} chars\n")

    configs = [
        ("codex_cli (default GPT-5)",
         ProviderConfig(name="codex_cli")),
        ("gemini_cli (default gemini-2.5-pro)",
         ProviderConfig(name="gemini_cli")),
        ("copilot_cli + gpt-5-mini",
         ProviderConfig(name="copilot_cli", model="gpt-5-mini")),
    ]

    results = []
    for label, cfg in configs:
        print(f"=== running {label} ===")
        res = await run_one(label, cfg, prompt)
        results.append(res)
        status = "PASS" if res["ok"] else "FAIL"
        print(f"  {status} — elapsed {res['elapsed_s']:.0f}s, chars={res['chars']}")
        if res["err"]:
            print(f"  err: {res['err'][:200]}")
        elif res["head"]:
            print(f"  head: {res['head']}")
        print()

    print("--- summary ---")
    fails = sum(1 for r in results if not r["ok"])
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  {status:4} {r['label']:40} elapsed={r['elapsed_s']:5.0f}s chars={r['chars']}")
    print(f"\n{len(results) - fails}/{len(results)} passed")
    return fails


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
