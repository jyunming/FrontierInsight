"""Integration verification for the model-escalation path in
``LLMClient._chat_cli``.

Directly exercises the production code path against the real
``claude`` binary on PATH:

  1. Build a claude_cli endpoint with provider.model="claude-sonnet-4-6"
  2. Construct LLMClient with the default node_model_fallbacks
     ({implement: claude-opus-4-7, write: claude-opus-4-7})
  3. Call .chat() with the OPC body prompt and node="implement"
  4. Observe: attempt 1 should fail (Sonnet runaway / rate-limit
     message guard / something), tenacity should catch, before_sleep
     callback should log the failure + next-attempt model, attempt 2
     should run with --model claude-opus-4-7 in argv and produce text.

This is not a unit test — it shells out to the real CLI and consumes
real LLM quota. Run by hand when verifying end-to-end behaviour of
the escalation path; not added to CI.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

# Make ``core`` importable.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.config import ProviderConfig  # noqa: E402
from core.provider import LLMClient, resolve_endpoint  # noqa: E402


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    # Path to the prompt file under test. Override via env var
    # ``FI_VERIFY_PROMPT_PATH`` so the script runs on any machine; the
    # Windows-temp default is a developer convenience, not a contract.
    import os
    default_path = (
        str(Path.home() / "AppData/Local/Temp/body_prompt_utf8.txt")
        if os.name == "nt" else "/tmp/body_prompt_utf8.txt"
    )
    body_prompt_path = Path(os.environ.get("FI_VERIFY_PROMPT_PATH", default_path))
    if not body_prompt_path.is_file():
        print(
            f"ERROR: body prompt file not found at {body_prompt_path} — "
            f"set FI_VERIFY_PROMPT_PATH to point at one, or drop a prompt at the default."
        )
        return 1

    prompt = body_prompt_path.read_text(encoding="utf-8", errors="replace")
    print(f"loaded prompt ({len(prompt)} chars)")

    ep = resolve_endpoint(
        ProviderConfig(name="claude_cli", model="claude-sonnet-4-6")
    )
    print(f"endpoint resolved: provider=claude_cli, primary model=claude-sonnet-4-6")

    # The escalation default that the production engine uses.
    client = LLMClient(
        ep,
        cli_timeout_s=300.0,
        cli_inactivity_timeout_s=180.0,
        node_cli_timeout_s={"implement": 1800.0},
        node_model_fallbacks={
            "implement": "claude-opus-4-7",
            "write": "claude-opus-4-7",
        },
    )

    started = time.monotonic()
    try:
        result = await client.chat(
            [{"role": "user", "content": prompt}],
            node="implement",
        )
        elapsed = time.monotonic() - started
        print(f"\nSUCCESS — response of {len(result)} chars in {elapsed:.0f}s")
        print(f"first 400 chars of response:\n{result[:400]}")
        return 0
    except Exception as e:
        elapsed = time.monotonic() - started
        print(f"\nFAILURE after {elapsed:.0f}s: {type(e).__name__}: {e}")
        return 2
    finally:
        await client.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
