---
title: Waiting out provider outages
sources: [core/provider.py]
updated: 2026-09-30
---
# Waiting out provider outages

A model provider's server that is down for a few minutes should not end a quest. On the HTTP connection (OpenAI, Gemini, Ollama, vLLM, any OpenAI-compatible `base_url`, and the local Claude Code / Copilot proxies), FI tells an outage from a real error:

- **Outage**: a server error (5xx, including Cloudflare's 520–529 except 524), or a "too many requests" (429) that is not a used-up quota. FI tries up to 6 times, waiting about 10, 20, 40, 60 and 90 s (±20 %), at most 300 s for one call. A `Retry-After` of up to 120 s is honoured.
- **Short hiccup** (a dropped connection or timeout alone, 501/505/524): up to 4 tries, at most 20 s apart. The same applies when a fallback provider is ready to take over.
- **No retry**: other client errors (4xx), and a 429 that says the quota or a daily/weekly/monthly limit is used up. That also switches to `provider.fallback` at once.

While FI waits it gives back its slot in the concurrent-call limit, so other quests can use it. Each wait writes one line to `run.log`, for example "attempt 2 of 6: the provider's server is down or busy (HTTP 521) … trying again in 21s".

There is no setting for this. For a longer outage, configure `provider.fallback`. It behaves the same from the CLI, the web page and VS Code, because it is part of the connection. CLI providers (4 tries, 30–90 s apart on a capacity error) and the VS Code chat model (6 tries) have their own retry rules.

Related: [[Model reasoning trace]] (what the model returned on the call that finally answered), [[Pauses]].
