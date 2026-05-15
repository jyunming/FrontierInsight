# Provider Layer Audit (03)

Scope: `core/provider.py` (1,030 LOC), `core/vscode_bridge.py` (281 LOC),
`docs/PROVIDERS.md` (129 LOC), and the policy hook in `core/engine.py`.
LOC counts here come from `wc -l`; an editor reading the
last-line-number reports N+1 (1031 / 282 / 130 for the three above) —
both numbers point at the same file. Date: 2026-05-15.

## Findings

### 1. Dispatch is concentrated and readable — but it's not a `Provider` protocol

There is no `Provider` ABC or `Protocol`. Instead, all 13 names converge
on a single `LLMClient` class whose `chat()` / `_chat_impl()` does a
3-way string branch on `endpoint.transport`:

- `core/provider.py:835` — `if self.endpoint.transport == "cli": return _chat_cli(...)`
- `core/provider.py:837` — `if self.endpoint.transport == "vscode_bridge": return _chat_vscode_bridge(...)`
- `core/provider.py:842` — fall-through to HTTP/OpenAI-compat path.

The 13 provider *names* (`ProviderName` Literal at
`core/config.py:16-35`) collapse onto 3 transport classes resolved at
`resolve_endpoint` / `resolve_endpoint_async`:

| Transport | Provider names | Resolution |
| --- | --- | --- |
| `http` | `codex`, `openai`, `gemini`, `ollama`, `vllm` (direct); plus `claude_code`, `github_copilot_cli`, `github_copilot_vscode` (after proxy spawn) | `_DIRECT_DEFAULTS` map at `provider.py:75`; proxy port substituted at `provider.py:706` |
| `cli` | `claude_cli`, `codex_cli`, `copilot_cli`, `gemini_cli` | `_CLI_SPECS` map at `provider.py:132` |
| `vscode_bridge` | `vscode_extension` | resolved synchronously at `provider.py:645-669` |

The configuration shape (`_DIRECT_DEFAULTS`, `_CLI_SPECS`) is good — new
HTTP and CLI providers can be added with a dict entry, no new branches
needed. But because the dispatch is *transport-level*, not
*provider-level*, downstream code that wants per-provider behavior (e.g.
"warn on `copilot_cli`", "force a specific endpoint shape for
`gemini`") has to scatter `if name == ...` checks across modules. The
sanction warning in `core/engine.py:2374` is one such example;
`provider.extra["bridge_port"]` for `vscode_extension` at
`provider.py:650` is another.

**Verdict — abstraction quality:** B+. The transport collapse is the
right factoring; the missing piece is a thin `ProviderAdapter` step
between the YAML name and the resolved endpoint, where each adapter
exposes `default_env`, `default_url`, `supports_streaming`, `is_sanctioned`,
`startup_hint` (e.g. "run `claude login`"). Today those live as separate
module-level structures (`_DIRECT_DEFAULTS`, `_CLI_SPECS`,
`_UNSANCTIONED_PROXY_PROVIDERS`, `_AGENTIC_CLI_PROVIDERS`) — easy to
miss when adding a new provider.

### 2. Transport instrumentation is uneven

**HTTP transport** is the most mature:

- Retry policy is explicit and well-documented: 4 attempts, exp backoff
  2–20 s, only `HTTPStatusError`/`TransportError`/`ReadTimeout`
  retried (`provider.py:886-892`). The cancellation contract is
  documented at length at `provider.py:862-885` — `CancelledError` is
  a `BaseException`, so `retry_if_exception_type(Exception)` does NOT
  swallow it, and there is a regression test
  (`test_chat_propagates_cancellation_promptly` at
  `tests/test_provider.py:342`).
- Auth handling correctly omits the `Authorization` header when the
  key sentinel is `_NO_KEY_SENTINEL` (`provider.py:857`).
- 4xx errors surface immediately, 5xx retried (`provider.py:896-899`).

**CLI transport** is well-instrumented but less symmetric:

- Retry: 4 attempts, exp backoff, retries `OSError` and
  `_CliTransientError` (`provider.py:1010-1015`).
- Per-call wall-clock timeout (`cli_timeout_s`, default 300 s) with
  forced `proc.kill()` + 5 s reap budget (`provider.py:556-585`).
- Windows-PATHEXT bug specifically fixed via `shutil.which`
  (`provider.py:491`) — a real-world papercut that took someone a
  while to track down (`asyncio.create_subprocess_exec` does not
  honor PATHEXT).
- Sub-second precision retained in error message
  (`provider.py:563-564`) — small but thoughtful.
- Stdout is `DEVNULL`'d when the answer lands in `--output-last-message`
  (`provider.py:524-528`) — avoids buffering large CLI logs.
- Tmpfile cleanup is in a single `finally` (`provider.py:604-606`) —
  comment notes a prior leak; nice fix.

**VSCode bridge transport** is more aggressive on retries:

- 6 attempts, exp backoff 4–60 s, custom predicate
  (`provider.py:962-967`), with `_TRANSIENT_BRIDGE_MARKERS` at
  `provider.py:373-402` listing 18 different transient strings to
  match (including `"bridge stalled"`).
- Wraps an "across 6 retry attempts" friendly message at
  `provider.py:973-981` — by far the best error UX of any transport.

The asymmetry: HTTP transport will retry on any 5xx, but a 429 from
`openai` is technically a 4xx and will *not* be retried — even though
that's the canonical transient. (`provider.py:899` calls
`r.raise_for_status()` after the 5xx guard.) Meanwhile, the bridge
transport explicitly lists `"rate limit"` in its transient markers
(`provider.py:387`). **HTTP's 429 handling is the most visible gap.**

### 3. Provider parity is good — except `streaming` is universally absent

Every provider supports the per-node model override:

- HTTP path (direct or proxy): `model` kwarg lands in
  POST body at `provider.py:843` (`(model or self.endpoint.model)`).
  Verified by `test_http_chat_uses_per_call_model_override`
  (`tests/test_per_node_model_routing.py:100`).
- CLI path: `model_override` flows into `_run_cli`'s `model` arg at
  `provider.py:1021-1028`; `_run_cli` injects `[spec.model_flag, value]`
  at `provider.py:501-504`. Verified at
  `tests/test_per_node_model_routing.py:162`.
- VSCode bridge path: `model_override` becomes `model_hint` at
  `provider.py:932-935`; an empty string means "use the Chat picker
  selection" (`provider.py:927-931`).

Every provider also honors the per-call timeout, *but* the two
timeouts are wired differently:

- HTTP: `timeout_s=120.0` (default) on `LLMClient` ctor → applied to
  the `httpx.AsyncClient` at `provider.py:732`. There is only ONE
  socket-level timeout; the retry budget extends wall time but each
  attempt is capped.
- CLI: `cli_timeout_s=300.0` (default) → bounded `asyncio.wait_for`
  around `proc.communicate` (`provider.py:556-558`). The retry layer
  then re-runs with the same budget.
- VSCode bridge: no per-call wall-clock timeout in the Python client.
  Instead, the TS bridge has a 180 s "no-chunks-for" stall budget that
  gets surfaced as `bridge stalled` (`provider.py:401`) and retried.

**Streaming** is *not* a first-class feature. `chat()` returns `str`
end-to-end, not `AsyncIterator[str]`. The bridge protocol *does* stream
(`lm_chunk` messages, `vscode_bridge.py:239-241`), but the chunks are
reassembled into a single string before `chat()` returns
(`vscode_bridge.py:251`). HTTP and CLI never stream. This is a
deliberate design choice (FI nodes consume JSON blobs all-or-nothing),
but it forecloses things like in-progress chat-panel updates and
mid-call cancellation on partial output.

**Ranking — provider parity:**

1. `vscode_extension` (best — has node tagging, model hint, modal
   clarify support, streaming on the wire even if not exposed)
2. HTTP-direct providers (`openai`, `gemini`, `ollama`, `vllm`,
   `codex`) — solid, well-tested
3. CLI providers (`claude_cli`, `codex_cli`, `gemini_cli`) — bounded
   and retryable, but argv-leak risk for `copilot_cli` (`-p <prompt>`
   on argv, `provider.py:172`) is a real concern
4. Proxy providers (`claude_code`, `github_copilot_*`) — second-class
   citizens; need a third-party install (`FI_CLAUDE_CODE_WRAPPER_DIR`,
   `npx copilot-api@latest`) and the warning calls them risky
5. `copilot_cli` — fourth-class; documented at the spec-level as
   broken-by-design (`provider.py:158-172`)

### 4. Error UX — actionable on the CLI path, abstract elsewhere

The CLI transport has by far the best error messages:

- Missing binary on PATH: includes the exact login commands users
  need (`provider.py:493-497`, `provider.py:546-550`).
- Timeout exceeded: includes wall-clock budget + sub-second precision +
  whether the kill itself was clean (`provider.py:582-585`).
- Non-zero exit: includes the last 500 chars of stderr
  (`provider.py:591-594`).

Proxy spawn errors are good (cite the install steps, env var name to
set: `provider.py:330-334`, `provider.py:346-353`). But once the proxy
is up, errors fall back to whatever the wrapper emits — and that's
typically a generic 5xx without provider attribution.

The HTTP path errors are *less* actionable. A 401 from OpenAI
surfaces as `httpx.HTTPStatusError` with whatever the body says. The
`_error_note` decorator at `provider.py:806-821` does attach
`provider=...,transport=...,model=...,node=...` to every exception
via `add_note` — that's a real win, since the user no longer sees a
bare httpx stack with no hint of which node failed. But there's no
"run `openai api keys list`"-style hint for, e.g., a bad
`OPENAI_API_KEY`. The bridge transport's "Copilot backend was
unavailable across 6 retry attempts" message
(`provider.py:975-981`) is the best of the three.

### 5. The `vscode_extension` provider — well-isolated, but the launch contract is implicit

The transport chain:

1. The user installs the FI VSCode extension and clicks "Start FI".
2. The extension picks a free port, listens on
   `127.0.0.1:<port>`, then spawns Python with
   `--vscode-bridge-port <port>` (`launch.py:260-269`).
3. `launch.py:_apply_vscode_bridge_override` (`launch.py:332-342`)
   sets `cfg.provider.name = "vscode_extension"` and
   `cfg.provider.extra["bridge_port"] = port` BEFORE the engine
   constructs.
4. `resolve_endpoint` (sync path, `provider.py:645-669`) reads
   `provider.extra["bridge_port"]` and produces a `ResolvedEndpoint`
   with `transport="vscode_bridge"` and `vscode_bridge_port=port`.
5. `LLMClient._chat_vscode_bridge` (`provider.py:903-984`) lazily
   constructs a `VSCodeBridgeClient` (one per `LLMClient`) on the
   first chat call.
6. The bridge client opens a persistent TCP connection, runs a
   background `_read_loop` (`vscode_bridge.py:210-235`) to demux
   responses by `id`, and serializes writes through an asyncio.Lock
   (`vscode_bridge.py:78`, `:203`).

The "supervisor" of this provider is the *VSCode extension*, not
`ProxySupervisor`. That asymmetry is fine but undocumented in the
module docstring — `ProxySupervisor`'s purpose comment at
`provider.py:248-255` refers to "Phase A leaves the spawn paths as
NotImplementedError; Phase C fills them in" but doesn't mention
that a fourth transport (vscode_bridge) sidesteps the supervisor
entirely.

The bridge protocol (newline-delimited JSON) is documented in the
`vscode_bridge.py:11-35` docstring with a complete wire-format spec.
That is excellent — it's the only transport whose wire protocol is
written down in the codebase.

**Concerns:**

- `provider.py:650`: `port = int(provider.extra.get("bridge_port", 0))`.
  An attacker who can poison `provider.extra` could redirect FI's LLM
  calls to a malicious local listener. The
  `_apply_vscode_bridge_override` chain is the only intended
  setter; users could in principle hand-write `provider.extra.bridge_port`
  in YAML. Worth a `provider.extra` allow-list in the future.
- `vscode_bridge.py:143`: `asyncio.get_event_loop().create_future()`.
  Deprecated in 3.10+, replaced by `asyncio.get_running_loop()`. The
  current behavior is fine but emits a warning under PYTHONDEVMODE.
- `vscode_bridge.py:101`: `except (asyncio.CancelledError, Exception):`
  *includes* `CancelledError`. This is in `aclose()` which is an
  intentional teardown, so it's correct here, but in any other
  context this pattern would silently swallow cancellation — worth a
  comment.

### 6. Sanction policy is enforced once per process, in the engine

`_warn_if_unsanctioned_provider` (`engine.py:2374-2454`) classifies
providers into three buckets:

- **`_UNSANCTIONED_PROXY_PROVIDERS`** (`engine.py:2359`):
  `github_copilot_cli`, `github_copilot_vscode`. Both route through
  the third-party `copilot-api` reverse-engineered proxy. The
  warning quotes the upstream README's own abuse-detection caveat.
- **`_AGENTIC_CLI_PROVIDERS`** (`engine.py:2369`):
  `copilot_cli`. Documented at the spec level as "broken as a chat
  backend" (`provider.py:158-172`).
- **Everything else** — no warning.

`docs/PROVIDERS.md:114-129` documents both warnings and the
`FI_SUPPRESS_PROXY_WARN=1` escape hatch. The matrix at
`docs/PROVIDERS.md:43-59` also visually flags the three risky
providers with `⚠️`. Good cross-referencing.

The warnings are clear, multi-line, opinionated, and point to the
sanctioned alternative for each category. This is probably the best
"actionable error/warning" path in the entire codebase.

**Two improvements possible:**

1. The warning is at engine init only. If `provider.name` is, say,
   `openai` (sanctioned) and `provider.node_models["write"]` is
   wired to a model the API doesn't expose, the user sees a 404
   from OpenAI with no hint that the issue is the *node-level*
   model override. The warning has no per-node visibility.
2. `claude_code` is documented in PROVIDERS.md as ⚠️ "Third-party
   wrapper" but does NOT trigger a runtime warning. That's an
   inconsistency — either it deserves a warning too, or the matrix
   should soften the marker.

### 7. There is no provider-level failover

When `openai` returns 429, the HTTP retry layer
(`provider.py:886-892`) waits exp-backoff and retries the same
endpoint. After 4 attempts it raises and the quest fails. There is
no logic anywhere in `LLMClient`, `Engine`, or the LangGraph nodes
that falls over to a secondary provider (e.g. `ollama`,
`claude_cli`).

The `external_fallback` knob in `config.py:232` is a *knowledge layer*
fallback (external source list when Axon is unavailable), not a
provider fallback. There is no `provider.fallback_chain: ["openai",
"ollama"]` field. The Phase O `node_models` mechanism is per-call but
not per-provider — every node still goes through the single
`LLMClient`.

The only thing close is the LangGraph checkpointer: a quest that
exhausts retries dies, the user re-runs with `--resume <quest_id>`
(`launch.py:251-258`), the saved checkpoint at
`<quest_root>/.fi/state.sqlite` lets it pick up at the last completed
node. That's a *whole-run* fallback (you can manually change provider
between runs), not an *in-run* one.

**Current behavior, documented:** if a provider fails, the quest dies
at that node, the user is responsible for retrying (after a delay)
or swapping providers and resuming.

### 8. Other observations

- `provider.py:610`: `_wait_for_openai_endpoint` blocks on
  `time.sleep(0.5)` in a loop. It's called from
  `asyncio.to_thread(self._spawn, ...)` (`provider.py:270`), so it
  doesn't block the event loop. Good.
- `provider.py:103-108`: `PROXY_PROVIDERS` is exported as a
  `frozenset`, with `_PROXY_PROVIDERS` retained as a back-compat
  alias. Same pattern for `CLI_PROVIDERS`. Nice immutability story.
- `provider.py:418-449`: `_extract_gemini_response` correctly handles
  the two failure modes the comment describes (trailing output,
  earlier `{` in warning lines). The incremental `JSONDecoder.raw_decode`
  walk is the right choice over `find/rfind` slicing.
- `provider.py:739-742`: `_bridge` is lazy. A non-VSCode quest never
  pays for the import. Good.
- `provider.py:728`: `cli_timeout_s` is a constructor parameter on
  `LLMClient`, but NOT exposed through YAML config — `engine.py:213`
  constructs `LLMClient(endpoint)` with default 300 s. Long-running
  quests on a heavily loaded gemini CLI have no way to extend that
  budget without code edits.
- `vscode_bridge.py:143`, `:176`: deprecated `get_event_loop()`. Use
  `asyncio.get_running_loop()` (no functional change today, but a
  trap for future asyncio versions).

## Recommendations

1. **[high impact] [medium effort] Introduce a `ProviderAdapter`
   protocol.** Replace `_DIRECT_DEFAULTS`, `_CLI_SPECS`,
   `_UNSANCTIONED_PROXY_PROVIDERS`, `_AGENTIC_CLI_PROVIDERS`, and the
   `vscode_extension` branch in `resolve_endpoint` with a single
   registry of `ProviderAdapter` subclasses, one per provider name.
   Each adapter exposes: `resolve_endpoint`, `is_sanctioned`,
   `sanction_warning`, `transport_kind`, `install_hint`,
   `supports_streaming`. The dispatch in `_chat_impl` stays
   transport-keyed; the *registration* becomes
   provider-keyed. Net win: adding `anthropic_direct` (the obvious
   missing entry today) becomes one new file, not three string
   additions across two modules.

2. **[high impact] [low effort] Retry 429 on the HTTP path.** Today
   only 5xx triggers a retry (`provider.py:896-899`). Add a 429-specific
   branch that honors the `Retry-After` header if present, otherwise
   the same exp backoff. The bridge transport already lists
   `"rate limit"` in `_TRANSIENT_BRIDGE_MARKERS` (`provider.py:387`);
   the HTTP path should parity.

3. **[high impact] [medium effort] Expose `cli_timeout_s` and
   `http_timeout_s` in `EngineConfig`.** Today they are constructor-
   only on `LLMClient`. A user whose `gemini_cli` is slow on cold
   start has no way to extend the 300 s budget without editing
   `core/engine.py:213`. Add `engine.cli_timeout_s` /
   `engine.http_timeout_s` to `config.py` and thread them through.

4. **[medium impact] [low effort] Add a provider failover chain.**
   `provider.fallback_chain: ["openai", "claude_cli"]` — when the
   primary provider exhausts its retry budget on a *retryable*
   error class (5xx, timeout, bridge connection drop, CLI rc != 0
   for a few attempts), construct a fresh `LLMClient` from the next
   entry and try once. Don't fall over on 4xx (auth/quota) — those
   are real bugs in the user's config, not transient ones.

5. **[medium impact] [low effort] Warn on `claude_code`.** Today
   `docs/PROVIDERS.md:57` flags it as ⚠️ "Third-party wrapper" but
   the runtime emits nothing. Add it to a new
   `_THIRD_PARTY_WRAPPER_PROVIDERS` bucket with its own warning that
   references the install steps and acknowledges the wrapper risk.
   Or, alternately, drop the ⚠️ from the doc and call it
   first-class — but the asymmetry is confusing.

6. **[medium impact] [medium effort] Expose streaming.** The bridge
   transport already streams (`vscode_bridge.py:239-241`); HTTP can
   stream trivially by passing `stream=True` to httpx; CLI can stream
   by reading stdout incrementally. Even if the engine nodes consume
   complete JSON, surfacing chunks to the VSCode chat panel as
   progress would dramatically improve perceived latency. Add a
   `chat_stream()` method that returns `AsyncIterator[str]`,
   reimplement `chat()` as `"".join(...)` over `chat_stream()`.

7. **[medium impact] [low effort] Move `cli_timeout_s` enforcement to
   the bridge transport too.** Today there's no Python-side timeout
   on bridge calls — only the TS bridge's 180 s stall budget. A
   hung extension (e.g. user closed VSCode but FI is still running)
   could hang a quest indefinitely. Add a per-call wall-clock cap
   to `_chat_vscode_bridge` matching the CLI transport's pattern.

8. **[low impact] [low effort] Allow-list `provider.extra` fields.**
   Today `provider.extra["bridge_port"]` is a free-form dict
   (`config.py:55`). Pydantic-validate that only known keys (today,
   just `bridge_port`) are accepted. Cheap defense against a
   YAML-injection that could redirect calls.

9. **[low impact] [low effort] Replace `asyncio.get_event_loop()`
   in `vscode_bridge.py:143` and `:176` with
   `asyncio.get_running_loop()`.** Future-proofs against the 3.12+
   deprecation.

10. **[low impact] [low effort] Add a `--list-providers` flag to
    `launch.py`** that prints the full matrix from `docs/PROVIDERS.md`
    plus runtime checks (binary on PATH? env var set? wrapper dir
    valid?). Removes "is my install working?" friction.

11. **[low impact] [medium effort] Probe at engine init.** Today a
    typo in `OPENAI_API_KEY` (or missing `gh auth login`) doesn't
    surface until the first chat call, minutes into a quest. Run a
    cheap probe (e.g. `GET /v1/models` for HTTP, `--version` for
    CLI, `ping` for bridge) in `Engine.run()` *before* graph
    compilation and fail fast with a hint pointing to the right
    login/install step.

12. **[low impact] [low effort] Document the launch contract for
    `vscode_extension`.** `core/provider.py:645-669` reads
    `provider.extra["bridge_port"]` but the only documentation of
    that wire is in `launch.py:332-342` and `docs/PROVIDERS.md`
    doesn't mention `extra` at all. Add a one-paragraph note to
    PROVIDERS.md.

## References

- `core/provider.py` (1,030 LOC)
  - `_DIRECT_DEFAULTS` map: lines 75-101
  - `PROXY_PROVIDERS`, `_PROXY_PROVIDERS`: lines 103-108
  - `_CliSpec` dataclass and `_CLI_SPECS` registry: lines 111-193
  - `ResolvedEndpoint`: lines 204-237
  - `ProxySupervisor` (refcounted lifecycle): lines 248-364
  - `_TRANSIENT_BRIDGE_MARKERS`: lines 373-402
  - `_extract_gemini_response`: lines 418-449
  - `_run_cli`: lines 478-606
  - `resolve_endpoint` / `resolve_endpoint_async`: lines 631-711
  - `LLMClient.chat` (with `add_note` error attribution): lines 754-804
  - `_chat_impl` (transport dispatch): lines 823-901
  - `_chat_vscode_bridge`: lines 903-984
  - `_chat_cli`: lines 986-1030
- `core/vscode_bridge.py` (281 LOC)
  - Wire protocol docstring: lines 1-36
  - `BridgeError`: lines 48-52
  - `VSCodeBridgeClient`: lines 55-281
- `core/engine.py`
  - `_warn_if_unsanctioned_provider`: lines 2374-2454
  - `_UNSANCTIONED_PROXY_PROVIDERS` / `_AGENTIC_CLI_PROVIDERS`:
    lines 2359-2369
  - `LLMClient(endpoint)` construction: line 213
  - `_chat` / `_chat_messages` / `_model_for_node` (Phase O routing):
    lines 1562-1609
- `core/config.py`
  - `ProviderName` Literal (the canonical list of 13): lines 16-35
  - `ProviderConfig.node_models`: line 74
- `docs/PROVIDERS.md` (129 LOC) — TL;DR matrix, provider matrix,
  per-node routing, cost expectations, sanction warnings.
- `launch.py:_apply_vscode_bridge_override`: lines 332-342
- Tests:
  - `tests/test_provider.py` (HTTP path, supervisor, cancellation,
    error notes)
  - `tests/test_provider_cli.py` (all 4 CLI providers, model flags,
    timeouts, missing-binary errors)
  - `tests/test_per_node_model_routing.py` (HTTP and CLI per-call
    overrides)
  - `tests/test_vscode_bridge.py` (bridge wire protocol, demux,
    error handling)
