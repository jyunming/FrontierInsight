# LLM Providers

Frontier Insight talks to LLMs through a unified `LLMClient.chat()`
surface that routes to four transports: VSCode bridge, HTTP direct,
CLI exec, and HTTP via local proxy. Pick the one that matches how
you want to authenticate and which models you want to reach.

## TL;DR — which one to pick

| Your situation | Use this | Why |
|---|---|---|
| You have GitHub Copilot and use VSCode | **`vscode_extension`** | Sanctioned `vscode.lm.*` API; picks up every model your Copilot Chat picker exposes (Copilot GPT family, Claude / Gemini when your subscription includes them); no extra API keys; calls show up in your Copilot usage dashboard. |
| You want headless runs (overnight fleets, CI) | **`claude_cli`** / **`codex_cli`** | Reuses the CLI's own OAuth (`claude login`, `codex login`). One-time sign-in, then zero ongoing config. `gemini_cli` is no longer a reliable third option here — see the provider matrix note below. |
| You have API keys and want full control | **`openai`** / **`gemini`** | Standard HTTP-direct via the OpenAI-compatible interface. |
| You're running everything locally | **`ollama`** / **`vllm`** | Self-hosted; zero API spend. |

## The headline: VSCode integration

Most users should start with `vscode_extension`. It's the path where
the integration story is strongest and the cost story is simplest:

- **One sanctioned API** (`vscode.lm.selectChatModels` +
  `model.sendRequest`). No reverse-engineered proxies, no scraped
  tokens.
- **One subscription** (your GitHub Copilot Chat plan) covers every
  model the Copilot picker federates — Copilot's GPT family today,
  plus Anthropic Claude 3.5/4 and Google Gemini families when your
  plan includes them. Whatever you pick in the Copilot Chat model
  dropdown, Frontier Insight uses *that exact model*.
- **Premium-request budget** is the only cost surface — same one
  you'd hit typing into Copilot Chat manually.

In Copilot Chat:

```
@fi /new
```

The chat participant walks you through 7 quick questions, generates a
config YAML, and runs the quest. Every LLM call streams through the
`vscode.lm` bridge to your selected model.

## Provider matrix

| `provider.name` | Transport | Auth | Models reachable | ToS / risk |
|---|---|---|---|---|
| `vscode_extension` | VSCode bridge | VSCode Copilot Chat sign-in | Whatever your Copilot subscription exposes (GPT family + Claude/Gemini families when federated) | ✅ Sanctioned via `vscode.lm` |
| `openai` | HTTP direct | `OPENAI_API_KEY` env | GPT family on the OpenAI API | ✅ Sanctioned |
| `codex` | HTTP direct | `OPENAI_API_KEY` env | Same as `openai`, separate alias for legacy YAMLs | ✅ Sanctioned |
| `gemini` | HTTP direct | `GEMINI_API_KEY` env | Gemini on Google's OpenAI-compat endpoint | ✅ Sanctioned |
| `ollama` | HTTP direct (local) | none | Whatever models you've pulled locally | ✅ Self-hosted |
| `vllm` | HTTP direct (local) | none | Local vLLM-served models | ✅ Self-hosted |
| `claude_cli` | CLI exec | `claude login` (Pro/Max OAuth) | Claude family via Anthropic's CLI | ✅ Sanctioned |
| `codex_cli` | CLI exec | `codex login` (ChatGPT Plus/Pro OAuth) | OpenAI Codex CLI's model selection | ✅ Sanctioned |
| `gemini_cli` | CLI exec | `gemini` OAuth / Google AI key | Gemini via `@google/gemini-cli` | ⚠️ Google has been migrating individual-account OAuth sign-in to Antigravity; verified live this session — a previously-working individual account now gets `Code Assist for individuals... migrate to the Antigravity suite` and the CLI exits rc=1 before ever reaching FI. A Workspace/enterprise account or an `AI_STUDIO`/`GEMINI_API_KEY` login may still work; use `antigravity_cli` or the direct `gemini` (API-key) provider instead if yours doesn't. |
| `copilot_cli` | CLI exec | `gh auth login` (Copilot sub) | n/a — see warning | ⚠️ Agentic. Replies conversationally to FI's structured prompts; not usable as an FI backend. Loud warning at engine init. |
| `claude_code` | HTTP via proxy | `claude login` + spawned wrapper | Anthropic via `claude-code-openai-wrapper` | ⚠️ Third-party wrapper |
| `github_copilot_cli` | HTTP via proxy | `gh auth login` + spawned `copilot-api` | Copilot models via reverse-engineered proxy | ⚠️ Against ToS spirit (use `vscode_extension` instead) |
| `github_copilot_vscode` | HTTP via proxy | VSCode Copilot extension + spawned `copilot-api` | Copilot models via reverse-engineered proxy | ⚠️ Against ToS spirit (use `vscode_extension` instead) |

## Per-node model routing

Different nodes of the research DAG can use different models. Cheap
model for clarify/cross_check, strong model for write/review. Set it
by hand-editing the YAML below, or from the interview's "Show advanced"
screen (`python launch.py --new`, `@fi /new`, or the web `/interview`
page) — the "Per-node model overrides" field takes the same
comma-separated `node:model` pairs (`poster:gpt-4o-mini,
slides:gpt-4o-mini`) and writes this exact block for you. Either way
gets you the same YAML:

```yaml
provider:
  name: vscode_extension
  model: gpt-5                  # global default
  node_models:
    clarify:        gpt-4o-mini
    cross_check:    gpt-4o-mini
    write:          claude-3-5-sonnet
    review:         gpt-5
    review_panel.statistician: claude-3-5-sonnet
    review_panel.devil_advocate: gpt-4o
    review_moderator: gpt-4o-mini
```

For the VSCode-extension transport, each `model_hint` is passed to
`vscode.lm.selectChatModels` as a family filter; the extension picks
the closest match in your Copilot subscription. If the hint matches
nothing your subscription exposes, that one call errors with a clear
"no Copilot model available for hint" message — VSCode handles the gate.

`poster` / `slides` / `speech` are valid `node_models` keys too. Those
three generators pour an already-written paper into a fixed template
rather than doing open-ended reasoning, so a cheaper model than the
quest's primary is usually just as good there:

```yaml
provider:
  name: claude_cli
  model: claude-sonnet-4-6       # global default
  node_models:
    poster: claude-haiku-4-5
    slides: claude-haiku-4-5
    speech: claude-haiku-4-5
```

Unset, all three use the primary model like any other node — no
behavior change until you opt in. There is no automatic "pick a cheap
model for me": the string you set here is passed straight to whichever
provider is already active, exactly like every other `node_models`
entry, so the model name still has to be one that provider's current
catalogue actually has. Model catalogues drift — verified live while
building this: `antigravity_cli` rejected `gemini-2.5-flash` outright
("model gemini-2.5-flash is not recognized... Available models: Gemini
3.8 Flash / 3.7 Flash / 3.6 Flash / ..."; the working id was
`gemini-3.6-flash-low`, from `agy models`). Check your provider's own
model-list command before setting a cheap-tier override, rather than
copying a model name from elsewhere.

## Cost expectations

Rough premium-request burn per quest (each is one LLM call against
your subscription budget):

| Quest shape | Approx. requests |
|---|---|
| Bare quest, `clarify_mode: off`, single reviewer | ~6 |
| Default (`clarify_mode: auto`, single reviewer, journal-length depth) | ~10 |
| 3-persona reviewer panel + moderator | +4 per review iteration |
| With cross-paper check (`cross_check_per_finding_k: 3`) | +1 per finding |
| Each revise iteration | +2 (design + implement) plus reviewer cost |

With Copilot Pro (~300 premium requests/month) a sensible default
quest budget is 15–30 quests/month. Enterprise plans have higher
ceilings.

## API-key environment variables

| Provider | Env var | Notes |
|---|---|---|
| `openai`, `codex` | `OPENAI_API_KEY` | Standard |
| `gemini` | `GEMINI_API_KEY` | Google AI Studio key |
| `ollama` | none | local |
| `vllm` | none | local |
| CLI providers (`claude_cli`, `codex_cli`, `gemini_cli`) | none — OAuth via the CLI's own login | reused from the CLI's keychain |
| `vscode_extension` | none — uses VSCode's auth | reused from VSCode's Copilot Chat sign-in |

## Provider warnings at engine init

Two warning classes fire automatically:

- **`copilot_cli`** — explicit warning that the standalone Copilot CLI
  is an agentic tool and not a chat backend; it replies
  conversationally to FI's structured node prompts and produces
  garbage output (paper.md filled with "Are you trying to debug X?",
  experiment.py reduced to a stub). Use `vscode_extension` instead.
- **`github_copilot_cli`** and **`github_copilot_vscode`** — third-party
  reverse-engineered proxy (`copilot-api`). GitHub explicitly warns
  about abuse-detection systems triggering on automated Copilot
  scraping. Use `vscode_extension` (sanctioned) instead.

Both warnings can be silenced with `FI_SUPPRESS_PROXY_WARN=1` if you
genuinely understand the tradeoffs.
