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

### Which providers can see images

The visual check's AI step (`output.visual_check_ai: true`; off by default) sends page screenshots of the paper, slides and poster (and of `slides.pptx`, when LibreOffice is installed) to the configured provider. Where a provider cannot take images, the check runs on its measurements alone and its report says so.

- **HTTP providers** (`ollama`, `vllm`, `openai`, `gemini`, …) send the screenshots as image parts. The model itself must accept images; checked with `gemma4:31b-cloud` on Ollama.
- **`claude_cli`** sends them inline in a stream-json turn. Checked with `haiku`.
- **`vscode_extension`** hands them to `vscode.lm` as image data. This needs a VS Code build that has `LanguageModelDataPart.image` and a chat model with image input.
- **`codex_cli`** passes each screenshot as a temporary file with `codex exec -i`. Checked with `gpt-5.6-luna` under the answer-only flags below: it named the colour of a test image.
- **`copilot_cli`, `gemini_cli`, `antigravity_cli`:** measurements only for now.

## Answer-only CLI calls

FI asks a CLI provider for an answer, not for work on your machine. Started
with their defaults, `codex` and `claude` are agents: one codex call inside a
quest node read the user's personal skills, ran shell commands and made 8 web
searches, 1.35M tokens for a single answer. FI therefore starts both
answer-only:

| Turned off | `codex_cli` | `claude_cli` |
|---|---|---|
| Web search | `-c web_search=disabled` | `--tools ""` (no built-in tool at all) |
| Shell and code execution | `--disable shell_tool`, `unified_exec`, `code_mode_host`; sandbox `-s read-only` | `--tools ""` |
| MCP servers | not loaded (`--ignore-user-config`) | `--strict-mcp-config` |
| Skills, plugins, apps | `--disable skill_search`, `plugins`, `apps` | `--disable-slash-commands`, `--safe-mode` |
| Subagents, browser and computer use, image generation and viewing, tool suggestions | `--disable multi_agent`, `browser_use`, `computer_use`, `image_generation`, `view_image`, `tool_suggest` | `--tools ""` |
| Memory and personal settings | `--disable memories`; `~/.codex/config.toml` not read | `CLAUDE.md`, auto-memory, hooks and other customisations off (`--safe-mode`) |
| Saved sessions | `--ephemeral` | `--no-session-persistence` |

Every CLI call, for every CLI provider, runs in a new, empty temporary
directory that is removed when the call ends, also after an error or a
timeout; codex is pointed at it with `-C`. A CLI no longer inherits FI's own
working directory (often a repository checkout) and cannot read what is there.
FI's own files for the call, such as codex's answer file and the visual check's
screenshots, are passed by absolute path from outside that directory.

**`codex_cli` does not read `~/.codex/config.toml`.** Custom model providers,
profiles, MCP servers and every other setting in that file are not used by
FI's calls; the `codex login` sign-in still is. Choose the model with
`provider.model` — left blank, codex uses its own default model, not the
`model` in config.toml — and the reasoning level with
`provider.reasoning_effort`. A model you can reach only through a custom
provider defined in config.toml cannot be used through `codex_cli`.

A model can still ask for a tool: `claude_cli` answers each such call with
"No such tool available" and nothing runs, and `codex_cli` has no tool to call.
A model can also *describe* running a command it never ran — in one check
Haiku wrote out an `echo hello` result while claude's event stream held no tool
call — so a CLI answer is text, not evidence that anything was executed.

Measured through FI's own client with a prompt asking the model to list the
directory, run `echo hello` and search the web. `codex_cli`
(`gpt-5.6-luna`, `reasoning_effort: low`): without these flags codex ran one
command and one web search and used 81,196 input tokens; with them its event
stream held no command, web-search, MCP or subagent item, it answered
`NO TOOLS`, and it used 12,988 input tokens. `claude_cli` (`claude-haiku-4-5`)
started with no tools, MCP servers or skills and no memory path, and reported
no web search or fetch. Its start-up event still names the installed plugins
and the built-in agents, but lists no tool, skill or MCP server from them, and
with no tools there is no way to start an agent.

Not restricted yet — these run in the empty directory, but their tools are on:

- **`antigravity_cli`** has no option that turns its tools off, and it works
  in its own fixed workspace (`~/.gemini/antigravity-cli/scratch`) wherever it
  is started.
- **`copilot_cli`** keeps `--allow-all-tools`. Restricting it needs a check
  against the real CLI, which could not be made while the account's Copilot
  premium-request quota was used up.
- **`gemini_cli`** keeps `--yolo`. The Gemini CLI no longer signs in
  individual Google accounts, so it could not be checked either.

Not checked: whether codex still reads a global `~/.codex/AGENTS.md`; that file
was empty on the machine these flags were checked on.

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

## Reasoning effort

`provider.reasoning_effort` sets how hard the model reasons before it
answers. Leave it unset (the default) and FI sends nothing, so each
provider keeps its own default: a CLI uses its own config (for example
`model_reasoning_effort` in `~/.codex/config.toml`), and a local Ollama
model does not think at all. Set it in the YAML, or from the interview's
"Show advanced" screen ("Reasoning effort") on the CLI, the web
`/interview` page or `@fi /new`.

```yaml
provider:
  name: ollama
  model: gemma4:31b-cloud
  reasoning_effort: high        # minimal | low | medium | high | xhigh | max
```

| `provider.name` | How the level is sent | Levels applied |
|---|---|---|
| `openai`, `codex`, `gemini`, `vllm` | `reasoning_effort` in the chat-completions body | all six, as-is; the model decides which it supports |
| `ollama` | `reasoning_effort` in the chat-completions body | `low`, `medium`, `high` — Ollama answers any other level with a 400 |
| `claude_cli` | `claude --effort <level>` | `low`, `medium`, `high`, `xhigh`, `max` |
| `codex_cli` | `codex exec -c model_reasoning_effort="<level>"` | all six; the model decides which it supports |
| `antigravity_cli` | `agy --effort <level>` | `low`, `medium`, `high` |
| `copilot_cli`, `gemini_cli`, `vscode_extension`, `claude_code`, `github_copilot_cli`, `github_copilot_vscode` | not sent — no setting FI can pass | none |

When a level cannot be applied — the provider has no such setting, or does
not accept that level — FI leaves it out and the quest log says so once,
for example `reasoning_effort=max is not applied on ollama: its server
accepts low, medium, high ...`. The call then runs at the provider's
default instead of failing. A value outside the six levels is rejected when
the config loads. Fallback providers (`provider.fallback`) inherit the level
and follow the same rules.

Measured on `gemma4:31b-cloud` through a local Ollama 0.17.7, one short
arithmetic question sent through FI's client: unset gave 6 completion
tokens and no reasoning text; `high` gave 229 completion tokens and 453
characters of reasoning text, with the same answer. The token log
(`.fi/cost.jsonl`) records those completion tokens, so a higher level costs
more tokens per call.

## Cost expectations

Measured premium-request burn per quest (each is one LLM call against
your subscription budget), counted from the token logs (`.fi/cost.jsonl`)
of 17 complete runs of one SIR simulation quest on gemma4 through Ollama,
with `knowledge.source_routing: manual` and slides and a poster:

| Quest shape | Requests |
|---|---|
| Default engine settings (`clarify_mode: off`, single reviewer, `cross_check_per_finding_k: 3`) | 21–26 |
| Slides, poster, talk script | +1 each |
| Design through review running a second time | 27–33 in four runs of the same quest on older engine versions, not counting slides and poster |
| `knowledge.source_routing: auto` (the default) | +1 per literature pass, +1 per cross-check lookup |
| Reviewer panel of N personas | N + 1 per review round (the personas plus a moderator) instead of 1 |

The spread within 21–26 comes from the model: 0–3 experiment repair
calls, one cross-check call per key finding that found related literature,
and a second write → claim check → review pass when the review asked for
a rewrite. With Copilot Pro (~300 premium requests/month) that is about
10–13 quests of this size a month. Enterprise plans have higher ceilings.

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
