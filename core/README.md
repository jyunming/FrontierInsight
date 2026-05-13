# `core/`

The engine. Async LangGraph DAG that turns a `Config` into a `QuestArtifacts`.

| File | What's in it |
|---|---|
| `engine.py` | The 11-node graph (`clarify → ideate → literature → design → implement → execute → execute_reflect → analyze → cross_check → write → review`) plus the three feedback loops. Owns `QuestState`, `Engine`, `_build_graph`, all `_node_*` methods, the JSON-leniency helpers (`_parse_json_lenient`, `_extract_result_json`, `_strip_outer_fence`), and the per-quest logger. |
| `config.py` | Pydantic v2 schema for the YAML quest file: `Config`, `ProviderConfig`, `EngineConfig`, `ExecutionConfig`, `KnowledgeConfig`, `OutputConfig`. `field_validator(..., mode="before")` expands `~` in path-shaped fields. |
| `execution.py` | The `Executor` protocol and its two impls: `VenvExecutor` (default; creates `<quest_root>/.venv/`) and `DockerExecutor` (network-disabled, `<quest_root>` bind-mounted at `/work`). `make_executor()` is the factory. |
| `provider.py` | Unified async LLM client. One `LLMClient.chat(messages) -> str` surface over three transports: HTTP-direct (openai/codex/gemini/ollama/vllm), HTTP-via-proxy (`claude_code` + the deprecated `github_copilot_*`), and CLI-exec (`claude_cli` / `codex_cli` / `copilot_cli` / `gemini_cli`). Includes `ProxySupervisor` for ref-counted proxy spawn. |
| `knowledge.py` | Axon-backed knowledge layer with three-tier retrieval (pinned local papers → Axon → external router). Tolerates missing axon install. `add_quest_artifacts` writes the structured-ingest bundle after accepted quests. |
| `vscode_bridge.py` | Python side of the Phase P TCP bridge — `VSCodeBridgeClient.chat()` shuttles JSON to the FI VSCode extension, which makes the actual `vscode.lm.*` call. |
| `platform.py` | `detect_system()` returning `"windows"`/`"macos"`/`"linux"`/`"unknown"` for OS-specific behaviors. |

Conventions: async-everywhere (`asyncio.create_subprocess_exec`, `httpx.AsyncClient`); no module-level singletons (multiple `Engine`s must coexist for `--fleet`); prompts are `string.Template` not f-strings; JSON parsing is lenient (`_parse_json_lenient`).
