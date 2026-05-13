# `docs/`

Reference documentation. The user-facing [`README.md`](../README.md) at
the repo root is the first-time-user guide; the docs in this folder
are deeper references and history.

| File | Purpose |
|---|---|
| [`capabilities.md`](capabilities.md) | Full capability inventory: every YAML field, every provider, every output kind, the three feedback loops, the provider matrix with ToS standing, the knowledge-layer retrieval flow, demo scripts, and the test suite layout. |
| [`architecture.md`](architecture.md) | Layered diagram and contracts (`Config`, `QuestState`, `QuestArtifacts`, `Executor`, `Generator`). Why FI owns the loop instead of wrapping DeepScientist. |
| [`plan.md`](plan.md) | Phased development history (Phases A–P). Each ✅ row points at the file(s) that landed for that phase. |

For agent-prompt content see [`../agents/`](../agents/) (one file per
DAG node, plus reviewer personas).
