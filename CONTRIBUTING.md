# Contributing to Frontier Insight

Thanks for your interest. FI is a Windows + Linux native automated research pipeline (async LangGraph engine, per-quest venv/Docker execution, Axon-backed knowledge layer). Contributions of all kinds welcome.

## Ways to contribute

- **Code** — improve any node body, generator, executor, or provider in `core/` and `generation/`. New paper templates (`templates/paper/<format>/template.tex`), slide themes, or poster layouts.
- **Tests** — expand the pytest suite in `tests/` (currently ~155 tests, all using fake LLMs via `monkeypatch`).
- **Examples** — add a config under `examples/<topic>/config.yaml` that demonstrates a new use case.
- **Bug reports / feature requests** — open an issue with a clear description and reproduction steps.

## Getting started

```bash
git clone https://github.com/jyunming/FrontierInsight.git
cd FrontierInsight
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
pip install pytest pytest-asyncio
python -m pytest -v
```

## Pull request guidelines

- Describe **what** the change does and **why** it is needed.
- Reference any related issues (e.g., `Closes #123`).
- Keep PRs focused — split unrelated changes into separate PRs.
- Update `docs/plan.md` / `docs/architecture.md` / `CLAUDE.md` when behavior or contracts change.
- All tests must pass: `python -m pytest -v`. New code should add direct tests for the module it touches.
- Tests must not call real LLM APIs — use `monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)` (or the analogous path for generators). Tests that need external CLIs (Docker, pandoc, marp, pdflatex) must `pytest.mark.skipif(shutil.which("...") is None, ...)`.

## Code style

- Python 3.10+, PEP 8.
- Async-first in `core/`: `async def`, `httpx.AsyncClient`, `asyncio.create_subprocess_exec`. Sync libraries (`docker-py`, `venv.EnvBuilder`) wrapped in `asyncio.to_thread`.
- `pathlib.Path` everywhere — no `os.path.join` with literal `/`.
- Pydantic v2 syntax (`field_validator(..., mode='before')`, `model_validate`, `model_dump`). Path-shaped Config fields use the `mode='before'` validator for tilde expansion.
- Prompts in `agents/*.md` use Python `string.Template` (`$placeholder`), not f-strings.
- The `RESULT_JSON: {...}` contract: experiment scripts emit one line on the LAST line of stdout; `_extract_result_json` parses it.

See `CLAUDE.md` for the longer rationale on conventions.

## Licensing

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

## Code of conduct

Be respectful, constructive, and inclusive.
