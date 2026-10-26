"""The coding agent's prompt (agents/implement_agent.md) fills in with the placeholders the implement prompt uses."""
from __future__ import annotations

import re
from pathlib import Path
from string import Template

AGENTS = Path(__file__).resolve().parent.parent / "agents"


def _identifiers(name: str) -> set[str]:
    return set(Template((AGENTS / name).read_text(encoding="utf-8")).get_identifiers())


def test_agent_prompt_fills_every_placeholder_the_implement_prompt_uses() -> None:
    used_by_implement = _identifiers("implement.md")
    used_by_agent = _identifiers("implement_agent.md")
    assert used_by_agent, "the agent prompt should use at least one placeholder"
    # Every placeholder the agent prompt uses must be one that the implement prompt's substitution supplies.
    assert used_by_agent <= (used_by_implement | _identifiers("implement_body.md"))

    dummies = {name: f"DUMMY_{name.upper()}" for name in used_by_agent | used_by_implement}
    text = (AGENTS / "implement_agent.md").read_text(encoding="utf-8")
    filled = Template(text).substitute(dummies)  # raises KeyError / ValueError on a bad placeholder

    assert not re.search(r"\$[A-Za-z_]", filled), "a placeholder was left unfilled"
    for name in used_by_agent:
        assert f"DUMMY_{name.upper()}" in filled


def test_agent_prompt_states_the_three_rules() -> None:
    text = (AGENTS / "implement_agent.md").read_text(encoding="utf-8")
    assert "expected values" in text and "hidden" in text
    assert ".fi/" in text and "needs/" in text and "plan.md" in text
    # No self-check script exists for the agent to run: the prompt must not tell it to run one.
    assert "fi_check.py" not in text
