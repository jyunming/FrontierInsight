"""Every template variable must have a value the engine actually passes.

``string.Template.substitute`` raises ``KeyError`` for a variable with no
matching keyword, and prompts are rendered deep inside a running quest — so
this class of mistake surfaces as a node failing minutes into a run, or as a
20-minute regression sweep going red, rather than at the point of the edit.

Adding a slot to a prompt without wiring it (or renaming a kwarg without
updating the prompt) is a one-line mistake that is easy to make and slow to
find. This makes it a one-second failure instead.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import core.engine as engine_mod

AGENTS = Path(__file__).resolve().parent.parent / "agents"
VAR = re.compile(r"\$(?:\{)?([A-Za-z_][A-Za-z0-9_]*)")


def _substitute_calls() -> dict[str, set[str]]:
    """Map prompt name -> the kwargs engine.py passes to its substitute()."""
    src = inspect.getsource(engine_mod)
    calls: dict[str, set[str]] = {}

    def _kwargs_from(start: int) -> set[str]:
        depth, i = 1, start
        while depth and i < len(src):
            depth += (src[i] == "(") - (src[i] == ")")
            i += 1
        return set(re.findall(r"(\w+)\s*=", src[start:i - 1]))

    for m in re.finditer(
        r'_prompts(?:\.get)?\(?\["?([a-z_]+)"?\]\)?[^\n]*?\.substitute\(', src
    ):
        calls.setdefault(m.group(1), set()).update(_kwargs_from(m.end()))

    # ``implement_body`` is fetched into a local first, so it needs its own
    # pattern — a rename there would otherwise slip past this check.
    for m in re.finditer(r"body_prompt\.substitute\(", src):
        calls.setdefault("implement_body", set()).update(_kwargs_from(m.end()))
    return calls


CALLS = _substitute_calls()


def test_the_scan_found_the_expected_prompts() -> None:
    """Guard the guard: a regex that silently matches nothing would make
    every test below vacuously pass."""
    assert len(CALLS) >= 15, sorted(CALLS)
    for required in ("design", "write", "review", "implement", "implement_body"):
        assert required in CALLS, f"{required} substitute call not found"


@pytest.mark.parametrize("name", sorted(CALLS))
def test_every_template_slot_is_supplied(name: str) -> None:
    path = AGENTS / f"{name}.md"
    if not path.is_file():
        pytest.skip(f"{name}.md not shipped")
    needed = set(VAR.findall(path.read_text(encoding="utf-8")))
    missing = needed - CALLS[name]
    assert not missing, (
        f"agents/{name}.md uses {sorted(missing)} but the engine does not "
        f"pass {'it' if len(missing) == 1 else 'them'} — substitute() would "
        f"raise KeyError at run time"
    )


def test_skills_slot_is_wired_everywhere_it_appears() -> None:
    """The skills block is opt-in per prompt; wherever the slot exists, the
    value must be passed."""
    for path in AGENTS.glob("*.md"):
        if "$skills_block" not in path.read_text(encoding="utf-8"):
            continue
        name = path.stem
        assert name in CALLS, f"{name}.md has a skills slot but is never rendered"
        assert "skills_block" in CALLS[name], (
            f"{name}.md declares $skills_block but the engine never passes it"
        )


@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_rendered_prompt_uses_at_least_one_slot(name: str) -> None:
    """The failure mode the checks above cannot see.

    ``substitute()`` raises when the TEMPLATE names something the engine does
    not pass. It says nothing in the other direction: unused keywords are
    ignored, so a prompt that lost its placeholders renders happily with none
    of its inputs, and the model answers a question it was never asked.

    Observed: ``select_skills.md`` was rewritten using ``{topic}`` and
    ``{catalogue}`` — wrong syntax, wrong names. Every test passed, every call
    succeeded, and the model returned an empty selection for every topic
    because it had been handed the literal text ``{topic}`` instead of one.
    Nothing failed; the answers were merely meaningless.

    A prompt the engine renders with arguments and which uses no slot at all
    is definitionally that bug, and no legitimate prompt looks like it.
    """
    path = AGENTS / f"{name}.md"
    if not path.is_file():
        pytest.skip(f"{name}.md not shipped")
    used = set(VAR.findall(path.read_text(encoding="utf-8")))
    assert used, (
        f"agents/{name}.md contains no $slot at all, yet the engine renders "
        f"it with {sorted(CALLS[name])} — every input is being dropped and "
        f"the model is answering with none of them"
    )


@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_supplied_input_is_not_written_in_brace_syntax(name: str) -> None:
    """Catches the specific slip: writing ``{topic}`` for a value the engine
    really does pass as ``topic``, when the prompt has no ``$topic``.

    Scoped to that case on purpose. Braces are legitimate elsewhere — example
    JSON, Python snippets — so flagging every ``{word}`` would be noise.
    """
    path = AGENTS / f"{name}.md"
    if not path.is_file():
        pytest.skip(f"{name}.md not shipped")
    text = path.read_text(encoding="utf-8")
    used = set(VAR.findall(text))
    for key in sorted(CALLS[name]):
        if key in used:
            continue
        assert "{" + key + "}" not in text, (
            f"agents/{name}.md writes {{{key}}} but never $" + key + " — the "
            f"engine passes {key}, so this is the wrong syntax and the value "
            f"reaches the model as literal text"
        )
