"""The line FI writes when it asks the plan's model to complete something reads plainly (no "the plan's plan")."""
from __future__ import annotations

import asyncio
from pathlib import Path

from tests.test_code_review import _engine


def _ask(tmp_path: Path, what: str) -> str:
    engine = _engine(tmp_path)
    gaps = ["The topic says to vary or compare x (1, 2), but the plan's settings do not include it."]

    async def complete(ask, path, **kwargs):  # the plan's model is not asked: the completion is stubbed
        return None

    engine._complete_plan_parts = complete  # type: ignore[method-assign]
    asyncio.run(engine._ask_plan_to_complete(
        {}, "sha", lacks=lambda design: gaps, request=lambda design, text: "ask", keep=lambda before, revised: revised,
        file="plan_wording_asked.json", what=what, short="it", note="n", ask_limit=1))
    return "\n".join(line for line in (engine.fi_dir / "run.log").read_text(encoding="utf-8").splitlines() if "[plan]" in line)


def test_what_the_topic_asks_for_reads_plainly(tmp_path: Path) -> None:
    text = _ask(tmp_path, "what the topic asks for")
    assert "the plan lacks what the topic asks for: The topic says to vary" in text
    assert "plan's plan" not in text and "was missing" not in text


def test_a_part_of_the_plan_keeps_its_wording(tmp_path: Path) -> None:
    text = _ask(tmp_path, "settings the experiment needs")
    assert "the plan's settings the experiment needs was missing" in text


def test_a_contradiction_the_code_reader_found_reads_plainly(tmp_path: Path) -> None:
    text = _ask(tmp_path, "what makes its parts agree")
    assert "the plan lacks what makes its parts agree: The topic says to vary" in text
    assert "plan's a plan" not in text and "was missing" not in text


def test_every_caller_names_a_part_of_the_plan_or_what_it_lacks() -> None:
    import ast
    import re
    whats = []
    for f in sorted((Path(__file__).resolve().parents[1] / "core").rglob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "_ask_plan_to_complete":
                what = next((k.value for k in node.keywords if k.arg == "what"), None)
                # every caller passes `what` as a literal keyword, so the sentence it makes can be read here
                assert isinstance(what, ast.Constant) and isinstance(what.value, str), f"{f.name}:{node.lineno}"
                whats.append(what.value)
    assert len(whats) >= 5, whats
    # "the plan's <what> was missing" reads badly when <what> is itself a whole phrase or speaks of a plan
    bad = [w for w in whats if not w.startswith("what ") and (re.match(r"(a|an|the)\b", w) or re.search(r"\bplan", w))]
    assert not bad, bad
