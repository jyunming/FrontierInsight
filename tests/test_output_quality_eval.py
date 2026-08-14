"""Output-quality golden eval (PR-9).

A deterministic, recorded/fake-driven end-to-end quest (no live provider, no
network, no cost) that asserts a battery of OUTPUT-QUALITY invariants on the
produced paper — the contract a shippable FI paper must satisfy:

  * structural completeness (title, non-trivial body),
  * figure-reference integrity (every ![](…) resolves to a real file — the
    broken-image regression class),
  * no scaffolding leakage (no [FI:…], TODO, PLACEHOLDER, lorem ipsum),
  * a real review verdict, and
  * cost accounting was recorded.

Unlike the fleet smoke test (which only checks that *a* paper + figure exist),
this pins the quality bar, so a refactor that silently degrades the output
contract fails here. The fake LLM is the "golden" input; the invariants are the
golden bar (robust to LLM wording, unlike a byte-for-byte golden file).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig,
    OutputConfig, ProviderConfig,
)
from core.engine import Engine, QuestArtifacts


_FAKE_EXPERIMENT_CODE = """\
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)
plt.figure()
plt.plot([0, 1, 2, 3], [0, 1, 4, 9])
plt.xlabel('x'); plt.ylabel('y')
plt.savefig('figures/scaling.png', dpi=72)
print('RESULT_JSON: {"score": 0.91, "slope": 3.0}')
"""

# A deliberately well-formed paper: H1 title, prose, a figure that the
# experiment actually produces, and NO scaffolding markers.
_FAKE_PAPER = """\
# Power-law scaling of the toy metric

## Abstract
We measure a toy metric across four settings and find a clean power-law
relationship, with the fitted slope matching the analytic expectation.

## Results
The metric grows quadratically with the input, as shown in Figure 1.

![Scaling of the metric with input](figures/scaling.png)

## Discussion
The observed slope of 3.0 is consistent with the design, and the fit is tight.
"""

_FAKE_RESPONSES = {
    "Ideation": json.dumps({
        "ideas": [{"title": "scaling", "summary": "toy power law",
                   "feasibility": "high", "novelty": "low"}],
        "chosen": {"title": "scaling", "rationale": "only candidate"},
    }),
    "Experiment Design": json.dumps({
        "hypothesis": "the metric scales as a power law",
        "variables": {"independent": ["x"], "dependent": ["y"], "controls": []},
        "method": "plot y vs x and fit",
        "expected_outcome": "monotonic power law",
        "figures_planned": ["scaling.png"],
        "dependencies": ["matplotlib"],
    }),
    "Implementation": json.dumps({"code": _FAKE_EXPERIMENT_CODE, "deps": ["matplotlib"]}),
    "Analysis": json.dumps({
        "summary": "The metric follows a power law; slope 3.0.",
        "key_findings": ["y grows quadratically with x", "fitted slope is 3.0"],
        "claims_supported": ["power-law scaling holds"],
        "claims_unsupported": [],
        "limitations": ["toy data only"],
    }),
    "Writing": _FAKE_PAPER,
    "Review": json.dumps({
        "verdict": "accept", "score": 4,
        "strengths": ["clear result"], "weaknesses": [],
        "suggestions": [], "blocking": "",
    }),
}


def _fake_response_for(prompt: str) -> str:
    head = prompt.lstrip().splitlines()[0] if prompt.strip() else ""
    for tag, resp in _FAKE_RESPONSES.items():
        if tag in head:
            return resp
    return "{}"


def _golden_cfg(tmp_path: Path) -> Config:
    return Config(
        topic="power-law scaling of a toy metric",
        title="scaling",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, clarify_mode="off",
            ideate_reflect=False, auto_accept_on_pass=True,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs", kinds=[]),
    )


_SCAFFOLD_MARKERS = ("[fi:", "todo", "placeholder", "lorem ipsum", "tktk", "xxx")
_FIG_REF_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _quality_findings(art: QuestArtifacts) -> list[tuple[str, bool, str]]:
    """Return (invariant, ok, detail) for each output-quality check."""
    out: list[tuple[str, bool, str]] = []

    assert art.paper_md is not None and art.paper_md.exists()
    text = art.paper_md.read_text(encoding="utf-8")

    out.append(("paper non-trivial", len(text) > 200, f"{len(text)} chars"))
    out.append((
        "has H1 title",
        any(ln.startswith("# ") for ln in text.splitlines()),
        "no top-level heading",
    ))

    # Figure-reference integrity: every ![](path) must resolve to a real file.
    refs = _FIG_REF_RE.findall(text)
    broken = [
        r for r in refs
        if not (art.quest_root / r).exists()
        and not (art.paper_md.parent / r).exists()
    ]
    out.append(("at least one figure referenced", len(refs) >= 1, f"{len(refs)} refs"))
    out.append(("no broken figure references", not broken, f"broken={broken}"))

    # No build scaffolding leaked into the shippable paper.
    low = text.lower()
    leaked = [m for m in _SCAFFOLD_MARKERS if m in low]
    out.append(("no scaffolding markers", not leaked, f"leaked={leaked}"))

    # A real review verdict was recorded.
    verdict = (art.raw_state.get("review", {}) or {}).get("verdict")
    out.append((
        "review verdict recorded",
        verdict in {"accept", "revise", "reject"},
        f"verdict={verdict!r}",
    ))

    # Cost accounting was written.
    cost = art.quest_root / ".fi" / "cost.jsonl"
    out.append(("cost.jsonl written", cost.is_file(), str(cost)))

    return out


@pytest.mark.slow
@pytest.mark.asyncio
async def test_golden_quest_meets_output_quality_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_chat(self, messages, **kw):  # noqa: ANN001, ANN003
        return _fake_response_for(messages[-1]["content"])

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    art = await Engine(_golden_cfg(tmp_path)).run()

    findings = _quality_findings(art)
    failed = [(name, detail) for name, ok, detail in findings if not ok]
    assert not failed, "output-quality invariants violated:\n" + "\n".join(
        f"  - {name}: {detail}" for name, detail in failed
    )
