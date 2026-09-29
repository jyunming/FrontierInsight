"""Running a quest again from any graph step that can safely be one (core/rerun_from.py, ``--resume <id> --from <step>``).

Every step offered has its own list of files to move aside and one plain sentence; the steps up to the design replace the
frozen protocol and need a named approval. No test calls a real model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from core import rerun_from
from core.engine import Engine
from tests.test_engine_smoke import _classify, _fake_response_for
from tests.test_rerun_from import _cfg, _Graph

NEW_STEPS = ["ideas", "literature", "plan", "design", "figures", "crosscheck", "evidence", "claims"]
# The nodes that are deliberately not a step, and the step that covers the others.
NOT_A_STEP = {"clarify", "pause_after_literature", "wait_for_data", "execute_reflect", "human_feedback"}

# Pinned here, apart from the module, so that dropping an entry from OUTPUTS fails a test: what each new step moves,
# and what it leaves.
MOVES = {
    "ideas": ["data/literature", "plan.md", ".fi/paused_at_plan.flag", "needs/FROZEN_PROTOCOL.json", "code", "paper.md"],
    "literature": ["data/literature", "plan.md", "needs/FROZEN_PROTOCOL.json", "results.json", "slides.pdf"],
    "plan": ["plan.md", ".fi/paused_at_plan.flag", "needs/FROZEN_PROTOCOL.json", "needs/protocol_versions",
             "needs/receipts/design_audit.json", "code", "paper"],
    "design": ["needs/FROZEN_PROTOCOL.json", "needs/protocol_versions", "needs/PROTOCOL_AMENDMENT_1.json",
               "needs/PROTOCOL_AMENDMENT_PENDING.json", "needs/AMENDMENT_APPROVAL.json", "needs/DESIGN_CRITIQUE.json",
               "needs/receipts", "needs/ORACLE_CHECK.json", "code", "figures", "raw", "results.json", "paper"],
    "figures": ["figures", "code/web_plots.py", "needs/receipts/evidence_gate.json", "paper", "paper.pdf"],
    "crosscheck": ["needs/EVIDENCE.json", "needs/receipts/evidence_gate.json", "paper", "paper.pdf", "slides.md"],
    "evidence": ["needs/EVIDENCE.json", "needs/receipts/evidence_gate.json", "paper", "paper.pdf"],
    "claims": ["paper/claims.json", "paper/CLAIMS.md", "paper/numeric_audit.json", "needs/receipts/claim_check.json",
               "slides.pdf", "poster.pdf"],
}
STAYS = {
    "ideas": [".fi/literature_queries.json", ".fi/audit.jsonl", ".fi/approved_plan.json", "needs/DESIGN_HISTORY.json"],
    "literature": [".fi/literature_queries.json", ".fi/audit.jsonl", ".fi/approved_plan.json", "needs/DESIGN_HISTORY.json"],
    "plan": ["data/literature", ".fi/approved_plan.json", "needs/DESIGN_HISTORY.json"],
    "design": ["plan.md", "data/literature", ".fi/approved_plan.json", "needs/DESIGN_HISTORY.json"],
    "figures": ["code/experiment.py", "results.json", "raw", "plan.md", "needs/FROZEN_PROTOCOL.json"],
    "crosscheck": ["code/experiment.py", "results.json", "figures", "needs/FROZEN_PROTOCOL.json", "needs/ORACLE_CHECK.json"],
    "evidence": ["code/experiment.py", "results.json", "figures", "needs/FROZEN_PROTOCOL.json"],
    "claims": ["paper/paper.md", "paper.pdf", "paper.md", "needs/EVIDENCE.json", "needs/receipts/evidence_gate.json",
               "results.json", "figures"],
}
BYSTANDERS = [".fi/literature_queries.json", ".fi/audit.jsonl", ".fi/attempts.jsonl", ".fi/approved_plan.json",
              "needs/DESIGN_HISTORY.json", "needs/NUMERIC_WARNINGS.json"]


def _concrete(pattern: str) -> str:
    return pattern.replace("*", "1")


def _lay_out(root: Path) -> None:
    """A quest folder holding one file for everything any step could move, plus files that are nobody's to move."""
    names = {_concrete(p) for step in rerun_from.STEPS for p in rerun_from.OUTPUTS[step]}
    names |= {"code/experiment.py", "paper/paper.md", "paper/claims.json", "needs/receipts/design_audit.json",
              "needs/receipts/claim_check.json", "needs/receipts/evidence_gate.json", "data/literature/lit_001.md",
              "figures/a.png", "raw/ledger.jsonl", "needs/protocol_versions/v1.json", *BYSTANDERS}
    for name in sorted(names):
        path = root / name
        if path.exists():  # a folder already made for a file inside it
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if "." not in Path(name).name:  # a folder such as "code" or "figures": give it a file
            path.mkdir(exist_ok=True)
            (path / "x.txt").write_text("x", encoding="utf-8")
        else:
            path.write_text("x", encoding="utf-8")


def test_every_offered_step_has_its_words_its_files_and_a_block() -> None:
    grouped = [s for members in rerun_from.GROUPS.values() for s in members]
    assert sorted(grouped) == sorted(rerun_from.STEPS), "every step is in exactly one block of the map"
    for step in rerun_from.STEPS:
        assert rerun_from.REDOES[step].strip() and rerun_from.OUTPUTS[step], step
        info = rerun_from.step_info(step, reached=True)
        assert set(info) == {"name", "group", "sentence", "needs_approval", "reached", "outputs"}
        assert info["group"] and info["outputs"] == rerun_from.outputs_of(step)
    for step in ("ideas", "literature", "plan", "design"):
        assert rerun_from.needs_approval(step) and "frozen protocol is replaced" in rerun_from.REDOES[step]
    assert not any(rerun_from.needs_approval(s) for s in rerun_from.STEPS if s not in ("ideas", "literature", "plan", "design"))


def test_every_graph_node_is_a_step_or_says_why_not(tmp_path: Path) -> None:
    nodes = set(Engine(_cfg(tmp_path))._build_graph().nodes)
    covered = {n for nodes_ in rerun_from.STEPS.values() for n in nodes_}
    aliased = {n for n in nodes if rerun_from.resolve(n)}  # select_skills is the skills step, which forks at the code
    assert nodes - covered - aliased - NOT_A_STEP == set(), "a node with no step and no reason is a gap"
    assert not covered - nodes, "a step names a node the graph does not have"
    for node in NOT_A_STEP:
        assert rerun_from.resolve(node) is None


def test_the_raw_node_names_are_accepted_as_the_steps() -> None:
    for alias, step in {"ideate": "ideas", "cross_check": "crosscheck", "evidence_gate": "evidence", "claim_check": "claims",
                        "web_plots": "figures", "web_figures": "figures", "Plan": "plan", "literature": "literature"}.items():
        assert rerun_from.resolve(alias) == step


@pytest.mark.asyncio
@pytest.mark.parametrize("step", NEW_STEPS)
async def test_the_checkpoint_is_the_state_before_the_step(step: str) -> None:
    first = rerun_from.STEPS[step][0]
    others = ["ideate", "literature", "select_skills", "plan", "design", "implement_outline", "implement", "execute",
              "web_plots", "web_figures", "analyze", "cross_check", "evidence_gate", "write", "claim_check", "review"]
    order = [(n,) for n in others]
    found = await rerun_from.checkpoint_before(_Graph(order), {}, step)
    assert found is not None and int(found["configurable"]["checkpoint_id"]) == others.index(first)
    # A second pass (a redesign loop): the latest pass is the one that is redone.
    graph = _Graph([*order, *order[others.index("design"):]])
    found = await rerun_from.checkpoint_before(graph, {}, "design")
    assert int(found["configurable"]["checkpoint_id"]) == len(order)


@pytest.mark.asyncio
@pytest.mark.parametrize("step", NEW_STEPS)
async def test_a_step_the_quest_never_reached_is_not_offered(step: str) -> None:
    nodes = rerun_from.STEPS[step]
    everything = ["ideate", "literature", "select_skills", "plan", "design", "implement_outline", "implement", "execute",
                  "analyze", "cross_check", "evidence_gate", "write", "claim_check", "review"]
    without = [(n,) for n in everything if n not in nodes]
    assert step not in await rerun_from.reached(_Graph(without), {})
    assert await rerun_from.checkpoint_before(_Graph(without), {}, step) is None
    assert step in await rerun_from.reached(_Graph([*without, (nodes[0],)]), {})
    text = rerun_from.listing("q-1", await rerun_from.reached(_Graph(without), {}))
    assert f"  {step} " not in text


@pytest.mark.parametrize("step", NEW_STEPS)
def test_the_backup_moves_what_the_step_lists_and_nothing_else(tmp_path: Path, step: str) -> None:
    _lay_out(tmp_path)
    before = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    where, moved = rerun_from.back_up(tmp_path, step)
    assert where is not None
    after = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    kept = {p.relative_to(where).as_posix() for p in where.rglob("*") if p.is_file()}
    assert kept | {f for f in after if not f.startswith(".fi/previous/")} == before, "nothing lost, nothing made up"
    for rel in MOVES[step]:
        assert (where / rel).exists(), f"{rel} was not moved aside"
    for rel in STAYS[step]:
        assert (tmp_path / rel).exists() and not (where / rel).exists(), f"{rel} was moved but it is the step's input"
    for rel in BYSTANDERS:
        assert (tmp_path / rel).exists(), rel
    assert all((where / rel).exists() for rel in moved)


@pytest.mark.parametrize("step", NEW_STEPS)
def test_dropping_a_listed_file_from_a_step_would_be_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    """The pinned lists above are independent of the module's, so a dropped entry leaves a pinned file behind."""
    entry = MOVES[step][0]
    listed = [p for p in rerun_from.OUTPUTS[step] if p != entry]
    monkeypatch.setitem(rerun_from.OUTPUTS, step, listed)
    _lay_out(tmp_path)
    where, _ = rerun_from.back_up(tmp_path, step)
    assert where is not None and not (where / entry).exists(), "the drop is visible as a file that stayed"


def test_the_amendment_files_are_matched_by_pattern(tmp_path: Path) -> None:
    for n in (1, 2):
        (tmp_path / "needs").mkdir(exist_ok=True)
        (tmp_path / "needs" / f"PROTOCOL_AMENDMENT_{n}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "needs" / "PROTOCOL_AMENDMENTS_NOTE.txt").write_text("x", encoding="utf-8")
    _, moved = rerun_from.back_up(tmp_path, "design")
    assert sorted(moved) == ["needs/PROTOCOL_AMENDMENT_1.json", "needs/PROTOCOL_AMENDMENT_2.json"]


async def _first_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Engine, list[str]]:
    calls: list[str] = []

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        calls.append(_classify(prompt))
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    first = Engine(_cfg(tmp_path))
    await first.run()
    return first, calls


@pytest.mark.asyncio
async def test_a_restart_at_the_design_needs_a_name_and_replaces_the_protocol_only_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    first, calls = await _first_run(tmp_path, monkeypatch)
    cfg = _cfg(tmp_path)
    frozen = first.quest_root / "needs" / "FROZEN_PROTOCOL.json"
    assert frozen.is_file(), "the quest froze a protocol before its run"
    old = frozen.read_text(encoding="utf-8")
    history = json.loads((first.quest_root / "needs" / "DESIGN_HISTORY.json").read_text(encoding="utf-8"))
    capsys.readouterr()

    # Without a name: plain refusal, nothing moved.
    refused = Engine(cfg, resume_quest_id=first.quest_id)
    await refused.run(from_step="design")
    out = capsys.readouterr().out
    assert "replaces its plan and its frozen protocol" in out and "--approve-as <you>" in out
    assert frozen.read_text(encoding="utf-8") == old and not (refused.fi_dir / "previous").exists()

    # With a name: the old protocol is kept aside, the audit says who approved and which protocol was replaced.
    again = Engine(cfg, resume_quest_id=first.quest_id)
    await again.run(from_step="design", approved_by="alice")
    assert "the old plan and frozen protocol are replaced" in capsys.readouterr().out
    # (The design step takes the design block of plan.md, so it may need no model call; the log shows it ran again.)
    log = (again.fi_dir / "run.log").read_text(encoding="utf-8")
    assert log.count("[design] Designing the experiment.") == 2, "the design was made again"
    previous = list((again.fi_dir / "previous").iterdir())
    assert len(previous) == 1 and (previous[0] / "needs" / "FROZEN_PROTOCOL.json").read_text(encoding="utf-8") == old
    events = [json.loads(line) for line in (again.fi_dir / "audit.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    event = next(e for e in events if e.get("kind") == "rerun_from")
    text = json.dumps(event)
    assert "alice" in text and "replaced_protocol_sha256" in text
    # The first design stays on record; the new one is added after it as made after results were seen.
    after = json.loads((again.quest_root / "needs" / "DESIGN_HISTORY.json").read_text(encoding="utf-8"))
    assert after[: len(history)] == history and len(after) == len(history) + 1 and after[-1]["post_hoc"] is True


@pytest.mark.asyncio
async def test_a_later_step_is_redone_without_a_name_and_keeps_the_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, calls = await _first_run(tmp_path, monkeypatch)
    frozen = first.quest_root / "needs" / "FROZEN_PROTOCOL.json"
    old = frozen.read_text(encoding="utf-8")
    again = Engine(_cfg(tmp_path), resume_quest_id=first.quest_id)
    offered = {s["name"]: s for s in await again.rerun_steps()}
    assert offered["claims"]["reached"] and offered["crosscheck"]["reached"] and offered["design"]["needs_approval"]
    assert not offered["claims"]["needs_approval"] and offered["claims"]["outputs"]
    writes = calls.count("Writing")
    await again.run(from_step="claims")
    assert calls.count("Writing") == writes, "the claim check does not write the paper again"
    assert frozen.read_text(encoding="utf-8") == old


def test_from_a_step_before_the_design_needs_approve_as_on_the_command_line() -> None:
    import subprocess

    root = Path(__file__).resolve().parent.parent
    base = [sys.executable, "-B", "launch.py", "--config", "examples/integrator_bakeoff/config.yaml", "--resume", "q-1"]
    done = subprocess.run([*base, "--from", "plan"], cwd=root, capture_output=True, text=True, timeout=180)
    assert done.returncode == 2 and "--approve-as <you>" in done.stderr and "replaces" in done.stderr, done.stderr[-500:]
