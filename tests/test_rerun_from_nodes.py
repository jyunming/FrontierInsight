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
NOT_A_STEP = {"clarify", "pause_after_literature", "wait_for_data", "execute_reflect", "human_feedback", "replot_layout"}

# Pinned here, apart from the module, so that dropping an entry from OUTPUTS fails a test: what each new step moves,
# and what it leaves.
MOVES = {
    "ideas": ["data/literature", "plan.md", ".fi/paused_at_plan.flag", "needs/FROZEN_PROTOCOL.json", "code", "paper.md"],
    "literature": ["data/literature", "plan.md", "needs/FROZEN_PROTOCOL.json", "results.json", "slides.pdf"],
    "plan": ["plan.md", ".fi/paused_at_plan.flag", ".fi/oracles_added.json", "needs/FROZEN_PROTOCOL.json",
             "needs/receipts/design_audit.json", "code", "paper", ".fi/oracle_guidance.json", ".fi/oracle_dry_run.json"],
    "design": ["needs/FROZEN_PROTOCOL.json", ".fi/oracle_dry_run.json",
               "needs/PROTOCOL_AMENDMENT_PENDING.json", "needs/AMENDMENT_APPROVAL.json", "needs/DESIGN_CRITIQUE.json",
               "needs/receipts", "needs/ORACLE_CHECK.json", "code", "figures", "raw", "results.json", "paper"],
    "figures": ["figures", "code/web_plots.py", "needs/receipts/evidence_gate.json", "paper", "paper.pdf"],
    "crosscheck": ["needs/EVIDENCE.json", "needs/receipts/evidence_gate.json", "paper", "paper.pdf", "slides.md"],
    "evidence": ["needs/EVIDENCE.json", "needs/receipts/evidence_gate.json", "paper", "paper.pdf"],
    "claims": ["paper/claims.json", "paper/CLAIMS.md", "paper/numeric_audit.json", "needs/receipts/claim_check.json",
               "slides.pdf", "poster.pdf"],
}
STAYS = {
    "ideas": ["needs/protocol_versions/v1.json", "needs/PROTOCOL_AMENDMENT_1.json", ".fi/literature_queries.json", ".fi/audit.jsonl", ".fi/approved_plan.json", "needs/DESIGN_HISTORY.json"],
    "literature": ["needs/protocol_versions/v1.json", "needs/PROTOCOL_AMENDMENT_1.json", ".fi/literature_queries.json", ".fi/audit.jsonl", ".fi/approved_plan.json", "needs/DESIGN_HISTORY.json"],
    "plan": ["needs/protocol_versions/v1.json", "needs/PROTOCOL_AMENDMENT_1.json", "data/literature", ".fi/approved_plan.json", "needs/DESIGN_HISTORY.json"],
    "design": ["needs/protocol_versions/v1.json", "needs/PROTOCOL_AMENDMENT_1.json", "plan.md", ".fi/oracles_added.json", ".fi/oracle_guidance.json", "data/literature", ".fi/approved_plan.json", "needs/DESIGN_HISTORY.json"],
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
              "figures/a.png", "raw/ledger.jsonl", "needs/protocol_versions/v1.json", "needs/PROTOCOL_AMENDMENT_1.json",
              *BYSTANDERS}
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
        assert rerun_from.needs_approval(step) and "experiment plan (frozen) is replaced" in rerun_from.REDOES[step]
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


def test_the_amendments_and_saved_versions_are_the_record_and_stay(tmp_path: Path) -> None:
    (tmp_path / "needs" / "protocol_versions").mkdir(parents=True)
    for n in (1, 2):
        (tmp_path / "needs" / f"PROTOCOL_AMENDMENT_{n}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "needs" / "protocol_versions" / "v1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "needs" / "FROZEN_PROTOCOL.json").write_text("{}", encoding="utf-8")
    _, moved = rerun_from.back_up(tmp_path, "design")
    assert moved == ["needs/FROZEN_PROTOCOL.json"]
    assert (tmp_path / "needs" / "PROTOCOL_AMENDMENT_2.json").exists() and (tmp_path / "needs" / "protocol_versions" / "v1.json").exists()


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
    assert "replaces its plan.md and its experiment plan (frozen)" in out and "--approve-as <you>" in out
    assert frozen.read_text(encoding="utf-8") == old and not (refused.fi_dir / "previous").exists()

    # With a name: the old protocol is kept aside, the audit says who approved and which protocol was replaced.
    again = Engine(cfg, resume_quest_id=first.quest_id)
    await again.run(from_step="design", approved_by="alice")
    assert "the old plan.md and experiment plan (frozen) are replaced" in capsys.readouterr().out
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


def test_replacing_a_frozen_protocol_is_recorded_as_a_change_made_after_results_were_seen(tmp_path: Path) -> None:
    from core import frozen_protocol as fp

    assert fp.record_replacement(tmp_path, approved_by="alice", step="design") is None, "nothing frozen, nothing replaced"
    old = fp.freeze(tmp_path, {"trials": 3}, approved_by="engine", source="design")
    record = fp.record_replacement(tmp_path, approved_by="alice", step="design")
    assert record and record["approved_by"] == "alice" and record["results_seen_before_change"] is True
    assert record["from_sha256"] == old["sha256"] and record["prespecified"] is False
    assert fp.open_replacement(tmp_path) == record
    (tmp_path / "needs" / "FROZEN_PROTOCOL.json").unlink()  # what back_up does
    new = fp.freeze(tmp_path, {"trials": 9}, approved_by="alice", source="run again from design",
                    version=record["from_version"] + 1, amendments=record["n"])
    fp.close_replacement(tmp_path, record, new["sha256"])
    assert fp.open_replacement(tmp_path) is None
    assert new["version"] == 2 and new["amendments"] == 1 and new["run_id"] == "run_2"
    assert (tmp_path / "needs" / "protocol_versions" / "v1.json").is_file() and (tmp_path / "needs" / "protocol_versions" / "v2.json").is_file()
    assert [a["to_sha256"] for a in fp.amendments(tmp_path)] == [new["sha256"]]
    assert fp.post_hoc(tmp_path), "the evidence level can no longer say the plan was fixed before the results"


@pytest.mark.asyncio
async def test_a_restart_at_the_design_freezes_the_new_protocol_as_an_amendment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core import frozen_protocol as fp

    first, _ = await _first_run(tmp_path, monkeypatch)
    old = fp.load(first.quest_root)
    assert old is not None
    again = Engine(_cfg(tmp_path), resume_quest_id=first.quest_id)
    await again.run(from_step="design", approved_by="alice")
    new = fp.load(first.quest_root)
    assert new is not None and new["version"] == old["version"] + 1 and new["amendments"] == 1
    found = fp.amendments(first.quest_root)
    assert len(found) == 1 and found[0]["approved_by"] == "alice" and found[0]["from_sha256"] == old["sha256"]
    assert found[0]["to_sha256"] == new["sha256"] and fp.post_hoc(first.quest_root)


def test_the_console_line_after_a_run_has_its_space(capsys: pytest.CaptureFixture[str]) -> None:
    text = (Path(__file__).resolve().parent.parent / "launch.py").read_text(encoding="utf-8")
    assert 'print(f"[FI] {art.quest_id} -> {art.quest_root}")' in text


def test_the_web_page_builds_its_rerun_menu_from_the_steps_the_quest_reached() -> None:
    page = (Path(__file__).resolve().parent.parent / "web" / "static" / "quest.html").read_text(encoding="utf-8")
    assert "loadRerunSteps" in page and "needs your name" in page


def test_a_run_again_keeps_the_protocol_and_oracle_checks_but_a_code_restart_moves_them(tmp_path: Path) -> None:
    kept = ["needs/PROTOCOL_CHECK.json", "needs/ORACLE_CHECK.json"]
    for step, moves in (("run", False), ("code", True)):
        root = tmp_path / step
        _lay_out(root)
        where, _ = rerun_from.back_up(root, step)
        for rel in kept:
            assert (root / rel).exists() is not moves, f"{rel} with --from {step}"
            if moves:
                assert where is not None and (where / rel).exists()


def test_a_backup_that_stops_half_way_puts_back_what_it_had_moved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _lay_out(tmp_path)
    before = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    real, calls = rerun_from._move, {"n": 0}

    def flaky(src: Path, target: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise PermissionError("in use")
        real(src, target)

    monkeypatch.setattr(rerun_from, "_move", flaky)
    with pytest.raises(PermissionError):
        rerun_from.back_up(tmp_path, "design")
    after = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before, "every file is back where it was"


@pytest.mark.asyncio
async def test_after_a_rerun_the_steps_of_the_abandoned_line_are_not_the_quest_s() -> None:
    """A real LangGraph fork: the quest ran to the end, was run again from the design and stopped before the run. The
    old line's later steps are not offered, and no checkpoint of that line is a place to run again from."""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph

    def node(_state: dict) -> dict:
        return {}

    builder = StateGraph(dict)
    order = ["ideate", "plan", "design", "execute", "write", "review"]
    for name in order:
        builder.add_node(name, node)
    builder.add_edge(START, order[0])
    for a, b in zip(order, order[1:]):
        builder.add_edge(a, b)
    builder.add_edge(order[-1], END)
    graph = builder.compile(checkpointer=MemorySaver(), interrupt_before=["execute"])
    cfg = {"configurable": {"thread_id": "q"}}
    await graph.ainvoke({}, cfg)
    await graph.ainvoke(None, cfg)  # the first line goes on to the end
    assert {"run", "writing", "review"} <= set(await rerun_from.reached(graph, cfg))
    fork = await rerun_from.checkpoint_before(graph, cfg, "design")
    fork = await graph.aupdate_state(fork, {}, as_node="plan")
    await graph.ainvoke(None, fork)  # the new line stops before execute again
    steps = await rerun_from.reached(graph, cfg)
    assert "run" in steps and not {"writing", "review"} & set(steps)
    assert await rerun_from.checkpoint_before(graph, cfg, "writing") is None
    assert await rerun_from.checkpoint_before(graph, cfg, "review") is None
