"""The coding agent's tripwire must not read FI's own writes (its heartbeat, logs, cost and model-call records) as the
agent's, while a real agent write elsewhere in the quest is still caught."""
from __future__ import annotations

from pathlib import Path

from core import agent_coder as ac
from core import run_marker


def _coder(quest: Path) -> ac.AgentCoder:
    return ac.AgentCoder(quest, object(), timeout_s=60)


def _quest(tmp_path: Path) -> Path:
    q = tmp_path / "q"
    (q / "code").mkdir(parents=True)
    (q / ".fi").mkdir()
    (q / "plan.md").write_text("plan", encoding="utf-8")
    (q / "results").mkdir()
    (q / "results" / "r.json").write_text("{}", encoding="utf-8")
    (q / ".fi" / "process.json").write_text('{"beat": 1}', encoding="utf-8")
    (q / ".fi" / "run.log").write_text("a\n", encoding="utf-8")
    (q / ".fi" / "progress.log").write_text("a\n", encoding="utf-8")
    return q


def _round(coder: ac.AgentCoder, q: Path, agent_writes) -> list[str]:
    before = coder._outside(with_code=True)
    # what FI itself writes while the agent works
    (q / run_marker.RECORD).write_text('{"beat": 2}', encoding="utf-8")
    with (q / run_marker.HISTORY).open("a", encoding="utf-8") as f:
        f.write("{}\n")
    for name in ("run.log", "progress.log", "model_calls.jsonl", "cost.jsonl", "cost.summary.json",
                 "record_heads.sqlite-wal"):
        with (q / ".fi" / name).open("a", encoding="utf-8") as f:
            f.write("x\n")
    agent_writes(q)
    return coder._restore_outside(before, with_code=True)


def test_fis_own_writes_during_the_round_do_not_trip_it(tmp_path: Path) -> None:
    q = _quest(tmp_path)
    assert _round(_coder(q), q, lambda q: None) == []
    assert (q / run_marker.RECORD).read_text(encoding="utf-8") == '{"beat": 2}', "FI's marker is not rolled back"


def test_an_agent_write_to_the_plan_still_trips_it(tmp_path: Path) -> None:
    q = _quest(tmp_path)
    cheats = _round(_coder(q), q, lambda q: (q / "plan.md").write_text("changed", encoding="utf-8"))
    assert any("plan.md" in c for c in cheats)


def test_an_agent_write_to_results_or_an_unknown_fi_file_still_trips_it(tmp_path: Path) -> None:
    q = _quest(tmp_path)

    def agent(q: Path) -> None:
        (q / "results" / "r.json").write_text('{"a": 1}', encoding="utf-8")
        (q / ".fi" / "agent_notes.json").write_text("{}", encoding="utf-8")

    cheats = _round(_coder(q), q, agent)
    assert any("r.json" in c for c in cheats) and any("agent_notes.json" in c for c in cheats)
