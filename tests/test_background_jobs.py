"""A real quest whose experiment is a background job, against a fake cluster.

The fake cluster is a folder: the job is "done" once the test drops a file in it.
The experiment script is the idempotent driver the prompts ask for — first run
submits, later runs check, a finished job is collected — so what is tested is FI's
half of the contract: pause on a pending line, never submit twice, resume or
``--watch`` until it delivers. Real HPC is not exercised here."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core import job_watch as jw
from core.config import (
    Config, EngineConfig, ExecutionConfig, KnowledgeConfig, OutputConfig,
    ProviderConfig,
)
from core.engine import Engine
from tests.test_engine_smoke import _classify, _fake_response_for

# A phrase only the protocol contains: the topic itself says "background job".
MARKER = "execution.background_jobs is on"

_DRIVER = """\
import json, os, pathlib

cluster = pathlib.Path(os.environ["FAKE_CLUSTER_DIR"])
job = pathlib.Path("job"); job.mkdir(exist_ok=True)
runs = job / "runs.count"
runs.write_text(str(int(runs.read_text() or 0) + 1 if runs.exists() else 1))
state = job / "state.json"
if not state.exists():
    state.write_text(json.dumps({"id": "J1"}))
    (cluster / "submitted").write_text("1")
    print("RESULT_JSON: " + json.dumps({"fi_job": {"status": "pending", "id": "J1", "note": "queued", "poll_s": 5}}))
elif not (cluster / "done").exists():
    print("RESULT_JSON: " + json.dumps({"fi_job": {"status": "pending", "id": "J1", "note": "running"}}))
else:
    print("RESULT_JSON: " + json.dumps({"score": 0.5, "job": json.loads(state.read_text())["id"]}))
"""


@pytest.fixture()
def cluster(tmp_path: Path, monkeypatch) -> Path:
    d = tmp_path / "cluster"
    d.mkdir()
    monkeypatch.setenv("FAKE_CLUSTER_DIR", str(d))
    return d


def _config(tmp_path: Path, *, jobs: bool = True, replicates: int = 1, pilot: bool = False) -> Config:
    return Config(
        topic="background job probe", title="background-job",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(
            max_iterations=1, review_loop=False, auto_accept_on_pass=True,
            execute_replicates=replicates, pilot_run=pilot,
        ),
        execution=ExecutionConfig(sandbox="venv", timeout_s=120, background_jobs=jobs),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


@pytest.fixture()
def prompts(monkeypatch) -> dict[str, str]:
    seen: dict[str, str] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        tag = _classify(prompt)
        seen.setdefault(tag, prompt)
        if tag == "Implementation":
            return json.dumps({"code": _DRIVER, "deps": []})
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)
    return seen


def _runs(engine: Engine) -> int:
    return int((engine.quest_root / "job" / "runs.count").read_text())


@pytest.mark.asyncio
async def test_a_pending_job_pauses_the_quest_and_each_resume_checks_it_once(
    tmp_path: Path, cluster: Path, prompts,
) -> None:
    # replicates and the pilot are on: neither may run the script, or the job is submitted again.
    cfg = _config(tmp_path, replicates=3, pilot=True)

    first = Engine(cfg)
    await first.run()

    assert MARKER in prompts["Implementation"], "the contract reaches the code-writing prompt"
    assert MARKER in prompts["Experiment Design"]
    assert _runs(first) == 1, "no pilot and no replicates for a background job"
    pending = jw.read_pending(first.fi_dir)
    assert pending["job"]["id"] == "J1" and pending["checks"] == 1
    assert (first.quest_root / "NEXT_STEP.md").is_file()
    assert "--watch" in (first.quest_root / "NEXT_STEP.md").read_text(encoding="utf-8")
    assert json.loads((first.fi_dir / "pause.json").read_text(encoding="utf-8"))["kind"] == "results"
    assert not (first.quest_root / "paper" / "paper.md").exists(), "nothing is written before the results exist"
    log = (first.fi_dir / "run.log").read_text(encoding="utf-8")
    assert "[execute] the job is pending" in log

    # Resumed while the job is still running: pauses again, does not submit again.
    second = Engine(cfg, resume_quest_id=first.quest_id)
    await second.run()
    assert jw.read_pending(second.fi_dir)["checks"] == 2
    assert _runs(second) == 2
    assert (cluster / "submitted").read_text() == "1"

    # The job finishes; the next resume collects it and the quest completes.
    (cluster / "done").write_text("1")
    third = Engine(cfg, resume_quest_id=first.quest_id)
    artifacts = await third.run()

    assert artifacts.paper_md is not None and artifacts.paper_md.exists()
    assert artifacts.raw_state["result_json"] == {"score": 0.5, "job": "J1"}
    assert jw.read_pending(third.fi_dir) is None, "the wait is over"
    assert _runs(third) == 3 and (cluster / "submitted").read_text() == "1"
    assert "the background job has finished" in (third.fi_dir / "run.log").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_watch_re_checks_on_a_timer_and_the_quest_then_completes(
    tmp_path: Path, cluster: Path, prompts, monkeypatch,
) -> None:
    monkeypatch.setattr(jw, "MIN_POLL_S", 0)
    cfg = _config(tmp_path)
    first = Engine(cfg)
    await first.run()
    assert jw.read_pending(first.fi_dir) is not None

    async def finish_soon() -> None:
        await asyncio.sleep(1.6)
        (cluster / "done").write_text("1")

    lines: list[str] = []
    watcher = Engine(cfg, resume_quest_id=first.quest_id)
    finisher = asyncio.create_task(finish_soon())
    outcome = await jw.watch(
        watcher.poll_job, fi_dir=watcher.fi_dir, every_s=1, max_wait_s=60, out=lines.append,
    )
    await finisher

    assert outcome == jw.DONE
    checks = [ln for ln in lines if ln.startswith("[watch]")]
    assert len(checks) >= 3 and "pending - running" in checks[0] and "done" in checks[-1], checks

    artifacts = await Engine(cfg, resume_quest_id=first.quest_id).run()
    assert artifacts.paper_md is not None and artifacts.raw_state["result_json"]["job"] == "J1"
    assert (cluster / "submitted").read_text() == "1", "watching never submits a second job"


@pytest.mark.asyncio
async def test_a_failed_job_ends_the_watch_and_the_quest_repairs_or_stops_as_any_failure_would(
    tmp_path: Path, cluster: Path, prompts, monkeypatch,
) -> None:
    """A script that exits non-zero is a failure, not a pending job."""
    monkeypatch.setattr(jw, "MIN_POLL_S", 0)
    cfg = _config(tmp_path)
    first = Engine(cfg)
    await first.run()
    (first.quest_root / "code" / "experiment.py").write_text("raise SystemExit(3)\n", encoding="utf-8")

    outcome = await jw.watch(
        Engine(cfg, resume_quest_id=first.quest_id).poll_job,
        fi_dir=first.fi_dir, every_s=1, max_wait_s=30, out=lambda s: None,
    )
    assert outcome == jw.FAILED


@pytest.mark.asyncio
async def test_without_background_jobs_the_prompts_and_the_run_are_unchanged(
    tmp_path: Path, monkeypatch,
) -> None:
    seen: dict[str, str] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        seen.setdefault(_classify(prompt), prompt)
        return _fake_response_for(prompt)

    monkeypatch.setattr("core.engine.LLMClient.chat", fake_chat)

    artifacts = await Engine(_config(tmp_path, jobs=False)).run()

    assert MARKER not in seen["Implementation"] and MARKER not in seen["Experiment Design"]
    assert artifacts.paper_md is not None and artifacts.paper_md.exists()


# --- the CLI: launch.py --watch ------------------------------------------------


@pytest.mark.asyncio
async def test_cli_watch_prints_and_logs_every_check_then_resumes_the_quest(
    tmp_path: Path, cluster: Path, prompts, monkeypatch, capsys,
) -> None:
    import argparse

    import launch

    monkeypatch.setattr(jw, "MIN_POLL_S", 0)
    cfg = _config(tmp_path)
    first = Engine(cfg)
    await first.run()
    yaml_path = tmp_path / "quest.yaml"
    yaml_path.write_text("topic: x", encoding="utf-8")

    resumed: list[str] = []

    async def fake_run_one(cfg, **kw):  # noqa: ANN001
        resumed.append(kw["resume_quest_id"])
        return {}

    monkeypatch.setattr(launch, "run_one", fake_run_one)

    async def finish_soon() -> None:
        await asyncio.sleep(1.2)
        (cluster / "done").write_text("1")

    finisher = asyncio.create_task(finish_soon())
    args = argparse.Namespace(
        watch=first.quest_id, watch_every=1, watch_max_hours=0.0,
        config=yaml_path, auto_accept_on_pass=None,
    )
    rc = await launch._watch_quest(cfg, args, supervisor=None)
    await finisher

    assert rc == 0 and resumed == [first.quest_id]
    out = capsys.readouterr().out
    assert "is waiting on its job (id J1" in out and "[watch]" in out and "the job is done; resuming quest" in out
    log = (first.fi_dir / "run.log").read_text(encoding="utf-8")
    assert log.count("[watch]") >= 3, "every check is in the quest's run.log too"


@pytest.mark.asyncio
async def test_cli_watch_refuses_a_quest_that_is_not_waiting(tmp_path: Path, capsys) -> None:
    import argparse

    import launch

    cfg = _config(tmp_path)
    args = argparse.Namespace(
        watch="1700000000-nothing-here", watch_every=0, watch_max_hours=0.0,
        config=tmp_path / "x.yaml", auto_accept_on_pass=None,
    )
    assert await launch._watch_quest(cfg, args, supervisor=None) == 1
    assert "is not waiting on a job" in capsys.readouterr().err
