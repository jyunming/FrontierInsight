"""The quest index from the command line: a quest is found by its (short) id from any folder, and runs as if from its
own. Every test uses a temporary ``FI_HOME`` (tests/conftest.py), never the real ``~/.frontier-insight``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import launch
from core import commission, quest_index
from tests import commissioned

REPO = Path(__file__).resolve().parent.parent
QID = "1790003131-energy-drift-479b06"


def _study(tmp_path: Path, qid: str = QID, title: str = "Energy drift") -> Path:
    """A quest started in ``study_a`` from its commission (its YAML's output_dir is relative, as `fi new` writes it)."""
    folder = tmp_path / "study_a"
    root = folder / "outputs" / qid
    (root / ".fi").mkdir(parents=True)
    (root / ".fi" / "state.sqlite").write_bytes(b"")
    (root / "config.yaml").write_text(
        f"{commissioned.line(folder)}title: {title}\nprovider:\n  name: openai\noutput:\n  output_dir: ./outputs\n",
        encoding="utf-8",
    )
    commission.stage(folder / "commission.yaml", root)
    quest_index.register(root, working_folder=folder, title=title)
    return root


@pytest.fixture()
def elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    other = tmp_path / "study_b"
    (other / "outputs").mkdir(parents=True)
    monkeypatch.chdir(other)
    return other


def test_resume_by_short_id_from_another_folder_uses_the_quests_own_config(tmp_path: Path, elsewhere: Path) -> None:
    root = _study(tmp_path)
    args = launch.parse_args(["--resume", "479b06"])
    assert Path(args.config) == root.resolve() / "config.yaml"
    assert args.resume == QID


def test_an_ambiguous_short_id_stops_with_the_candidates(tmp_path: Path, elsewhere: Path,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    _study(tmp_path, "1790000001-first-c0ffee", "First")
    _study(tmp_path, "1790000002-second-c0ffee", "Second")
    with pytest.raises(SystemExit):
        launch.parse_args(["--resume", "c0ffee"])
    err = capsys.readouterr().err
    assert "more than one quest" in err and "1790000001-first-c0ffee" in err and "Second" in err


async def test_the_resumed_quest_runs_in_its_own_outputs_folder_not_this_one(
    tmp_path: Path, elsewhere: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _study(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen: dict = {}

    async def fake_run_one(cfg, **kw):  # noqa: ANN001, ANN003
        seen["output_dir"] = Path(cfg.output.output_dir)
        seen["resume"] = kw.get("resume_quest_id")
        return {}

    monkeypatch.setattr(launch, "run_one", fake_run_one)
    args = launch.parse_args(["--resume", "479b06", "--no-axon-sidecar"])
    assert await launch.main_async(args) == 0
    assert seen["resume"] == QID
    assert seen["output_dir"].resolve() == root.parent.resolve()


async def test_an_explicit_config_with_a_short_id_finds_the_quest_through_the_index(
    tmp_path: Path, elsewhere: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _study(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mine = elsewhere / "mine.yaml"
    mine.write_text(f"{commissioned.line(tmp_path / 'study_a')}provider:\n  name: openai\noutput:\n  output_dir: ./outputs\n",
                    encoding="utf-8")
    seen: dict = {}

    async def fake_run_one(cfg, **kw):  # noqa: ANN001, ANN003
        seen["output_dir"] = Path(cfg.output.output_dir)
        seen["resume"] = kw.get("resume_quest_id")
        return {}

    monkeypatch.setattr(launch, "run_one", fake_run_one)
    args = launch.parse_args(["--config", str(mine), "--resume", "479b06", "--no-axon-sidecar"])
    assert await launch.main_async(args) == 0
    assert seen["resume"] == QID and seen["output_dir"].resolve() == root.parent.resolve()
    # The YAML given is the person's own: its relative paths mean this folder, so FI stays here.
    assert Path.cwd().resolve() == elsewhere.resolve()


async def test_run_one_records_the_quest_in_the_index_with_its_folder_and_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.config import Config, OutputConfig, ProviderConfig

    monkeypatch.chdir(tmp_path)
    cfg = Config(topic="t", title="A title", provider=ProviderConfig(name="openai"),
                 output=OutputConfig(output_dir=tmp_path / "outputs"))
    src = tmp_path / "q.yaml"
    src.write_text("topic: t\ntitle: A title\n", encoding="utf-8")
    engine = MagicMock()
    engine.quest_id = "1790000009-recorded-9a9a9a"
    engine.quest_root = tmp_path / "outputs" / engine.quest_id
    monkeypatch.setattr(launch, "Engine", lambda *_a, **_kw: engine)
    art = MagicMock()
    art.quest_id, art.quest_root, art.paper_md = engine.quest_id, engine.quest_root, None

    async def fake_maybe(*_a, **_kw):  # noqa: ANN001, ANN003
        (engine.quest_root / ".fi").mkdir(parents=True, exist_ok=True)
        return art

    monkeypatch.setattr(launch, "_maybe_profiled", fake_maybe)
    monkeypatch.setattr(launch, "_pick_clarify_callback", lambda *a, **kw: None)
    monkeypatch.setattr(launch, "_run_generators", AsyncMock(return_value={}))
    await launch.run_one(cfg, supervisor=MagicMock(), source_yaml_path=src)
    entry = quest_index.load()[engine.quest_id]
    assert entry["quest_root"] == str(engine.quest_root.resolve())
    assert entry["config"] == str(engine.quest_root.resolve() / "config.yaml")
    assert entry["working_folder"] == str(tmp_path.resolve())
    assert entry["title"] == "A title"


async def test_a_quest_found_elsewhere_runs_from_the_folder_it_was_started_in(
    tmp_path: Path, elsewhere: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the web page's launch paths do: --config <its yaml> --output <this server's outputs> <full id>."""
    root = _study(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen: dict = {}

    async def fake_run_one(cfg, **kw):  # noqa: ANN001, ANN003
        seen["output_dir"] = Path(cfg.output.output_dir)
        seen["cwd"] = Path.cwd()
        seen["yaml"] = kw.get("source_yaml_path")
        return {}

    monkeypatch.setattr(launch, "run_one", fake_run_one)
    args = launch.parse_args(["--config", str(root / "config.yaml"), "--resume", QID,
                              "--output", str(elsewhere / "outputs"), "--no-axon-sidecar"])
    assert await launch.main_async(args) == 0
    assert seen["output_dir"].resolve() == root.parent.resolve()
    assert seen["cwd"].resolve() == (tmp_path / "study_a").resolve()
    assert Path(seen["yaml"]) == (root / "config.yaml").resolve()


def test_trace_takes_a_short_id_from_another_folder(tmp_path: Path, elsewhere: Path,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    root = _study(tmp_path)
    from core import audit_log

    audit_log.AuditLog(root / ".fi" / "audit.jsonl", QID).append("node_started", node="clarify")
    assert launch._show_trace("479b06", "", "summary", Path("outputs")) in (0, 1)
    assert "no quest" not in capsys.readouterr().out


def _cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "FI_SKIP_BOOTSTRAP": "1"}
    return subprocess.run([sys.executable, str(REPO / "launch.py"), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=180)


def test_fi_tools_quests_lists_every_quest_with_its_short_id_from_any_folder(tmp_path: Path) -> None:
    root = _study(tmp_path)
    other = tmp_path / "study_b"
    other.mkdir()
    done = _cli(["tools", "quests"], other)
    assert done.returncode == 0, done.stderr
    assert "479b06" in done.stdout and "Energy drift" in done.stdout and str(root.resolve()) in done.stdout


def test_fi_tools_quests_prune_drops_quests_whose_folder_is_gone(tmp_path: Path) -> None:
    import shutil

    root = _study(tmp_path)
    shutil.rmtree(root)
    done = _cli(["tools", "quests", "--prune"], tmp_path)
    assert done.returncode == 0, done.stderr
    assert QID in done.stdout
    assert quest_index.load() == {}


def test_rename_by_short_id_from_another_folder_changes_the_title_there_and_in_the_index(tmp_path: Path) -> None:
    root = _study(tmp_path)
    other = tmp_path / "study_b"
    other.mkdir()
    done = _cli(["tools", "rename", "479b06", "Energy", "drift", "revisited"], other)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Energy drift revisited" in (root / "config.yaml").read_text(encoding="utf-8")
    assert quest_index.load()[QID]["title"] == "Energy drift revisited"


def _own_folder_quest(folder: Path, qid: str) -> Path:
    root = folder / "outputs" / qid
    (root / ".fi").mkdir(parents=True)
    return root


def test_the_quest_list_puts_this_folders_quests_first_then_every_other_one(tmp_path: Path) -> None:
    """The VS Code extension offered other folders' quests only when its own folder had none, so a round with one
    folder per study showed one study of three. ``listed`` is the one list the web page and the extension show."""
    a = _own_folder_quest(tmp_path / "study_a", "1790000001-first-aaaaaa")
    b = _own_folder_quest(tmp_path / "study_b", "1790000002-second-bbbbbb")
    c = _own_folder_quest(tmp_path / "study_c", "1790000003-third-cccccc")
    for root, folder in ((a, "study_a"), (b, "study_b"), (c, "study_c")):
        quest_index.register(root, working_folder=tmp_path / folder, title=root.name)
    # a quest in this folder the index never heard of is listed too
    unknown = _own_folder_quest(tmp_path / "study_b", "1790000009-old-dddddd")
    got = [e.quest_id for e in quest_index.listed(b.parent)]
    assert got[:2] == ["1790000009-old-dddddd", "1790000002-second-bbbbbb"]
    assert sorted(got[2:]) == ["1790000001-first-aaaaaa", "1790000003-third-cccccc"]
    assert len(got) == len(set(got)) == 4 and unknown.is_dir()
    # no folder given (or one that is not there): the index alone, as before
    assert len(quest_index.listed(None)) == 3
    assert len(quest_index.listed(tmp_path / "nowhere")) == 3


def test_the_quest_list_json_carries_each_quests_running_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The extension reads this and offers no second run of a running quest; the status is Python's own rule."""
    import json
    import os
    import time

    psutil = pytest.importorskip("psutil")
    a = _own_folder_quest(tmp_path / "study_a", "1790000001-first-aaaaaa")
    b = _own_folder_quest(tmp_path / "study_b", "1790000002-second-bbbbbb")
    quest_index.register(a, working_folder=tmp_path / "study_a", title="First")
    quest_index.register(b, working_folder=tmp_path / "study_b", title="Second")
    me = {"pid": os.getpid(), "started": psutil.Process(os.getpid()).create_time(), "seen_at": time.time(), "ended": None}
    (a / ".fi" / "process.json").write_text(json.dumps(me), encoding="utf-8")
    assert launch._list_quests(as_json=True, output_root=b.parent) == 0
    rows = json.loads(capsys.readouterr().out)["quests"]
    assert [r["quest_id"] for r in rows] == ["1790000002-second-bbbbbb", "1790000001-first-aaaaaa"]
    assert [r["status"] for r in rows] == ["stopped", "running"]


def test_the_quest_list_has_one_line_per_physical_folder(tmp_path: Path) -> None:
    """Two index ids on one folder, a letter-case variant of it, or this folder's own quest recorded again, are one
    quest in the list: the first is kept."""
    a = _own_folder_quest(tmp_path / "study_a", "1790000001-first-aaaaaa")
    quest_index.register(a, working_folder=tmp_path / "study_a", title="First")
    other = _own_folder_quest(tmp_path / "study_b", "1790000002-second-bbbbbb")
    quest_index.register(other, working_folder=tmp_path / "study_b", title="Second")
    # the same folder under a second id
    path = quest_index.index_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    twin = dict(data["quests"]["1790000001-first-aaaaaa"])
    data["quests"]["1790000005-twin-eeeeee"] = twin
    path.write_text(json.dumps(data), encoding="utf-8")
    got = [e.quest_id for e in quest_index.listed(other.parent)]
    assert len(got) == 2 and got[0] == "1790000002-second-bbbbbb"
    assert sum(g in ("1790000001-first-aaaaaa", "1790000005-twin-eeeeee") for g in got) == 1


def test_the_web_page_lists_quests_in_the_same_order_as_the_shared_list(tmp_path: Path) -> None:
    """Two quests elsewhere whose most recently seen one has the smaller id: the page and `fi tools quests` agree."""
    from fastapi.testclient import TestClient

    from web.server import make_app

    mine = _own_folder_quest(tmp_path / "study_a", "1790000003-mine-cccccc")
    old = _own_folder_quest(tmp_path / "study_b", "1790000009-zz-old-ffffff")
    new = _own_folder_quest(tmp_path / "study_c", "1790000001-aa-new-111111")
    for root, folder in ((mine, "study_a"), (old, "study_b"), (new, "study_c")):
        quest_index.register(root, working_folder=tmp_path / folder, title=root.name)
    path = quest_index.index_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    data["quests"]["1790000009-zz-old-ffffff"]["last_seen"] = "2026-01-01T00:00:00"
    data["quests"]["1790000001-aa-new-111111"]["last_seen"] = "2026-06-01T00:00:00"
    path.write_text(json.dumps(data), encoding="utf-8")
    shared = [e.quest_id for e in quest_index.listed(mine.parent)]
    page = [q["quest_id"] for q in TestClient(make_app(mine.parent)).get("/api/quests").json()["quests"]]
    assert page == shared and shared[-2:] == ["1790000001-aa-new-111111", "1790000009-zz-old-ffffff"]
