"""code/ is a project that runs on its own: README, requirements, run.py (and study.json for the trial contract)."""
import json
import subprocess
import sys

from core import attempt_records, code_project, trial_runner

SIMULATE = '''
def run_trial(cell, trial_id, seed):
    return {"seed": float(seed), "x": float(cell["n"]) * 2}
'''
ANALYSIS = '''
import json, os
data = json.load(open(os.environ["FI_TRIALS"]))
seeds = {c["key"]: c["metrics"]["seed"]["values"] for c in data["cells"]}
print("RESULT_JSON: " + json.dumps({"seeds": seeds, "raw": os.environ["FI_RAW_DIR"]}))
open("out.txt", "w").write("done")
'''
PROTOCOL = {"grid": {"n": [1, 2]}, "runs_per_setting": 3, "thresholds": {}}


def _quest(tmp_path, *, two=True):
    root = tmp_path / "q"
    (root / "code").mkdir(parents=True)
    (root / "code" / "experiment.py").write_text(ANALYSIS if two else "print('hi')\n", encoding="utf-8")
    if two:
        (root / "code" / "simulate.py").write_text(SIMULATE, encoding="utf-8")
    return root


def _run(root):
    return subprocess.run([sys.executable, "run.py"], cwd=root / "code", capture_output=True, text=True, timeout=120)


def test_writes_project_files_and_run_py_matches_fi_seeds(tmp_path):
    root = _quest(tmp_path)
    written = code_project.refresh(root, deps=["numpy"], protocol=PROTOCOL, title="T", question="Does n matter?")
    assert set(written) == {"README.md", "requirements.txt", "run.py", "study.json"}
    readme = (root / "code" / "README.md").read_text(encoding="utf-8")
    assert "python run.py" in readme and "Does n matter?" in readme and "simulate.py" in readme
    done = _run(root)
    assert done.returncode == 0, done.stderr
    assert (root / "run_output" / "out.txt").read_text() == "done"
    assert not (root / "code" / "out.txt").exists() and not (root / "raw").exists()
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT_JSON: ")][-1]
    result = json.loads(line[len("RESULT_JSON: "):])
    assert result["raw"] == "raw"
    for key, seeds in result["seeds"].items():
        assert seeds == [float(trial_runner.trial_seed(0, key, t)) for t in range(3)]


def test_idempotent_and_follows_the_scripts(tmp_path):
    root = _quest(tmp_path)
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    assert code_project.refresh(root, deps=[], protocol=PROTOCOL) == []
    code_project.refresh(root, deps=[], protocol={**PROTOCOL, "runs_per_setting": 5})
    assert json.loads((root / "code" / "study.json").read_text())["runs_per_setting"] == 5


def test_a_file_the_person_edited_is_left_alone(tmp_path):
    root = _quest(tmp_path)
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    readme = root / "code" / "README.md"
    readme.write_text("my own notes\n", encoding="utf-8")
    written = code_project.refresh(root, deps=["scipy"], protocol=PROTOCOL, title="New title")
    assert "README.md" not in written and readme.read_text() == "my own notes\n"
    assert "scipy" in (root / "code" / "requirements.txt").read_text()


def test_a_file_fi_never_wrote_is_left_alone(tmp_path):
    root = _quest(tmp_path, two=False)
    (root / "code" / "README.md").write_text("theirs\n", encoding="utf-8")
    written = code_project.refresh(root, deps=[], protocol=None)
    assert "README.md" not in written and (root / "code" / "README.md").read_text() == "theirs\n"
    assert {"requirements.txt", "run.py"} <= set(written) and not (root / "code" / "study.json").exists()


def test_one_script_quest_runs_alone(tmp_path):
    root = _quest(tmp_path, two=False)
    code_project.refresh(root, deps=[], protocol=None)
    done = _run(root)
    assert done.returncode == 0 and "hi" in done.stdout


def test_requirements_add_imported_packages_the_list_lacks(tmp_path):
    root = _quest(tmp_path, two=False)
    (root / "code" / "experiment.py").write_text(
        "import json, helper\nimport numpy as np\nfrom sklearn.linear_model import Ridge\n"
        "from mpl_toolkits.mplot3d import Axes3D\ntry:\n    import fancy\nexcept ImportError:\n    fancy = None\n",
        encoding="utf-8")
    (root / "code" / "helper.py").write_text("x = 1\n", encoding="utf-8")
    code_project.refresh(root, deps=["numpy>=1.24"], protocol=None)
    lines = (root / "code" / "requirements.txt").read_text().split()
    assert lines == ["numpy>=1.24", "scikit-learn"]


def test_requirements_start_from_the_installed_list_and_still_add_what_the_scripts_import(tmp_path):
    root = _quest(tmp_path, two=False)
    (root / "code" / "experiment.py").write_text("import numpy\nimport scipy\n", encoding="utf-8")
    code_project.record_installed(root, ["numpy==1.0"])
    code_project.refresh(root, deps=["made-up-name"], protocol=None)
    assert (root / "code" / "requirements.txt").read_text().split() == ["numpy==1.0", "scipy"]


def test_study_json_removed_when_the_simulation_leaves_the_trial_contract(tmp_path):
    root = _quest(tmp_path)
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    (root / "code" / "simulate.py").write_text("print('old style')\n", encoding="utf-8")
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    assert not (root / "code" / "study.json").exists()


def test_split_off_run_py_does_not_run_the_simulation(tmp_path):
    root = _quest(tmp_path)
    (root / "code" / "simulate.py").write_text("open('simulated.txt', 'w').write('x')\n", encoding="utf-8")
    (root / "code" / "experiment.py").write_text("print('analysis only')\n", encoding="utf-8")
    written = code_project.refresh(root, deps=[], protocol=PROTOCOL, split=False)
    assert json.loads((root / "code" / "study.json").read_text()) == {"simulate": False} and "study.json" in written
    done = _run(root)
    assert done.returncode == 0 and "analysis only" in done.stdout, done.stderr
    assert not (root / "run_output" / "simulated.txt").exists()
    assert "It runs `experiment.py`." in (root / "code" / "README.md").read_text()


def test_nothing_is_written_without_an_experiment(tmp_path):
    (tmp_path / "q" / "code").mkdir(parents=True)
    assert code_project.refresh(tmp_path / "q", deps=[], protocol=None) == []


def test_run_py_reads_the_quests_data_and_leaves_its_results_alone(tmp_path):
    root = _quest(tmp_path, two=False)
    (root / "data").mkdir()
    (root / "data" / "in.txt").write_text("hello", encoding="utf-8")
    (root / "code" / "experiment.py").write_text("print(open('data/in.txt').read())\n", encoding="utf-8")
    code_project.refresh(root, deps=[], protocol=None)
    done = _run(root)
    assert done.returncode == 0 and "hello" in done.stdout, done.stderr
    assert (root / "run_output" / "data" / "in.txt").is_file()


def test_a_failed_trial_leaves_no_values_and_is_counted(tmp_path):
    root = _quest(tmp_path)
    (root / "code" / "simulate.py").write_text(
        "def run_trial(cell, trial_id, seed):\n"
        "    if trial_id == 1:\n        return {'x': True}\n"
        "    return {'x': 1.0, 'y': 2.0}\n", encoding="utf-8")
    (root / "code" / "experiment.py").write_text(
        "import json\nd = json.load(open('raw/trials.json'))\nprint(json.dumps(d['cells'][0]))\n", encoding="utf-8")
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    done = _run(root)
    cell = json.loads(done.stdout.strip().splitlines()[-1])
    assert cell["ok"] == 2 and cell["failed"] == 1
    assert cell["metrics"]["x"]["values"] == [1.0, 1.0] and cell["metrics"]["y"]["count"] == 2


def test_generated_files_do_not_count_as_code_unless_edited(tmp_path):
    root = _quest(tmp_path)
    before = attempt_records.script_hashes(root)
    code_project.refresh(root, deps=["numpy"], protocol=PROTOCOL)
    assert set(attempt_records.script_hashes(root)) == set(before) | {"study.json"}
    (root / "code" / "README.md").write_text("mine\n", encoding="utf-8")
    assert "README.md" in attempt_records.script_hashes(root)


def test_a_lost_record_leaves_existing_files_alone(tmp_path):
    root = _quest(tmp_path)
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    (root / ".fi" / "code_project.json").write_text("{not json", encoding="utf-8")
    (root / "code" / "README.md").write_text("mine\n", encoding="utf-8")
    assert "README.md" not in code_project.refresh(root, deps=[], protocol=PROTOCOL)
    assert (root / "code" / "README.md").read_text() == "mine\n"


def test_a_record_write_failure_is_reported(tmp_path, monkeypatch, caplog):
    import logging

    root = _quest(tmp_path)
    (root / ".fi").mkdir()
    (root / ".fi" / "code_project.json").mkdir()  # a folder where the record file goes
    log = logging.getLogger("t")
    with caplog.at_level(logging.WARNING, logger="t"):
        code_project.refresh(root, deps=[], protocol=PROTOCOL, log=log)
    assert "code_project.json" in caplog.text


def test_pin_uses_installed_versions_and_keeps_what_it_cannot_resolve():
    import importlib.metadata as md

    out = code_project.pin(sys.executable, ["pytest", "not-a-real-package-xyz", "numpy>=1 ; python_version>'3'"])
    assert out[0] == f"pytest=={md.version('pytest')}"
    assert out[1] == "not-a-real-package-xyz" and out[2].startswith("numpy>=1")


def test_requirements_use_the_pinned_installed_list(tmp_path):
    root = _quest(tmp_path)
    code_project.record_installed(root, ["numpy==1.2.3"])
    code_project.refresh(root, deps=["numpy"], protocol=PROTOCOL)
    assert (root / "code" / "requirements.txt").read_text().split() == ["numpy==1.2.3"]


def _git_available():
    import shutil

    return shutil.which("git") is not None


def test_each_change_is_one_commit_and_one_changelog_entry(tmp_path):
    import pytest

    if not _git_available():
        pytest.skip("git not installed")
    root = _quest(tmp_path)
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    assert code_project.record_change(root, "code written") is True
    assert code_project.record_change(root, "again, nothing changed") is False
    (root / "code" / "experiment.py").write_text("print('v2')\n", encoding="utf-8")
    assert code_project.record_change(root, "added what a refine asked for: X") is True
    log = subprocess.run(["git", "log", "--format=%s"], cwd=root / "code", capture_output=True, text=True).stdout.split("\n")
    assert [ln for ln in log if ln] == ["added what a refine asked for: X", "code written"]
    text = (root / "code" / "CHANGELOG.md").read_text()
    assert text.count("## ") == 2 and "experiment.py" in text
    first = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=root / "code",
                           capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "checkout", first, "--", "experiment.py"], cwd=root / "code", check=True, capture_output=True)
    assert "hi" in (root / "code" / "experiment.py").read_text() or "seeds" in (root / "code" / "experiment.py").read_text()


def test_history_and_changelog_are_not_code_for_the_attempt_hash(tmp_path):
    import pytest

    if not _git_available():
        pytest.skip("git not installed")
    root = _quest(tmp_path)
    before = attempt_records.script_hashes(root)
    code_project.record_change(root, "code written")
    assert attempt_records.script_hashes(root) == before


def test_an_edited_file_is_asked_about_once(tmp_path):
    root = _quest(tmp_path)
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    assert code_project.unasked_conflicts(root) == []
    (root / "code" / "run.py").write_text("# mine\n", encoding="utf-8")
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    assert code_project.unasked_conflicts(root) == ["run.py"]
    code_project.mark_asked(root, ["run.py"])
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    assert code_project.unasked_conflicts(root) == []
    assert (root / "code" / "run.py").read_text() == "# mine\n"
    (root / "code" / "run.py").unlink()  # deleting it lets FI write its own again
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    assert (root / "code" / "run.py").read_text().startswith('"""Runs this study')
    assert code_project.unasked_conflicts(root) == []


def test_verify_says_it_ran_in_a_clean_environment(tmp_path):
    root = _quest(tmp_path, two=False)
    code_project.refresh(root, deps=[], protocol=None)
    result = code_project.verify(root)
    assert result["ok"] is True, result
    assert json.loads((root / "needs" / "CODE_PROJECT_CHECK.json").read_text())["ok"] is True
    assert not (root / ".fi" / "project_check").exists()


def test_verify_warns_in_plain_words_when_it_does_not_run(tmp_path):
    root = _quest(tmp_path, two=False)
    (root / "code" / "experiment.py").write_text("raise SystemExit('boom in the script')\n", encoding="utf-8")
    code_project.refresh(root, deps=[], protocol=None)
    result = code_project.verify(root)
    assert result["ok"] is False and "did NOT run" in result["says"] and "boom in the script" in result["says"]


def test_the_web_zip_leaves_out_run_output(tmp_path):
    import io
    import zipfile

    __import__("pytest").importorskip("fastapi")
    from fastapi.testclient import TestClient

    from web.server import make_app

    out = tmp_path / "out"
    q = out / "q1"
    (q / "code").mkdir(parents=True)
    (q / "code" / "run.py").write_text("x", encoding="utf-8")
    (q / "run_output").mkdir()
    (q / "run_output" / "big.bin").write_text("y", encoding="utf-8")
    body = TestClient(make_app(out)).get("/api/quests/q1/download")
    names = zipfile.ZipFile(io.BytesIO(body.content)).namelist()
    assert "code/run.py" in names and not any(n.startswith("run_output") for n in names)


def _engine_with_an_edited_file(tmp_path, *, ask):
    from core.config import Config, ExecutionConfig, KnowledgeConfig, OutputConfig, PausesConfig, ProviderConfig
    from core.engine import Engine

    cfg = Config(
        topic="t", title="t", provider=ProviderConfig(name="openai"),
        execution=ExecutionConfig(sandbox="venv", timeout_s=60), knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
        pauses=PausesConfig(review="ask" if ask else "off"),
    )
    engine = Engine(cfg)
    engine.quest_root.mkdir(parents=True, exist_ok=True)
    (engine.quest_root / "code").mkdir(exist_ok=True)
    (engine.quest_root / "code" / "experiment.py").write_text("print('x')\n", encoding="utf-8")
    code_project.refresh(engine.quest_root, deps=[], protocol=None)
    (engine.quest_root / "code" / "README.md").write_text("mine\n", encoding="utf-8")
    code_project.refresh(engine.quest_root, deps=[], protocol=None)
    return engine


def test_an_interactive_quest_pauses_once_before_touching_an_edited_file(tmp_path):
    engine = _engine_with_an_edited_file(tmp_path, ask=True)
    seen = []
    engine._pause_for_human = lambda **kw: seen.append(kw)
    engine._ask_about_edited_project_files()
    assert len(seen) == 1 and seen[0]["kind"] == "code_project" and seen[0]["payload"]["files"] == ["README.md"]
    engine._ask_about_edited_project_files()  # resumed: the same file is not asked about again
    assert len(seen) == 1
    assert (engine.quest_root / "code" / "README.md").read_text() == "mine\n"


def test_a_quest_that_does_not_ask_keeps_the_file_and_says_so(tmp_path):
    engine = _engine_with_an_edited_file(tmp_path, ask=False)
    engine._pause_for_human = lambda **kw: (_ for _ in ()).throw(AssertionError("must not pause"))
    said = []
    engine._log = type("L", (), {"warning": lambda self, msg, *a: said.append(msg % a), "info": lambda *a, **k: None})()
    engine._ask_about_edited_project_files()
    assert any("code/README.md" in m for m in said)
    assert (engine.quest_root / "code" / "README.md").read_text() == "mine\n"


def test_rerunning_from_code_keeps_the_projects_history(tmp_path):
    import pytest

    from core import rerun_from

    if not _git_available():
        pytest.skip("git not installed")
    root = _quest(tmp_path)
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    code_project.record_change(root, "code written")
    rerun_from.back_up(root, "code")
    assert not (root / "code" / "experiment.py").exists()
    log = subprocess.run(["git", "log", "--format=%s"], cwd=root / "code", capture_output=True, text=True).stdout
    assert "code written" in log


def test_the_pause_kind_has_plain_advice():
    from core import todo

    decide, rec, alts = todo.advice("code_project")
    assert decide and rec and alts


def test_a_simulation_that_uses_a_process_pool_runs_from_run_py(tmp_path):
    root = _quest(tmp_path)
    (root / "code" / "simulate.py").write_text(
        "import multiprocessing as mp\n\ndef _sq(x):\n    return x * x\n\n"
        "def run_trial(cell, trial_id, seed):\n    with mp.Pool(2) as pool:\n"
        "        return {'seed': float(seed), 'x': float(sum(pool.map(_sq, [1, 2, cell['n']])))}\n", encoding="utf-8")
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    done = _run(root)
    assert done.returncode == 0, done.stderr[-500:]
    trials = json.loads((root / "run_output" / "raw" / "trials.json").read_text())
    assert [c["ok"] for c in trials["cells"]] == [3, 3]


def test_run_py_stops_with_an_error_when_no_setting_produced_a_result(tmp_path):
    root = _quest(tmp_path)
    (root / "code" / "simulate.py").write_text(
        "def run_trial(cell, trial_id, seed):\n    raise RuntimeError('broken')\n", encoding="utf-8")
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    done = _run(root)
    assert done.returncode != 0 and "no setting produced a result" in done.stderr


def test_a_failed_commit_leaves_no_changelog_entry_behind(tmp_path, monkeypatch):
    import pytest

    if not _git_available():
        pytest.skip("git not installed")
    root = _quest(tmp_path)
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    real = code_project._git

    def refuse_commit(code_dir, *args):
        if args and args[0] == "commit":
            return subprocess.CompletedProcess(args, 1, "", "hook said no")
        return real(code_dir, *args)

    monkeypatch.setattr(code_project, "_git", refuse_commit)
    for _ in range(3):
        assert code_project.record_change(root, "code written") is False
    assert not (root / "code" / "CHANGELOG.md").exists()
    monkeypatch.setattr(code_project, "_git", real)
    assert code_project.record_change(root, "code written") is True
    assert (root / "code" / "CHANGELOG.md").read_text().count("## ") == 1


def test_a_file_with_windows_line_endings_is_not_taken_for_an_edit(tmp_path):
    root = _quest(tmp_path)
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    path = root / "code" / "README.md"
    path.write_bytes(path.read_bytes().decode("utf-8").replace("\n", "\r\n").encode("utf-8"))
    code_project.refresh(root, deps=[], protocol=PROTOCOL)
    assert code_project.unasked_conflicts(root) == []
