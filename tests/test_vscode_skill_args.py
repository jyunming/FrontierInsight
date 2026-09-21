"""``--config <quest.yaml>`` beside the skill commands in the VSCode chat panel.

Runs the compiled parser (``vscode-frontier-insight/out/skill-args.js``) with Node, and checks that each
of the four skill commands hands the option to the ``launch.py`` call it makes, so the panel reaches a
skill living only in a folder a quest's YAML names, as the command line does with ``--config``.

Skips when Node or a current compiled extension is missing: the Python CI job does not build the
extension (its own job compiles it).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

EXT = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"
ARGS_JS = EXT / "out" / "skill-args.js"
ARGS_TS = EXT / "src" / "skill-args.ts"
SKILLS_TS = EXT / "src" / "skills.ts"


def _run(expression: str):
    node = shutil.which("node")
    if node is None or not ARGS_JS.is_file():
        pytest.skip("needs node and a compiled vscode-frontier-insight/out/skill-args.js")
    if ARGS_JS.stat().st_mtime < ARGS_TS.stat().st_mtime:
        pytest.skip("out/skill-args.js is older than the source; run npm run compile")
    script = f"const m = require(process.argv[1]); process.stdout.write(JSON.stringify({expression}));"
    done = subprocess.run(
        [node, "-e", script, str(ARGS_JS)], capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@pytest.mark.parametrize(
    "text,rest,config",
    [
        ("uncertainty-and-units", "uncertainty-and-units", ""),
        ("", "", ""),
        ("name --config C:\\quests\\a.yaml", "name", "C:\\quests\\a.yaml"),
        ("--config C:\\quests\\a.yaml name", "name", "C:\\quests\\a.yaml"),
        ('name --config "C:\\my quests\\a.yaml"', "name", "C:\\my quests\\a.yaml"),
        ("name --config 'C:\\my quests\\a.yaml' ", "name", "C:\\my quests\\a.yaml"),
        ("name --config=/home/me/q.yaml", "name", "/home/me/q.yaml"),
        ("--config q.yaml", "", "q.yaml"),
    ],
)
def test_the_config_is_taken_out_of_the_text(text: str, rest: str, config: str) -> None:
    got = _run(f"m.splitSkillArgs({json.dumps(text)})")
    assert got["rest"] == rest and got["config"] == config and "error" not in got


@pytest.mark.parametrize("text", ["name --config", "--config", "name --config  ", "name --config="])
def test_a_config_without_a_path_is_reported(text: str) -> None:
    got = _run(f"m.splitSkillArgs({json.dumps(text)})")
    assert got["config"] == "" and "needs the path" in got["error"]
    assert "--config" not in got["rest"]


def test_the_arguments_and_the_hint_for_a_config() -> None:
    assert _run('m.configArgs("")') == []
    assert _run('m.configArgs("q.yaml")') == ["--config", "q.yaml"]
    assert _run('m.configHint("")') == ""
    assert _run('m.configHint("q.yaml")') == " --config q.yaml"
    assert _run('m.configHint("C:\\\\my quests\\\\a.yaml")') == ' --config "C:\\my quests\\a.yaml"'


def _function_body(name: str) -> str:
    source = SKILLS_TS.read_text(encoding="utf-8")
    start = source.index(f"export async function {name}(")
    end = source.find("\nexport ", start + 10)
    return source[start:end if end > 0 else len(source)]


@pytest.mark.parametrize(
    "function,flag",
    [
        ("runListSkills", '"--skills"'),
        ("runScanSkill", '"--scan-skill"'),
        ("runApproveSkill", '"--approve-skill"'),
        ("runRevokeSkill", '"--revoke-skill"'),
    ],
)
def test_each_skill_command_passes_the_config_to_launch(function: str, flag: str) -> None:
    body = _function_body(function)
    assert "splitSkillArgs(promptArgs)" in body
    assert "...configArgs(config)" in body
    assert flag in body


def test_the_approval_call_and_the_review_before_it_both_carry_the_config() -> None:
    body = _function_body("runApproveSkill")
    assert len(re.findall(r"\.\.\.configArgs\(config\)", body)) == 2  # the scan, then the approval


def test_the_list_command_is_given_the_text_after_it() -> None:
    extension = (EXT / "src" / "extension.ts").read_text(encoding="utf-8")
    assert "runListSkills(prompt, stream, token)" in extension
