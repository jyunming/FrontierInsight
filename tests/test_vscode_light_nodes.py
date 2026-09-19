"""The VSCode "Per-node model overrides" editor: pick models from the list VSCode offers,
with the five measured light nodes as one entry, instead of typing node:model pairs.

Three layers, like the ensemble picker: the constants and prompt text shared with Python, the
compiled editor driven in real Node against a scripted stand-in ``vscode`` (menus answer from a
script; nothing here renders VSCode's own UI), and the answer it produces read back through the
real ``Config``. Skips without Node or a current compile, like the other extension tests."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from core.config import Config
from core.interview import LIGHT_NODES, QUESTIONS
from core.provider import model_for_node

EXT = Path(__file__).resolve().parent.parent / "vscode-frontier-insight"
CORE_TS = EXT / "src" / "interview-core.ts"
INTERVIEW_TS = EXT / "src" / "interview.ts"
CORE_JS = EXT / "out" / "interview-core.js"
INTERVIEW_JS = EXT / "out" / "interview.js"
DRIVER = Path(__file__).resolve().parent / "fixtures" / "vscode_pick_node_models.js"

MODELS = [
    {"id": "gpt-x-mini", "name": "GPT X mini", "vendor": "copilot", "family": "gpt-x-mini"},
    {"id": "big-model", "name": "Big Model", "vendor": "copilot", "family": "big"},
    {"id": "gemma3:4b", "name": "Gemma 3 4B", "vendor": "ollama", "family": "gemma3"},
]


def _need_node() -> str:
    node = shutil.which("node")
    if node is None or not INTERVIEW_JS.is_file() or not CORE_JS.is_file():
        pytest.skip("needs node and a compiled vscode-frontier-insight/out/")
    newest = max(p.stat().st_mtime for p in (EXT / "src").glob("*.ts"))
    if min(INTERVIEW_JS.stat().st_mtime, CORE_JS.stat().st_mtime) < newest:
        pytest.skip("out/ is older than the sources; run npm run compile")
    return node


def _drive(node_models: str, script: list, models: list | None = None) -> dict:
    done = subprocess.run(
        [_need_node(), str(DRIVER), str(INTERVIEW_JS),
         json.dumps({"node_models": node_models, "script": script, "models": MODELS if models is None else models})],
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


# --- what Python and TypeScript share --------------------------------------------------


def test_the_light_nodes_are_the_same_list_in_python_and_typescript() -> None:
    src = CORE_TS.read_text(encoding="utf-8")
    block = re.search(r"export const LIGHT_NODES[^=]*=\s*\[(.*?)\]", src, re.S)
    assert block, "LIGHT_NODES not found in interview-core.ts"
    assert tuple(re.findall(r'"([a-z_]+)"', block.group(1))) == LIGHT_NODES


def test_the_question_names_the_measured_nodes_and_gives_no_unmeasured_advice() -> None:
    (q,) = [q for q in QUESTIONS if q.id == "node_models"]
    for node in LIGHT_NODES:
        assert node in q.prompt and node in q.placeholder
    # It used to recommend a cheaper model for clarify and speech and a stronger one for
    # write/review; none of those was measured.
    assert "stronger" not in q.prompt and "clarify" not in q.prompt and "speech" not in q.prompt
    assert "untested" in q.prompt and "does not choose the model" in q.prompt


def test_the_editor_replaces_the_text_box_for_this_field() -> None:
    ts = INTERVIEW_TS.read_text(encoding="utf-8")
    assert 'which.value === "node_models"' in ts and "await editNodeModels(a)" in ts
    assert "vscode.lm.selectChatModels()" in ts
    assert 'placeHolder: which.value === "node_models"' not in ts, "the old typed-example placeholder is gone"


# --- the editor, driven --------------------------------------------------------------------


def test_light_nodes_go_to_the_one_model_picked_and_nothing_else_moves() -> None:
    got = _drive("write:big-model", ["Light nodes", "GPT X mini", "Done"])
    pairs = dict(p.split(":", 1) for p in got["node_models"].split(", "))
    assert pairs == {"write": "big-model", **{n: "gpt-x-mini" for n in LIGHT_NODES}}
    # The model list is what the stand-in VSCode offered, sorted, with the remove entry first.
    model_menu = got["menus"][1]
    assert model_menu["labels"] == ["Use my Chat-picker model (remove)", "Big Model", "Gemma 3 4B", "GPT X mini"]


def test_nothing_is_chosen_until_the_user_chooses() -> None:
    assert _drive("", ["Light nodes", "ESC", "Done"])["node_models"] == ""
    assert _drive("poster:big-model", ["ESC"])["node_models"] == "poster:big-model"
    assert _drive("", ["Done"])["node_models"] == ""


def test_one_node_shows_the_measured_ones_first_and_marks_the_rest_untested() -> None:
    got = _drive("", ["One node", "poster", "GPT X mini", "Done"])
    assert got["node_models"] == "poster:gpt-x-mini"
    node_menu = got["menus"][1]
    assert node_menu["labels"][: len(LIGHT_NODES)] == list(LIGHT_NODES)
    assert "write" in node_menu["labels"] and "implement" in node_menu["labels"]


def test_the_remove_entry_and_clear_undo_an_override() -> None:
    got = _drive("poster:big-model, slides:big-model", ["One node", "poster", "Use my Chat-picker", "Done"])
    assert got["node_models"] == "slides:big-model"
    assert _drive("poster:big-model, slides:big-model", ["Clear all", "Done"])["node_models"] == ""


def test_a_model_id_with_a_colon_survives_and_typing_still_works() -> None:
    got = _drive("", ["Light nodes", "Gemma 3 4B", "Done"])
    assert got["node_models"].startswith("cross_check:gemma3:4b, ")
    typed = _drive("", ["Type node", {"input": " poster:foo , slides:bar "}, "Done"])
    assert typed["node_models"] == "poster:foo , slides:bar"


def test_no_models_offered_warns_instead_of_offering_an_empty_list() -> None:
    got = _drive("", ["Light nodes", "Done"], models=[])
    assert got["node_models"] == ""
    assert any("lists no language models" in m.get("warning", "") for m in got["menus"])


def test_a_model_id_with_a_comma_cannot_be_written_and_is_refused_with_a_message() -> None:
    odd = [{"id": "a,b", "name": "Odd", "vendor": "x", "family": "y"}]
    got = _drive("poster:keep", ["Light nodes", "Odd", "Done"], models=odd)
    assert got["node_models"] == "poster:keep"
    assert any("cannot be written" in m.get("warning", "") for m in got["menus"])


# --- what the editor writes reaches the engine ----------------------------------------------


def test_what_the_editor_writes_routes_exactly_those_nodes_in_a_real_config(tmp_path: Path) -> None:
    got = _drive("", ["Light nodes", "GPT X mini", "One node", "write", "Big Model", "Done"])
    node = shutil.which("node")
    script = (
        "const c = require(process.argv[1]);"
        "process.stdout.write(c.answersToYaml(JSON.parse(process.argv[2])));"
    )
    base = {
        "topic": "picker probe", "title": "picker", "output_kinds": ["paper_md"], "paper_format": "generic",
        "clarify_mode": "auto", "review_panel": [], "knowledge_enabled": False, "no_simulation": False,
        "study_depth": "journal-length", "comparative_baseline": "", "success_metric": "", "budget": "",
        "provider_model": "", "max_iterations": 2, "audience": "external", "knowledge_top_k": 8,
        "node_models": got["node_models"],
    }
    yaml_text = subprocess.run(
        [node, "-e", script, str(CORE_JS), json.dumps(base)],
        capture_output=True, text=True, encoding="utf-8", timeout=60, check=True,
    ).stdout
    path = tmp_path / "picked.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    cfg = Config.from_yaml(path)
    routed = {n: model_for_node(cfg.provider.node_models, n) for n in (*LIGHT_NODES, "write", "implement", "review")}
    assert routed == {**{n: "gpt-x-mini" for n in LIGHT_NODES}, "write": "big-model", "implement": None, "review": None}
