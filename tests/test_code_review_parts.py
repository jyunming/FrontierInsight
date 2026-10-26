"""The code reader always sees every file: when the files do not fit in one request they are read in parts and the
verdicts are merged from FI's own record (core/code_review.py). Fake models, neutral topics (a mass on a damped spring)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from core import code_review as cr
from tests.test_code_review import PROTOCOL, WITH_PLAN, _engine

pytestmark = pytest.mark.usefixtures("this_python")

#: What a requirement's text names, and the text of the code that shows it is computed.
EVIDENCE = {"quality_factor": "quality_factor", "settling_time": "settling_time", "decay.png": "decay.png",
            "E1": "# E1", "parameter c": "grid_c", "'quality'": "'quality'"}


def _write(code: Path, files: dict[str, str]) -> None:
    code.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (code / name).write_text(text, encoding="utf-8")


def test_everything_that_fits_is_one_part_unchanged(tmp_path: Path) -> None:
    _write(tmp_path, {"simulate.py": "x = 1\n", "experiment.py": "y = 2\n"})
    parts = cr.source_parts(tmp_path)
    assert len(parts) == 1 and "### simulate.py" in parts[0] and "### experiment.py" in parts[0]
    assert cr.sources_text(tmp_path) == (parts[0], False)
    assert cr.source_parts(tmp_path / "nowhere") == []


def test_every_file_and_every_line_is_shown_in_some_part(tmp_path: Path) -> None:
    big = "\n".join(f"value_{i} = {i}" for i in range(200)) + "\n"
    _write(tmp_path, {"simulate.py": big[:1500], "experiment.py": "z = 3\n" * 100})
    _write(tmp_path / "pkg", {"__init__.py": "", "model.py": big})
    parts = cr.source_parts(tmp_path, limit=1200)
    assert len(parts) > 2 and all(len(p) <= 1200 for p in parts)
    shown = "\n".join(parts)
    assert "### simulate.py" in shown and "### experiment.py" in shown
    for i in range(200):  # a split file keeps its own line numbers
        assert f"{i + 1:4d}  value_{i} = {i}\n" in shown
    assert re.search(r"### \S*model\.py \(lines 1-\d+ of 200", shown) or "model.py (lines" in shown
    assert sum(1 for line in shown.splitlines() if line.strip().endswith("z = 3")) == 100


def test_a_requirement_found_in_any_part_is_implemented_and_one_found_in_none_is_missing() -> None:
    reqs = cr.requirements({"variables": {"dependent": ["a_value", "b_value"]}}, None)
    a = cr.parse({"requirements": [{"id": "R1", "status": "implemented", "file": "x.py", "line": 3},
                                   {"id": "R2", "status": "missing", "note": "not in this part"}]}, reqs)
    b = cr.parse({"requirements": [{"id": "R1", "status": "missing", "note": "not in this part"},
                                   {"id": "R2", "status": "missing", "note": "not here either"}]}, reqs)
    got = {v["id"]: v for v in cr.merge_parts([a, b])}
    assert got["R1"]["status"] == "implemented" and got["R1"]["file"] == "x.py" and got["R1"]["line"] == 3
    assert got["R2"]["status"] == "missing"
    # Order does not decide it, and something found to differ is not lost to a part that did not hold it.
    assert {v["id"]: v["status"] for v in cr.merge_parts([b, a])} == {"R1": "implemented", "R2": "missing"}
    c = cr.parse({"requirements": [{"id": "R2", "status": "different", "note": "another quantity"}]}, reqs)
    assert {v["id"]: v["status"] for v in cr.merge_parts([b, c])}["R2"] == "different"


class _PartReader:
    """Says a requirement is implemented only when the code it is shown holds the text that proves it."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if not prompt.lstrip().startswith("# Code Against the Plan"):
            return ""
        self.prompts.append(prompt)
        code = prompt.split("# The code", 1)[1].split("# What the code gave", 1)[0]
        reqs = json.loads(prompt.split("```json", 1)[1].split("```", 1)[0])
        out = []
        for r in reqs:
            needs = [proof for key, proof in EVIDENCE.items() if key in r["text"]]
            found = bool(needs) and all(p in code for p in needs)
            out.append({"id": r["id"], "status": "implemented" if found else "missing",
                        "file": "x.py" if found else "", "line": 2 if found else None,
                        "note": "" if found else "not in the code shown"})
        return json.dumps({"requirements": out, "checked": ["each requirement against the code shown"], "summary": "s"})

    async def aclose(self) -> None:
        return None


SIM = "# E1\ndef run_cell(cell):\n    return {'quality': 1.0}\n" + "# pad\n" * 20
EXP = ("grid_c = [0.2, 0.5, 1.0]\nresult = {'quality_factor': 1, 'settling_time': 2}\nsave('decay.png')\n"
       + "# pad\n" * 20)


@pytest.mark.asyncio
async def test_a_requirement_computed_in_a_file_that_does_not_fit_with_the_others_is_not_called_missing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": EXP})
    monkeypatch.setattr(cr, "_MAX_CODE_CHARS", 700)  # simulate.py and experiment.py do not fit in one request
    assert len(cr.source_parts(engine.quest_root / "code")) == 2
    reader = _PartReader()
    engine._client = reader
    assert await engine._code_review_gate(WITH_PLAN) is None
    assert len(reader.prompts) == 2, "one request per part"
    assert "### experiment.py" in reader.prompts[1] and "Part 2 of 2" in reader.prompts[1]
    record = cr.load(engine.quest_root)
    assert [v["status"] for v in record["verdicts"]] == ["implemented"] * len(record["verdicts"]) and record["verdicts"]
    assert record["problems"] == [] and record["final"] is True


@pytest.mark.asyncio
async def test_a_requirement_in_no_part_is_still_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": EXP.replace("settling_time", "other_name")})
    monkeypatch.setattr(cr, "_MAX_CODE_CHARS", 700)
    reader = _PartReader()
    engine._client = reader
    await engine._code_review_gate(WITH_PLAN)
    record = cr.load(engine.quest_root)
    missing = {v["name"] for v in record["verdicts"] if v["status"] == "missing"}
    assert missing == {"settling_time"}, missing


@pytest.mark.asyncio
async def test_code_that_fits_is_read_in_one_request(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": EXP})
    reader = _PartReader()
    engine._client = reader
    await engine._code_review_gate(WITH_PLAN)
    assert len(reader.prompts) == 1 and "Part 1 of" not in reader.prompts[0]


def test_every_script_and_the_study_settings_are_read(tmp_path: Path) -> None:
    _write(tmp_path, {"simulate.py": "a = 1\n", "experiment.py": "b = 2\n", "run.py": "c = 3\n", "study.json": '{"d": 4}\n',
                      "fi_search.py": "e = 5\n", "test_x.py": "f = 6\n"})
    shown = "\n".join(cr.source_parts(tmp_path))
    assert all(f"### {n}" in shown for n in ("simulate.py", "experiment.py", "run.py", "study.json"))
    assert "### fi_search.py" in shown and "test_x.py" not in shown
    assert cr.part_files(shown) >= {"run.py", "study.json", "fi_search.py"}


@pytest.mark.asyncio
async def test_a_fault_named_in_a_file_the_request_was_not_shown_is_not_kept(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": EXP})
    monkeypatch.setattr(cr, "_MAX_CODE_CHARS", 700)
    reader = _PartReader()
    plain = reader.chat

    async def chat(messages, **kw):  # noqa: ANN001
        out = json.loads(await plain(messages, **kw) or "{}")
        if out:  # every part claims a fault in experiment.py, which only the second part holds
            out["findings"] = [{"where": "experiment.py:2", "what": "a figure is missing", "why": "w", "fix": "f",
                                "severity": "high"}]
        return json.dumps(out)

    reader.chat = chat  # type: ignore[method-assign]
    engine._client = reader
    await engine._code_review_gate(WITH_PLAN)
    record = cr.load(engine.quest_root)
    assert record["code_parts"] == 2 and record["files_read"] == ["experiment.py", "simulate.py"]
    logic = [p for p in record["problems"] if p.get("kind") == "logic"]
    assert len(logic) == 1 and logic[0]["file"] == "experiment.py"


def test_the_studys_own_files_come_before_fi_s_search_and_small_files_fill_the_gaps(tmp_path: Path) -> None:
    _write(tmp_path, {"simulate.py": "a = 1\n" * 100, "experiment.py": "b = 2\n" * 100, "fi_search.py": "s = 9\n" * 100,
                      "study.json": "{}\n"})
    parts = cr.source_parts(tmp_path, limit=1500)
    order = [h for p in parts for h in re.findall(r"^### (\S+)", p, re.M)]
    assert order.index("fi_search.py") > order.index("experiment.py")
    assert all(len(p) <= 1500 for p in parts)
    assert any("### study.json" in p and "### simulate.py" in p for p in parts), "a small file goes into a part with room"
