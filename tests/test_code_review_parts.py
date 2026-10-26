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
                        "file": ("experiment.py" if "### experiment.py" in code else "simulate.py") if found else "",
                        "line": 2 if found else None,
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


def test_a_change_to_any_file_the_reader_is_given_is_another_code_version(tmp_path: Path) -> None:
    _write(tmp_path, {"simulate.py": "a = 1\n", "experiment.py": "b = 2\n", "run.py": "c = 3\n", "study.json": "{}\n",
                      "fi_search.py": "s = 9\n"})
    seen = {cr.version(tmp_path)}
    for name, text in (("run.py", "c = 4\n"), ("study.json", '{"x": 1}\n'), ("fi_search.py", "s = 8\n")):
        (tmp_path / name).write_text(text, encoding="utf-8")
        assert cr.version(tmp_path) not in seen, name
        seen.add(cr.version(tmp_path))


def test_a_place_counts_as_shown_only_inside_the_line_ranges_of_the_part() -> None:
    shown = {"pkg/model.py": [[1, 40]], "run.py": [[1, 10], [30, 50]]}
    assert cr.shown_in_part(shown, "model.py", 40) and not cr.shown_in_part(shown, "run.py", None)
    assert cr.shown_in_part(shown, "run.py", 35)
    assert not cr.shown_in_part(shown, "model.py", 41) and not cr.shown_in_part(shown, "run.py", 20)
    assert not cr.shown_in_part(shown, "experiment.py", 1) and not cr.shown_in_part(shown, "", None)
    reqs = cr.requirements({"variables": {"dependent": ["a_value"]}}, None)
    got = cr.parse({"requirements": [{"id": "R1", "status": "implemented", "file": "experiment.py", "line": 2}]}, reqs)
    outside: list = []
    kept = cr.keep_seen(got, shown, outside)
    assert kept == [] and outside == [{"id": "R1", "status": "implemented", "file": "experiment.py",
                                                           "line": 2}]


def test_the_ranges_of_a_split_file_are_recorded_for_each_part(tmp_path: Path) -> None:
    _write(tmp_path, {"simulate.py": "\n".join(f"v{i} = {i}" for i in range(300)) + "\n"})
    parts = cr.source_parts_detail(tmp_path, limit=1500)
    ranges = sorted(r for p in parts for r in p["shown"]["simulate.py"])
    assert len(parts) > 2 and ranges[0][0] == 1 and ranges[-1][1] == 300
    assert all(a[1] + 1 == b[0] for a, b in zip(ranges, ranges[1:])), "every line is in exactly one range"


def test_a_line_longer_than_the_cap_is_cut_and_the_cut_is_recorded(tmp_path: Path) -> None:
    _write(tmp_path, {"simulate.py": "a = 1\n", "study.json": '{"x": "' + "y" * 9000 + '"}\n'})
    parts = cr.source_parts_detail(tmp_path, limit=2000)
    assert all(len(p["text"]) <= 2000 for p in parts)
    cut = [c for p in parts for c in p["cut"]]
    assert cut == [{"file": "study.json", "line": 1, "chars": 9000 + 9 - 500}] or cut[0]["file"] == "study.json"
    assert "more characters of this line are not shown" in "\n".join(p["text"] for p in parts)


@pytest.mark.asyncio
async def test_a_part_that_cites_a_file_it_was_not_shown_does_not_find_the_requirement(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": EXP})
    monkeypatch.setattr(cr, "_MAX_CODE_CHARS", 700)
    seen: list[str] = []

    class Liar:
        async def chat(self, messages, **kw):  # noqa: ANN001
            prompt = messages[-1]["content"]
            if not prompt.lstrip().startswith("# Code Against the Plan"):
                return ""
            code = prompt.split("# The code", 1)[1].split("# What the code gave", 1)[0]
            seen.append("experiment.py" in code)
            reqs = json.loads(prompt.split("```json", 1)[1].split("```", 1)[0])
            have = "### experiment.py" in code
            out = [{"id": r["id"], "status": "missing" if have else "implemented", "file": "" if have else "experiment.py",
                    "line": None if have else 2, "note": "n"} for r in reqs]
            return json.dumps({"requirements": out, "checked": ["x"], "summary": "s"})

        async def aclose(self) -> None:
            return None

    engine._client = Liar()
    await engine._code_review_gate(WITH_PLAN)
    record = cr.load(engine.quest_root)
    assert seen == [False, False, True], "the first part is asked again for what it left unjudged"
    assert {v["status"] for v in record["verdicts"]} == {"missing"}, "a claim about a file it did not see is not counted"
    assert record["cited_outside_what_was_shown"]
    assert record["requests"][0]["files"].keys() == {"simulate.py"} and record["code_parts"] == 2


@pytest.mark.asyncio
async def test_the_readers_requests_of_a_quest_are_counted_and_a_reading_that_cannot_be_afforded_is_not_made(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": EXP})
    monkeypatch.setattr(cr, "_MAX_CODE_CHARS", 700)
    reader = _PartReader()
    engine._client = reader
    await engine._code_review_gate(WITH_PLAN)
    assert cr.load(engine.quest_root)["reader_calls"] == len(reader.prompts) == 2
    # the budget cannot cover the two requests of another reading: nothing is asked and the record says so
    engine2 = _engine(tmp_path / "second")
    _write(engine2.quest_root / "code", {"simulate.py": SIM, "experiment.py": EXP})
    monkeypatch.setattr(cr, "MAX_READER_CALLS", 1)
    reader2 = _PartReader()
    engine2._client = reader2
    await engine2._code_review_gate(WITH_PLAN)
    record = cr.load(engine2.quest_root)
    assert reader2.prompts == [] and "not read in full" in record["error"] and not record.get("verdicts")


class _Cites:
    """Answers every requirement implemented, citing the place it is given, and reports one finding at a place too."""

    def __init__(self, file: str, line, where: str) -> None:
        self.file, self.line, self.where, self.prompts = file, line, where, []

    async def chat(self, messages, **kw):  # noqa: ANN001
        prompt = messages[-1]["content"]
        if not prompt.lstrip().startswith("# Code Against the Plan"):
            return ""
        self.prompts.append(prompt)
        reqs = json.loads(prompt.split("```json", 1)[1].split("```", 1)[0])
        out = [{"id": r["id"], "status": "implemented", "file": self.file, "line": self.line, "note": ""} for r in reqs]
        finding = {"where": self.where, "what": "a step is skipped", "why": "w", "fix": "f", "severity": "high"}
        return json.dumps({"requirements": out, "findings": [finding], "checked": ["x"], "summary": "s"})

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("file,line,where", [("other.py", 1, "other.py:1"), ("simulate.py", None, "simulate.py"),
                                             ("simulate.py", 9999, "simulate.py:9999")])
async def test_one_request_counts_only_a_citation_of_a_file_and_a_line_it_was_shown(
        tmp_path: Path, file: str, line, where: str) -> None:
    engine = _engine(tmp_path)
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": EXP})
    reader = _Cites(file, line, where)
    engine._client = reader
    await engine._code_review_gate(WITH_PLAN)
    record = cr.load(engine.quest_root)
    assert len(reader.prompts) == 2 and record["code_parts"] == 1, "asked once more for what it left unjudged"
    assert not record.get("verdicts") and not [p for p in record.get("problems", []) if p.get("kind") == "logic"]
    assert record["error"] == "its answers cited no place in the code it was shown"
    assert record["cited_outside_what_was_shown"]


@pytest.mark.asyncio
async def test_a_citation_of_a_line_inside_what_was_shown_is_counted_in_one_request(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": EXP})
    engine._client = _Cites("experiment.py", 2, "simulate.py:3")
    await engine._code_review_gate(WITH_PLAN)
    record = cr.load(engine.quest_root)
    assert {v["status"] for v in record["verdicts"]} == {"implemented"}
    assert [p["line"] for p in record["problems"] if p.get("kind") == "logic"] == [3]
    assert "cited_outside_what_was_shown" not in record


@pytest.mark.asyncio
async def test_a_line_that_was_cut_is_not_fully_shown_so_a_citation_of_it_is_not_counted(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine(tmp_path)
    one_line = '{"padding": "' + "p" * 20000 + '", "grid": {"quality_factor": [0, 1]}}\n'
    _write(engine.quest_root / "code", {"simulate.py": SIM, "experiment.py": EXP, "study.json": one_line})
    monkeypatch.setattr(cr, "_MAX_CODE_CHARS", 8000)  # a line over a quarter of it (2000) is cut
    parts = cr.source_parts_detail(engine.quest_root / "code")
    cut = [c for p in parts for c in p["cut"]]
    assert cut and cut[0]["file"] == "study.json" and cut[0]["line"] == 1
    assert not cr.shown_in_part({"study.json": [[1, 1]]}, "study.json", 1, cut)
    reader = _Cites("study.json", 1, "study.json:1")  # the key it needs is past the cut
    engine._client = reader
    await engine._code_review_gate(WITH_PLAN)
    record = cr.load(engine.quest_root)
    assert not record.get("verdicts"), "a verdict whose only evidence is the cut line is not counted"
    assert record["lines_cut"][0]["file"] == "study.json"
    assert any(o.get("cut") for o in record["cited_outside_what_was_shown"])
