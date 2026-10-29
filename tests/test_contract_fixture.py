"""dev/evaluation/contract_fixture.py: a scratch quest from a recorded raw and a contract-form design."""

import json

import pytest

from dev.evaluation import contract_fixture


def _record(tmp_path):
    quest = tmp_path / "recorded"
    (quest / "raw").mkdir(parents=True)
    (quest / ".fi").mkdir()
    (quest / "raw" / "trials.json").write_text(json.dumps({"schema": "fi.trials/v1", "trials": [{"seed": 1}]}))
    (quest / "config.yaml").write_text("topic: tank cooling\n")
    (quest / ".fi" / "run_facts.json").write_text("{}")
    out = tmp_path / "parts" / "run1"
    out.mkdir(parents=True)
    (out / "design.json").write_text(json.dumps({"protocol": {"form": "contract", "metrics": []}}))
    (out / "contract.json").write_text(json.dumps({"form": "contract"}))
    return quest, out.parent


def test_builds_scratch_quest(tmp_path):
    quest, parts = _record(tmp_path)
    scratch = tmp_path / "scratch"
    copied = contract_fixture.build(quest, parts, scratch)
    assert (scratch / "raw" / "trials.json").read_text() == (quest / "raw" / "trials.json").read_text()
    assert json.loads((scratch / "design.json").read_text())["protocol"]["form"] == "contract"
    assert (scratch / "contract.json").is_file() and (scratch / "config.yaml").is_file()
    assert (scratch / ".fi" / "run_facts.json").is_file()
    assert not (scratch / "raw" / "trials.analysis.json").exists()
    assert set(json.loads((scratch / "fixture.json").read_text())) == set(copied)
    assert "raw/trials.json" in copied and "design.json" in copied


def test_refuses_scratch_inside_recorded_quest_and_missing_raw(tmp_path):
    quest, parts = _record(tmp_path)
    with pytest.raises(SystemExit):
        contract_fixture.build(quest, parts, quest / "scratch")
    (quest / "raw" / "trials.json").unlink()
    with pytest.raises(SystemExit):
        contract_fixture.build(quest, parts, tmp_path / "scratch")
