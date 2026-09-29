"""How the retrieved sources were judged is kept, not thrown away.

The embedding floor and the batched 0-3 screen decide which sources reach the paper. Each source's score or grade, whether
it was kept and why go to ``.fi/literature_queries.json`` (stage ``floor`` and ``screen``), beside the queries that found
them. That file was already sealed, so the seal names the same file as before and an older seal still verifies.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.passages as pmod
from core import evidence
from core.engine import _query_set_digest, _query_set_standing
from core.knowledge import RetrievedDoc
from tests.test_literature_query_ledger import _sealed, _write  # noqa: F401
from tests.test_literature_screen import TOPIC, _engine, _grader, _page, _paper


def _entries(eng, stage: str) -> list[dict]:
    path = eng.fi_dir / "literature_queries.json"
    return [e for e in json.loads(path.read_text(encoding="utf-8"))["entries"] if e.get("stage") == stage]


@pytest.mark.asyncio
async def test_every_source_gets_its_grade_and_the_reason_for_its_verdict(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    own = RetrievedDoc(content="mine", metadata={"title": "My own paper", "source": "local_paper"})
    docs = [_paper("Action figures and play", "10.1/3"), _paper("Toys in the 1980s", "10.1/2"),
            _paper("Plastic polymer moulding", "10.1/1"),
            _page("A collector's blog on He-Man", "https://blog.example/he-man"),
            _page("Buy action figures now", "https://shop.example"), own]
    eng._chat = _grader({0: 3, 1: 2, 2: 1, 3: 1, 4: 0, 5: 0})
    await eng._screen_literature(TOPIC, docs, iteration=2)
    (entry,) = _entries(eng, "screen")
    assert entry["outcome"] == "graded" and entry["iteration"] == 2
    rows = {r["title"]: r for r in entry["sources"]}
    assert rows["Action figures and play"]["grade"] == 3 and rows["Action figures and play"]["kept"] is True
    assert rows["Plastic polymer moulding"]["kept"] is False
    assert rows["Plastic polymer moulding"]["why"] == "graded 1; needs 2 or higher"
    assert rows["A collector's blog on He-Man"]["kept"] is True
    assert rows["Buy action figures now"] == {**rows["Buy action figures now"], "grade": 0, "kept": False}
    assert rows["My own paper"]["grade"] is None and rows["My own paper"]["kept"] is True
    assert "never screened" in rows["My own paper"]["why"]
    assert rows["Action figures and play"]["source"] == "doi:10.1/3"
    assert entry["digest"] == _query_set_digest(entry) and _query_set_standing(entry) == "verified"


@pytest.mark.asyncio
async def test_a_source_kept_only_to_reach_the_minimum_says_so(tmp_path: Path) -> None:
    eng = _engine(tmp_path, min_keep=3)
    docs = [_paper(f"Paper {c} on toys and childhood", f"10.1/{c}") for c in "abcd"]
    eng._chat = _grader({0: 1, 1: 0, 2: 2, 3: 1})
    await eng._screen_literature(TOPIC, docs)
    (entry,) = _entries(eng, "screen")
    assert entry["minimum"] == 3
    why = {r["source"]: r["why"] for r in entry["sources"]}
    assert why["doi:10.1/a"] == "graded 1, below 2; kept to reach the minimum of 3 sources"
    assert why["doi:10.1/b"] == "graded 0; needs 2 or higher"
    assert why["doi:10.1/c"] == "graded 2; 2 or higher is kept"


@pytest.mark.parametrize("reply,outcome", [
    ("not json at all", "unreadable"), ('{"verdict": "fine"}', "unreadable"),
    ('{"grades": []}', "unreadable"), (RuntimeError("down"), "failed")])
@pytest.mark.asyncio
async def test_a_screen_that_could_not_grade_is_recorded_as_keeping_everything(tmp_path: Path, reply, outcome) -> None:
    eng = _engine(tmp_path)

    async def chat(prompt, node=""):
        if isinstance(reply, Exception):
            raise reply
        return reply

    eng._chat = chat
    await eng._screen_literature(TOPIC, [_paper("A", "10.1/a"), _paper("B", "10.1/b")])
    (entry,) = _entries(eng, "screen")
    assert entry["outcome"] == outcome
    assert all(r["kept"] and r["grade"] is None and "every source is kept" in r["why"] for r in entry["sources"])


@pytest.mark.asyncio
async def test_a_screen_that_is_switched_off_leaves_that_in_the_record(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    eng.config.knowledge.literature_screen = False
    await eng._screen_literature(TOPIC, [_paper("A", "10.1/a")])
    (entry,) = _entries(eng, "screen")
    assert entry["outcome"] == "off" and entry["sources"][0]["kept"] is True


@pytest.mark.asyncio
async def test_the_screen_call_is_named_by_its_id_in_the_record_of_model_calls(tmp_path: Path) -> None:
    from core import attempt_records as ar
    from tests.test_literature_query_once import _Client

    eng = _engine(tmp_path)
    eng._client = _Client([], json.dumps({"grades": [{"i": 0, "grade": 3}]}))
    await eng._screen_literature(TOPIC, [_paper("A", "10.1/a")])
    (entry,) = _entries(eng, "screen")
    ok = {r["call_id"] for r in ar.read(eng.fi_dir, ar.MODEL_CALLS)
          if r.get("node") == "literature_screen" and r.get("outcome") == "ok"}
    assert entry["call_id"] and entry["call_id"] in ok


def test_the_floor_records_each_sources_score(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path, min_keep=1)
    eng.config.knowledge.relevance_min_score = 0.3
    docs = [_paper("On topic", "10.1/a"), _paper("Off topic", "10.1/b"), _paper("Borderline", "10.1/c")]
    monkeypatch.setattr(pmod, "_embed_scores", lambda blobs, q: [0.8, 0.1, 0.35])
    stats: dict = {}
    kept = eng._filter_docs_by_relevance(TOPIC, docs, stats=stats)
    eng._record_floor_verdicts(docs, kept, stats, 1)
    (entry,) = _entries(eng, "floor")
    assert entry["outcome"] == "scored" and entry["threshold"] == 0.3
    rows = {r["source"]: r for r in entry["sources"]}
    assert rows["doi:10.1/a"]["score"] == 0.8 and rows["doi:10.1/a"]["kept"] is True
    assert rows["doi:10.1/b"]["kept"] is False and rows["doi:10.1/b"]["why"] == "score 0.10 is below 0.30"
    assert rows["doi:10.1/c"]["kept"] is True
    assert _query_set_standing(entry) == "verified"


def test_an_unscored_floor_is_recorded_as_such(tmp_path: Path, monkeypatch) -> None:
    eng = _engine(tmp_path)
    eng.config.knowledge.relevance_min_score = 0.3
    docs = [_paper("A", "10.1/a")]
    monkeypatch.setattr(pmod, "_embed_scores", lambda blobs, q: None)
    stats: dict = {}
    eng._record_floor_verdicts(docs, eng._filter_docs_by_relevance(TOPIC, docs, stats=stats), stats, 1)
    (entry,) = _entries(eng, "floor")
    assert entry["outcome"] == "not_scored" and entry["sources"] == []


def test_the_digest_of_an_entry_written_before_verdicts_existed_is_unchanged() -> None:
    old = {"stage": "literature", "key": "k", "iteration": 1, "queries": ["q"], "prompt_sha256": "p", "model": "m",
           "call_id": "c", "reason": "r", "revised_by": "fi", "source_sha256": None}
    import hashlib

    expected = hashlib.sha256(json.dumps(
        {k: old.get(k) for k in ("stage", "key", "iteration", "queries", "prompt_sha256", "model", "call_id", "reason",
                                 "revised_by", "source_sha256")},
        sort_keys=True, default=str).encode("utf-8")).hexdigest()
    assert _query_set_digest(old) == expected


def test_a_verdict_edited_after_it_was_written_no_longer_checks_out(tmp_path: Path) -> None:
    eng = _engine(tmp_path)
    entry = {"stage": "screen", "iteration": 1, "outcome": "graded", "threshold": None, "minimum": 0, "call_id": None,
             "sources": [{"source": "doi:1", "title": "t", "grade": 0, "kept": False, "why": "graded 0"}]}
    from core.engine import _record_query_set

    _record_query_set(eng.fi_dir, entry)
    (saved,) = _entries(eng, "screen")
    assert _query_set_standing(saved) == "verified"
    saved["sources"][0]["grade"], saved["sources"][0]["kept"] = 3, True
    assert _query_set_standing(saved) == "edited"


def test_the_seal_still_names_the_same_file_and_an_edit_of_a_verdict_after_it_is_a_gap(tmp_path: Path) -> None:
    assert evidence.SEALED_QUERIES == ".fi/literature_queries.json" and evidence.SEAL_RECORDS == 2
    root = _sealed(tmp_path)
    assert evidence.read(root)["trace_seal"] == "verified"
    edited = {"stage": "screen", "queries": None, "sources": [{"grade": 3, "kept": True}], "digest": "d"}
    (root / evidence.SEALED_QUERIES).write_text(json.dumps({"schema": 2, "entries": [edited]}), encoding="utf-8")
    read = evidence.read(root)
    assert read["trace_seal"] == "not_verified"
    assert any("literature_queries.json changed after the quest was sealed" in g
               for g in read["all_gaps"]["publication_ready"])
