"""A quest's result lives in Axon under one standing at a time (the 2026-09-27b re-audit, P1-1).

A quest accepted once and run again to a preliminary result used to leave its accepted copy under the accepted kinds,
still citable. Now the copy under the other standing is removed before the new one is written (first, because Axon
skips a text it has stored before), and ``fi tools tidy-knowledge`` removes the older copy that earlier write-backs
left behind.
"""

from __future__ import annotations

from pathlib import Path

from core.knowledge import STANDING_KINDS, _delete_in_process
from tests.test_knowledge import _enabled_knowledge_with


class _HttpLikeBrain:
    """What FI's HTTP brain offers: ingest, delete by document id, list document ids, a filtered search."""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.log: list[tuple[str, object]] = []

    def ingest(self, documents: list[dict]) -> None:
        self.log.append(("ingest", [d["id"] for d in documents]))
        for d in documents:
            self.docs[d["id"]] = d

    def finalize_ingest(self) -> None:
        pass

    def delete_documents(self, ids: list[str]) -> None:
        self.log.append(("delete", list(ids)))
        for i in ids:
            self.docs.pop(i, None)

    def list_sources(self) -> list[str]:
        return list(self.docs)

    def search_raw(self, query, *, filters=None, overrides=None):  # noqa: ANN001
        rows = [{"metadata": d["metadata"]} for d in self.docs.values()
                if all(d["metadata"].get(k) == v for k, v in (filters or {}).items())]
        return rows, {}, None


def _write(k, tmp_path: Path, standing: str) -> None:
    paper = tmp_path / "paper.md"
    paper.write_text("# the same paper", encoding="utf-8")
    assert k.add_quest_artifacts(quest_id="q1", paper_md_path=paper, summary="s",
                                 metadata={"topic": "t", "title": "T", "standing": standing})


def _kinds(brain: _HttpLikeBrain) -> set[str]:
    return {d["metadata"]["kind"] for d in brain.docs.values()}


def test_a_standing_change_removes_the_other_copy_before_writing(tmp_path: Path) -> None:
    brain = _HttpLikeBrain()
    k = _enabled_knowledge_with(brain)
    _write(k, tmp_path, "accepted")
    assert set(STANDING_KINDS["accepted"]) <= _kinds(brain)
    _write(k, tmp_path, "preliminary")
    kinds = _kinds(brain)
    assert set(STANDING_KINDS["preliminary"]) <= kinds
    assert not set(STANDING_KINDS["accepted"]) & kinds, "the accepted copy is gone: it can no longer be cited"
    last_delete = max(i for i, (op, _) in enumerate(brain.log) if op == "delete")
    last_ingest = max(i for i, (op, _) in enumerate(brain.log) if op == "ingest")
    assert last_delete < last_ingest, "removed first: Axon would skip the same text written again"
    _write(k, tmp_path, "accepted")
    assert not set(STANDING_KINDS["preliminary"]) & _kinds(brain)


class _Store:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_by_ids(self, ids: list[str]) -> None:
        self.deleted += ids

    def delete_documents(self, ids: list[str]) -> None:
        self.deleted += ids


class _Bm25:
    def __init__(self, corpus: list[dict]) -> None:
        self.corpus = corpus

    def delete_documents(self, ids: list[str]) -> None:
        self.corpus = [c for c in self.corpus if c["id"] not in ids]


class _InProcessBrain:
    def __init__(self, corpus: list[dict]) -> None:
        self._own_bm25 = _Bm25(corpus)
        self._own_vector_store = _Store()
        self._graph_backend = _Store()
        self._ingested_hashes = {"h:" + c["text"] for c in corpus}
        self.saved = 0

    def _doc_hash(self, chunk: dict) -> str:
        return "h:" + chunk["text"]

    def _save_hash_store(self) -> None:
        self.saved += 1


def test_in_process_every_chunk_and_its_duplicate_record_go() -> None:
    chunk = lambda i, kind, qid="q1": {"id": f"ns::{kind}:{qid}_p0_chunk_{i}", "text": f"{kind}{qid}{i}",  # noqa: E731
                                       "metadata": {"kind": kind, "quest_id": qid}}
    corpus = [chunk(0, "fi_quest_paper"), chunk(1, "fi_quest_paper"), chunk(0, "fi_paper_spine"),
              chunk(0, "fi_quest_paper", "q2"), chunk(0, "fi_preliminary_paper"),
              {"id": "raptor_x", "text": "sum", "metadata": {"children_ids": ["ns::fi_quest_paper:q1_p0_chunk_0"]}}]
    brain = _InProcessBrain(corpus)
    assert _delete_in_process(brain, quest_id="q1", kinds=STANDING_KINDS["accepted"])
    left = {c["id"] for c in brain._own_bm25.corpus}
    assert left == {"ns::fi_quest_paper:q2_p0_chunk_0", "ns::fi_preliminary_paper:q1_p0_chunk_0"}
    assert "raptor_x" in brain._own_vector_store.deleted, "a summary built over a removed chunk goes too"
    assert "h:fi_quest_paperq11" not in brain._ingested_hashes and "h:fi_quest_paperq20" in brain._ingested_hashes
    assert brain.saved == 1


def test_tidy_removes_the_older_copy_only(tmp_path: Path) -> None:
    brain = _HttpLikeBrain()
    k = _enabled_knowledge_with(brain)
    _write(k, tmp_path, "preliminary")
    # An accepted copy written later by FI before this change existed: both standings are there.
    older = {i: d for i, d in brain.docs.items()}
    for d in older.values():
        d["metadata"]["ingested_at"] = "2026-01-01T00:00:00+00:00"
    brain.docs.update({f"{kind}:q1:x": {"id": f"{kind}:q1:x", "text": "t",
                                        "metadata": {"kind": kind, "quest_id": "q1",
                                                     "ingested_at": "2026-02-01T00:00:00+00:00"}}
                       for kind in STANDING_KINDS["accepted"]})
    (plan,) = k.retire_stale_standing(dry_run=True)
    assert plan["kept"] == "accepted" and plan["removed"] == "preliminary" and "ok" not in plan
    assert set(STANDING_KINDS["preliminary"]) <= _kinds(brain), "a check changes nothing"
    (done,) = k.retire_stale_standing()
    assert done["ok"] and not set(STANDING_KINDS["preliminary"]) & _kinds(brain)
    assert set(STANDING_KINDS["accepted"]) <= _kinds(brain)
    assert k.retire_stale_standing() == []


def test_the_tool_is_listed() -> None:
    import launch

    assert "tidy-knowledge" in launch._tools_help()
    assert launch._expand_tools_argv(["tools", "tidy-knowledge", "check"]) == ["--tidy-knowledge", "check"]


def test_tidy_removes_nothing_when_it_cannot_tell_which_copy_is_newer(tmp_path: Path) -> None:
    brain = _HttpLikeBrain()
    k = _enabled_knowledge_with(brain)
    _write(k, tmp_path, "preliminary")
    for d in brain.docs.values():
        d["metadata"].pop("ingested_at", None)
    brain.docs.update({f"{kind}:q1:x": {"id": f"{kind}:q1:x", "text": "t", "metadata": {"kind": kind, "quest_id": "q1"}}
                       for kind in STANDING_KINDS["accepted"]})
    before = set(brain.docs)
    (entry,) = k.retire_stale_standing()
    assert entry["undetermined"] and set(brain.docs) == before


class _SkippingBrain(_HttpLikeBrain):
    """The Axon service over HTTP after a removal: the paper's text was stored before, so it does not land again."""

    def ingest(self, documents: list[dict]) -> None:
        super().ingest([d for d in documents if d["metadata"]["kind"] != "fi_preliminary_paper"])


def test_a_document_that_did_not_land_is_said_in_the_log(tmp_path: Path, caplog) -> None:
    import logging

    brain = _SkippingBrain()
    k = _enabled_knowledge_with(brain)
    _write(k, tmp_path, "accepted")
    with caplog.at_level(logging.WARNING):
        _write(k, tmp_path, "preliminary")
    assert any("did not land" in r.getMessage() and "fi_preliminary_paper" in r.getMessage() for r in caplog.records)
