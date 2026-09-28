"""The knowledge base holds one current copy of a quest's result, and cited papers' cards name only accepted quests
(the 2026-09-28 re-audit, F-01 and F-08).

- F-01: when the copy under the other standing cannot be removed (or cannot be shown absent), nothing new is written,
  the write-back says it failed, and the person is told how to fix it (run.log and the quest's card). After a write,
  the knowledge base is checked again for a leftover copy.
- F-08: a cited paper's card keeps every accepted quest that used it (a union, not the last writer); a quest whose
  result becomes preliminary is taken off its cards, and a card left with no quest is removed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core import todo
from core.engine import Engine, QuestArtifacts
from core.knowledge import REF_SPINE, STANDING_KINDS, _CITED_BY
from tests.test_axon_standing import _HttpLikeBrain
from tests.test_knowledge import _BrainAddText, _enabled_knowledge_with

REF = {"title": "A real paper", "doi": "10.1/x", "authors": ["A"], "year": 2020, "abstract": "about it"}
OTHER_REF = {"title": "Another paper", "doi": "10.1/y", "authors": ["B"], "year": 2021, "abstract": "more"}


class _Http(_HttpLikeBrain):
    """The HTTP-like fake, returning text with its search rows (the service does)."""

    def __init__(self, *, delete_fails: bool = False) -> None:
        super().__init__()
        self.delete_fails = delete_fails

    def delete_documents(self, ids: list[str]) -> None:
        if self.delete_fails:
            raise RuntimeError("the Axon service is not reachable")
        super().delete_documents(ids)

    def search_raw(self, query, *, filters=None, overrides=None):  # noqa: ANN001
        rows = [{"text": d["text"], "metadata": d["metadata"]} for d in self.docs.values()
                if all(d["metadata"].get(k) == v for k, v in (filters or {}).items())]
        return rows, {}, None


class _InProcess(_BrainAddText):
    """An in-process-like brain with a readable chunk index whose deletion can be made to fail."""

    def __init__(self, *, delete_fails: bool = False) -> None:
        super().__init__()
        self.delete_fails = delete_fails
        self._own_vector_store = SimpleNamespace(delete_by_ids=lambda ids: None)
        index = self._own_bm25
        original = index.delete_documents

        def delete(ids):  # noqa: ANN001
            if self.delete_fails:
                raise RuntimeError("the index is locked")
            original(ids)

        index.delete_documents = delete


def _write(k, tmp_path: Path, standing: str, *, quest: str = "q1", refs: list | None = None) -> bool:
    paper = tmp_path / f"{quest}.md"
    paper.write_text("# the same paper", encoding="utf-8")
    return k.add_quest_artifacts(quest_id=quest, paper_md_path=paper, summary="s",
                                 metadata={"topic": "t", "title": "T", "standing": standing,
                                           "external_refs": list(refs or [])})


def _kinds_http(brain: _Http, quest: str = "q1") -> set[str]:
    return {d["metadata"]["kind"] for d in brain.docs.values() if d["metadata"].get("quest_id") == quest}


def _kinds_in_process(brain: _InProcess, quest: str = "q1") -> set[str]:
    return {c["metadata"]["kind"] for c in brain._own_bm25.corpus if c["metadata"].get("quest_id") == quest}


# --- F-01 -----------------------------------------------------------------------------------------------------------

def test_over_http_a_failed_removal_writes_nothing_new(tmp_path: Path) -> None:
    brain = _Http()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted")
    brain.delete_fails = True
    assert _write(k, tmp_path, "preliminary") is False
    kinds = _kinds_http(brain)
    assert set(STANDING_KINDS["accepted"]) <= kinds and not set(STANDING_KINDS["preliminary"]) & kinds
    assert "still holds this quest's earlier accepted copy" in k.last_writeback_problem
    assert "fi tools tidy-knowledge" in k.last_writeback_problem


def test_in_process_a_failed_removal_writes_nothing_new(tmp_path: Path) -> None:
    brain = _InProcess()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted")
    brain.delete_fails = True
    assert _write(k, tmp_path, "preliminary") is False
    assert not set(STANDING_KINDS["preliminary"]) & _kinds_in_process(brain)


def test_an_unreadable_index_is_not_taken_for_no_earlier_copy(tmp_path: Path) -> None:
    brain = _InProcess()
    k = _enabled_knowledge_with(brain)
    brain._own_bm25 = None  # nothing FI can read
    assert _write(k, tmp_path, "preliminary") is False
    assert "or could not show it does not" in k.last_writeback_problem


def test_a_successful_change_leaves_one_standing(tmp_path: Path) -> None:
    for brain, kinds in ((_Http(), _kinds_http), (_InProcess(), _kinds_in_process)):
        k = _enabled_knowledge_with(brain)
        assert _write(k, tmp_path, "accepted")
        assert _write(k, tmp_path, "preliminary") and k.last_writeback_problem is None
        assert not set(STANDING_KINDS["accepted"]) & kinds(brain)


def test_a_leftover_after_the_write_is_caught(tmp_path: Path) -> None:
    brain = _Http()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted")
    brain.delete_documents = lambda ids: None  # says it removed them, removes nothing
    assert _write(k, tmp_path, "preliminary") is False
    assert "still holds its earlier accepted copy" in k.last_writeback_problem


def test_the_engine_puts_the_problem_on_the_card_and_in_run_log(tmp_path: Path) -> None:
    warnings: list[str] = []

    class _Knowledge:
        enabled = True
        last_writeback_problem = None

        def add_quest_artifacts(self, **kwargs):
            self.last_writeback_problem = "the knowledge base still holds this quest's earlier accepted copy"
            return False

    paper = tmp_path / "paper.md"
    paper.write_text("# Probe\n", encoding="utf-8")
    eng = object.__new__(Engine)
    eng.quest_id, eng.quest_root, eng.fi_dir = "probe", tmp_path, tmp_path / ".fi"
    eng.fi_dir.mkdir()
    eng.knowledge = _Knowledge()
    eng.config = SimpleNamespace(
        knowledge=SimpleNamespace(write_back_quests=True, write_back_only_on_accept=True),
        provider=SimpleNamespace(name="fake", model="fake"), effective_result_use="research")
    eng._log = SimpleNamespace(info=lambda *a, **k: None, warning=lambda msg, *a: warnings.append(msg % a),
                               debug=lambda *a, **k: None)
    artifacts = QuestArtifacts(quest_id="probe", quest_root=tmp_path, paper_md=paper, paper_pdf=None,
                               figures_dir=None, bundle_manifest=None, raw_state={})
    eng._write_back_knowledge(artifacts, {"review": {"verdict": "accept", "status": "ok"}, "analysis": {},
                                          "design": {}}, "publication_ready")
    assert any("earlier accepted copy" in w for w in warnings)
    record = json.loads((eng.fi_dir / "knowledge_problem.json").read_text(encoding="utf-8"))
    assert "earlier accepted copy" in record["problem"]
    items = [i for i in todo.waiting(tmp_path) if i.kind == "knowledge"]
    assert items and "fi tools tidy-knowledge" in items[0].recommended
    # A later successful write-back clears it.
    eng.knowledge.add_quest_artifacts = lambda **kw: True
    eng.knowledge.last_writeback_problem = None
    eng._write_back_knowledge(artifacts, {"review": {"verdict": "accept", "status": "ok"}, "analysis": {},
                                          "design": {}}, "publication_ready")
    assert not (eng.fi_dir / "knowledge_problem.json").exists()


# --- F-08 -----------------------------------------------------------------------------------------------------------

def _cards(brain) -> dict[str, dict]:
    docs = brain.docs.values() if isinstance(brain, _Http) else brain._own_bm25.corpus
    return {d["metadata"]["paper_id"]: d for d in docs if d["metadata"].get("kind") == REF_SPINE}


def test_a_card_names_every_accepted_quest_that_used_the_paper(tmp_path: Path) -> None:
    for brain in (_Http(), _InProcess()):
        k = _enabled_knowledge_with(brain)
        assert _write(k, tmp_path, "accepted", quest="q1", refs=[REF])
        assert _write(k, tmp_path, "accepted", quest="q2", refs=[REF])
        (card,) = _cards(brain).values()
        assert card["metadata"]["consumed_by_quests"] == ["q1", "q2"], "a union, not the last writer"
        assert card["text"].rstrip().endswith(f"{_CITED_BY}q1, q2")


def test_a_quest_that_becomes_preliminary_is_taken_off_its_cards(tmp_path: Path) -> None:
    for brain in (_Http(), _InProcess()):
        k = _enabled_knowledge_with(brain)
        assert _write(k, tmp_path, "accepted", quest="q1", refs=[REF, OTHER_REF])
        assert _write(k, tmp_path, "accepted", quest="q2", refs=[REF])
        assert _write(k, tmp_path, "preliminary", quest="q1", refs=[REF, OTHER_REF])
        cards = _cards(brain)
        assert list(cards) == [next(iter(cards))] and cards[next(iter(cards))]["metadata"]["consumed_by_quests"] == ["q2"]
        assert "q1" not in cards[next(iter(cards))]["text"].splitlines()[-1], "the card no longer names q1"


def test_tidy_takes_a_preliminary_only_quest_off_old_cards(tmp_path: Path) -> None:
    for brain in (_Http(), _InProcess()):
        k = _enabled_knowledge_with(brain)
        assert _write(k, tmp_path, "accepted", quest="q1", refs=[REF])
        assert _write(k, tmp_path, "accepted", quest="q2", refs=[REF, OTHER_REF])
        # q2 became preliminary before the cards were kept current: its result is preliminary, its cards still name it.
        for d in list(brain.docs.values() if isinstance(brain, _Http) else brain._own_bm25.corpus):
            meta = d["metadata"]
            if meta.get("quest_id") == "q2" and meta.get("kind") in STANDING_KINDS["accepted"]:
                meta["kind"] = STANDING_KINDS["preliminary"][STANDING_KINDS["accepted"].index(meta["kind"])]
        if isinstance(brain, _Http):  # the service lists documents by id, which starts with the kind
            brain.docs = {(f"{d['metadata']['kind']}:{d['id'].split(':', 1)[1]}"
                           if d["metadata"].get("quest_id") == "q2" else d["id"]): d
                          for d in brain.docs.values()}
        plan = k.retire_stale_ref_spines(dry_run=True)
        assert {e["paper"] for e in plan} == set(_cards(brain)) and all("ok" not in e for e in plan)
        done = k.retire_stale_ref_spines()
        assert all(e.get("ok") for e in done)
        cards = _cards(brain)
        assert len(cards) == 1, "the card only q2 used is gone"
        assert next(iter(cards.values()))["metadata"]["consumed_by_quests"] == ["q1"]
        assert k.retire_stale_ref_spines() == []


def test_a_card_that_cannot_be_read_is_left_alone(tmp_path: Path) -> None:
    brain = _Http()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted", quest="q1", refs=[REF])
    brain.search_raw = lambda *a, **kw: ([], {}, None)  # listed, but its metadata cannot be read
    assert _write(k, tmp_path, "accepted", quest="q2", refs=[REF])
    (card,) = _cards(brain).values()
    assert card["metadata"]["consumed_by_quests"] == ["q1"], "never overwritten when its consumers cannot be read"
