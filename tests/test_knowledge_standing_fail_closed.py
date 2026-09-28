"""The knowledge base holds one current copy of a quest's result, and cited papers' entries name only accepted quests
(the 2026-09-28 re-audit, F-01 and F-08).

- F-01: when the copy under the other standing cannot be removed (or cannot be shown absent), nothing new is written,
  the write-back says it failed, and the person is told how to fix it (run.log and the quest's card). After a write,
  the knowledge base is checked again for a leftover copy.
- F-08: a cited paper's entry keeps every accepted quest that used it (a union, not the last writer); a quest whose
  result becomes preliminary is taken off its entries, and an entry left with no quest is removed.

The fakes behave as the two real brains do: over HTTP the service files a document under ``metadata.source`` (which a
document's own metadata overrides), lists documents by it, and a search keeps only matches above its similarity
threshold unless asked for 0; in process the brain deletes a document by its id through the pieces'
``source_id`` and lists nothing. One test runs a real in-process AxonBrain when its model is available offline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import todo
from core.engine import Engine, QuestArtifacts
from core.knowledge import REF_SPINE, STANDING_KINDS, _CITED_BY
from tests.test_knowledge import _enabled_knowledge_with

REF = {"title": "A real paper", "doi": "10.1/x", "authors": ["A"], "year": 2020, "abstract": "about it",
       "source": "arxiv"}
OTHER_REF = {"title": "Another paper", "doi": "10.1/y", "authors": ["B"], "year": 2021, "abstract": "more",
             "source": "openalex"}


class _Http:
    """The Axon service as FI's HTTP brain sees it."""

    def __init__(self, *, delete_fails: bool = False) -> None:
        self.docs: dict[str, dict] = {}
        self.delete_fails = delete_fails

    def ingest(self, documents: list[dict]) -> None:
        for d in documents:
            meta = {"source": d["id"]}
            meta.update(d.get("metadata") or {})  # /add_texts: the document's own metadata wins
            self.docs[d["id"]] = {"id": d["id"], "text": d["text"], "metadata": meta}

    def finalize_ingest(self) -> None:
        pass

    def delete_documents(self, ids: list[str]) -> None:
        if self.delete_fails:
            raise RuntimeError("the Axon service is not reachable")
        wanted = set(ids)
        self.docs = {k: d for k, d in self.docs.items() if k not in wanted and d["metadata"].get("source") not in wanted}

    def list_sources(self) -> list[str]:
        return sorted({str(d["metadata"].get("source") or "") for d in self.docs.values()})

    def search_raw(self, query, *, filters=None, overrides=None):  # noqa: ANN001
        if (overrides or {}).get("threshold") != 0.0:
            return [], {}, None  # a paper id as the query scores below the default threshold
        rows = [{"id": d["id"], "text": d["text"], "metadata": dict(d["metadata"])} for d in self.docs.values()
                if all(d["metadata"].get(k) == v for k, v in (filters or {}).items())]
        return rows, {}, None


class _Index:
    def __init__(self) -> None:
        self._loaded: list[dict] = []
        self.corpus: list[dict] = []  # empty until loaded, as after a restart

    def ensure_corpus_loaded(self) -> None:
        self.corpus = self._loaded

    def delete_documents(self, ids: list[str]) -> None:
        self._loaded[:] = [c for c in self._loaded if c["id"] not in ids]


class _InProcess:
    """An in-process AxonBrain: pieces named ``<id>_p0_chunk_0`` with ``source_id`` ``<id>_p0``; deletion by
    document id through them; no listing, no search FI relies on."""

    def __init__(self, *, delete_fails: bool = False) -> None:
        self._own_bm25 = _Index()
        self.delete_fails = delete_fails

    def ingest(self, documents: list[dict]) -> None:
        for d in documents:
            meta = {**(d.get("metadata") or {}), "source_id": f"{d['id']}_p0", "chunk_index": 0}
            self._own_bm25._loaded.append({"id": f"{d['id']}_p0_chunk_0", "text": d["text"], "metadata": meta})

    def finalize_ingest(self) -> None:
        pass

    def delete_documents(self, ids: list[str]) -> dict:
        if self.delete_fails:
            raise RuntimeError("the index is locked")
        wanted = set(ids)
        hit = [c["id"] for c in self._own_bm25._loaded
               if c["id"] in wanted or c["metadata"]["source_id"] in wanted
               or c["metadata"]["source_id"].rsplit("_p", 1)[0] in wanted]
        self._own_bm25.delete_documents(hit)
        return {"deleted": len(hit)}


def _write(k, tmp_path: Path, standing: str, *, quest: str = "q1", refs: list | None = None) -> bool:
    paper = tmp_path / f"{quest}.md"
    paper.write_text(f"# the paper of {quest}", encoding="utf-8")  # two quests' papers differ, as in life
    return k.add_quest_artifacts(quest_id=quest, paper_md_path=paper, summary=f"what {quest} found",
                                 metadata={"topic": "t", "title": f"T {quest}", "standing": standing,
                                           "external_refs": list(refs or [])})


def _pieces(brain) -> list[dict]:
    if isinstance(brain, _Http):
        return list(brain.docs.values())
    brain._own_bm25.ensure_corpus_loaded()
    return list(brain._own_bm25.corpus)


def _kinds(brain, quest: str = "q1") -> set[str]:
    return {d["metadata"]["kind"] for d in _pieces(brain) if d["metadata"].get("quest_id") == quest}


def _entries(brain) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in _pieces(brain):
        if d["metadata"].get("kind") == REF_SPINE:
            assert d["metadata"]["paper_id"] not in out, "one entry per paper"
            out[d["metadata"]["paper_id"]] = d
    return out


BRAINS = [pytest.param(_Http, id="http"), pytest.param(_InProcess, id="in_process")]


# --- F-01 -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("make", BRAINS)
def test_a_failed_removal_writes_nothing_new(tmp_path: Path, make) -> None:
    brain = make()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted")
    brain.delete_fails = True
    assert _write(k, tmp_path, "preliminary") is False
    kinds = _kinds(brain)
    assert set(STANDING_KINDS["accepted"]) <= kinds and not set(STANDING_KINDS["preliminary"]) & kinds
    assert "still holds this quest's earlier accepted copy" in k.last_writeback_problem
    assert "fi tools tidy-knowledge" in k.last_writeback_problem


def test_an_unreadable_index_is_not_taken_for_no_earlier_copy(tmp_path: Path) -> None:
    brain = _InProcess()
    k = _enabled_knowledge_with(brain)
    brain._own_bm25 = None  # nothing FI can read
    brain.ingest = lambda docs: None
    brain.delete_fails = True
    assert _write(k, tmp_path, "preliminary") is False
    assert "or could not show it does not" in k.last_writeback_problem
    # A removal that reports success is not enough either: afterwards nothing shows the old copy is gone.
    brain.delete_documents = lambda ids: {}
    assert _write(k, tmp_path, "preliminary") is False
    assert "could not be shown to hold no earlier accepted copy" in k.last_writeback_problem


@pytest.mark.parametrize("make", BRAINS)
def test_a_successful_change_leaves_one_standing(tmp_path: Path, make) -> None:
    brain = make()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted")
    assert _write(k, tmp_path, "preliminary") and k.last_writeback_problem is None
    assert not set(STANDING_KINDS["accepted"]) & _kinds(brain)


def test_a_leftover_after_the_write_is_caught(tmp_path: Path) -> None:
    brain = _Http()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted")
    brain.delete_documents = lambda ids: None  # says it removed them, removes nothing
    assert _write(k, tmp_path, "preliminary") is False
    assert "still holds its earlier accepted copy" in k.last_writeback_problem


def test_a_listing_cut_short_is_not_proof_of_absence(tmp_path: Path) -> None:
    brain = _Http()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted")
    brain.list_sources = lambda: []  # a store that lists nothing (cut short), while the documents are there
    brain.delete_fails = True
    assert _write(k, tmp_path, "preliminary") is False, "the search still finds the accepted copy"


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
    state = {"review": {"verdict": "accept", "status": "ok"}, "analysis": {}, "design": {}}
    eng._write_back_knowledge(artifacts, state, "publication_ready")
    assert any("earlier accepted copy" in w for w in warnings)
    record = json.loads((eng.fi_dir / "knowledge_problem.json").read_text(encoding="utf-8"))
    assert "earlier accepted copy" in record["problem"]
    items = [i for i in todo.waiting(tmp_path) if i.kind == "knowledge"]
    assert items and "fi tools tidy-knowledge" in items[0].recommended
    # A later successful write-back clears it; so does a quest that no longer writes back at all.
    eng.knowledge.add_quest_artifacts = lambda **kw: True
    eng.knowledge.last_writeback_problem = None
    eng._write_back_knowledge(artifacts, state, "publication_ready")
    assert not (eng.fi_dir / "knowledge_problem.json").exists()
    (eng.fi_dir / "knowledge_problem.json").write_text('{"problem": "x"}', encoding="utf-8")
    eng.config.knowledge.write_back_quests = False
    eng._write_back_knowledge(artifacts, state, "publication_ready")
    assert not (eng.fi_dir / "knowledge_problem.json").exists()


def test_a_failed_write_after_removing_a_copy_is_on_the_card(tmp_path: Path) -> None:
    brain = _InProcess()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted")

    def broken(docs):  # noqa: ANN001
        raise RuntimeError("the store went away")

    brain.ingest = broken
    assert _write(k, tmp_path, "preliminary") is False
    assert "may hold no copy of this result now" in k.last_writeback_problem
    assert "resume the quest" in k.last_writeback_problem


def test_a_skipped_write_back_keeps_only_what_is_true(tmp_path: Path) -> None:
    class _Knowledge:
        enabled = True
        last_writeback_problem = None
        conflict: bool | None = False

        def standing_conflict(self, quest_id):  # noqa: ANN001
            return self.conflict

    paper = tmp_path / "paper.md"
    paper.write_text("# Probe\n", encoding="utf-8")
    eng = object.__new__(Engine)
    eng.quest_id, eng.quest_root, eng.fi_dir = "probe", tmp_path, tmp_path / ".fi"
    eng.fi_dir.mkdir()
    eng.knowledge = _Knowledge()
    eng.config = SimpleNamespace(knowledge=SimpleNamespace(write_back_quests=True, write_back_only_on_accept=True))
    eng._log = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None)
    artifacts = QuestArtifacts(quest_id="probe", quest_root=tmp_path, paper_md=paper, paper_pdf=None,
                               figures_dir=None, bundle_manifest=None, raw_state={})
    note = eng.fi_dir / "knowledge_problem.json"
    rejected = {"review": {"verdict": "revise", "status": "ok"}}
    note.write_text('{"problem": "still holds the earlier copy", "kind": "both"}', encoding="utf-8")
    eng._write_back_knowledge(artifacts, rejected, None)
    assert not note.exists(), "no longer true: the note goes"
    eng.knowledge.conflict = True
    eng._write_back_knowledge(artifacts, rejected, None)
    assert "holds both" in json.loads(note.read_text(encoding="utf-8"))["problem"], "still true: it stays"


def test_a_note_is_kept_when_the_check_cannot_tell(tmp_path: Path) -> None:
    """Both copies planted, the service's list and search failing, a revise verdict: the note stays."""
    brain = _Http()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted")
    for d in list(brain.docs.values()):  # a preliminary copy beside it
        if d["metadata"].get("kind") in STANDING_KINDS["accepted"]:
            kind = STANDING_KINDS["preliminary"][STANDING_KINDS["accepted"].index(d["metadata"]["kind"])]
            brain.docs[d["id"].replace(d["metadata"]["kind"], kind, 1)] = {
                **d, "metadata": {**d["metadata"], "kind": kind, "source": d["id"].replace(d["metadata"]["kind"], kind, 1)}}
    assert k.standing_conflict("q1") is True
    eng = _skip_engine(tmp_path, k)
    note = eng.fi_dir / "knowledge_problem.json"
    note.write_text(json.dumps({"problem": "holds both", "kind": "both"}), encoding="utf-8")

    def down(*a, **kw):  # noqa: ANN001
        raise RuntimeError("the Axon service is down")

    brain.search_raw = down
    brain.list_sources = down
    assert k.standing_conflict("q1") is None
    eng._write_back_knowledge(_artifacts(tmp_path), {"review": {"verdict": "revise", "status": "ok"}}, None)
    assert note.exists(), "cannot tell: left as it is"


def test_a_not_written_note_is_cleared_only_by_a_write_that_succeeds(tmp_path: Path) -> None:
    brain = _Http()
    k = _enabled_knowledge_with(brain)
    eng = _skip_engine(tmp_path, k)
    note = eng.fi_dir / "knowledge_problem.json"
    note.write_text(json.dumps({"problem": "the write failed", "kind": "unwritten"}), encoding="utf-8")
    assert k.standing_conflict("probe") is False
    eng._write_back_knowledge(_artifacts(tmp_path), {"review": {"verdict": "revise", "status": "ok"}}, None)
    assert note.exists(), "a skipped write-back does not settle an unwritten result"


def test_a_failed_first_write_does_not_speak_of_an_earlier_copy(tmp_path: Path) -> None:
    brain = _InProcess()
    k = _enabled_knowledge_with(brain)

    def broken(docs):  # noqa: ANN001
        raise RuntimeError("the store went away")

    brain.ingest = broken
    assert _write(k, tmp_path, "accepted") is False
    assert "earlier copy" not in k.last_writeback_problem and "resume the quest" in k.last_writeback_problem


def _skip_engine(tmp_path: Path, knowledge) -> Engine:  # noqa: ANN001
    eng = object.__new__(Engine)
    eng.quest_id, eng.quest_root, eng.fi_dir = "probe", tmp_path, tmp_path / ".fi"
    eng.fi_dir.mkdir(exist_ok=True)
    eng.knowledge = knowledge
    eng.config = SimpleNamespace(knowledge=SimpleNamespace(write_back_quests=True, write_back_only_on_accept=True))
    eng._log = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None)
    return eng


def _artifacts(tmp_path: Path) -> QuestArtifacts:
    paper = tmp_path / "probe.md"
    paper.write_text("# Probe", encoding="utf-8")
    return QuestArtifacts(quest_id="probe", quest_root=tmp_path, paper_md=paper, paper_pdf=None,
                          figures_dir=None, bundle_manifest=None, raw_state={})


def test_a_lock_it_cannot_open_is_waited_for_then_said_plainly(tmp_path: Path) -> None:
    from core.axon_http import AxonUnavailable, _ProcessLock

    blocked = tmp_path / "a-file" / "x.lock"
    (tmp_path / "a-file").write_text("not a folder", encoding="utf-8")  # the lock's folder cannot be made
    lock = _ProcessLock(blocked, timeout=0.5, what="the knowledge base's paper entries")
    with pytest.raises(AxonUnavailable, match="could not be opened"):
        with lock.held():
            pass


def test_the_lock_names_what_it_guards(tmp_path: Path) -> None:
    from core.axon_http import AxonUnavailable, _ProcessLock

    path = tmp_path / "busy.lock"
    holder = _ProcessLock(path, what="the knowledge base's paper entries")
    waiter = _ProcessLock(path, timeout=0.5, what="the knowledge base's paper entries")
    with holder.held():
        with pytest.raises(AxonUnavailable, match="held the knowledge base's paper entries"):
            with waiter.held():
                pass


# --- F-08 -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("make", BRAINS)
def test_an_entry_names_every_accepted_quest_that_used_the_paper(tmp_path: Path, make) -> None:
    brain = make()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted", quest="q1", refs=[REF])
    assert _write(k, tmp_path, "accepted", quest="q2", refs=[REF])
    (entry,) = _entries(brain).values()
    assert entry["metadata"]["consumed_by_quests"] == ["q1", "q2"], "a union, not the last writer"
    assert entry["text"].rstrip().endswith(f"{_CITED_BY}q1, q2")
    assert entry["metadata"]["ref_source"] == "arxiv"
    if isinstance(brain, _Http):
        assert f"{REF_SPINE}:doi:10.1/x" in brain.list_sources(), "listed under its own id, not the paper's source"


@pytest.mark.parametrize("make", BRAINS)
def test_a_quest_that_becomes_preliminary_is_taken_off_its_entries(tmp_path: Path, make) -> None:
    brain = make()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted", quest="q1", refs=[REF, OTHER_REF])
    assert _write(k, tmp_path, "accepted", quest="q2", refs=[REF])
    assert _write(k, tmp_path, "preliminary", quest="q1", refs=[REF, OTHER_REF])
    entries = _entries(brain)
    assert list(entries) == ["doi:10.1/x"], "the entry only q1 used is gone"
    assert entries["doi:10.1/x"]["metadata"]["consumed_by_quests"] == ["q2"]
    assert "q1" not in entries["doi:10.1/x"]["text"].splitlines()[-1]


def test_a_failed_write_restores_the_entries_it_removed(tmp_path: Path) -> None:
    brain = _Http()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted", quest="q1", refs=[REF])
    real_ingest = brain.ingest
    calls = {"n": 0}

    def failing_once(docs):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the service went away mid-write")
        real_ingest(docs)

    brain.ingest = failing_once
    assert _write(k, tmp_path, "accepted", quest="q2", refs=[REF]) is False
    (entry,) = _entries(brain).values()
    assert entry["metadata"]["consumed_by_quests"] == ["q1"], "q1's entry is back as it was"


def test_a_write_that_landed_before_its_answer_was_lost_is_not_restored_twice(tmp_path: Path) -> None:
    brain = _Http()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted", quest="q1", refs=[REF])
    real_ingest = brain.ingest
    calls = {"n": 0}

    def lands_then_fails(docs):  # noqa: ANN001
        calls["n"] += 1
        real_ingest(docs)
        if calls["n"] == 1:
            raise RuntimeError("the answer was lost")

    brain.ingest = lands_then_fails
    assert _write(k, tmp_path, "accepted", quest="q2", refs=[REF]) is False
    (entry,) = _entries(brain).values()
    assert entry["metadata"]["consumed_by_quests"] == ["q1", "q2"], "the entry that landed is kept, not replaced"
    assert calls["n"] == 1, "nothing written back over it"


def test_chroma_s_joined_list_is_read(tmp_path: Path) -> None:
    brain = _InProcess()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted", quest="q1", refs=[REF])
    assert _write(k, tmp_path, "accepted", quest="q2", refs=[REF])
    for c in brain._own_bm25._loaded:
        if c["metadata"].get("kind") == REF_SPINE:
            c["metadata"]["consumed_by_quests"] = "q1|q2"  # how Chroma stores a list
    assert _write(k, tmp_path, "preliminary", quest="q1", refs=[REF])
    (entry,) = _entries(brain).values()
    assert entry["metadata"]["consumed_by_quests"] == ["q2"]


@pytest.mark.parametrize("make", BRAINS)
def test_tidy_takes_a_preliminary_only_quest_off_old_entries(tmp_path: Path, make) -> None:
    brain = make()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted", quest="q1", refs=[REF])
    assert _write(k, tmp_path, "accepted", quest="q2", refs=[REF, OTHER_REF])
    # q2 became preliminary before entries were kept current: its result is preliminary, its entries still name it.
    for d in _pieces(brain):
        meta = d["metadata"]
        if meta.get("quest_id") == "q2" and meta.get("kind") in STANDING_KINDS["accepted"]:
            meta["kind"] = STANDING_KINDS["preliminary"][STANDING_KINDS["accepted"].index(meta["kind"])]
            if isinstance(brain, _Http):
                meta["source"] = f"{meta['kind']}:{meta['source'].split(':', 1)[1]}"
    plan = k.retire_stale_ref_spines(dry_run=True)
    assert {e["paper"] for e in plan} == {"doi:10.1/x", "doi:10.1/y"} and all("ok" not in e for e in plan)
    done = k.retire_stale_ref_spines()
    assert all(e.get("ok") for e in done)
    entries = _entries(brain)
    assert list(entries) == ["doi:10.1/x"] and entries["doi:10.1/x"]["metadata"]["consumed_by_quests"] == ["q1"]
    assert k.retire_stale_ref_spines() == []


def test_tidy_rewrites_an_entry_an_older_write_back_hid_from_the_list(tmp_path: Path) -> None:
    brain = _Http()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted", quest="q1", refs=[REF])
    for d in brain.docs.values():  # as an older write-back stored it: the paper's source over the entry's own id
        if d["metadata"].get("kind") == REF_SPINE:
            d["metadata"]["source"] = "arxiv"
            d["metadata"].pop("ref_source", None)
            d["metadata"].pop("entry_layout", None)
    assert f"{REF_SPINE}:doi:10.1/x" not in brain.list_sources()
    done = k.retire_stale_ref_spines()
    assert [e for e in done if e.get("migrated")] and all(e.get("ok") for e in done)
    assert f"{REF_SPINE}:doi:10.1/x" in brain.list_sources()
    (entry,) = _entries(brain).values()
    assert entry["metadata"]["ref_source"] == "arxiv" and entry["metadata"]["consumed_by_quests"] == ["q1"]
    before = dict(brain.docs)
    assert not [e for e in k.retire_stale_ref_spines() if e.get("migrated")], "a migrated entry is not done again"
    assert brain.docs == before


def test_an_entry_that_cannot_be_read_is_left_alone(tmp_path: Path) -> None:
    brain = _Http()
    k = _enabled_knowledge_with(brain)
    assert _write(k, tmp_path, "accepted", quest="q1", refs=[REF])
    brain.search_raw = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("search is down"))
    assert _write(k, tmp_path, "accepted", quest="q2", refs=[REF]) is False  # its own copy cannot be shown either
    (entry,) = _entries(brain).values()
    assert entry["metadata"]["consumed_by_quests"] == ["q1"], "never overwritten when its quests cannot be read"


# --- a real in-process AxonBrain ------------------------------------------------------------------------------------

def _real_brain(tmp_path: Path):
    cache = Path(os.environ.get("FI_TEST_AXON_MODEL_CACHE") or Path.home() / ".axon" / "model_cache" / "fastembed")
    try:
        from axon.config import AxonConfig
        from axon.main import AxonBrain
    except Exception:  # noqa: BLE001
        pytest.skip("axon is not importable here")
    if not (cache / "models--qdrant--all-MiniLM-L6-v2-onnx").is_dir():
        pytest.skip(f"the embedding model is not cached at {cache} (no download in tests)")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return AxonBrain(AxonConfig(axon_store_base=str(tmp_path / "store"), embedding_provider="fastembed",
                                    embedding_model_path=str(cache)))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"a real AxonBrain could not be built offline: {e}")


@pytest.mark.slow
def test_a_real_axon_brain_keeps_one_standing_and_current_entries(tmp_path: Path) -> None:
    brain = _real_brain(tmp_path)
    assert not hasattr(brain, "list_sources") and callable(getattr(brain, "delete_documents", None))
    k = _enabled_knowledge_with(brain)

    def corpus() -> list[dict]:
        brain._own_bm25.ensure_corpus_loaded()
        return list(brain._own_bm25.corpus)

    def kinds(quest: str) -> set[str]:
        return {c["metadata"]["kind"] for c in corpus() if c["metadata"].get("quest_id") == quest}

    def entry_quests() -> dict[str, str]:
        return {c["metadata"]["paper_id"]: str(c["metadata"].get("consumed_by_quests")) for c in corpus()
                if c["metadata"].get("kind") == REF_SPINE}

    assert _write(k, tmp_path, "accepted", quest="q1", refs=[REF]), k.last_writeback_problem
    assert _write(k, tmp_path, "accepted", quest="q2", refs=[REF, OTHER_REF]), k.last_writeback_problem
    quests = entry_quests()
    assert sorted(quests) == ["doi:10.1/x", "doi:10.1/y"]
    assert "q1" in quests["doi:10.1/x"] and "q2" in quests["doi:10.1/x"], "two quests citing one paper: both named"
    assert _write(k, tmp_path, "preliminary", quest="q2", refs=[REF, OTHER_REF]), k.last_writeback_problem
    assert not set(STANDING_KINDS["accepted"]) & kinds("q2") and set(STANDING_KINDS["preliminary"]) <= kinds("q2")
    quests = entry_quests()
    assert list(quests) == ["doi:10.1/x"], "the entry only q2 used is gone"
    assert "q2" not in quests["doi:10.1/x"] and "q1" in quests["doi:10.1/x"]
    assert k.retire_stale_standing() == [] and k.retire_stale_ref_spines() == []
    # Written back again under the same standing (a resumed quest): no "id already exists", still one standing.
    assert _write(k, tmp_path, "preliminary", quest="q2", refs=[REF, OTHER_REF]), k.last_writeback_problem
    assert set(STANDING_KINDS["preliminary"]) <= kinds("q2")
