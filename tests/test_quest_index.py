"""The quest index (core/quest_index.py): every quest FI ran on this computer, found by id from any folder.

Every test runs against a temporary ``FI_HOME`` (tests/conftest.py:_isolate_fi_home), never the real
``~/.frontier-insight``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from core import quest_index

REPO = Path(__file__).resolve().parent.parent


def _quest(folder: Path, qid: str, title: str = "") -> Path:
    root = folder / "outputs" / qid
    (root / ".fi").mkdir(parents=True)
    (root / ".fi" / "state.sqlite").write_bytes(b"")
    (root / "config.yaml").write_text(f"title: {title or qid}\noutput:\n  output_dir: ./outputs\n", encoding="utf-8")
    return root


def test_the_index_lives_in_the_per_person_folder_and_the_tests_never_touch_the_real_one(tmp_path: Path) -> None:
    home = Path(os.environ["FI_HOME"])
    assert home != Path.home() / ".frontier-insight"
    assert quest_index.index_path() == home / "quests.json"


def test_register_records_where_a_quest_is_and_keeps_when_it_started(tmp_path: Path) -> None:
    root = _quest(tmp_path / "study_a", "1790003131-energy-drift-479b06", "Energy drift")
    quest_index.register(root, working_folder=tmp_path / "study_a", title="Energy drift")
    raw = json.loads(quest_index.index_path().read_text(encoding="utf-8"))
    assert raw["version"] == 1
    e = raw["quests"]["1790003131-energy-drift-479b06"]
    assert e["quest_root"] == str(root.resolve())
    assert e["config"] == str(root.resolve() / "config.yaml")
    assert e["working_folder"] == str((tmp_path / "study_a").resolve())
    assert e["title"] == "Energy drift"
    first = e["created_at"]
    quest_index.register(root, title="")  # a resume without a title keeps the one it had
    e = json.loads(quest_index.index_path().read_text(encoding="utf-8"))["quests"]["1790003131-energy-drift-479b06"]
    assert e["created_at"] == first and e["title"] == "Energy drift"
    assert e["working_folder"] == str((tmp_path / "study_a").resolve())


def test_short_id_is_the_six_characters_after_the_last_dash() -> None:
    assert quest_index.short_id("1790003131-energy-drift-479b06") == "479b06"
    assert quest_index.short_id("my-own-name") == "my-own-name"


def test_a_quest_is_found_from_another_folder_by_full_id_start_or_end(tmp_path: Path) -> None:
    root = _quest(tmp_path / "study_a", "1790003131-energy-drift-479b06")
    quest_index.register(root, working_folder=tmp_path / "study_a")
    elsewhere = tmp_path / "study_b" / "outputs"
    elsewhere.mkdir(parents=True)
    for text in ("1790003131-energy-drift-479b06", "479b06", "b06", "1790003131-ene"):
        if len(text) < quest_index.MIN_SHORT:
            with pytest.raises(quest_index.QuestNotFound):
                quest_index.find(text, [elsewhere])
            continue
        found = quest_index.find(text, [elsewhere])
        assert found.root == root.resolve() and found.via == "index"
        assert found.working_folder == (tmp_path / "study_a").resolve()


def test_the_local_outputs_folder_wins_over_the_index(tmp_path: Path) -> None:
    here = _quest(tmp_path / "here", "1790000000-local-aaaaaa")
    there = _quest(tmp_path / "there", "1790000000-local-aaaaaa")
    quest_index.register(there)
    found = quest_index.find("1790000000-local-aaaaaa", [tmp_path / "here" / "outputs"])
    assert found.root == here.resolve() and found.via == "local"
    # And a shortened id is looked up among the local quests first too.
    assert quest_index.find("aaaaaa", [tmp_path / "here" / "outputs"]).via == "local"


def test_a_folder_path_is_still_accepted(tmp_path: Path) -> None:
    root = _quest(tmp_path / "s", "1790000000-by-path-bbbbbb")
    assert quest_index.find(str(root)).via == "folder"


def test_an_ambiguous_short_id_lists_each_quest_with_its_folder_and_title(tmp_path: Path) -> None:
    a = _quest(tmp_path / "a", "1790000001-first-study-c0ffee", "First study")
    b = _quest(tmp_path / "b", "1790000002-second-study-c0ffee", "Second study")
    quest_index.register(a, title="First study")
    quest_index.register(b, title="Second study")
    with pytest.raises(quest_index.AmbiguousQuest) as e:
        quest_index.find("c0ffee", [tmp_path / "nowhere"])
    text = str(e.value)
    assert "more than one quest" in text
    for needle in ("1790000001-first-study-c0ffee", "First study", str(a.resolve()),
                   "1790000002-second-study-c0ffee", "Second study", str(b.resolve())):
        assert needle in text
    assert len(e.value.candidates) == 2


def test_no_match_says_so_plainly_and_names_close_ones(tmp_path: Path) -> None:
    root = _quest(tmp_path / "a", "1790000001-energy-drift-479b06")
    quest_index.register(root)
    with pytest.raises(quest_index.QuestNotFound) as e:
        quest_index.find("1790000001-energy-drfit-479b06", [tmp_path / "nowhere"])
    text = str(e.value)
    assert "no quest" in text and "1790000001-energy-drift-479b06" in text and "fi tools quests" in text


def test_a_quest_whose_folder_is_gone_is_dropped_when_looked_up(tmp_path: Path) -> None:
    import shutil

    root = _quest(tmp_path / "a", "1790000001-moved-away-123456")
    keep = _quest(tmp_path / "b", "1790000002-still-here-654321")
    quest_index.register(root)
    quest_index.register(keep)
    shutil.rmtree(root)
    with pytest.raises(quest_index.QuestNotFound) as e:
        quest_index.find("123456", [])
    assert "is gone" in str(e.value)
    assert set(quest_index.load()) == {"1790000002-still-here-654321"}


def test_a_moved_quest_is_recorded_again_at_its_new_place(tmp_path: Path) -> None:
    import shutil

    old = _quest(tmp_path / "old", "1790000001-moving-abcdef")
    quest_index.register(old)
    new = tmp_path / "new" / "outputs" / old.name
    new.parent.mkdir(parents=True)
    shutil.move(str(old), str(new))
    quest_index.register(new, working_folder=tmp_path / "new")  # what resuming it from its new folder does
    assert quest_index.find("abcdef").root == new.resolve()


def test_prune_drops_only_the_gone_entries(tmp_path: Path) -> None:
    import shutil

    gone = _quest(tmp_path / "a", "1790000001-gone-111111")
    keep = _quest(tmp_path / "b", "1790000002-kept-222222")
    quest_index.register(gone)
    quest_index.register(keep)
    shutil.rmtree(gone)
    assert quest_index.prune() == ["1790000001-gone-111111"]
    assert list(quest_index.load()) == ["1790000002-kept-222222"]


def test_an_unreadable_index_is_kept_aside_not_written_over(tmp_path: Path) -> None:
    path = quest_index.index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert quest_index.load() == {}
    root = _quest(tmp_path / "a", "1790000001-after-bad-333333")
    quest_index.register(root)
    assert list(quest_index.load()) == ["1790000001-after-bad-333333"]
    kept = list(path.parent.glob("quests.json.unreadable-*"))
    assert kept and kept[0].read_text(encoding="utf-8") == "{ not json"


def test_a_short_id_matching_a_local_quest_and_one_elsewhere_is_ambiguous(tmp_path: Path) -> None:
    _quest(tmp_path / "here", "1790000001-local-abcabc")
    there = _quest(tmp_path / "there", "1790000002-remote-abcabc")
    quest_index.register(there)
    with pytest.raises(quest_index.AmbiguousQuest):
        quest_index.find("abcabc", [tmp_path / "here" / "outputs"])


def test_the_web_lookup_never_takes_a_folder_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _quest(tmp_path, "1790000001-beside-cwd-777777")
    monkeypatch.chdir(root.parent)
    assert quest_index.find(root.name).via == "folder"
    with pytest.raises(quest_index.QuestNotFound):
        quest_index.find(root.name, allow_folder=False)


def test_a_quest_on_a_disconnected_drive_is_left_out_not_forgotten(tmp_path: Path) -> None:
    path = quest_index.index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    offline = "Q:\\\\studies\\\\outputs\\\\1790000001-offline-888888" if os.name == "nt" \
        else "/nonexistent-mount-fi-test/outputs/1790000001-offline-888888"
    path.write_text(json.dumps({"version": 1, "quests": {"1790000001-offline-888888": {"quest_root": offline}}}),
                    encoding="utf-8")
    if Path(offline).parent.exists():
        pytest.skip("the made-up folder exists on this machine")
    assert quest_index.entries() == []
    assert "1790000001-offline-888888" in quest_index.load()


def test_an_index_that_cannot_be_read_is_not_written_over(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = _quest(tmp_path / "a", "1790000001-kept-999999")
    quest_index.register(a)
    before = quest_index.index_path().read_text(encoding="utf-8")

    def locked(_path: Path) -> dict:
        raise PermissionError("held open by another program")

    monkeypatch.setattr(quest_index, "_read", locked)
    b = _quest(tmp_path / "b", "1790000002-new-aaaaaa")
    with pytest.raises(OSError):
        quest_index.register(b)
    assert quest_index.index_path().read_text(encoding="utf-8") == before


def test_rename_updates_the_title_in_the_index(tmp_path: Path) -> None:
    root = _quest(tmp_path / "a", "1790000001-old-name-444444", "Old name")
    quest_index.register(root, title="Old name")
    quest_index.set_title(root, "New name")
    assert quest_index.load()["1790000001-old-name-444444"]["title"] == "New name"


_WRITER = textwrap.dedent("""
    import sys
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    from core import quest_index
    base = Path(sys.argv[2])
    for i in range(int(sys.argv[3])):
        root = base / f"17900000{i:02d}-writer-{sys.argv[4]}{i:04x}"
        (root / ".fi").mkdir(parents=True, exist_ok=True)
        quest_index.register(root, title=f"quest {i}")
""")


def test_two_processes_writing_at_once_keep_every_entry(tmp_path: Path) -> None:
    n = 25
    procs = [
        subprocess.Popen([sys.executable, "-c", _WRITER, str(REPO), str(tmp_path / tag), str(n), tag],
                         env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for tag in ("aa", "bb", "cc")
    ]
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err.decode(errors="replace")
    quests = quest_index.load()
    assert len(quests) == 3 * n
    for tag in ("aa", "bb", "cc"):
        assert sum(1 for q in quests if f"-writer-{tag}" in q) == n
    assert not list(quest_index.index_path().parent.glob("quests.json.*.tmp"))


def test_entries_and_listing_show_short_id_status_title_and_folder(tmp_path: Path) -> None:
    root = _quest(tmp_path / "a", "1790000001-listed-555555", "Listed study")
    (root / "frontier_insight_summary.json").write_text("{}", encoding="utf-8")
    quest_index.register(root, title="Listed study")
    items = quest_index.entries()
    assert [e.quest_id for e in items] == ["1790000001-listed-555555"]
    text = quest_index.listing(items)
    assert "555555" in text and "finished" in text and "Listed study" in text and str(root.resolve()) in text
