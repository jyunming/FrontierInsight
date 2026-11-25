"""What counts as the person's data in a study with no simulation: FI's own folders under data/ never do."""
from __future__ import annotations

from pathlib import Path

from core.engine import _list_user_data_files


def test_papers_fi_wrote_into_data_literature_are_not_the_persons_data(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "literature" / ".full_text").mkdir(parents=True)
    (data / "literature" / "lit_001_a.md").write_text("a paper", encoding="utf-8")
    (data / "literature" / ".full_text" / "k.txt").write_text("its text", encoding="utf-8")
    (data / "auto_collected").mkdir()
    (data / "auto_collected" / "page.md").write_text("an old page", encoding="utf-8")
    assert _list_user_data_files(data) == [], "only FI's own folders: the study still waits for data"
    (data / "table.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    assert [p.name for p in _list_user_data_files(data)] == ["table.csv"]
