"""Direct tests for `core.summarizer`.

The summarizer is one big async function over a folder. Unit-tested
here at a fine granularity: file walker, extension classifier,
content-kind detector, prompt rendering, Axon ingestion call shape,
plus an end-to-end with a mocked LLM. No real LLM calls; no real Axon.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.config import ProviderConfig
from core.summarizer import (
    FileEntry,
    _classify_extension,
    _detect_folder_kind,
    _ingest_to_axon,
    _new_summary_id,
    _render_content_blocks,
    _render_file_manifest,
    _slugify_folder,
    _walk_folder,
    summarize_folder,
)


# ---- _classify_extension ------------------------------------------------


def test_classify_extension_papers() -> None:
    for ext in (".pdf", ".md", ".txt", ".rst"):
        assert _classify_extension(ext) == "paper"


def test_classify_extension_code() -> None:
    for ext in (".py", ".ts", ".rs", ".go", ".java", ".cpp"):
        assert _classify_extension(ext) == "code"


def test_classify_extension_logs() -> None:
    for ext in (".log", ".jsonl", ".csv"):
        assert _classify_extension(ext) == "log"


def test_classify_extension_unknown() -> None:
    assert _classify_extension(".xyz") == "other"
    assert _classify_extension("") == "other"


def test_classify_extension_is_case_insensitive() -> None:
    assert _classify_extension(".PDF") == "paper"
    assert _classify_extension(".PY") == "code"


# ---- _detect_folder_kind ------------------------------------------------


def _entry(kind: str, ident: int = 1) -> FileEntry:
    return FileEntry(
        ident=ident, path=Path("/fake"), rel_path=f"f{ident}.x",
        kind=kind, size_bytes=100,
    )


def test_detect_folder_kind_pure_papers_is_literature() -> None:
    entries = [_entry("paper", i) for i in range(5)]
    assert _detect_folder_kind(entries) == "literature"


def test_detect_folder_kind_pure_code_is_code() -> None:
    entries = [_entry("code", i) for i in range(5)]
    assert _detect_folder_kind(entries) == "code"


def test_detect_folder_kind_pure_logs_is_execution() -> None:
    entries = [_entry("log", i) for i in range(5)]
    assert _detect_folder_kind(entries) == "execution"


def test_detect_folder_kind_balanced_code_papers_is_study() -> None:
    """Mixed code + papers → study project (the user's research-folder case)."""
    entries = (
        [_entry("paper", i) for i in range(3)] +
        [_entry("code", i + 3) for i in range(3)]
    )
    assert _detect_folder_kind(entries) == "study"


def test_detect_folder_kind_balanced_code_logs_is_execution() -> None:
    entries = (
        [_entry("code", i) for i in range(2)] +
        [_entry("log", i + 2) for i in range(3)]
    )
    assert _detect_folder_kind(entries) == "execution"


def test_detect_folder_kind_empty_is_mixed() -> None:
    assert _detect_folder_kind([]) == "mixed"


def test_detect_folder_kind_only_configs_is_mixed() -> None:
    """Config-only folders don't have a dominant content signal."""
    entries = [_entry("config", i) for i in range(3)]
    assert _detect_folder_kind(entries) == "mixed"


# ---- _walk_folder -------------------------------------------------------


def test_walk_folder_skips_noise_dirs(tmp_path: Path) -> None:
    """`.git`, `node_modules`, `__pycache__`, `outputs/`, `.fi`, dot-dirs
    must all be excluded — they're noise for summarization."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("gitconfig")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "leftpad.js").write_text("ok")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "foo.cpython-311.pyc").write_bytes(b"\x00")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "prior.md").write_text("old quest")
    (tmp_path / "real.py").write_text("print('keep me')")
    (tmp_path / "readme.md").write_text("keep me too")

    entries = _walk_folder(tmp_path)
    paths = {e.rel_path for e in entries}
    assert paths == {"real.py", "readme.md"}


def test_walk_folder_ignores_dotfiles(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("SECRET=42")
    (tmp_path / ".gitignore").write_text("*.pyc")
    (tmp_path / "real.md").write_text("hi")
    entries = _walk_folder(tmp_path)
    assert {e.rel_path for e in entries} == {"real.md"}


def test_walk_folder_recurses_into_subdirs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("a = 1")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "intro.md").write_text("# intro")
    entries = _walk_folder(tmp_path)
    assert {e.rel_path for e in entries} == {"src/a.py", "docs/intro.md"}


def test_walk_folder_assigns_deterministic_idents(tmp_path: Path) -> None:
    """File IDs must be stable across runs for citations to be
    reproducible (tests rely on this)."""
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "c.md").write_text("c")
    entries = _walk_folder(tmp_path)
    # Sorted walk → alphabetical idents.
    assert [e.rel_path for e in entries] == ["a.md", "b.md", "c.md"]
    assert [e.ident for e in entries] == [1, 2, 3]


def test_walk_folder_raises_on_non_directory(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        _walk_folder(f)


def test_walk_folder_loads_previews_for_text_files(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("print('hello world')")
    (tmp_path / "doc.md").write_text("# heading\n\nbody")
    entries = _walk_folder(tmp_path)
    by_name = {e.rel_path: e for e in entries}
    assert "print('hello world')" in by_name["code.py"].preview
    assert "heading" in by_name["doc.md"].preview


# ---- prompt rendering ---------------------------------------------------


def test_render_file_manifest_empty(tmp_path: Path) -> None:
    assert _render_file_manifest([]) == "(empty — no files found)"


def test_render_file_manifest_table_shape(tmp_path: Path) -> None:
    entries = [
        FileEntry(ident=1, path=Path("/fake/a.py"), rel_path="a.py",
                  kind="code", size_bytes=2048),       # exactly 2 KB
        FileEntry(ident=2, path=Path("/fake/b.md"), rel_path="b.md",
                  kind="paper", size_bytes=512),        # ceil → 1 KB
        FileEntry(ident=3, path=Path("/fake/c.md"), rel_path="c.md",
                  kind="paper", size_bytes=1536),       # ceil → 2 KB (PR #37 bot)
        FileEntry(ident=4, path=Path("/fake/d.md"), rel_path="d.md",
                  kind="paper", size_bytes=0),          # empty → 0 KB
    ]
    out = _render_file_manifest(entries)
    assert "| ID | path | kind | size_kb |" in out
    assert "| [1] | `a.py` | code | 2 |" in out
    assert "| [2] | `b.md` | paper | 1 |" in out
    assert "| [3] | `c.md` | paper | 2 |" in out   # ceil math, not floor
    assert "| [4] | `d.md` | paper | 0 |" in out


def test_render_content_blocks_skips_empty_previews() -> None:
    entries = [
        FileEntry(ident=1, path=Path("/x.py"), rel_path="x.py", kind="code",
                  size_bytes=10, preview="print('hi')"),
        FileEntry(ident=2, path=Path("/y.bin"), rel_path="y.bin", kind="other",
                  size_bytes=10, preview=""),
    ]
    out = _render_content_blocks(entries)
    assert "[1]" in out and "print('hi')" in out
    assert "[2]" not in out


def test_render_content_blocks_empty_returns_marker() -> None:
    entries = [
        FileEntry(ident=1, path=Path("/y.bin"), rel_path="y.bin", kind="other",
                  size_bytes=10, preview=""),
    ]
    assert _render_content_blocks(entries) == "(no readable content found)"


# ---- ID + slug ----------------------------------------------------------


def test_slugify_folder_basics() -> None:
    # Trailing dashes (from non-alnum suffix chars) are stripped.
    assert _slugify_folder(Path("/tmp/My Project!")) == "my-project"
    assert _slugify_folder(Path("/tmp/papers")) == "papers"
    # Empty / all-punct → safe fallback.
    assert _slugify_folder(Path("/tmp/!!!")) == "summary"


def test_new_summary_id_includes_slug() -> None:
    qid = _new_summary_id(Path("/tmp/my-cool-folder"))
    assert qid.startswith("summary-")
    assert "my-cool-folder" in qid


def test_new_summary_id_unique() -> None:
    ids = {_new_summary_id(Path("/x")) for _ in range(20)}
    assert len(ids) == 20


# ---- _ingest_to_axon ----------------------------------------------------


def test_ingest_to_axon_calls_add_text_per_file_and_summary() -> None:
    """Each input file with non-empty axon_content → one fi_summary_input
    call; the summary itself → one fi_summary call. Total = N+1."""
    k = MagicMock()
    k.add_text = MagicMock(return_value=True)
    entries = [
        FileEntry(ident=1, path=Path("/x.py"), rel_path="x.py", kind="code",
                  size_bytes=10, preview="print(1)", axon_content="print(1)"),
        FileEntry(ident=2, path=Path("/y.md"), rel_path="y.md", kind="paper",
                  size_bytes=20, preview="hello", axon_content="hello full body"),
        FileEntry(ident=3, path=Path("/z.bin"), rel_path="z.bin", kind="other",
                  size_bytes=30, preview="", axon_content=""),  # skipped
    ]
    ok = _ingest_to_axon(
        k, summary_id="sum-1", folder=Path("/work"),
        detected_kind="study", entries=entries,
        summary_markdown="# summary\n",
    )
    assert ok is True
    # 2 fi_summary_input + 1 fi_summary = 3 calls.
    assert k.add_text.call_count == 3
    kinds_seen = [
        call.kwargs.get("kind") for call in k.add_text.call_args_list
    ]
    assert kinds_seen.count("fi_summary_input") == 2
    assert kinds_seen.count("fi_summary") == 1


def test_ingest_to_axon_uses_full_axon_content_not_prompt_preview() -> None:
    """Regression for PR #37 review: ingest must use the larger
    ``axon_content`` field so Axon receives more than the 4 KB the
    prompt saw. The preview / axon_content split is the whole point
    of the two caps."""
    k = MagicMock()
    captured: list[str] = []
    k.add_text = MagicMock(
        side_effect=lambda **kw: (captured.append(kw["text"]) or True),
    )
    big = "X" * 40_000
    entries = [
        FileEntry(ident=1, path=Path("/big.md"), rel_path="big.md", kind="paper",
                  size_bytes=40_000, preview=big[:4_000], axon_content=big),
    ]
    _ingest_to_axon(
        k, summary_id="sum-1", folder=Path("/work"),
        detected_kind="literature", entries=entries,
        summary_markdown="# s\n",
    )
    # The first call (fi_summary_input) carried the FULL axon_content,
    # not the truncated preview.
    assert len(captured[0]) == 40_000


def test_ingest_to_axon_returns_false_when_knowledge_returns_false() -> None:
    """`Knowledge.add_text` returns False on failure — propagate that
    so the caller can surface 'partial ingest' to the user."""
    k = MagicMock()
    k.add_text = MagicMock(return_value=False)
    entries = [
        FileEntry(ident=1, path=Path("/x.py"), rel_path="x.py", kind="code",
                  size_bytes=10, preview="hi", axon_content="hi"),
    ]
    ok = _ingest_to_axon(
        k, summary_id="sum-2", folder=Path("/work"),
        detected_kind="code", entries=entries,
        summary_markdown="# x\n",
    )
    assert ok is False


# ---- summarize_folder end-to-end (with mocked LLM + Axon) ---------------


@pytest.mark.asyncio
async def test_summarize_folder_writes_summary_md_and_returns_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end with fake LLM: walks folder, calls LLM once, writes
    summary.md, returns SummaryArtifacts with the expected fields."""
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "paper.md").write_text("# A toy paper\n\nIt says things.")
    (folder / "code.py").write_text("x = 1\n")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    captured: dict[str, str] = {}

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        captured["prompt"] = messages[-1]["content"]
        captured["node"] = kw.get("node", "")
        return "# Folder Summary\n\nbody.\n"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    # Mock the Knowledge handle: enabled + add_text returns True so
    # the summarizer reports `ingested_to_axon=True`.
    knowledge = MagicMock()
    knowledge.enabled = True
    knowledge.add_text = MagicMock(return_value=True)

    art = await summarize_folder(
        folder,
        provider=ProviderConfig(name="openai"),
        output_dir=output_dir,
        kind="auto",
        knowledge=knowledge,
    )

    # The summary file exists and has our canned body.
    assert art.summary_path.is_file()
    body = art.summary_path.read_text(encoding="utf-8")
    assert body.startswith("# Folder Summary")
    # 1 paper + 1 code → "study" by the detector (paper + code both ≥ 20%).
    assert art.detected_kind == "study"
    assert art.file_count == 2
    assert art.ingested_to_axon is True
    # Prompt included our manifest IDs and the canned content.
    assert "[1]" in captured["prompt"]
    assert "[2]" in captured["prompt"]
    assert "A toy paper" in captured["prompt"]
    assert captured["node"] == "summarize"


@pytest.mark.asyncio
async def test_summarize_folder_explicit_kind_overrides_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User-passed kind beats the heuristic — important when the user
    knows better than the file-mix detector."""
    folder = tmp_path / "in"
    folder.mkdir()
    (folder / "only.md").write_text("hi")  # would auto-detect literature
    output_dir = tmp_path / "out"

    async def fake_chat(self, messages, **kw):  # noqa: ANN001
        return "# Forced Kind\n"

    monkeypatch.setattr("core.provider.LLMClient.chat", fake_chat)

    art = await summarize_folder(
        folder,
        provider=ProviderConfig(name="openai"),
        output_dir=output_dir,
        kind="execution",       # explicit override
        knowledge=None,
    )
    assert art.detected_kind == "execution"
    assert art.ingested_to_axon is False  # no Knowledge supplied


@pytest.mark.asyncio
async def test_summarize_folder_raises_on_invalid_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        await summarize_folder(
            tmp_path,
            provider=ProviderConfig(name="openai"),
            output_dir=tmp_path / "out",
            kind="totally-made-up",
        )


@pytest.mark.asyncio
async def test_summarize_folder_raises_on_missing_dir(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(NotADirectoryError):
        await summarize_folder(
            missing,
            provider=ProviderConfig(name="openai"),
            output_dir=tmp_path / "out",
        )
