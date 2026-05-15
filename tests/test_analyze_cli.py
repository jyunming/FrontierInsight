"""Tests for ``core/analyze_cli.py`` — the helper that powers
``fi --analyze <data_path>``."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.analyze_cli import (
    build_analyze_config,
    mint_quest_id,
    prepare_analyze_quest,
    stage_data,
)


def test_mint_quest_id_shape() -> None:
    """quest_id matches the codebase convention ``<epoch>-<slug>-<nonce>``."""
    qid = mint_quest_id("Compare integrators on a damped oscillator")
    parts = qid.split("-")
    assert len(parts) >= 3, f"expected epoch-slug-nonce shape; got {qid!r}"
    epoch = parts[0]
    nonce = parts[-1]
    assert epoch.isdigit() and len(epoch) >= 10
    assert len(nonce) == 6 and all(c in "0123456789abcdef" for c in nonce)


def test_mint_quest_id_uses_topic_slug() -> None:
    qid = mint_quest_id("EUV stochastics roadmap", now_epoch=1778800000)
    assert qid.startswith("1778800000-euv-stochastics-roadmap-")


def test_stage_data_copies_files_recursively(tmp_path: Path) -> None:
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    (src / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (src / "sub" / "notes.md").write_text("# Notes\n", encoding="utf-8")
    dest = tmp_path / "out"
    n = stage_data(src, dest)
    assert n == 2
    assert (dest / "a.csv").read_text(encoding="utf-8").startswith("x,y")
    assert (dest / "sub" / "notes.md").exists()


def test_stage_data_skips_noise_files(tmp_path: Path) -> None:
    """``.DS_Store``, ``__pycache__/foo.pyc``, ``.git/HEAD`` get skipped
    so we don't bloat the data_load prompt with junk."""
    src = tmp_path / "in"
    src.mkdir()
    (src / "keep.txt").write_text("real", encoding="utf-8")
    (src / ".DS_Store").write_bytes(b"\x00")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
    dest = tmp_path / "out"
    n = stage_data(src, dest)
    assert n == 1
    assert (dest / "keep.txt").exists()
    assert not (dest / ".DS_Store").exists()
    assert not (dest / ".git").exists()
    assert not (dest / "__pycache__").exists()


def test_stage_data_rejects_non_directory(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("not a dir", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an existing directory"):
        stage_data(f, tmp_path / "out")


def test_stage_data_skips_symlinks(tmp_path: Path) -> None:
    """Symlinks aren't followed — a malicious symlink could otherwise
    point at ``/etc/passwd`` and end up in the quest's data dir."""
    if not hasattr(Path, "symlink_to"):
        pytest.skip("symlink_to not available on this platform")
    target = tmp_path / "secret.txt"
    target.write_text("sensitive", encoding="utf-8")
    src = tmp_path / "in"
    src.mkdir()
    try:
        (src / "link.txt").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted on this system")
    (src / "real.csv").write_text("a,b\n", encoding="utf-8")
    dest = tmp_path / "out"
    n = stage_data(src, dest)
    assert n == 1, "only the real file should copy; the symlink is skipped"
    assert (dest / "real.csv").exists()
    assert not (dest / "link.txt").exists()


def test_build_analyze_config_defaults_are_no_simulation(tmp_path: Path) -> None:
    """The config must skip implement/execute and route through
    auto_collect_data → wait_for_data → data_load → analyze."""
    cfg = build_analyze_config(
        topic="Belgium vs Taiwan cultural attitudes",
        output_dir=tmp_path / "outputs",
    )
    assert cfg.engine.no_simulation is True
    # auto_collect_data: False because the user already supplied data;
    # asking Axon to add more would dilute the prompt.
    assert cfg.engine.auto_collect_data is False
    # No clarify call — topic is the source of truth.
    assert cfg.engine.clarify_mode == "off"
    # One pass — analyze workflows usually don't want revise cycles.
    assert cfg.engine.review_loop is False
    assert cfg.engine.enable_analyze_reroute is False
    assert cfg.engine.cross_check_per_finding_k == 0


def test_prepare_analyze_quest_end_to_end(tmp_path: Path) -> None:
    """The public entry point: mints quest_id, stages data, returns
    a working Config + count. No LLM call."""
    src = tmp_path / "data_in"
    src.mkdir()
    (src / "trends.csv").write_text("year,value\n2020,1\n2021,2\n", encoding="utf-8")
    (src / "notes.md").write_text("# Observations\n", encoding="utf-8")
    output_root = tmp_path / "outputs"

    cfg, quest_id, files_staged = prepare_analyze_quest(
        data_path=src, topic="Identify the trend",
        output_root=output_root,
    )
    assert files_staged == 2
    assert cfg.engine.no_simulation is True
    # Data was copied into the new quest dir under output_root.
    staged_dir = output_root.resolve() / quest_id / "data"
    assert (staged_dir / "trends.csv").exists()
    assert (staged_dir / "notes.md").exists()


def test_prepare_analyze_quest_empty_dir_returns_zero(tmp_path: Path) -> None:
    """Empty input directory is allowed by stage_data (returns 0). The
    launch.py CLI surfaces a clear error in that case — pinned at a
    higher layer."""
    src = tmp_path / "empty"
    src.mkdir()
    cfg, quest_id, n = prepare_analyze_quest(
        data_path=src, topic="t", output_root=tmp_path / "outputs",
    )
    assert n == 0
    assert cfg.topic == "t"
