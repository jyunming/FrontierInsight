"""Direct tests for launch.py — CLI parsing, generator failure isolation,
and fleet counter accounting. No real LLM calls; no real quest run."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import launch
from core.config import (
    Config,
    EngineConfig,
    ExecutionConfig,
    KnowledgeConfig,
    OutputConfig,
    ProviderConfig,
)
from core.engine import QuestArtifacts


# ---- parse_args ----------------------------------------------------------


def test_parse_args_config_only() -> None:
    args = launch.parse_args(["--config", "x.yaml"])
    assert args.config == Path("x.yaml")
    assert args.fleet is None


def test_parse_args_fleet_multiple() -> None:
    args = launch.parse_args(["--fleet", "a.yaml", "b.yaml"])
    assert args.fleet == [Path("a.yaml"), Path("b.yaml")]
    assert args.config is None


def test_parse_args_no_mode_raises_systemexit() -> None:
    with pytest.raises(SystemExit):
        launch.parse_args([])


def test_parse_args_config_and_fleet_mutually_exclusive() -> None:
    """Both --config and --fleet at once must be rejected."""
    with pytest.raises(SystemExit):
        launch.parse_args(["--config", "x.yaml", "--fleet", "a.yaml"])


def test_parse_args_profile_and_memory_cap() -> None:
    args = launch.parse_args(
        ["--config", "x.yaml", "--profile", "--memory-cap-mb", "2048"]
    )
    assert args.profile is True
    assert args.memory_cap_mb == 2048


def test_parse_args_resume_requires_config() -> None:
    """--resume without --config is rejected (fleet/serve/ingest can't resume)."""
    with pytest.raises(SystemExit):
        launch.parse_args(["--fleet", "a.yaml", "--resume", "some-quest-id"])


def test_parse_args_resume_with_config_ok() -> None:
    args = launch.parse_args(["--config", "x.yaml", "--resume", "q-123"])
    assert args.resume == "q-123"


def test_parse_args_summarize_mode() -> None:
    """--summarize is its own top-level mode; takes a folder Path."""
    args = launch.parse_args(["--summarize", "./papers"])
    assert args.summarize == Path("./papers")
    assert args.summarize_kind == "auto"


def test_parse_args_summarize_with_kind() -> None:
    args = launch.parse_args(["--summarize", ".", "--summarize-kind", "literature"])
    assert args.summarize_kind == "literature"


def test_parse_args_summarize_rejects_invalid_kind() -> None:
    """Unknown kinds rejected at the parser level (argparse `choices=`
    does the work) so we never reach the summarizer with a string
    that can't drive the prompt."""
    with pytest.raises(SystemExit):
        launch.parse_args([
            "--summarize", ".", "--summarize-kind", "not-a-real-kind",
        ])


def test_parse_args_summarize_mutually_exclusive_with_config() -> None:
    with pytest.raises(SystemExit):
        launch.parse_args(["--config", "x.yaml", "--summarize", "./papers"])


def test_parse_args_install_tectonic() -> None:
    """--install-tectonic is its own mode, no config/fleet needed."""
    args = launch.parse_args(["--install-tectonic"])
    assert args.install_tectonic is True


def test_parse_args_install_tectonic_mutually_exclusive() -> None:
    """Like --ingest/--serve, --install-tectonic is in the mode group."""
    with pytest.raises(SystemExit):
        launch.parse_args(["--install-tectonic", "--config", "x.yaml"])


def test_install_tectonic_unsupported_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """If sys.platform / arch isn't in _TECTONIC_ASSET_NAMES, abort
    with a clear message rather than try a garbage URL."""
    monkeypatch.setattr(launch.sys, "platform", "freebsd14")
    import platform
    monkeypatch.setattr(platform, "machine", lambda: "powerpc64")
    rc = launch._install_tectonic()
    assert rc == 1


def test_install_tectonic_skips_when_already_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `tools/tectonic.exe` already exists, the function returns 0
    immediately without re-downloading."""
    # Pin REPO_ROOT to tmp_path; create the tools/ marker file.
    fake_exe = "tectonic.exe" if launch.sys.platform == "win32" else "tectonic"
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / fake_exe).write_bytes(b"already-installed")

    # Pretend launch.py lives at tmp_path/launch.py so repo_root resolves
    # to tmp_path. The function reads `Path(__file__).resolve().parent`,
    # which we'd have to patch via a module-level shim; simpler: monkey-
    # patch the inner Path call by overriding sys.modules entry. The
    # cleanest patch is to set launch's `__file__` indirectly via
    # `Path(launch.__file__).resolve().parent` — we shim the module's
    # path attribute.
    monkeypatch.setattr(
        launch, "__file__", str(tmp_path / "launch.py"),
    )

    rc = launch._install_tectonic()
    assert rc == 0


def test_install_tectonic_rejects_sha_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mocked download where the archive content doesn't match the
    SHA-256 listed in the fake SHA256SUMS file → return 1, leave no
    binary on disk."""
    monkeypatch.setattr(launch, "__file__", str(tmp_path / "launch.py"))
    monkeypatch.setattr(launch.sys, "platform", "win32")
    import platform
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")

    asset_name = launch._TECTONIC_ASSET_NAMES[("win32", "AMD64")]
    # SHA256SUMS lists a fixed hash; the archive bytes hash differently.
    sums_body = f"deadbeef{'0' * 56}  {asset_name}\n".encode("utf-8")

    class _FakeResp:
        def __init__(self, body: bytes) -> None:
            self._body = body
        def __enter__(self):  # noqa: ANN101
            return self
        def __exit__(self, *_a):  # noqa: ANN001
            return False
        def read(self) -> bytes:
            return self._body

    def fake_urlopen(url, **_kw):  # noqa: ANN001
        if url.endswith("SHA256SUMS"):
            return _FakeResp(sums_body)
        # The archive itself — return body whose SHA doesn't match.
        return _FakeResp(b"WRONG-BODY-WONT-MATCH-HASH")

    monkeypatch.setattr(
        "urllib.request.urlopen", fake_urlopen,
    )

    rc = launch._install_tectonic()
    assert rc == 1
    tectonic_exe = tmp_path / "tools" / (
        "tectonic.exe" if launch.sys.platform == "win32" else "tectonic"
    )
    assert not tectonic_exe.exists(), "binary should NOT land on disk after checksum failure"


# ---- _validate_resume_quest_id ------------------------------------------


def test_validate_resume_quest_id_rejects_path_separators(tmp_path: Path) -> None:
    """Path-traversal hardening: the bot review on PR #26 flagged that any
    string was accepted. Reject anything outside the strict alphabet."""
    for bad in ("../etc/passwd", "..\\windows", "a/b", "a\\b", "../sibling"):
        err = launch._validate_resume_quest_id(bad, tmp_path)
        assert err is not None, f"should reject {bad!r}"
        assert "invalid quest id" in err or "outside" in err


def test_validate_resume_quest_id_rejects_missing_checkpoint(tmp_path: Path) -> None:
    """Well-formed id but no prior run → return a clear error rather than
    silently creating an empty quest dir under that name."""
    err = launch._validate_resume_quest_id("nonexistent-quest-id", tmp_path)
    assert err is not None
    assert "no checkpoint" in err


def test_validate_resume_quest_id_accepts_real_checkpoint(tmp_path: Path) -> None:
    """A valid id + an existing .fi/state.sqlite returns None (accept)."""
    qid = "1700000000-resume-good-cafe11"
    fi_dir = tmp_path / qid / ".fi"
    fi_dir.mkdir(parents=True)
    (fi_dir / "state.sqlite").write_bytes(b"")
    assert launch._validate_resume_quest_id(qid, tmp_path) is None


# ---- _run_generators -----------------------------------------------------


def _make_cfg(tmp_path: Path) -> Config:
    return Config(
        topic="launch unit",
        title="launch-unit",
        provider=ProviderConfig(name="openai"),
        engine=EngineConfig(max_iterations=1, review_loop=False),
        execution=ExecutionConfig(sandbox="venv", timeout_s=30),
        knowledge=KnowledgeConfig(enabled=False),
        output=OutputConfig(output_dir=tmp_path / "outputs"),
    )


def _make_artifacts(tmp_path: Path) -> QuestArtifacts:
    quest_root = tmp_path / "quest"
    quest_root.mkdir(parents=True, exist_ok=True)
    return QuestArtifacts(
        quest_id="qid-test",
        quest_root=quest_root,
        paper_md=quest_root / "paper" / "paper.md",
        figures_dir=quest_root / "figures",
    )


async def test_run_generators_continues_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One generator raising must not abort the other three."""
    cfg = _make_cfg(tmp_path)
    art = _make_artifacts(tmp_path)
    supervisor = MagicMock()

    def boom_paper(self, art, out_dir):  # noqa: ANN001
        raise RuntimeError("paper exploded")

    async def ok_slides(self, art, out_dir, *, supervisor):  # noqa: ANN001
        return {"slides_md": out_dir / "slides.md"}

    async def boom_poster(self, art, out_dir, *, supervisor):  # noqa: ANN001
        raise RuntimeError("poster exploded")

    async def ok_speech(self, art, out_dir, *, supervisor):  # noqa: ANN001
        return {"speech_md": out_dir / "speech.md"}

    monkeypatch.setattr("launch.PaperGenerator.generate", boom_paper)
    monkeypatch.setattr("launch.SlideGenerator.generate", ok_slides)
    monkeypatch.setattr("launch.PosterGenerator.generate", boom_poster)
    monkeypatch.setattr("launch.SpeechGenerator.generate", ok_speech)

    written = await launch._run_generators(cfg, art, supervisor=supervisor)

    # Both surviving generators contributed; failed ones did not abort the run.
    assert "slides_md" in written
    assert "speech_md" in written
    assert "paper_md" not in written  # paper raised
    assert "poster_md" not in written  # poster raised


# ---- _maybe_profiled -----------------------------------------------------


async def test_maybe_profiled_no_profile_calls_run_directly() -> None:
    engine = MagicMock()
    engine.run = AsyncMock(return_value="sentinel")
    result = await launch._maybe_profiled(engine, profile=False)
    assert result == "sentinel"
    engine.run.assert_awaited_once()


async def test_maybe_profiled_viztracer_missing_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If viztracer isn't installed, _maybe_profiled prints a warning and
    still runs the engine."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):  # noqa: ANN001
        if name == "viztracer":
            raise ImportError("not installed")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    engine = MagicMock()
    engine.run = AsyncMock(return_value="sentinel-no-viz")
    result = await launch._maybe_profiled(engine, profile=True)
    assert result == "sentinel-no-viz"
    engine.run.assert_awaited_once()


# ---- _await_under_cap ----------------------------------------------------


async def test_await_under_cap_returns_immediately_when_psutil_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """psutil-missing path: _rss_mb returns None, _await_under_cap returns."""
    monkeypatch.setattr(launch, "_rss_mb", lambda: None)
    # Tight cap that would otherwise block; should still return immediately.
    await launch._await_under_cap(cap_mb=1, poll_s=0.01)


async def test_await_under_cap_returns_when_under_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launch, "_rss_mb", lambda: 100.0)
    await launch._await_under_cap(cap_mb=4096, poll_s=0.01)


# ---- run_fleet counters --------------------------------------------------


async def test_run_fleet_counters_at_end_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two fake quests both succeed: done=2, failed=0, return code 0."""
    cfg_a = _make_cfg(tmp_path)
    cfg_b = _make_cfg(tmp_path)

    async def fake_run_one(cfg, *, supervisor, profile, engine=None, **_kw):  # noqa: ANN001
        # Engine is now constructed in gated() and passed through — accept it.
        qid = engine.quest_id if engine is not None else "q-" + cfg.title
        return {"quest_id": qid, "ok": True}

    monkeypatch.setattr(launch, "run_one", fake_run_one)

    printed: list[str] = []
    monkeypatch.setattr(
        "builtins.print",
        lambda *a, **kw: printed.append(" ".join(str(x) for x in a)),
    )

    rc = await launch.run_fleet(
        [cfg_a, cfg_b],
        supervisor=MagicMock(),
        max_concurrent=2,
        memory_cap_mb=None,
        profile=False,
    )
    assert rc == 0
    done_lines = [ln for ln in printed if "done " in ln and "/2" in ln]
    assert any("done=2/2" in ln for ln in done_lines)
    assert all("failed=0" in ln for ln in done_lines)


async def test_run_fleet_failure_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One quest raises, the other succeeds: rc=1, failed counted, both ran."""
    cfg_a = _make_cfg(tmp_path)
    cfg_b = _make_cfg(tmp_path)

    # First call raises, subsequent calls succeed. max_concurrent=1
    # serializes the gather so order is deterministic.
    state = {"first": True}

    async def fake_run_one(cfg, *, supervisor, profile, engine=None, **_kw):  # noqa: ANN001
        if state["first"]:
            state["first"] = False
            raise RuntimeError("first quest exploded")
        qid = engine.quest_id if engine is not None else "q-ok"
        return {"quest_id": qid, "ok": True}

    monkeypatch.setattr(launch, "run_one", fake_run_one)
    monkeypatch.setattr("builtins.print", lambda *a, **kw: None)

    rc = await launch.run_fleet(
        [cfg_a, cfg_b],
        supervisor=MagicMock(),
        max_concurrent=1,
        memory_cap_mb=None,
        profile=False,
    )
    assert rc == 1


# ---- source_yaml_path → <quest_root>/config.yaml -------------------------


async def test_run_one_copies_source_yaml_into_quest_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User complaint: `/resume` had to slug-match YAMLs in `_drafts/`.
    The fix drops a copy of the source YAML at `<quest_root>/config.yaml`
    on quest startup so future resumes can find the config trivially.
    Pin that behavior."""
    cfg = _make_cfg(tmp_path)
    src_yaml = tmp_path / "my_quest.yaml"
    src_yaml.write_text("topic: copied yaml\n", encoding="utf-8")

    # Stub engine so we don't actually run the LangGraph.
    fake_engine = MagicMock()
    fake_engine.quest_id = "qid-copy-test"
    fake_engine.quest_root = tmp_path / "outputs" / "qid-copy-test"

    # Stub Engine() so run_one's no-engine path uses our stub.
    monkeypatch.setattr(launch, "Engine", lambda *_a, **_kw: fake_engine)

    # Stub _maybe_profiled to skip the whole engine.run path.
    fake_art = MagicMock()
    fake_art.quest_id = fake_engine.quest_id
    fake_art.quest_root = fake_engine.quest_root
    fake_art.paper_md = None

    async def fake_maybe(*_a, **_kw):  # noqa: ANN001
        fake_art.quest_root.mkdir(parents=True, exist_ok=True)
        return fake_art

    monkeypatch.setattr(launch, "_maybe_profiled", fake_maybe)
    monkeypatch.setattr(launch, "_pick_clarify_callback", lambda *a, **kw: None)
    monkeypatch.setattr(launch, "_run_generators", AsyncMock(return_value={}))

    await launch.run_one(
        cfg, supervisor=MagicMock(), source_yaml_path=src_yaml,
    )

    copied = fake_engine.quest_root / "config.yaml"
    assert copied.is_file(), "expected config.yaml dropped into quest_root"
    assert copied.read_text(encoding="utf-8") == "topic: copied yaml\n"


async def test_run_one_skips_copy_when_dest_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On --resume the destination already exists from the original run;
    we must NOT clobber it (preserves the user's edits if they tweaked
    the YAML between runs)."""
    cfg = _make_cfg(tmp_path)
    src_yaml = tmp_path / "new.yaml"
    src_yaml.write_text("topic: new\n", encoding="utf-8")

    quest_root = tmp_path / "outputs" / "qid-preserve"
    quest_root.mkdir(parents=True)
    (quest_root / "config.yaml").write_text("topic: original\n", encoding="utf-8")

    fake_engine = MagicMock()
    fake_engine.quest_id = "qid-preserve"
    fake_engine.quest_root = quest_root

    monkeypatch.setattr(launch, "Engine", lambda *_a, **_kw: fake_engine)
    fake_art = MagicMock()
    fake_art.quest_id = fake_engine.quest_id
    fake_art.quest_root = fake_engine.quest_root
    fake_art.paper_md = None

    async def fake_maybe(*_a, **_kw):  # noqa: ANN001
        return fake_art

    monkeypatch.setattr(launch, "_maybe_profiled", fake_maybe)
    monkeypatch.setattr(launch, "_pick_clarify_callback", lambda *a, **kw: None)
    monkeypatch.setattr(launch, "_run_generators", AsyncMock(return_value={}))

    await launch.run_one(
        cfg, supervisor=MagicMock(), source_yaml_path=src_yaml,
    )

    preserved = (quest_root / "config.yaml").read_text(encoding="utf-8")
    assert preserved == "topic: original\n", "must not clobber existing config.yaml"


# ---- _apply_paper_venue_override ----------------------------------------


def _art_with_venue(tmp_path: Path, venue: object) -> QuestArtifacts:
    return QuestArtifacts(
        quest_id="q-x", quest_root=tmp_path,
        raw_state={"clarify_answers": {"paper_venue": venue}},
    )


def test_paper_venue_override_applies_when_yaml_is_default(tmp_path: Path) -> None:
    """User wrote `paper_format: generic` (or left default) AND clarify
    picked `neurips` — override applies."""
    cfg = _make_cfg(tmp_path)
    assert cfg.output.paper_format == "generic"
    launch._apply_paper_venue_override(cfg, _art_with_venue(tmp_path, "neurips"))
    assert cfg.output.paper_format == "neurips"


def test_paper_venue_override_respects_explicit_yaml(tmp_path: Path) -> None:
    """User explicitly set `paper_format: ieee_access` in YAML — clarify
    must NOT silently override that."""
    cfg = _make_cfg(tmp_path)
    cfg.output.paper_format = "ieee_access"
    launch._apply_paper_venue_override(cfg, _art_with_venue(tmp_path, "neurips"))
    assert cfg.output.paper_format == "ieee_access"


def test_paper_venue_override_ignores_unknown_venue(tmp_path: Path) -> None:
    """A clarify answer like `paper_venue: 'some-future-template'` must
    be silently dropped — pydantic would otherwise reject it."""
    cfg = _make_cfg(tmp_path)
    launch._apply_paper_venue_override(
        cfg, _art_with_venue(tmp_path, "made-up-venue"),
    )
    assert cfg.output.paper_format == "generic"


def test_paper_venue_override_handles_missing_clarify_answers(tmp_path: Path) -> None:
    """clarify_mode=off → no clarify_answers → must not raise."""
    cfg = _make_cfg(tmp_path)
    art = QuestArtifacts(quest_id="q", quest_root=tmp_path, raw_state={})
    launch._apply_paper_venue_override(cfg, art)
    assert cfg.output.paper_format == "generic"


def test_main_catches_keyboard_interrupt_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """Ctrl-C during --serve, --fleet, or a long quest used to dump a
    multi-frame traceback (CancelledError + KeyboardInterrupt) that
    looked like a crash. The fix wraps ``asyncio.run`` in
    try/except KeyboardInterrupt and exits with rc=130 + a clean
    goodbye message. Pin both."""
    monkeypatch.setattr("sys.argv", ["launch.py", "--serve"])

    def boom(_):
        raise KeyboardInterrupt()

    monkeypatch.setattr("launch.asyncio.run", boom)

    rc = launch.main()
    assert rc == 130, "POSIX conventional exit code for SIGINT"
    captured = capsys.readouterr()
    assert "interrupted" in captured.err.lower()
    # No tracebacks in the captured streams.
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out
