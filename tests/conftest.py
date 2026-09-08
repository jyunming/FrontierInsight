"""Make the repo root importable so `from core...` works inside tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _no_embed_model_download(monkeypatch):
    """Keep the test suite hermetic: never let passage ranking download the
    sentence-transformer model (CI has no cache, and a download is slow +
    network-dependent). Force `_embed_model()` to return None so ranking
    stays on the deterministic lexical path unless a test injects its own
    (mocked) embedder. Real quests are unaffected — they load the model on
    first use."""
    import core.passages as _passages
    monkeypatch.setattr(_passages, "_EMBED_TRIED", True, raising=False)
    monkeypatch.setattr(_passages, "_EMBED_MODEL", None, raising=False)


@pytest.fixture(autouse=True)
def _isolate_skill_library(tmp_path_factory, monkeypatch):
    """No test may read the developer's real skill library.

    Skills live outside the repo, in ``FI_SKILLS_DIR`` (default
    ``~/.frontier-insight/skills``), because they are per-machine state
    rather than repository content. Nothing pointed the tests away from it,
    so every engine test discovered whatever happened to be installed —
    invisible while that was one skill, and decisive once it was 34: engine
    smoke tests began exercising real discovery, running each skill's
    self-test, and making a live selection call. Three of them failed on
    behaviour that had nothing to do with what they were testing, and the
    full sweep went from ~20 minutes to 1h55.

    A test that needs skills sets ``FI_SKILLS_DIR`` itself; the default is
    an empty directory so results depend on the repository, not the machine.
    """
    empty = tmp_path_factory.mktemp("fi_skills_empty")
    monkeypatch.setenv("FI_SKILLS_DIR", str(empty))
    # The approval ledger lives outside the repo too, and an approval
    # recorded by a test must never reach the real one.
    monkeypatch.setenv(
        "FI_SKILLS_APPROVALS", str(empty.parent / "approvals.json"),
    )
