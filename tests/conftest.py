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
