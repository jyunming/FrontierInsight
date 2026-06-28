"""Tests for relevance-ranked passage selection (core/passages.py).

All deterministic + offline: the lexical ranker needs no model or network.
"""

from core.passages import chunk_text, select_relevant_excerpt, _lexical_scores


def test_chunk_text_packs_paragraphs_and_preserves_order() -> None:
    text = "First para.\n\nSecond para.\n\nThird para."
    chunks = chunk_text(text, chunk_chars=20)
    # Each para is < 20 chars but packing two would exceed it → one per chunk.
    assert chunks == ["First para.", "Second para.", "Third para."]


def test_chunk_text_hard_splits_overlong_paragraph() -> None:
    text = "x" * 2500  # single paragraph, no breaks
    chunks = chunk_text(text, chunk_chars=1000)
    assert [len(c) for c in chunks] == [1000, 1000, 500]
    assert "".join(chunks) == text


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_select_relevant_surfaces_buried_passage() -> None:
    """The relevant passage sits past the first `budget` chars; selection
    must surface it where a naive first-N slice would miss it."""
    head = "Title: Greenspace and Health\n\n" + (
        "Generic framing about cities and wellbeing. " * 60   # pushes finding past 2000
    )
    finding = (
        "\n\nResults: the hazard ratio for all-cause mortality was 0.96 per "
        "0.1 NDVI increase and systolic blood pressure fell 2.3 mmHg.\n\n"
    )
    tail = "Unrelated methodological caveats and acknowledgements. " * 40
    content = head + finding + tail
    query = "all-cause mortality hazard ratio blood pressure NDVI"

    out = select_relevant_excerpt(content, query, budget_chars=2000, mode="lexical")
    assert "0.96 per 0.1 NDVI" in out and "2.3 mmHg" in out
    # The head (title/abstract) is kept for orientation.
    assert out.lstrip().startswith("Title: Greenspace")
    # And the naive first-N slice would NOT have contained the finding.
    assert "0.96 per 0.1 NDVI" not in content[:2000]


def test_select_relevant_degenerate_cases() -> None:
    content = "Some content here.\n\n" + ("more text. " * 100)
    # No query → first-N slice (back-compatible).
    assert select_relevant_excerpt(content, "", budget_chars=50) == content[:50]
    # Content already under budget → returned whole.
    assert select_relevant_excerpt("short", "query", budget_chars=500) == "short"
    # Whitespace-only query → first-N.
    assert select_relevant_excerpt(content, "   ", budget_chars=50) == content[:50]


def test_select_relevant_respects_budget() -> None:
    content = "head\n\n" + "\n\n".join(f"para {i} " + ("word " * 50) for i in range(20))
    out = select_relevant_excerpt(content, "para word", budget_chars=1000, mode="lexical")
    assert len(out) <= 1000


def test_lexical_scores_rank_query_terms() -> None:
    chunks = [
        "completely unrelated text about weather and sports",
        "mortality hazard ratio rose with air pollution exposure",
    ]
    scores = _lexical_scores(chunks, "mortality hazard ratio pollution")
    assert scores[1] > scores[0]
    # An empty query yields no signal.
    assert _lexical_scores(chunks, "") == [0.0, 0.0]


def test_stopwords_keep_scientific_terms() -> None:
    from core.passages import _content_terms
    terms = _content_terms("the study results show an effect and association")
    for w in ("study", "results", "effect", "association"):
        assert w in terms          # load-bearing scientific terms survive
    assert "the" not in terms and "and" not in terms  # function words dropped


def test_hybrid_surfaces_paraphrase_with_mocked_embedder(monkeypatch) -> None:
    import core.passages as p

    class _Fake:
        def encode(self, texts, normalize_embeddings=True):
            import numpy as np
            # query + the paraphrase chunk share a direction; off-topic is orthogonal
            return np.array([[1.0, 0.0] if ("mortality" in t or "death rate" in t)
                             else [0.0, 1.0] for t in texts])

    monkeypatch.setattr(p, "_EMBED_TRIED", True)
    monkeypatch.setattr(p, "_EMBED_MODEL", _Fake())
    chunks = ["off-topic weather and sports coverage",
              "each rise in greenery cut the death rate among elders"]
    scores = p._hybrid_scores(chunks, "mortality")
    assert scores[1] > scores[0]   # paraphrase chunk wins via the embedding term


def test_hybrid_falls_back_to_lexical_offline(monkeypatch) -> None:
    import core.passages as p
    monkeypatch.setenv("FI_OFFLINE", "1")
    monkeypatch.setattr(p, "_EMBED_TRIED", False)
    monkeypatch.setattr(p, "_EMBED_MODEL", None)
    assert p._embed_model() is None   # FI_OFFLINE → no model download
    chunks = ["mortality rose sharply", "weather and sports"]
    assert p._hybrid_scores(chunks, "mortality") == p._lexical_scores(chunks, "mortality")
