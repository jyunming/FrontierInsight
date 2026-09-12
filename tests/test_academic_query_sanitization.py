"""Academic source adapters must receive a search query, not a topic blob.

Found by running a real quest. FI's `topic:` is routinely a paragraph-length
block with newlines and a "GOALS:" list, and `_route_external` handed it to
every adapter verbatim. Measured against the live APIs:

* arXiv answered **HTTP 500**, OpenAlex **HTTP 400**, on the raw multi-line
  topic. Crossref tolerated it. Two of the three configured academic sources
  therefore contributed *nothing*, and the run's corpus was Crossref plus
  general web search -- which is how a quest ends up citing a homework
  assignment and a web simulator alongside real papers.
* The 400 was **syntax, not length**: the topic states its model as
  ``m * x'' + c * x' + k * x = 0``, and a bare ``*`` is a wildcard operator to
  OpenAlex. The same query is accepted at 365 characters once the ``*`` is
  gone, while a 187-character query containing one is still rejected.

`_web_search` already sanitized for the same class of reason (Brave 422s on
embedded newlines). Sanitizing inside `_route_external` covers all six
academic adapters at once, including any added later.

NOTE: this fixes the hard *error*. It does not make a topic statement a good
query -- the same sanitized 365-character query returns HTTP 200 with **zero**
results, where an 79-character keyword query returns three on-target papers.
That is a separate, larger issue about deriving a query from a topic.
"""

from __future__ import annotations

import asyncio

import pytest

import core.knowledge as K

TOPIC = (
    "TOPIC: Compare three numerical integrators (forward Euler, 4th-order\n"
    "Runge-Kutta, and Velocity-Verlet) on a 1-D damped harmonic oscillator\n"
    "m * x'' + c * x' + k * x = 0, m = k = 1, c = 0.1, x(0) = 1, x'(0) = 0.\n"
    "\n"
    "GOALS:\n"
    "1. Implement all three integrators in Python (numpy).\n"
)


def test_operator_characters_are_removed() -> None:
    """`*` is what made OpenAlex answer 400."""
    out = K._sanitize_search_query(TOPIC, strip_operators=True)
    assert "*" not in out


def test_operators_become_spaces_not_deletions() -> None:
    """Deleting rather than replacing would weld two terms into a word that
    appears in no paper: `energy*drift` -> `energydrift`."""
    out = K._sanitize_search_query("energy*drift", strip_operators=True)
    assert out == "energy drift"


def test_newlines_are_collapsed() -> None:
    out = K._sanitize_search_query(TOPIC, strip_operators=True)
    assert "\n" not in out


def test_sanitizing_is_off_by_default() -> None:
    """General web search treats `*` as an ordinary character; only the
    scholarly APIs parse it. The default must not change that path."""
    assert "*" in K._sanitize_search_query("a * b")


def test_route_external_sanitizes_before_dispatch() -> None:
    """The regression itself: adapters must never see the raw topic. Asserted
    at the dispatcher, so an adapter added later inherits the fix instead of
    having to remember it."""
    seen: list[str] = []

    def _spy(query: str, top_k: int, *, timeout_s: float = 10.0) -> list:
        seen.append(query)
        return []

    original = dict(K._SOURCE_REGISTRY)
    K._SOURCE_REGISTRY["_spy"] = _spy  # type: ignore[assignment]
    try:
        asyncio.run(K._route_external(TOPIC, 5, ["_spy"], timeout_s=1))
    finally:
        K._SOURCE_REGISTRY.clear()
        K._SOURCE_REGISTRY.update(original)

    assert seen, "the adapter was never called"
    got = seen[0]
    assert "\n" not in got, "raw newlines reached an academic adapter"
    assert "*" not in got, "a wildcard operator reached an academic adapter"


@pytest.mark.parametrize("bad", ["*", "?", "~", "^", "\\"])
def test_each_known_operator_is_stripped(bad: str) -> None:
    out = K._sanitize_search_query(f"alpha {bad} beta", strip_operators=True)
    assert bad not in out
    assert "alpha" in out and "beta" in out
