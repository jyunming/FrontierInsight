"""The headless-render navigation guard must not print a traceback when the page closed under it.

A render's page (or the Playwright driver connection) can close while a route handler is still
answering a request — the render finished or timed out. Playwright then raises from
``route.abort()`` / ``continue_()`` / ``fulfill()``, and because the handler runs inside
Playwright's event dispatcher the error is printed to the console as a full traceback, which
reads as if FI crashed. The guard ignores exactly that closed-page case and nothing else.
"""
from __future__ import annotations

import pytest

import core.knowledge as kn

_DRIVER_CLOSED = "Route.abort: Connection closed while reading from the driver"


class _Req:
    def __init__(self, url: str, *, nav: bool = True):
        self.url = url
        self._nav = nav

    def is_navigation_request(self) -> bool:
        return self._nav


class _Resp:
    def __init__(self, status: int = 200, location: str | None = None):
        self.status = status
        self.headers = {"location": location} if location else {}


class _Route:
    """A fake Playwright route whose calls raise ``exc`` (when given)."""

    def __init__(self, *, exc: BaseException | None = None, fetch_exc: BaseException | None = None,
                 resp: _Resp | None = None):
        self.exc = exc
        self.fetch_exc = fetch_exc
        self.resp = resp or _Resp()
        self.calls: list[str] = []

    def _act(self, name: str):
        self.calls.append(name)
        if self.exc is not None:
            raise self.exc

    def abort(self, *a, **k):
        self._act("abort")

    def continue_(self, *a, **k):
        self._act("continue")

    def fulfill(self, *a, **k):
        self._act("fulfill")

    def fetch(self, *a, **k):
        self.calls.append("fetch")
        if self.fetch_exc is not None:
            raise self.fetch_exc
        return self.resp


def _deny(url: str) -> bool:
    return False


def _allow(url: str) -> bool:
    return True


def test_abort_after_driver_closed_is_ignored():
    guard = kn._make_navigation_guard(_deny)
    route = _Route(exc=Exception(_DRIVER_CLOSED))
    guard(route, _Req("https://refused.example/"))  # must not raise
    assert route.calls == ["abort"]


def test_fetch_and_abort_after_driver_closed_is_ignored():
    # The traceback seen in a real quest: fetch failed because the driver was gone, then the
    # fallback abort failed for the same reason.
    guard = kn._make_navigation_guard(_allow)
    route = _Route(exc=Exception(_DRIVER_CLOSED), fetch_exc=Exception("Route.fetch: Connection closed while reading from the driver"))
    guard(route, _Req("https://ok.example/"))
    assert route.calls == ["fetch", "abort"]


def test_continue_and_fulfill_after_target_closed_are_ignored():
    closed = Exception("Route.continue_: Target page, context or browser has been closed")
    kn._make_navigation_guard(_allow)(_Route(exc=closed), _Req("https://ok.example/x.js", nav=False))
    route = _Route(exc=Exception("Route.fulfill: Target page, context or browser has been closed"))
    kn._make_navigation_guard(_allow)(route, _Req("https://ok.example/"))
    assert route.calls == ["fetch", "fulfill"]


def test_playwright_target_closed_error_type_is_ignored():
    errors = pytest.importorskip("playwright._impl._errors")
    guard = kn._make_navigation_guard(_deny)
    guard(_Route(exc=errors.TargetClosedError()), _Req("https://refused.example/"))


def test_unrelated_error_still_raises():
    guard = kn._make_navigation_guard(_deny)
    with pytest.raises(ValueError):
        guard(_Route(exc=ValueError("something else broke")), _Req("https://refused.example/"))


def test_redirect_to_refused_address_is_still_aborted():
    guard = kn._make_navigation_guard(lambda u: "evil" not in u)
    route = _Route(resp=_Resp(302, "https://evil.example/"))
    guard(route, _Req("https://ok.example/"))
    assert route.calls == ["fetch", "abort"]


def test_allowed_navigation_is_fulfilled():
    route = _Route()
    kn._make_navigation_guard(_allow)(route, _Req("https://ok.example/"))
    assert route.calls == ["fetch", "fulfill"]
