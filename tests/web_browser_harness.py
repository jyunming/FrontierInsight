"""Drive FI's web pages in a real browser with no network and no server socket.

Every request the page makes is answered by the FastAPI app in-process (Starlette's TestClient) through Playwright's
request interception: the page is at ``http://fi.test``, anything else is refused and recorded (an offline machine).
A test injects faults per path: answer 500, drop the connection, hold the answer (a slow server), or let the server
do the work and then drop its answer (the answer lost on the way back). ``restart()`` swaps in a new app on the same
output folder, as a restarted ``fi --serve`` would be.

Skips when Playwright or its Chromium is not installed (CI's slow tier installs both).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import pytest

HOST = "fi.test"
BASE = f"http://{HOST}"

# A fault: called with (route, request); True when it answered the request itself.
Fault = Callable[[Any, Any], bool]


class Site:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.launches: list[str] = []
        self.external: list[str] = []
        self.requests: list[tuple[str, str]] = []
        self.faults: dict[str, Fault] = {}
        self.held: list[Any] = []
        self.restart()

    def restart(self) -> None:
        from fastapi.testclient import TestClient

        from web.server import make_app

        app = make_app(self.output_root)

        def fake_launch(*, quest_id: str, yaml_path: Path) -> SimpleNamespace:
            self.launches.append(quest_id)
            return SimpleNamespace(quest_id=quest_id, pid=4242)

        app.state.launcher.launch = fake_launch
        self.client = TestClient(app)

    # Answers no test needs from the real handlers (they would probe the machine): a fixed reply.
    _STUBS: dict[str, Any] = {
        "/api/axon/status": {"running": False},
        "/api/provider/models": {"source": "static", "models": []},
        "/api/knowledge/info": {"enabled": False, "documents": []},
    }

    def forward(self, route: Any, request: Any) -> Any:
        """Have the app answer ``request``; returns its response (not yet sent to the page)."""
        parts = urlsplit(request.url)
        path = parts.path + (f"?{parts.query}" if parts.query else "")
        headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "idempotency-key")}
        return self.client.request(request.method, path, content=request.post_data_buffer, headers=headers)

    @staticmethod
    def fulfill(route: Any, resp: Any) -> None:
        skip = {"content-length", "content-encoding", "transfer-encoding", "connection"}
        route.fulfill(status=resp.status_code, body=resp.content,
                      headers={k: v for k, v in resp.headers.items() if k.lower() not in skip})

    def handle(self, route: Any, request: Any) -> None:
        parts = urlsplit(request.url)
        if parts.hostname != HOST:
            self.external.append(request.url)
            route.abort("internetdisconnected")
            return
        self.requests.append((request.method, parts.path))
        fault = self.faults.get(parts.path)
        if fault is not None and fault(route, request):
            return
        if parts.path in self._STUBS and request.method == "GET":
            import json

            route.fulfill(status=200, content_type="application/json", body=json.dumps(self._STUBS[parts.path]))
            return
        self.fulfill(route, self.forward(route, request))

    # --- faults ---------------------------------------------------------------------------------------------------

    def fail_once(self, path: str, status: int = 500) -> None:
        def fault(route: Any, _request: Any) -> bool:
            self.faults.pop(path, None)
            route.fulfill(status=status, content_type="application/json", body='{"detail": "injected failure"}')
            return True
        self.faults[path] = fault

    def drop_once(self, path: str) -> None:
        def fault(route: Any, _request: Any) -> bool:
            self.faults.pop(path, None)
            route.abort("connectionreset")
            return True
        self.faults[path] = fault

    def hold_once(self, path: str) -> None:
        """No answer at all (a server that hangs): the page's own timeout must notice."""
        def fault(route: Any, _request: Any) -> bool:
            self.faults.pop(path, None)
            self.held.append(route)
            return True
        self.faults[path] = fault

    def lose_answer_once(self, path: str) -> None:
        """The server does the work; its answer never reaches the page."""
        def fault(route: Any, request: Any) -> bool:
            self.faults.pop(path, None)
            self.forward(route, request)
            route.abort("connectionreset")
            return True
        self.faults[path] = fault


@contextlib.contextmanager
def browser_page(site: Site, *, init_script: str = "") -> Iterator[Any]:
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception as e:  # noqa: BLE001 -- no browser installed: nothing to drive
            pytest.skip(f"Chromium for Playwright is not installed ({e.__class__.__name__})")
        try:
            context = browser.new_context()
            context.route("**/*", site.handle)
            if init_script:
                context.add_init_script(init_script)
            page = context.new_page()
            page.set_default_timeout(20000)
            yield page
            for route in site.held:
                with contextlib.suppress(Exception):
                    route.abort()
        finally:
            browser.close()


def fill_first_screen(page: Any, topic: str = "Heat flow through a cooling fin") -> None:
    """The answers a new quest needs before its review screen: the topic, 'exploring', and a provider."""
    page.wait_for_selector("#q-topic")
    page.fill("#q-topic", topic)
    page.select_option("#q-result_use", "explore")
    page.select_option("#q-provider", "openai")
