"""FI's brain over the running Axon service (``core/axon_http.py``): the bracket that switches the service's project and back.

A stand-in for the service keeps one active project, answers ``/health/ready`` with it, and refuses (409) a request that names
another project, as the real one does. Two OS processes run operations against it at once, which is the case the file lock is
for; the real service is exercised separately (see the pull request).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.axon_http import AxonHTTPBrain, AxonUnavailable, _lock_path_for, _ProcessLock

FI = "frontier-insight"


class _Service:
    """A stand-in Axon service on a local port."""

    def __init__(self, *, active: str = "default", delay: float = 0.0, ready_delay: float = 0.0) -> None:
        self.active, self.delay, self.ready_delay = active, delay, ready_delay
        self.projects: set[str] = {"default"}
        self.calls: list[tuple[str, str]] = []
        self.refused: list[str] = []
        self.written: list[dict] = []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a) -> None:  # noqa: D401 -- silence
                return

            def _send(self, status: int, payload) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> dict:
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}") if n else {}

            def do_GET(self) -> None:  # noqa: N802
                service.calls.append(("GET", self.path))
                if self.path == "/health/ready":
                    seen = service.active
                    if service.ready_delay:  # the answer is stale by the time it arrives, as on a busy service
                        time.sleep(service.ready_delay)
                    self._send(200, {"status": "ok", "project": seen})
                else:
                    self._send(404, {"detail": "no such route"})

            def do_POST(self) -> None:  # noqa: N802
                body = self._body()
                service.calls.append(("POST", self.path))
                if self.path == "/project/new":
                    service.projects.add(body["name"])
                    return self._send(200, {"status": "success"})
                if self.path == "/project/switch":
                    if body["project_name"] not in service.projects:
                        return self._send(404, {"detail": "no such project"})
                    service.active = body["project_name"]
                    return self._send(200, {"status": "success"})
                if body.get("project") not in (None, service.active):
                    service.refused.append(self.path)
                    return self._send(409, {"detail": f"active project is {service.active!r}"})
                if service.delay:
                    time.sleep(service.delay)
                if self.path == "/add_texts":
                    service.written.extend(body["docs"])
                    return self._send(200, [{"id": d["doc_id"], "status": "created"} for d in body["docs"]])
                if self.path == "/search/raw":
                    hit = {"id": "d1", "text": "a passage", "score": 0.5, "metadata": {"kind": "fi_local_paper"}}
                    return self._send(200, {"query": body["query"], "results": [hit] * int(body.get("top_k") or 1), "diagnostics": {"n": 1}})
                if self.path == "/graph/finalize":
                    return self._send(200, {"status": "ok"})
                self._send(404, {"detail": "no such route"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def service():
    s = _Service()
    yield s
    s.close()


def _brain(url: str, tmp_path: Path) -> AxonHTTPBrain:
    return AxonHTTPBrain(url, FI, lock=_ProcessLock(tmp_path / "fi.lock", timeout=30.0))


def test_a_search_runs_in_fis_project_and_the_active_project_is_put_back(service, tmp_path) -> None:
    rows, diagnostics, trace = _brain(service.url, tmp_path).search_raw("epidemic threshold", overrides={"top_k": 3})
    assert len(rows) == 3 and diagnostics == {"n": 1} and trace is None
    assert service.active == "default"
    assert service.refused == []
    assert FI in service.projects  # created on first use


def test_a_service_already_on_fis_project_is_not_switched_at_all(tmp_path) -> None:
    service = _Service(active=FI)
    service.projects.add(FI)
    try:
        _brain(service.url, tmp_path).search_raw("q")
        assert ("POST", "/project/switch") not in service.calls
        assert service.active == FI
    finally:
        service.close()


def test_ingest_sends_the_documents_with_their_ids_and_metadata_and_counts_the_new_ones(service, tmp_path) -> None:
    brain = _brain(service.url, tmp_path)
    n = brain.ingest([
        {"id": "fi_local_paper:a", "text": "first body", "metadata": {"kind": "fi_local_paper", "source": "a.pdf"}},
        {"id": "fi_local_paper:empty", "text": "   ", "metadata": {}},
        {"id": "fi_local_paper:b", "text": "second body", "metadata": {"kind": "fi_local_paper"}},
    ])
    assert n == 2  # the empty document is not sent
    assert [d["doc_id"] for d in service.written] == ["fi_local_paper:a", "fi_local_paper:b"]
    assert service.written[0]["metadata"] == {"kind": "fi_local_paper", "source": "a.pdf"}
    brain.finalize_ingest()
    assert ("POST", "/graph/finalize") in service.calls
    assert service.active == "default"


def test_nested_sessions_switch_once_and_switch_back_once(service, tmp_path) -> None:
    brain = _brain(service.url, tmp_path)
    with brain.session():
        brain.ingest([{"id": "x", "text": "t", "metadata": {}}])
        brain.finalize_ingest()
        brain.search_raw("q")
        assert service.active == FI
    assert service.active == "default"
    assert service.calls.count(("POST", "/project/switch")) == 2  # there and back, not once per operation


def test_the_project_is_put_back_when_the_operation_fails(service, tmp_path) -> None:
    brain = _brain(service.url, tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with brain.session():
            assert service.active == FI
            raise RuntimeError("boom")
    assert service.active == "default"


def test_a_service_that_is_not_there_says_so_in_the_error(tmp_path) -> None:
    brain = _brain("http://127.0.0.1:9", tmp_path)  # nothing listens on the discard port
    with pytest.raises(AxonUnavailable, match="could not reach Axon at http://127.0.0.1:9"):
        brain.search_raw("q")


def test_a_refusal_carries_the_services_reason(service, tmp_path) -> None:
    brain = _brain(service.url, tmp_path)
    brain._known_project_exists = True  # noqa: SLF001
    with brain._lock.held():  # noqa: SLF001 -- talk to the service outside a session, so the project is not FI's
        with pytest.raises(AxonUnavailable, match="409.*active project is 'default'"):
            brain._request("POST", "/search/raw", {"query": "q", "project": FI})  # noqa: SLF001


_WORKER = """
import sys
import time
from pathlib import Path
from core.axon_http import AxonHTTPBrain, _ProcessLock
brain = AxonHTTPBrain(sys.argv[1], "frontier-insight", lock=_ProcessLock(Path(sys.argv[2]), timeout=60.0))
time.sleep(max(0.0, float(sys.argv[3]) - time.time()))  # both processes start together
for _ in range(15):
    brain.search_raw("q")
"""


def test_two_processes_operating_at_once_never_switch_under_each_other(tmp_path) -> None:
    """Without the lock the second process reads FI's project as 'previous', skips its switch, and the first process's
    switch-back then leaves it on the wrong project: the service answers 409."""
    service = _Service(delay=0.05, ready_delay=0.08)
    try:
        lock = tmp_path / "shared.lock"
        root = Path(__file__).resolve().parent.parent
        start = time.time() + 3.0
        procs = [
            subprocess.Popen([sys.executable, "-c", _WORKER, service.url, str(lock), str(start)], cwd=root, stderr=subprocess.PIPE, text=True)
            for _ in range(2)
        ]
        errors = [p.communicate(timeout=120)[1] for p in procs]
        assert [p.returncode for p in procs] == [0, 0], errors
        assert service.refused == []
        assert service.active == "default"
    finally:
        service.close()


def test_the_lock_is_named_after_the_service_so_every_process_takes_the_same_one() -> None:
    assert _lock_path_for("http://127.0.0.1:8420") == _lock_path_for("http://127.0.0.1:8420/")
    assert _lock_path_for("http://127.0.0.1:8420") != _lock_path_for("http://127.0.0.1:8421")


def test_a_lock_that_is_never_free_times_out_with_a_message(tmp_path) -> None:
    held = _ProcessLock(tmp_path / "x.lock", timeout=30.0)
    waiting = _ProcessLock(tmp_path / "x.lock", timeout=0.6)
    with held.held():
        with pytest.raises(AxonUnavailable, match="another FI process has held the Axon project"):
            with waiting.held():
                pass
    with waiting.held():  # free once the holder is done
        pass


# ---------------------------------------------------------------------------------------------------------------------
# Knowledge: how the layer picks its brain, and what it does when the service is not there
# ---------------------------------------------------------------------------------------------------------------------


def _knowledge(monkeypatch, **cfg):
    import core.knowledge as kn
    from core.config import KnowledgeConfig

    monkeypatch.setattr(kn, "_AXON_AVAILABLE", True)
    monkeypatch.setattr(kn, "AxonRetriever", lambda **kw: ("retriever", kw["brain"]))
    return kn, KnowledgeConfig(enabled=True, seed_source_catalog=False, **cfg)


def test_http_is_the_default_and_in_process_stays_selectable() -> None:
    from core.config import KnowledgeConfig

    assert KnowledgeConfig().axon_mode == "http"
    assert KnowledgeConfig(axon_mode="in_process").axon_mode == "in_process"
    with pytest.raises(Exception):  # noqa: B017, PT011 -- pydantic's validation error
        KnowledgeConfig(axon_mode="sidecar")


def test_the_knowledge_layer_uses_the_running_service(monkeypatch, service) -> None:
    from core import axon_sidecar

    monkeypatch.delenv("FI_NO_AXON_SIDECAR", raising=False)
    monkeypatch.setattr(axon_sidecar, "ensure_axon_up", lambda **kw: {"running": True, "ready": True, "url": service.url, "error": None, "source": "test"})
    kn, cfg = _knowledge(monkeypatch)
    k = kn.Knowledge(cfg)
    assert k.enabled and isinstance(k._brain, AxonHTTPBrain)
    assert k._brain.base_url == service.url and k._brain.project == kn.FI_AXON_PROJECT


def test_without_the_service_the_quest_runs_without_the_knowledge_base_and_the_log_says_why(monkeypatch, caplog) -> None:
    from core import axon_sidecar

    monkeypatch.delenv("FI_NO_AXON_SIDECAR", raising=False)
    monkeypatch.setattr(axon_sidecar, "ensure_axon_up", lambda **kw: {"running": False, "ready": False, "url": "http://127.0.0.1:8420", "error": "boot timeout after 30s", "source": "spawned"})
    kn, cfg = _knowledge(monkeypatch)
    with caplog.at_level("ERROR", logger="fi.knowledge"):
        k = kn.Knowledge(cfg)
    assert k.enabled is False and k._brain is None
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "runs without it" in text and "boot timeout after 30s" in text and "axon_mode: in_process" in text


def test_no_axon_sidecar_means_use_the_service_if_it_is_up_and_never_start_one(monkeypatch) -> None:
    from core import axon_sidecar

    monkeypatch.setenv("FI_NO_AXON_SIDECAR", "1")

    def _never(**kw):
        raise AssertionError("must not start the service")

    monkeypatch.setattr(axon_sidecar, "ensure_axon_up", _never)
    monkeypatch.setattr(axon_sidecar, "axon_status", lambda *a, **k: {"running": False, "ready": False, "url": "http://127.0.0.1:8420", "error": "not listening", "source": "probe"})
    kn, cfg = _knowledge(monkeypatch)
    assert kn.Knowledge(cfg).enabled is False


def test_launch_hands_no_axon_sidecar_to_the_web_server_through_the_environment() -> None:
    import inspect

    import launch

    assert 'os.environ["FI_NO_AXON_SIDECAR"] = "1"' in inspect.getsource(launch.main_async)


# ---------------------------------------------------------------------------------------------------------------------
# Auto-start: a lock a killed server left behind
# ---------------------------------------------------------------------------------------------------------------------


def test_a_process_that_exited_is_not_running_and_this_one_is() -> None:
    import os

    from core.axon_sidecar import _pid_is_running

    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait(timeout=60)
    assert _pid_is_running(os.getpid()) is True
    assert _pid_is_running(done.pid) is False
    assert _pid_is_running(0) is False


def _fake_axon(monkeypatch, lock: Path) -> None:
    import types

    import core.axon_endpoint as ep

    client = types.ModuleType("axon.server_client")
    client._store_lock_path = lambda config: lock
    monkeypatch.setitem(sys.modules, "axon.server_client", client)
    monkeypatch.setattr(ep, "_axon_config", lambda: object())


def test_a_stale_store_lock_is_removed_before_the_service_is_started(monkeypatch, tmp_path) -> None:
    from core.axon_sidecar import _clear_stale_store_lock

    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait(timeout=60)
    lock = tmp_path / ".axon-api.lock"
    lock.write_text(json.dumps({"host": "127.0.0.1", "port": 8420, "pid": done.pid}), encoding="utf-8")
    _fake_axon(monkeypatch, lock)
    said: list[str] = []
    _clear_stale_store_lock(said.append)
    assert not lock.exists()
    assert said and "stale Axon store lock" in said[0] and str(done.pid) in said[0]


def test_a_lock_whose_server_is_running_or_unreadable_is_left_alone(monkeypatch, tmp_path) -> None:
    import os

    from core.axon_sidecar import _clear_stale_store_lock

    lock = tmp_path / ".axon-api.lock"
    _fake_axon(monkeypatch, lock)
    lock.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    _clear_stale_store_lock(lambda m: None)
    assert lock.exists()
    lock.write_text("not json", encoding="utf-8")
    _clear_stale_store_lock(lambda m: None)
    assert lock.exists()


def test_looking_for_the_service_does_not_print_axons_retired_key_warnings(tmp_path, monkeypatch, capsys) -> None:
    """Axon's config loader warns once per retired key, through the ``Axon`` logger; FI reads only the address."""
    import logging
    import types

    import core.axon_endpoint as ep

    class _Config:
        @staticmethod
        def load(path=None):
            logging.getLogger("Axon").warning("config: 'graph_rag_x' was removed in 0.5.0")
            return types.SimpleNamespace(api_host="127.0.0.1", api_port=8420)

    fake = types.ModuleType("axon.config")
    fake.AxonConfig = _Config
    monkeypatch.setitem(sys.modules, "axon.config", fake)
    handler = logging.StreamHandler()
    records: list[str] = []
    handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
    axon_log = logging.getLogger("Axon")
    axon_log.addHandler(handler)
    try:
        config = ep._axon_config()
    finally:
        axon_log.removeHandler(handler)
    assert config is not None and config.api_port == 8420
    assert records == []
