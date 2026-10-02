"""The quick ``fi --doctor``: this computer only, nothing loaded, a few seconds, even with an empty model cache and
no network. It used to load the embedding model, which on a fresh machine meant trying Hugging Face, and sat for
more than a minute with nothing on screen. ``--deep`` is the one that may use the network, and it says so first.

The subprocess test runs the real command with a fresh FI home, empty model caches and every connection to anything
but this computer refused (a ``sitecustomize`` that replaces ``socket.connect`` / ``getaddrinfo``), and counts the
attempts.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import launch

ROOT = Path(__file__).resolve().parent.parent

_BLOCK_NETWORK = r'''
import os, socket

_LOG = os.environ["FI_TEST_NET_LOG"]
_LOCAL = ("127.0.0.1", "::1", "localhost", "0.0.0.0")


def _note(what):
    with open(_LOG, "a", encoding="utf-8") as fh:
        fh.write(repr(what) + "\n")


def _is_local(address):
    host = address[0] if isinstance(address, tuple) else address
    return isinstance(host, str) and (host in _LOCAL or host.startswith("127."))


_connect = socket.socket.connect
_connect_ex = socket.socket.connect_ex
_getaddrinfo = socket.getaddrinfo


def connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(address):
        _note(("connect", address))
        raise OSError(10013 if os.name == "nt" else 101, "network blocked by the test")
    return _connect(self, address)


def connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(address):
        _note(("connect_ex", address))
        return 101
    return _connect_ex(self, address)


def getaddrinfo(host, *args, **kwargs):
    if host is not None and not _is_local(host if isinstance(host, str) else host.decode()):
        _note(("getaddrinfo", host))
        raise socket.gaierror(-2, "name lookup blocked by the test")
    return _getaddrinfo(host, *args, **kwargs)


socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
socket.getaddrinfo = getaddrinfo
'''


def _doctor_env(tmp_path: Path) -> dict[str, str]:
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_BLOCK_NETWORK, encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("HF_", "HUGGINGFACE", "TRANSFORMERS_", "SENTENCE_TRANSFORMERS", "FI_"))
           and k.upper() not in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")}  # a proxy would hide a connection
    env.update({
        "PYTHONPATH": os.pathsep.join([str(site), str(ROOT)]),
        "FI_TEST_NET_LOG": str(tmp_path / "net.log"),
        "FI_HOME": str(tmp_path / "fi_home"),
        "HF_HOME": str(tmp_path / "hf_home"),           # an empty model cache
        "XDG_CACHE_HOME": str(tmp_path / "xdg_cache"),
        "SENTENCE_TRANSFORMERS_HOME": str(tmp_path / "st_home"),
        "PYTHONIOENCODING": "utf-8",
    })
    return env


def _run_doctor(tmp_path: Path, env: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "launch.py"), "--doctor", "--no-axon-sidecar"],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120,
        stdin=subprocess.DEVNULL,
    )
    return proc, time.monotonic() - started


def test_quick_doctor_is_fast_offline_and_makes_no_connection(tmp_path: Path) -> None:
    env = _doctor_env(tmp_path)
    _run_doctor(tmp_path, env)  # once untimed: the .pyc files of a fresh checkout are written here
    (tmp_path / "net.log").unlink(missing_ok=True)
    proc, took = _run_doctor(tmp_path, env)
    out = proc.stdout
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert not (tmp_path / "net.log").exists(), (tmp_path / "net.log").read_text(encoding="utf-8")
    assert "Embeddings (MiniLM)" in out and ("not downloaded yet" in out or "not installed" in out)
    assert "Model providers" in out
    assert "--deep" in out
    assert took < 5.0, f"quick doctor took {took:.1f} s (Python start and FI's imports included)\n{out}"
    print(f"quick doctor: {took:.2f} s wall clock; {out.strip().splitlines()[-3:]}")


@pytest.fixture
def fake_finders(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real finders start programs (a browser for its version): replaced, as in test_doctor_and_marp."""
    monkeypatch.setattr("generation._pandoc.find_pandoc", lambda *a, **k: "/p/pandoc")
    monkeypatch.setattr("generation._pdf_engine.find_pdf_engine", lambda *a, **k: ("pdflatex", "/x/pdflatex"))
    monkeypatch.setattr("generation._html_pdf.find_html_browser", lambda *a, **k: ("edge", "/e/edge"))
    monkeypatch.setattr("generation._marp.find_marp", lambda *a, **k: "/m/marp")
    monkeypatch.setattr("generation._office_pdf.find_libreoffice", lambda *a, **k: "/l/soffice")


def test_the_quick_doctor_does_not_import_the_engine(tmp_path: Path) -> None:
    """Importing launch for `fi --doctor` alone runs no `core` import (the engine is skipped, so it is quick), so a
    missing package is caught where the bootstrap can set FI up: main()'s guarded import, not a module-level one."""
    code = ("import sys; sys.argv = ['fi', '--doctor']; import launch; "
            "print('core.engine' in sys.modules, 'core.config' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[-2:] == ["False", "False"]


def test_quick_doctor_never_loads_the_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fake_finders: None,
) -> None:
    loaded: list[str] = []
    monkeypatch.setattr("core.passages._embed_model", lambda: loaded.append("loaded"))
    monkeypatch.setattr("core.passages.embed_model_cached", lambda: (True, "all-MiniLM-L6-v2 downloaded (cache)"))
    assert launch._doctor() == 0
    assert loaded == []
    assert "all-MiniLM-L6-v2 downloaded" in capsys.readouterr().out


def test_a_check_that_hangs_is_reported_and_the_rest_still_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fake_finders: None,
) -> None:
    import threading

    release = threading.Event()
    monkeypatch.setattr(launch, "_DOCTOR_CHECK_TIMEOUT_S", 0.3)
    monkeypatch.setattr("generation._pandoc.find_pandoc", lambda *a, **k: release.wait(30))
    monkeypatch.setattr("core.passages.embed_model_cached", lambda: (False, "not downloaded"))
    started = time.monotonic()
    try:
        assert launch._doctor() == 0
    finally:
        release.set()
    out = capsys.readouterr().out
    assert "not checked: did not answer within 0.3 s" in out
    assert "Marp CLI" in out and "Model providers" in out  # the checks after it ran
    # Not answering is not missing: no install advice for it, and paper_pdf is "not known", not MISS.
    assert "pypandoc_binary" not in out
    assert "paper_pdf                    not known" in out
    assert time.monotonic() - started < 15


def test_no_usable_provider_is_something_to_fix(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fake_finders: None,
) -> None:
    from core import provider_readiness as pr

    monkeypatch.setattr("core.passages.embed_model_cached", lambda: (True, "downloaded"))
    monkeypatch.setattr(pr.shutil, "which", lambda name: None)
    for env in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    assert launch._doctor() == 0
    out = capsys.readouterr().out
    assert "none of the providers above can be used yet" in out
    assert "Everything checked is available" not in out


def test_a_gemini_key_alone_is_a_usable_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fake_finders: None,
) -> None:
    """fi demo would use it, so the doctor does not say that no model can be used."""
    from core import provider_readiness as pr

    monkeypatch.setattr("core.passages.embed_model_cached", lambda: (True, "downloaded"))
    monkeypatch.setattr(pr.shutil, "which", lambda name: "/bin/copilot" if name == "copilot" else None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    assert launch._doctor() == 0
    out = capsys.readouterr().out
    assert "gemini" in out and "none of the providers above" not in out
    # A copilot command alone is not one fi demo uses: then the line is shown.
    monkeypatch.delenv("GEMINI_API_KEY")
    assert launch._doctor() == 0
    assert "none of the providers above can be used yet" in capsys.readouterr().out


def test_deep_doctor_does_not_hold_ollama_to_a_default_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fake_finders: None,
) -> None:
    """The deep doctor asks what Ollama has; only a launch with no model named looks for the provider's default."""
    from core import provider_readiness as pr

    async def tags(url, headers, timeout_s):  # noqa: ANN001, ANN202
        return (200, {"models": [{"name": "gemma3:4b"}]}) if url.endswith("/api/tags") else (None, None)

    monkeypatch.setattr(pr, "_get_json", tags)
    monkeypatch.setattr(pr, "_run_command", lambda argv, timeout_s: (None, ""))
    monkeypatch.setattr(pr.shutil, "which", lambda name: "/bin/ollama" if name == "ollama" else None)
    for env in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr("core.passages._embed_model", lambda: object())
    assert launch._doctor(deep=True) == 0
    out = capsys.readouterr().out
    assert "Reachable \u2014 Running at http://127.0.0.1:11434 (1 model(s) downloaded)." in out
    assert "none of the providers above" not in out


def test_deep_doctor_says_what_it_will_do_before_doing_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fake_finders: None,
) -> None:
    from core import provider_readiness as pr

    order: list[str] = []

    def load():  # noqa: ANN202
        order.append(capsys.readouterr().out)
        return object()

    async def fake_check_all(names, **kwargs):  # noqa: ANN001, ANN202
        assert kwargs.get("call_service") is True
        return [pr.Readiness(n, "sign_in_unknown", "FI cannot tell.") for n in names]

    monkeypatch.setattr("core.passages._embed_model", load)
    monkeypatch.setattr(pr, "check_all", fake_check_all)
    assert launch._doctor(deep=True) == 0
    assert order and "Hugging Face" in order[0] and "can take a minute" in order[0]


def test_deep_goes_with_doctor_only() -> None:
    with pytest.raises(SystemExit):
        launch.parse_args(["--new", "--deep"])
    assert launch.parse_args(["--doctor", "--deep"]).deep is True
