"""FI's own verdict on equation cases in tests: the functions are measured in a child process given only their inputs
(core/equation_measure.py) and judged here, the way the engine does (core/equation_tests.run_and_judge)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from core import equation_tests as eqt


def fi_verdict(case_list: list[dict[str, Any]], cwd: Path, *, code: str = ".", only: list[str] | None = None,
               env: dict[str, str] | None = None, timeout: int = 240) -> list[dict[str, Any]]:
    nonce = eqt.new_nonce()
    done = subprocess.run([sys.executable, "-B", *eqt.measure_argv(case_list, nonce=nonce, only=only, code=code)],
                          cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          env=env if env is not None else dict(os.environ))
    found = eqt.judged(case_list, done.stdout, nonce=nonce, only=only)
    assert found is not None, done.stderr[-1500:]
    return found
