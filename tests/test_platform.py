"""Direct tests for `core.platform.detect_system`.

Confirms the function returns one of the four documented values on the
current host. The detailed branch logic (Linux vs WSL2 via
`/proc/version`) is exercised on the appropriate host kernels in CI;
here we only assert the contract.
"""

from __future__ import annotations

from core.platform import detect_system


def test_detect_system_returns_valid_value() -> None:
    assert detect_system() in {"linux", "wsl2", "macos", "windows"}
