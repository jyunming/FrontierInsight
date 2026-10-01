"""Start a program so that it, and every program it starts, can be stopped together.

Killing a program alone leaves behind what it started: ``git`` leaves ``git-remote-https``, ``soffice.exe``
leaves ``soffice.bin``. Whatever is left keeps running, and keeps the output handle it inherited open, so a
caller waiting on that output waits for ever.

On Windows the program is started suspended, put into its own Job Object and only then resumed, so nothing it
starts can be created outside the job (``taskkill /T`` instead walks the parent links at the moment it runs: a
helper created while it walks, or one whose parent has already exited, is missed). The job is set to kill every
process in it when it is closed, so the tree also goes if this process dies. On POSIX the program leads a new
session, and the whole process group is killed.

Synchronous on purpose: the callers that need it (``scripts/import_scientist_skills.py``,
``generation/_office_pdf.py``) run a blocking ``subprocess.Popen``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any

__all__ = ["ProcessTree"]

# How long ``kill`` waits for every process in the tree to be gone before giving up on the wait.
_DESCENDANT_WAIT_S = 10.0

if os.name == "nt":  # pragma: no cover - exercised on Windows only
    import ctypes
    from ctypes import wintypes

    _CREATE_SUSPENDED = 0x00000004
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JobObjectBasicAccountingInformation = 1
    _JobObjectExtendedLimitInformation = 9

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_longlong),
            ("TotalKernelTime", ctypes.c_longlong),
            ("ThisPeriodTotalUserTime", ctypes.c_longlong),
            ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")

    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
    ]
    _kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    _ntdll.NtResumeProcess.restype = ctypes.c_long

    def _new_job() -> int:
        job = _kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _kernel32.SetInformationJobObject(
            job, _JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info),
        ):
            err = ctypes.get_last_error()
            _kernel32.CloseHandle(job)
            raise ctypes.WinError(err)
        return job

    def _active_processes(job: int) -> int:
        info = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        if not _kernel32.QueryInformationJobObject(
            job, _JobObjectBasicAccountingInformation, ctypes.byref(info), ctypes.sizeof(info), None,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(info.ActiveProcesses)


class ProcessTree:
    """``subprocess.Popen`` (``.proc``) whose whole tree ``kill()`` stops.

    Use it as a context manager, or call ``close()`` when done: on Windows that releases the job, which also
    stops anything the program left running.
    """

    def __init__(self, argv: list[str], **popen_kwargs: Any) -> None:
        self._job: int | None = None
        if os.name == "nt":
            self._start_windows(argv, popen_kwargs)
        else:
            popen_kwargs["start_new_session"] = True
            self.proc = subprocess.Popen(argv, **popen_kwargs)

    # --- Windows ---------------------------------------------------------------

    def _start_windows(self, argv: list[str], popen_kwargs: dict[str, Any]) -> None:  # pragma: no cover
        flags = popen_kwargs.pop("creationflags", 0)
        self.proc = subprocess.Popen(argv, creationflags=flags | _CREATE_SUSPENDED, **popen_kwargs)
        handle = int(self.proc._handle)  # noqa: SLF001 - the only way to reach the process handle
        try:
            try:
                job = _new_job()
            except OSError:
                job = None  # no job: kill() falls back to taskkill /T
            if job is not None and not _kernel32.AssignProcessToJobObject(job, handle):
                _kernel32.CloseHandle(job)
                job = None
            self._job = job
        finally:
            # Never leave the program suspended: it would look like a hang to whoever waits on it.
            if _ntdll.NtResumeProcess(handle) < 0:
                self.proc.kill()
                self.proc.wait()
                self.close()
                raise OSError(f"could not resume {argv[0]} after starting it")

    # --- both ------------------------------------------------------------------

    @property
    def pid(self) -> int:
        return self.proc.pid

    def kill(self) -> None:
        """Stop the program and everything it started, and wait (up to a few seconds) until all of it is gone."""
        deadline = time.monotonic() + _DESCENDANT_WAIT_S
        if os.name == "nt":
            self._kill_windows(deadline)
        else:
            self._kill_posix(deadline)

    def _kill_windows(self, deadline: float) -> None:  # pragma: no cover - exercised on Windows only
        if self._job is not None:
            _kernel32.TerminateJobObject(self._job, 1)
            while time.monotonic() < deadline:
                try:
                    if _active_processes(self._job) == 0:
                        break
                except OSError:
                    break
                time.sleep(0.05)
        else:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                capture_output=True, stdin=subprocess.DEVNULL, check=False,
            )
        self._reap(deadline)

    def _kill_posix(self, deadline: float) -> None:
        # The program leads its own session, so its process group id is its pid; the group outlives its
        # leader, which is what catches a helper whose parent has already exited.
        pgid = self.proc.pid
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self._reap(deadline)
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                break
            time.sleep(0.05)

    def _reap(self, deadline: float) -> None:
        try:
            self.proc.kill()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass

    def close(self) -> None:
        """Release the job (Windows). Anything of the tree still running is stopped with it."""
        job, self._job = self._job, None
        if job is not None and os.name == "nt":  # pragma: no cover - exercised on Windows only
            _kernel32.CloseHandle(job)

    def __enter__(self) -> ProcessTree:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass
