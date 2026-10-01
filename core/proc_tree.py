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

import logging
import os
import signal
import subprocess
import time
from typing import Any

__all__ = ["ProcessTree"]

_log = logging.getLogger("frontier_insight.proc_tree")

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

    Use it as a context manager, or call ``close()`` when done. Leaving the block (or ``close()``, or dropping the
    object) while the program still runs stops the whole tree; on Windows it also stops anything a finished program
    left running. So keep the ``ProcessTree``, not just its ``.proc``, for as long as the program should run.
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
        resumed = False
        try:
            handle = int(self.proc._handle)  # noqa: SLF001 - the only way to reach the process handle
            try:
                job = _new_job()
            except OSError as exc:
                _log.warning("could not create a job for %s (%s); a timeout will fall back to taskkill", argv[0], exc)
                job = None
            if job is not None and not _kernel32.AssignProcessToJobObject(job, handle):
                _log.warning(
                    "could not put %s into a job (%s); a timeout will fall back to taskkill",
                    argv[0], ctypes.WinError(ctypes.get_last_error()),
                )
                _kernel32.CloseHandle(job)
                job = None
            self._job = job
            # Never leave the program suspended: it would look like a hang to whoever waits on it.
            resumed = _ntdll.NtResumeProcess(handle) >= 0
        finally:
            if not resumed:
                self.proc.kill()
                self.proc.wait()
                for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                    if stream is not None:
                        stream.close()
                self.close()
        if not resumed:
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
        if self._job is not None and _kernel32.TerminateJobObject(self._job, 1):
            while time.monotonic() < deadline:
                try:
                    if _active_processes(self._job) == 0:
                        break
                except OSError:
                    break
                time.sleep(0.05)
            else:
                _log.warning("processes started by %s were still running %.0f s after being stopped",
                             self.proc.args, _DESCENDANT_WAIT_S)
        else:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                capture_output=True, stdin=subprocess.DEVNULL, check=False,
            )
        self._reap(deadline)

    def _kill_posix(self, deadline: float) -> None:
        # The program leads its own session, so its process group id is its pid; the group outlives its
        # leader, which is what catches a helper whose parent has already exited. While any member is left the
        # id stays reserved, so it cannot name an unrelated group; call this only before the program has been
        # waited for with its group empty (``close()`` checks ``poll()`` first; the callers kill on a timeout).
        pgid = self.proc.pid
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        self._reap(deadline)
        # Killed members linger as zombies until their new parent reaps them; one that never does (a container
        # without an init process) only costs this bounded wait.
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                break
            time.sleep(0.05)
        else:
            _log.debug("process group %d was still present %.0f s after being killed", pgid, _DESCENDANT_WAIT_S)

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
        """Stop the tree if the program is still running (an exception or Ctrl+C left the ``with`` block early),
        then release the job (Windows; closing it also stops anything the program left running)."""
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            self.kill()
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
