"""Start a program so that it, and every program it starts, can be stopped together.

Killing a program alone leaves behind what it started: ``git`` leaves ``git-remote-https``, ``soffice.exe``
leaves ``soffice.bin``. Whatever is left keeps running, and keeps the output handle it inherited open, so a
caller waiting on that output waits for ever.

On Windows the program is started suspended, put into its own Job Object and only then resumed, so nothing it
starts can be created outside the job (``taskkill /T`` instead walks the parent links at the moment it runs: a
helper created while it walks, or one whose parent has already exited, is missed). The job is set to kill every
process in it when it is closed, so the tree also goes if this process dies. On POSIX the program leads a new
session, and the whole process group is killed.

Two forms. ``ProcessTree`` wraps a blocking ``subprocess.Popen`` (``scripts/import_scientist_skills.py``,
``generation/_office_pdf.py``, the provider proxies in ``core/provider.py``). ``AsyncProcessTree`` wraps
``asyncio.create_subprocess_exec`` for the async callers: the experiment script run by ``core/execution.py`` and the
CLI providers in ``core/provider.py``, whose timeouts and cancellations must not leave a pool of workers or a CLI's
helpers running.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import subprocess
import time
from typing import Any

__all__ = ["AsyncProcessTree", "ProcessTree"]

_log = logging.getLogger("frontier_insight.proc_tree")

# How long ``kill`` waits for every process in the tree to be gone before giving up on the wait.
_DESCENDANT_WAIT_S = 10.0

# The real class, kept before any test can patch ``subprocess.Popen`` with a stand-in (``ProcessTree.real``).
_REAL_POPEN = subprocess.Popen

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
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateProcess.restype = wintypes.BOOL

    # AssignProcessToJobObject needs PROCESS_SET_QUOTA and PROCESS_TERMINATE; NtResumeProcess needs
    # PROCESS_SUSPEND_RESUME.
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_SUSPEND_RESUME = 0x0800
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _ADOPT_ACCESS = _PROCESS_TERMINATE | _PROCESS_SET_QUOTA | _PROCESS_SUSPEND_RESUME | _PROCESS_QUERY_LIMITED_INFORMATION

    def _adopt_suspended(pid: int, name: str) -> int | None:
        """Put the suspended process ``pid`` into a new job and resume it. Returns the job, or None when no job could
        be made (a stop then falls back to taskkill). Raises OSError, with the process stopped, if it cannot be
        resumed: a process left suspended would look like a hang to whoever waits on it."""
        handle = _kernel32.OpenProcess(_ADOPT_ACCESS, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        job: int | None = None
        try:
            try:
                job = _new_job()
            except OSError as exc:
                _log.warning("could not create a job for %s (%s); a timeout will fall back to taskkill", name, exc)
            if job is not None and not _kernel32.AssignProcessToJobObject(job, handle):
                _log.warning(
                    "could not put %s into a job (%s); a timeout will fall back to taskkill",
                    name, ctypes.WinError(ctypes.get_last_error()),
                )
                _kernel32.CloseHandle(job)
                job = None
            if _ntdll.NtResumeProcess(handle) < 0:
                _kernel32.TerminateProcess(handle, 1)
                if job is not None:
                    _kernel32.CloseHandle(job)
                raise OSError(f"could not resume {name} after starting it")
            return job
        finally:
            _kernel32.CloseHandle(handle)

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
        if not self.real:
            return  # a test's stand-in: no handle to put in a job (int() of a mock is 1, a real handle number)
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

    @property
    def real(self) -> bool:
        """A process ``subprocess.Popen`` started, not a test's stand-in (a test that patches ``subprocess.Popen``).
        A stand-in gets no job and no process group: ``kill()`` only calls its own ``kill()``."""
        return issubclass(type(self.proc), _REAL_POPEN)  # type(), not isinstance: a spec mock fakes __class__

    def kill(self) -> None:
        """Stop the program and everything it started, and wait (up to a few seconds) until all of it is gone."""
        if not self.real:
            try:
                self.proc.kill()
            except OSError:
                pass
            return
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


class AsyncProcessTree:
    """``asyncio.subprocess.Process`` (``.proc``) whose whole tree ``await kill()`` stops.

    Start it with ``await AsyncProcessTree.start(*argv, **kwargs)`` (the arguments of
    ``asyncio.create_subprocess_exec``) and end it with ``await aclose()`` in a ``finally``: a program still running
    then (an error or a cancellation ended the call early) is stopped with everything it started. On Windows
    ``aclose()`` also stops anything a finished program left running. Keep the ``AsyncProcessTree``, not just its
    ``.proc``, for as long as the program should run: dropping it releases the job, which stops the tree.

    Only a real ``asyncio.subprocess.Process`` gets a job or a process group. A stand-in (a test's mock) is left as
    it is, and ``kill()`` then only calls its own ``kill()``: a mock's made-up pid must never reach
    ``OpenProcess`` or ``killpg``, where it could name somebody else's process.
    """

    def __init__(self, proc: Any, job: int | None = None) -> None:
        self.proc = proc
        self._job = job

    @classmethod
    async def start(cls, *argv: Any, **kwargs: Any) -> AsyncProcessTree:
        # ``asyncio.create_subprocess_exec`` is looked up when called, so a test that patches it reaches this too.
        if os.name == "nt":  # pragma: no cover - exercised on Windows only
            flags = kwargs.pop("creationflags", 0) or 0
            proc = await asyncio.create_subprocess_exec(*argv, creationflags=flags | _CREATE_SUSPENDED, **kwargs)
            if not _is_real(proc):
                return cls(proc)
            try:
                job = _adopt_suspended(proc.pid, str(argv[0]))
            except OSError:
                # Stopped, or never resumed: reap it so no pipe or handle is left behind.
                with contextlib.suppress(Exception):
                    proc.kill()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=_DESCENDANT_WAIT_S)
                raise
            return cls(proc, job)
        kwargs["start_new_session"] = True
        return cls(await asyncio.create_subprocess_exec(*argv, **kwargs))

    @property
    def pid(self) -> Any:
        return self.proc.pid

    def running(self) -> bool:
        """The program itself has not been seen to exit (what it started may still run either way)."""
        return getattr(self.proc, "returncode", 0) is None

    def _signal(self, *, group_after_exit: bool) -> bool:
        """Stop the whole tree now, without waiting: safe as the first step of a cancelled task's cleanup, since no
        ``await`` comes before it. True when it went through the job or the process group.

        Windows: when the job cannot be ended, the program itself is left alone, so that ``kill()``'s taskkill can
        still walk the tree from it.

        POSIX: the program leads its own session, so its group id is its pid. While the program runs (or is an
        unreaped zombie) that id is its own. Once it has exited and been reaped, the group outlives it as long as
        any member is left (which is what catches a helper still holding the output pipe), and the id stays
        reserved for it; ``group_after_exit`` asks for that case too (a timeout or a cancellation). An empty group's
        id is free again, and could name an unrelated group only after the system has handed out every other pid
        in between, which takes far longer than the moment between the program's exit and this stop.
        """
        stopped = False
        real = _is_real(self.proc)
        if os.name == "nt":  # pragma: no cover - exercised on Windows only
            stopped = self._job is not None and bool(_kernel32.TerminateJobObject(self._job, 1))
            if real and not stopped:
                return False
        elif real and (group_after_exit or self.running()):
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
                stopped = True
            except (ProcessLookupError, PermissionError):
                pass
        try:
            self.proc.kill()  # an exited program: ProcessLookupError, ignored
        except (ProcessLookupError, OSError):
            pass
        return stopped

    async def kill(self) -> bool:
        """Stop the program and everything it started, wait (up to a few seconds) until all of it is gone, and reap
        the program. True when the program was reaped (and, on Windows, its job emptied) within the wait. On POSIX
        the members of the stopped group are waited for too, but a member that stays a zombie (its new parent never
        reaps it, as in a container with no init process) does not count against the result."""
        deadline = time.monotonic() + _DESCENDANT_WAIT_S
        via_tree = self._signal(group_after_exit=True)
        if os.name == "nt" and not via_tree and _is_real(self.proc) and self.running():  # pragma: no cover
            # No job, or it could not be ended: taskkill walks the tree from the program while it still runs. An
            # exited program's pid could already name another process, and there is no tree to walk from it.
            try:
                await asyncio.to_thread(
                    subprocess.run, ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                    capture_output=True, stdin=subprocess.DEVNULL, check=False,
                )
            finally:
                try:
                    self.proc.kill()
                except (ProcessLookupError, OSError):
                    pass
        gone = True
        if os.name == "nt" and via_tree:  # pragma: no cover - exercised on Windows only
            gone = await self._wait_job_empty(deadline)
        reaped = await self._reap(deadline)
        if os.name != "nt" and via_tree:
            await self._wait_group_gone(self.proc.pid, deadline)
        return gone and reaped

    async def _wait_job_empty(self, deadline: float) -> bool:  # pragma: no cover - exercised on Windows only
        while time.monotonic() < deadline:
            try:
                if self._job is None or _active_processes(self._job) == 0:
                    return True
            except OSError:
                return True
            await asyncio.sleep(0.05)
        _log.debug("processes started by pid %s were still running %.0f s after being stopped",
                   self.proc.pid, _DESCENDANT_WAIT_S)
        return False

    async def _wait_group_gone(self, pgid: int, deadline: float) -> bool:
        # Killed members linger as zombies until their new parent reaps them; one that never does (a container
        # without an init process) only costs this bounded wait.
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)
            except (ProcessLookupError, PermissionError):
                return True
            await asyncio.sleep(0.05)
        _log.debug("process group %d was still present %.0f s after being killed", pgid, _DESCENDANT_WAIT_S)
        return False

    async def _reap(self, deadline: float) -> bool:
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=max(0.1, deadline - time.monotonic()))
            return True
        except asyncio.TimeoutError:
            return False

    def _release(self) -> None:
        job, self._job = self._job, None
        if job is not None and os.name == "nt":  # pragma: no cover - exercised on Windows only
            _kernel32.CloseHandle(job)

    async def aclose(self, *, aborted: bool = False) -> None:
        """Stop the tree if the program is still running, wait for it, then release the job (Windows; closing it
        also stops anything a finished program left running).

        ``aborted`` (the call was cancelled): stop the tree even when the program itself has already exited. A
        helper it started can still be running and holding the output pipe the call was waiting on; on POSIX
        nothing else would stop it."""
        try:
            if aborted or self.running():
                await self.kill()  # its first step, the stop itself, runs before anything can interrupt it
        finally:
            self._release()

    def __del__(self) -> None:
        try:
            if self.running() and not self._signal(group_after_exit=False):
                self.proc.kill()
            self._release()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass


def _is_real(proc: Any) -> bool:
    """A process asyncio started, not a test's stand-in (``MagicMock(spec=Process)`` passes ``isinstance``)."""
    return issubclass(type(proc), asyncio.subprocess.Process)
