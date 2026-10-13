"""Process checks for recovering a local run after the Studio server restarts."""
import os
import subprocess
from pathlib import Path

# Reuse the vendored Prompt Studio identity contract: executable plus OS creation
# marker. A PID alone can be reused by an unrelated process after a restart.
from .vendor.promptstudio.llamacpp_watchdog import _process_snapshot, _process_snapshots_match


def identity(pid):
    value = _process_snapshot(pid)
    if value is None and running(pid):
        raise ValueError('Cannot verify the trainer process identity. Leave it running and retry recovery.')
    return value


def same_process(actual, expected):
    return _process_snapshots_match(actual, expected)


def command_args(pid):
    """Read an old trainer's command for one-time migration; never execute it."""
    if type(pid) is not int or pid <= 0:
        raise ValueError('Invalid trainer process ID')
    if os.name != 'nt':
        return Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    import ctypes
    from ctypes import wintypes
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
        f'[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; (Get-CimInstance Win32_Process -Filter "ProcessId = {pid}").CommandLine'],
        capture_output=True, encoding='utf-8', timeout=10, check=True,
        creationflags=subprocess.CREATE_NO_WINDOW)
    command = result.stdout.strip().lstrip('\ufeff')
    if not command:
        raise ValueError('Cannot verify the previous trainer command')
    shell = ctypes.WinDLL('shell32', use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    count = ctypes.c_int()
    argv = shell.CommandLineToArgvW(command, ctypes.byref(count))
    if not argv:
        raise ValueError('Cannot read the previous trainer command')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    try:
        return [argv[i] for i in range(count.value)]
    finally:
        kernel.LocalFree(argv)

def running(pid):
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError('No recorded trainer process ID. Verify the old run manually before acknowledging it.')
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:  # ERROR_INVALID_PARAMETER: PID no longer exists
                return False
            raise ValueError('Unable to verify the previous trainer process. Check it manually.')
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ValueError('Unable to read the previous trainer process status.')
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False

def terminate_tree(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        import signal
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
