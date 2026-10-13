# Derived from tiko13/ComfyUI_PromptStudio; GPL-3.0-only. See NOTICE.md.


import hashlib, hmac, json, logging, math, os, re, secrets, subprocess, sys, threading, time, urllib.parse, uuid
from pathlib import Path
from .providers import _get_json, _reasoning_effort, _post_json, _clean_base_url, _clean_llamacpp_base_url, _list_llamacpp_models
BASE_DIR = str(Path(__file__).parent)
_WIRE_ENUMS = {'thinking_modes': ['Disabled', 'Minimal', 'Low', 'Medium', 'High', 'XHigh', 'Einstein', 'Spoon', 'Instruct Low', 'Instruct Medium', 'Instruct XHigh', 'Instruct Einstein', 'Instruct Spoon']}

LLAMACPP_CONFIG_DIRECTORY = None

LLAMACPP_PROCESS_STATE_PATH = None

LLAMACPP_OUTPUT_LOG_PATH = None

LLAMACPP_WATCHDOG_PATH = None

LLAMACPP_AUTOSTART_PATH = None

GPU_HANDOFF_TIMEOUT_SECONDS = 30 * 60

KOBOLD_ADMIN_TIMEOUT_SECONDS = 5 * 60

_KOBOLD_ADMIN_UNLOADED = set()

_LLAMACPP_PROCESS_LOCK = threading.RLock()

_LLAMACPP_PROCESS = None

_LLAMACPP_PROCESS_DETAILS = {}

_LLAMACPP_PICKER_LOCK = threading.Lock()

MAX_LLAMACPP_CONFIG_BYTES = 64 * 1024

LLAMACPP_LLM_PROFILE_DEFAULTS = {
    "thinking_mode": "Disabled",
    "thinking_modes": ["Disabled", "Minimal", "Low", "Medium", "High"],
    "max_response_tokens": 800,
    "llamacpp_reasoning_budget_tokens": 0,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 100,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "rep_pen": 1.05,
    "rep_pen_range": 360,
    "thinking_temperature": 0.7,
    "thinking_top_p": 0.9,
    "thinking_top_k": 100,
    "thinking_min_p": 0.0,
    "thinking_presence_penalty": 0.0,
    "thinking_rep_pen": 1.05,
    "thinking_rep_pen_range": 360,
    "sampler_seed": -1,
    "request_timeout": 120,
    "stop_sequence": "",
}

LLAMACPP_THINKING_MODES = tuple(_WIRE_ENUMS["thinking_modes"])

MAX_LLAMACPP_PROCESS_STATE_BYTES = 16 * 1024

MAX_LLAMACPP_AUTOSTART_BYTES = 16 * 1024

MAX_LLAMACPP_OUTPUT_TAIL_BYTES = 16 * 1024

LLAMACPP_STARTUP_GRACE_SECONDS = 1.25

LLAMACPP_PARENT_EXIT_GRACE_SECONDS = 120

def _llamacpp_output_tail():
    try:
        with open(LLAMACPP_OUTPUT_LOG_PATH, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - MAX_LLAMACPP_OUTPUT_TAIL_BYTES))
            output = handle.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""
    return output[-MAX_LLAMACPP_OUTPUT_TAIL_BYTES:]

def _llamacpp_error_detail(output):
    error_lines = [
        line.strip()
        for line in output.splitlines()
        if re.search(r"(?:^|\s)[EF]\s|error|invalid|failed", line, re.IGNORECASE)
    ]
    return "\n".join((error_lines or output.splitlines())[-8:]).strip()

def _llamacpp_startup_error(return_code):
    detail = _llamacpp_error_detail(_llamacpp_output_tail())
    message = f"Llama.cpp exited during startup with code {return_code}."
    return f"{message}\n{detail}" if detail else message

def _llamacpp_normalize_device_list(label, value):
    raw_value = _text(value).strip()
    if not raw_value:
        return ""
    devices = []
    for raw_device in raw_value.split(","):
        device = raw_device.strip()
        if not device or any(character.isspace() for character in device):
            raise ValueError(
                f"Llama.cpp config '{label}' must be a comma-separated device list "
                "without whitespace in device names"
            )
        devices.append(device)
    return ",".join(devices)

def _llamacpp_effective_main_gpu(main_gpu, cuda_devices):
    devices = [value.strip() for value in cuda_devices.split(",") if value.strip()]
    if not devices:
        return main_gpu
    physical_cuda_device = f"cuda{main_gpu}"
    for index, device in enumerate(devices):
        if device.casefold() == physical_cuda_device:
            return index
    if main_gpu >= len(devices):
        raise ValueError(
            "Llama.cpp config 'main_gpu' is not present in cuda_devices and exceeds the "
            f"filtered device index range 0-{len(devices) - 1}"
        )
    return main_gpu

def _normalized_process_path(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(_text(path).strip())))

def _windows_process_snapshot(pid, *, terminate=False, expected=None):
    import ctypes
    from ctypes import wintypes

    query = 0x1000  # PROCESS_QUERY_LIMITED_INFORMATION
    synchronize = 0x00100000
    terminate_access = 0x0001 if terminate else 0
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    access = query | synchronize | terminate_access
    handle = kernel32.OpenProcess(access, False, int(pid))
    if not handle:
        return None
    try:
        if kernel32.WaitForSingleObject(handle, 0) != 0x00000102:  # WAIT_TIMEOUT
            return None
        buffer = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buffer))
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return None
        created = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return None
        marker = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
        snapshot = {
            "executable": _normalized_process_path(buffer.value),
            "creation_marker": f"windows:{marker}",
        }
        if terminate:
            if not _process_snapshots_match(snapshot, expected):
                return None
            kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
            if not kernel32.TerminateProcess(handle, 1):
                raise OSError(ctypes.get_last_error(), "Could not terminate recovered Llama.cpp process")
        return snapshot
    finally:
        kernel32.CloseHandle(handle)

def _process_snapshot(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if os.name == "nt":
        try:
            return _windows_process_snapshot(pid)
        except (OSError, ValueError):
            return None
    proc_root = f"/proc/{pid}"
    try:
        executable = os.readlink(os.path.join(proc_root, "exe"))
        with open(os.path.join(proc_root, "stat"), "r", encoding="utf-8") as handle:
            stat = handle.read()
        fields = stat.rsplit(") ", 1)[1].split()
        start_ticks = fields[19]
    except (OSError, IndexError, ValueError):
        return None
    return {
        "executable": _normalized_process_path(executable),
        "creation_marker": f"proc:{start_ticks}",
    }

def _process_snapshots_match(actual, expected):
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return False
    actual_executable = _text(actual.get("executable")).strip()
    expected_executable = _text(expected.get("executable")).strip()
    actual_marker = _text(actual.get("creation_marker")).strip()
    expected_marker = _text(expected.get("creation_marker")).strip()
    return bool(
        actual_executable
        and expected_executable
        and actual_marker
        and expected_marker
        and _normalized_process_path(actual_executable)
        == _normalized_process_path(expected_executable)
        and hmac.compare_digest(actual_marker, expected_marker)
    )

def _remove_llamacpp_process_state():
    for path in (LLAMACPP_PROCESS_STATE_PATH, LLAMACPP_PROCESS_STATE_PATH + ".tmp"):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logging.warning("Could not remove stale Prompt Studio Llama.cpp process state: %s", exc)

def _write_llamacpp_process_state_payload(state):
    temporary_path = LLAMACPP_PROCESS_STATE_PATH + ".tmp"
    try:
        Path(LLAMACPP_PROCESS_STATE_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary_path, LLAMACPP_PROCESS_STATE_PATH)
    except OSError as exc:
        try:
            os.remove(temporary_path)
        except OSError:
            pass
        raise RuntimeError(
            "Could not save Prompt Studio Llama.cpp process ownership for restart recovery"
        ) from exc

def _write_llamacpp_process_state(process, details, watchdog):
    snapshot = None
    for _attempt in range(10):
        snapshot = _process_snapshot(process.pid)
        if snapshot is not None or process.poll() is not None:
            break
        time.sleep(0.05)
    if snapshot is None:
        _remove_llamacpp_process_state()
        raise RuntimeError(
            "Prompt Studio could not record the Llama.cpp process identity, so the server "
            "was not left running without restart-safe process control."
        )
    state = {
        "version": 2,
        "pid": int(process.pid),
        "identity": snapshot,
        "details": dict(details),
        "watchdog": dict(watchdog),
    }
    _write_llamacpp_process_state_payload(state)
    return state

def _read_llamacpp_process_state():
    try:
        if os.path.getsize(LLAMACPP_PROCESS_STATE_PATH) > MAX_LLAMACPP_PROCESS_STATE_BYTES:
            raise ValueError("process state exceeds the size limit")
        with open(LLAMACPP_PROCESS_STATE_PATH, "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logging.warning("Ignoring invalid Prompt Studio Llama.cpp process state: %s", exc)
        _remove_llamacpp_process_state()
        return None
    if not isinstance(state, dict) or state.get("version") not in {1, 2}:
        _remove_llamacpp_process_state()
        return None
    return state

def _read_llamacpp_autostart_config():
    try:
        if os.path.getsize(LLAMACPP_AUTOSTART_PATH) > MAX_LLAMACPP_AUTOSTART_BYTES:
            raise ValueError("autostart config exceeds the size limit")
        with open(LLAMACPP_AUTOSTART_PATH, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logging.warning("Ignoring invalid Prompt Studio Llama.cpp autostart config: %s", exc)
        return None
    if (
        not isinstance(config, dict)
        or config.get("version") != 1
        or config.get("enabled") is not True
        or not _text(config.get("llamacpp_executable")).strip()
        or not _text(config.get("llamacpp_config_profile")).strip()
    ):
        return None
    return config

def _llamacpp_autostart_status():
    config = _read_llamacpp_autostart_config()
    if config is None:
        return {"enabled": False, "llamacpp_executable": "", "llamacpp_config_profile": ""}
    return {
        "enabled": True,
        "llamacpp_executable": config["llamacpp_executable"],
        "llamacpp_config_profile": config["llamacpp_config_profile"],
        **({"keep_models_loaded": config["keep_models_loaded"]}
           if isinstance(config.get("keep_models_loaded"), bool) else {}),
    }

def _launch_llamacpp_watchdog(watchdog):
    command = [
        sys.executable,
        LLAMACPP_WATCHDOG_PATH,
        "--state-path",
        LLAMACPP_PROCESS_STATE_PATH,
        "--token",
        watchdog["token"],
    ]
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "shell": False,
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(command, **kwargs)
    except OSError as exc:
        raise RuntimeError(f"Could not start the Llama.cpp parent-process watchdog: {exc}") from exc

def _arm_llamacpp_watchdog(process, details, previous_state=None):
    if _process_snapshot(process.pid) is None:
        raise RuntimeError(
            "Prompt Studio could not record the Llama.cpp process identity, so the server "
            "was not left running without restart-safe process control."
        )
    owner_identity = _process_snapshot(os.getpid())
    if owner_identity is None:
        raise RuntimeError(
            "Prompt Studio could not record the ComfyUI process identity for the Llama.cpp watchdog"
        )
    watchdog = {
        "token": secrets.token_hex(16),
        "owner_pid": os.getpid(),
        "owner_identity": owner_identity,
        "grace_seconds": LLAMACPP_PARENT_EXIT_GRACE_SECONDS,
    }
    _write_llamacpp_process_state(process, details, watchdog)
    try:
        _launch_llamacpp_watchdog(watchdog)
    except Exception:
        if previous_state is None:
            _remove_llamacpp_process_state()
        else:
            _write_llamacpp_process_state_payload(previous_state)
        raise
    return watchdog

def _terminate_recovered_process(pid, expected, *, force=False):
    if os.name == "nt":
        if _windows_process_snapshot(pid, terminate=True, expected=expected) is None:
            raise ProcessLookupError(f"Recovered Llama.cpp process {pid} is no longer running")
        return
    import signal

    actual = _process_snapshot(pid)
    if not _process_snapshots_match(actual, expected):
        raise ProcessLookupError(f"Recovered Llama.cpp process {pid} is no longer running")
    os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)

class _RecoveredLlamacppProcess:
    def __init__(self, pid, identity):
        self.pid = int(pid)
        self.identity = dict(identity)
        self.returncode = None

    def poll(self):
        if _process_snapshots_match(_process_snapshot(self.pid), self.identity):
            return None
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        _terminate_recovered_process(self.pid, self.identity)

    def kill(self):
        _terminate_recovered_process(self.pid, self.identity, force=True)

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.identity.get("executable"), timeout)
            time.sleep(0.05)
        return self.returncode

def _recover_llamacpp_process_locked():
    global _LLAMACPP_PROCESS, _LLAMACPP_PROCESS_DETAILS
    if _LLAMACPP_PROCESS is not None:
        return
    state = _read_llamacpp_process_state()
    if state is None:
        return
    try:
        pid = int(state.get("pid"))
    except (TypeError, ValueError):
        _remove_llamacpp_process_state()
        return
    identity = state.get("identity")
    details = state.get("details")
    executable = _text(identity.get("executable") if isinstance(identity, dict) else "").strip()
    if (
        not isinstance(details, dict)
        or os.path.basename(executable).casefold()
        not in {"llama.exe", "llama", "llama-server.exe", "llama-server"}
        or not _process_snapshots_match(_process_snapshot(pid), identity)
    ):
        _remove_llamacpp_process_state()
        return
    _LLAMACPP_PROCESS = _RecoveredLlamacppProcess(pid, identity)
    _LLAMACPP_PROCESS_DETAILS = dict(details)
    try:
        _arm_llamacpp_watchdog(_LLAMACPP_PROCESS, _LLAMACPP_PROCESS_DETAILS, state)
    except Exception as exc:
        logging.warning("Could not renew the Prompt Studio Llama.cpp watchdog: %s", exc)

def _keep_models_loaded(data):
    value = data.get("keep_models_loaded", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return bool(value)

def _validate_llamacpp_picker_selection(kind, path):
    kind = _text(kind).strip().casefold()
    selected = os.path.abspath(_text(path).strip()) if _text(path).strip() else ""
    if kind not in {"executable", "model", "mmproj"}:
        raise ValueError("Llama.cpp picker kind must be executable, model, or mmproj")
    if not selected:
        return ""
    if not os.path.isfile(selected):
        raise ValueError(f"Selected file was not found: {selected}")
    if kind != "executable":
        if os.path.splitext(selected)[1].casefold() != ".gguf":
            raise ValueError("Select a GGUF file")
        return selected
    if os.path.basename(selected).casefold() not in {"llama.exe", "llama-server.exe"}:
        raise ValueError("Select llama.exe or llama-server.exe")
    return selected

def _pick_llamacpp_file(kind, current_path=""):
    if os.name != "nt":
        raise RuntimeError("The Llama.cpp file picker is currently available on Windows only")
    kind = _text(kind).strip().casefold()
    if kind not in {"executable", "model", "mmproj"}:
        raise ValueError("Llama.cpp picker kind must be executable, model, or mmproj")
    if not _LLAMACPP_PICKER_LOCK.acquire(blocking=False):
        raise RuntimeError("A Llama.cpp file picker is already open")
    root = None
    try:
        import tkinter
        from tkinter import filedialog

        current = _text(current_path).strip()
        current_directory = os.path.dirname(os.path.abspath(current)) if current else BASE_DIR
        if not os.path.isdir(current_directory):
            current_directory = BASE_DIR
        options = {
            "initialdir": current_directory,
            "title": "Select the Llama.cpp server executable",
            "filetypes": [
                ("Llama.cpp server", "llama.exe llama-server.exe"),
                ("Executable files", "*.exe"),
                ("All files", "*.*"),
            ],
        }
        if current and os.path.basename(current):
            options["initialfile"] = os.path.basename(current)
        if kind != "executable":
            options.update(title=f"Select {kind} GGUF", filetypes=[("GGUF files", "*.gguf")])
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.update_idletasks()
        selected = filedialog.askopenfilename(parent=root, **options)
        return _validate_llamacpp_picker_selection(kind, selected)
    except ImportError as exc:
        raise RuntimeError("This Python installation does not provide the native file picker") from exc
    finally:
        if root is not None:
            root.destroy()
        _LLAMACPP_PICKER_LOCK.release()

def _llamacpp_config_profile(value, default=""):
    profile = _text(value).strip() or default
    if not profile:
        raise ValueError("Select a Llama.cpp config profile")
    if any(char in profile for char in ("/", "\\", ":")) or profile != os.path.basename(profile) or os.path.splitext(profile)[1].casefold() != ".json":
        raise ValueError("Llama.cpp config profile must be a JSON filename")
    if profile in {".", ".."} or len(profile) > 255 or "\x00" in profile:
        raise ValueError("Llama.cpp config profile name is invalid")
    return profile

def _llamacpp_config_directory():
    directory = os.path.abspath(LLAMACPP_CONFIG_DIRECTORY)
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Could not create the Llama.cpp config profiles folder: {exc}") from exc
    return directory

def _list_llamacpp_config_profiles():
    config_directory = _llamacpp_config_directory()
    profiles = []
    try:
        entries = os.scandir(config_directory)
        with entries:
            for entry in entries:
                if entry.is_file() and os.path.splitext(entry.name)[1].casefold() == ".json":
                    profiles.append(entry.name)
    except OSError as exc:
        raise ValueError(f"Could not read the Llama.cpp config profiles folder: {exc}") from exc
    return sorted(profiles, key=str.casefold)

def _read_llamacpp_config_document(profile):
    config_path = _llamacpp_resolve_config_path(profile)
    if not os.path.isfile(config_path):
        raise ValueError(f"Llama.cpp config file was not found: {config_path}")
    try:
        with open(config_path, "rb") as handle:
            raw_config = handle.read(MAX_LLAMACPP_CONFIG_BYTES + 1)
        if len(raw_config) > MAX_LLAMACPP_CONFIG_BYTES:
            raise ValueError("Llama.cpp config file exceeds the 64 KB limit")
        config = json.loads(raw_config.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read Llama.cpp config JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Llama.cpp config JSON must contain an object at the root")
    return config, raw_config, config_path

def _inspect_llamacpp_config_profiles():
    profiles = []
    for name in _list_llamacpp_config_profiles():
        item = {"name": name}
        try:
            config, _, _ = _read_llamacpp_config_document(name)
            for key in ("model", "mmproj"):
                # Match launcher alias precedence and relative-path resolution.
                value = _text(config.get(key, config.get(key + "_gguf"))).strip()
                item[key] = {
                    "path": value,
                    "present": os.path.isfile(os.path.abspath(value)) if value else None,
                }
        except ValueError as exc:
            item["error"] = str(exc)
        profiles.append(item)
    return profiles

def _delete_llamacpp_config_profile(profile):
    with _LLAMACPP_PROCESS_LOCK:
        path = _llamacpp_resolve_config_path(profile)
        if not os.path.isfile(path):
            raise ValueError("Llama.cpp config profile was already deleted or is unavailable")
        startup = _read_llamacpp_autostart_config()
        startup_matches = startup and os.path.normcase(
            _llamacpp_resolve_config_path(startup["llamacpp_config_profile"])
        ) == os.path.normcase(path)
        if startup_matches:
            # Fail rather than silently leave startup pointing at a deleted config.
            for startup_path in (LLAMACPP_AUTOSTART_PATH + ".tmp", LLAMACPP_AUTOSTART_PATH):
                try:
                    os.remove(startup_path)
                except FileNotFoundError:
                    pass
        # Unlink only this directory entry; never remove referenced assets or router files.
        os.remove(path)
        return {"deleted": os.path.basename(path), "autostart": _llamacpp_autostart_status()}

def _llamacpp_profile_mode_options(profile):
    return list(dict.fromkeys([
        *profile.get("thinking_modes", []),
        *(mode if mode == "Disabled" else f"Instruct {mode}"
          for mode in profile.get("instruct_modes", [])),
    ]))

def _normalize_llamacpp_llm_profile(value):
    source = value if isinstance(value, dict) else {}
    defaults = LLAMACPP_LLM_PROFILE_DEFAULTS

    def number(key, minimum, maximum, *, integer=False):
        requested = source.get(key, defaults[key])
        try:
            normalized = int(requested) if integer else float(requested)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Llama.cpp config 'llm_profile.{key}' must be a number") from exc
        if not math.isfinite(normalized) or normalized < minimum or normalized > maximum:
            raise ValueError(
                f"Llama.cpp config 'llm_profile.{key}' must be between {minimum} and {maximum}"
            )
        return int(normalized) if integer else normalized

    split_modes = "instruct_modes" in source
    requested_modes = source.get("thinking_modes", [] if split_modes else defaults["thinking_modes"])
    if not isinstance(requested_modes, list):
        raise ValueError("Llama.cpp config 'llm_profile.thinking_modes' must be a list")
    modes = []
    for requested in requested_modes:
        mode = next(
            (candidate for candidate in LLAMACPP_THINKING_MODES
             if candidate.casefold() == _text(requested).strip().casefold()),
            None,
        )
        if mode is None:
            raise ValueError(
                "Llama.cpp config 'llm_profile.thinking_modes' may contain only "
                + ", ".join(LLAMACPP_THINKING_MODES)
            )
        if split_modes and _reasoning_effort(mode) == "none":
            raise ValueError(f"Llama.cpp mode '{mode}' belongs in instruct_modes")
        if mode not in modes:
            modes.append(mode)
    instruct_modes = []
    if split_modes:
        requested_instruct = source["instruct_modes"]
        if not isinstance(requested_instruct, list):
            raise ValueError("Llama.cpp config 'llm_profile.instruct_modes' must be a list")
        for requested in requested_instruct:
            mode = next((candidate for candidate in ("Disabled", "Low", "Medium", "XHigh", "Einstein", "Spoon")
                         if candidate.casefold() == _text(requested).strip().casefold()), None)
            if mode is None:
                raise ValueError("Llama.cpp instruct_modes may contain only Disabled, Low, Medium, XHigh, Einstein, Spoon")
            if mode not in instruct_modes:
                instruct_modes.append(mode)
    available_modes = _llamacpp_profile_mode_options({"thinking_modes": modes, "instruct_modes": instruct_modes})
    if not available_modes:
        raise ValueError("Llama.cpp config must define at least one thinking or instruct mode")
    requested_default = _text(source.get("thinking_mode"), defaults["thinking_mode"]).strip()
    thinking_mode = next(
        (mode for mode in available_modes if mode.casefold() == requested_default.casefold()),
        None,
    )
    if thinking_mode is None:
        raise ValueError(
            "Llama.cpp config 'llm_profile.thinking_mode' must be included in the available modes"
        )
    stop_sequence = source.get("stop_sequence", defaults["stop_sequence"])
    if not isinstance(stop_sequence, str) or len(stop_sequence) > 4096:
        raise ValueError("Llama.cpp config 'llm_profile.stop_sequence' must be a string up to 4096 characters")
    return {
        "thinking_mode": thinking_mode,
        "thinking_modes": modes,
        **({"instruct_modes": instruct_modes} if split_modes else {}),
        "max_response_tokens": number("max_response_tokens", 0, 8192, integer=True),
        "llamacpp_reasoning_budget_tokens": number(
            "llamacpp_reasoning_budget_tokens", 0, 262144, integer=True
        ),
        "temperature": number("temperature", 0, 5),
        "top_p": number("top_p", 0, 1),
        "top_k": number("top_k", 0, 200, integer=True),
        "min_p": number("min_p", 0, 1),
        "presence_penalty": number("presence_penalty", -2, 2),
        "rep_pen": number("rep_pen", 0.5, 3),
        "rep_pen_range": number("rep_pen_range", 0, 4096, integer=True),
        "thinking_temperature": number("thinking_temperature", 0, 5),
        "thinking_top_p": number("thinking_top_p", 0, 1),
        "thinking_top_k": number("thinking_top_k", 0, 200, integer=True),
        "thinking_min_p": number("thinking_min_p", 0, 1),
        "thinking_presence_penalty": number("thinking_presence_penalty", -2, 2),
        "thinking_rep_pen": number("thinking_rep_pen", 0.5, 3),
        "thinking_rep_pen_range": number("thinking_rep_pen_range", 0, 4096, integer=True),
        "sampler_seed": number("sampler_seed", -1, 999999, integer=True),
        "request_timeout": number("request_timeout", 5, 600, integer=True),
        "stop_sequence": stop_sequence,
    }

def _llamacpp_config_llm_profile(profile):
    config, _, _ = _read_llamacpp_config_document(profile)
    return _normalize_llamacpp_llm_profile(config.get("llm_profile"))

def _llamacpp_configured_generation_data(data):
    if _text(data.get("llm_provider"), "koboldcpp").strip().casefold() != "llamacpp":
        return data
    profile_name = _text(data.get("llamacpp_config_profile")).strip()
    if not profile_name:
        # Preserve direct/external llama-server callers that do not opt into a
        # Prompt Studio config file. The UI selects a config whenever one exists.
        return data
    profile = _llamacpp_config_llm_profile(profile_name)
    requested_mode = _text(data.get("thinking_mode"), profile["thinking_mode"]).strip()
    thinking_mode = next(
        (mode for mode in _llamacpp_profile_mode_options(profile) if mode.casefold() == requested_mode.casefold()),
        profile["thinking_mode"],
    )
    thinking = _reasoning_effort(thinking_mode) != "none"
    sampler_keys = (
        ("temperature", "thinking_temperature"),
        ("top_p", "thinking_top_p"),
        ("top_k", "thinking_top_k"),
        ("min_p", "thinking_min_p"),
        ("presence_penalty", "thinking_presence_penalty"),
        ("rep_pen", "thinking_rep_pen"),
        ("rep_pen_range", "thinking_rep_pen_range"),
    )
    configured = {
        "thinking_mode": thinking_mode,
        "max_response_tokens": profile["max_response_tokens"],
        "llamacpp_reasoning_budget_tokens": profile["llamacpp_reasoning_budget_tokens"],
        "sampler_seed": profile["sampler_seed"],
        "request_timeout": profile["request_timeout"],
        "stop_sequence": profile["stop_sequence"],
    }
    configured.update({
        standard: profile[thinking_key if thinking else standard]
        for standard, thinking_key in sampler_keys
    })
    internal_overrides = data.get("_llamacpp_generation_overrides")
    if isinstance(internal_overrides, dict):
        configured.update(internal_overrides)
    return {**data, **configured}

def _llamacpp_resolve_config_path(profile, *, default_profile=""):
    config_directory = _llamacpp_config_directory()
    config_profile = _llamacpp_config_profile(profile, default_profile)
    config_path = os.path.abspath(os.path.join(config_directory, config_profile))
    if os.path.dirname(config_path) != config_directory:
        raise ValueError("Llama.cpp config profile must stay inside the configured folder")
    return config_path

class LlamacppConfigConflict(ValueError):
    pass

def _edit_llamacpp_config(data):
    """Read/save local profiles without launching an editor or model process."""
    action = data.get("action", "load")
    if action == "new":
        profile = dict(LLAMACPP_LLM_PROFILE_DEFAULTS)
        profile.update(thinking_mode="Low", thinking_modes=["Minimal", "Low", "Medium", "High"], instruct_modes=[])
        return {"config": {"llm_profile": profile}, "revision": None, "native_editor": True,
                "can_browse": os.name == "nt"}
    name = _llamacpp_config_profile(data.get("llamacpp_config_profile"))
    path = _llamacpp_resolve_config_path(name)
    if action == "load":
        config, raw, _ = _read_llamacpp_config_document(name)
        config["llm_profile"] = {**(config.get("llm_profile") or {}),
                                 **_normalize_llamacpp_llm_profile(config.get("llm_profile"))}
        return {"config": config, "revision": hashlib.sha256(raw).hexdigest(),
                "config_profile": name, "native_editor": True, "can_browse": os.name == "nt"}
    if action != "save":
        raise ValueError("Unknown Llama.cpp config editor action")
    config = data.get("config")
    if not isinstance(config, dict):
        raise ValueError("Llama.cpp config JSON must contain an object at the root")
    config = dict(config)
    if not isinstance(config.get("llm_profile", {}), dict):
        raise ValueError("Llama.cpp llm_profile must be an object")
    config["llm_profile"] = {**config.get("llm_profile", {}),
                             **_normalize_llamacpp_llm_profile(config.get("llm_profile"))}
    raw = (json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > MAX_LLAMACPP_CONFIG_BYTES:
        raise ValueError("Llama.cpp config file exceeds the 64 KB limit")
    _validate_llamacpp_launcher_document(config, "llama-server", path, raw)
    with _LLAMACPP_PROCESS_LOCK:
        if os.path.lexists(path):
            if os.path.islink(path):
                raise ValueError("Cannot save a symlinked Llama.cpp profile")
            _, previous, _ = _read_llamacpp_config_document(name)
            if data.get("revision") != hashlib.sha256(previous).hexdigest():
                raise LlamacppConfigConflict("This profile already exists or changed since opening. Reopen it or use a different name.")
        elif data.get("revision") is not None:
            raise LlamacppConfigConflict("This profile was deleted. Reopen the editor to create a new profile.")
        temporary = path + "." + uuid.uuid4().hex + ".tmp"
        try:
            with open(temporary, "xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
    return {"saved": True, "config_profile": name, "revision": hashlib.sha256(raw).hexdigest(),
            "llm_profile": config["llm_profile"]}

def _llamacpp_launcher_paths(data):
    executable = os.path.abspath(_text(data.get("llamacpp_executable")).strip())
    if not _text(data.get("llamacpp_executable")).strip():
        raise ValueError("Set the Llama.cpp executable path in Prompt Studio settings")
    if not os.path.isfile(executable):
        raise ValueError(f"Llama.cpp executable was not found: {executable}")
    if os.path.basename(executable).casefold() not in {
        "llama.exe", "llama", "llama-server.exe", "llama-server",
    }:
        raise ValueError("Llama.cpp executable must be llama.exe or llama-server.exe")
    profile_value = _text(data.get("llamacpp_config_profile")).strip()
    legacy_path = _text(data.get("llamacpp_config_path")).strip()
    if not profile_value and legacy_path:
        profile_value = profile_value or os.path.basename(legacy_path)
    config_path = _llamacpp_resolve_config_path(profile_value)
    if not os.path.isfile(config_path):
        raise ValueError(f"Llama.cpp config file was not found: {config_path}")
    if os.path.getsize(config_path) > MAX_LLAMACPP_CONFIG_BYTES:
        raise ValueError("Llama.cpp config file exceeds the 64 KB limit")
    return executable, config_path

def _load_llamacpp_launcher_config(data):
    executable, config_path = _llamacpp_launcher_paths(data)
    try:
        with open(config_path, "rb") as handle:
            raw_config = handle.read(MAX_LLAMACPP_CONFIG_BYTES + 1)
        if len(raw_config) > MAX_LLAMACPP_CONFIG_BYTES:
            raise ValueError("Llama.cpp config file exceeds the 64 KB limit")
        config = json.loads(raw_config.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read Llama.cpp config JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Llama.cpp config JSON must contain an object at the root")

    return _validate_llamacpp_launcher_document(config, executable, config_path, raw_config)

def _validate_llamacpp_launcher_document(config, executable, config_path, raw_config):
    """Shared validation for launch and the native config editor; never starts a process."""
    def first(*keys, default=None):
        for key in keys:
            if key in config:
                return config[key]
        return default

    def integer(label, *keys, default, minimum, maximum):
        try:
            requested = first(*keys, default=default)
            value = int(requested)
            if isinstance(requested, bool) or float(requested) != value:
                raise ValueError("not an integer")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Llama.cpp config '{label}' must be an integer") from exc
        if value < minimum or value > maximum:
            raise ValueError(f"Llama.cpp config '{label}' must be between {minimum} and {maximum}")
        return value

    def number(label, *keys, default, minimum, maximum):
        try:
            value = float(first(*keys, default=default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Llama.cpp config '{label}' must be a number") from exc
        if not math.isfinite(value) or value < minimum or value > maximum:
            raise ValueError(
                f"Llama.cpp config '{label}' must be between {minimum} and {maximum}"
            )
        return value

    model = os.path.abspath(_text(first("model", "model_gguf")).strip())
    if not _text(first("model", "model_gguf")).strip() or not os.path.isfile(model):
        raise ValueError(f"Llama.cpp model GGUF was not found: {model}")
    mmproj_value = _text(first("mmproj", "mmproj_gguf")).strip()
    mmproj = os.path.abspath(mmproj_value) if mmproj_value else ""
    if mmproj and not os.path.isfile(mmproj):
        raise ValueError(f"Llama.cpp MMProj GGUF was not found: {mmproj}")
    gpu_layers_value = first("gpu_layers", "n_gpu_layers", default="all")
    gpu_layers = _text(gpu_layers_value).strip().casefold()
    if gpu_layers != "all":
        try:
            gpu_layers = str(int(gpu_layers_value))
        except (TypeError, ValueError) as exc:
            raise ValueError("Llama.cpp config 'gpu_layers' must be an integer or 'all'") from exc
        if int(gpu_layers) < 0:
            raise ValueError("Llama.cpp config 'gpu_layers' must be non-negative or 'all'")
    split_mode = _text(first("split_mode", default="layer")).strip().casefold()
    if split_mode not in {"none", "layer", "row", "tensor"}:
        raise ValueError("Llama.cpp config 'split_mode' must be none, layer, row, or tensor")
    flash_attention = _text(first("flash_attention", "flash_attn", default="auto")).strip().casefold()
    if flash_attention not in {"auto", "on", "off"}:
        raise ValueError("Llama.cpp config 'flash_attention' must be auto, on, or off")
    auto_fit = _text(first("auto_fit", "fit", default="default")).strip().casefold()
    if auto_fit not in {"default", "on", "off"}:
        raise ValueError("Llama.cpp config 'auto_fit' must be default, on, or off")
    cache_type_k = _text(first("kv_cache_k", "cache_type_k", default="f16")).strip()
    cache_type_v = _text(first("kv_cache_v", "cache_type_v", default="f16")).strip()
    if not cache_type_k or not cache_type_v:
        raise ValueError("Llama.cpp KV cache types must not be empty")
    mtp_value = first("mtp_enabled", "mtp", default="off")
    if isinstance(mtp_value, bool):
        mtp_enabled = mtp_value
    else:
        normalized_mtp = _text(mtp_value).strip().casefold()
        if normalized_mtp not in {"on", "off"}:
            raise ValueError("Llama.cpp config 'mtp_enabled' must be on or off")
        mtp_enabled = normalized_mtp == "on"
    mtp_draft_tokens = integer(
        "mtp_draft_tokens", "mtp_draft_tokens", "spec_draft_n_max",
        default=3, minimum=1, maximum=1024,
    )
    mtp_min_draft_tokens = integer(
        "mtp_min_draft_tokens", "mtp_min_draft_tokens", "spec_draft_n_min",
        default=0, minimum=0, maximum=1024,
    )
    if mtp_min_draft_tokens > mtp_draft_tokens:
        raise ValueError(
            "Llama.cpp config 'mtp_min_draft_tokens' must not exceed 'mtp_draft_tokens'"
        )
    mtp_min_probability = number(
        "mtp_min_probability", "mtp_min_probability", "spec_draft_p_min",
        default=0.0, minimum=0.0, maximum=1.0,
    )
    mtp_gpu_layers_value = first(
        "mtp_gpu_layers", "spec_draft_gpu_layers", default="auto",
    )
    mtp_gpu_layers = _text(mtp_gpu_layers_value).strip().casefold()
    if mtp_gpu_layers not in {"auto", "all"}:
        try:
            mtp_gpu_layers = str(int(mtp_gpu_layers_value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Llama.cpp config 'mtp_gpu_layers' must be an integer, 'auto', or 'all'"
            ) from exc
        if int(mtp_gpu_layers) < 0:
            raise ValueError(
                "Llama.cpp config 'mtp_gpu_layers' must be non-negative, 'auto', or 'all'"
            )
    mtp_device = _llamacpp_normalize_device_list(
        "mtp_device", first("mtp_device", "spec_draft_device", default=""),
    )
    mtp_cache_type_k = _text(first(
        "mtp_kv_cache_k", "spec_draft_cache_type_k", default="f16",
    )).strip()
    mtp_cache_type_v = _text(first(
        "mtp_kv_cache_v", "spec_draft_cache_type_v", default="f16",
    )).strip()
    if not mtp_cache_type_k or not mtp_cache_type_v:
        raise ValueError("Llama.cpp MTP KV cache types must not be empty")
    host = _text(first("host", default="127.0.0.1")).strip() or "127.0.0.1"
    if any(character.isspace() for character in host):
        raise ValueError("Llama.cpp config 'host' is invalid")
    port = integer("port", "port", default=8080, minimum=1, maximum=65535)
    parallel = integer(
        "parallel_slots", "parallel_slots", "parallel", default=1, minimum=1, maximum=1024,
    )
    context_size = integer(
        "context_size", "context_size", "ctx_size",
        default=32768, minimum=128, maximum=16_777_216,
    )
    main_gpu = integer(
        "main_gpu", "main_gpu", "main_gpu_index", default=0, minimum=0, maximum=1024,
    )
    tensor_split = _text(first("tensor_split", default="")).strip()
    if tensor_split and not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:,[0-9]+(?:\.[0-9]+)?)*", tensor_split):
        raise ValueError("Tensor split must be a comma-separated numeric list, such as 2,1,1")
    cuda_devices = _llamacpp_normalize_device_list(
        "cuda_devices", first("cuda_devices", default=""),
    )
    cuda_visible_devices = _text(first("cuda_visible_devices", default="")).strip()
    effective_main_gpu = _llamacpp_effective_main_gpu(main_gpu, cuda_devices)
    extra_args = first("extra_args", default=[])
    if not isinstance(extra_args, list) or not all(isinstance(value, str) for value in extra_args):
        raise ValueError("Llama.cpp config 'extra_args' must be a list of strings")
    if len(extra_args) > 128 or any("\x00" in value or len(value) > 4096 for value in extra_args):
        raise ValueError("Llama.cpp config 'extra_args' is too large or contains invalid values")
    extra_args = [value.strip() for value in extra_args if value.strip()]

    command = [executable]
    if os.path.basename(executable).casefold() in {"llama.exe", "llama"}:
        command.append("serve")
    command.extend(["--model", model])
    if mmproj:
        command.extend(["--mmproj", mmproj])
    command.extend([
        "--ctx-size", str(context_size),
        "--gpu-layers", gpu_layers,
        "--split-mode", split_mode,
        "--main-gpu", str(effective_main_gpu),
        "--flash-attn", flash_attention,
        "--cache-type-k", cache_type_k,
        "--cache-type-v", cache_type_v,
        "--parallel", str(parallel),
        "--host", host,
        "--port", str(port),
        "--slots",
    ])
    if tensor_split:
        command.extend(["--tensor-split", tensor_split])
    if cuda_devices:
        command.extend(["--device", cuda_devices])
    if auto_fit != "default":
        command.extend(["--fit", auto_fit])
    if mtp_enabled:
        command.extend([
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(mtp_draft_tokens),
            "--spec-draft-n-min", str(mtp_min_draft_tokens),
            "--spec-draft-p-min", format(mtp_min_probability, ".15g"),
            "--spec-draft-ngl", mtp_gpu_layers,
            "--spec-draft-type-k", mtp_cache_type_k,
            "--spec-draft-type-v", mtp_cache_type_v,
        ])
        if mtp_device:
            command.extend(["--spec-draft-device", mtp_device])
    command.extend(extra_args)
    url_host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return {
        "executable": executable,
        "model": model,
        "mmproj": mmproj,
        "config_path": config_path,
        "config_revision": hashlib.sha256(raw_config).hexdigest(),
        "command": command,
        "environment": {"CUDA_VISIBLE_DEVICES": cuda_visible_devices} if cuda_visible_devices else {},
        "host": host,
        "port": port,
        "url": f"http://{url_host}:{port}",
    }

def _llamacpp_router_command(launcher, *, keep_models_loaded=False):
    """Keep the HTTP server alive while its selected model can be unloaded."""
    model = launcher["model"]
    mmproj = launcher["mmproj"]
    if any(char in model for char in "\r\n]") or any(char in mmproj for char in "\r\n"):
        raise ValueError("Llama.cpp model paths contain characters unsupported by router presets")
    preset_path = launcher["config_path"] + ".router.ini"
    # A Windows drive colon in a preset section is interpreted as a quant tag.
    # Use the filename as the ID and retain the old full-path ID as an alias.
    model_id = os.path.basename(model)
    lines = ["version = 1", f"[{model_id}]", f"model = {model}", f"alias = {model}"]
    if mmproj:
        lines.append(f"mmproj = {mmproj}")
    if keep_models_loaded:
        lines.append("load-on-startup = true")
    # Remaining CLI options are inherited by the model worker, including GPU,
    # MTP, cache settings and extra arguments.
    command = list(launcher["command"])
    for flag in ("--model", "--mmproj"):
        if flag in command:
            index = command.index(flag)
            del command[index:index + 2]
    with open(preset_path + ".tmp", "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    os.replace(preset_path + ".tmp", preset_path)
    command.extend(["--models-preset", preset_path, "--models-max", "1"])
    return command

def _llamacpp_managed_process_status(data=None):
    global _LLAMACPP_PROCESS
    with _LLAMACPP_PROCESS_LOCK:
        _recover_llamacpp_process_locked()
        process = _LLAMACPP_PROCESS
        details = dict(_LLAMACPP_PROCESS_DETAILS)
        if process is None:
            running = False
            return_code = details.get("return_code")
        else:
            return_code = process.poll()
            running = return_code is None
            if not running:
                details["return_code"] = return_code
                output = _llamacpp_output_tail()
                if output:
                    details["last_output"] = output
                _LLAMACPP_PROCESS_DETAILS.update(details)
                _LLAMACPP_PROCESS = None
                _remove_llamacpp_process_state()
        status = {
            "managed": process is not None or bool(details.get("started_at")),
            "running": running,
            "pid": process.pid if running else None,
            "return_code": return_code,
        }
        for key in (
            "url", "executable", "config_path", "config_profile", "config_revision", "started_at", "model",
            "last_output",
        ):
            if details.get(key) is not None:
                status[key] = details[key]
        if data:
            status["configured"] = bool(
                _text(data.get("llamacpp_executable")).strip()
                and (
                    _text(data.get("llamacpp_config_profile")).strip()
                    or _text(data.get("llamacpp_config_path")).strip()
                )
            )
            if running and details.get("config_revision"):
                try:
                    selected = _load_llamacpp_launcher_config(data)
                except (OSError, ValueError):
                    status["config_changed"] = True
                else:
                    status["config_changed"] = (
                        _normalized_process_path(selected["config_path"])
                        != _normalized_process_path(details.get("config_path"))
                        or not hmac.compare_digest(
                            selected["config_revision"],
                            details["config_revision"],
                        )
                    )
                    if not status["config_changed"]:
                        status["model"] = os.path.basename(selected["model"])
        return status

def _start_llamacpp_server(data, *, allow_external=True):
    global _LLAMACPP_PROCESS, _LLAMACPP_PROCESS_DETAILS
    with _LLAMACPP_PROCESS_LOCK:
        _recover_llamacpp_process_locked()
        current = _LLAMACPP_PROCESS
        if current is not None and current.poll() is None:
            return {**_llamacpp_managed_process_status(data), "already_running": True}
        # Config Builder replaces profiles atomically. Reading inside the launch lock,
        # immediately before Popen, keeps the command and revision on one saved version.
        launcher = _load_llamacpp_launcher_config(data)
        if isinstance(_get_json(urllib.parse.urljoin(launcher["url"] + "/", "health"), 2, "Llama.cpp"), dict):
            if not allow_external:
                raise RuntimeError(
                    "The previous Llama.cpp server stopped, but its endpoint is still responding. "
                    "The edited config was not started; retry Restart after the endpoint goes offline."
                )
            return {
                "managed": False,
                "running": True,
                "already_running": True,
                "external": True,
                "url": launcher["url"],
            }
        environment = os.environ.copy()
        if "CUDA_VISIBLE_DEVICES" not in launcher["environment"]:
            environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment.update(launcher["environment"])
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            output_handle = open(LLAMACPP_OUTPUT_LOG_PATH, "wb")
        except OSError as exc:
            raise RuntimeError(f"Could not open the Llama.cpp startup log: {exc}") from exc
        try:
            try:
                process = subprocess.Popen(
                    _llamacpp_router_command(launcher, keep_models_loaded=_keep_models_loaded(data)),
                    cwd=os.path.dirname(launcher["executable"]),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=output_handle,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    creationflags=creation_flags,
                )
            except OSError as exc:
                raise RuntimeError(f"Could not start Llama.cpp: {exc}") from exc
        finally:
            output_handle.close()
        try:
            return_code = process.wait(timeout=LLAMACPP_STARTUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            return_code = None
        if return_code is not None:
            _LLAMACPP_PROCESS = None
            _LLAMACPP_PROCESS_DETAILS = {
                "url": launcher["url"],
                "executable": launcher["executable"],
                "config_path": launcher["config_path"],
                "config_profile": os.path.basename(launcher["config_path"]),
                "config_revision": launcher["config_revision"],
                "return_code": return_code,
                "last_output": _llamacpp_output_tail(),
            }
            _remove_llamacpp_process_state()
            raise RuntimeError(_llamacpp_startup_error(return_code))
        _LLAMACPP_PROCESS = process
        _LLAMACPP_PROCESS_DETAILS = {
            "url": launcher["url"],
            "model": os.path.basename(launcher["model"]),
            "executable": launcher["executable"],
            "config_path": launcher["config_path"],
            "config_profile": os.path.basename(launcher["config_path"]),
            "config_revision": launcher["config_revision"],
            "started_at": time.time(),
        }
        try:
            _arm_llamacpp_watchdog(process, _LLAMACPP_PROCESS_DETAILS)
        except Exception:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            _LLAMACPP_PROCESS = None
            _LLAMACPP_PROCESS_DETAILS = {}
            _remove_llamacpp_process_state()
            raise
        return _llamacpp_managed_process_status(data)

def _stop_llamacpp_server(data=None):
    global _LLAMACPP_PROCESS
    with _LLAMACPP_PROCESS_LOCK:
        _recover_llamacpp_process_locked()
        process = _LLAMACPP_PROCESS
        if process is None or process.poll() is not None:
            _LLAMACPP_PROCESS = None
            _remove_llamacpp_process_state()
            return {**_llamacpp_managed_process_status(data), "stopped": False}
        try:
            process.terminate()
        except ProcessLookupError:
            process.returncode = 0
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    with _LLAMACPP_PROCESS_LOCK:
        _LLAMACPP_PROCESS_DETAILS["return_code"] = process.returncode
        _LLAMACPP_PROCESS = None
        _remove_llamacpp_process_state()
        return {**_llamacpp_managed_process_status(data), "stopped": True}

def _restart_llamacpp_server(data):
    # Validate before interrupting a healthy server, then reload after stopping so edits
    # saved while the old process is shutting down are included in the new command.
    _load_llamacpp_launcher_config(data)
    stopped = _stop_llamacpp_server(data)
    return _start_llamacpp_server(data, allow_external=not stopped.get("stopped"))

def _kobold_model_name(base_url):
    result = _get_json(urllib.parse.urljoin(base_url + "/", "api/v1/model"), 3)
    if not isinstance(result, dict):
        return None
    name = result.get("result")
    return name.strip() if isinstance(name, str) and name.strip() else None

def _kobold_model_is_inactive(name):
    return not name or name.casefold() in {"inactive", "none", "no model", "unloaded"}

def _kobold_admin_request(data, target):
    base_url = _clean_base_url(data.get("kobold_url"))
    headers = {}
    admin_password = os.environ.get("PROMPT_STUDIO_KOBOLD_ADMIN_PASSWORD", "").strip()
    if admin_password:
        headers["Authorization"] = f"Bearer {admin_password}"
    result = _post_json(
        urllib.parse.urljoin(base_url + "/", "api/admin/reload_config"),
        {"filename": target},
        15,
        "KoboldCpp admin",
        headers=headers,
    )
    if not isinstance(result, dict) or result.get("success") is not True:
        raise RuntimeError(
            "KoboldCpp could not switch models through its admin API. Start it with Admin Mode "
            "enabled and an Admin Directory configured, or enable 'Keep models loaded' only when "
            "the LLM and ComfyUI use separate GPUs. If Admin Password is enabled, set "
            "PROMPT_STUDIO_KOBOLD_ADMIN_PASSWORD before starting ComfyUI."
        )
    return base_url

def _wait_for_kobold_model_state(base_url, inactive, timeout=KOBOLD_ADMIN_TIMEOUT_SECONDS):
    deadline = time.monotonic() + timeout
    while True:
        name = _kobold_model_name(base_url)
        if name is not None and _kobold_model_is_inactive(name) is inactive:
            return name
        if time.monotonic() >= deadline:
            state = "unload" if inactive else "reload"
            raise RuntimeError(f"KoboldCpp did not finish its model {state} before the GPU handoff.")
        time.sleep(0.25)

def _unload_kobold_model(data):
    base_url = _kobold_admin_request(data, "unload_model")
    _wait_for_kobold_model_state(base_url, True)
    _KOBOLD_ADMIN_UNLOADED.add(base_url)
    return {"provider": "koboldcpp", "unloaded": True}

def _reload_kobold_model_if_needed(data):
    if not _KOBOLD_ADMIN_UNLOADED:
        return
    base_url = _clean_base_url(data.get("kobold_url"))
    if base_url not in _KOBOLD_ADMIN_UNLOADED:
        return
    current_name = _kobold_model_name(base_url)
    if current_name is not None and not _kobold_model_is_inactive(current_name):
        _KOBOLD_ADMIN_UNLOADED.discard(base_url)
        return
    _kobold_admin_request(data, "initial_model")
    _wait_for_kobold_model_state(base_url, False)
    _KOBOLD_ADMIN_UNLOADED.discard(base_url)

def _unload_llamacpp_model(llamacpp_url, llamacpp_model, request_timeout=15):
    """Unload a llama-server router model before ComfyUI takes the shared GPU."""
    base_url = _clean_llamacpp_base_url(llamacpp_url)
    model = _text(llamacpp_model).strip()
    if not model:
        models = _list_llamacpp_models(base_url, request_timeout)
        if len(models) == 1:
            model = models[0]
        else:
            raise ValueError("Select a Llama.cpp model in Prompt Studio settings")
    catalog = _get_json(urllib.parse.urljoin(base_url + "/", "models"), int(request_timeout), "Llama.cpp")
    entries = catalog.get("data", []) if isinstance(catalog, dict) else []
    model = next((item["id"] for item in entries if isinstance(item, dict)
                  and model in item.get("aliases", []) and item.get("id")), model)
    if any(isinstance(item, dict) and item.get("id") == model
           and isinstance(item.get("status"), dict) and item["status"].get("value") == "unloaded"
           for item in entries):
        return {"provider": "llamacpp", "model": model, "unloaded": True, "already_released": True}
    try:
        result = _post_json(
            urllib.parse.urljoin(base_url + "/", "models/unload"),
            {"model": model},
            int(request_timeout),
            "Llama.cpp",
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "Llama.cpp shared-GPU handoff requires router mode. Restart it through Prompt Studio's "
            "Backend settings to enable model unloading, or start an external llama-server with "
            "--models-dir. Enable Keep models "
            "loaded only when Llama.cpp and ComfyUI use separate GPUs."
        ) from exc
    if not isinstance(result, dict) or result.get("success") is not True:
        raise RuntimeError(f"Llama.cpp could not unload '{model}': {result}")
    # The router acknowledges its stop request before the worker has exited.
    # Do not let ComfyUI allocate until the router confirms memory is released.
    deadline = time.monotonic() + GPU_HANDOFF_TIMEOUT_SECONDS
    while True:
        catalog = _get_json(urllib.parse.urljoin(base_url + "/", "models"), int(request_timeout), "Llama.cpp")
        entries = catalog.get("data", []) if isinstance(catalog, dict) else []
        if any(isinstance(item, dict) and item.get("id") == model
               and isinstance(item.get("status"), dict) and item["status"].get("value") == "unloaded"
               for item in entries):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Llama.cpp did not finish unloading '{model}' before the GPU handoff.")
        time.sleep(0.1)
    return {"provider": "llamacpp", "model": model, "unloaded": True}

def _text(value, default=""):
    if value is None:
        return default
    return str(value)
