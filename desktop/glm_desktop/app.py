import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import ipaddress
import json
import math
import os
from pathlib import Path
import queue
import re
import sys
import time
import webbrowser

from .api import Api, DEFAULT_API, origin
from .runtime import Job, Runtime
from .directory import public_servers
from .state import benchmark_view, training_view

DEVICES = ("auto", "cuda", "rocm", "cpu")


def settings_path():
    base = Path(os.environ.get("APPDATA", Path.home() / "Library/Application Support")) if sys.platform in ("win32", "darwin") else Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "GLM" / "desktop.json"


PREFERENCES = ("api_url", "python_path", "use_wsl", "distro", "host", "device_index", "bundle", "peer", "worker_name", "shard", "output", "resume", "setup_complete")


def save_preferences(path, values):
    # An explicit allowlist keeps keys/passwords out of disk even if a caller
    # accidentally supplies the complete window state.
    safe = {name: values[name] for name in PREFERENCES if name in values}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(safe, indent=2), encoding="utf-8")
    temporary.replace(path)


class Controller:
    def __init__(self, window):
        self.ui = window
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.events = queue.Queue()
        self.jobs = {}
        self.api = None
        self.state = {}
        self.generation = 0
        self.polling = False
        self.next_poll = 0
        self.last_snapshot = 0
        self.server_anchor = None
        self.instance = None
        self.instance_session = None
        self.log_lines = deque(maxlen=200)
        self.target_tokens = self.target_epochs = 0
        self.checked_runtime = None
        self.pending = 0
        self.closing = False
        self.auth_error_until = 0
        window.login_busy = False
        window.login_retry_blocked = False
        window.login_error = ""
        window.busy_label = ""
        window.environment_status = "Not checked"
        window.driver_status = "Not checked"
        window.network_status = "Not checked"
        window.directory_status = "Finding public servers…"
        window.server_selected = False
        window.selected_server_name = "Your team"
        window.selected_server_origin = window.api_url
        window.use_wsl = sys.platform == "win32"
        window.python_path = "/home/YOUR_USER/gatorllm-venv/bin/python" if window.use_wsl else str(Path.home() / "gatorllm-venv/bin/python")
        window.output = "output/desktop"
        try:
            values = json.loads(settings_path().read_text(encoding="utf-8"))
            for key in PREFERENCES:
                if key in values:
                    setattr(window, key, values[key])
        except (OSError, ValueError, TypeError):
            pass
        window.setup_complete = bool(getattr(window, "setup_complete", False))
        window.session_mode = 0
        self.room_state = {}
        self.room_instance = None
        self.bundle_fingerprint = None
        try:
            from .vault import load
            remembered = load(window.api_url)
            if remembered:
                window.server_selected = True
                window.selected_server_origin = window.api_url
                window.access_key = remembered
                self.connect()
                window.setup_step = 1
                if window.setup_complete:
                    window.page = 1
        except Exception:
            pass
        callbacks = {
            "choose_server": self.choose_server, "browse_servers": self.browse_servers,
            "refresh_directory": self.refresh_directory,
            "open_admin_setup": self.open_admin_setup,
            "resume_setup": lambda: setattr(self.ui, "page", 0),
            "finish_setup": self.finish_setup,
            "check_driver": lambda: self.launch("check_gpu"),
            "check_network": lambda: self.launch("check_network"),
            "remember_key": self.remember_key, "forget_key": self.forget_key,
            "admin_login": self.admin_login, "install_tools": lambda: self.launch("install"),
            "create_training_room": self.create_room, "join_room": lambda: self.launch("room"),
            "start_training_room": lambda: self.room_action("start"), "cancel_training_room": lambda: self.room_action("cancel"), "stop_training_discovery": lambda: self.stop("bootstrap"),
            "connect_team": self.connect, "disconnect_team": self.disconnect,
            "check_computer": lambda: self.launch("check"), "join_benchmark": lambda: self.launch("agent"),
            "stop_benchmark": lambda: self.stop("agent"), "toggle_ready": self.ready,
            "open_dashboard": self.open_dashboard, "refresh": self.refresh,
            "save_preferences": self.save, "start_coordinator": lambda: self.launch("coordinator"),
            "find_python": self.find_python, "setup_guide": lambda: webbrowser.open("https://github.com/michaelkrymski/distributed-compute/blob/main/training/BENCHMARK.md"),
            "stop_coordinator": lambda: self.stop("coordinator"), "start_benchmark": self.start_benchmark,
            "inspect_bundle": lambda: self.launch("inspect"), "join_training": lambda: self.launch("train"),
            "stop_training": lambda: self.stop("train"), "poll_ui": self.tick,
        }
        for name, callback in callbacks.items():
            setattr(window, name, callback)
        self.refresh_directory()

    def refresh_directory(self):
        self.ui.directory_status = "Finding public servers…"
        def read():
            try:
                return (public_servers(DEFAULT_API), None) if DEFAULT_API else ([], None)
            except Exception:
                return [], "The directory is unavailable. You can still add a private server."
        def loaded(result):
            import slint
            self.ui.public_servers = slint.ListModel(result[0])
            self.ui.directory_status = result[1] or ("Choose a server to continue." if result[0] else "No public servers are listed yet. Add your team's private address below.")
        self.async_call(read, loaded, busy=False)

    def choose_server(self, address, name):
        if any(self.running(n) for n in ("agent", "train", "room", "coordinator", "bootstrap")):
            self.note("Stop your workers before switching servers.", True)
            return
        try:
            address = origin(address)
        except Exception as error:
            self.note(error, True)
            return
        self.disconnect()
        self.ui.api_url = address
        self.ui.selected_server_origin = address
        self.ui.selected_server_name = name.strip() or "Private server"
        self.ui.server_selected = True
        self.ui.login_choice = 0
        self.ui.access_key = ""
        self.ui.admin_password = ""
        self.ui.login_error = ""
        self.auth_error_until = 0
        self.ui.directory_status = ""
        self.note("Server selected. Choose how you'd like to sign in.")

    def browse_servers(self):
        if any(self.running(n) for n in ("agent", "train", "room", "coordinator", "bootstrap")):
            self.note("Stop your workers before switching servers.", True)
            return
        self.disconnect()
        self.ui.server_selected = False
        self.ui.setup_step = 0
        self.ui.page = 0
        self.ui.access_key = ""
        self.ui.admin_password = ""
        self.refresh_directory()
        self.note("Choose a server to connect with your team.")

    def note(self, message, error=False):
        if self.api:
            message = str(message).replace(self.api.key, "[access key hidden]")
        self.ui.notice, self.ui.notice_error = str(message)[:900], error
        self.log(str(message))

    def log(self, line):
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).replace("\x00", "")
        if self.api:
            line = line.replace(self.api.key, "[access key hidden]")
        self.log_lines.append(line[:2000])
        self.ui.logs = "\n".join(self.log_lines)

    def async_call(self, work, done, *, busy=True):
        generation = self.generation
        if busy:
            self.pending += 1
            self.ui.busy = True
        future = self.pool.submit(work)
        def finished(f):
            try:
                value, error = f.result(), None
            except Exception as exc:
                value, error = None, str(exc)
            self.events.put(("task", (generation, done, value, error, busy)))
        future.add_done_callback(finished)

    def runtime(self):
        return Runtime(self.ui.python_path.strip(), self.ui.use_wsl, self.ui.distro.strip())

    def find_python(self):
        wsl, distro = self.ui.use_wsl, self.ui.distro
        def found(path):
            self.ui.python_path = path
            self.ui.environment_status = "Python found"
            self.note("Found a Python executable. Click Check this computer to verify its training packages.")
        self.ui.busy_label = "Finding your Python environment…"
        self.async_call(lambda: Runtime.discover(wsl, distro), found)

    def identity(self):
        return (self.ui.python_path, self.ui.use_wsl, self.ui.distro, self.ui.device_index)

    def running(self, name):
        return name in self.jobs and self.jobs[name].running

    def connect(self):
        try:
            api = Api(self.ui.api_url, self.ui.access_key)
        except Exception as error:
            self.note(error, True)
            return
        self.note("Connecting to your team…")
        def connected(state):
            self.accept_snapshot(state)
            self.api = api
            self.ui.access_key = ""
            self.note("Connected. Check this computer, then join the shared session.")
        self.async_call(api.request, connected)

    def login_error(self, error):
        self.ui.login_busy = False
        self.ui.login_error = str(error)[:250]
        self.log("Sign-in: " + str(error))
        self.auth_error_until = time.monotonic() + 4
        self.ui.login_retry_blocked = True
        self.note("Sign-in failed. Check the highlighted message and retry.", True)

    def login_call(self, work, done):
        if self.ui.login_busy or time.monotonic() < self.auth_error_until:
            return
        self.ui.login_busy = True
        self.ui.login_error = ""
        self.ui.busy_label = "Signing in…"
        def perform():
            try:
                return work(), None
            except Exception as error:
                return None, str(error)
        def completed(result):
            self.ui.login_busy = False
            if result[1]:
                self.login_error(result[1])
            else:
                try:
                    done(result[0])
                except Exception as error:
                    self.login_error(error)
        self.async_call(perform, completed)

    def finish_setup(self):
        if not self.ui.connected or self.checked_runtime != self.identity():
            self.note("Connect and check this computer before finishing setup.", True)
            return
        self.ui.setup_complete = True
        if not self.save():
            self.ui.setup_complete = False
            return
        if self.ui.setup_complete:
            self.ui.page = 1

    def remember_key(self):
        try:
            from .vault import save
            api = Api(self.ui.api_url, self.ui.access_key)
            key = api.key
            def verified(state):
                save(api.address, key)
                self.api = api
                self.accept_snapshot(state)
                self.ui.access_key = ""
                self.note("Credential saved securely. Continue setup.")
            self.login_call(api.request, verified)
        except Exception as error:
            self.login_error(error)

    def forget_key(self):
        if any(self.running(n) for n in ("agent", "coordinator", "train", "room")):
            self.note("Stop your workers before replacing credentials", True)
            return
        try:
            from .vault import forget
            forget(origin(self.ui.api_url))
            self.disconnect()
        except Exception as error:
            self.login_error(error)

    def admin_login(self):
        password = self.ui.admin_password
        self.ui.admin_password = ""
        try:
            api = Api(self.ui.api_url, "login")
            def logged(data):
                from .vault import save
                save(api.address, data["token"])
                self.ui.access_key = data["token"]
                self.connect()
                self.note("Administrator device signed in. Your password was not saved.")
            self.login_call(lambda: api.request({"password": password}, "desktop-auth"), logged)
        except Exception as error:
            self.login_error(error)

    def room_action(self, action, **values):
        if not self.api:
            return
        room = self.room_state.get("room") or {}
        api = self.api
        self.async_call(lambda: api.request(dict(action=action, roomId=room.get("id"), **values), "rooms"), lambda _: self.note("Training room updated"))

    def create_room(self):
        if not self.ui.admin or not self.bundle_fingerprint:
            self.note("Verify the prepared bundle first", True)
            return
        self.launch("bootstrap")

    def publish_room(self):
        self.room_action("create", name=self.ui.room_name, fingerprint=self.bundle_fingerprint, peer=self.ui.peer, targetTokens=self.target_tokens, shards=getattr(self,"bundle_shards",1))

    def disconnect(self):
        if any(self.running(name) for name in ("agent", "coordinator", "room", "train", "bootstrap")):
            self.note("Stop your benchmark agent and coordinator before disconnecting.", True)
            return
        self.generation += 1
        self.api = None
        self.polling = False
        self.state = {}
        self.room_state = {}
        self.room_instance = None
        self.ui.session_mode = 0
        self.instance = None
        self.ui.connected = False
        self.ui.access_key = ""
        self.ui.user_label = "Not connected"
        self.note("Disconnected. The access key has been cleared from this session.")

    def accept_snapshot(self, state):
        if type(state.get("serverTime")) not in (float, int) or not math.isfinite(state["serverTime"]) or not isinstance(state.get("user"), dict):
            raise ValueError("The server response has no shared session clock or user identity.")
        self.state = state
        self.server_anchor = (state["serverTime"], time.monotonic())
        self.last_snapshot = time.monotonic()
        self.ui.connected = True
        self.ui.user_label = "Connected as " + str(state["user"].get("id", "team member"))
        self.next_poll = time.monotonic() + 2

    def refresh(self):
        if not self.api or self.polling:
            return
        self.polling = True
        api = self.api
        def receive(pair):
            self.room_state = pair[1]
            self.accept_snapshot(pair[0])
        self.async_call(lambda: (api.request(), api.request(endpoint="rooms")), receive, busy=False)

    def action(self, name, **values):
        session = self.state.get("session") or {}
        if not self.api or not session.get("id") or self.ui.stale:
            self.note("Refresh the shared session before changing it.", True)
            return
        api, session_id = self.api, session["id"]
        def done(_):
            self.note("Session updated.")
            self.next_poll = 0
        self.async_call(lambda: api.action(name, session_id, **values), done)

    def ready(self):
        if getattr(self.ui, "session_mode", 0) == 1:
            self.room_action("ready", instanceId=self.room_instance, ready=not self.ui.local_ready)
            return
        if self.ui.can_ready and self.instance:
            self.action("request_ready", instanceId=self.instance, ready=not self.ui.local_ready)

    def start_benchmark(self):
        if self.ui.can_start:
            self.action("start")

    def launch(self, operation):
        try:
            if operation in ("check", "install", "check_gpu", "check_network", "inspect") and any(self.running(n) for n in ("agent", "train", "room")):
                raise ValueError("Stop your compute worker before changing or checking its runtime.")
            if self.running(operation):
                raise ValueError("This process is already running.")
            if operation in ("agent", "train", "room") and any(self.running(n) for n in ("agent", "train", "room")):
                raise ValueError("Stop the current compute worker before starting another.")
            if operation not in ("check", "install", "check_gpu", "check_network") and self.checked_runtime != self.identity():
                raise ValueError("Check this computer with the selected runtime first.")
            if operation == "room":
                if not self.api or self.ui.stale:
                    raise ValueError("Connect to your team first")
            payload = dict(operation=operation, device=DEVICES[self.ui.device_index], output=self.ui.output.strip())
            if operation in ("agent", "coordinator", "train", "room", "bootstrap"):
                host = self.ui.host.strip()
                address = ipaddress.ip_address(host)
                if address.is_unspecified or address.is_loopback or address.version != 4:
                    raise ValueError("Use this runtime's reachable Tailscale IPv4, not a loopback address.")
                payload["host"] = host
                if not payload["output"]:
                    raise ValueError("Choose a local results/checkpoint output directory.")
            if operation in ("agent", "coordinator"):
                if not self.api or self.ui.stale:
                    raise ValueError("Connect and refresh your team session first.")
                if operation == "agent" and not self.ui.can_join:
                    raise ValueError("Wait for an open lobby before joining.")
                if operation == "coordinator" and not self.ui.admin:
                    raise ValueError("A coordinator access key is required to open a lobby.")
                payload.update(api=self.api.address, key=self.api.key)
                if operation == "agent":
                    payload["session_id"] = self.state["session"]["id"]
                    self.instance = self.instance_session = None
            if operation in ("inspect", "train", "bootstrap"):
                if not self.ui.bundle.strip():
                    raise ValueError("Enter the complete run bundle folder from your organizer.")
                payload["bundle"] = self.ui.bundle.strip()
            if operation == "train":
                if not re.fullmatch(r"/ip[46]/[^\s]+/tcp/\d+/p2p/[A-Za-z0-9]+", self.ui.peer.strip()):
                    raise ValueError("Paste the complete peer address from your organizer.")
                shard = int(self.ui.shard)
                if shard < 0 or not self.ui.worker_name.strip():
                    raise ValueError("Choose a worker name and a nonnegative assigned shard.")
                payload.update(peer=self.ui.peer.strip(), name=self.ui.worker_name.strip(), shard=shard, resume=self.ui.resume)
                self.target_tokens = self.target_epochs = 0
                self.ui.training_progress = 0
                self.ui.training_status = "Joining training — loading bundle and peer state…"
                self.ui.training_summary = "Waiting for the worker's first progress event."
            if operation == "room":
                payload.update(api=self.api.address, key=self.api.key, bundle=self.ui.bundle.strip(), name=self.ui.worker_name, resume=self.ui.resume)
            runtime = self.runtime()
            identity = self.identity()
            def launched(job):
                self.jobs[operation] = job
                if operation == "check":
                    self.checking_runtime = identity
                self.note("Computer check started…" if operation == "check" else f"{operation.capitalize()} started. See Debug for details.")
            self.ui.busy_label = {"check":"Checking your GPU and Tailscale…", "check_gpu":"Checking GPU driver…", "check_network":"Checking Tailscale…", "install":"Installing training packages…", "inspect":"Verifying run bundle…"}.get(operation, "Connecting your computer…")
            self.async_call(lambda: Job(runtime, payload, lambda kind, data: self.events.put(("job", (operation, kind, data)))), launched)
        except Exception as error:
            self.note(error, True)

    def stop(self, operation):
        if operation == "train" and self.running("room"):
            operation = "room"
        if operation == "room" and (self.room_state.get("room") or {}).get("phase") == "lobby":
            self.room_action("leave", instanceId=self.room_instance)
        job = self.jobs.get(operation)
        if job:
            self.note("Stopping your local process gracefully…")
            self.async_call(job.stop, lambda _: self.note("Stop request completed. Checkpoints remain in your output folder."))

    def save(self):
        try:
            origin(self.ui.api_url)
            save_preferences(settings_path(), {name: getattr(self.ui, name) for name in PREFERENCES})
            self.note("Settings saved. Credentials remain in the OS credential store.")
            return True
        except Exception as error:
            self.note(error, True)

    def open_admin_setup(self):
        try:
            webbrowser.open(origin(self.ui.api_url))
            self.note("On the server website, use your coordinator key for one-time admin password setup. If a password is already configured, sign in with it instead.")
        except Exception as error:
            self.note(error, True)

    def open_dashboard(self):
        try:
            webbrowser.open(origin(self.ui.api_url))
        except Exception as error:
            self.note(error, True)

    def job_event(self, operation, kind, data):
        if kind == "exit":
            self.log(f"{operation}: {'stopped' if data['stopped'] else 'exited'} (code {data['code']})")
            if operation == "train":
                self.ui.training_status = "Worker stopped — resume from the last checkpoint" if data["stopped"] else "Worker finished" if data["code"] == 0 else "Worker exited with an error — see Debug"
            if operation == "agent":
                self.instance = None
                self.next_poll = 0
            if data["code"] != 0 and not data["stopped"]:
                self.note(f"The {operation} process exited with an error. See Debug for the cause.", True)
            return
        line, row = data
        if line:
            if isinstance(row, dict) and row.get("event") == "desktop_check":
                self.log(f"Device: {row.get('device', 'unknown')} | PyTorch: {row.get('packages', {}).get('torch', '?')} | Tailscale: {row.get('host') or 'not connected'}")
            elif isinstance(row, dict) and row.get("event") == "desktop_bundle":
                self.log(f"Run bundle verified: {row.get('run')}")
            else:
                self.log(line)
        if not isinstance(row, dict):
            return
        event = row.get("event")
        if operation == "bootstrap" and row.get("initial_peers"):
            self.ui.peer = row["initial_peers"][0]
            self.publish_room()
        if event == "desktop_gpu":
            self.ui.driver_status = row["device"]
        elif event == "desktop_network":
            self.ui.network_status = "Connected" if row.get("host") else "Sign-in needed"
            if row.get("host"):
                self.ui.host = row["host"]
        elif event == "install_stage":
            self.ui.setup_progress = row["progress"]
            self.ui.setup_progress_label = row["label"]
        elif event == "install_complete":
            self.ui.setup_progress = 1
            self.ui.setup_progress_label = "Packages installed. Check this computer next."
        elif event == "room_connected":
            self.room_instance = row["instanceId"]
        elif event == "desktop_error":
            self.note(row.get("message", "Worker error"), True)
        elif event == "desktop_check":
            if row.get("platform") == "Windows":
                self.note("Training tools were found in native Windows. Select WSL for Hivemind training.", True)
                return
            self.checked_runtime = getattr(self, "checking_runtime", self.identity())
            self.ui.device_label = row.get("device", "CPU")
            self.ui.driver_status = self.ui.device_label
            self.ui.environment_status = "Verified"
            self.ui.network_status = "Connected" if row.get("host") else "Sign-in needed"
            self.ui.runtime_label = f"Training tools found · {row.get('platform')} · PyTorch {row.get('packages', {}).get('torch', '?')}"
            if row.get("host"):
                self.ui.host = row["host"]
            else:
                self.checked_runtime = None
                self.note("GPU found. Sign into Tailscale, then check this computer again.", True)
                return
            self.note("Computer checked. Confirm its Tailscale address, then join the session.")
        elif event == "agent_connected":
            member = row.get("member", "")
            if ":" in member:
                self.instance = member.rsplit(":", 1)[1]
                self.instance_session = row.get("sessionId")
                self.note("Your computer joined. Choose Mark ready when you are ready to contribute.")
                self.next_poll = 0
        elif event == "desktop_bundle":
            self.bundle_fingerprint = row.get("fingerprint")
            self.bundle_shards = row.get("shards", 1)
            self.target_tokens = row.get("target_tokens", 0)
            self.ui.bundle_summary = f"Verified {row.get('run')} · {self.target_tokens:,} target tokens · {row.get('shards')} assigned shards"
        elif event == "ready" and operation in ("train", "room"):
            self.target_epochs = row.get("target_epochs", 0)
            self.ui.training_status = "Training worker connected"
        elif event in ("epoch", "step", "training_progress") and operation in ("train", "room"):
            view = training_view(row, self.target_tokens, self.target_epochs)
            self.ui.training_progress = view["progress"]
            self.ui.training_summary = view["summary"]
            self.ui.loss = view["loss"]
            self.ui.training_status = "Training — receiving worker updates"

    def tick(self):
        try:
            for _ in range(100):
                try:
                    kind, payload = self.events.get_nowait()
                except queue.Empty:
                    break
                if kind == "job":
                    self.job_event(*payload)
                elif kind == "task":
                    generation, done, value, error, busy = payload
                    if busy:
                        self.pending = max(0, self.pending - 1)
                    elif generation == self.generation:
                        self.polling = False
                    if generation != self.generation:
                        continue
                    if error:
                        self.note(error, True)
                        self.next_poll = time.monotonic() + 5
                    else:
                        done(value)
            checks_running = any(self.running(n) for n in ("check", "install", "inspect", "check_gpu", "check_network"))
            self.ui.busy = self.pending > 0 or checks_running
            self.ui.login_retry_blocked = time.monotonic() < self.auth_error_until
            self.ui.login_error_visible = time.monotonic() < self.auth_error_until - 0.4
            self.ui.joined = self.running("agent") or self.running("room") or self.running("train")
            if not self.ui.busy:
                self.ui.busy_label = ""

            self.ui.checked = self.checked_runtime == self.identity()
            self.ui.agent_running = self.running("agent")
            self.ui.training_running = self.running("train") or self.running("room")
            self.ui.coordinator_running = self.running("coordinator")
            self.ui.stale = bool(self.api) and time.monotonic() - self.last_snapshot > 8
            if self.api and time.monotonic() >= self.next_poll:
                self.refresh()
            now = self.server_anchor[0] + (time.monotonic() - self.server_anchor[1]) * 1000 if self.server_anchor else 0
            view = benchmark_view(self.state, now)
            session = self.state.get("session") or {}
            self.ui.phase = view["phase"].capitalize()
            self.ui.phase_title = {"lobby": "Ready when your team is.", "warmup": "Warming up together.", "measuring": "Measuring shared performance.", "finished": "The shared run is complete.", "cancelled": "This session was cancelled.", "countdown": "Starting together."}.get(view["phase"], "Ready for a shared experiment.")
            self.ui.countdown = f"{view['remaining']//60}:{view['remaining']%60:02}" if view["phase"] in ("warmup", "measuring", "countdown") else "Waiting" if view["phase"] == "lobby" else "—"
            self.ui.bench_progress = view["progress"]
            self.ui.team_summary, self.ui.workload = view["summary"], view["workload"]
            self.ui.result_text = view["result"]
            # Update row models only when their content changes, preserving UI focus.
            if getattr(self, "last_rows", None) != view["rows"]:
                import slint
                self.ui.computers = slint.ListModel(view["rows"])
                self.last_rows = view["rows"]
            own = next((a for a in self.state.get("agents", []) if a.get("instanceId") == self.instance), None)
            self.ui.local_ready = bool(own and own.get("requestedReady"))
            self.ui.can_ready = bool(own and self.running("agent") and self.instance_session == session.get("id") and view["phase"] == "lobby" and not self.ui.stale)
            self.ui.can_join = bool(self.api and self.ui.checked and view["phase"] == "lobby" and not self.running("agent") and not self.running("train") and not self.running("room") and not self.ui.stale)
            self.ui.admin = self.state.get("user", {}).get("role") == "coordinator"
            self.ui.lobby_label = ((self.room_state.get("room") or {}).get("name") or "Model training") if getattr(self.ui, "session_mode", 0) == 1 else "Team benchmark"
            self.ui.lobby_server = self.api.address if self.api else self.ui.api_url
            self.ui.can_start = bool(self.ui.admin and session.get("canStart") and not self.ui.stale)
            self.update_room_view()
            self.ui.has_session = bool(session and view["phase"] not in ("finished", "cancelled"))
        except Exception as error:
            self.note(str(error), True)

    def update_room_view(self):
        reports = [a for a in self.state.get("agents", []) if (a.get("result") or {}).get("processedTokens", 0) > 0]
        total = sum(a["result"]["processedTokens"] for a in reports)
        import slint
        self.ui.leaderboard = slint.ListModel([dict(device=a["device"]["name"], share=a["result"]["processedTokens"]/total) for a in sorted(reports, key=lambda a: -a["result"]["processedTokens"])])
        complete = reports and all((a.get("result") or {}).get("complete") for a in reports)
        expected = (self.state.get("session") or {}).get("expected", [])
        if complete and len(reports) == len(expected):
            seconds = max(a["result"]["measuredSeconds"] for a in reports)
            rate = benchmark_view(self.state, 0)["rate"]
            target = (self.state.get("session") or {}).get("config", {}).get("trainingTokens", 500000000)
            self.ui.forecast_ready = rate > 0
            self.ui.estimate = f"{target:,} tokens at {rate:,.0f} tokens/s: approximately {target/rate/3600:.1f} hours. For the benchmarked model. Excludes data loading, evaluation and checkpoints." if rate else "Estimate unavailable"
        room = self.room_state.get("room")
        if not room or room.get("phase") == "cancelled":
            return
        self.ui.session_mode = 1
        members = self.room_state.get("members", [])
        own = next((m for m in members if m["instance_id"] == self.room_instance), None)
        self.ui.local_ready = bool(own and own["ready"])
        self.ui.can_ready = bool(own and room["phase"] == "lobby" and not self.ui.stale)
        self.ui.phase_title = "Training complete." if room["phase"] == "finished" else "Ready for model training." if room["phase"] == "lobby" else "Training together."
        self.ui.team_summary = f"{len(members)} computers in training room"
        self.ui.workload = room["name"]
        self.ui.computers = slint.ListModel([dict(name=m["name"], device=m["device"], status="Offline" if self.room_state.get("serverTime",0)-m["last_seen"]>20000 else "Ready" if m["ready"] else "Connected", ready=bool(m["ready"]) and self.room_state.get("serverTime",0)-m["last_seen"]<=20000) for m in members])

    def close(self):
        self.closing = True
        self.generation += 1
        # Finish launches already in flight, then stop the exact jobs they own.
        self.pool.shutdown(wait=True, cancel_futures=True)
        while not self.events.empty():
            kind, payload = self.events.get()
            if kind == "task":
                _, done, value, error, _ = payload
                if not error and isinstance(value, Job):
                    value.stop()
        for job in self.jobs.values():
            job.stop()
        self.api = None


def main():
    parser = argparse.ArgumentParser(description="GLM Slint team companion")
    parser.add_argument("--check-ui", action="store_true", help="Compile the UI without connecting or launching training")
    parser.add_argument("--choose-server", action="store_true", help="Open server selection without deleting saved credentials")
    args = parser.parse_args()
    import slint
    components = slint.load_file(str(Path(__file__).with_name("ui") / "main.slint"), style="fluent-dark")
    if args.check_ui:
        print("Slint UI compiled successfully")
        return
    ui = components.MainWindow()
    controller = Controller(ui)
    if args.choose_server:
        controller.browse_servers()
    try:
        ui.run()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
