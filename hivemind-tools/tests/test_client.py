import json
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from glmbench.client import Client


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.headers.get("Authorization") == "Bearer slow":
            with self.server.counter_lock:
                self.server.slow_calls += 1
                call = self.server.slow_calls
            if call <= 2:
                time.sleep(0.15)
        if self.headers.get("Authorization") == "Bearer clock-probe":
            self.server.clock_calls += 1
            payload = json.dumps({"serverTime": int(time.time() * 1000)}).encode()
            if self.server.clock_calls == 2:
                time.sleep(2.2)
            status = 200
        elif self.headers.get("User-Agent", "").startswith("Python-urllib"):
            status, payload = 403, b"Access denied"
        elif self.headers.get("Authorization") == "Bearer member":
            status = 403
            payload = json.dumps({"error": "Only the coordinator can create a lobby."}).encode()
        elif self.headers.get("Authorization") == "Bearer blocked":
            status, payload = 403, b"<html>Access denied</html>"
        else:
            status, payload = 200, b'{"id":"test-lobby"}'
        self.send_response(status)
        self.end_headers()
        try:
            self.wfile.write(payload)
        except ConnectionError:
            pass  # The timeout regression test deliberately disconnects.

    def log_message(self, *args):
        pass


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.counter_lock = threading.Lock()
        cls.server.slow_calls = 0
        cls.server.clock_calls = 0
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_post_reaches_api_with_application_identity(self):
        self.assertEqual(Client(self.origin, "coordinator").post(action="create"), {"id": "test-lobby"})

    def test_api_permission_error_is_preserved(self):
        with self.assertRaisesRegex(RuntimeError, "403: Only the coordinator"):
            Client(self.origin, "member").post(action="create")

    def test_non_json_hosting_error_is_explained(self):
        with self.assertRaisesRegex(RuntimeError, "403: Non-JSON response from the hosting layer"):
            Client(self.origin, "blocked").post(action="create")

    def test_heartbeat_recovers_after_real_http_read_timeouts(self):
        self.server.slow_calls = 0
        open_request = urllib.request.urlopen
        with patch("glmbench.client.urllib.request.urlopen", side_effect=lambda request, timeout: open_request(request, timeout=0.04)):
            result = Client(self.origin, "slow").post(action="coordinator_heartbeat")
        self.assertEqual(result, {"id": "test-lobby"})
        self.assertEqual(self.server.slow_calls, 3)

    def test_lobby_creation_is_not_replayed_after_timeout(self):
        self.server.slow_calls = 0
        open_request = urllib.request.urlopen
        with patch("glmbench.client.urllib.request.urlopen", side_effect=lambda request, timeout: open_request(request, timeout=0.04)):
            with self.assertRaisesRegex(RuntimeError, "Backend connection timed out"):
                Client(self.origin, "slow").post(action="create")
        self.assertEqual(self.server.slow_calls, 1)

    def test_real_delayed_response_retains_recent_clock_accuracy(self):
        self.server.clock_calls = 0
        client = Client(self.origin, "clock-probe")
        client.post(action="coordinator_heartbeat")
        client.post(action="coordinator_heartbeat")
        self.assertLess(client.uncertainty_ms, 1000)
        self.assertEqual(client.clock_jump_ms, 0)
