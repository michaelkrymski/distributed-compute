import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import pytest

from glm_desktop.api import Api, ApiError, origin
from glm_desktop.app import save_preferences
from glm_desktop.runtime import Runtime, wsl_path
from glm_desktop.state import benchmark_view, training_view


@pytest.fixture
def http_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/servers":
                self.server.requests.append((self.path, self.headers.get("Authorization"), None))
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({"servers":[
                    {"name":"Public team","description":"A team","origin":"https://public.example"},
                    {"name":"Invalid","description":"","origin":"https://user:secret@bad.example"},
                    {"name":"Invalid","description":"","origin":"http://bad.example"}
                ]}).encode())
                return
            if self.path == "/api/benchmark" and self.headers.get("Authorization") == "Bearer redirect":
                self.send_response(302)
                self.send_header("Location", "/must-not-receive-credentials")
                self.end_headers()
                return
            self.server.requests.append((self.path, self.headers.get("Authorization"), None))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(dict(serverTime=100000, user=dict(id="fixture",role="member"),session=None,agents=[])).encode())

        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.server.requests.append((self.path,self.headers.get("Authorization"),body))
            self.send_response(403 if body.get("action") == "start" else 200)
            self.end_headers()
            self.wfile.write(json.dumps(dict(error="Admin access required") if body.get("action") == "start" else dict(ok=True)).encode())

        def log_message(self,*args):
            pass
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    server.requests=[]
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    yield server
    server.shutdown();server.server_close();thread.join()


def test_existing_http_contract_and_own_instance_readiness(http_server):
    api=Api(f"http://127.0.0.1:{http_server.server_port}","fixture-key")
    assert api.request()["session"] is None
    api.action("request_ready","session-1",instanceId="my-instance",ready=True)
    assert http_server.requests[-1] == ("/api/benchmark","Bearer fixture-key",dict(action="request_ready",sessionId="session-1",instanceId="my-instance",ready=True))
    with pytest.raises(ApiError,match="403: Admin access required"):
        api.action("start","session-1")
    assert len(http_server.requests)==3  # Failed mutations are not replayed.


def test_public_directory_is_credential_free_and_validates_listed_origins(http_server):
    from glm_desktop.directory import public_servers
    rows=public_servers(f"http://127.0.0.1:{http_server.server_port}")
    assert rows==[dict(name="Public team",description="A team",origin="https://public.example")]
    assert http_server.requests==[("/api/servers",None,None)]


def test_credentials_are_not_forwarded_on_redirect(http_server):
    api=Api(f"http://127.0.0.1:{http_server.server_port}","redirect")
    with pytest.raises(ApiError,match="not forwarded"):
        api.request()
    assert not http_server.requests


@pytest.mark.parametrize("url",["http://example.com","https://user:password@example.com","https://example.com/path","https://example.com/?key=secret","file:///tmp/key"])
def test_only_explicit_secure_origins(url):
    with pytest.raises(ApiError):origin(url)


def test_preferences_never_save_credentials(tmp_path):
    path=tmp_path/"desktop.json"
    save_preferences(path,dict(api_url="https://example.com",python_path="/a path/python",access_key="secret",password="secret",key="secret"))
    assert json.loads(path.read_text())==dict(api_url="https://example.com",python_path="/a path/python")


def test_wsl_command_keeps_paths_as_literal_arguments(monkeypatch):
    monkeypatch.setattr("glm_desktop.runtime.BRIDGE",r"C:\app path\bridge.py")
    command=Runtime("/home/a user/python",True,"Ubuntu").command()
    assert command[:5]==["wsl.exe","--distribution","Ubuntu","--","/home/a user/python"]
    assert wsl_path(r"C:\Users\A B\bridge.py")=="/mnt/c/Users/A B/bridge.py"
    with pytest.raises(ValueError):wsl_path(r"\\server\share\bridge.py")
    with pytest.raises(ValueError):Runtime("--bad").command()


def test_incomplete_group_never_gets_a_throughput_estimate():
    result=dict(complete=True,measuredRounds=10,localBatchTokens=16384,measuredSeconds=120)
    state=dict(session=dict(phase="finished",expected=["a","b"],startsAt=100000,config={}),agents=[dict(userId="a",result=result)])
    assert "unavailable" in benchmark_view(state,200000)["result"]
    state["agents"].append(dict(userId="b",result=result))
    assert "2,731" in benchmark_view(state,200000)["result"]
    state["agents"][1]["result"]={**result,"complete":False}
    assert "unavailable" in benchmark_view(state,200000)["result"]


def test_phase_progress_and_readiness_use_server_time():
    state=dict(session=dict(phase="measuring",startsAt=100000,config=dict(warmupSeconds=60,measureSeconds=120)),agents=[dict(lastSeenAt=200000,ready=True,requestedReady=True,device=dict(name="GPU",hostname="machine"))])
    view=benchmark_view(state,220000)
    assert view["progress"]==.5 and view["remaining"]==60
    assert view["rows"][0]["status"]=="Offline"


def test_training_progress_is_observed_and_clamped():
    view=training_view(dict(epoch=10,nominal_global_tokens=327680,loss=3.2),500000000,0)
    assert view["progress"]==327680/500000000 and view["loss"]=="3.2000"
    assert training_view(dict(epoch=1000),0,100)["progress"]==1
    assert training_view({},0,0)["progress"]==0


def test_slint_document_compiles():
    import slint
    import glm_desktop
    slint.load_file(str(Path(glm_desktop.__file__).with_name("ui")/"main.slint"))


def test_proportional_local_steps_count_actual_processed_tokens():
    from glm_desktop.state import benchmark_view
    state = {"session": {"phase": "finished", "expected": ["a", "b"], "config": {"localSteps": 4}}, "agents": [
        {"userId": "a", "result": {"complete": True, "measuredRounds": 10, "measuredSeconds": 100, "localBatchTokens": 1024, "processedTokens": 40960}},
        {"userId": "b", "result": {"complete": True, "measuredRounds": 10, "measuredSeconds": 100, "localBatchTokens": 2048, "processedTokens": 81920}}]}
    assert benchmark_view(state, 0)["result"] == "1,229 tokens/sec combined"
