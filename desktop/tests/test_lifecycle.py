import json
import os
from pathlib import Path
import sys
import time

import pytest

from glm_desktop.runtime import Job
from glm_desktop import bridge


def test_owned_subprocess_input_redaction_and_exit(tmp_path):
    script=tmp_path/"child fixture.py"
    script.write_text("import json,sys\np=json.loads(sys.stdin.readline())\nprint(json.dumps({'event':'fixture','message':p['key']}),flush=True)\n")
    class FixtureRuntime:
        wsl=False
        def command(self):return [sys.executable,"-u",str(script)]
    events=[]
    job=Job(FixtureRuntime(),dict(key="private-test-token"),lambda kind,data:events.append((kind,data)))
    job.thread.join(10)
    assert not job.running
    assert "private-test-token" not in repr(events)
    assert events[-1]==("exit",dict(code=0,stopped=False))
    assert job.secret == ""
    job.stop()  # Stopping a completed process must not signal another process.


def test_stop_targets_only_owned_process(tmp_path):
    script=tmp_path/"waiting child.py"
    script.write_text("import time\nprint('waiting',flush=True)\ntime.sleep(60)\n")
    class FixtureRuntime:
        wsl=False
        def command(self):return [sys.executable,"-u",str(script)]
    events=[]
    job=Job(FixtureRuntime(),{},lambda kind,data:events.append((kind,data)))
    try:
        job.stop()
        job.thread.join(10)
        assert not job.running and events[-1][1]["stopped"]
    finally:
        if job.running:job.process.kill()


def test_corpus_bridge_preserves_cpu_choice_and_resume(monkeypatch):
    import io
    from types import ModuleType, SimpleNamespace
    data=ModuleType("glmtrain.data")
    data.load_bundle=lambda _:dict(run_id="fixture",fingerprint="a"*64,training=dict(target_tokens=1000,data_shards=2,target_batch_tokens=128))
    cli=ModuleType("glmtrain.__main__")
    observed=[]
    cli.main=lambda:observed.extend(sys.argv)
    package=ModuleType("glmtrain")
    package.__main__=cli
    monkeypatch.setitem(sys.modules,"glmtrain",package)
    monkeypatch.setitem(sys.modules,"glmtrain.data",data)
    monkeypatch.setitem(sys.modules,"glmtrain.__main__",cli)
    monkeypatch.setattr(bridge.platform,"system",lambda:"Linux")
    monkeypatch.setattr(bridge,"send",lambda **_:None)
    monkeypatch.setattr(bridge.signal,"signal",lambda *args:None)
    if os.name != "nt":monkeypatch.setattr(bridge.os,"setsid",lambda:None)
    payload=dict(operation="train",nonce="fixture",bundle="/run bundle",host="127.0.0.1",peer="peer",name="name",shard=1,output="/output",device="cpu",resume=True)
    monkeypatch.setattr(sys,"stdin",io.StringIO(json.dumps(payload)+"\n"))
    monkeypatch.setattr(sys,"argv",[])
    bridge.main()
    assert observed[observed.index("--device")+1]=="cpu"
    assert observed[observed.index("--bundle")+1]=="/run bundle"
    assert "--resume" in observed and "--start-new" not in observed


def test_bridge_stop_unwinds_existing_engine_cleanup():
    with pytest.raises(KeyboardInterrupt):
        bridge.request_stop(None,None)


def test_packaged_install_finds_internal_sources_without_working_directory(tmp_path):
    app = tmp_path / "Installed App"
    script = app / "_internal/glm_desktop/bridge.py"
    script.parent.mkdir(parents=True)
    script.touch()
    project = app / "_internal/training"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    assert bridge.training_source("training", script) == project
    with pytest.raises(RuntimeError, match="missing"):
        bridge.training_source("hivemind-tools", script)


def test_utf16_wsl_stderr_cannot_corrupt_authenticated_utf8_stdout(tmp_path):
    script = tmp_path / "mixed encodings.py"
    script.write_text("""import json, sys
p=json.loads(sys.stdin.readline())
sys.stderr.buffer.write('wsl: Unknown key — café\\n'.encode('utf-16-le'))
sys.stderr.buffer.flush()
print(json.dumps(dict(desktop_event='started',nonce=p['nonce'],group=12345)),flush=True)
print(json.dumps(dict(event='desktop_check',device='fixture')),flush=True)
""", encoding="utf-8")
    class FixtureRuntime:
        wsl=False
        def command(self): return [sys.executable, "-u", str(script)]
    events=[]
    job=Job(FixtureRuntime(), {}, lambda kind,data: events.append((kind,data)))
    job.thread.join(10)
    assert not job.running and job.group==12345
    lines=[value[0] for kind,value in events if kind=='line']
    assert any('café' in line for line in lines)
    assert all('\x00' not in line and 'nonce' not in line for line in lines)
    assert any(value[1] and value[1].get('event')=='desktop_check' for kind,value in events if kind=='line')
