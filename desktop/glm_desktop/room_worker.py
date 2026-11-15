"""A local verified worker waits for a backend-authorized room start."""
import json
import os
import signal
import sys
import threading
import time
import uuid

def run(payload, manifest, cli):
    from glmbench.client import Client
    from glmbench.device import select_device
    import torch
    selected,_=select_device(payload.get("device","auto"),0)
    client=Client(payload["api"],payload["key"])
    client.url=payload["api"].rstrip("/")+"/api/rooms"
    state=client.get()
    room=state.get("room")
    if not room or room["phase"]!="lobby" or room["fingerprint"]!=manifest["fingerprint"]:
        raise RuntimeError("Open training room and matching prepared bundle required")
    instance=uuid.uuid4().hex
    room_id=room["id"]
    client.post(action="join",roomId=room_id,instanceId=instance,fingerprint=manifest["fingerprint"],name=payload["name"],device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print(json.dumps(dict(event="room_connected",instanceId=instance)),flush=True)
    mine=None
    while True:
        client.post(action="heartbeat",roomId=room_id,instanceId=instance)
        state=client.get();room=state.get("room") or {}
        if room.get("id")!=room_id or room.get("phase") in ("cancelled","finished"):
            raise RuntimeError("Training room closed")
        mine=next((m for m in state["members"] if m["instance_id"]==instance),None)
        if room["phase"]=="running":
            if mine is None or mine["shard"] is None:
                # Start assignments may be committed just after the frozen roster.
                if instance not in json.loads(room["expected"]):
                    raise RuntimeError("This computer was excluded from the started room")
            else: break
        time.sleep(2)
    if mine["leader"] and payload.get("resume"):
        raise RuntimeError("Fresh room leader cannot resume: create a fresh run or use the existing recovery CLI")
    stop=threading.Event();progress={}; original=sys.stdout
    class Reporter:
        def write(self,text):
            original.write(text)
            for line in text.splitlines():
                try:
                    row=json.loads(line)
                    if row.get("event") in ("epoch","step","training_progress"):
                        progress.clear();progress.update(row)
                except (ValueError,AttributeError):pass
            return len(text)
        def flush(self):original.flush()
    def heartbeat():
        last=time.monotonic()
        while not stop.wait(3):
            try:
                client.post(action="heartbeat",roomId=room_id,instanceId=instance,progress=dict(progress))
                current=client.get().get("room") or {}
                if current.get("id")!=room_id or current.get("phase")!="running":
                    os.kill(os.getpid(),signal.SIGTERM);return
                last=time.monotonic()
            except Exception:
                if time.monotonic()-last>30:
                    os.kill(os.getpid(),signal.SIGTERM);return
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    sys.argv=["glmtrain","train","--bundle",payload["bundle"],"--host",payload["host"],"--peer",room["peer"],"--name",instance,"--shard",str(mine["shard"]),"--output",payload["output"],"--min-peers",str(len(json.loads(room["expected"]))),"--join-timeout","300","--device","cpu" if selected.type=="cpu" else "cuda"]
    if mine["leader"]:sys.argv.append("--start-new")
    elif payload.get("resume"):sys.argv.append("--resume")
    sys.stdout=Reporter()
    try:
        cli.main()
        client.post(action="heartbeat",roomId=room_id,instanceId=instance,progress={"event":"worker_finished"})
    finally:
        stop.set();thread.join(timeout=20);sys.stdout=original
