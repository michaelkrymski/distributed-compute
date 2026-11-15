"""Credential-free server discovery. Results are metadata, never authority to sign in."""
import json
import urllib.request
from .api import NoRedirect, origin

def public_servers(directory_origin):
    address=origin(directory_origin)
    request=urllib.request.Request(address+"/api/servers", headers={"User-Agent":"glm-desktop/0.2", "Accept":"application/json"})
    with urllib.request.build_opener(NoRedirect()).open(request,timeout=15) as response:
        payload=json.loads(response.read(100_000))
    rows=payload.get("servers") if isinstance(payload,dict) else None
    if not isinstance(rows,list) or len(rows)>50:
        raise ValueError("Invalid server directory")
    result=[]
    for row in rows:
        if not isinstance(row,dict) or not all(isinstance(row.get(k),str) for k in ("name","description","origin")):
            continue
        try:
            address=origin(row["origin"])
        except (ValueError, RuntimeError):
            continue
        if not address.startswith("https://"):
            continue
        result.append(dict(name=row["name"][:80],description=row["description"][:240],origin=address))
    return result
