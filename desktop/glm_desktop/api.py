import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit

DEFAULT_API = os.environ.get("COMPUTE_DIRECTORY_URL", "").strip()


class ApiError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise ApiError("The server redirected this request. Check the server address; the key was not forwarded.")


def origin(value):
    value = value.strip().rstrip("/")
    url = urlsplit(value)
    if (url.scheme != "https" and not (url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1", "::1"))) or not url.hostname:
        raise ApiError("Use an HTTPS server address. HTTP is allowed only for localhost development.")
    if url.username or url.password or url.query or url.fragment or url.path:
        raise ApiError("Enter just the server origin, without a path, password or query.")
    return value


class Api:
    def __init__(self, address, key):
        self.address = origin(address)
        if not key.strip() or len(key) > 4096 or "\n" in key or "\r" in key:
            raise ApiError("Enter your private team access key, not your admin password.")
        self.key = key.strip()
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, body=None, endpoint="benchmark"):
        request = urllib.request.Request(self.address + "/api/" + endpoint,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json", "User-Agent": "glmbench/0.4.0 glm-desktop/0.1"})
        try:
            with self.opener.open(request, timeout=15) as response:
                data = json.loads(response.read(2_000_000))
        except urllib.error.HTTPError as error:
            try:
                message = json.loads(error.read(16000)).get("error", "Request rejected")
            except (ValueError, AttributeError):
                message = "The hosting service rejected the request. Check the server address and access key."
            raise ApiError(f"Server {error.code}: {str(message).replace(self.key, '[access key hidden]')}") from None
        except (OSError, ValueError) as error:
            raise ApiError("Could not read the server response. Check your connection and server address.") from error
        if not isinstance(data, dict):
            raise ApiError("Unexpected server response; this may not be a GLM server.")
        return data

    def action(self, action, session_id, **values):
        # Mutations are deliberately never automatically retried.
        return self.request(dict(action=action, sessionId=session_id, **values))
