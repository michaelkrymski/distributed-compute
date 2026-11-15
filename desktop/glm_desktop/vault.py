"""OS credential storage; refuse plaintext/fallback keyring backends."""
import hashlib

def account(origin):
    return hashlib.sha256(origin.rstrip("/").encode()).hexdigest()

def backend():
    import keyring
    store = keyring.get_keyring()
    module = type(store).__module__
    if not module.startswith(("keyring.backends.Windows", "keyring.backends.SecretService", "keyring.backends.kwallet", "keyring.backends.macOS")):
        raise RuntimeError("Unlock your OS credential store (Windows Credential Manager or Linux Secret Service). No plaintext fallback is used.")
    return store

def save(origin, key):
    backend().set_password("Distributed Compute", account(origin), key)

def load(origin):
    return backend().get_password("Distributed Compute", account(origin))

def forget(origin):
    store = backend()
    if store.get_password("Distributed Compute", account(origin)):
        store.delete_password("Distributed Compute", account(origin))
