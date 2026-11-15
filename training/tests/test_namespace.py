import pytest
from glmtrain.__main__ import scoped_manifest


def test_room_namespace_preserves_bundle_rendezvous_and_valid_averaging_prefix():
    bundle = {"run_id": "glm0-bundle", "fingerprint": "immutable"}
    room = scoped_manifest(bundle, "project-id.room-id")
    assert "." not in room["run_id"]
    assert room["rendezvous_run_id"] == bundle["run_id"]
    assert room == scoped_manifest(bundle, "project-id.room-id")
    assert room["run_id"] != scoped_manifest(bundle, "other-project.room-id")["run_id"]
    assert room["fingerprint"] == bundle["fingerprint"]
    assert "rendezvous_run_id" not in bundle


def test_namespace_rejects_invalid_input():
    with pytest.raises(ValueError):
        scoped_manifest({"run_id": "glm0-bundle"}, "untrusted/path")
